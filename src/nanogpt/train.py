from dataclasses import dataclass
from pathlib import Path
import time
import torch
import torch.nn as nn
from typing import Tuple
from nanogpt.model import ModelConfig, NanoGPT
from nanogpt.data.batch import get_batch
from torch.utils.tensorboard import SummaryWriter

@dataclass
class TrainConfig:
    # dataset
    data_dir: str|Path

    # device
    device: torch.device = torch.device('cpu')

    # model
    n_vocab: int = 0
    n_layer: int = 4
    n_head: int = 8
    d_emb: int = 256
    dropout_p: float = 0.2
    mlp_width_multiplier: int = 4

    n_block: int = 4
    n_batch: int = 4

    # optimizer
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    betas: Tuple[float, float] = (0.9, 0.999)

    # eval
    eval_iters: int = 10
    eval_every: int = 100

    # train loop
    grad_clip: float = 1.0
    max_iter: int = 1000
    save_every: int = 500
    out_dir: str|Path = "checkpoints"

def build_model(cfg: TrainConfig) -> NanoGPT:
    model_cfg = ModelConfig(
        n_vocab=cfg.n_vocab,
        n_block=cfg.n_block,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        d_emb=cfg.d_emb,
        dropout_p=cfg.dropout_p,
        mlp_width_multiplier=cfg.mlp_width_multiplier,
    )

    return NanoGPT(model_cfg)

def build_optimizer(cfg: TrainConfig, model: NanoGPT) -> torch.optim.Optimizer:
    # ps that requires gradient
    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}
    
    decay_params = []
    nodecay_params = []
    for nm, p in param_dict.items():
        # 1D params (biases/layernorms) or position embeddings get no decay
        if p.dim() < 2 or "position_embedding" in nm:
            nodecay_params.append(p)
        else:
            decay_params.append(p)

    optim_groups = [
        {'params': decay_params, 'weight_decay': cfg.weight_decay},
        {'params': nodecay_params, 'weight_decay': 0.0}
    ]
    
    n_decay_params = sum(p.numel() for p in decay_params)
    n_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(f"number of decayed parameter tensors: {len(decay_params)}, with {n_decay_params:,} parameters")
    print(f"number of non-decayed parameter tensors: {len(nodecay_params)}, with {n_nodecay_params:,} parameters")

    opt = torch.optim.AdamW(optim_groups, lr=cfg.learning_rate, betas=cfg.betas)

    return opt

@torch.no_grad()
def eval_loss(
    cfg: TrainConfig,
    model: NanoGPT,
) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
    """Run eval on train/val and return estimated losses"""

    loss_dict = {}
    eval_time = {}
    model.eval()

    for batch in ('train', 'val'):
        t0 = time.perf_counter()
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):
            xb, yb = get_batch(
                cfg.data_dir,
                batch,
                cfg.n_batch,
                cfg.n_block,
                cfg.device
            )
            y_pred, loss = model(xb, yb)
            losses[k] = loss
        loss_dict[batch] = losses.mean()
        eval_time[batch] = time.perf_counter() - t0
    
    model.train()

    return loss_dict, eval_time

def _build_tb_writer(out_dir: Path):
    tb_dir = out_dir / "tb"
    tb_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(log_dir=str(tb_dir))

def train_loop(
    cfg: TrainConfig,
    model: NanoGPT,
    opt: torch.optim.Optimizer,
):
    model.train()
    out_dir = Path(cfg.out_dir)
    ckpt_dir = out_dir / "checkpoints"
    eval_every = cfg.eval_every
    if cfg.save_every > 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if eval_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    n_batch = cfg.n_batch
    n_block = cfg.n_block
    device = cfg.device
    save_every = cfg.save_every
    grad_clip = cfg.grad_clip
    tb_writer = _build_tb_writer(out_dir)

    loss_fh = None
    if eval_every > 0:
        loss_csv_path = out_dir / "losses.csv"
        write_header = not loss_csv_path.exists() or loss_csv_path.stat().st_size == 0
        loss_fh = loss_csv_path.open("a", buffering=1)
        if write_header:
            loss_fh.write("iter,train_loss,val_loss,train_time_avg,eval_train_time,eval_val_time\n")

    train_time_sum = 0.0
    train_time_steps = 0
    for it in range(cfg.max_iter):
        t_iter = time.perf_counter()

        xb, yb = get_batch(
            cfg.data_dir,
            'train',
            n_batch,
            n_block,
            device,
        )

        _, loss = model(xb, yb) # forward
        opt.zero_grad(set_to_none=True) # zero grad
        loss.backward() # backward pass
        # gradient clipping (returns global grad norm before clipping)
        grad_norm = None
        if grad_clip and grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"))
        
        # optimizer step
        opt.step()

        tb_writer.add_scalar("grad/global_norm", float(grad_norm), it)
        
        train_time_sum += time.perf_counter() - t_iter
        train_time_steps += 1

        # eval
        if eval_every > 0 and it % eval_every == 0:
            avg_train_time = train_time_sum / train_time_steps
            loss_dict, eval_time = eval_loss(cfg, model)
            train_loss = loss_dict["train"].item()
            val_loss = loss_dict["val"].item()
            train_eval_time = eval_time["train"]
            val_eval_time = eval_time["val"]
            print(
                f"iter {it}: train {train_loss:.4f}, val {val_loss:.4f}, "
                f"train_time/step {avg_train_time:.2f}s, "
                f"eval_train {train_eval_time:.2f}s, eval_val {val_eval_time:.2f}s"
            )
            if loss_fh is not None:
                loss_fh.write(
                    f"{it},{train_loss:.6f},{val_loss:.6f},"
                    f"{avg_train_time:.6f},{train_eval_time:.6f},{val_eval_time:.6f}\n"
                )
            tb_writer.add_scalar("loss/train", train_loss, it)
            tb_writer.add_scalar("loss/val", val_loss, it)
            tb_writer.add_scalar("time/train_iter_avg_s", avg_train_time, it)
            tb_writer.add_scalar("time/eval_train_s", train_eval_time, it)
            tb_writer.add_scalar("time/eval_val_s", val_eval_time, it)
            train_time_sum = 0.0
            train_time_steps = 0

        # check point
        if save_every > 0 and it % save_every == 0:
            ckpt_path = ckpt_dir / f"ckpt_{it:06d}.pt"
            torch.save(
                {
                    "iter": it,
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "cfg": cfg,
                },
                ckpt_path,
            )

    if loss_fh is not None:
        loss_fh.close()
    tb_writer.close()
