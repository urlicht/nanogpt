from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from typing import Tuple
from nanogpt.model import ModelConfig, NanoGPT
from nanogpt.data.batch import get_batch

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

def eval_loss(
    cfg: TrainConfig,
    model: NanoGPT,
) -> dict[str, float]:
    """Run eval on train/val and return estimated losses"""

    loss_dict = {}
    for batch in ('train', 'validation'):
        xb, yb = get_batch(
            cfg.data_dir,
            batch,
            cfg.n_batch,
            cfg.n_block,
            cfg.device
        )
        y_pred, loss = model(xb, yb)
        loss_dict[batch] = loss

    return loss_dict