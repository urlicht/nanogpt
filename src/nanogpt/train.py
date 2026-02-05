from dataclasses import dataclass
from pathlib import Path
import torch
import torch.nn as nn
from nanogpt.model import ModelConfig, NanoGPT

@dataclass
class TrainConfig:
    # model
    n_vocab: int = 0
    n_layer: int = 4
    n_head: int = 8
    d_emb: int = 256
    dropout_p: float = 0.2
    mlp_width_multiplier: int = 4

    n_block: int = 4
    n_batch: int = 4

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