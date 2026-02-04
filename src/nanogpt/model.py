import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class Config:
    n_vocab:int = 512
    n_block:int = 256 # max context
    n_layer:int = 4
    n_head:int = 12
    d_emb:int = 768
    dropout_p:float = 0.2
    mlp_width_multiplier:int = 4


    # d_emb = 384
    # n_batch = 64
    # eval_iters = 500
    # max_iters = 5000
    # eval_interval = 1000
    # learning_rate = 3e-4
    # d_head = d_emb // n_head
    # ffn_width_multiplier: int = 4

class AttentionHead(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.d_head = config.d_emb // config.n_head
        self.n_block = config.n_block
        self.d_emb = config.d_emb

        self.query = nn.Linear(self.d_emb, self.d_head, bias=False)
        self.key = nn.Linear(self.d_emb, self.d_head, bias=False)
        self.value = nn.Linear(self.d_emb, self.d_head, bias=False)
        self.dropout = nn.Dropout(config.dropout_p)

        self.register_buffer('tril', torch.tril(torch.ones(self.n_block, self.n_block)))

    def forward(self, x):
        n_token = x.shape[1]

        w = self.query(x) @ self.key(x).transpose(-2,-1)
        w = w.masked_fill(self.tril[:n_token, :n_token] == 0, float('-inf'))
        w = F.softmax(w * (self.d_head ** -0.5), dim=-1)
        w = self.dropout(w)

        output = w @ self.value(x)

        return output

class MultiheadAttention(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.n_head = config.n_head
        self.d_head = config.d_emb // config.n_head
        self.d_emb = config.d_emb
        self.dropout_p = config.dropout_p

        self.heads = nn.ModuleList([AttentionHead(config) for _ in range(self.n_head)])
        self.proj = nn.Linear(self.d_emb, self.d_emb)
        self.dropout = nn.Dropout(self.dropout_p)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        
        return out
    
class MLP(nn.Module):
    def __init__(self, config: Config):
        super().__init__()

        self.d_emb = config.d_emb
        self.dropout_p = config.dropout_p
        self.mlp_width_x = config.mlp_width_multiplier

        self.network = nn.Sequential(
            nn.Linear(self.d_emb, self.d_emb * self.mlp_width_x),
            nn.GELU(),
            nn.Linear(self.d_emb * self.mlp_width_x, self.d_emb),
            nn.GELU(),
            nn.Dropout(self.dropout_p)
        )

    def forward(self, x):
        return self.network(x)
