import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class ModelConfig:
    n_vocab:int = 512
    n_block:int = 256 # max context
    n_layer:int = 4
    n_head:int = 12
    d_emb:int = 768
    dropout_p:float = 0.2
    mlp_width_multiplier:int = 4

class AttentionHead(nn.Module):
    def __init__(self, config: ModelConfig):
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
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_emb % config.n_head == 0, "Embedding dim must be divisible by n_head"
        
        self.n_head = config.n_head
        self.d_head = config.d_emb // config.n_head
        self.d_emb = config.d_emb

        # Q, K, V into one linear layer for efficiency
        self.qkv_proj = nn.Linear(config.d_emb, 3 * config.d_emb, bias=False)
        self.out_proj = nn.Linear(config.d_emb, config.d_emb, bias=False)
        
        self.dropout_p = config.dropout_p

    def forward(self, x):
        n_batch, n_token, d_emb = x.shape # batch, sequence length, embedding dim

        # linear projection of all heads
        qkv = self.qkv_proj(x) # (n_batch, n_token, d_emb * 3)
        
        # split into q, k, v and reshape to (n_batch, n_head, d_head)
        q, k, v = qkv.split(self.d_emb, dim=2)

        q = q.view(n_batch, n_token, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(n_batch, n_token, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(n_batch, n_token, self.n_head, self.d_head).transpose(1, 2)
        
        # scaled dot product
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout_p if self.training else 0,
            is_causal=True
        )

        # combine the heads and project back
        out = out.transpose(1, 2).contiguous().view(n_batch, n_token, d_emb)

        return self.out_proj(out)

class MLP(nn.Module):
    def __init__(self, config: ModelConfig):
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

class AttentionBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_emb = config.d_emb
        self.dropout_p = config.dropout_p
        self.d_head = config.d_emb // config.n_head

        self.mh_attn = MultiheadAttention(config)
        self.ffn = MLP(config)
        self.ln1 = nn.LayerNorm(self.d_emb)
        self.ln2 = nn.LayerNorm(self.d_emb)

    def forward(self, x):
        x = self.ln1(self.mh_attn(x)) + x
        x = self.ln2(self.ffn(x)) + x

        return x

class NanoGPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.d_emb = config.d_emb
        self.n_vocab = config.n_vocab
        self.n_block = config.n_block

        self.token_embedding = nn.Embedding(self.n_vocab, self.d_emb)
        self.position_embedding = nn.Embedding(self.n_block, self.d_emb)

        self.blocks = nn.Sequential(
            *[AttentionBlock(config) for _ in range(config.n_layer)],
            nn.LayerNorm(self.d_emb)
        )
        self.prj_out = nn.Linear(self.d_emb, self.n_vocab)

    def forward(self, x, y_target=None):
        n_batch, n_token = x.shape[:2]
    
        # embedding
        emb = self.token_embedding(x) # (B, T, C) where C = d_emb, T = n_token, B = n_batch
        emb_pos = self.position_embedding(torch.arange(n_token, device=x.device)) # (T, C)
        x = emb + emb_pos # (B, T, C)
        x = self.blocks(x) # (B, T, C)

        logits = self.prj_out(x) # (B, T, n_vocab)

        if y_target is None:
            loss = None
        else:
            logits = logits.view(n_batch * n_token, self.n_vocab)
            targets = y_target.view(n_batch * n_token,)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    @torch.no_grad()
    def generate(self, x, max_n_token):
        # x is (B, T) of current tokens
        for _ in range(max_n_token):
            # crop
            x_crop = x[:, -self.n_block:]
            
            # pred
            logits, loss = self(x_crop)

            # last token
            logits = logits[:,-1,:] # (B, C)

            # sample and append
            x_next = torch.multinomial(F.softmax(logits, dim=1), num_samples=1)
            x = torch.cat([x, x_next], dim=1) # (B, T+1)

        return x