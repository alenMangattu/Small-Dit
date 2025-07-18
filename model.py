# This script provides a minimal, educational implementation of the Diffusion Transformer (DiT)
# model, as introduced in the paper "Scalable Diffusion Models with Transformers" by
# William Peebles and Saining Xie (https://arxiv.org/abs/2212.09748).
#
# This implementation is intended for educational purposes to demystify the DiT architecture.
# It is not an official implementation from the paper's authors.

import torch
import torch.nn as nn
from torch.nn import functional as F
import math
from dataclasses import dataclass

@dataclass
class DiTConfig:
    image_size: int = 256
    patch_size: int = 16
    in_channels: int = 3
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    dropout: float = 0.1

    @property
    def n_ctx(self):
        return (self.image_size // self.patch_size) ** 2

class SelfAttention(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        y = F.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.attn_dropout.p, is_causal=False)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        y = self.resid_dropout(self.c_proj(y))
        return y

class FeedForward(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

class DiTBlock(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.attn = SelfAttention(config)
        self.norm2 = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.mlp = FeedForward(config)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 6 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x

class FinalLayer(nn.Module):
    def __init__(self, config: DiTConfig):
        super().__init__()
        self.norm_final = nn.LayerNorm(config.n_embd, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(config.n_embd, config.patch_size * config.patch_size * config.in_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.n_embd, 2 * config.n_embd, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        return self.mlp(t_freq)

class DiT(nn.Module):
    def __init__(self, config: DiTConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.in_channels = config.in_channels
        self.out_channels = config.in_channels
        self.patch_size = config.patch_size
        self.num_heads = config.n_head

        self.x_embedder = nn.Conv2d(self.in_channels, config.n_embd, kernel_size=self.patch_size, stride=self.patch_size)
        self.t_embedder = TimestepEmbedder(config.n_embd)
        self.y_embedder = nn.Embedding(num_classes, config.n_embd)

        self.pos_embed = nn.Parameter(torch.zeros(1, config.n_ctx, config.n_embd))

        self.blocks = nn.ModuleList([
            DiTBlock(config) for _ in range(config.n_layer)
        ])
        self.final_layer = FinalLayer(config)
        self.initialize_weights()

    def initialize_weights(self):
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        nn.init.normal_(self.pos_embed, std=0.02)

        nn.init.normal_(self.y_embedder.weight, std=0.02)

        nn.init.kaiming_normal_(self.x_embedder.weight, mode="fan_in", nonlinearity="relu")

        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)

    def unpatchify(self, x):
        c = self.out_channels
        p = self.patch_size
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, w * p))
        return imgs

    def forward(self, x, t, y):
        x = self.x_embedder(x).flatten(2).transpose(1, 2)
        x = x + self.pos_embed

        t = self.t_embedder(t)
        y = self.y_embedder(y)
        c = t + y

        for block in self.blocks:
            x = block(x, c)

        x = self.final_layer(x, c)
        x = self.unpatchify(x)
        return x