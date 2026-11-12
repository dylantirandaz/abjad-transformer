"""A small transformer usable as a causal language model or a bidirectional tagger."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

MLP_EXPANSION = 4
INIT_STD = 0.02
# Query, key and value projections share one linear layer.
QKV_PARTS = 3


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int
    n_out: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    causal: bool


class SelfAttention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.n_head = config.n_head
        self.causal = config.causal
        self.dropout = config.dropout
        self.qkv = nn.Linear(config.n_embd, QKV_PARTS * config.n_embd, bias=False)
        self.proj = nn.Linear(config.n_embd, config.n_embd, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        combined: torch.Tensor = self.qkv(x)
        q, k, v = (
            part.view(batch, length, self.n_head, width // self.n_head).transpose(1, 2)
            for part in combined.chunk(QKV_PARTS, dim=2)
        )
        attended = F.scaled_dot_product_attention(
            q, k, v, is_causal=self.causal, dropout_p=self.dropout if self.training else 0.0
        )
        projected: torch.Tensor = self.proj(attended.transpose(1, 2).reshape(batch, length, width))
        return projected


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.norm_attn = nn.LayerNorm(config.n_embd)
        self.attn = SelfAttention(config)
        self.norm_mlp = nn.LayerNorm(config.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, MLP_EXPANSION * config.n_embd),
            nn.GELU(),
            nn.Linear(MLP_EXPANSION * config.n_embd, config.n_embd),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended: torch.Tensor = x + self.attn(self.norm_attn(x))
        transformed: torch.Tensor = attended + self.mlp(self.norm_mlp(attended))
        return transformed


class Transformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_out = nn.LayerNorm(config.n_embd)
        self.head = nn.Linear(config.n_embd, config.n_out, bias=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=INIT_STD)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        length = ids.shape[1]
        if length > self.config.block_size:
            message = f"Sequence length {length} exceeds block size {self.config.block_size}"
            raise ValueError(message)
        positions = torch.arange(length, device=ids.device)
        x: torch.Tensor = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        logits: torch.Tensor = self.head(self.norm_out(x))
        return logits

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
