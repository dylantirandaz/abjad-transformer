"""A transformer whose every position is a chord: one host token plus the run of satellites before it.

Each pass reads the sum of a run embedding and a host embedding, and makes two picks for the
next chord. The first pick is looked up in a table and added to the hidden state before the
second pick. That addition is the only link between the two halves; there is no second network.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from chords import HOST_COLUMN, RUN_COLUMN
from model import Block, init_weights

RUN_FIRST = "run_first"
HOST_FIRST = "host_first"
INDEPENDENT = "independent"
ORDERS = (RUN_FIRST, HOST_FIRST, INDEPENDENT)


@dataclass(frozen=True)
class ChordConfig:
    n_runs: int
    n_hosts: int
    block_size: int
    n_layer: int
    n_head: int
    n_embd: int
    dropout: float
    order: str  # which half of the chord is picked first

    @property
    def causal(self) -> bool:
        return True


def sample_classes(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    return torch.multinomial(F.softmax(logits.float() / temperature, dim=-1), 1).squeeze(-1)


class ChordTransformer(nn.Module):
    def __init__(self, config: ChordConfig) -> None:
        super().__init__()
        if config.order not in ORDERS:
            message = f"Unknown chord order: {config.order}"
            raise ValueError(message)
        self.config = config
        self.run_embedding = nn.Embedding(config.n_runs, config.n_embd)
        self.host_embedding = nn.Embedding(config.n_hosts, config.n_embd)
        self.position_embedding = nn.Embedding(config.block_size, config.n_embd)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.n_layer))
        self.norm_out = nn.LayerNorm(config.n_embd)
        self.run_head = nn.Linear(config.n_embd, config.n_runs, bias=False)
        self.host_head = nn.Linear(config.n_embd, config.n_hosts, bias=False)
        first_pick_sizes = {RUN_FIRST: config.n_runs, HOST_FIRST: config.n_hosts}
        self.first_pick_embedding: nn.Embedding | None = (
            nn.Embedding(first_pick_sizes[config.order], config.n_embd) if config.order in first_pick_sizes else None
        )
        self.apply(init_weights)

    def hidden(self, chords: torch.Tensor) -> torch.Tensor:
        """Map (batch, length, 2) chords to (batch, length, width) hidden states."""
        length = chords.shape[1]
        if length > self.config.block_size:
            message = f"Sequence length {length} exceeds block size {self.config.block_size}"
            raise ValueError(message)
        positions = torch.arange(length, device=chords.device)
        embedded = self.run_embedding(chords[..., RUN_COLUMN]) + self.host_embedding(chords[..., HOST_COLUMN])
        x: torch.Tensor = self.dropout(embedded + self.position_embedding(positions))
        for block in self.blocks:
            x = block(x)
        normalized: torch.Tensor = self.norm_out(x)
        return normalized

    def condition(self, hidden: torch.Tensor, first_pick: torch.Tensor) -> torch.Tensor:
        """Hidden state used for the second pick, given the first."""
        if self.first_pick_embedding is None:
            return hidden
        conditioned: torch.Tensor = hidden + self.first_pick_embedding(first_pick)
        return conditioned

    def forward(self, chords: torch.Tensor, next_chords: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (run logits, host logits) for the chord after each position, given the true first pick."""
        hidden = self.hidden(chords)
        if self.config.order == HOST_FIRST:
            host_logits: torch.Tensor = self.host_head(hidden)
            run_logits: torch.Tensor = self.run_head(self.condition(hidden, next_chords[..., HOST_COLUMN]))
            return run_logits, host_logits
        run_logits = self.run_head(hidden)
        host_logits = self.host_head(self.condition(hidden, next_chords[..., RUN_COLUMN]))
        return run_logits, host_logits

    @torch.no_grad()
    def sample_next(self, hidden: torch.Tensor, temperature: float) -> torch.Tensor:
        """Sample one chord for each row of (batch, width) hidden states. Returns (batch, 2)."""
        if self.config.order == HOST_FIRST:
            hosts = sample_classes(self.host_head(hidden), temperature)
            runs = sample_classes(self.run_head(self.condition(hidden, hosts)), temperature)
        else:
            runs = sample_classes(self.run_head(hidden), temperature)
            hosts = sample_classes(self.host_head(self.condition(hidden, runs)), temperature)
        columns = {RUN_COLUMN: runs, HOST_COLUMN: hosts}
        return torch.stack([columns[column] for column in sorted(columns)], dim=-1)

    def count_parameters(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
