"""
UnoNet — shared MLP trunk with three output heads.

Architecture
------------
  Input  : state vector (STATE_DIM = 169)
  Trunk  : n_layers × [Linear → LayerNorm → ReLU → Dropout]
  Head 1 : card_head  → logits over ACTION_DIM (54 card plays + draw)
  Head 2 : color_head → logits over COLOR_DIM  (Red / Green / Blue / Yellow)
  Head 3 : value_head → scalar V(s) for PPO / actor-critic methods

The card head is queried every turn; the color head only after a wild is
played; the value head provides the baseline for GAE advantage estimation.
Supervised training ignores the value head — it is trained from scratch
when fine-tuning with PPO.

Saving / loading
----------------
    model.save("checkpoints/my_model.pt")
    model = UnoNet.load("checkpoints/my_model.pt")

Backward compatibility
----------------------
Old checkpoints (without value_head weights) are loaded with strict=False so
the value head is silently random-initialized rather than raising a key error.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from uno.encoding import ACTION_DIM, COLOR_DIM, STATE_DIM


class UnoNet(nn.Module):
    def __init__(
        self,
        state_dim:  int   = STATE_DIM,
        action_dim: int   = ACTION_DIM,
        color_dim:  int   = COLOR_DIM,
        hidden:     int   = 256,
        n_layers:   int   = 3,
        dropout:    float = 0.1,
    ) -> None:
        super().__init__()
        # Store full config so save() / load() can reconstruct exactly
        self._config = dict(
            state_dim=state_dim, action_dim=action_dim, color_dim=color_dim,
            hidden=hidden, n_layers=n_layers, dropout=dropout,
        )
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.color_dim  = color_dim

        # Shared trunk
        layers: list[nn.Module] = []
        in_dim = state_dim
        for _ in range(n_layers):
            layers += [
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            in_dim = hidden
        self.trunk      = nn.Sequential(*layers)

        # Policy heads
        self.card_head  = nn.Linear(hidden, action_dim)
        self.color_head = nn.Linear(hidden, color_dim)

        # Value head (used by PPO; ignored during supervised training)
        self.value_head = nn.Linear(hidden, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        x : float tensor of shape (batch, state_dim)

        Returns
        -------
        card_logits  : (batch, action_dim)
        color_logits : (batch, color_dim)
        value        : (batch,)   — V(s) estimate; use for PPO, ignore otherwise
        """
        h     = self.trunk(x)
        # tanh bounds V(s) to (-1, 1), matching the ±1 terminal reward range.
        # This prevents gradient explosion in the value head early in RL training.
        value = self.value_head(h).squeeze(-1).tanh()
        return self.card_head(h), self.color_head(h), value

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save({"state_dict": self.state_dict(), "config": self._config}, path)

    @classmethod
    def load(cls, path: str, **override_kwargs) -> "UnoNet":
        data   = torch.load(path, map_location="cpu", weights_only=False)
        config = {**data.get("config", {}), **override_kwargs}
        net    = cls(**config)
        missing, unexpected = net.load_state_dict(data["state_dict"], strict=False)
        if missing:
            print(
                f"[UnoNet] {len(missing)} key(s) not in checkpoint "
                f"(e.g. value_head) — initialized randomly. "
                f"This is expected when loading a supervised checkpoint for RL."
            )
        return net
