"""
UnoNet — shared MLP trunk with two output heads.

Architecture
------------
  Input  : state vector (STATE_DIM = 169)
  Trunk  : n_layers × [Linear → LayerNorm → ReLU → Dropout]
  Head 1 : card_head  → logits over ACTION_DIM (54 card plays + draw)
  Head 2 : color_head → logits over COLOR_DIM  (Red / Green / Blue / Yellow)

The card head is queried every turn; the color head is queried only after a
wild card is played.  Both heads share the trunk so color knowledge can inform
card selection and vice-versa.

Saving / loading
----------------
    model.save("checkpoints/my_model.pt")
    model = UnoNet.load("checkpoints/my_model.pt")
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
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.color_dim  = color_dim

        # Build shared trunk
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
        self.card_head  = nn.Linear(hidden, action_dim)
        self.color_head = nn.Linear(hidden, color_dim)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        x : float tensor of shape (batch, state_dim)

        Returns
        -------
        card_logits  : (batch, action_dim)
        color_logits : (batch, color_dim)
        """
        h = self.trunk(x)
        return self.card_head(h), self.color_head(h)

    # ------------------------------------------------------------------ #
    #  Persistence                                                         #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "state_dim":  self.state_dim,
                "action_dim": self.action_dim,
                "color_dim":  self.color_dim,
            },
        }, path)

    @classmethod
    def load(cls, path: str, **override_kwargs) -> "UnoNet":
        data   = torch.load(path, map_location="cpu", weights_only=False)
        config = {**data.get("config", {}), **override_kwargs}
        net    = cls(**config)
        net.load_state_dict(data["state_dict"])
        return net
