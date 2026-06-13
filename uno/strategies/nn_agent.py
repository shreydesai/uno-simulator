"""
Neural-network-powered UNO agent.

Inference
---------
The agent runs a forward pass through ``UnoNet`` to obtain card logits and
color logits, masks illegal actions, then either samples or argmax-picks.

RL training mode
----------------
Set ``training=True`` to make the agent record a ``Transition`` for every
decision.  After the game ends, call ``reset_trajectory()`` to retrieve the
episode data for a policy-gradient update.

The color head result is cached after ``choose_card`` so that
``choose_color`` reuses the same forward pass (no redundant compute).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import numpy as np

from uno.agent import Agent, GameState
from uno.card import Card, Color
from uno.encoding import (
    ACTION_DIM, DRAW_ACTION,
    build_action_mask, card_type_idx, card_to_action,
    action_to_card, color_to_idx, idx_to_color,
    encode_state,
)

try:
    import torch
    import torch.nn.functional as F
    _TORCH_OK = True
except ImportError:
    _TORCH_OK = False


# ------------------------------------------------------------------ #
#  Transition dataclass (used for RL trajectory collection)            #
# ------------------------------------------------------------------ #

@dataclass
class Transition:
    """
    One step in a training episode.

    Fields
    ------
    state         : encoded state vector at decision time  (STATE_DIM,)
    action        : chosen action index (0-54)
    log_prob      : log π(action | state) at sample time
    mask          : legal-action boolean mask              (ACTION_DIM,)
    color_action  : chosen color index 0-3, or None if no wild was played
    color_log_prob: log π(color | state) for the color choice, or None
    """
    state:          np.ndarray
    action:         int
    log_prob:       float
    mask:           np.ndarray
    color_action:   Optional[int]   = None
    color_log_prob: Optional[float] = None


# ------------------------------------------------------------------ #
#  Agent                                                               #
# ------------------------------------------------------------------ #

class NNAgent(Agent):
    """
    Parameters
    ----------
    name        : display name shown in game output
    model       : a ``UnoNet`` instance (see ``training/model.py``)
    device      : ``'cpu'`` or ``'cuda'``
    greedy      : ``True`` → argmax;  ``False`` → sample from the policy
    temperature : softmax temperature (used only when ``greedy=False``)
    training    : record ``Transition`` objects for RL training
    """

    def __init__(
        self,
        name: str,
        model: "UnoNet",           # noqa: F821  (imported lazily)
        device: str = "cpu",
        greedy: bool = False,
        temperature: float = 1.0,
        training: bool = False,
    ) -> None:
        if not _TORCH_OK:
            raise ImportError(
                "PyTorch is required for NNAgent.\n"
                "Install it with:  pip install torch"
            )
        super().__init__(name)
        self.model       = model.to(device)
        self.device      = device
        self.greedy      = greedy
        self.temperature = temperature
        self.training_mode = training

        # Cached color logits from the last forward pass
        self._cached_color_logits: Optional["torch.Tensor"] = None
        # Episode trajectory (populated when training_mode=True)
        self.trajectory: list[Transition] = []

    # ------------------------------------------------------------------ #
    #  Agent interface                                                     #
    # ------------------------------------------------------------------ #

    def choose_card(self, state: GameState) -> Optional[Card]:
        import torch

        self._cached_color_logits = None

        state_vec = encode_state(state)
        mask      = build_action_mask(state)
        state_t   = torch.tensor(state_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        ctx = torch.no_grad() if not self.training_mode else torch.enable_grad()
        with ctx:
            card_logits, color_logits, _value = self.model(state_t)

        # Cache color logits for choose_color (same forward pass)
        self._cached_color_logits = color_logits.squeeze(0).detach()

        # Mask illegal actions
        clogits = card_logits.squeeze(0).clone()
        mask_t  = torch.tensor(mask, dtype=torch.bool, device=self.device)
        clogits[~mask_t] = float("-inf")
        clogits = clogits / self.temperature

        if self.greedy:
            action   = int(clogits.argmax().item())
            log_prob = float(F.log_softmax(clogits, dim=0)[action].item())
        else:
            probs    = F.softmax(clogits, dim=0)
            action   = int(torch.multinomial(probs, 1).item())
            log_prob = float(torch.log(probs[action] + 1e-10).item())

        if self.training_mode:
            self.trajectory.append(Transition(
                state=state_vec,
                action=action,
                log_prob=log_prob,
                mask=mask,
            ))

        playable = state.playable_cards()
        return action_to_card(action, playable)

    def choose_color(self, state: GameState) -> Color:
        import torch

        logits = self._cached_color_logits
        if logits is None:
            # Fallback: re-run forward pass (shouldn't normally happen)
            state_t = torch.tensor(encode_state(state), dtype=torch.float32,
                                   device=self.device).unsqueeze(0)
            with torch.no_grad():
                _, logits, _val = self.model(state_t)
            logits = logits.squeeze(0)

        if self.greedy:
            color_idx = int(logits.argmax().item())
        else:
            probs     = F.softmax(logits / self.temperature, dim=0)
            color_idx = int(torch.multinomial(probs, 1).item())

        color_log_prob = float(F.log_softmax(logits, dim=0)[color_idx].item())

        # Annotate the most recent trajectory entry with color info
        if self.training_mode and self.trajectory:
            t = self.trajectory[-1]
            t.color_action    = color_idx
            t.color_log_prob  = color_log_prob

        self._cached_color_logits = None
        return idx_to_color(color_idx)

    # ------------------------------------------------------------------ #
    #  RL helpers                                                          #
    # ------------------------------------------------------------------ #

    def reset_trajectory(self) -> list[Transition]:
        """Return and clear the accumulated episode transitions."""
        traj, self.trajectory = self.trajectory, []
        return traj
