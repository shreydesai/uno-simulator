"""
Collect (state, action, mask) transitions from games played by existing agents.

Usage
-----
    from uno.strategies.smart_agent import SmartAgent
    from training.data_collector import DataCollector

    collector = DataCollector(teacher=SmartAgent("Smart"))
    dataset   = collector.collect(n_games=10_000, verbose=True)
    dataset.save("data/smart_10k.npz")
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from uno.agent import Agent, GameState
from uno.card import Card, Color
from uno.encoding import (
    build_action_mask, card_to_action, color_to_idx, encode_state,
)
from uno.game import UnoGame


# ------------------------------------------------------------------ #
#  Recording wrapper                                                   #
# ------------------------------------------------------------------ #

class _RecordingAgent(Agent):
    """
    Transparent wrapper that intercepts every ``choose_card`` /
    ``choose_color`` call and stores the (state, action, mask) tuple.
    """

    def __init__(self, inner: Agent) -> None:
        super().__init__(inner.name)
        self._inner          = inner
        self.records:         list[dict] = []
        self._pending: Optional[dict]    = None

    def choose_card(self, state: GameState) -> Optional[Card]:
        choice = self._inner.choose_card(state)
        record = {
            "state":        encode_state(state),
            "action":       card_to_action(choice),
            "mask":         build_action_mask(state),
            "color_action": None,
        }
        self._pending = record
        self.records.append(record)
        return choice

    def choose_color(self, state: GameState) -> Color:
        color = self._inner.choose_color(state)
        if self._pending is not None:
            self._pending["color_action"] = color_to_idx(color)
            self._pending = None
        return color

    def pop_records(self) -> list[dict]:
        recs, self.records, self._pending = self.records, [], None
        return recs


# ------------------------------------------------------------------ #
#  Dataset container                                                   #
# ------------------------------------------------------------------ #

@dataclass
class TransitionDataset:
    """
    Numpy arrays of collected transitions.

    Arrays (all aligned on axis-0)
    --------------------------------
    states  : (N, STATE_DIM)   float32
    actions : (N,)             int32    — card action index 0-54
    masks   : (N, ACTION_DIM)  bool
    colors  : (N,)             int32    — 0-3 for color choice; -1 if no wild
    rewards : (N,)             float32  — +1.0 win, -1.0 loss
    """

    states:  np.ndarray
    actions: np.ndarray
    masks:   np.ndarray
    colors:  np.ndarray
    rewards: np.ndarray

    def __len__(self) -> int:
        return len(self.states)

    def save(self, path: str) -> None:
        import os
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        np.savez_compressed(
            path,
            states=self.states,
            actions=self.actions,
            masks=self.masks,
            colors=self.colors,
            rewards=self.rewards,
        )
        print(f"Saved {len(self):,} transitions → {path}")

    @classmethod
    def load(cls, path: str) -> "TransitionDataset":
        data = np.load(path)
        return cls(
            states=data["states"],
            actions=data["actions"],
            masks=data["masks"],
            colors=data["colors"],
            rewards=data["rewards"],
        )

    @classmethod
    def merge(cls, *datasets: "TransitionDataset") -> "TransitionDataset":
        """Concatenate multiple datasets into one."""
        return cls(
            states=np.concatenate([d.states  for d in datasets]),
            actions=np.concatenate([d.actions for d in datasets]),
            masks=np.concatenate([d.masks   for d in datasets]),
            colors=np.concatenate([d.colors  for d in datasets]),
            rewards=np.concatenate([d.rewards for d in datasets]),
        )


# ------------------------------------------------------------------ #
#  Collector                                                           #
# ------------------------------------------------------------------ #

class DataCollector:
    """
    Play games and record every decision made by *teacher*.

    Parameters
    ----------
    teacher      : agent whose moves we record (acts as the source of
                   supervision signal for imitation learning)
    opponent     : agent to play against — defaults to a fresh SmartAgent
    record_loser : if True, also store losing-game transitions (reward=-1)
                   which is useful for contrastive / RL-from-demos training
    """

    def __init__(
        self,
        teacher:      Agent,
        opponent:     Optional[Agent] = None,
        record_loser: bool = False,
    ) -> None:
        self.teacher      = teacher
        self.opponent     = opponent
        self.record_loser = record_loser

    def collect(self, n_games: int, verbose: bool = False) -> TransitionDataset:
        from uno.strategies.smart_agent import SmartAgent  # lazy import

        opp = self.opponent or SmartAgent(f"opp_{self.teacher.name}")

        all_states:  list[np.ndarray] = []
        all_actions: list[int]        = []
        all_masks:   list[np.ndarray] = []
        all_colors:  list[int]        = []
        all_rewards: list[float]      = []
        wins = 0

        for i in range(n_games):
            recorder = _RecordingAgent(self.teacher)
            # Alternate who goes first to avoid first-player bias
            agents   = [recorder, opp] if i % 2 == 0 else [opp, recorder]
            result   = UnoGame(agents).play()
            records  = recorder.pop_records()

            won    = result.winner == recorder.name
            reward = 1.0 if won else -1.0
            wins  += int(won)

            if won or self.record_loser:
                for r in records:
                    all_states.append(r["state"])
                    all_actions.append(r["action"])
                    all_masks.append(r["mask"])
                    all_colors.append(
                        r["color_action"] if r["color_action"] is not None else -1
                    )
                    all_rewards.append(reward)

            if verbose and (i + 1) % 500 == 0:
                wr = wins / (i + 1) * 100
                print(
                    f"  [{i+1:>6,}/{n_games:,}]  "
                    f"win_rate={wr:.1f}%  "
                    f"transitions={len(all_states):,}"
                )

        return TransitionDataset(
            states=np.array(all_states,  dtype=np.float32),
            actions=np.array(all_actions, dtype=np.int32),
            masks=np.array(all_masks,   dtype=bool),
            colors=np.array(all_colors,  dtype=np.int32),
            rewards=np.array(all_rewards, dtype=np.float32),
        )
