"""
PyTorch Dataset wrapper for ``TransitionDataset``.

Each ``__getitem__`` returns a dict:
  state   : FloatTensor  (STATE_DIM,)
  action  : LongTensor   scalar
  mask    : BoolTensor   (ACTION_DIM,)
  color   : LongTensor   scalar   (-1 means no color was chosen this step)
  reward  : FloatTensor  scalar
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from training.data_collector import TransitionDataset


class UnoTorchDataset(Dataset):
    """
    Parameters
    ----------
    dataset      : a ``TransitionDataset`` instance
    winners_only : if True, only include transitions where reward > 0
                   (typical for imitation learning from winning games)
    """

    def __init__(
        self,
        dataset:      TransitionDataset,
        winners_only: bool = False,
    ) -> None:
        if winners_only:
            idx = dataset.rewards > 0
            self.states  = torch.tensor(dataset.states[idx],  dtype=torch.float32)
            self.actions = torch.tensor(dataset.actions[idx], dtype=torch.long)
            self.masks   = torch.tensor(dataset.masks[idx],   dtype=torch.bool)
            self.colors  = torch.tensor(dataset.colors[idx],  dtype=torch.long)
            self.rewards = torch.tensor(dataset.rewards[idx], dtype=torch.float32)
        else:
            self.states  = torch.tensor(dataset.states,  dtype=torch.float32)
            self.actions = torch.tensor(dataset.actions, dtype=torch.long)
            self.masks   = torch.tensor(dataset.masks,   dtype=torch.bool)
            self.colors  = torch.tensor(dataset.colors,  dtype=torch.long)
            self.rewards = torch.tensor(dataset.rewards, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.states)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "state":  self.states[idx],
            "action": self.actions[idx],
            "mask":   self.masks[idx],
            "color":  self.colors[idx],
            "reward": self.rewards[idx],
        }
