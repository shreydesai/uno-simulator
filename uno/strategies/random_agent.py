from __future__ import annotations

import random
from typing import Optional

from uno.agent import Agent, GameState
from uno.card import Card, Color, NON_WILD_COLORS


class RandomAgent(Agent):
    """Uniformly random valid card; random color when playing a wild."""

    def choose_card(self, state: GameState) -> Optional[Card]:
        playable = state.playable_cards()
        return random.choice(playable) if playable else None

    def choose_color(self, state: GameState) -> Color:
        return random.choice(NON_WILD_COLORS)
