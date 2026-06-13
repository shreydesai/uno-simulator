from __future__ import annotations

import random
from collections import Counter
from typing import Optional

from uno.agent import Agent, GameState
from uno.card import Card, Color, CardType, NON_WILD_COLORS


# Higher value → play this card first
_PRIORITY: dict[CardType, int] = {
    CardType.WILD_DRAW_FOUR: 60,
    CardType.DRAW_TWO:       50,
    CardType.SKIP:           40,
    CardType.REVERSE:        30,
    CardType.WILD:           20,
    CardType.NUMBER:          0,
}


def dominant_color(hand: list[Card]) -> Color:
    """Most frequent non-wild color in *hand*; random if hand is all wilds."""
    counts = Counter(c.color for c in hand if c.color != Color.WILD)
    if not counts:
        return random.choice(NON_WILD_COLORS)
    return counts.most_common(1)[0][0]


class GreedyAgent(Agent):
    """
    Always plays the highest-priority card available.

    Priority: Wild+4 > Draw+2 > Skip > Reverse > Wild > Number (highest digit).
    Color choice: most frequent color in remaining hand.
    """

    def choose_card(self, state: GameState) -> Optional[Card]:
        playable = state.playable_cards()
        if not playable:
            return None
        return max(playable, key=lambda c: (_PRIORITY[c.card_type], c.number or 0))

    def choose_color(self, state: GameState) -> Color:
        return dominant_color(state.hand)
