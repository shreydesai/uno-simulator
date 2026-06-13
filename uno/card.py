from __future__ import annotations
from enum import Enum
from dataclasses import dataclass
from typing import Optional


class Color(Enum):
    RED = "Red"
    GREEN = "Green"
    BLUE = "Blue"
    YELLOW = "Yellow"
    WILD = "Wild"


class CardType(Enum):
    NUMBER = "Number"
    SKIP = "Skip"
    REVERSE = "Reverse"
    DRAW_TWO = "DrawTwo"
    WILD = "Wild"
    WILD_DRAW_FOUR = "WildDrawFour"


NON_WILD_COLORS = [Color.RED, Color.GREEN, Color.BLUE, Color.YELLOW]


@dataclass(frozen=True)
class Card:
    color: Color
    card_type: CardType
    number: Optional[int] = None  # only for NUMBER cards

    def is_wild(self) -> bool:
        return self.color == Color.WILD

    def can_play_on(self, top_card: Card, current_color: Color) -> bool:
        if self.is_wild():
            return True
        if self.color == current_color:
            return True
        # match by face value only when top card is not wild
        if not top_card.is_wild() and self.card_type == top_card.card_type:
            if self.card_type == CardType.NUMBER:
                return self.number == top_card.number
            return True
        return False

    def __str__(self) -> str:
        if self.card_type == CardType.NUMBER:
            return f"{self.color.value} {self.number}"
        return f"{self.color.value} {self.card_type.value}"

    def __repr__(self) -> str:
        return str(self)
