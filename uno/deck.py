from __future__ import annotations
import random
from uno.card import Card, Color, CardType, NON_WILD_COLORS


def build_deck() -> list[Card]:
    cards: list[Card] = []

    for color in NON_WILD_COLORS:
        # 0 appears once per color; 1-9 appear twice
        cards.append(Card(color, CardType.NUMBER, 0))
        for n in range(1, 10):
            cards.append(Card(color, CardType.NUMBER, n))
            cards.append(Card(color, CardType.NUMBER, n))
        # action cards appear twice per color
        for _ in range(2):
            cards.append(Card(color, CardType.SKIP))
            cards.append(Card(color, CardType.REVERSE))
            cards.append(Card(color, CardType.DRAW_TWO))

    # 4 wilds and 4 wild-draw-fours
    for _ in range(4):
        cards.append(Card(Color.WILD, CardType.WILD))
        cards.append(Card(Color.WILD, CardType.WILD_DRAW_FOUR))

    random.shuffle(cards)
    return cards
