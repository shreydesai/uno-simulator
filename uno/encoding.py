"""
Fixed-size encoding of UNO game states and actions for neural-network training.

This module is the bridge between the game engine and any ML model.  Keeping
all feature-engineering here means strategies and training scripts stay clean,
and changes to the representation only need to happen in one place.

Card-type index mapping (0 – 53)
---------------------------------
  0  – 12   Red   (0-9, Skip, Reverse, DrawTwo)
  13 – 25   Green
  26 – 38   Blue
  39 – 51   Yellow
  52         Wild
  53         Wild Draw Four

Action index (0 – 54)
----------------------
  0 – 53   Play card of that type
  54        Draw / pass

Color index (0 – 3)
--------------------
  0 Red   1 Green   2 Blue   3 Yellow

State vector layout  (STATE_DIM = 169)
---------------------------------------
  [  0 –  53]  hand_counts      raw count of each card type held  (max 4)
  [ 54 – 107]  top_card_onehot  one-hot of the top discard card
  [108 – 111]  current_color    one-hot over four non-wild colors
  [112]        opp_hand_total   sum of opponent hand sizes / 108
  [113]        cards_in_deck    deck size / 108
  [114 – 167]  can_play_mask    binary: which card types in hand are playable
  [168]        draw_valid       always 1.0 (bias / draw-is-legal signal)
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from uno.agent import GameState
from uno.card import Card, CardType, Color, NON_WILD_COLORS

# ------------------------------------------------------------------ #
#  Dimension constants                                                 #
# ------------------------------------------------------------------ #

NUM_CARD_TYPES: int = 54
ACTION_DIM:     int = 55   # 54 card plays + 1 draw
COLOR_DIM:      int = 4
DRAW_ACTION:    int = 54

# 54 + 54 + 4 + 1 + 1 + 54 + 1
STATE_DIM: int = NUM_CARD_TYPES + NUM_CARD_TYPES + COLOR_DIM + 1 + 1 + NUM_CARD_TYPES + 1

# ------------------------------------------------------------------ #
#  Internal lookup tables (built once at import time)                  #
# ------------------------------------------------------------------ #

# Slot 10 = Skip, 11 = Reverse, 12 = DrawTwo within each color block
_ACTION_TYPES = [CardType.SKIP, CardType.REVERSE, CardType.DRAW_TWO]

# Precomputed forward mapping
_CARD_TO_IDX: dict[tuple, int] = {}
for _ci, _col in enumerate(NON_WILD_COLORS):
    _base = _ci * 13
    for _n in range(10):
        _CARD_TO_IDX[(_col, CardType.NUMBER, _n)] = _base + _n
    for _ai, _at in enumerate(_ACTION_TYPES):
        _CARD_TO_IDX[(_col, _at, None)] = _base + 10 + _ai
_CARD_TO_IDX[(Color.WILD, CardType.WILD,           None)] = 52
_CARD_TO_IDX[(Color.WILD, CardType.WILD_DRAW_FOUR, None)] = 53


# ------------------------------------------------------------------ #
#  Card ↔ type index                                                   #
# ------------------------------------------------------------------ #

def card_type_idx(card: Card) -> int:
    """Map a card to its canonical type index (0 – 53)."""
    key = (card.color, card.card_type, card.number)
    try:
        return _CARD_TO_IDX[key]
    except KeyError:
        raise ValueError(f"Unknown card: {card!r}")


def idx_to_card_type(idx: int) -> tuple[Color, CardType, Optional[int]]:
    """Inverse of card_type_idx — returns (color, card_type, number)."""
    if idx == 52:
        return Color.WILD, CardType.WILD, None
    if idx == 53:
        return Color.WILD, CardType.WILD_DRAW_FOUR, None
    color = NON_WILD_COLORS[idx // 13]
    slot  = idx % 13
    if slot <= 9:
        return color, CardType.NUMBER, slot
    return color, _ACTION_TYPES[slot - 10], None


# ------------------------------------------------------------------ #
#  Action ↔ card                                                       #
# ------------------------------------------------------------------ #

def card_to_action(card: Optional[Card]) -> int:
    """Card → action index.  ``None`` (draw) → ``DRAW_ACTION``."""
    return DRAW_ACTION if card is None else card_type_idx(card)


def action_to_card(action: int, playable: list[Card]) -> Optional[Card]:
    """
    Action index → the first matching card in *playable*.

    Returns ``None`` for ``DRAW_ACTION`` or when no matching card is found
    (the caller should treat that as a draw).
    """
    if action == DRAW_ACTION:
        return None
    for card in playable:
        if card_type_idx(card) == action:
            return card
    return None


# ------------------------------------------------------------------ #
#  Color ↔ index                                                       #
# ------------------------------------------------------------------ #

def color_to_idx(color: Color) -> int:
    return NON_WILD_COLORS.index(color)


def idx_to_color(idx: int) -> Color:
    return NON_WILD_COLORS[idx]


# ------------------------------------------------------------------ #
#  State encoding                                                      #
# ------------------------------------------------------------------ #

def encode_state(state: GameState) -> np.ndarray:
    """
    Encode a ``GameState`` into a fixed-size ``float32`` vector of shape
    ``(STATE_DIM,)`` = ``(169,)``.
    """
    vec    = np.zeros(STATE_DIM, dtype=np.float32)
    offset = 0

    # 1. Hand card-type counts (raw; max 4 per type)
    for card in state.hand:
        vec[offset + card_type_idx(card)] += 1.0
    offset += NUM_CARD_TYPES                                  # → 54

    # 2. Top-card one-hot
    vec[offset + card_type_idx(state.top_card)] = 1.0
    offset += NUM_CARD_TYPES                                  # → 108

    # 3. Current color one-hot (4 dims; zero if wild color somehow active)
    if state.current_color in NON_WILD_COLORS:
        vec[offset + color_to_idx(state.current_color)] = 1.0
    offset += COLOR_DIM                                       # → 112

    # 4. Total opponent hand cards, normalised
    vec[offset] = sum(state.opponent_hand_sizes.values()) / 108.0
    offset += 1                                               # → 113

    # 5. Deck size, normalised
    vec[offset] = state.cards_in_deck / 108.0
    offset += 1                                               # → 114

    # 6. Can-play mask — which card types in hand are legally playable now
    for card in state.playable_cards():
        vec[offset + card_type_idx(card)] = 1.0
    offset += NUM_CARD_TYPES                                  # → 168

    # 7. Draw-always-valid bias feature
    vec[offset] = 1.0                                         # → 169

    return vec


def build_action_mask(state: GameState) -> np.ndarray:
    """
    Boolean mask of shape ``(ACTION_DIM,)`` = ``(55,)``.
    ``True`` at index ``i`` means action ``i`` is currently legal.

    If no card can be played, only the draw action (54) is legal.
    """
    mask     = np.zeros(ACTION_DIM, dtype=bool)
    playable = state.playable_cards()
    if playable:
        for card in playable:
            mask[card_type_idx(card)] = True
    else:
        mask[DRAW_ACTION] = True
    return mask
