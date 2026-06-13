from __future__ import annotations

import random
from typing import Optional

from uno.agent import Agent, GameState
from uno.card import Card, Color, CardType
from uno.strategies.greedy_agent import dominant_color


class SmartAgent(Agent):
    """
    Multi-heuristic UNO strategy.

    Rules applied in order
    ----------------------
    1. **Danger response** — if any opponent has ≤ 2 cards, immediately play
       the most punishing card (+4 > +2 > Skip > Reverse > Wild).
    2. **Finishing setup** — with exactly 2 cards left, lead with the
       non-wild so the wild becomes a guaranteed closing card.
    3. **Action cards** — prefer Draw+2 and Skip over plain number cards;
       among those, favour cards matching the current color to maintain
       color control.
    4. **Color steering** — number cards are picked from the dominant color
       so that the hand stays easy to play after the turn.
    5. **Wild conservation** — plain Wild before Wild+4; Wild+4 only as last
       resort (saving the nuclear option).
    """

    def choose_card(self, state: GameState) -> Optional[Card]:
        playable = state.playable_cards()
        if not playable:
            return None

        opp_min = min(state.opponent_hand_sizes.values(), default=7)
        danger  = opp_min <= 2

        w4       = [c for c in playable if c.card_type == CardType.WILD_DRAW_FOUR]
        d2       = [c for c in playable if c.card_type == CardType.DRAW_TWO]
        skips    = [c for c in playable if c.card_type == CardType.SKIP]
        reverses = [c for c in playable if c.card_type == CardType.REVERSE]
        wilds    = [c for c in playable if c.card_type == CardType.WILD]
        numbers  = [c for c in playable if c.card_type == CardType.NUMBER]

        # ── 1. Opponent about to win ──────────────────────────────────────
        if danger:
            for group in [w4, d2, skips, reverses]:
                if group:
                    return group[0]
            if wilds:
                return wilds[0]

        # ── 2. Finishing setup ────────────────────────────────────────────
        if len(state.hand) == 2:
            non_wilds = [c for c in playable if not c.is_wild()]
            if non_wilds:
                other = next((c for c in state.hand if c not in non_wilds), None)
                if other and other.is_wild():
                    return non_wilds[0]

        # ── 3. Action cards (colour-matched first) ────────────────────────
        for group in [d2, skips]:
            if group:
                same = [c for c in group if c.color == state.current_color]
                return same[0] if same else group[0]

        # ── 4. Number cards from dominant colour ──────────────────────────
        dom      = dominant_color(state.hand)
        dom_nums = [c for c in numbers if c.color == dom]
        if dom_nums:
            return max(dom_nums, key=lambda c: c.number or 0)
        if numbers:
            return max(numbers, key=lambda c: c.number or 0)

        # ── 5. Reverses (lower disruption value) ─────────────────────────
        if reverses:
            return reverses[0]

        # ── 6. Wild (switch to dominant colour) ──────────────────────────
        if wilds:
            return wilds[0]

        # ── 7. Wild+4 last resort ────────────────────────────────────────
        if w4:
            return w4[0]

        return random.choice(playable)

    def choose_color(self, state: GameState) -> Color:
        return dominant_color(state.hand)
