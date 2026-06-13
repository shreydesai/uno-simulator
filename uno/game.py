from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Optional

from uno.agent import Agent, GameState
from uno.card import Card, Color, CardType
from uno.deck import build_deck


@dataclass
class GameResult:
    winner: str
    turns: int
    final_hand_sizes: dict[str, int]


class UnoGame:
    """
    Drives a complete UNO game between two or more agents.

    Parameters
    ----------
    agents    : ordered list of players (turn order follows list order initially)
    hand_size : cards dealt to each player at the start (default 7)
    max_turns : safety cap to prevent infinite games (default 500)
    """

    def __init__(
        self,
        agents: list[Agent],
        hand_size: int = 7,
        max_turns: int = 500,
    ) -> None:
        if len(agents) < 2:
            raise ValueError("Need at least 2 agents")
        self.agents    = agents
        self.hand_size = hand_size
        self.max_turns = max_turns

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def play(self, verbose: bool = False) -> GameResult:
        deck: list[Card]              = build_deck()
        hands: dict[str, list[Card]]  = {a.name: [] for a in self.agents}

        # Deal
        for agent in self.agents:
            hands[agent.name] = [deck.pop() for _ in range(self.hand_size)]

        # First card must not be a wild
        discard: list[Card] = []
        while True:
            card = deck.pop()
            if not card.is_wild():
                discard.append(card)
                break
            deck.insert(0, card)

        current_color: Color = discard[-1].color
        direction: int       = 1     # +1 clockwise, -1 counter-clockwise
        n: int               = len(self.agents)
        current_idx: int     = 0

        # Apply effect of the opening card
        opening = discard[0]
        if opening.card_type == CardType.SKIP:
            current_idx = self._advance(current_idx, direction, n)
        elif opening.card_type == CardType.REVERSE:
            direction = -1
            if n == 2:                           # Reverse == Skip in 2-player
                current_idx = self._advance(current_idx, direction, n)
        elif opening.card_type == CardType.DRAW_TWO:
            first = self._advance(current_idx, direction, n)
            for _ in range(2):
                self._draw(deck, discard, hands[self.agents[first].name])
            current_idx = self._advance(first, direction, n)

        for turn in range(self.max_turns):
            agent   = self.agents[current_idx]
            hand    = hands[agent.name]
            playable = [c for c in hand if c.can_play_on(discard[-1], current_color)]

            if verbose:
                opp = {nm: len(h) for nm, h in hands.items() if nm != agent.name}
                print(
                    f"T{turn+1:03d} | {agent.name:22s} | "
                    f"hand={len(hand):3d} | top={str(discard[-1]):25s} | "
                    f"color={current_color.value:6s} | opp={opp}"
                )

            played: Optional[Card] = None

            if playable:
                state  = self._make_state(agent.name, hand, discard, current_color, hands, deck)
                choice = agent.choose_card(state)
                if choice is None or choice not in playable:
                    choice = random.choice(playable)  # enforce must-play rule
                played = choice
            else:
                drawn = self._draw(deck, discard, hand)
                if verbose:
                    print(f"        draws {drawn}")
                if drawn.can_play_on(discard[-1], current_color):
                    state  = self._make_state(agent.name, hand, discard, current_color, hands, deck)
                    choice = agent.choose_card(state)
                    if choice == drawn:
                        played = drawn

            skip_next = False

            if played is not None:
                hand.remove(played)
                discard.append(played)

                if played.is_wild():
                    col_state     = self._make_state(agent.name, hand, discard, current_color, hands, deck)
                    current_color = agent.choose_color(col_state)
                else:
                    current_color = played.color

                if verbose:
                    suffix = f" → {current_color.value}" if played.is_wild() else ""
                    print(f"        plays {played}{suffix}  (hand→{len(hand)})")

                ct = played.card_type
                if ct == CardType.SKIP:
                    skip_next = True
                elif ct == CardType.REVERSE:
                    direction *= -1
                    if n == 2:
                        skip_next = True
                elif ct == CardType.DRAW_TWO:
                    nxt = self._advance(current_idx, direction, n)
                    for _ in range(2):
                        self._draw(deck, discard, hands[self.agents[nxt].name])
                    if verbose:
                        print(f"        {self.agents[nxt].name} draws 2")
                    skip_next = True
                elif ct == CardType.WILD_DRAW_FOUR:
                    nxt = self._advance(current_idx, direction, n)
                    for _ in range(4):
                        self._draw(deck, discard, hands[self.agents[nxt].name])
                    if verbose:
                        print(f"        {self.agents[nxt].name} draws 4")
                    skip_next = True

            # Check for win
            if len(hand) == 0:
                if verbose:
                    print(f"\n>>> {agent.name} wins in {turn + 1} turns!")
                return GameResult(
                    winner=agent.name,
                    turns=turn + 1,
                    final_hand_sizes={a.name: len(hands[a.name]) for a in self.agents},
                )

            current_idx = self._advance(current_idx, direction, n)
            if skip_next:
                current_idx = self._advance(current_idx, direction, n)

        # Timeout — award win to the player with the fewest cards
        winner = min(self.agents, key=lambda a: len(hands[a.name]))
        return GameResult(
            winner=winner.name,
            turns=self.max_turns,
            final_hand_sizes={a.name: len(hands[a.name]) for a in self.agents},
        )

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _advance(idx: int, direction: int, n: int) -> int:
        return (idx + direction) % n

    @staticmethod
    def _draw(deck: list[Card], discard: list[Card], hand: list[Card]) -> Card:
        """Draw one card, reshuffling the discard pile if the deck is empty."""
        if not deck:
            top = discard.pop()
            random.shuffle(discard)
            deck.extend(discard)
            discard.clear()
            discard.append(top)
        card = deck.pop()
        hand.append(card)
        return card

    def _make_state(
        self,
        agent_name: str,
        hand: list[Card],
        discard: list[Card],
        current_color: Color,
        hands: dict[str, list[Card]],
        deck: list[Card],
    ) -> GameState:
        return GameState(
            hand=list(hand),
            top_card=discard[-1],
            current_color=current_color,
            opponent_hand_sizes={nm: len(h) for nm, h in hands.items() if nm != agent_name},
            discard_pile=list(discard),
            cards_in_deck=len(deck),
        )
