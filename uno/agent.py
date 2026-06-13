from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

from uno.card import Card, Color


@dataclass
class GameState:
    """
    Snapshot of observable game state delivered to an agent each turn.

    Agents may read every field freely — in a real tournament you might
    hide the discard pile or deck count, but for research purposes full
    information is available so agents can implement more sophisticated
    strategies.
    """

    hand: list[Card]
    top_card: Card
    current_color: Color            # Active color; differs from top_card.color after a wild
    opponent_hand_sizes: dict[str, int]
    discard_pile: list[Card]        # Oldest → newest
    cards_in_deck: int

    def playable_cards(self) -> list[Card]:
        """Cards in hand that can legally be played right now."""
        return [c for c in self.hand if c.can_play_on(self.top_card, self.current_color)]


class Agent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name

    @abstractmethod
    def choose_card(self, state: GameState) -> Optional[Card]:
        """
        Return a card from *state.hand* to play, or ``None`` to draw / pass.

        If ``state.playable_cards()`` is non-empty the game engine treats a
        ``None`` return (or an invalid card) as a rule violation and picks a
        random valid card on the agent's behalf.
        """
        ...

    @abstractmethod
    def choose_color(self, state: GameState) -> Color:
        """
        Called immediately after the agent plays a wild card.
        Must return one of the four non-wild colors.
        """
        ...
