#!/usr/bin/env python3
"""
UNO strategy simulator — pit agents against each other over many games.

Usage
-----
    python simulate.py                   # run all matchups, 10k games each
    python simulate.py -n 50000          # more games for tighter estimates
    python simulate.py --sample          # print a verbose sample game
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from statistics import mean, stdev

from uno.game import UnoGame
from uno.strategies.greedy_agent import GreedyAgent
from uno.strategies.random_agent import RandomAgent
from uno.strategies.smart_agent import SmartAgent


# ------------------------------------------------------------------ #
#  Core runner (also imported by training scripts for evaluation)      #
# ------------------------------------------------------------------ #

def run_matchup(
    agent1,
    agent2,
    n_games: int = 10_000,
    seed: int | None = None,
) -> dict:
    """
    Play *n_games* between agent1 and agent2, alternating who goes first.
    Returns a stats dict with win rates, average game length, etc.
    """
    if seed is not None:
        random.seed(seed)

    wins: dict[str, int] = defaultdict(int)
    turn_counts: list[int] = []

    for i in range(n_games):
        agents = [agent1, agent2] if i % 2 == 0 else [agent2, agent1]
        result = UnoGame(agents).play()
        wins[result.winner] += 1
        turn_counts.append(result.turns)

    return {
        "wins": dict(wins),
        "win_rates": {
            agent1.name: wins[agent1.name] / n_games,
            agent2.name: wins[agent2.name] / n_games,
        },
        "avg_turns": mean(turn_counts),
        "std_turns": stdev(turn_counts) if len(turn_counts) > 1 else 0.0,
        "n_games":   n_games,
    }


def print_matchup(title: str, stats: dict) -> None:
    print(f"\n{'─' * 58}")
    print(f"  {title}")
    print(f"{'─' * 58}")
    print(
        f"  Games : {stats['n_games']:,}  |  "
        f"Avg turns : {stats['avg_turns']:.1f} ± {stats['std_turns']:.1f}"
    )
    for name, rate in sorted(stats["win_rates"].items(), key=lambda x: -x[1]):
        bar = "█" * int(rate * 34)
        print(f"  {name:26s}  {rate * 100:5.1f}%  {bar}")


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="UNO Strategy Simulator")
    parser.add_argument("-n", "--games", type=int,  default=10_000,
                        help="Games per matchup (default 10,000)")
    parser.add_argument("--seed",        type=int,  default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--sample",      action="store_true",
                        help="Print a verbose walkthrough of one game")
    args = parser.parse_args()

    print(
        f"\n  UNO Strategy Simulator "
        f"— {args.games:,} games per matchup  (seed={args.seed})"
    )

    matchups = [
        ("Random   vs  Greedy", RandomAgent("Random"), GreedyAgent("Greedy")),
        ("Random   vs  Smart",  RandomAgent("Random"), SmartAgent("Smart")),
        ("Greedy   vs  Smart",  GreedyAgent("Greedy"), SmartAgent("Smart")),
    ]

    for title, a1, a2 in matchups:
        stats = run_matchup(a1, a2, n_games=args.games, seed=args.seed)
        print_matchup(title, stats)

    if args.sample:
        print("\n\n─── Sample game: Greedy vs Smart ───\n")
        random.seed(args.seed)
        result = UnoGame([GreedyAgent("Greedy"), SmartAgent("Smart")]).play(verbose=True)
        print(f"\nWinner : {result.winner}")
        print(f"Turns  : {result.turns}")
        print(f"Hands  : {result.final_hand_sizes}")


if __name__ == "__main__":
    main()
