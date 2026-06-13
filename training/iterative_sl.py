"""
Iterative Supervised Learning (ISL) with self-play data generation.

Motivation
----------
Pure RL (PPO) struggled because:
  - Sparse ±1 terminal reward → weak gradient through 30+ turns
  - Pure self-play → gradients cancel when both players share weights

This approach sidesteps both problems:
  - Collect self-play games between the current model and itself
  - Keep only the WINNER's transitions as supervised training examples
  - Retrain with cross-entropy (dense, stable gradient on every step)
  - Repeat — each round the model gets harder training data

This is a variant of Expert Iteration / DAgger, where the "expert"
is the winning side of the previous model playing against itself.

Round structure
---------------
  Round 0 : Train on SmartAgent games                  → NN-SL  (baseline)
  Round 1 : Run NN-SL vs NN-SL, collect winners        → NN-ISL-1
  Round 2 : Run NN-ISL-1 vs NN-ISL-1, collect winners  → NN-ISL-2
  ...

Data mixing
-----------
Each round blends self-play winner data with a fraction of the original
Smart data to prevent catastrophic forgetting of basic card-play rules.

Usage
-----
    # Standard run (5 rounds, warm-start from supervised checkpoint)
    python -m training.iterative_sl

    # More rounds with more data per round
    python -m training.iterative_sl --rounds 8 --games 10000 --epochs 20

    # No mixing — pure self-play data only
    python -m training.iterative_sl --mix-ratio 0.0
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np

from training.data_collector import TransitionDataset, collect_self_play_data
from training.model import UnoNet
from training.supervised import train as sl_train

SL_DATA_PATH = "data/supervised.npz"
SL_CKPT_PATH = "checkpoints/supervised.pt"
ISL_DIR      = "checkpoints/isl"


# ------------------------------------------------------------------ #
#  Helpers                                                             #
# ------------------------------------------------------------------ #

def _sample_dataset(ds: TransitionDataset, n: int) -> TransitionDataset:
    """Random subsample of n rows from ds (without replacement if possible)."""
    idx = np.random.choice(len(ds), min(n, len(ds)), replace=False)
    return TransitionDataset(
        states=ds.states[idx],
        actions=ds.actions[idx],
        masks=ds.masks[idx],
        colors=ds.colors[idx],
        rewards=ds.rewards[idx],
    )


def _eval_vs_smart(model: UnoNet, n_games: int = 2_000, device: str = "cpu") -> dict:
    """Quick evaluation: win rates vs Greedy and Smart."""
    from simulate import run_matchup
    from uno.strategies.greedy_agent import GreedyAgent
    from uno.strategies.smart_agent import SmartAgent
    from uno.strategies.nn_agent import NNAgent

    random.seed(0)
    nn = NNAgent("NN", model, device=device, greedy=True)
    return {
        "vs_greedy": run_matchup(nn, GreedyAgent("Greedy"), n_games=n_games)["win_rates"]["NN"],
        "vs_smart":  run_matchup(nn, SmartAgent("Smart"),   n_games=n_games)["win_rates"]["NN"],
    }


# ------------------------------------------------------------------ #
#  Main loop                                                           #
# ------------------------------------------------------------------ #

def run_iterative_sl(
    n_rounds:        int   = 5,
    games_per_round: int   = 5_000,
    epochs_per_round:int   = 15,
    batch_size:      int   = 512,
    lr:              float = 1e-4,     # lower than initial SL — fine-tuning rate
    mix_ratio:       float = 0.3,      # fraction of original SL data blended in
    device:          str   = "cpu",
    init_ckpt:       str   = SL_CKPT_PATH,
    sl_data_path:    str   = SL_DATA_PATH,
    eval_games:      int   = 2_000,
) -> UnoNet:
    """
    Run *n_rounds* of iterative supervised learning.

    Parameters
    ----------
    n_rounds         : number of self-play → retrain iterations
    games_per_round  : self-play games to collect each round
    epochs_per_round : CE training epochs per round
    lr               : learning rate for fine-tuning (lower than initial SL)
    mix_ratio        : fraction of the batch drawn from original Smart data
                       (0 = pure self-play; 0.3 = 30% Smart + 70% self-play)
    init_ckpt        : starting checkpoint (default: supervised.pt)
    sl_data_path     : original SL dataset for data mixing
    eval_games       : games per evaluation run
    """
    os.makedirs(ISL_DIR, exist_ok=True)

    # Load initial model and (optionally) original SL data for mixing
    model = UnoNet.load(init_ckpt)

    sl_data: TransitionDataset | None = None
    if mix_ratio > 0 and os.path.exists(sl_data_path):
        sl_data = TransitionDataset.load(sl_data_path)
        print(f"Loaded SL data for mixing: {len(sl_data):,} transitions")
    elif mix_ratio > 0:
        print(f"[warn] mix_ratio={mix_ratio} but {sl_data_path} not found — skipping mix")

    # Baseline eval
    print(f"\n{'═'*58}")
    print(f"  Baseline (before any self-play rounds)")
    print(f"{'═'*58}")
    baseline = _eval_vs_smart(model, n_games=eval_games, device=device)
    print(f"  vs Greedy : {baseline['vs_greedy']*100:.1f}%")
    print(f"  vs Smart  : {baseline['vs_smart']*100:.1f}%  ← target to beat")
    best_vs_smart = baseline["vs_smart"]

    for r in range(1, n_rounds + 1):
        print(f"\n{'═'*58}")
        print(f"  Round {r}/{n_rounds}")
        print(f"{'═'*58}")

        # ── 1. Collect self-play data ──────────────────────────────────
        print(f"\n[1] Collecting {games_per_round:,} self-play games (sampling)...")
        model.eval()
        sp_data = collect_self_play_data(
            model,
            n_games=games_per_round,
            greedy=False,    # sample to get diverse gameplay
            record_loser=False,
            verbose=True,
        )
        print(f"    → {len(sp_data):,} winner transitions")

        # ── 2. Mix with original SL data ───────────────────────────────
        if sl_data is not None and mix_ratio > 0:
            n_sl = int(len(sp_data) * mix_ratio / max(1 - mix_ratio, 1e-6))
            sl_sample = _sample_dataset(sl_data, n_sl)
            mixed = TransitionDataset.merge(sp_data, sl_sample)
            print(
                f"[2] Mixed: {len(sp_data):,} self-play"
                f" + {len(sl_sample):,} Smart = {len(mixed):,} total"
            )
        else:
            mixed = sp_data
            print(f"[2] No mixing — using pure self-play data")

        # ── 3. Retrain with CE loss ────────────────────────────────────
        print(f"[3] Training for {epochs_per_round} epochs (lr={lr:.1e})...")
        model.train()
        model = sl_train(
            mixed,
            model,
            epochs=epochs_per_round,
            batch_size=batch_size,
            lr=lr,
            device=device,
        )

        # ── 4. Evaluate ────────────────────────────────────────────────
        print(f"[4] Evaluating ({eval_games:,} games each)...")
        model.eval()
        rates = _eval_vs_smart(model, n_games=eval_games, device=device)
        g, s  = rates["vs_greedy"] * 100, rates["vs_smart"] * 100
        delta = s / 100 - best_vs_smart
        arrow = "▲" if delta > 0 else "▼"
        print(
            f"    vs Greedy : {g:.1f}%\n"
            f"    vs Smart  : {s:.1f}%  "
            f"{arrow} {abs(delta)*100:+.1f}pp vs best so far"
        )

        if rates["vs_smart"] > best_vs_smart:
            best_vs_smart = rates["vs_smart"]

        # ── 5. Save checkpoint ─────────────────────────────────────────
        ckpt = os.path.join(ISL_DIR, f"round_{r}.pt")
        model.save(ckpt)
        print(f"    Saved → {ckpt}")

    print(f"\n{'═'*58}")
    print(f"  ISL complete.  Best vs Smart: {best_vs_smart*100:.1f}%")
    print(f"{'═'*58}\n")
    return model


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Iterative SL with self-play data")
    parser.add_argument("--rounds",    type=int,   default=5,
                        help="Number of self-play → retrain rounds")
    parser.add_argument("--games",     type=int,   default=5_000,
                        help="Self-play games to collect per round")
    parser.add_argument("--epochs",    type=int,   default=15,
                        help="CE training epochs per round")
    parser.add_argument("--batch",     type=int,   default=512)
    parser.add_argument("--lr",        type=float, default=1e-4,
                        help="Fine-tuning LR (lower than initial SL)")
    parser.add_argument("--mix-ratio", type=float, default=0.3,
                        help="Fraction of original Smart data blended in (0=pure SP)")
    parser.add_argument("--eval-games",type=int,   default=2_000)
    parser.add_argument("--device",    type=str,   default="cpu")
    parser.add_argument("--init",      type=str,   default=SL_CKPT_PATH,
                        help="Starting checkpoint path")
    args = parser.parse_args()

    run_iterative_sl(
        n_rounds=args.rounds,
        games_per_round=args.games,
        epochs_per_round=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        mix_ratio=args.mix_ratio,
        device=args.device,
        init_ckpt=args.init,
        eval_games=args.eval_games,
    )


if __name__ == "__main__":
    main()
