"""
Supervised (imitation) learning from teacher-agent trajectories.

Pipeline
--------
Step 1 — Collect data from a teacher agent:
    python -m training.supervised --collect --games 10000

Step 2 — Train the model with cross-entropy loss:
    python -m training.supervised --train --epochs 20

Step 3 — Evaluate the trained model:
    python -m training.supervised --eval --games 2000

Loss
----
    L = CE(card_logits_masked, card_action)
      + 0.5 * CE(color_logits, color_action)    # only on wild-card turns

The card logits are masked to illegal actions before computing CE so the
model is never rewarded for assigning probability to unplayable cards.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from training.data_collector import DataCollector, TransitionDataset
from training.dataset import UnoTorchDataset
from training.model import UnoNet

DATA_PATH = "data/supervised.npz"
CKPT_PATH = "checkpoints/supervised.pt"


# ------------------------------------------------------------------ #
#  Training loop                                                       #
# ------------------------------------------------------------------ #

def train(
    dataset:    TransitionDataset,
    model:      UnoNet,
    epochs:     int   = 20,
    batch_size: int   = 512,
    lr:         float = 3e-4,
    device:     str   = "cpu",
    val_split:  float = 0.1,
) -> UnoNet:
    torch_ds = UnoTorchDataset(dataset, winners_only=True)
    n_val    = max(1, int(len(torch_ds) * val_split))
    n_train  = len(torch_ds) - n_val
    train_ds, val_ds = random_split(torch_ds, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=0)

    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce    = nn.CrossEntropyLoss()

    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        # ── Train ──────────────────────────────────────────────────────
        model.train()
        t_loss = t_acc = n_train_steps = 0

        for batch in train_loader:
            states  = batch["state"].to(device)
            actions = batch["action"].to(device)
            masks   = batch["mask"].to(device)
            colors  = batch["color"].to(device)

            card_logits, color_logits = model(states)

            # Mask illegal card actions
            card_logits_m = card_logits.clone()
            card_logits_m[~masks] = float("-inf")
            card_loss = ce(card_logits_m, actions)

            # Color loss — only on turns where a wild was played
            wild_mask  = colors >= 0
            color_loss = (
                ce(color_logits[wild_mask], colors[wild_mask])
                if wild_mask.any()
                else torch.tensor(0.0, device=device)
            )

            loss = card_loss + 0.5 * color_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            preds           = card_logits_m.argmax(dim=1)
            t_acc          += (preds == actions).float().sum().item()
            t_loss         += loss.item() * len(states)
            n_train_steps  += len(states)

        sched.step()

        # ── Validate ───────────────────────────────────────────────────
        model.eval()
        v_loss = v_acc = n_val_steps = 0

        with torch.no_grad():
            for batch in val_loader:
                states  = batch["state"].to(device)
                actions = batch["action"].to(device)
                masks   = batch["mask"].to(device)

                card_logits, _ = model(states)
                card_logits[~masks] = float("-inf")
                loss  = ce(card_logits, actions)
                preds = card_logits.argmax(dim=1)

                v_acc      += (preds == actions).float().sum().item()
                v_loss     += loss.item() * len(states)
                n_val_steps += len(states)

        train_acc = t_acc  / n_train_steps
        val_acc   = v_acc  / n_val_steps
        val_loss  = v_loss / n_val_steps

        print(
            f"Epoch {epoch:3d}/{epochs}  "
            f"train_loss={t_loss/n_train_steps:.4f}  train_acc={train_acc:.3f}  "
            f"val_loss={val_loss:.4f}  val_acc={val_acc:.3f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            model.save(CKPT_PATH)
            print(f"  ✓ new best → {CKPT_PATH}")

    return model


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(model_path: str, n_games: int, device: str = "cpu") -> None:
    from simulate import run_matchup, print_matchup
    from uno.strategies.greedy_agent import GreedyAgent
    from uno.strategies.smart_agent import SmartAgent
    from uno.strategies.nn_agent import NNAgent

    model    = UnoNet.load(model_path)
    nn_agent = NNAgent("NN-Supervised", model, device=device, greedy=True)

    for label, opp in [
        ("NN-Supervised  vs  Greedy", GreedyAgent("Greedy")),
        ("NN-Supervised  vs  Smart",  SmartAgent("Smart")),
    ]:
        stats = run_matchup(nn_agent, opp, n_games=n_games)
        print_matchup(label, stats)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised UNO training")
    parser.add_argument("--collect",  action="store_true", help="Collect game data")
    parser.add_argument("--train",    action="store_true", help="Train the model")
    parser.add_argument("--eval",     action="store_true", help="Evaluate a saved model")
    parser.add_argument("--games",    type=int,   default=10_000)
    parser.add_argument("--epochs",   type=int,   default=20)
    parser.add_argument("--batch",    type=int,   default=512)
    parser.add_argument("--lr",       type=float, default=3e-4)
    parser.add_argument("--device",   type=str,   default="cpu")
    args = parser.parse_args()

    if args.collect:
        from uno.strategies.smart_agent import SmartAgent
        print(f"Collecting {args.games:,} games from SmartAgent...")
        collector = DataCollector(SmartAgent("Smart"), record_loser=False)
        td = collector.collect(args.games, verbose=True)
        td.save(DATA_PATH)

    if args.train:
        td    = TransitionDataset.load(DATA_PATH)
        model = UnoNet()
        print(
            f"Training on {len(td):,} winning transitions "
            f"for {args.epochs} epochs..."
        )
        train(td, model, epochs=args.epochs, batch_size=args.batch,
              lr=args.lr, device=args.device)

    if args.eval:
        evaluate(CKPT_PATH, n_games=args.games, device=args.device)


if __name__ == "__main__":
    main()
