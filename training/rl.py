"""
Reinforcement learning training for UNO.

Algorithms
----------
REINFORCE  (implemented)
    Vanilla policy gradient (Williams 1992) with entropy regularisation.
    Simple, unbiased, but high variance — good first RL baseline.

PPO / A2C  (extension points, see comments marked "# PPO:")
    To add PPO: add a value head to UnoNet, collect multiple episodes per
    update, compute GAE advantages, clip the policy ratio.  The main loop
    structure below already supports it — swap the loss computation section.

Usage
-----
    # Train against SmartAgent
    python -m training.rl --episodes 50000 --opponent smart

    # Resume from checkpoint
    python -m training.rl --episodes 50000 --resume

    # Evaluate a saved model
    python -m training.rl --eval --eval-games 2000
"""

from __future__ import annotations

import argparse
import os
from collections import deque
from statistics import mean

import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import UnoNet
from uno.game import UnoGame
from uno.strategies.greedy_agent import GreedyAgent
from uno.strategies.nn_agent import NNAgent, Transition
from uno.strategies.smart_agent import SmartAgent

CKPT_PATH = "checkpoints/rl_reinforce.pt"


# ------------------------------------------------------------------ #
#  REINFORCE                                                           #
# ------------------------------------------------------------------ #

def reinforce(
    model:           UnoNet,
    opponent_factory,
    n_episodes:  int   = 50_000,
    lr:          float = 1e-4,
    gamma:       float = 0.99,
    entropy_coef:float = 0.01,
    log_every:   int   = 500,
    save_every:  int   = 5_000,
    device:      str   = "cpu",
) -> UnoNet:
    """
    Vanilla REINFORCE with entropy regularisation.

    At the end of each episode the discounted return G_t is computed
    for every decision step and the policy is updated via:

        L = -E[ log π(a_t | s_t) * G_t ] - β * H(π)

    where H(π) is the entropy of the masked action distribution.

    Parameters
    ----------
    model            : UnoNet to train (modified in-place)
    opponent_factory : callable → Agent  (called fresh every episode)
    n_episodes       : total training episodes
    lr               : Adam learning rate
    gamma            : discount factor
    entropy_coef     : weight of the entropy bonus  (β)
    log_every        : print stats every N episodes
    save_every       : save checkpoint every N episodes
    device           : torch device string
    """
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)

    recent_wins: deque[int]   = deque(maxlen=log_every)
    recent_lens: deque[int]   = deque(maxlen=log_every)
    recent_loss: deque[float] = deque(maxlen=log_every)

    for ep in range(1, n_episodes + 1):
        # ── Collect episode ───────────────────────────────────────────
        nn_agent = NNAgent("NN", model, device=device, greedy=False, training=True)
        opponent = opponent_factory()
        # Alternate who goes first
        agents   = [nn_agent, opponent] if ep % 2 == 1 else [opponent, nn_agent]
        result   = UnoGame(agents).play()
        traj: list[Transition] = nn_agent.reset_trajectory()

        if not traj:
            continue

        won = result.winner == "NN"
        recent_wins.append(int(won))
        recent_lens.append(result.turns)

        # ── Discounted returns ────────────────────────────────────────
        T       = len(traj)
        reward  = 1.0 if won else -1.0
        returns = [reward * (gamma ** (T - t - 1)) for t in range(T)]

        states_t  = torch.tensor(
            [t.state for t in traj], dtype=torch.float32, device=device
        )                                            # (T, STATE_DIM)
        masks_t   = torch.tensor(
            [t.mask  for t in traj], dtype=torch.bool,    device=device
        )                                            # (T, ACTION_DIM)
        actions_t = torch.tensor(
            [t.action for t in traj], dtype=torch.long,   device=device
        )                                            # (T,)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=device)

        # Normalise returns for training stability
        if returns_t.std() > 1e-6:
            returns_t = (returns_t - returns_t.mean()) / (returns_t.std() + 1e-8)

        # ── Policy gradient update ────────────────────────────────────
        card_logits, _, __ = model(states_t)          # (T, ACTION_DIM)
        card_logits[~masks_t] = float("-inf")

        log_probs = F.log_softmax(card_logits, dim=1)  # (T, ACTION_DIM)
        probs     = log_probs.exp()                     # (T, ACTION_DIM)

        # -log π(a_t | s_t) * G_t  summed over the episode
        selected_lp = log_probs[range(T), actions_t]   # (T,)
        pg_loss     = -(selected_lp * returns_t).mean()

        # Entropy bonus: H(π) = -Σ p log p  (maximise → subtract from loss)
        # nansum handles the -inf → 0*-inf = nan entries from masked logits
        entropy = -(probs * log_probs).nansum(dim=1).mean()
        loss    = pg_loss - entropy_coef * entropy

        # PPO extension point:
        #   old_log_probs = [t.log_prob for t in traj]  # sampled at collection
        #   ratio         = (selected_lp - old_log_probs).exp()
        #   clipped_ratio = ratio.clamp(1-ε, 1+ε)
        #   pg_loss       = -torch.min(ratio * returns_t, clipped_ratio * returns_t).mean()

        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        recent_loss.append(float(loss.item()))

        # ── Logging ───────────────────────────────────────────────────
        if ep % log_every == 0:
            wr = mean(recent_wins) * 100
            al = mean(recent_lens)
            lv = mean(recent_loss)
            print(
                f"Episode {ep:>7,}/{n_episodes:,}  "
                f"win_rate={wr:5.1f}%  avg_len={al:5.1f}  "
                f"loss={lv:.4f}  entropy={float(entropy.item()):.4f}"
            )

        if ep % save_every == 0:
            os.makedirs(os.path.dirname(CKPT_PATH), exist_ok=True)
            model.save(CKPT_PATH)
            print(f"  ✓ checkpoint → {CKPT_PATH}")

    return model


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(model_path: str, n_games: int, device: str = "cpu") -> None:
    from simulate import run_matchup, print_matchup

    model    = UnoNet.load(model_path)
    nn_agent = NNAgent("NN-RL", model, device=device, greedy=True)

    for label, opp in [
        ("NN-RL  vs  Greedy", GreedyAgent("Greedy")),
        ("NN-RL  vs  Smart",  SmartAgent("Smart")),
    ]:
        stats = run_matchup(nn_agent, opp, n_games=n_games)
        print_matchup(label, stats)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="RL training for UNO")
    parser.add_argument("--episodes",   type=int,   default=50_000)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--gamma",      type=float, default=0.99)
    parser.add_argument("--entropy",    type=float, default=0.01)
    parser.add_argument("--opponent",   choices=["greedy", "smart"], default="smart")
    parser.add_argument("--eval",       action="store_true")
    parser.add_argument("--eval-games", type=int,   default=2_000)
    parser.add_argument("--device",     type=str,   default="cpu")
    parser.add_argument("--resume",     action="store_true",
                        help="Load existing checkpoint before training")
    args = parser.parse_args()

    opp_map = {
        "greedy": lambda: GreedyAgent("Greedy"),
        "smart":  lambda: SmartAgent("Smart"),
    }

    if args.eval:
        evaluate(CKPT_PATH, n_games=args.eval_games, device=args.device)
        return

    model = (
        UnoNet.load(CKPT_PATH)
        if args.resume and os.path.exists(CKPT_PATH)
        else UnoNet()
    )

    print(
        f"REINFORCE | opponent={args.opponent} | "
        f"episodes={args.episodes:,} | lr={args.lr} | γ={args.gamma}"
    )
    reinforce(
        model,
        opponent_factory=opp_map[args.opponent],
        n_episodes=args.episodes,
        lr=args.lr,
        gamma=args.gamma,
        entropy_coef=args.entropy,
        device=args.device,
    )


if __name__ == "__main__":
    main()
