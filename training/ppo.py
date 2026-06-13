"""
PPO training for UNO.

Implements the core "37 details" (Huang et al. 2022) that matter most for
a discrete-action environment with sparse rewards:

  ✓ GAE-λ advantage estimation           (rollout.py)
  ✓ Per-minibatch advantage normalisation
  ✓ Clipped policy objective  (ε = 0.2)
  ✓ Clipped value-function loss
  ✓ Entropy bonus
  ✓ Gradient clipping  (max-norm 0.5)
  ✓ Linear LR annealing
  ✓ Approximate KL early stopping  (target_kl = 0.01)
  ✓ Orthogonal weight initialisation    (model.py)
  ✓ Opponent pool curriculum (Random → Greedy → Smart → mix)

Scalability
-----------
  Rollout collection is parallelised across N CPU workers (see rollout.py).
  The gradient update runs on the configured device (MPS / CUDA / CPU).
  Collection and training alternate, so the GPU is never idle during rollouts.

Usage
-----
    # Warm-start from supervised checkpoint, train vs Smart
    python -m training.ppo --init supervised --opponent smart --iters 300

    # Train from scratch vs opponent pool
    python -m training.ppo --opponent pool --iters 500

    # Evaluate
    python -m training.ppo --eval
"""

from __future__ import annotations

import argparse
import os
import time
from collections import deque
from statistics import mean

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from training.model import UnoNet
from training.rollout import RolloutBatch, collect_parallel

CKPT_PPO  = "checkpoints/ppo.pt"
CKPT_SL   = "checkpoints/supervised.pt"


# ------------------------------------------------------------------ #
#  Diagnostics                                                         #
# ------------------------------------------------------------------ #

def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Fraction of variance in y_true explained by y_pred (−∞ to 1.0)."""
    var_y = np.var(y_true)
    return float(1.0 - np.var(y_true - y_pred) / (var_y + 1e-8))


# ------------------------------------------------------------------ #
#  Single PPO update step                                              #
# ------------------------------------------------------------------ #

def ppo_update(
    model:          UnoNet,
    opt:            torch.optim.Optimizer,
    batch:          RolloutBatch,
    ppo_epochs:     int   = 4,
    mini_batch_size:int   = 256,
    clip_eps:       float = 0.2,
    vf_coef:        float = 0.5,
    ent_coef:       float = 0.01,
    max_grad_norm:  float = 0.5,
    target_kl:      float = 0.01,
    device:         str   = "cpu",
) -> dict:
    """
    Run K epochs of PPO minibatch updates over *batch*.

    Returns a dict of training diagnostics for logging.
    """
    N = len(batch)

    # Move everything to device once
    states_t    = torch.tensor(batch.states,        dtype=torch.float32, device=device)
    actions_t   = torch.tensor(batch.actions,       dtype=torch.long,    device=device)
    old_lps_t   = torch.tensor(batch.old_log_probs, dtype=torch.float32, device=device)
    old_vals_t  = torch.tensor(batch.old_values,    dtype=torch.float32, device=device)
    masks_t     = torch.tensor(batch.masks,         dtype=torch.bool,    device=device)
    advs_t      = torch.tensor(batch.advantages,    dtype=torch.float32, device=device)
    returns_t   = torch.tensor(batch.returns,       dtype=torch.float32, device=device)
    col_acts_t  = torch.tensor(batch.color_actions, dtype=torch.long,    device=device)
    col_lps_t   = torch.tensor(batch.color_log_probs, dtype=torch.float32, device=device)

    # Diagnostics accumulators
    pg_losses, vf_losses, entropies, kl_divs, clip_fracs = [], [], [], [], []
    n_early_stop = 0

    for epoch in range(ppo_epochs):
        perm = torch.randperm(N, device=device)

        for start in range(0, N, mini_batch_size):
            idx = perm[start : start + mini_batch_size]
            if len(idx) < 2:          # skip tiny trailing shards
                continue

            mb_states  = states_t[idx]
            mb_actions = actions_t[idx]
            mb_old_lps = old_lps_t[idx]
            mb_old_vs  = old_vals_t[idx]
            mb_masks   = masks_t[idx]
            mb_advs    = advs_t[idx]
            mb_returns = returns_t[idx]
            mb_col_a   = col_acts_t[idx]

            # ── Forward pass ─────────────────────────────────────────
            card_logits, color_logits, values = model(mb_states)

            # Additive mask: use a large negative constant rather than
            # -inf so that backward never touches ±inf arithmetic.
            # -1e9 makes illegal-action probabilities effectively 0
            # while keeping gradients finite everywhere.
            card_logits_m = card_logits + (~mb_masks).float() * (-1e9)

            # torch.distributions.Categorical handles the masked softmax
            # correctly and provides a numerically-stable entropy.
            dist    = torch.distributions.Categorical(logits=card_logits_m)
            new_lps = dist.log_prob(mb_actions)   # (mb,)
            entropy = dist.entropy().mean()        # scalar

            log_ratio = new_lps - mb_old_lps
            ratio     = log_ratio.exp()

            # ── Per-minibatch advantage normalisation ────────────────
            # (Huang et al. "37 details" §3.3 — normalise within the
            #  minibatch, not the full batch, for stability)
            mb_advs_norm = (mb_advs - mb_advs.mean()) / (mb_advs.std() + 1e-8)

            # ── Clipped policy loss ───────────────────────────────────
            pg_loss1 = -mb_advs_norm * ratio
            pg_loss2 = -mb_advs_norm * ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps)
            pg_loss  = torch.max(pg_loss1, pg_loss2).mean()

            # ── Clipped value loss ────────────────────────────────────
            vf_loss_unclip = (values - mb_returns) ** 2
            values_clipped = mb_old_vs + (values - mb_old_vs).clamp(-clip_eps, clip_eps)
            vf_loss_clip   = (values_clipped - mb_returns) ** 2
            vf_loss        = 0.5 * torch.max(vf_loss_unclip, vf_loss_clip).mean()

            # ── Color head loss (only on wild-card turns) ─────────────
            wild_mask  = mb_col_a >= 0
            color_loss = (
                F.cross_entropy(color_logits[wild_mask], mb_col_a[wild_mask])
                if wild_mask.any()
                else torch.tensor(0.0, device=device)
            )

            # ── Total loss ────────────────────────────────────────────
            loss = pg_loss + vf_coef * vf_loss - ent_coef * entropy + 0.1 * color_loss

            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()

            # ── Diagnostics ───────────────────────────────────────────
            with torch.no_grad():
                approx_kl  = (mb_old_lps - new_lps).mean().item()
                clip_frac  = ((ratio - 1.0).abs() > clip_eps).float().mean().item()

            pg_losses.append(pg_loss.item())
            vf_losses.append(vf_loss.item())
            entropies.append(entropy.item())
            kl_divs.append(approx_kl)
            clip_fracs.append(clip_frac)

            # ── KL early stopping ─────────────────────────────────────
            if approx_kl > target_kl:
                n_early_stop += 1
                break   # stop this epoch

        if n_early_stop > 0:
            break       # stop all remaining epochs

    # Explained variance of the value function (on CPU for numpy)
    with torch.no_grad():
        _, _, pred_vals = model(states_t)
    ev = explained_variance(
        pred_vals.cpu().numpy(), returns_t.cpu().numpy()
    )

    return {
        "pg_loss":    float(np.mean(pg_losses)),
        "vf_loss":    float(np.mean(vf_losses)),
        "entropy":    float(np.mean(entropies)),
        "approx_kl":  float(np.mean(kl_divs)),
        "clip_frac":  float(np.mean(clip_fracs)),
        "expl_var":   ev,
        "early_stop": n_early_stop > 0,
    }


# ------------------------------------------------------------------ #
#  Main training loop                                                  #
# ------------------------------------------------------------------ #

def quick_eval(model: UnoNet, n_games: int = 500, device: str = "cpu") -> dict[str, float]:
    """
    Evaluate win rates vs Greedy and Smart (greedy inference, no training).
    Used as the periodic ground-truth signal during RL training.
    """
    import random
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


def train_ppo(
    model:           UnoNet,
    opponent_names:  list[str]  = ("random", "greedy", "smart"),
    n_iterations:    int   = 300,
    episodes_per_iter: int  = 256,
    n_workers:       int   = 4,
    ppo_epochs:      int   = 4,
    mini_batch_size: int   = 256,
    clip_eps:        float = 0.2,
    vf_coef:         float = 0.5,
    ent_coef:        float = 0.01,
    max_grad_norm:   float = 0.5,
    target_kl:       float = 0.01,
    lr:              float = 3e-4,
    lr_decay:        bool  = True,
    gamma:           float = 0.99,
    gae_lambda:      float = 0.95,
    reward_shaping:  float = 0.0,
    log_every:       int   = 10,
    save_every:      int   = 50,
    eval_every:      int   = 25,   # 0 to disable; runs quick_eval vs Greedy+Smart
    device:          str   = "cpu",
    ckpt_path:       str   = CKPT_PPO,
) -> UnoNet:
    model = model.to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, eps=1e-5)

    win_history:  deque[float] = deque(maxlen=log_every)
    len_history:  deque[float] = deque(maxlen=log_every)

    print(
        f"\nPPO | iters={n_iterations} | eps/iter={episodes_per_iter} "
        f"| workers={n_workers} | ppo_epochs={ppo_epochs} "
        f"| opponent={list(opponent_names)} | device={device}"
    )
    print(f"{'─'*72}")

    for it in range(1, n_iterations + 1):
        t0 = time.time()

        # ── 1. Linear LR annealing ─────────────────────────────────────
        if lr_decay:
            frac = 1.0 - (it - 1) / n_iterations
            for pg in opt.param_groups:
                pg["lr"] = lr * frac

        # ── 2. Collect rollouts (parallel, CPU) ────────────────────────
        model.eval()
        batch = collect_parallel(
            model,
            opponent_names=list(opponent_names),
            n_episodes=episodes_per_iter,
            n_workers=n_workers,
            gamma=gamma,
            gae_lambda=gae_lambda,
            reward_shaping=reward_shaping,
            base_seed=it * 1000,
        )

        win_history.append(batch.wins / batch.episodes)
        # approximate avg game length from batch size
        len_history.append(len(batch) / batch.episodes)

        # ── 3. PPO update (GPU / MPS) ──────────────────────────────────
        model.train()
        diag = ppo_update(
            model, opt, batch,
            ppo_epochs=ppo_epochs,
            mini_batch_size=mini_batch_size,
            clip_eps=clip_eps,
            vf_coef=vf_coef,
            ent_coef=ent_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            device=device,
        )

        elapsed = time.time() - t0

        # ── 4. Logging ─────────────────────────────────────────────────
        if it % log_every == 0 or it == 1:
            wr  = mean(win_history) * 100
            agl = mean(len_history)
            cur_lr = opt.param_groups[0]["lr"]
            es  = "✓KL" if diag["early_stop"] else "   "
            print(
                f"Iter {it:4d}/{n_iterations}  "
                f"win={wr:5.1f}%  len={agl:5.1f}  "
                f"pg={diag['pg_loss']:+.4f}  vf={diag['vf_loss']:.4f}  "
                f"ent={diag['entropy']:.4f}  kl={diag['approx_kl']:.4f}  "
                f"clip={diag['clip_frac']:.3f}  ev={diag['expl_var']:+.3f}  "
                f"lr={cur_lr:.2e}  {es}  [{elapsed:.1f}s]"
            )

        # ── 5. Periodic evaluation (ground-truth signal) ───────────────
        if eval_every > 0 and it % eval_every == 0:
            model.eval()
            rates = quick_eval(model, n_games=500, device=device)
            model.train()
            g, s = rates["vs_greedy"] * 100, rates["vs_smart"] * 100
            print(
                f"  ├─ eval  vs_greedy={g:.1f}%  vs_smart={s:.1f}%  "
                f"{'▲ above SL baseline' if s > 50.3 else '▼ below SL baseline'}"
            )

        # ── 6. Checkpoint ──────────────────────────────────────────────
        if it % save_every == 0:
            os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
            model.save(ckpt_path)
            print(f"  ✓ checkpoint → {ckpt_path}")

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    model.save(ckpt_path)
    print(f"\nTraining complete. Final checkpoint → {ckpt_path}")
    return model


# ------------------------------------------------------------------ #
#  Evaluation                                                          #
# ------------------------------------------------------------------ #

def evaluate(model_path: str, n_games: int, device: str = "cpu") -> None:
    from simulate import run_matchup, print_matchup
    from uno.strategies.random_agent import RandomAgent
    from uno.strategies.greedy_agent import GreedyAgent
    from uno.strategies.smart_agent import SmartAgent
    from uno.strategies.nn_agent import NNAgent

    model    = UnoNet.load(model_path)
    nn_agent = NNAgent("NN-PPO", model, device=device, greedy=True)

    print(f"\n  Evaluating {model_path} ({n_games:,} games each)")
    for label, opp in [
        ("NN-PPO  vs  Random", RandomAgent("Random")),
        ("NN-PPO  vs  Greedy", GreedyAgent("Greedy")),
        ("NN-PPO  vs  Smart",  SmartAgent("Smart")),
    ]:
        stats = run_matchup(nn_agent, opp, n_games=n_games)
        print_matchup(label, stats)


# ------------------------------------------------------------------ #
#  CLI                                                                 #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="PPO training for UNO")
    parser.add_argument("--iters",       type=int,   default=300,
                        help="Number of PPO iterations")
    parser.add_argument("--eps-per-iter",type=int,   default=256,
                        help="Episodes collected per iteration")
    parser.add_argument("--workers",     type=int,   default=4,
                        help="Parallel rollout workers")
    parser.add_argument("--ppo-epochs",  type=int,   default=4)
    parser.add_argument("--mini-batch",  type=int,   default=256)
    parser.add_argument("--clip-eps",    type=float, default=0.2)
    parser.add_argument("--vf-coef",     type=float, default=0.5)
    parser.add_argument("--ent-coef",    type=float, default=0.01)
    parser.add_argument("--lr",          type=float, default=3e-4)
    parser.add_argument("--gamma",       type=float, default=0.99)
    parser.add_argument("--gae-lambda",  type=float, default=0.95)
    parser.add_argument("--reward-shaping", type=float, default=0.0,
                        help="Per-step reward penalty (e.g. -0.001 discourages long games)")
    parser.add_argument("--target-kl",   type=float, default=0.01)
    parser.add_argument("--eval-every",  type=int,   default=25,
                        help="Evaluate vs Greedy+Smart every N iters (0=off)")
    parser.add_argument("--opponent",
                        choices=["random", "greedy", "smart", "pool", "self"],
                        default="self",
                        help="'self'=pure self-play  'pool'=mixed heuristics")
    parser.add_argument("--init",
                        choices=["random", "supervised"],
                        default="supervised",
                        help="'supervised' warm-starts from the SL checkpoint")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from existing PPO checkpoint")
    parser.add_argument("--device",      type=str, default="cpu")
    parser.add_argument("--eval",        action="store_true")
    parser.add_argument("--eval-games",  type=int, default=5_000)
    args = parser.parse_args()

    if args.eval:
        evaluate(CKPT_PPO, n_games=args.eval_games, device=args.device)
        return

    # ── Opponent selection ─────────────────────────────────────────────
    opp_map = {
        "random": ["random"],
        "greedy": ["greedy"],
        "smart":  ["smart"],
        "pool":   ["random", "greedy", "smart"],
        "self":   ["self"],
    }
    opponent_names = opp_map[args.opponent]

    # ── Model initialisation ───────────────────────────────────────────
    if args.resume and os.path.exists(CKPT_PPO):
        print(f"Resuming from {CKPT_PPO}")
        model = UnoNet.load(CKPT_PPO)
    elif args.init == "supervised" and os.path.exists(CKPT_SL):
        print(f"Warm-starting from supervised checkpoint: {CKPT_SL}")
        model = UnoNet.load(CKPT_SL)   # value_head will be randomly initialised
    else:
        print("Starting from random initialisation")
        model = UnoNet()

    train_ppo(
        model,
        opponent_names=opponent_names,
        n_iterations=args.iters,
        episodes_per_iter=args.eps_per_iter,
        n_workers=args.workers,
        ppo_epochs=args.ppo_epochs,
        mini_batch_size=args.mini_batch,
        clip_eps=args.clip_eps,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        lr=args.lr,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        reward_shaping=args.reward_shaping,
        target_kl=args.target_kl,
        eval_every=args.eval_every,
        device=args.device,
    )


if __name__ == "__main__":
    main()
