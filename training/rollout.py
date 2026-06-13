"""
Parallel rollout collection for PPO.

Design
------
Game collection is purely CPU-bound (sequential decisions + NN inference at
batch size 1).  We use ProcessPoolExecutor with spawn-based workers so each
process gets a clean PyTorch state and MPS / CUDA locks don't carry over.
The gradient-update step in ppo.py is the only place we use an accelerator.

Public API
----------
    batch = collect_parallel(
        model, opponent_names=["random","greedy","smart"],
        n_episodes=256, n_workers=4,
        gamma=0.99, gae_lambda=0.95,
    )
    # batch is a RolloutBatch (numpy arrays ready for PPO)

Each worker independently:
  1. Reconstructs UnoNet from serialised weights (no shared-memory issues)
  2. Creates a PPOCollectorAgent wrapping the model
  3. Runs n_episodes/n_workers games, alternating first-player
  4. Computes GAE advantages within each episode
  5. Returns a flat numpy dict that is concatenated in the main process
"""

from __future__ import annotations

import multiprocessing as mp
import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Optional

import numpy as np

from uno.encoding import STATE_DIM, ACTION_DIM


# ------------------------------------------------------------------ #
#  RolloutBatch                                                        #
# ------------------------------------------------------------------ #

@dataclass
class RolloutBatch:
    """
    Flat batch of transitions collected from one PPO iteration.

    All arrays are aligned on axis-0 (one row per agent decision).

    states          (N, STATE_DIM)  float32
    actions         (N,)            int32
    old_log_probs   (N,)            float32  — log π_old(a|s) at collection time
    old_values      (N,)            float32  — V_old(s) at collection time
    masks           (N, ACTION_DIM) bool
    advantages      (N,)            float32  — GAE advantages
    returns         (N,)            float32  — advantages + old_values  (V-targets)
    color_actions   (N,)            int32    — -1 if no color was chosen
    color_log_probs (N,)            float32  — 0.0 if no color was chosen
    """
    states:          np.ndarray
    actions:         np.ndarray
    old_log_probs:   np.ndarray
    old_values:      np.ndarray
    masks:           np.ndarray
    advantages:      np.ndarray
    returns:         np.ndarray
    color_actions:   np.ndarray
    color_log_probs: np.ndarray
    wins:            int
    episodes:        int

    def __len__(self) -> int:
        return len(self.states)


# ------------------------------------------------------------------ #
#  GAE                                                                 #
# ------------------------------------------------------------------ #

def compute_gae(
    rewards:    np.ndarray,   # (T,)
    values:     np.ndarray,   # (T,)
    dones:      np.ndarray,   # (T,)  1.0 at episode end
    gamma:      float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generalised Advantage Estimation (Schulman et al. 2016).

    δ_t   = r_t + γ · V(s_{t+1}) · (1 − done_t) − V(s_t)
    A_t   = δ_t + (γλ) · (1 − done_t) · A_{t+1}
    G_t   = A_t + V(s_t)          ← V-function targets (returns)

    Returns (advantages, returns), both shape (T,).
    """
    T          = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    gae        = 0.0
    next_val   = 0.0   # bootstrap value after terminal = 0

    for t in reversed(range(T)):
        mask     = 1.0 - dones[t]
        delta    = rewards[t] + gamma * next_val * mask - values[t]
        gae      = delta + gamma * gae_lambda * mask * gae
        advantages[t] = gae
        next_val = values[t]

    returns = advantages + values
    return advantages, returns


# ------------------------------------------------------------------ #
#  PPOCollectorAgent                                                   #
# ------------------------------------------------------------------ #

class PPOCollectorAgent:
    """
    Lightweight agent that wraps a UnoNet for rollout collection.

    Unlike NNAgent it always samples (never greedy), always runs on CPU,
    and records (state, action, log_prob, value, mask) for every decision.
    The value head output is stored so GAE can be computed after the game.
    """

    def __init__(self, model: "UnoNet") -> None:  # noqa: F821
        import torch
        self.model  = model
        self.name   = "NN"
        self._steps: list[dict] = []
        self._pending: Optional[dict] = None
        self._cached_color_logits = None

    # ---- Agent interface (subset) ------------------------------------

    def choose_card(self, state) -> Optional["Card"]:  # noqa: F821
        import torch
        import torch.nn.functional as F
        from uno.encoding import encode_state, build_action_mask, action_to_card

        state_vec = encode_state(state)
        mask      = build_action_mask(state)
        state_t   = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            card_logits, color_logits, value = self.model(state_t)

        self._cached_color_logits = color_logits.squeeze(0)

        clogits = card_logits.squeeze(0).clone()
        mask_t  = torch.tensor(mask, dtype=torch.bool)
        clogits[~mask_t] = float("-inf")

        probs = F.softmax(clogits, dim=0)
        # Guard: replace any NaN / inf that can arise from extreme logits
        # (e.g. MPS numerical edge-cases or early PPO updates).
        # Fall back to uniform over legal actions if the distribution collapses.
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() < 1e-9:
            probs = mask_t.float()
        probs = probs / probs.sum()

        action   = int(torch.multinomial(probs, 1).item())
        log_prob = float(torch.log(probs[action] + 1e-10).item())
        val      = float(value.item())

        step = {
            "state":        state_vec,
            "action":       action,
            "log_prob":     log_prob,
            "value":        val,
            "mask":         mask,
            "color_action": -1,
            "color_lp":     0.0,
        }
        self._pending = step
        self._steps.append(step)

        return action_to_card(action, state.playable_cards())

    def choose_color(self, state) -> "Color":  # noqa: F821
        import torch
        import torch.nn.functional as F
        from uno.encoding import idx_to_color

        logits = self._cached_color_logits
        probs  = F.softmax(logits, dim=0)
        # Same NaN guard as choose_card — protects against corrupted
        # model weights after aggressive early PPO updates.
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        if probs.sum() < 1e-9:
            probs = torch.ones(probs.shape) / probs.numel()
        else:
            probs = probs / probs.sum()
        color_idx = int(torch.multinomial(probs, 1).item())
        color_lp  = float(torch.log(probs[color_idx] + 1e-10).item())

        if self._pending is not None:
            self._pending["color_action"] = color_idx
            self._pending["color_lp"]     = color_lp
            self._pending = None

        self._cached_color_logits = None
        return idx_to_color(color_idx)

    # ---- Episode management -----------------------------------------

    def flush(
        self,
        won:            bool,
        gamma:          float,
        gae_lambda:     float,
        reward_shaping: float = 0.0,
    ) -> list[dict]:
        """
        Assign rewards, compute GAE, and return completed steps.
        Clears internal state for the next episode.
        """
        steps = self._steps
        T     = len(steps)

        if T == 0:
            return []

        # Assign rewards: 0 for all intermediate steps, ±1 at terminal.
        # Optional: subtract a small per-step time penalty to encourage
        # shorter games (reward shaping preserves optimal policy when γ<1).
        base = 1.0 if won else -1.0
        rewards = np.zeros(T, dtype=np.float32)
        rewards[-1] = base + reward_shaping * T   # e.g. -0.001 * T length penalty

        values = np.array([s["value"] for s in steps], dtype=np.float32)
        dones  = np.zeros(T, dtype=np.float32)
        dones[-1] = 1.0

        advantages, returns = compute_gae(rewards, values, dones, gamma, gae_lambda)

        for i, s in enumerate(steps):
            s["reward"]    = float(rewards[i])
            s["advantage"] = float(advantages[i])
            s["return"]    = float(returns[i])

        self._steps   = []
        self._pending = None
        return steps


# ------------------------------------------------------------------ #
#  Subprocess worker                                                   #
# ------------------------------------------------------------------ #

def _worker(args: tuple) -> dict:
    """
    Subprocess entry point.  Receives plain-Python / numpy data only
    (no live model objects) so pickling is safe across spawn boundaries.
    """
    import random, torch
    (
        state_dict, model_config,
        n_episodes, opponent_names, seed,
        gamma, gae_lambda, reward_shaping,
    ) = args

    random.seed(seed)
    torch.manual_seed(seed)

    # Reconstruct model on CPU
    from training.model import UnoNet
    model = UnoNet(**model_config)
    model.load_state_dict(state_dict)
    model.eval()

    from uno.game import UnoGame
    from uno.strategies.random_agent import RandomAgent
    from uno.strategies.greedy_agent import GreedyAgent
    from uno.strategies.smart_agent import SmartAgent

    _opp_map = {
        "random": lambda: RandomAgent("Random"),
        "greedy": lambda: GreedyAgent("Greedy"),
        "smart":  lambda: SmartAgent("Smart"),
    }

    all_states, all_actions, all_lps, all_vals = [], [], [], []
    all_masks, all_advs, all_rets             = [], [], []
    all_col_acts, all_col_lps                 = [], []
    wins = 0

    for ep in range(n_episodes):
        agent    = PPOCollectorAgent(model)
        opp_name = random.choice(opponent_names)
        opp      = _opp_map[opp_name]()

        agents = [agent, opp] if ep % 2 == 0 else [opp, agent]

        # Monkey-patch: UnoGame calls agent.name and agent.choose_card/color
        result = UnoGame(agents).play()
        won    = result.winner == "NN"
        wins  += int(won)

        steps = agent.flush(won, gamma, gae_lambda, reward_shaping)
        for s in steps:
            all_states.append(s["state"])
            all_actions.append(s["action"])
            all_lps.append(s["log_prob"])
            all_vals.append(s["value"])
            all_masks.append(s["mask"])
            all_advs.append(s["advantage"])
            all_rets.append(s["return"])
            all_col_acts.append(s["color_action"])
            all_col_lps.append(s["color_lp"])

    return {
        "states":          np.array(all_states,   dtype=np.float32),
        "actions":         np.array(all_actions,  dtype=np.int32),
        "old_log_probs":   np.array(all_lps,      dtype=np.float32),
        "old_values":      np.array(all_vals,      dtype=np.float32),
        "masks":           np.array(all_masks,    dtype=bool),
        "advantages":      np.array(all_advs,     dtype=np.float32),
        "returns":         np.array(all_rets,      dtype=np.float32),
        "color_actions":   np.array(all_col_acts, dtype=np.int32),
        "color_log_probs": np.array(all_col_lps,  dtype=np.float32),
        "wins":            wins,
        "episodes":        n_episodes,
    }


# ------------------------------------------------------------------ #
#  Public collection API                                               #
# ------------------------------------------------------------------ #

def collect_parallel(
    model,
    opponent_names: list[str],
    n_episodes:     int,
    n_workers:      int   = 4,
    gamma:          float = 0.99,
    gae_lambda:     float = 0.95,
    reward_shaping: float = 0.0,
    base_seed:      int   = 0,
) -> RolloutBatch:
    """
    Collect *n_episodes* across *n_workers* parallel processes.

    Parameters
    ----------
    model           : UnoNet (CPU or GPU — weights are copied to workers as CPU)
    opponent_names  : list of "random" | "greedy" | "smart" to sample from
    n_episodes      : total episodes to collect this iteration
    n_workers       : number of subprocess workers
    gamma           : discount factor for GAE
    gae_lambda      : GAE-λ smoothing parameter
    reward_shaping  : per-step coefficient added to terminal reward
                      (e.g. -0.001 discourages long games)
    base_seed       : each worker gets base_seed + worker_id as its RNG seed
    """
    import torch

    # Serialise model weights to CPU (safe to pickle for spawn)
    state_dict   = {k: v.cpu() for k, v in model.state_dict().items()}
    model_config = model._config

    # Distribute episodes across workers
    eps_per_worker = [n_episodes // n_workers] * n_workers
    for i in range(n_episodes % n_workers):
        eps_per_worker[i] += 1

    worker_args = [
        (
            state_dict, model_config,
            eps_per_worker[i],
            opponent_names,
            base_seed + i,
            gamma, gae_lambda, reward_shaping,
        )
        for i in range(n_workers)
    ]

    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as pool:
        results = list(pool.map(_worker, worker_args))

    # Concatenate worker results
    keys = ["states", "actions", "old_log_probs", "old_values", "masks",
            "advantages", "returns", "color_actions", "color_log_probs"]
    merged = {k: np.concatenate([r[k] for r in results], axis=0) for k in keys}

    total_wins     = sum(r["wins"]     for r in results)
    total_episodes = sum(r["episodes"] for r in results)

    return RolloutBatch(
        wins=total_wins,
        episodes=total_episodes,
        **merged,
    )
