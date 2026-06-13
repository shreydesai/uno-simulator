# UNO Simulator & Strategy Research

A fully digital UNO engine used to study what makes a winning strategy — starting from hand-coded heuristics, moving through supervised imitation learning, and progressively toward self-play reinforcement learning.

---

## Scoreboard

Win rates over **10,000 games** each (alternating first player). Rows = the agent under evaluation.

| Agent | vs Random | vs Greedy | vs Smart | vs NN-SL | Notes |
|-------|:---------:|:---------:|:--------:|:--------:|-------|
| Random | — | 44.9% | 38.7% | 39.4% | Baseline |
| Greedy | 55.1% | — | 45.1% | 44.8% | Action-card priority |
| Smart | 61.3% | 54.9% | — | 49.7% | Multi-heuristic |
| **NN-SL** | **60.6%** | **55.2%** | **50.3%** | — | Imitation from Smart |
| NN-PPO (fixed opp.) | 56.5% | 51.4% | 46.1% | 45.8% | PPO vs mixed pool — *regressed* |
| **NN-SP** | TBD | TBD | TBD | TBD | PPO self-play *(in progress)* |

> NN-SL baseline to beat: **50.3% vs Smart**

---

## Agents

| Agent | File | Strategy |
|-------|------|----------|
| `RandomAgent` | `uno/strategies/random_agent.py` | Uniform random valid card |
| `GreedyAgent` | `uno/strategies/greedy_agent.py` | Plays highest-priority action card; picks dominant color for wilds |
| `SmartAgent` | `uno/strategies/smart_agent.py` | Danger response, finishing setup, color steering, wild conservation |
| `NNAgent` | `uno/strategies/nn_agent.py` | Shared-trunk MLP with card + color + value heads; supports greedy or sampled inference |

---

## Quick Start

```bash
# No dependencies for the simulator itself
python3 simulate.py                      # run all heuristic matchups

# Install ML stack (only needed for training)
pip install torch numpy

# Supervised: collect data → train → eval
python3 -m training.supervised --collect --games 10000
python3 -m training.supervised --train   --epochs 30 --device mps
python3 -m training.supervised --eval    --games 5000

# PPO self-play (recommended)
python3 -m training.ppo --opponent self --init supervised --iters 300 --device mps

# PPO vs fixed pool (for comparison)
python3 -m training.ppo --opponent pool --init supervised --iters 300 --device mps

# Evaluate a saved checkpoint
python3 -m training.ppo --eval --eval-games 5000
```

---

## Architecture

```
State vector (169 dims)
  hand_counts[54]      count of each card type in hand
  top_card[54]         one-hot of discard-pile top card
  current_color[4]     one-hot of active color
  opp_hand_total[1]    opponent card count / 108
  deck_size[1]         remaining deck size / 108
  can_play_mask[54]    binary: which card types are currently legal
  draw_valid[1]        always 1.0

UnoNet
  Trunk   : 3 × [Linear(→256) → LayerNorm → ReLU → Dropout(0.1)]
  card_head  → logits over 55 actions (54 card types + draw)
  color_head → logits over 4 colors (for wild card play)
  value_head → V(s) ∈ (-1, 1) via tanh  [used by PPO only]

PPO training
  GAE-λ advantage estimation (γ=0.99, λ=0.95)
  Clipped objective ε=0.2, clipped value loss
  Per-minibatch advantage normalization
  Entropy bonus (coef=0.01)
  Linear LR annealing
  KL early stopping (target_kl=0.05)
  Parallel rollout collection via ProcessPoolExecutor (spawn)
```

---

## Experiment Log

### Experiment 1 — Heuristic Baselines
**What:** Three hand-coded agents from weakest to strongest: Random → Greedy → Smart.

**Key findings:**
- Smart beats Greedy 54.9% in 39 avg turns. This is our *minimum bar* — any learned agent needs to clear this to be interesting.
- Games have high variance (std ≈ 25 turns). UNO is a high-luck game, which means any strategy advantage is diluted. Expect win rates well below 70% even with a much better agent.

---

### Experiment 2 — Supervised / Imitation Learning
**What:** Collected ~79k transitions from SmartAgent winning games. Trained UnoNet with masked cross-entropy loss for 30 epochs on MPS.

**Result:** `NN-SL vs Smart = 50.3%` — essentially a coin flip against its own teacher.

**Learnings:**
- Imitation learning hits the teacher's ceiling almost perfectly. The model learned Smart's card priorities and color choices; it cannot exceed them because it never sees situations where Smart was *wrong*.
- The ceiling was reached quickly (96.6% val accuracy after 30 epochs), and training for longer doesn't help — the bottleneck is the training signal, not model capacity.
- **This is the baseline to beat for all future RL experiments.**

---

### Experiment 3 — PPO vs Fixed Opponent Pool
**What:** Warm-started from the SL checkpoint. Ran 300 PPO iterations training against [Random, Greedy, Smart] uniformly.

**Result:** `NN-PPO vs Smart = 46.1%` — *regressed* below the SL baseline. NN-SL beats NN-PPO 54.2% head-to-head.

**Learnings / Failure analysis:**
- `target_kl=0.01` was far too conservative. KL early-stopping triggered every single iteration after the first minibatch, meaning `pg_loss ≈ 0` throughout — the policy barely moved.
- Training against fixed opponents creates a ceiling: once you win 55% against the pool, the gradient signal goes flat. The opponent never adapts.
- The mixed pool (Random + Greedy + Smart) may have encouraged the agent to become a "jack of all trades" — slightly better than Random, not quite as good as Smart — without ever mastering any one opponent.

---

### Experiment 4 — PPO Self-Play *(in progress)*
**What:** Pure self-play — the current model plays against itself every iteration. Both players' transitions are collected (doubles data per game). Periodic evaluation every 25 iterations measures win rate vs Smart as the ground-truth signal.

**Setup:**
- Warm-start from SL checkpoint
- `target_kl=0.05` (raised from 0.01 to allow meaningful gradient updates)
- 256 episodes/iter × 4 workers on MPS
- Results logged every 10 iterations; eval vs Smart+Greedy every 25

**Key assumption being tested:** Does the model improve its win rate vs Smart when trained only against itself? If yes, self-play provides a real gradient signal and serves as a foundation for a stronger agent.

**Early results (4 iterations):**
- vs_smart jumped from 50.3% (SL baseline) → 52–53% in just 4 iterations ✓
- `len` ≈ 30 turns (vs 39 for heuristic games) — self-play converges faster
- KL values 0.02–0.04 with target_kl=0.05 allow multiple epochs per iteration

*Full results will be appended here.*
