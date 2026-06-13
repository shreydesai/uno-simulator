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
| NN-SP (pure self-play) | 60.1% | 54.8% | 49.7% | 49.0% | Maintained quality, didn't improve |
| NN-ISL (R2, best) | 61.2% | 54.6% | 50.9% | ~50.0% | Iterative SL on self-play data — within noise of baseline |

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

### Experiment 4 — PPO Pure Self-Play
**What:** Pure self-play — current model plays against itself each iteration. Both players' transitions collected (doubles data per game). `target_kl=0.05` to allow meaningful updates. Periodic eval vs Smart every 25 iters as ground truth.

**Result:** `NN-SP vs Smart = 49.7%` — statistically equivalent to SL baseline (50.3%). Maintained quality, didn't improve.

**Training curve (vs Smart, every 25 iters):**
```
Iter  25: 43.2%  Iter  50: 43.4%  Iter  75: 44.8%  Iter 100: 45.8%
Iter 125: 47.6%  Iter 150: 49.0%  Iter 175: 44.8%  Iter 200: 48.2%
Iter 225: 43.8%  Iter 250: 50.2%  Iter 275: 48.6%  Iter 300: 46.8%
```

**Failure analysis — why pure self-play doesn't work:**

The training diagnostics reveal two smoking guns:

1. **`pg_loss ≈ 0` throughout.** Policy gradient is essentially zero. When both agents share the same weights W_t, the winning trajectory's gradient and losing trajectory's gradient cancel each other — they're drawn from the same distribution. Only the random asymmetry between two identical players produces any gradient, which is tiny and noisy.

2. **Entropy increased from 0.11 → 0.24.** The policy became *more random*, not more strategic. In a symmetric zero-sum game, pure self-play's stable equilibrium is the Nash equilibrium — approximately uniform random play. The model was slowly drifting toward that equilibrium, which is why it stayed near 50% quality but eroded the structured knowledge from supervised training.

The value function (`ev` 0.22–0.27) was healthy and well-calibrated, so the problem isn't GAE — it's that there's no stable training *direction* when both players are the same.

**Insight:** Pure self-play (shared weights) doesn't work for competitive games. You need a *stable opponent target*. The fix is **frozen opponent / lagged self-play**: train against a snapshot of the model from K iterations ago. This is what AlphaGo used and what made self-play viable in practice.

---

### Experiment 5 — Iterative Supervised Learning on Self-Play Data *(in progress)*

**The pivot:** Rather than fighting RL instability, use self-play purely as a **data engine**. Two copies of the current model play each other; the winner's transitions become the next round's supervised training set. Retrain with cross-entropy. Repeat.

**Why this avoids the RL problems:**
- No sparse reward signal — CE loss gives a dense gradient on every step
- No gradient cancellation — we're not backpropagating through game outcomes, just treating winning moves as labels
- No value function needed — supervised training is inherently stable
- The model naturally gets harder training data each round as it improves

**Round structure:**
```
Round 0: Train on SmartAgent winning games         → NN-SL  (50.3% vs Smart)
Round 1: Train on NN-SL vs NN-SL winning games     → NN-ISL-1  (? vs Smart)
Round 2: Train on NN-ISL-1 vs NN-ISL-1 wins       → NN-ISL-2  (? vs Smart)
...
```

**Key hypothesis:** Self-play games between equally-matched strong agents surface more nuanced decision-making than Smart's heuristics — better color control when opponents are close to winning, more precise wild-card timing, tighter endgame play. These get distilled into the next model via CE.

**Data mixing:** Each round blends self-play winner data with a fraction of the original Smart data to prevent catastrophic forgetting of basic card rules.

**Results (10k games each):**

| Round | vs Smart | vs SL baseline |
|-------|:--------:|:--------------:|
| Baseline (NN-SL) | 49.9% | — |
| Round 1 | 49.9% | = |
| Round 2 | **50.9%** | +1.0pp |
| Round 3 | 50.1% | +0.2pp |
| Round 4 | 49.5% | -0.4pp |
| Round 5 | 49.8% | -0.1pp |

All differences are within statistical noise (1 stderr ≈ ±0.5pp). **Iterative SL does not meaningfully improve over the original SL baseline.**

**Why it doesn't work — root cause:**

The winner in NN vs NN games is not systematically making *better decisions* — they're getting *luckier card draws*. Since both players use the same underlying model, they make nearly identical choices in similar states. The winner diverges from the loser mostly when one player draws a good run of cards, not because they found a superior strategy.

Training on the winner's transitions is therefore not "imitate the better player" — it's "imitate a random 50% subset of this model's own outputs." Iterating this process can't escape the original strategy: the model converges back to NN-SL each round.

The evidence: val accuracy *increases* across rounds (96.8% → 97.7%) but game win rate stays flat. The model is learning to imitate the winner's moves more precisely — it's just that those moves aren't systematically better.

This is the fundamental ceiling of supervised / imitation learning: **you can't exceed the quality of your training signal.** To break through the ~50% vs Smart ceiling, the training signal itself needs to be stronger than "who got luckier in this game."

**What would actually work:**

| Approach | Why it would help |
|----------|-------------------|
| MCTS-guided data | Search finds *genuinely better* moves at each state, not just the winner's lucky choices |
| Dense reward shaping | Gives RL real signal at every step rather than just terminal ±1 |
| Lagged self-play (RL) | Stable opponent creates a clear gradient direction; the current model can't cancel it |
| State augmentation | Add card-counting from discard pile → the model sees information it currently ignores |
