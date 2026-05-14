---
title: P3 Specialization — First Experiment of the Slepian-Wolf Protocol
date: 2026-05-13
status: planning
companion_paper: ../../slepian-wolf-marl/paper/slepian-wolf-marl.4/paper.pdf
---

# P3 Specialization — First Experiment of the Slepian-Wolf Protocol

## Context

The Slepian-Wolf MARL paper (`../../slepian-wolf-marl/paper/slepian-wolf-marl.4/paper.tex`, v4 ready for preprint) reframes coordination without communication as distributed source coding and pre-registers five experimental predictions in Section 7. This notebook entry begins **execution** of that protocol on the existing Bucket Brigade environment.

The environment, scenarios, PPO training loop, and Nash/Double-Oracle solvers are already in place. We are missing: (a) information-theoretic estimators, (b) the paper's proposed conditional-MI regularizer, (c) a centralized teacher policy, (d) capacity-proxy infrastructure. We tackle (a) + (b) first; (c) is needed for P1/P2/P4 and is queued.

## Why P3 First

Of the five predictions:

| Prediction | Needs centralized oracle? | Tests the paper's proposed method? | Cost |
|---|---|---|---|
| P1 Capacity | yes | partially | high (capacity proxies + oracle + sweep) |
| P2 Sample complexity | yes | no | high (oracle + multi-radius sweep) |
| **P3 Specialization** | **no** | **yes** | **low** |
| P4 Communication | yes | no | medium |
| P5 Symmetry | no | no | low |

**P3 wins on three axes**: it tests the conditional-MI regularizer (the paper's *proposed method*), it doesn't need the centralized teacher (the expensive infrastructure item), and it produces a publishable result *or* a clean falsification with ~1 week of work.

## P3 Specification (from paper §5.3 + §7)

**Claim.** Training with a conditional-MI redundancy penalty `λ_red · I(Ẑ_i; Ẑ_j | R)` should reduce inter-agent representation redundancy monotonically in `λ_red`, while not degrading reward in tasks where redundancy is wasteful. Agents should also become more robust to single-agent dropout as `λ_red` increases.

**Manipulation.** Train at `λ_red ∈ {0, 1e-3, 1e-2, 1e-1}`.

**Outcomes.** Track over training:
- `I(Ẑ_i; Ẑ_j | R)` between encoder outputs of all agent pairs.
- Role entropy `H(A_i^*)` proxy via marginal action distributions.
- Agent-dropout robustness: replace one agent with a no-op at eval; measure team-reward drop.
- Team reward.

**Falsifier.** Either (a) no monotone decrease in conditional redundancy as `λ_red` increases, or (b) reward strictly worse at every `λ_red > 0` (penalty just hurts, never helps).

**Statistical protocol.** ≥20 random seeds per cell; mean ± 95% bootstrap CI; permutation tests for directional claims.

**Scenarios.** Three picked to span the conditional-entropy spectrum:
1. `trivial_cooperation` — low `H(A_i^* | A_{-i}^*)`, agents' optimal actions strongly co-determined. Specialization penalty should help *least*.
2. `default` — medium conditional entropy.
3. `chain_reaction` — high conditional entropy, requires distributed sub-task allocation. Specialization penalty should help *most*.

## Implementation Plan

### Step 1 — Information-Theoretic Estimators (~½ day)

New module: `bucket_brigade/analysis/info_theory.py`

API sketch:
```python
def entropy_discrete(samples: ndarray, bias_correction: str = "miller-madow") -> float
def conditional_entropy(x: ndarray, y: ndarray) -> float          # H(X|Y) = H(X,Y) − H(Y)
def mutual_information(x: ndarray, y: ndarray) -> float           # I(X;Y) = H(X) + H(Y) − H(X,Y)
def conditional_mi(x: ndarray, y: ndarray, z: ndarray) -> float   # I(X;Y|Z) via stratification on Z
def bootstrap_ci(estimator_fn, samples: ndarray, n_boot: int = 1000) -> tuple[float, float, float]
```

Tests in `tests/test_info_theory.py`:
- Uniform-on-k entropy ≈ log₂(k) within bias correction.
- Independent variables: I(X;Y) ≈ 0.
- Perfect correlation: I(X;Y) ≈ H(X).
- Synthetic conditioning: I(X;Y|Z) ≈ 0 when Z d-separates X and Y.

### Step 2 — Encoder Tap in `networks.py` (~1 hour)

Expose `PolicyNetwork.encoder_output(obs)` returning the trunk activations pre-action-heads. This is what we'll feed into `conditional_mi`. Discretize via small-codebook quantization (VQ-style) or just use the raw activations and use a kernel/InfoNCE estimator — start with quantization for simplicity and a clean plug-in estimator.

### Step 3 — Joint Multi-Agent Trainer (~1 day)

**Architecture decision (post-survey):** The existing PPO infrastructure is per-agent population training (each agent in a separate `PolicyLearner` process). Computing `I(Ẑ_i; Ẑ_j | S)` across agents requires cross-process batch synchronization, which is doable but adds complexity. Instead, we write a new trainer alongside the existing one:

`bucket_brigade/training/joint_trainer.py` — one PPO process, four separate `PolicyNetwork` instances trained on shared rollout batches. MI penalty becomes trivial: same minibatch, run all 4 encoders, compute pairwise `Î(Ẑ_i; Ẑ_j | S)` on the encoder outputs.

Architecture is independent learners (IPPO) but synchronously trained — equivalent to running 4 PolicyLearners in lockstep with shared rollout buffers. No info shared between agents at *inference* time; the MI penalty only operates at *training* time on the joint batch.

This same infrastructure later supports a 5th *centralized* policy (sees joint state, outputs joint action) as the teacher for P1/P2/P4.

```
L_red = λ_red · Σ_{i<j} Î(Ẑ_i; Ẑ_j | S)   # batch-level estimate
L_total = Σ_i L_ppo[i] + L_red
```

Conditioning on `S` avoids PMIC's failure mode. For discrete-quantized `Ẑ_i` with the small Bucket Brigade state space, plug-in estimation is tractable per minibatch.

### Step 4 — P3 Experiment Harness (~½ day)

New: `experiments/p3_specialization/`

- `run_sweep.py` — λ_red × scenario × seed grid runner; logs to `experiments/p3_specialization/runs/{scenario}/{lambda}/seed{N}/`.
- `dropout_eval.py` — load a trained policy, replace one agent with no-op, measure reward drop. Run on all checkpoints in the sweep.
- `analyze.py` — bootstrap CIs across seeds; plots; pre-register the falsifiers before looking at plots.

Pre-registered seeds: `range(42, 62)` (20 seeds).

### Step 5 — Smoke Test, Then Full Sweep

- Smoke test: 1 scenario × `λ_red = 0.01` × 3 seeds, ~30 episodes. Verify the IT measurements look sane and the loss term doesn't blow up gradients.
- Full sweep: 4 × `λ_red` × 3 scenarios × 20 seeds = 240 PPO runs. Mac Studio overnight.

### Step 6 — Write-Up

If results support: append a `2026-05-NN_p3_specialization_results.md` to this notebook; update the paper protocol's P3 cell from TBD to measured values; rev the paper to v5.

If results falsify: append a clean negative result. The paper's pre-registered falsifier was *exactly* this; honest reporting strengthens the credibility of the framework.

## Open Decisions

- **Quantization for `Ẑ_i`.** Simplest is fixed-codebook (e.g., 16-bin VQ) at the encoder output. Alternative: continuous-`Ẑ` + InfoNCE estimator. Start quantized; revisit if results are estimator-bound.
- **Conditioning variable.** Paper main text mentions `I(Ẑ_i; Ẑ_j | R)` (reward-conditioned). For implementation, `| S` (state-conditioned) is more natural for online MI estimation since `S` is observed at the timestep. Defensible because reward is a function of (S, A); state-conditioning is the more conservative cut. Doc both in the paper update.

## Parallel Track: Centralized PPO Teacher (for P1/P2/P4 later)

Independent of P3. The plan:

- New: `experiments/centralized_teacher/`
- Train a single PPO policy on the *full* joint state with the *joint* action. This is a centralized-training centralized-execution oracle.
- Use it as the source of `A_i^* ~ Z_i(·|S)` samples for entropy estimation.
- Acknowledge limitations in the paper: it's *trained-optimal*, not *provably optimal*; serves as the best computable teacher in the absence of tractable joint VI.

Defer until P3 is running. ~2-3 days of work when we get to it.

## Convergence Criteria for P3

P3 is "done" when:

1. The sweep has completed (240 runs).
2. Bootstrap CIs are computed for redundancy / reward / dropout-robustness curves.
3. The pre-registered falsifier is evaluated, with the answer (supported / partial / falsified) recorded *before* any post-hoc analysis.
4. A short results notebook entry is written.
5. The paper's Section 7 P3 cell is updated from TBD to measured values.

## Timeline

- **Day 1**: Steps 1+2 (info_theory module + encoder tap), with unit tests.
- **Day 2**: Step 3 (loss term) + smoke test.
- **Day 3**: Step 4 (experiment harness) + dropout eval.
- **Day 4**: Step 5 full sweep, runs overnight.
- **Day 5**: Step 6 analysis + write-up.

Total: one workweek if uninterrupted.
