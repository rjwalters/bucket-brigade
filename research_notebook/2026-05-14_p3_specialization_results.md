---
title: P3 Specialization — Results (Fresh Sweep, Fixed Env)
date: 2026-05-14
status: falsified-as-preregistered, with one scenario-specific signal
companion_paper: ../../slepian-wolf-marl/paper/slepian-wolf-marl.4/paper.pdf
companion_plan: 2026-05-13_p3_specialization_plan.md
---

# P3 Specialization — Results (Fresh Sweep, Fixed Env)

## TL;DR

Pre-registered claim F1 (monotone decrease in `I(Ẑ_i; Ẑ_j | R)` as `λ_red` increases) is **falsified** in all three scenarios. CMI is essentially flat in λ.

Pre-registered claim F2 (reward strictly worse at every λ > 0) is **not triggered** — reward is essentially constant in λ for trivial_cooperation and default; chain_reaction shows a weak negative trend (within bootstrap CIs).

A *behavioural* version of the prediction — robustness to one-agent dropout — is supported only on **chain_reaction** (≈30% drop reduction at λ = 1e-1 vs λ = 0). On `default` the effect runs the wrong way; on `trivial_cooperation` there is no effect.

This is the cleanest reading of the data the protocol allows. It is publishable as a negative result.

## Headline numbers

Bootstrap mean ± 95% CI over 20 seeds per cell, final-iteration metrics (iter 49 of 50, rollout_steps = 2048).

| scenario | λ_red | reward | CMI (3-D proj, n_bins=4) | dropout drop |
|---|---|---|---|---|
| trivial_cooperation | 0 | 400.29 [399.97, 400.57] | 1.077 [0.91, 1.23] | -5.60 |
| trivial_cooperation | 1e-3 | 400.29 [399.98, 400.57] | 1.073 [0.90, 1.23] | -5.54 |
| trivial_cooperation | 1e-2 | 400.31 [400.00, 400.58] | 1.110 [0.91, 1.29] | -5.47 |
| trivial_cooperation | 1e-1 | 400.28 [399.97, 400.57] | 1.121 [0.93, 1.29] | -5.55 |
| default | 0 | 294.58 [292.16, 297.18] | 0.881 [0.75, 1.01] | 25.73 |
| default | 1e-3 | 294.78 [292.24, 297.56] | 0.928 [0.81, 1.05] | 27.99 |
| default | 1e-2 | 294.79 [292.29, 297.56] | 0.863 [0.75, 0.97] | 29.64 |
| default | 1e-1 | 295.09 [292.19, 298.17] | 0.909 [0.78, 1.03] | 30.86 |
| chain_reaction | 0 | 224.47 [221.18, 227.89] | 0.776 [0.65, 0.90] | 65.30 |
| chain_reaction | 1e-3 | 224.11 [220.21, 228.00] | 0.793 [0.65, 0.93] | 59.68 |
| chain_reaction | 1e-2 | 224.28 [220.82, 228.10] | 0.810 [0.67, 0.94] | 62.40 |
| chain_reaction | 1e-1 | 222.37 [218.29, 226.75] | 0.793 [0.68, 0.91] | 45.27 |

Reward is per-step team total summed over agents (mean over rollout_steps × num_episodes per cell). Dropout drop is `baseline - mean(agent_i_replaced)` over 50 eval episodes per condition.

## Falsifier verdicts

| scenario | F1: CMI monotone ↓ in λ | F2: reward strictly ↓ at λ > 0 |
|---|---|---|
| trivial_cooperation | **falsified** (flat to slightly increasing) | not triggered (penalty doesn't hurt) |
| default | **falsified** (flat) | not triggered (penalty doesn't hurt) |
| chain_reaction | **falsified** (flat) | "falsified-method" (λ=1e-1 reward 222.37 < λ=0 reward 224.47, but CIs overlap) |

The strict pre-registered version of P3 — that conditional MI between encoder outputs decreases monotonically in `λ_red` — is not supported.

## The buggy-sweep artifact

The first sweep (240 cells, run before the env-reset bug was found) showed an apparent positive signal on `default`: dropout drop fell from 103 (λ=0) → 55 (λ=1e-1). With the fixed env this metric goes 25.7 → 30.9 — *opposite direction*. The apparent effect was an artifact of the env-reset bug (`BucketBrigadeEnv.reset()` didn't clear `RUINED` houses across episodes; agents inherited accumulated damage and reward collapsed to ~-400/step regardless of policy). Fixed in commit d6d15c76 (issue #132).

This is exactly the failure mode the pre-registered protocol guards against — claim a positive effect on the basis of a bug. Worth noting in the paper.

## The chain_reaction signal

`chain_reaction` is the one scenario where dropout robustness drops monotonically in λ over the tested range:

```
λ=0     → 65.3 reward drop on single-agent removal
λ=1e-3  → 59.7
λ=1e-2  → 62.4
λ=1e-1  → 45.3  (~30% improvement vs λ=0)
```

The paper predicted exactly this: scenarios with high conditional entropy (distributed sub-task allocation) should benefit most from the specialization regularizer. CMI doesn't move, but the *behavioural* consequence of specialization shows up in robustness. The 3-D projected plug-in CMI may be missing higher-order structure that the linear cross-correlation surrogate is driving.

This is a single-scenario signal, however, and it does not satisfy the protocol's monotone-CMI requirement. Treating it as the headline finding would be cherry-picking. It belongs in the discussion of caveats and follow-ups.

## Methodology caveats

Three reasons F1 might be falsified in this sweep without the underlying prediction being wrong:

1. **The differentiable surrogate is linear; the plug-in measurement is nonlinear.** The training penalty `Σ_{i<j} ||corr(Ẑ_i, Ẑ_j)||_F^2 / d^2` penalizes only *linear* per-feature correlations on standardized features. The CMI we measure quantizes a 3-D random projection of the 64-D encoder output and then runs a plug-in estimator. These quantities are not the same. The penalty can drive linear correlation to zero while leaving nonlinear redundancy untouched.

2. **Team-reward conditioning is near-vacuous.** Team reward `R = Σ_i r_i` is approximately constant within `trivial_cooperation` and small-variance within the other scenarios. CMI ≈ MI in every row of the table. The paper main-text quantity is `I(Ẑ_i; Ẑ_j | R)`; in this sweep it collapses to `I(Ẑ_i; Ẑ_j)`. A useful conditioning variable in this environment is an open methodological question.

3. **Training horizon may be too short.** 50 iterations × 2048 steps × 4 PPO epochs. The reward signal saturates within the first 5 iterations (the agents land at the random-policy baseline immediately and don't improve from there — see [open issue](#what-comes-next)). The redundancy penalty's effect on the encoder may need many more updates to crystallize once the policy is stable.

## Phase 1 plateau diagnostics (added 2026-05-14)

A read-only follow-up (issue [#145](https://github.com/rjwalters/bucket-brigade/issues/145)) plotted the per-iteration metrics already on disk under `experiments/p3_specialization/runs/`. Diagnostic plots and a summary live in `experiments/p3_specialization/diagnostics/`:

- `reward_vs_baseline.png` — per-iteration team reward vs random/heuristic baseline, all three scenarios at λ=0.
- `entropy_per_agent.png` — action-distribution entropy collapses from ~0.66 nats to ~0.12–0.17 nats over 50 iterations.
- `value_loss_log.png` — value loss starts at ~10^5 and decreases slowly.
- `loss_decomposition.png` — after coefficient scaling, `value_coef * value_loss` is ~10^7 × larger than `|policy_loss|` and ~10^7 × larger than `entropy_coef * entropy`.
- `loss_term_scales.png` — bar chart of the same dominance at iter 0 vs iter 49.
- `summary.md` — numerical summary; e.g. on `default`, iter-0 value_term/policy_term ≈ 5×10^6, and reward iter 0→49 goes 293.46 → 294.58 (below random = 308).

These plots confirm the two leading hypotheses in issue #145: (a) the value-loss term dominates the gradient by ~7 orders of magnitude, and (b) the per-agent action-distribution entropy collapses early. A targeted one-variable fix (Phase 2; not yet implemented) is the natural next step — likely return normalisation, a value-loss coefficient sweep, or a higher / scheduled entropy bonus — but should be run on `rwalters-sandbox-1`, not locally.

Regenerate the diagnostics with::

    uv run python experiments/p3_specialization/analyze_plateau.py \
        --sweep-root experiments/p3_specialization/runs

## What comes next

Per the team's current operating posture (`file issues, don't do work`), the followups belong as upstream issues, not as a P3-v2 experiment from this notebook:

- **Reward-attribution mismatch between Python env and Rust core** (bucket-brigade#135). The Rust core emits only per-step work/rest cost; house outcomes only appear in `compute_final_rewards()`. Until this is resolved, thrust-side sweeps aren't comparable to Python-side sweeps.

- **Agents do not exceed random/heuristic baselines** (no upstream issue yet). On `default`, `chain_reaction`, and `trivial_cooperation`, the trained policies match but do not beat random actions. PPO is structurally not learning much. Hypotheses: reward magnitude per step is too large relative to value-head capacity; entropy_coef is too small; rollout horizon is too short. Worth a focused issue separating "P3 mostly null" from "PPO doesn't learn this task."

- **Conditioning variable for CMI**. `R` is vacuous here. Alternatives: coarsened state (number of burning houses?), per-agent observed reward, agent identity one-hot. Each has its own statistical caveats.

- **Differentiable proxy for CMI**. MINE / InfoNCE neural critic, or HSIC, would close the gap between the training surrogate and the measurement.

## Reproducibility

- Sweep grid: 3 scenarios × 4 λ_red ∈ {0, 1e-3, 1e-2, 1e-1} × 20 seeds (42..61) × 50 iterations × 2048 rollout steps × 4 PPO epochs.
- Per-cell wall clock: ~63 s on Mac Studio (M-series CPU). Total sweep: ~4.3 hours.
- Cell artifacts: `experiments/p3_specialization/runs/{scenario}/{lambda}/seed_{N}/` containing `metrics.json`, `config.json`, `policies/agent_*.pt`, `dropout_results.json`.
- Aggregated: `experiments/p3_specialization/analysis.json`.
- Code: this commit + d6d15c76 (env fix) + 99fa47ce (experiment infrastructure).
