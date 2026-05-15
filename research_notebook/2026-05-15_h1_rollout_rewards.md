# H1 diagnostic: per-step rollout reward distribution

**Issue:** [#190](https://github.com/rjwalters/bucket-brigade/issues/190)
**Date:** 2026-05-15
**Author:** Builder (agent-5)

## TL;DR

**Verdict: H1 is *ambiguous-leaning-falsified*.** The pure "near-constant per-step reward" version of H1 (CV < 0.05) is **falsified**: aggregate per-agent CV sits at **0.33-0.36** for the trained Phase-3-stuck cell. However the *spirit* of H1 — that the action-controllable component of the per-step reward is too weak to drive policy improvement — is **broadly supported**: the variance of the per-step reward is almost entirely carried by the team-reward term (residual std ≈ 25), while the analytic work/rest action term has std ≈ 0.4, and the action-reward R² on the full 20-class packed action is **0.011-0.036** — i.e. ≤ ~4% of per-step reward variance is explainable by the agent's own action choice. Trained-policy R² is within a factor of 2-6 of the random-init baseline, indicating the trained policy is barely better than random at producing action-reward coupling.

The right fix is *not* "the signal is constant" (#193's framing of H1 as written). It is: **per-step reward is dominated by a team term that is approximately exogenous from each agent's instantaneous action.** Sibling issues #191 (per-agent attribution) and #193 (reward-shaping) are the natural follow-ups.

## Method

- **Cell:** `runs/p3_183_phase3/L1_norm/default/lambda_0e0/seed_42` from `$COMPUTE_HOST_PRIMARY` (`robbs-mac-studio`); 500 iterations, scenario=`default`, λ_red=0.0, seed=42, `normalize_returns=true`. Rsynced to `/tmp/h1_cell`.
- **Script:** [`experiments/p3_specialization/diagnostics/inspect_rollout_rewards.py`](../experiments/p3_specialization/diagnostics/inspect_rollout_rewards.py).
- **Procedure:** load the 4 trained policies; run one `trainer.collect_rollout(2048)`; stack `rollout.rewards` (a `Dict[int, torch.Tensor]`) into an `ndarray[N=4, T=2048]`; compute per-agent stats, analytic work/rest decomposition, and action-reward R² on the 20-class packed action `a[:,0]*2 + a[:,1]`. Repeat with a fresh random-init policy as a baseline.

The curator-noted dtype claim (`Dict[int, torch.Tensor]`, each `[T]` `float32`) was **verified** at runtime via `assert` — no quiet patching needed.

## Per-agent numbers (trained cell, scenario=default, 500-iter, seed 42)

| Agent | mean   | std    | CV      | p10    | p50    | p90    | work_rate | wr_term std | residual std | R²(packed) | R²(work) |
|-------|-------:|-------:|--------:|-------:|-------:|-------:|----------:|------------:|-------------:|-----------:|---------:|
| 0     | 74.19  | 24.98  | 0.337   | 37.5   | 79.5   | 99.5   | 0.998     | 0.044       | 24.98        | 0.0126     | 0.00019  |
| 1     | 74.88  | 25.19  | 0.336   | 38.5   | 80.5   | 100.5  | 0.137     | 0.344       | 25.17        | 0.0136     | 0.00247  |
| 2     | 75.07  | 24.74  | 0.330   | 38.5   | 80.5   | 100.5  | 0.294     | 0.456       | 24.73        | 0.0109     | 0.00153  |
| 3     | 75.10  | 24.85  | 0.331   | 38.5   | 80.5   | 100.5  | 0.202     | 0.401       | 24.81        | 0.0361     | 0.01273  |

Random-init baseline (same env / config) shows CV ≈ 0.355, R²(packed) ≈ 0.006-0.011 — i.e. **the trained policy lifts R² from ~0.008 to ~0.018 on average, a meaningful but tiny absolute amount; it does not reduce CV** (variance is set by env dynamics, not policy).

## Histogram (trained cell, all (N × T) per-step rewards)

```
  [ -44.500,  -34.767)      4
  [ -34.767,  -25.033)      4
  [ -25.033,  -15.300)      0
  [ -15.300,   -5.567)      8
  [  -5.567,    4.167)     60 #
  [   4.167,   13.900)     39
  [  13.900,   23.633)    337 ########
  [  23.633,   33.367)    100 ##
  [  33.367,   43.100)    688 ################
  [  43.100,   52.833)    208 ####
  [  52.833,   62.567)   1100 ##########################
  [  62.567,   72.300)    372 ########
  [  72.300,   82.033)   2252 #####################################################
  [  82.033,   91.767)    516 ############
  [  91.767,  101.500)   2504 ############################################################
```

The distribution is clearly **non-degenerate** — it's bimodal-skewed with most mass at ~80 (most houses safe ⇒ team_reward ≈ 90 of 100) and a secondary mass near 35-45 (partial saves). The original H1 framing ("near-constant signal, CV < 0.05") doesn't fit this picture.

## Interpretation against #190's rubric

The curator's rubric in the issue:

| Condition | Verdict |
|-----------|---------|
| CV < 0.05 AND R² < 0.01 | H1 confirmed (degenerate signal) |
| CV low, R² > 0.05 | H1 ruled out; PPO failing |
| CV high, R² > 0.05 | H1 ruled out; signal informative |

Our case is **CV high (~0.33) AND R² low (~0.018)**. The rubric doesn't list this cell explicitly. The right reading:

- **CV >> 0.05** → the per-step reward is not "near-constant." The aggregate-CV operationalization of H1 is **falsified**.
- **But R² is also low (~0.018, well under 0.05)** → most of that variance is action-independent. The signal is "informative-about-the-world" (team state changes) but **not informative-about-the-agent's-own-instantaneous-action**.

In other words: the variance in the per-step reward is coming from the team term reacting to environment stochasticity (`prob_house_catches_fire=0.02`, fire spread, etc.), not from the agent's action choice. Per the env code at `bucket_brigade_env.py:248-298`, the team term is `100 * saved_fraction - 100 * burned_fraction`, shared across all 4 agents, and this dominates per-step magnitude (mean ~75 from team vs. mean |wr_term| ≈ 0.4 from work/rest).

This points the post-mortem at **credit assignment / advantage estimation** rather than "the signal is constant":

- The team term is a shared baseline that GAE will mostly route into the value-function and zero out in the advantage (good).
- What remains in the advantage *should* be the action-controllable component — but R²(work) ≈ 0.001-0.013 means the work/rest signal is genuinely faint per-step relative to the residual stochasticity.
- The 10-way location action is similarly weak in R² (R²(packed) is only modestly higher than R²(work)).

So H1's underlying intuition has a kernel of truth — the action-controllable per-step reward is dwarfed by env stochasticity — but the *operationalization* (low aggregate CV) was the wrong proxy. The right diagnostic going forward is **R²(advantage, action)** post-GAE, not R²(raw reward, action).

## Recommended follow-ups

1. **#191 (per-agent attribution audit):** is the per-agent reward actually distinguishing the agent from its teammates? Given the work_rates `(0.998, 0.137, 0.294, 0.202)`, agent_0 is in a "work-always" basin while the others are mostly resting — yet the four mean per-step rewards differ by < 1 unit (74.2 vs 75.1). The team term is *swamping* per-agent distinctions.
2. **Post-GAE diagnostic:** compute R² of the *GAE advantage* on the action (not the raw reward). This is the quantity PPO's loss actually depends on. The reward signal could be informative-in-principle but the advantage flat-in-practice (or vice versa).
3. **#193 (reward-shaping proposal):** consider down-weighting the team term in the per-agent reward (e.g. credit-assignment baseline subtraction, or a difference-rewards/AAMAS-style decomposition). Even halving the team coefficient would lift R²(packed) by an order of magnitude purely from the variance-ratio shift.

## Artifacts

- Script: `experiments/p3_specialization/diagnostics/inspect_rollout_rewards.py` (committed).
- Raw JSON summary: `/tmp/h1_summary.json` (not committed — reproducible from the script).
- Trained cell: `/tmp/h1_cell` (not committed; rsync from `$COMPUTE_HOST_PRIMARY` per script docstring).

## One-line verdict

**H1 as written (CV < 0.05) is falsified, but its underlying intuition is supported: per-step reward variance is dominated by environment stochasticity in the team term, leaving the action-controllable component below R² ≈ 0.02 on the packed 20-class action — too faint for PPO to extract reliably.**
