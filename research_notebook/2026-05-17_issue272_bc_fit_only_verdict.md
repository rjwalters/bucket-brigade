# Issue #272: Specialist BC-fit-only diagnostic — verdict

**Date:** 2026-05-17
**Branch:** `feature/issue-272`
**Script:** `experiments/p3_specialization/bc_fit_only.py`
**Result JSON:** `experiments/p3_specialization/bc_fit_only_result.json`
**Scenario:** `minimal_specialization` (4 agents × 10 houses, round-robin ownership)

## TL;DR — Verdict: INDUCTIVE_BIAS_GAP with strong evidence of an architecture-level failure on the discriminating sub-task

Strict per-curator thresholds give **INDUCTIVE_BIAS_GAP** (eval loss = 0.59, joint accuracy = 0.94 — between the two clean cells). But the per-head breakdown shows the headline accuracy is an artifact of class imbalance, not real fitting. **On the 5% of rows where the specialist actually does something (mode == WORK), the house head sits at 0.35 accuracy — essentially chance over the 3 owned houses (1/3 ≈ 0.33)**. The trunk has learned the per-agent "favorite REST house" lookup but has **not** learned to route the identity one-hot together with the burning-houses signal into a specific owned-burning index. This is the smallest plausible mismatch the curator predicted.

## Headline numbers

| Metric | Value |
|---|---|
| `obs_dim` | 42 (10 houses + 4 signals + 4 locations + 8 last_actions + 12 scenario_info + 4 identity) |
| `n_params` | 7,887 (default `hidden_size=64`) |
| `n_train / n_eval` | 8,000 / 2,000 |
| `final eval_loss` | 0.5903 (after 50 epochs; converged) |
| **house acc (all rows)** | **0.964** |
| **mode acc** | **0.943** (= REST-class prior — head is not discriminating) |
| **signal acc** | **0.943** (= REST-class prior — head is not discriminating) |
| **joint acc** | **0.943** |
| **house acc \| mode=WORK** | **0.354** (n=113; ≈ 1/3 chance over 3 owned houses) |
| Data work fraction | 0.050 (low; specialist is dominated by REST) |
| Reproducibility (seed=0) | bit-exact (0.8058 at epoch 10, both runs) |

## Loss curve (10 epochs)

```
epoch  1: train=2.1354  eval=1.9745  house=0.279  mode=0.943  signal=0.943  joint=0.275
epoch  5: train=1.6884  eval=1.7134  house=0.939  mode=0.943  signal=0.943  joint=0.920
epoch 10: train=0.7606  eval=0.8058  house=0.964  mode=0.943  signal=0.943  joint=0.943
```

Loss continues to fall after epoch 10 but plateaus by epoch ~30 around eval=0.59. Crucially, **per-head accuracy stops improving after epoch 7**: the heads lock onto the majority-class strategy and never escape.

## Per-head diagnostic — the key finding

Each head's headline accuracy is exactly the class prior of its dominant label:

- `mode`: 0.943 ≈ P(mode == REST) on this dataset
- `signal`: 0.943 ≈ same (specialist signals honestly, so signal == mode)
- `house`: 0.964 ≈ P(predicted == agent's lowest-owned-house) because under REST the specialist always picks that fixed index per agent

In other words **all three heads have learned to ignore the obs and predict the per-agent majority action**. The 4-bit identity tail is being used (per-agent constant predictions differ across agents — otherwise house accuracy would be 1/10 = 0.10, not 0.96), but the `houses` slice is essentially being ignored.

The conditional accuracy on the discriminating sub-task is the smoking gun:

| Subset | n | house accuracy |
|---|---|---|
| All eval rows | 2000 | 0.964 |
| `mode == REST` | 1887 | ≥ 0.98 (predicts identity-conditioned constant) |
| **`mode == WORK`** | **113** | **0.354** |

A specialist trained to perfection scores 1.000 on both subsets. Random over the 3 owned houses scores 0.333. The model scores 0.354 — barely above random over owned. It has not learned the burning-owned argmin at all.

## Implication for the PPO debugging plan

Per curator's threshold table, the verdict is "**INDUCTIVE_BIAS_GAP**" — eval loss is between the clean cells (0.05 < 0.59 < 0.5? — actually loss 0.59 > 0.5, but min-head accuracy 0.94 is well above 0.50, so we fall to the in-between bucket).

But the per-head story upgrades the practical interpretation:

> **The standard `PolicyNetwork(obs_dim=42, action_dims=[10,2,2], hidden_size=64)` cannot represent the specialist's burning-owned-argmin behavior even under direct supervision with 8k labeled examples.** The trunk has learned to use the identity one-hot for constant per-agent predictions, but has not learned to route the identity bits together with the `houses` slice to select among an agent's *owned* houses when the burning-state matters.

**Practical reading**: PPO failure on `minimal_specialization` is **not purely** a path-finding / exploration / credit-assignment problem. There is a real representational gap on the sub-task that produces the bulk of the reward. Sister issues #270 (BC-init then PPO) and #271 (random-init best-of-N) are still worth running — but if BC alone cannot get >0.5 accuracy on the WORK subset, BC-init will not bootstrap PPO out of the basin either.

## Recommended follow-ups (do NOT do in this PR — scope discipline)

These are concrete next steps that would unblock the architecture story; file as separate issues if approved:

1. **Wider/deeper trunk**: rerun this exact diagnostic with `hidden_size=128`, then `hidden_size=256`, then a 3-layer trunk. If WORK accuracy jumps to ≥0.9 at any of these, we know the capacity is the bottleneck and the PPO net should be widened repo-wide.
2. **Transformer policy** (`TransformerPolicyNetwork` already exists): rerun with the transformer; the identity-conditioned attention over houses is exactly what this task needs.
3. **Class-balanced BC**: oversample WORK rows (or use focal/weighted CE) — pure diagnostic, not a fix; would confirm that the model *can* fit the WORK cases when not drowned by REST.
4. **Identity-aware obs encoding**: instead of a 4-bit one-hot tail, gate the `houses` slice by an owned-mask derived from `agent_id` *before* the trunk. (Architecturally invasive — would change every PPO experiment.)

## What this changes about the PPO failure-mode hypothesis

Project memory currently reads:

> independent-PPO still plateaus at 15% of optimum on a sanity-check scenario. Next intervention is MAPPO / CTDE.

This diagnostic suggests that interpretation is **incomplete**. MAPPO/CTDE changes the value-baseline side of the algorithm but not the actor architecture, so it should not be expected to fix a representational gap in `PolicyNetwork`. The first follow-up (item 1 above — wider trunk) is cheap and may be the real fix, or at least a necessary precondition.

I am flagging this in project memory, not editing the original entry — the empirical finding (#197/#198/#199 working, MAPPO open) is unchanged; what's added is a representational concern that lives upstream of any RL algorithm choice.

## Files

- `experiments/p3_specialization/bc_fit_only.py` — implementation (251 lines, CLI-driven)
- `experiments/p3_specialization/bc_fit_only_result.json` — full run history at default args (`--epochs 10 --seed 0`)

## Reproduction

```bash
uv run python experiments/p3_specialization/bc_fit_only.py
# default: --num-steps 2500 --epochs 10 --seed 0
# runtime: ~5 sec data gen + ~3 sec train = <10 sec on CPU
```

Reproducibility verified: two runs at `--seed 0` give bit-identical final `eval_loss` (0.8058 at epoch 10).
