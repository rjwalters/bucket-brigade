# H3 diagnostic: re-derived "308" random baseline for the P3 specialization sweep

**Date:** 2026-05-15
**Issue:** [#192](https://github.com/rjwalters/bucket-brigade/issues/192)
**Diagnostic:** `experiments/p3_specialization/diagnostics/random_baseline.py` (NEW; canonical going forward)

## TL;DR

| Source | Per-step team reward | Notes |
|---|---|---|
| **Uniform-random (1000 ep, n=5 seeds)** | **293.39, 95% CI [288.87, 297.78]** | re-derived this PR |
| **Random-init MLP iter-0 (250 ep, n=5 seeds)** | **287.74, 95% CI [278.61, 296.67]** | this PR; matches #183's L1_norm iter-0 = 290.52 |
| Uniform-random (50 ep, single seed, #145 protocol) | 289.46, 95% CI [271.91, 306.09] | this PR (re-run for backwards compat) |
| Cited "308" (#145 body) | 308 | hard-coded in `analyze_plateau.py:48-52`, `analyze_174.py:47`; **outside the n=1000 CI** |

**Verdict:** the cited 308 is **about 5% too high**. Uniform-random and random-init MLP agree with each other (well within their respective CIs of each other), and both agree with the iter-0 value from #183's phase-3 cells. They do **not** agree with 308. The original 50-episode measurement in #145 has 308 sitting just past the upper edge of its 95% CI (306.09) — it was a slightly unlucky sample on the right tail of a high-variance reward distribution.

## Methodology

`experiments/p3_specialization/diagnostics/random_baseline.py` runs three passes on `default_scenario(num_agents=4)`:

1. **Uniform-random:** 1000 episodes (5 seeds × 200). Per step, sample `actions[:, 0] ~ U{0..9}` and `actions[:, 1] ~ U{0..1}` for each agent — matches the `MultiDiscrete([10, 2])` action space verified at `puffer_env.py:77` and the `(N, 2)` shape documented at `bucket_brigade_env.py:117`.
2. **Random-init MLP iter-0:** 250 episodes (5 seeds × 50). Constructs `JointPPOTrainer` with the #183 phase-3 `CellConfig` defaults (`hidden_size=64`, `num_agents=4`, `action_dims=[10, 2]`, seed 42–46) and runs episodes through the untrained `PolicyNetwork`s. **No PPO updates.**
3. **#145 reproduction:** 50 episodes single seed, the original protocol from the issue body.

All per-step normalization uses the **actual `env.night` at done**, not a fixed 13. `default_scenario` has `min_nights=12` (`scenarios_generated.py:83`); episodes terminate after `min_nights` once no fires remain (`bucket_brigade_env.py:303-314`). The observed median episode length is **13 nights**, mean 13.18, range 13–16.

Reporting uses bootstrap 95% CIs (10000 resamples) over the per-episode samples.

## Provenance of "308"

Curator located the chain in the issue #192 comment:

- **Origin:** issue #145 body reports "Random actions across 50 episodes on `default`: 4012.34 per episode (~+308/step)." The script that produced 4012.34 was never committed.
- **Hard-coded thereafter:**
  - `experiments/p3_specialization/analyze_plateau.py:48-52` — `BASELINES["default"]["random"] = 308.0`
  - `experiments/p3_specialization/analyze_174.py:47` — `RAND_BASELINE = 308.0`
- **Transitively cited in:** `diagnostics/summary.md`, `results_174_ablation/summary.md`, `research_notebook/2026-05-14_p3_specialization_results.md`, `research_notebook/2026-05-15_p3_plateau_ablation.md`.

The issue body's "13 nights" was an approximation; correcting to actual episode length (median 13) reproduces ~298 from the re-derived `4012.34 / 13.18 = 304.4` framing — **slightly closer to 308 but still high.** The bigger source of the gap is sampling variance on n=50.

## Why uniform-random and random-init MLP agree (and 308 doesn't)

- **Sample noise:** at n=50, the bootstrap CI for per-step is ±17. At n=1000, ±5. The original 4012.34/308 number is one realization from the right tail of a high-variance distribution.
- **No structural bias from MLP init:** `PolicyNetwork`'s un-trained softmax over the two discrete heads is close enough to uniform that iter-0 lands on the same ~290 as uniform-random. **PPO at iter-0 is not "worse than random" — it is at random.**
- **Per-step vs per-episode is robust:** because actual length is 13 ± 1 (not the assumed 13), the difference between `total / 13` (308.6) and `total / actual_length` (~293) accounts for ~5 reward of the gap.

## Implications for prior P3 results (informational; no code changes here)

This diagnostic is **diagnostic-only**. It does NOT change `BASELINES["default"]["random"] = 308.0` in `analyze_plateau.py` or `RAND_BASELINE = 308.0` in `analyze_174.py` — that's a separate decision out of scope for #192.

If the true random baseline is ~293 rather than 308:

- Phase-3 cells reported at "iter-0 ≈ 290.52" are **at random**, not below it. The narrative "PPO performs below random" is incorrect under the corrected baseline.
- Acceptance bars derived from 308 (e.g. `ACCEPTANCE_BAR = 320.0` in `analyze_plateau.py`, "> 320" in `results_174_ablation/summary.md`) are arguably ~12 reward too high.
- A separate issue can decide whether to (a) update the hard-coded constants, (b) lower the acceptance bar, (c) re-frame the "below random" wording, or some combination.

## Durable artifact

Per the issue's bonus deliverable: since #145's measurement has no committed script, **`experiments/p3_specialization/diagnostics/random_baseline.py` is now the canonical random baseline going forward.** Re-running it on a different machine should reproduce ~293 ± 5 (per-step) and ~3870 ± 60 (per-episode) within the cited CIs.

## Reproduction

```bash
# Full sweep (1000 random + 250 MLP, ~2 min on CPU)
uv run python experiments/p3_specialization/diagnostics/random_baseline.py

# Reproduce #145 protocol exactly (50 episodes, single seed)
uv run python experiments/p3_specialization/diagnostics/random_baseline.py \
    --episodes-per-seed 50 --seeds 1 --no-mlp
```
