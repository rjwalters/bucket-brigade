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

## Amendment (2026-05-16): chain_reaction sibling re-derivation (issue #219)

Sibling baseline to PR #213's `default`-only audit: the `chain_reaction`
random baseline cited as `233.0` in `analyze_plateau.py` and
`diagnostics/summary.md` has the same uncommitted #145 n=50 provenance as
`default`'s old `308`. Re-derived on `COMPUTE_HOST_PRIMARY` against current
main (commit `b053d38e`, which contains the post-#197/#198 reward function)
using `experiments/p3_specialization/diagnostics/random_baseline.py
--scenario chain_reaction --episodes-per-seed 200 --seeds 5` (n=1000 total):

- **Uniform-random per-step team reward on `chain_reaction`**: **220.75, 95% CI [215.39, 225.86]** (n=1000).
- **Uniform-random per-episode**: 3573.19, 95% CI [3486.31, 3656.27].
- **Episode length**: median 16, mean 16.18, range [16, 20]. `chain_reaction` has `min_nights=15` (vs 12 for `default`); fire-spread frequently extends episodes 1–5 nights past the floor.
- **Random-init MLP iter-0 per-step**: 218.79, 95% CI [208.96, 228.54] (n=250). Mean differs from uniform-random by 1.96 (well inside the ±5 verdict window); CIs overlap heavily — the MLP and uniform-random policies are statistically indistinguishable at this n.
- **#145-protocol reproduction (n=50, single seed)**: 235.41, 95% CI [214.23, 257.33]. The cited 233 sits inside this wide CI, mirroring PR #213's finding that `default`'s 308 was an unlucky tail-sample on a high-variance reward distribution.

Side-by-side:

| baseline | per-step mean | 95% CI | provenance |
|---|---|---|---|
| #145 cited | 233.0 | — (n=50, single seed) | uncommitted protocol |
| **#219 re-derivation (post-#197/#198, current)** | **220.75** | **[215.39, 225.86] (n=1000)** | this run / `b053d38e` |
| #219 #145-reproduction (n=50) | 235.41 | [214.23, 257.33] | this run (sampling-noise demo) |

**Verdict:** `Uniform-random per-step CI contains cited 233.0: False` —
the cited number sits ~5% above the n=1000 mean and outside its CI.
Same pattern as `default`'s old 308 (also ~5% above the n=1000 mean,
also outside the post-#197/#198 CI).

**Implications:**

- `BASELINES["chain_reaction"]["random"]` in `experiments/p3_specialization/analyze_plateau.py` updated from `233.0` → `220.75`.
- `experiments/p3_specialization/diagnostics/summary.md` line 24 baseline literal updated; the surrounding `reward iter 0 -> iter 49: 224.21 -> 224.47` numbers are unchanged (they come from the existing `runs/` data, not the constant).
- The H3 regression test (`tests/test_env_health_diagnostics.py`) remains `default`-only by design; extending it to `chain_reaction` is out of scope for this issue.
- The `chain_reaction` heuristic baseline (`226.0`) still traces to the same uncommitted #145 protocol and remains flagged for a separate heuristic-policy diagnostic pass (out of scope for #219).
- The narrative reading of the prior P3 sweep on `chain_reaction` (iter-49 reward `222.37 [218.29, 226.75]` per `2026-05-14_p3_specialization_results.md:38`) is unchanged in spirit: that CI overlaps the new `220.75` random CI almost exactly, so the "PPO sits at random on `chain_reaction`" framing is preserved — it just now sits *at* the corrected random baseline rather than *below* the inflated one.

The diagnostic script (`experiments/p3_specialization/diagnostics/random_baseline.py`) was extended in this PR with a `SCENARIO_CITED_VALUES` table (curator's Option A from issue #219's enhancement). `chain_reaction` is now a table entry; the verdict block generalizes per-scenario without hardcoded `if scenario == "default"` branches.

References: issue #219; PR #213 (sister `default` post-#197/#198 fix);
issue #218 (its parent audit); PR #228 (the `--scenario` CLI foundation
this PR builds on); PR #196 (canonical re-derivation pattern); issue
#145 (origin of the wrong 233); issue #202 (audit-policy issue that
deferred `chain_reaction` from PR #213 to here).

## Amendment (2026-05-16): post-#236 re-derivation across all 14 scenarios (issue #237)

PR #236 (commit ``dffe1060``) widened the joint action space to
``MultiDiscrete([10, 2, 2])`` — the third dim is the broadcast signal,
now sampled independently rather than mirrored from the work/rest bit.
This invalidates all pre-#236 cited baselines (including the 220.75 and
247.58 figures above).

Re-derived on ``COMPUTE_HOST_PRIMARY`` for all 14 named scenarios at
commit ``dffe1060`` (n=1000 each via ``--episodes-per-seed 200 --seeds 5
--no-mlp``). Headline shifts:

- `default`: 247.58 → **251.23** (CI [244.86, 257.51]; +3.65)
- `chain_reaction`: 220.75 → **227.39** (CI [221.96, 232.70]; +6.64)
- `minimal_specialization`: -96.07 → **-87.72** (CI [-93.31, -82.16]; +8.35, sign preserved)
- `positional_default`: 247.09 → **250.73** (CI [244.36, 257.01]; +3.64)

Full 14-scenario table is in
``research_notebook/2026-05-14_p3_specialization_results.md`` (2026-05-16
post-#237 amendment).

**Critical test-suite bug fix surfaced by this audit:**
``tests/test_env_health_diagnostics.py::_random_baseline_per_step_default``
(lines 249-257 pre-fix) was still sampling 2-dim actions after PR #236 —
PR #236 missed this call site, so the H3 regression test was silently
measuring a pre-#236 policy. Fixed in this PR to 3-dim sampling with an
anti-regression grep-assert against
``random_baseline.ACTION_DIMS == [10, 2, 2]``. Without this fix the H3
verdict was meaningless. The H3 window ``H3_RANDOM_PER_STEP_RANGE_DEFAULT
= [220, 290]`` already brackets 251.23 — no widening required.

**MLP iter-0 confirmation:** re-ran with ``--mlp-episodes-per-seed 50``
on `default`; random-init MLP per-step came back at 247.40 (CI
[234.76, 259.62]), within ±5 of the new uniform-random mean as expected
(curator predicted unchanged since #236 did not alter the MLP forward
pass). The 290.52 figure cited in `SCENARIO_CITED_VALUES["default"]
["mlp_iter0"]` retains its separate provenance (specific seeded #183
phase-3 cell, not a free MLP-init average).

References: issue #237; PR #236 (signal as first-class action,
``dffe1060``); PRs #218 and #229 (immediate prior re-derivations now
superseded).
