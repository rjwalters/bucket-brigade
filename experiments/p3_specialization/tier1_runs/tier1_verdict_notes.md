## Notes

### het_ppo / rest_trap (#429 / #434): recalibrated — no longer scored on the gap ladder

The vacuous `closed` verdict originally recorded for this row was a
baseline-scale artifact: the pre-#434 `run_tier1_cell.py::gap_closed` scored
**every** scenario against the `minimal_specialization` references, and on
rest_trap's reward scale uniform-random play alone mapped to
`gap_closed ≈ 6.58` (`iter0_gap_closed_mean = 6.573`), so the 0.49 gate was
passed by any policy, trained or not.

Issue #434 made `gap_closed` scenario-aware
(`bucket_brigade.baselines.SCENARIO_GAP_REFERENCES`) and this table was
regenerated from the committed per-seed `metrics.json` via
`run_tier1_cell.py --summarize-only` (no training re-runs). The het_ppo row
now reports:

- `gap_closed = null`, `gap_source = "degenerate_reference"`, verdict
  `not_scored_degenerate_reference` — the fraction ladder is **not
  applicable** on rest_trap. The frozen NE (`3×free_rider + 1×firefighter`,
  `team_payoff = 2984.04`/episode, i.e. ≤ 248.7/step at ≥ 12 nights;
  `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`) sits
  *below* the 302.87/step random baseline: rest_trap's equilibrium is
  team-suboptimal by construction (social trap), so no valid upper reference
  exists yet. (The NE payoff is per-episode while gap metrics are per-step —
  the ≤ 248.7/step bound suffices to establish degeneracy.)
- The honest scenario-scale headline is `uplift_over_random = +3.39 ± 7.34`
  /step (trailing-5 `306.26` vs random `302.87`, n=20 seeds) — marginal, not
  a rescue, consistent with #356's expectation that tier-1 trainers fail on
  rest_trap.
- The pre-#434 MINSPEC-scale value is preserved as the audit column
  `gap_closed_minspec_legacy_mean = 6.639`.

### minimal_specialization rows: gap values refreshed to the #416 specialist constant

The four `*_minimal_specialization` summaries committed in #346 were computed
at commit `734bec7e` with the historical `MINSPEC_SPECIALIST = -22.07`
(n=50); issue #416 re-derived the constant to `-28.38` (n=10k) but the
committed artifacts were never recomputed — there was no recompute tool until
`--summarize-only` (#434). The regeneration therefore shifted the minspec gap
fractions slightly (denominator 65.65 → 59.34; e.g. ippo `0.1128 → 0.1248`)
while the underlying per-seed `trailing5` team rewards and all four
`insufficient` verdicts are unchanged. The scenario-aware formula itself is
bit-identical to the historical MINSPEC formula for this scenario
(`SCENARIO_GAP_REFERENCES["minimal_specialization"]` = `(MINSPEC_RANDOM,
MINSPEC_SPECIALIST)`, drift-guarded in `tests/test_baselines_constants.py`).

### het_ppo Phase 2 (asymmetric_only phase-diagram cells): blocked

Phase 1 ran to completion (see recalibrated row above), but the Phase 2
scenarios (`asym_b05_k05_c09`, `asym_b05_k09_c09`) are not registered in
`bucket_brigade/envs` — they exist only in the runbook/launcher docs. Phase 2
is blocked on the #358 cell-to-scenario follow-up (see
`experiments/p3_specialization/het_ppo_runbook.md` and issue #429).

### het_ppo / rest_trap (#436): trap-escape verdict — `escaped_trap` (marginal)

Issue #436 added a measured upper anchor and a categorical four-way
trap-escape verdict for degenerate-reference (social-trap) rows, computed by
`run_tier1_cell.classify_trap_verdict` at re-summarization time
(`--summarize-only`, no retraining):

- **Anchors** (per-step team reward): NE bound ≤ 248.67 (frozen NE payoff
  2984.04/episode ÷ min_nights = 12), random 302.87
  (`SCENARIO_RANDOM_BASELINES`), and the measured `scripted_best` = 386.60
  [386.17, 387.03] (homogeneous `specialist` team, n=10k, issue #436 Part A;
  artifact `experiments/p3_specialization/scripted_battery/rest_trap.{json,md}`).
- **Rule**: nested one-sided ladder on the lower bound of the seed-bootstrap
  95% CI of the trailing-5 per-step team reward (10k resamples, fixed seed):
  `lo > scripted_best_ci_hi` → `above_scripted_best`; `lo > random` →
  `escaped_trap`; `lo > ne_bound` → `at_random`; else `trapped_at_ne`.
- **het_ppo result (20 seeds)**: trailing-5 mean 306.26, CI [302.95, 309.33]
  → **`escaped_trap`**, but *marginal*: the CI lower bound clears the random
  anchor by only 0.08/step, and the #436 fresh n=10k re-measurement of the
  uniform baseline (302.94 [301.46, 304.31]) sits essentially on that lower
  bound. Read this as "statistically distinguishable from random under the
  committed anchor, far below scripted_best (Δ ≈ −80/step)" — PPO escapes
  the *resting* trap direction but captures almost none of the measurable
  scripted headroom. The scripted battery also shows `always_rest` (288.55)
  and the NE bound (≤ 248.67) are both *below* random: the trap is real, and
  a fully-scripted specialist team demonstrates the scenario is not
  reward-capped at random-level play.
