# Tier-1 sweep verdict

Verdict ladder: `gap_closed_mean >= 0.88` -> **closed**; `0.49 <= mean < 0.88` -> **partial_upper**; `0.20 <= mean < 0.49` -> **partial_lower**; `mean < 0.20` -> **insufficient**. Rows with a null `gap_closed` (**not_scored** / **not_scored_degenerate_reference**, #434) are never classified on the ladder; read their `uplift_over_random` (per-step, scenario scale) instead.

| Trainer | Scenario | gap_closed (mean ± std) | uplift_over_random (mean ± std) | n_seeds | Verdict |
|---------|----------|--------------------------|---------------------------------|---------|---------|
| ippo | minimal_specialization | 0.125 ± 0.082 | +7.405 ± 4.862 | 3 ok | insufficient |
| influence | minimal_specialization | 0.120 ± 0.036 | +7.101 ± 2.121 | 3 ok | insufficient |
| hca | minimal_specialization | 0.081 ± 0.139 | +4.806 ± 8.268 | 3 ok | insufficient |
| lola | minimal_specialization | 0.005 ± 0.053 | +0.267 ± 3.132 | 3 ok | insufficient |
| het_ppo | rest_trap | n/a | +3.394 ± 7.340 | 20 ok | not_scored_degenerate_reference |

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
