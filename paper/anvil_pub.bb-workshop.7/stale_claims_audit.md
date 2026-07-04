# Stale-claim audit: anvil_pub.bb-workshop.6 → .7 (issue #446)

Scope mandated by issue #446: sweep the paper for every claim citing
**conditional entropy**, **gap_closed on rest_trap**, **het_ppo**, or
**tier-1 verdicts**, and reconcile each against the current committed
artifacts (`experiments/p3_specialization/tier1_runs/tier1_verdict.md`,
`experiments/nash/phase_diagram/entropy_vs_trainability.md`,
`experiments/p3_specialization/scripted_battery/rest_trap.md`,
`docs/PAPER_RESULTS.md` §6–7). The audit baseline is v6
(`paper/anvil_pub.bb-workshop.6/main.tex`), the latest revision in the
artifact trail.

## Part A — the four mandated claim families

Grep sweep of v6 `main.tex` (case-insensitive) for `entropy`, `het_ppo`,
`tier`, `gap_closed`-on-rest_trap contexts, and `rest_trap`:

| Claim family | Occurrences in v6 | Finding |
|---|---|---|
| Conditional entropy as trainability predictor | **0** | The paper never adopted the #368 entropy predictor, so no in-paper claim needed retraction. The retirement (ρ = 0.007, p = 0.97) is **added** to §4 as a first-class negative result so the paper cannot be read as leaving the predictor alive. |
| `gap_closed` on rest_trap (incl. any "gate pass") | **0** | v6 never scored rest_trap on the gap ladder. The historical vacuous "pass" lived in the pre-#434 tier-1 tooling, not the paper. §5 (new) documents the mechanism and **explicitly retracts** the vacuous pass so no earlier internal write-up can be cited as a trap escape. |
| het_ppo | **0** | No het_ppo claim existed. §5 (new) adds the corrected reading: verdict `at_random`, CI [302.95, 309.33] vs random-anchor upper bound 304.31; `uplift_over_random = +3.39 ± 7.34`/step; never described as a gate pass. |
| Tier-1 verdicts | **0** | No tier-1 claims existed. §5 adds the minimal_specialization tier-1 battery context (ippo/influence/hca/lola all `insufficient`, gap_closed 0.005–0.125) with citation. |

v6's three `rest-trap` mentions (§2 template naming, Fig. 1 caption, App.
A free-rider template) are definitional, carry no measurements, and are
unchanged.

**Net Part A finding: 0 stale claims to correct; 4 additions made so the
paper reflects the corrected evidentiary base rather than omitting it.**

## Part B — stale claims actually present in v6: β-independence (issue #442)

The sweep surfaced a family of genuinely stale claims the issue's caveat
(a) anticipates: v6 presents β as a live sweep dimension and counts
"β-independence" as an empirically confirmed prediction of the mean-field
reduction. Issue #442 (verified mechanistically during PR #441 review)
shows that in the bernoulli extinguish mode the sweep uses, burn-out
clears every burning house before the spread phase, so spread never
fires, draws zero RNG, and the environment is **bit-identical across β**
— β-invariance holds by construction and cross-β residuals measure
double-oracle solver noise.

| # | Location (v6) | Old text (abridged) | Corrected text (v7, abridged) | Artifact citation |
|---|---|---|---|---|
| B1 | Abstract | "The reduction correctly predicts the qualitative phase order … and β-independence" | "…its β-independence prediction turns out to hold *by construction* in the sweep's bernoulli extinguish mode … cross-β residuals measure solver noise rather than testing the reduction" | issue #442 (mechanism: `bucket-brigade-core/src/engine/{core,phases}.rs`) |
| B2 | §3 "Predicted vs. observed" | "The single splitting row is (κ=0.9, c=0.5) … a small but real anomaly we return to in §6" | New caveat paragraph: β-invariance is exact by construction; the splitting row and the cross-β payoff deltas (80.9 vs 72.0 at (κ=0.9, c=0.5); −614.4 vs −648.0 at (κ=0.5, c=0.5)) are direct measurements of double-oracle solver nondeterminism; NE *payoffs* carry an unquantified solver-noise error bar, verdicts (12/13 rows) are more robust | `experiments/nash/phase_diagram/results.json` (bit-identical env, differing payoffs); issue #442 |
| B3 | §3 "Headline finding" | "correctly predicts *three* structural properties … the qualitative phase order, β-independence …, and the … collapse regime" | "correctly predicts *two* structural properties … (A third prediction, β-independence …, is correct but … true by construction …, so we no longer count it as an empirical confirmation.)" | issue #442 |
| B4 | Fig. 1 caption | "…matching the prediction in (S)–(C) but at substantially different κ thresholds" | "…the environment is bit-identical across β (repo issue #442), so the β axis functions as a solver-noise probe rather than a test of (S)–(C)" | issue #442 |
| B5 | §4 Results | "PPO is consistent with β-independence but does not sharpen the test beyond the solver-level evidence" | "cross-β variation in the PPO metric at fixed (κ, c) cannot reflect differences in game dynamics and instead bounds the run-to-run noise of the training pipeline" | issue #442; `entropy_vs_trainability.md` β-invariance table |
| B6 | §6 Threats | "rigorous separation requires a ρ-sweep to falsify β-independence outside the small-βρ regime" | "requires a sweep in which β is actually live (contagion-active phase order or continuous-extinguish variant) — in the bernoulli mode used here β is provably inert, so no ρ-sweep can activate it" | issue #442 |
| B7 | App. B "β-independence as leading-order, not exact" | "the second-order claim … is approximately true … but should break for βρ ≫ ρ … sweeping ρ ∈ {0.05, 0.10} would let the framework be falsified" | Paragraph rewritten: "exact by construction in bernoulli mode" — spread never has a Burning source in *any* night; no across-night path for β; cross-β solver deltas quantified as a free noise probe | issue #442; `results.json` |
| B8 | App. B follow-ups item (2) | "sweep ρ ∈ {0.05, 0.10} at fixed c=0.5 to test the β-independence breakage" | "re-run a β slice under a variant in which β is live … so the reduction's β treatment can actually be falsified (in bernoulli mode it cannot be)" | issue #442 |
| B9 | Conclusion | "correctly predict the qualitative phase order and the β-independence under the canonical phase order" | "correctly predict the qualitative phase order … (its β-independence prediction, while correct, holds by construction … and now serves as a solver-noise probe rather than a confirmation)" | issue #442 |

**Net Part B finding: 9 stale β-independence claims corrected.**

## Part C — additions required by issue #446 (traceability)

| Addition | Where | Every number traces to |
|---|---|---|
| Negative result: entropy predictor retired (ρ = 0.007, p = 0.97, n = 31; Bonferroni α = 0.0063 kills the lone p = 0.039 entry) | §4 | `experiments/nash/phase_diagram/entropy_vs_trainability.{py,json,md}` |
| β-invariance impossibility argument (identical NE profile across β; gap range up to 0.36; example −0.32/−0.11/−0.00 at κ=0.1, c=1.0) | §4 | `entropy_vs_trainability.md` β-invariance table |
| Noise ceiling (per-cell std ≤ 0.99, median 0.24, n = 4 seeds) | §4 | `entropy_vs_trainability.md`; `phase_diagram_ppo_v2/recalibrated_verdict.json` |
| k\* hypothesis, validation **pending** (thrust#269 artifact not landed; 20-seed buy-down open) | §4 | repo issue #443; rjwalters/thrust#259/#268/#269 |
| k = 1 oracle constraint (+13.4/step, paired 95% CI [+10.5, +16.6]) | §4 | `experiments/nash/phase_diagram/improvability_oracle.md` |
| Four-anchor ladder: NE ≤ 248.67 < always_rest 288.55 [285.20, 291.65] < random 302.87 / 302.94 [301.46, 304.31] < het_ppo 306.26 [302.95, 309.33] < scripted 386.60 [386.17, 387.03]; paired Δ +83.67 [+82.36, +84.89] | §5 (new), Table 4 | `scripted_battery/rest_trap.{json,md}`; `tier1_runs/tier1_verdict.md`; `bucket_brigade/baselines/release/local/nash/rest_trap-v1.json`; `docs/PAPER_RESULTS.md` §7 |
| Vacuous-pass retraction (pre-#434 gate: uniform-random alone → gap_closed ≈ 6.58 ≫ 0.49 gate) | §5 | `tier1_verdict.md` notes (§ het_ppo/rest_trap #429/#434) |
| Trap-escape verdict rule (seed-bootstrap CI, nested ladder, random-ci95-hi rung rationale ±1.4/step) | §5 | `docs/PAPER_RESULTS.md` §7; issue #436; PR #440 |
| het_ppo `at_random`; `uplift_over_random = +3.39 ± 7.34`/step; no positive interventional evidence for asymmetry rescue | §5 | `tier1_verdict.md`; `docs/PAPER_RESULTS.md` §7 |
| Tier-1 minspec battery context (`insufficient`, 0.005–0.125) | §5 | `tier1_verdict.md` |
| Solver-noise hedge on rest_trap NE payoff (robustness via always_rest < random) | §5 | issue #442 point 2; `scripted_battery/rest_trap.md` |
| Reproducibility: frozen `-v1` registry, parity manifest + CLI, ~7× incident, reporting convention | §6 | PR #441 (`bucket_brigade/envs/registry.py`); PR #439 / `docs/PARITY.md`; PR #432 |

## Summary

- **Stale claims corrected: 9** (all in the β-independence family; Part B).
- **Mandated claim families with stale in-paper claims: 0 of 4** — the
  paper had never adopted the entropy predictor, the rest_trap gate pass,
  or any het_ppo/tier-1 claim; v7 adds the corrected material (Part A/C)
  so the paper and the committed artifacts are in lockstep.
- **Fresh caveats landed:** β-inertness in bernoulli mode (issue #442
  point 1) wherever β appears as a sweep dimension; double-oracle
  solver-noise hedge on equilibrium payoffs (issue #442 point 2) in §3
  and §5.
