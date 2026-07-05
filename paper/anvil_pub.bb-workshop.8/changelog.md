# changelog: anvil_pub.bb-workshop.7 → .8

## Trigger

v8 is a **content revision** driven by the three headline results that
landed after v7 (repo issue #462; PRs #455, #456, #460; issues #443,
#444, #445), not a critic-loop revision. v7 (PR #448) integrated results
through 2026-07-02; the three post-v7 results were committed only in
`docs/PAPER_RESULTS.md` §6–9 and the experiment artifacts. Every number
added in v8 traces to a committed artifact path (and, where noted, to
the independent verification in the judge review comments on PRs
#455/#456/#460); the full stale-claim reconciliation is committed
alongside at `stale_claims_audit.md`.

## Headline changes

### 1. §5: budget-scaling ladder — marginal statistical trap escape at 16× (PR #460, issue #444)

- **Anchor table (Table 4) gains the 16× row**: `het_ppo` at 16× budget,
  trailing-5 mean 307.83, seed-bootstrap 95% CI [305.00, 310.71] —
  the lone trained configuration whose CI lower bound clears the random
  anchor's measured n=10k upper bound (304.31, by +0.69/step). The
  standard-budget row is relabelled accordingly.
- **New "Budget scaling" paragraph** reporting the full pre-registered
  ladder (issue #444; 4×/16× × het_ppo/ippo × 20 seeds,
  `experiments/p3_specialization/tier1_runs_trap_escape/`): three of
  four cells stay `at_random` (lower bounds 304.03 / 301.77 / 303.46);
  16× het_ppo is the first `escaped_trap` verdict on rest_trap. The
  crossing's robustness is cited from the PR #460 review's independent
  recomputation: clears the anchor in 300/300 bootstrap RNG/resample
  configurations, and the non-bootstrap t-interval on the 20 seed means
  is [304.67, 310.99].
- **Marginality language is load-bearing and carried from
  PAPER_RESULTS §9**: no dose-response on the mean (306.26 → 307.66 →
  307.83; all paired same-seed contrasts n.s., e.g. 16×−1× =
  +1.57 ± 4.80/step, t = 1.46); the flip is variance-driven (uplift std
  8.36 → 6.59, CI lower bound 302.95 → 304.03 → 305.00 around a flat
  plateau); mean uplift ≈6% of the 83.7/step scripted headroom (best
  seed ≈27%, still ≈61/step below `scripted_best`); CRN-coupled seed
  streams + one-of-four-tests multiplicity caveat stated; the row is
  scored as "marginally but significantly above random," explicitly
  **never** as "PPO solves the trap at scale."
- Abstract, §1 contribution (4), §5 headline italics, the closing
  benchmark-statement ladder (now ≤248.67 < 302.87 < 307.83 < 386.60,
  with `escaped_trap` reached exactly once, marginally), and the
  Conclusion all updated to the precise post-#444 statement: "budget
  buys precision, not performance"; the hardness headline is sharpened,
  not weakened.
- The v7 "no positive interventional evidence for asymmetry rescue"
  claim is kept but now cross-references the ladder: het_ppo crosses
  where ippo at identical budget does not, but the two trainers' means
  are statistically indistinguishable, so this is not counted as
  positive evidence either.

### 2. §5: rest_trap symmetric-DO cycling confirmed and characterized (PR #455, issue #445)

- **New paragraph "Symmetric-oracle cycling is a property of the
  scenario, not a hole in the characterization"**: the battery-seeded
  double-oracle retry ran 50/50 iterations without converging (min
  improvement 11.79 vs ε = 0.01; payoff band [1611.56, 2316.61], same
  basin as the unseeded run — cycling is not a seeding-coverage
  artifact); the final mixed profile is exploitable by ≥407.50/episode
  (best deviation `always_rest`); the genome-mapped specialist team is
  exploitable at every position (+605 to +870/episode); the cycling
  mechanism is solver noise (support strategies score −258 to
  +237/episode against the mixture at the verification budget). Source:
  `experiments/nash/rest_trap_seeded_do/RESULTS.md` + committed trace
  and exploitability JSONs.
- **The canonical asymmetric FR×3+FF NE (2984.04/episode) stands** as
  rest_trap's characterization; symmetric non-convergence is documented
  as a property of the game (no symmetric equilibrium at the measured
  bounds — role differentiation required). **12/12 frozen-scenario
  equilibrium coverage** is now claimed (11 converged + 1
  exploitability-bounded cycling entry).
- The v7 §5 hedge "a better equilibrium may exist" is resolved: the
  solver-noise error bar on the NE payoff value stays, but the sentence
  now records that the seeded retry searched for a better equilibrium
  and found nothing that stands. Abstract and Conclusion carry the
  closed-characterization claim.

### 3. §4: 20-seed noise buy-down (PR #456, issue #443)

- **New paragraph "The noise ceiling, bought down at 20 seeds"**
  directly under the v7 noise-ceiling threat: 95% CI half-widths on
  `gap_closed_ne` shrink 2.00–3.64× (median 2.65; homogeneous column
  1.84–3.59×), consistent with √5 ≈ 2.24; **0/8 trainability verdicts
  flip** (all stay `insufficient`); the entropy retirement is
  re-verified at n=20 (Spearman ρ = 0.109, p = 0.56, all four
  aggregates insignificant); the n=4 point estimates were
  noise-dominated but directionally right (largest shift 0.372 → 0.043
  at κ=0.3, c=1.0). Source:
  `experiments/nash/phase_diagram/noise_buydown_precision.md`,
  `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`.
- **k\* hypothesis paragraph updated**: the buy-down enables a first
  class test under the partial thrust#259/#268 stratification (2
  proven k\*≥2 cells vs 6 k\*=1 candidates) — not significant (exact
  one-sided Mann–Whitney p = 0.43; exhaustive permutation p = 0.39),
  with the power caveat (min achievable p ≈ 0.036) stated; the full
  test remains blocked on thrust#269 (open). "Validation explicitly
  pending" wording replaced by the precise post-#443 status throughout
  (abstract, §1 contribution (3c), §4, Conclusion).
- §6 Threats notes the buy-down covers 8 of 37 cells (verdicts
  unchanged) without yet covering the rest of the grid.

### 4. Open threads cited (issue #462 references)

- §3 and §6 Threats: repo issue #459 — whether the phase-diagram
  asymmetric profiles are themselves ε-NE under CRN re-evaluation
  (deferred from #445/#450).
- §6 Reproducibility: repo issue #461 — reproduction drift in the
  *superseded* v1 recalibrated tables; the v2 artifacts this paper
  reports reproduce byte-for-byte.

## Not changed

Figures 1 and 2 and their source scripts are carried forward unchanged
from v7 (no new figure data; the anchor-table and ladder numbers are
textual). refs.bib unchanged (new material cites in-repo artifact paths
and repo issue/PR numbers, per the PAPER_RESULTS.md convention). §2
(environment), Related-work table, and Appendices A/B unchanged. All v7
critic conclusions and the v7 β-inertness corrections are preserved.
