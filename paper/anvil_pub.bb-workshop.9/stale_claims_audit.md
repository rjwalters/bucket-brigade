# Stale-claim audit: anvil_pub.bb-workshop.8 → .9 (issue #477)

Scope mandated by issue #477: sweep v8 for every claim the k\*
coordination-threshold vs trainability join (PR #475, issue #430
Task 2) changes, and reconcile each against the committed artifact
(`experiments/nash/phase_diagram/kstar_vs_trainability.{py,json,md}`,
built from the byte-verified thrust#269/#290 75-cell k\* artifact and
`experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`)
and the independent re-derivation in the PR #475 judge review (which
recomputed the β-dedup, the κ-confounding, the exact/permutation
p-values, the power floors, and the pooling arithmetic from the raw
JSON). The audit baseline is v8
(`paper/anvil_pub.bb-workshop.8/main.tex`), the latest revision in the
artifact trail.

Grep sweep of v8 `main.tex` (case-insensitive) for `k^{*}` / `k*`,
`coordination`, `pending`, `unvalidated`, `blocked`, `thrust`,
`hypothesis`, `stratification`, and `predictor` contexts. Every prose
hit is accounted for below; the remaining `coordination` hits (§1
benchmark survey, §7 related work, Appendix A Stag-Hunt discussion)
and the Appendix numeric hit on `181` do not concern k\* and are
untouched.

## Part A — the "validation pending" family: binary k\* threshold falsified

v8 was written when the full k\* class test was blocked on the
downstream per-cell artifact (thrust#269, then open). That artifact
landed (thrust#290) and the join has run. The verdict: the
pre-registered binary threshold test (k\*=1 vs k\*≥2 on pooled
`gap_closed_ne`) finds a genuine zero — rank-biserial 0.00 (U exactly
at the null center), exact one-sided Mann-Whitney p = 0.539,
exhaustive 165-assignment permutation Δmean = −0.002 (two-sided
p = 0.994) — so every v8 "unvalidated / pending / blocked" framing of
the hypothesis is stale, and so is the hypothesis itself in its binary
form. Unlike the v7→v8 corrections (which qualified claims), these
corrections **resolve** a deliberately-suspended claim: the suspense
was the stale part.

| # | Location (v8) | Old text (abridged) | Corrected text (v9, abridged) | Artifact citation |
|---|---|---|---|---|
| A1 | Abstract | "the replacement coordination-threshold (k\*) hypothesis remains unvalidated (its class test is insignificant under a partial stratification)" | "…is now also retired in its binary form: joining a downstream per-cell k\* oracle against the sweep, PPO closes the NE gap on k\*≥2 columns as well as on k\*=1 columns (rank-biserial 0.00, exact Mann-Whitney p = 0.54, permutation Δmean −0.002) — the second retired predictor," + post-hoc k\*=k_max sentence with both flags | `kstar_vs_trainability.md` primary test; PR #475 review (independent re-derivation) |
| A2 | §1 contribution (3c) | "…remains unvalidated (class test insignificant under a partial stratification; the full test is blocked on a pending per-cell k\* artifact)" | Rewritten as a **double** negative result: "we retire two intuitive predictors in sequence and identify the honest remaining signal"; falsification stated with rank-biserial 0.00 / exact p = 0.54; remaining signal flagged post hoc + κ-equivalent | same |
| A3 | §4 paragraph heading + framing | "The coordination-threshold hypothesis (validation pending)." / "We state this as a hypothesis, not a finding." | "The coordination-threshold hypothesis, falsified in its binary form." — the hypothesis is stated in the past tense and the full-grid test result replaces the suspense | same |
| A4 | §4 partial class test | "The 20-seed buy-down above enables a first class-comparison test under a *partial* k\* stratification (2 proven k\*≥2 cells … 6 k\*=1 candidates): not significant (exact MW p = 0.43; permutation p = 0.39) … little power … absence of evidence, not evidence of absence" | Superseded and said so: the full-grid pre-registered test (3 vs 8 columns, all 13 joinable columns stratified) replaces it; the power floor (min achievable one-sided p = 0.006) is carried, together with the observation that the measured effect is exactly zero (U at the null center), i.e. not a low-power miss | `kstar_vs_trainability.md` §"Primary test" + "Relationship to prior tests" (explicit supersession note) |
| A5 | §4 full-test status | "The full test (all 13 effective cells, actual k\* values, the gap_closed_ne column where defined) remains blocked on the per-cell k\* artifact (rjwalters/thrust#269, open)" | The full test has run; §4 now reports it, with the β-dedup/column-pooling methodology (13 of 15 effective columns joinable; two no_convergence columns drop out of the NE-gap outcome, leaving 11) | same; thrust#290 (artifact landed) |
| A6 | §4 stand-behind sentence | "the retirement of the entropy predictor (re-verified at n=20) and the β-invariance impossibility argument are the results we stand behind" | "…the retirement of the entropy predictor (re-verified at n=20), the β-invariance impossibility argument, **and the falsification of the binary k\* threshold** are the results we stand behind — the k\*=k_max failure zone is offered as an exploratory, κ-confounded observation awaiting an out-of-grid test" | same |
| A7 | Conclusion | "with the coordination-threshold k\* hypothesis still unvalidated (its class test is insignificant under a partial, low-power stratification and the full test is blocked on a pending per-cell k\* artifact)" | "the replacement coordination-threshold k\* hypothesis is falsified in its binary form by the full-grid join (k\*=1 and k\*≥2 columns indistinguishable on the pooled NE gap: rank-biserial 0.00, exact p = 0.54). Two intuitive predictors retired in sequence is itself the finding; the honest remaining signal … is post hoc and … observationally equivalent to a κ effect" | same |
| A8 | Header comment block (lines 24–30) | "the k\* class test under the partial thrust#259/#268 stratification is not significant (exact MW p = 0.43) and remains blocked on thrust#269" | Replaced wholesale by the v9 header describing the falsification and the flags | same |

**Net Part A: 8 stale claim sites resolved** (7 prose + 1 header). One
v8 conclusion is **reversed in the only sense available**: v8
committed to no verdict ("unvalidated"), and the verdict came back
negative. No v8 *finding* is reversed — the entropy retirement, the
β-invariance argument, the buy-down precision numbers, and all §3/§5
results are untouched by the join.

## Part B — new material added with the result (flag discipline)

These are additions, not corrections; listed so the flag discipline is
auditable.

| Addition | Where | Flags carried at the site | Traces to |
|---|---|---|---|
| Post-hoc failure zone k\*=k_max=4: three k\*=4 columns below all ten k\*≤2 columns on the homogeneous gap (rank-biserial −1.00, exact one-sided p = 0.0035 = combinatorial floor at 3 vs 10, permutation two-sided p = 0.021) | Abstract, §1 (3c), §4, §6 Threats, Conclusion | "post hoc" / "exploratory" at **every** one of the five mentions; "split chosen after seeing the primary null" stated in §4 | `kstar_vs_trainability.md` §"Post-hoc failure zone" (JSON `registration: post_hoc`) |
| κ-confounding: k\* is a pure function of κ (0.1→4, 0.3–0.7→2, 0.9→1); every k\* result observationally equivalent to "κ=0.1 is untrainable"; coalition mechanism a candidate explanation, not independently identified | Abstract, §1 (3c), §4, §6 Threats, Conclusion | stated at **every** k\*=k_max mention (all five) | `kstar_vs_trainability.md` β-dedup table; PR #475 review (recomputed: pure function of κ, zero β-violations) |
| Selection-bias caveat: no_convergence cells have no NE baseline and drop out of the NE-gap join by construction — biased against k\* where its prediction is strongest | §4 | — (is itself a caveat) | `kstar_vs_trainability.md` §"k\* vs the retired entropy predictor" note |
| k\*-vs-entropy comparison: Spearman k\* vs homogeneous gap ρ = −0.556 (p = 0.0004, n = 37 cells) / ρ = −0.635 (p = 0.020, n = 13 columns) — only significant pair in either family — vs retired entropy's n.s. ρ = 0.109 / 0.342 | §4 | κ-equivalence restated in the same sentence | same, correlation table; entropy constants cross-checked against `entropy_vs_trainability.json` in the PR #475 review |
| Cross-tabulation: k\* neither reproduces nor refines the NE-verdict classes (every k\* level mixes ≥2 verdict classes; asymmetric_only cells appear at all three k\* levels); single clean alignment no_convergence ⊆ k\*=4 | §4 | tied to the post-hoc reading | same, cross-tabulation section (counts re-checked against the committed table: k\*=4 row is 3 asymmetric_only + 6 no_convergence, so the paper does **not** repeat the artifact's looser "every k\* level contains at least two of \[the converged classes\]" phrasing) |
| §6 Threats: the two structural limits of the join (within-grid κ-confounding — deconfounding needs cells where k\* varies within κ, which the grid lacks; power floor 0.006 with "measured zero, not failure-to-reach-significance") | §6 Threats | — | same |

## Part C — reporting-discipline checks

- **Issue #476 (display medians)**: the artifact's `class_comparison`
  display reports the upper-middle order statistic as "median" for
  even-sized groups (e.g. primary k\*≥2 group displayed 0.111 vs
  conventional 0.089). v9 quotes **no group median from that display**
  — verified by grep of the v9 diff: every quoted statistic is a mean,
  Δmean, rank-biserial, exact p, permutation p, ρ, or power floor,
  none of which issue #476 affects.
- **Exploratory/post-hoc flag coverage**: grep of v9 `main.tex` for
  `k_{\max}` returns 5 prose sites (abstract, §1, §4, §6, conclusion);
  each carries "post hoc"/"exploratory" and the κ-confounding
  equivalence in the same sentence or clause.
- **No live/pending binary threshold**: grep of v9 for `pending`,
  `unvalidated`, `blocked` returns no k\*-related prose hit (remaining
  hits are the v9 header's description of what changed).

## Part D — pointer updates outside the paper

| Change | Where | Traces to |
|---|---|---|
| Published-paper and Figure-2-source pointers moved `.8` → `.9` | `docs/PAPER_RESULTS.md` (4 sites) | trail convention (same move as PR #463's `.7` → `.8`) |
| §6f "the coordination-threshold account (k\* > 1 …) and plain exploration failure remain the live hypotheses" | `docs/PAPER_RESULTS.md` §6f | stale for the same reason as Part A; updated to record the PR #475 falsification and the surviving exploratory reading |

## Summary

- **Stale claim sites resolved: 8** (Part A; all in the single
  "k\* validation pending" family v8 carried across abstract, §1, §4
  ×4, conclusion, header). One suspended claim resolved negative; no
  v8 finding reversed.
- **Fresh claims added, each with a committed artifact path and an
  independent verification trail** (PR #475 judge review): the primary
  falsification statistics, the post-hoc failure zone, the
  κ-confounding equivalence, the selection-bias caveat, the
  k\*-vs-entropy comparison, and the cross-tabulation (Part B).
- **Flag discipline verified** (Part C): post-hoc/exploratory +
  κ-confounded at every k\*=k_max mention; no #476-affected median
  quoted; no passage presents the binary threshold as live or pending.
- Figures, refs.bib, §2, §3, §5, Related work, Reproducibility, and
  Appendices A/B: no claims affected by the join; carried forward
  unchanged.
