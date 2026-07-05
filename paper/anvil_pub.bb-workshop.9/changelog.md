# changelog: anvil_pub.bb-workshop.8 → .9

## Trigger

v9 is a **content revision** driven by a single paper-consequential
result that landed after v8 (repo issue #477; PR #475, issue #430
Task 2): the k\* coordination-threshold vs PPO-trainability join
against the full 75-cell downstream k\* artifact
(rjwalters/thrust#269/#290), committed at
`experiments/nash/phase_diagram/kstar_vs_trainability.{py,json,md}`.
The test v8 reported as "blocked on thrust#269" has now run, and its
verdict is negative: **the binary threshold prediction the v8 §4
framed as the pending replacement for the retired entropy predictor is
falsified for PPO on this grid.** Every number added in v9 traces to
the committed artifact (and to the independent re-derivation in the
PR #475 judge review); the full stale-claim reconciliation is
committed alongside at `stale_claims_audit.md`.

## Headline changes

### 1. §4: the coordination-threshold hypothesis, falsified in its binary form

- The v8 paragraph "The coordination-threshold hypothesis (validation
  pending)" is **replaced** by two paragraphs. The first states the
  falsification plainly: on the pre-registered class test (k\*=1,
  3 columns, vs k\*≥2, 8 columns, on pooled `gap_closed_ne`) the
  Mann-Whitney U sits exactly at the null center — rank-biserial 0.00,
  exact one-sided p = 0.539, exhaustive 165-assignment permutation
  Δmean = −0.002 (two-sided p = 0.994); the same split on the
  homogeneous gap (13 columns) is likewise insignificant
  (rank-biserial 0.40, exact p = 0.19). The power floor (minimum
  achievable one-sided p = 0.006) is stated together with the
  observation that the measured effect is zero, not modest-and-missed.
  PPO closes the NE gap at k\* = 2 as well as at k\* = 1; the literal
  "unilateral-BR methods fail whenever k\* > 1" account is dead on
  this grid. This supersedes the underpowered 2-vs-6
  partial-stratification test v8 reported (p = 0.43) and is framed as
  the paper's **second retired predictor** — the negative-results arc
  ("we retire two intuitive predictors and identify the honest
  remaining signal") is now leaned into as a strength.
- The β-dedup methodology is stated: k\* is byte-identical across β
  within every (κ, c) column, so class tests run at column level with
  n-seed-weighted pooling (13 of 15 effective columns joinable; two
  no_convergence columns drop out of the NE-gap outcome, leaving 11).
- A second paragraph, "What survives the falsification, and what it is
  confounded with," carries (a) the **post-hoc failure zone**
  k\* = k_max = 4: the three k\*=4 columns fall below all ten k\*≤2
  columns on the homogeneous gap (rank-biserial −1.00, exact one-sided
  p = 0.0035 = the combinatorial floor at 3 vs 10, permutation
  two-sided p = 0.021) — flagged exploratory (split chosen after the
  primary null) at every mention; (b) the **κ-confounding flag**: k\*
  is a pure function of κ (0.1→4, 0.3–0.7→2, 0.9→1), so every k\*
  result is observationally equivalent to "κ = 0.1 is untrainable" —
  the coalition mechanism is a candidate explanation of the κ effect,
  not independently identified; (c) the **selection-bias caveat**:
  no_convergence cells drop out of the NE-gap join by construction,
  biased against k\* where its prediction is strongest; (d) the
  **k\*-vs-entropy comparison**: Spearman k\* vs homogeneous gap
  ρ = −0.556 (p = 0.0004, n = 37 cells; ρ = −0.635, p = 0.020 at
  column level) is the only significant pair in either predictor
  family, vs the retired entropy predictor's n.s. ρ = 0.109/0.342;
  (e) the **cross-tabulation**: k\* neither reproduces nor refines the
  NE-verdict classes (every k\* level mixes ≥2 verdict classes;
  asymmetric_only cells appear at all three k\* levels); the single
  clean alignment is no_convergence ⊆ k\*=4 — a coarser, κ-driven
  trainability marker, not a sharper equilibrium taxonomy.
- The improvability-oracle "awkward datum" (+13.4/step scripted k=1
  best response on cells where PPO stays flat) is kept and now reads
  as part of the same pattern: neither single-agent improvability nor
  the binary k\* threshold separates trainable from untrainable cells.
- The closing stand-behind sentence now lists three results: the
  entropy retirement (re-verified at n = 20), the β-invariance
  impossibility argument, and the falsification of the binary k\*
  threshold; the k\*=k_max failure zone is offered as an exploratory,
  κ-confounded observation awaiting an out-of-grid test.

### 2. Every other k\* mention resolved against the landed verdict

- **Abstract**: "the replacement coordination-threshold (k\*)
  hypothesis remains unvalidated (its class test is insignificant
  under a partial stratification)" → the hypothesis "is now also
  retired in its binary form" with the committed statistics
  (rank-biserial 0.00, exact p = 0.54, permutation Δmean −0.002),
  plus the post-hoc k\*=k_max observation with both flags (post hoc;
  pure function of κ ⇒ observationally a κ effect).
- **§1 contribution (3c)**: rewritten as "a double negative result …
  we retire two intuitive predictors in sequence and identify the
  honest remaining signal"; all "unvalidated / blocked on a pending
  per-cell k\* artifact" language removed.
- **§6 Threats**: gains the two structural limits of the k\* join —
  within-grid κ-confounding (deconfounding needs cells where k\*
  varies within κ, which the grid does not contain) and the class-test
  power floor, with the explicit note that the primary null is a
  measured zero effect, not a failure to reach significance.
- **Conclusion**: "k\* hypothesis still unvalidated … blocked on a
  pending per-cell k\* artifact" → the double-negative statement
  (binary form falsified; "two intuitive predictors retired in
  sequence is itself the finding"), with the surviving signal carried
  post-hoc + κ-confounded.
- No passage anywhere in v9 presents the binary k\* threshold as live
  or pending.

### 3. Reporting-discipline note (repo issue #476)

The artifact's `class_comparison` display reports the upper-middle
order statistic as "median" for even-sized groups (repo issue #476,
open, display-only). v9 quotes **no group median** from that display:
all statistics quoted are means, rank-biserials, exact Mann-Whitney
p-values, and exhaustive-permutation p-values/Δmeans, none of which is
affected.

## Not changed

Figures 1 and 2 and their source scripts are carried forward unchanged
from v8 (no new figure data; the k\* material is textual). refs.bib
unchanged (new material cites in-repo artifact paths and repo/thrust
issue/PR numbers, per the PAPER_RESULTS.md convention). §2
(environment), §3 (equilibrium structure), §5 (social-trap hardness),
Related-work table, Reproducibility, and Appendices A/B unchanged. All
v8 content — the 16× budget-scaling ladder with its marginality
guards, the rest_trap cycling characterization, the 20-seed noise
buy-down — is preserved.
