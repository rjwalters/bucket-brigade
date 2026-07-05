# Audit flags for anvil_pub.bb-workshop.11

## Critical flags (block advancement to AUDITED)

**None.** 0 critical flags.

- All 18 `\cite` keys resolve; 0 orphan refs.bib entries.
- 0 claim-support failures (18/18 partial — no primary PDFs on disk, but
  on-disk secondary/resolver evidence covers every key; see
  citation-audit.md).
- 0 numerical inconsistencies; the changelog's no-numbers-moved
  attestation is diff-verified TRUE, and every headline statistic
  re-verifies against the committed artifacts at repo HEAD `73911a75`
  (see numerical-audit.md).
- Build: `pdflatex → bibtex → pdflatex ×2` all exit 0; 0 undefined
  citations/references, 0 `??` in `pdftotext` of the rendered PDF,
  0 overfull hboxes, 0 bibtex warnings, 0 duplicate-destination
  warnings (the v10 nit class is confirmed eliminated by
  `hypertexnames=false`); 28 pages. See compile-log.txt.
- The v10 audit's NC-3 (§4 skipped-cell wording) is verified FIXED and
  the replacement wording is exactly artifact-true: the Nash artifact's
  (β=0.1, κ=0.9, c=0.5) cell carries the splitting row's `mixed`
  verdict and is absent from the PPO artifact, whose (κ=0.9, c=0.5)
  row holds two `asymmetric_only` samples (β=0.5, 0.9).
- The v10 review's single major (Figure 1 caption transposition) is
  verified FIXED against `figures/src/phase_diagram.py` (rows = β,
  columns = κ).

## Non-critical notes

1. **Partial citations (18)** — carried v10 NC-2, narrowed: no primary
   PDF of any cited paper is in `<thread>/refs/`. 7 benchmark keys are
   verified only against the secondary
   `refs/benchmark_comparison.md` evidence index; the other 11 keys
   (4 confirmed + 7 newly merged) carry on-disk resolver-verified
   metadata provenance (`anvil_pub.bb-workshop.10.litsearch/`), which
   verifies bibliographic identity but not full-text claim support.
   Primary-PDF acquisition remains an operator task before camera-ready.
2. **`shapley1953stochastic` attribution precision** (new, minor):
   App. A.6 reads "Every formal result for finite stochastic
   games~\citep{shapley1953stochastic}---minimax, equilibrium existence
   in stationary mixed strategies, value iteration
   convergence---applies directly." Shapley (1953) supplies the class
   definition, the (zero-sum) minimax value, and the iterative
   solution; *N-player* equilibrium existence in stationary strategies
   is Fink (1964)/Takahashi (1964). The cite sits in definitional
   position so this is not a claim-support failure, but a camera-ready
   polish could add a Fink citation or scope the list. Evidence:
   main.tex lines 1691–1694.
3. **Tradition cited via modern representative** (documented decision,
   not a defect): §6 credits "the oldest form of NE-ground-truth MARL
   evaluation ... the climbing/penalty-game tradition" to
   `christianos2022pareto` (2022) because the tradition's primary
   (Claus & Boutilier 1998) has no resolvable identifier and the
   litsearch write contract forbids hand-inventing the entry
   (changelog.md, litsearch cluster 2/gap rows). If the author
   hand-enters a verified Claus & Boutilier BibTeX before camera-ready,
   the §6 clause should absorb it.
4. **§5 footnote provenance-standard exception** — carried v10 NC-4,
   disclosed in-paper: the 300/300 bootstrap RNG/resample sweep and the
   t-interval [304.67, 310.99] trace to the PR #460 judge recomputation
   via committed paper-trail files
   (`paper/anvil_pub.bb-workshop.8/{changelog.md,stale_claims_audit.md}`),
   not a regenerable `experiments/` artifact. Regeneration is a
   camera-ready operator task.
5. **HuggingFace baseline hosting "in progress"** (§7) — carried:
   complete or remove the hosting promise before submission; the
   in-paper status statement remains true as written.
6. **Figure mtime micro-skew** (informational, NOT stale): both
   `figures/src/*.py` scripts are 0.1–0.6 ms newer than their rendered
   PDFs — a `cp -p` copy-ordering artifact of the v10→v11 carryover.
   Figures and scripts are byte-identical to v10 (v10 audit: no stale
   figures). No re-render needed; counted as 0 stale figures in the
   summary.
7. **Cosmetic build warnings** (pre-existing class behavior): two
   `h`→`ht` float-specifier moves and the `OT1/cmr/bx/sc`
   font-substitution notice. Present in the v10 log as well; no action.

## State machine consequence

Zero critical flags AND the paper's v10 review is `advance: true`
(38/44) with v11 a numbers-invariant polish pass consuming all four v10
critic siblings: the thread reaches **AUDITED** at version 11.
