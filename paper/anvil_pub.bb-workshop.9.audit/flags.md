# Audit flags for anvil_pub.bb-workshop.9

## Critical flags (block advancement to AUDITED)

- **CF-1 — Numerical disagreement with the committed artifacts as cited
  (artifact supersession, §4 + §6 + Figure 2).** The §4 primary-sweep
  statistics match only **superseded committed revisions** of the two
  artifacts the paper cites by path, not the files at HEAD. Commit
  `a5b8ccdc` (2026-07-04, issue #456 — one day before v9 landed at
  `bb100b81`) regenerated
  `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
  and `experiments/nash/phase_diagram/entropy_vs_trainability.{json,md}`
  with n=20 means substituted on the 8 buy-down cells. Specific mismatches
  a reader hits when opening the cited files at HEAD:
  - Class means quoted in the abstract and §4 (`gap_closed_ne` 0.180 /
    0.107 / 0.059 / −0.049) recompute at HEAD to 0.142 / 0.096 / 0.060 /
    −0.024.
  - The §4-Caveats robustness claim — the homogeneous metric "yields the
    same qualitative ordering (symmetric 0.051 > mixed 0.050 > …)" — had
    a margin of 0.0005 and **inverts at HEAD** (symmetric 0.041 < mixed
    0.046). As written, the "robust to the metric choice" sentence is not
    supported by the currently committed artifact.
  - §4 quotes "Spearman ρ = 0.007 (p = 0.97)" and immediately cites
    `entropy_vs_trainability.{py,json,md}` as the committed source; the
    HEAD file's headline for the same 31-cell computation is ρ = 0.109
    (p = 0.56). Likewise "the single nominally significant entry (spread
    vs homogeneous, p = 0.039)" is, at HEAD, "min vs homogeneous
    (p = 0.038)". The β-column example (−0.32/−0.11/−0.00; "up to 0.36")
    is also the superseded revision's row (HEAD: −0.32/0.01/−0.00; max
    range 0.34).
  - §6 claims the v2 recalibrated artifacts "reproduce byte-for-byte from
    the committed per-seed summaries" and that Figure 2's column "can be
    re-derived" — regeneration at HEAD reproduces the n=20 blend, not the
    n=4 values the text and the rendered Figure 2 carry (Figure 2's
    extracted cell values match revision `73b49b08`).

  The paper's numbers are not *wrong* — they trace exactly to committed
  revisions `73b49b08` (verdict JSON) and `22b1fda6` (entropy artifact),
  and the paper's own buy-down paragraph quotes the n=20 recomputation —
  but the citation-by-path is now ambiguous, one robustness claim fails
  against HEAD, and Figure 2 is not reproducible from its declared source.
  **Reviser options**: (a) pin the git revisions (or commit an n=4-only
  companion artifact) wherever §4/§6 cite the two paths, and qualify the
  homogeneous-ordering sentence (state it holds on the n=4 sweep and that
  the sym/mixed separation, 0.0005, is not resolved at HEAD precision); or
  (b) recompute §4 and re-render Figure 2 against the HEAD blend. Option
  (a) is the minimal, honest fix consistent with the paper's existing
  "conclusions drawn from the n=4 means stand" framing.

- **CF-2 — Internal inconsistency: Table 1 caption contradicts the Table 1
  body, and the unsampled-cell descriptor is wrong (three sites).** The
  caption asserts "the row totals are 9 except at κ=0.7 (6 cells, after the
  high-κ×c=0.5 corner is subsumed by c=1.0)", but the table body's own
  κ=0.3 row also totals 6 (it reads "6 symmetric … 6/6"), and
  `results.json` confirms per-κ totals 9/6/9/6/9. Additionally, the six
  unsampled c=0.5 cells sit at κ∈{0.3, 0.7} — the *middle* of the κ range —
  so the phrase "high-κ × c=0.5 corner" (used in §3 prose, the Table 1
  caption, and the Figure 1 caption) misdescribes them; κ=0.9, c=0.5 IS
  sampled and is in fact the lone β-splitting row the paper analyses.
  Evidence: `experiments/nash/phase_diagram/results.json` grid/cells
  (per-κ verdict counts recomputed: 0.1→9, 0.3→6, 0.5→9, 0.7→6, 0.9→9);
  main.tex lines ~381–383 (§3), ~466–469 (Table 1 caption), ~529–531
  (Fig 1 caption). Carried verbatim since v6; the v6 audit recorded the
  κ∈{0.3,0.7} truth in prose without flagging the caption. Fix: "…except
  at κ∈{0.3, 0.7} (6 cells each…)" and replace "high-κ" with "mid-κ" (or
  name the two columns explicitly).

## Non-critical notes

- **NC-1 — Unverified citations (2)**: `ppo2017`, `mappo2022` have no
  source material in `<thread>/refs/`; claim support could not be verified
  on disk. Both are the canonically correct references for the claims they
  back; author should verify off-disk.
- **NC-2 — Secondary-source-only verification (6 benchmark keys)**: the
  Overcooked / Melting Pot / Hanabi / SMAC+SMACv2 / MAgent / PettingZoo
  claims and Table 3 rows verify point-for-point against the
  author-supplied `refs/benchmark_comparison.md` evidence index, but the
  primary paper PDFs are not on disk — verdicts recorded as `partial`.
- **NC-3 — refs.bib self-declares unverified IDs**: header comment asks
  pub-audit to re-check arXiv IDs and venues. All nine eprint IDs match the
  auditor's knowledge of the canonical papers (no web fetch performed).
  `pettingzoo2021` / `mappo2022` appeared in the NeurIPS Datasets &
  Benchmarks track; entries use the plain NeurIPS proceedings string —
  acceptable, consider naming the track for camera-ready.
- **NC-4 — "4×" denotes different multipliers in §4 vs §5**: §4's
  no-convergence sweep is 200 iter × 4096 steps (8× the 50×2048 base in
  env steps; "4×" counts iterations), §5's ladder 4× is 200 × 2048 (true
  4×). Each matches its committed artifact; a clarifying footnote would
  remove the ambiguity.
- **NC-5 — Provenance of the 16× escape-robustness numbers**: the
  "300/300 bootstrap RNG/resample combinations" and the t-interval
  [304.67, 310.99] trace to the PR #460 judge recomputation as recorded in
  the committed v8 paper trail (`anvil_pub.bb-workshop.8/changelog.md`,
  `.8/stale_claims_audit.md`), not to a regenerable `experiments/`
  artifact. Committed, but weaker than the paper's usual artifact standard.
- **NC-6 — Stale figures**: none by the mtime rule (renders and sources
  share timestamps). Figure 2's reproducibility-from-HEAD issue is CF-1,
  not staleness.
- **Build**: clean. `pdflatex` + `bibtex` + 2×`pdflatex` exit 0; final PDF
  26 pages; zero unresolved citations or cross-references (`??` count 0 in
  the rendered text); remaining warnings are cosmetic (font-shape
  substitutions, `h`→`ht` float moves, over/underfull boxes).

## Verdict

2 critical flags → the thread remains **READY-WITH-AUDIT-FLAGS** (not
AUDITED). Recommend `pub-revise` consuming this sibling; both flags are
narrow, mechanical edits (revision pinning + one caption/descriptor fix)
that do not disturb any headline result.
