# changelog: anvil_pub.bb-workshop.4 → .5

## Trigger

v5 is a **content + polish revision** driven by two threads:

1. **User-driven content change.** The two `@misc` placeholder citations
   `bbenvspec` and `bbnestructure` pointed at in-repo memos
   (`paper/anvil_memo.env_spec.1/`, `paper/anvil_memo.ne_structure.1/`)
   that were never going to make it onto arXiv before submission. The user
   asked for the memo content to be **inlined into the paper** rather than
   carried as forward-referenced citations. v5 lands the memo material as
   two new appendices and removes the placeholder bib entries.
2. **Carryover polish from the v2 reviewer.** The v2 review surfaced
   several `major`-tagged findings that survived through v3 and v4
   unaddressed (notably the buried per-cell-baseline contribution and
   the over-claimed "confirming" language in §4). v5 lands the
   straightforward polish items now that the appendix work touched
   adjacent prose anyway.

The v4 audit (`anvil_pub.bb-workshop.4.audit/`) issued an `advance`
verdict with no critical flags; v5 is **not** required by that audit.
v5 is a polish-and-self-containment revision. Carrying forward all v4
audit conclusions; adding the v5-specific changes below.

## Headline changes

### 1. Two new appendices inline the memo content

- **Appendix A — Formal environment specification.** Direct LaTeX port
  of `paper/anvil_memo.env_spec.1/env_spec.md` (247 markdown lines →
  one LaTeX appendix with §A.1 Players and parameters, §A.2 State /
  action / observation, §A.3 Transition dynamics, §A.4 Reward
  structure, §A.5 Horizon / information / symmetries / variants, §A.6
  Relationship to canonical templates, §A.7 Notation summary).
- **Appendix B — Analytical NE characterisation for the 4-agent game.**
  Direct LaTeX port of `paper/anvil_memo.ne_structure.1/ne_structure.md`
  (294 markdown lines → §B.1 Setup, §B.2 Mean-field stage-game
  reduction, §B.3 The three NE candidates, §B.4 Predicted vs.
  empirical phase table (7-cell preview), §B.5 Where the reduction
  breaks, §B.6 Implications and recommended follow-ups).

Body §2 and §3 are unchanged in content; the five citation sites that
pointed at the memos (L180, L318, L330, L359, L377 in v4) now point at
the appendices (`\ref{app:envspec}`, `\ref{app:nestructure}`).

### 2. `refs.bib`: deleted the two memo placeholder entries

`@misc{bbenvspec}` and `@misc{bbnestructure}` deleted. The pre-amble
comment block explaining the `refs/<file>.md` placeholder convention
deleted alongside (no longer applicable). One stray
`\citep{diekmann1985}` introduced during the appendix draft was
removed; the Volunteer's-Dilemma paragraph now opens with the named
attribution alone, which is consistent with the rest of Appendix A's
informal-attribution style.

### 3. §1 Contributions split

v4 (3) collapsed three findings into one bullet. The v2 reviewer
flagged this as the paper's most novel finding (the per-cell baseline
correction) being buried in a sub-clause. v5 splits (3) into:

- **(3a)** Protocol + the first empirical results + 4× falsification.
- **(3b)** The per-cell NE-anchored metric as a standalone
  methodological contribution, with the single-cell-baseline-inverts-
  the-ordering finding stated up front.

Also softened the (3a) claim from "matches the analytical prediction"
to "is consistent with the analytical prediction," paired with the §4
calibration of language below.

### 4. §3 inline Table 1: per-κ predicted-vs-empirical comparison

A 5-row table inserted between the "Predicted vs. observed thresholds"
paragraph and the "We own this gap" paragraph. Columns:
$\kappa$ | Predicted class | Empirical distribution | Predicted-class share.

Source: counts derived from
`experiments/nash/phase_diagram/results.json`. The 5 rows are at
$\kappa\in\{0.1, 0.3, 0.5, 0.7, 0.9\}$; row totals are 9 except at
$\kappa=0.7$ (6 cells, the high-$\kappa\times c=0.5$ corner being
subsumed). The reduction predicts the modal empirical class on 3/5
$\kappa$ rows; the two off-rows ($\kappa\in\{0.1, 0.7\}$) are the
collapse-threshold and asymmetric-threshold disagreements documented
in the prose.

v2 reviewer requested at least a small inline truth table in §3 so
readers see the predicted-vs-empirical structure without flipping to
the appendix. v5 lands this at the full 39-cell scale (an upgrade on
the v2-era 7-cell preview the reviewer originally asked for).

### 5. §4 Calibrated statistical language

- "in exact agreement with the analytical prediction" →
  "consistent with the analytical prediction".
- Added one new sentence to the Results paragraph:

  > "We report the ordering, not statistical significance: per-class
  > std ($\approx 0.21$-$0.44$) is larger than the class-mean
  > separations ($0.073, 0.048, 0.108$), so the data is consistent
  > with the predicted ordering but does not reject the null of equal
  > class means under a standard test. A cross-class 4×-budget sweep
  > is in scope for the camera-ready and is the natural significance
  > gate."

- The Caveats paragraph's redundant sample-size disclosure
  ($n=11, 9, 11, 6$) compressed because the new sentence above already
  carries the ordering-not-significance framing.

Source numbers all verified at v5 cut:

- Class means (gap_closed_ne for sym/mix/asy; gap_closed_homogeneous
  for collapse): 0.1799 / 0.107 / 0.0595 / -0.0485 → matches paper
  prose at 0.180 / 0.107 / 0.059 / -0.049.
- Separations: 0.073 / 0.048 / 0.108 verified.
- Per-class std range 0.21-0.44 carried forward from v4 (already audited).

### 6. §6 Threats merge

v4's two paragraphs `Threats to the analytical contribution` and
`Threats to the empirical contribution` were merged into one
`Threats to the contributions` paragraph with analytical-side and
empirical-side sub-clauses. The per-cell-baseline methodological
finding is now explicitly load-bearing-independent of both unresolved
questions (ring-Markov refinement on the analytical side; cross-class
4×-budget on the empirical side).

Saves ~80 words and removes the v2 reviewer's flagged redundancy
(both v4 paragraphs ended with the same "in scope for the
camera-ready" gate).

### 7. Figure 1 / Figure 2 caption polish

- **Figure 1**: v4's 125-word dense caption split into two sentences
  (the parameter-cell / DO-solver framing now a stand-alone opening
  sentence; the four-regime description and β-independence note kept
  as a second sentence).
- **Figure 2**: added one sentence explaining why
  `gap_closed_homogeneous` is the natural fallback metric on
  no-pure-NE cells (the all-Hero SpecialistPolicy is the strongest
  available stationary policy when no NE policy profile exists, even
  though it is not itself a NE). The closing "matches the analytical
  prediction" softened to "is consistent with the analytical
  prediction" to match the §4 calibration.

### 8. Conclusion: per-cell-baseline finding promoted

v4 Conclusion folded the methodological finding into one half-sentence.
v5 promotes to sentence-and-a-half, naming the
single-cell-baseline-inverts-the-ordering finding directly:

> "The per-cell NE-anchored metric is itself a standalone
> methodological finding: on this paper's data the previously-used
> single-cell baseline inverted the per-class ordering, so per-cell
> calibration is necessary, not cosmetic, for any cross-cell PPO
> comparison in which the random-policy return varies across cells."

## What is unchanged

- All §2 content (the body env spec stays compressed; Appendix A is
  the formal expansion).
- The §3 closed-form derivation (eq. S/A/C, the "Predicted thresholds"
  paragraph, the "We own this gap" three-bias paragraph): unchanged.
- The §3 "12 of 13 (κ,c) rows" β-independence count (v4 audit's M3 was
  already correctly resolved in v4 prose; v5 carries it forward
  unchanged).
- The §4 4×-budget paragraph: unchanged.
- The §4 methodological-observation paragraph (the v1-baseline
  inversion story): unchanged content; only the Conclusion's
  reference to it strengthened.
- The §5 Related work + Table 2 benchmark comparison: unchanged.
- The §6 "Bucket Brigade is intentionally small and artificial"
  (a)/(b)/(c) paragraph: unchanged.
- The §6 Reproducibility paragraph: unchanged.

## Page count

v4 PDF: 11 pages. v5 PDF: 20 pages (the appendix inlining is the
dominant growth driver, ~7-8 pages; the other v5 polish items are
roughly net-zero on page count).

The user opted out of workshop page-budget enforcement at v4 audit
time; v5 inherits that opt-out. The body alone (through Conclusion,
before References) is **8 pages**, so a strict-budget submission
target could promote the appendices to a supplementary-materials file
without disturbing the body.

## Soft spots carried forward into v5

- The closed-form κ-thresholds remain off by 3-10× on the 39-cell grid
  (the v3/v4 claim survives). §3 owns this; v5 does not change the
  bias accounting.
- The closed-form bound does not predict the empirical `mixed` class.
  §3 and the new Appendix B §B.5 both surface this honestly.
- Cross-class 4×-budget sweep (the natural significance gate, named in
  the new §4 sentence) is **not yet run** at v5 cut.
- HuggingFace baselines pathway: still "in progress" at v5 cut
  (unchanged from v4).
- The 2 cells skipped in the PPO sweep
  (`b=0.10, k=0.50, c=0.50` and `b=0.10, k=0.90, c=0.50`) remain
  disclosed in §4 protocol; the figure marks them n/a.
