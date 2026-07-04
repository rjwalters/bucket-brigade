# Line-level comments — anvil_pub.bb-workshop.6

Comments grouped by section heading and severity. Severities:
**blocker** (must fix before advance), **major** (should fix before
camera-ready), **minor** (worth a look), **nit** (cosmetic).

## Blockers

**None.**

## Majors

**None.** All five v5/v5-audit punch-list items are closed.

## Minors

### Appendix A §A.6 (Relationship to canonical game-theory templates) — informal attributions without `\cite{}`

Carry-forward from v5. The named-template attributions to Volunteer's
Dilemma, $N$-player Public Goods, Stag Hunt, free-rider problem, and
finite stochastic game (L972–1018) ship without `\cite{}` links. The
v5 changelog flagged this as intentional to match Appendix A's
informal-attribution style, but for the camera-ready a workshop
reviewer may expect a `\cite{}` per named template. Not enough to
deduct dim 8 (the structural side is clean) but worth a pass before
submission.

### Appendix B §B.4 / §B.5 boundary — layout accident

Carry-forward from v5. §B.4 "Predicted vs.\ empirical phase table
(7-cell preview)" contains one lead-in sentence (L1287–1288) and
then immediately the table float (Table 5, L1290–1318) before §B.5
begins. The boundary reads as a layout accident rather than a
structural choice. Fix: either fold Table 5 into §B.4 proper with
explicit framing (one paragraph after the lead-in sentence
explaining what's in the table), or merge §B.4 into §B.5's
"Where the reduction breaks" with the 7-cell preview as a worked
example of the gap.

### §B.2 — small-$q(k)$ Taylor expansion not named as a bias source

Carry-forward from v5. The derivation of the survival coefficient $A$
in §B.2 (L1154–1170) uses $(1{-}q(k))^{T_{\min}}\approx 1{-}T_{\min}q(k)$,
which is a small-$q(k)$ linearisation distinct from the five bias
sources §B.5 enumerates. A careful reader will notice the linearisation
step and ask whether it contributes to the 3–10× $\kappa$-threshold
gap. Naming it in §B.5 as a sixth bias (or footnoting at the Taylor
step that it is the source of one such bias) would close the loop.
Not load-bearing for the qualitative claim.

### §6 Reproducibility — HuggingFace baselines pathway "in progress"

The "publication pathway for the baselines is in progress and not yet
complete at the time of this draft" language (L685–687) is honest but
will need a status update for the camera-ready. Tracked; not a v6
issue.

## Nits

### Figure 1 caption — explicit panel layout

The new Figure 1 caption (L399–417) says "across $(\beta,\kappa,c)$"
and "Four regimes appear across the grid" but does not explicitly
say "three panels, one per $c\in\{0.5,1.0,2.0\}$" the way Figure 2's
caption does. A reader scanning only the caption may not realise the
$c$-axis is the panel split. One half-sentence addition ("Panels by
cost $c\in\{0.5,1.0,2.0\}$.") would parallel Figure 2.

### Figure 1 axis label — "extinguish prob" vs. "$\kappa$ (extinguish prob)"

In the rendered figure, the $\kappa$-axis label reads `κ (extinguish
prob)` and the $\beta$-axis label reads `β (spread prob)` — consistent
with Figure 2. No change needed; flagging only as confirmation.

### §3 L289–292 — "lone splitting row" hyphenation

Minor: "a small but real anomaly we return to in §6" uses "small but
real" without a hyphen; "small-but-real" would be more consistent
with the paper's compound-modifier hyphenation elsewhere
("load-bearing", "ordering-not-significance"). Style nit only.

### Table 1 caption (L339–351) — readability

The caption is dense (12 lines of mostly-prose). The headline finding
("The reduction predicts the modal empirical class on $3/5$ $\kappa$
rows") is buried mid-caption. Promoting it to the first sentence
would let a reader extract the load-bearing fact without reading the
full caption. Not blocking.

## Spot-check confirmations (no action needed)

- **Figure 1 PDF inspection**: regenerated as 3-panel categorical
  heatmap, all four regimes visible, n/a cells hatched grey with
  "n/a" labels, no inline numerics. Matches caption.
- **Table 2 PDF inspection (page 7 of rendered v6)**: "Coop/comp"
  column header and all entries fit within the body width. No
  overfull \hbox warning expected (compile log not re-checked at
  reviewer level; audit will confirm).
- **§3 L393–394 cross-reference**: now reads "The full bias
  accounting and the 7-cell preview predicted-vs-empirical table
  appear in Appendix B." Matches Appendix B §B.4 Table 5 + §B.5
  delivery.
- **§4 L476 estimator naming**: "per-class mean within-cell std
  (≈ 0.21–0.44)" names the estimator inline.
- **§4 L477–481 separation arithmetic**: 0.001 = 0.051 − 0.050,
  0.017 = 0.050 − 0.033, 0.082 = 0.033 − (−0.049). All three on
  `gap_closed_homogeneous`. Metric-consistency clause is in place.
- **`figures/src/phase_diagram.py`**: source script ships
  alongside the generated PDF, matching the
  `figures/src/recalibrated_heatmap.py` pattern for Figure 2.
- **Page count**: 19 pages (v5 was 20pp per audit; v6 changelog
  says "expected 20 pages"). One-page tightening is incidental,
  not a regression — the body changes are within-paragraph or
  cell-text only.
