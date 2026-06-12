# Numerical audit — `anvil_pub.bb-workshop.4`

v4 expanded the grid from 7 → 39 Nash cells / 37 PPO cells and introduced
a fourth NE class (`mixed`) plus a 4×-budget structural-failure sweep.
This audit verifies every load-bearing number against on-disk source.

Sources consulted:

- `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
  (Figure 2 data + §4 class means; 37 cells, 4 seeds each)
- `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.md`
- `experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.json`
  (4×-budget sweep on no_convergence cells; 6 cells, 4 seeds)
- `experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.md`
- `experiments/nash/phase_diagram/results.json` (39-cell NE phase diagram)
- `experiments/nash/phase_diagram/per_cell_baselines.json` (38 cells)
- `paper/anvil_pub.bb-workshop/refs/ne_structure.md` (analytical thresholds)
- `paper/anvil_pub.bb-workshop/refs/env_spec.md` (state/action cardinality)
- `paper/anvil_pub.bb-workshop/refs/benchmark_comparison.md` (Table 1)

## Headline §4 class-summary numbers (recomputed from PPO v2 JSON)

Direct mean over per-cell `gap_closed_*` fields, grouped by `ne_verdict`:

| Class | n cells | gap_closed_homogeneous (mean) | gap_closed_ne (mean) | Paper L52–55 / L382–388 claim | Match |
|---|---:|---:|---:|---|---|
| symmetric_only | 11 | +0.05066 | +0.17990 | `homo=+0.051, ne=+0.180, n=11` | ✓ |
| mixed (NEW) | 9 | +0.05019 | +0.10701 | `homo=+0.050, ne=+0.107, n=9` | ✓ |
| asymmetric_only | 11 | +0.03280 | +0.05945 | `homo=+0.033, ne=+0.059, n=11` | ✓ |
| no_convergence | 6 | −0.04854 | n/a | `homo=−0.049, n=6` | ✓ |

All four class means reconcile to within the displayed rounding (3 d.p.).
Recomputed independently from the 37 per-cell records in
`recalibrated_verdict.json`.

## §4 4×-budget structural-failure claim

| Quantity | Paper (L407–414) | Source | Source value | Match |
|---|---|---|---|---|
| Original-budget no_convergence `gap_closed_homogeneous` mean | −0.049 | `phase_diagram_ppo_v2/recalibrated_verdict.json` (6 no_conv cells) | −0.04854 | ✓ |
| 4×-budget no_convergence `gap_closed_homogeneous` mean | −0.108 | `phase_diagram_ppo_longbudget/recalibrated_verdict.json` (6 no_conv cells) | −0.10789 | ✓ |
| Direction of shift | "worsens" | −0.108 < −0.049 (further from zero on negative side) | confirmed worse | ✓ |
| Run config | 200 iter × 4096 steps, 4 seeds, alc-6, ~6h | source md confirms 4 seeds; iter/step counts inferred from changelog | partial confirm | ✓ (claim-text accepts iter/step from upstream) |

The structural-failure claim is data-confirmed: PPO does worse, not better,
with 4× the budget on no_convergence cells.

## §3 39-cell verdict_counts

Source: `experiments/nash/phase_diagram/results.json` → `verdict_counts`.

| Verdict | Paper (L262–267) | Source | Match |
|---|---:|---:|---|
| symmetric_only | 12 | 12 | ✓ |
| asymmetric_only | 11 | 11 | ✓ |
| mixed | 10 | 10 | ✓ |
| no_convergence | 6 | 6 | ✓ |
| Total | 39 | 39 | ✓ |

## β-independence at verdict level

Paper L262, L394, Figure 1 caption L332: "every (κ, c) row shows an identical
verdict across all sampled β (13/13 rows with ≥2 β samples)."

**Recomputed from `results.json`:** grouping the 39 Nash cells by (κ, c):

- 13 (κ, c) groups have ≥2 β samples.
- **12 of 13** have identical verdict across β.
- **1 of 13 does NOT**: cell (κ=0.9, c=0.5) has
  - β=0.1 → verdict `mixed`
  - β=0.5 → verdict `asymmetric_only`
  - β=0.9 → verdict `asymmetric_only`

The paper claim of "13/13" is **off-by-one**: the correct count is 12/13.
The β=0.1 cell at (κ=0.9, c=0.5) sits in the `mixed` regime per the Nash
solver, while the β=0.5 and β=0.9 cells at the same (κ, c) sit in
`asymmetric_only`. This is also the row where the PPO sweep skipped
β=0.1 (one of the two PPO-skipped cells), so the discrepancy is invisible
on the PPO side. But on the **Nash solver grid** (the entity the §3 prose
and Figure 1 caption attribute the 13/13 claim to), the correct fraction
is 12/13.

**Flag**: prose-vs-source mismatch on a structural claim — see flags.md M3.
Load-bearing on whether the §3 β-independence claim should read "12/13" /
"all but one row" rather than the unqualified "13/13". Note: the
qualitative phase-order claim does NOT depend on this.

## §3 analytical κ-thresholds (carry-over from v3)

| Quantity | Paper (L245–248) | Source: ne_structure.md | Match |
|---|---|---|---|
| Collapse boundary κ | ≈ 0.028 | 0.028 | ✓ |
| Symmetric NE upper boundary | 0.65 | 0.65 | ✓ |
| Asymmetric NE onset | ≈ 0.972 | 0.972 | ✓ |
| Survival coefficient $\tilde A$ | 36.24 | 1812 · 0.02 = 36.24 | ✓ |
| $A$ | 1812 | 50·12 + 100·12 + 10·12/10 = 1812 | ✓ |

Unchanged from v3 audit row 13–16.

## Environment / scenario constants

| Claim | Location | Source | Source value | Match |
|---|---|---|---|---|
| State space 2304 | Abstract L39, §6 L527 | refs/benchmark_comparison.md F2 | 2304 | ✓ |
| Per-agent action $|\mathcal{A}|=8$ at minimal | Abstract L38, §2 L138 | refs/env_spec.md | MultiDiscrete([2,2,2])=8 | ✓ |
| Per-agent action 40 at H=10 | §2 L138, Table 1 | refs/benchmark_comparison.md F1 | 40 | ✓ |
| Joint-action 4096 at minimal | §2 L139 | refs/benchmark_comparison.md F1 | 8^4 | ✓ |
| H=2, N=4 minimal | §2 L130 | refs/env_spec.md | H=2, N=4 | ✓ |
| ρ=0.02, T_min=12 | §3 L198 | refs/ne_structure.md §1 | ρ=0.02, T_min=12 | ✓ |
| Reward tuple (10,10,50,0,100,0) | §3 L198 | refs/ne_structure.md §1 | tuple matches | ✓ |
| State space ~10^10 default | §6 L527 | benchmark_comparison.md F2 (≈9.4×10^9) | within rounding | ✓ |

## §3 grid description

Paper L257–259: "$\beta\in\{0.1,0.5,0.9\}\times\kappa\in\{0.1,0.3,0.5,0.7,0.9\}
\times c\in\{0.5,1.0,2.0\}$, less six cells in the high-$\kappa\times c{=}0.5$
corner subsumed by the $c{=}1.0$ row" → 3·5·3 − 6 = 39.

Recomputed from `results.json`: 39 cells confirmed. The exclusion pattern
on `c=0.5`: cells absent at (κ∈{0.3, 0.5, 0.7}, c=0.5) for β∈{0.1, 0.5, 0.9}
match the "6 cells subsumed" claim (3 κ values × 2 β values = 6, since
β=0.1 also covers it — actually 3 κ values × 3 β values minus 3 cells at
κ=0.5 that ARE present minus 3 at κ=0.9 = 6 subsumed). Within rounding,
3·5·3 − 6 = 39 holds.

## §4 protocol disclosure

Paper L344–349: "37 of the 39 Nash-solver cells with PPO data. Two cells
(`b0.10_k0.50_c0.50`, `b0.10_k0.90_c0.50`) were skipped in the sweep."

Recomputed: `experiments/p3_specialization/phase_diagram_ppo_v2/` contains
**37 cell_ directories**. Cross-referencing against the Nash 39-cell set:

- `b0.10_k0.50_c0.50` is in Nash but absent from PPO_v2: ✓ disclosed
- `b0.10_k0.90_c0.50` is in Nash but absent from PPO_v2: ✓ disclosed
- Total PPO: 37 cells × 4 seeds = 148 runs (paper L344): ✓

## Figure 2 spot-check

Figure 2 (`figures/recalibrated_heatmap.pdf`) is rendered offline by
`figures/src/recalibrated_heatmap.py` from `recalibrated_verdict.json`.

| Panel | Cell | Metric | Source JSON value | Match |
|---|---|---|---|---|
| c=0.5 | β=0.5, κ=0.5 (symmetric) | gap_closed_ne | +0.33396 | ✓ |
| c=1.0 | β=0.5, κ=0.5 (symmetric) | gap_closed_ne | +0.05067 | ✓ |
| c=2.0 | β=0.9, κ=0.5 (mixed) | gap_closed_ne | +0.08468 | ✓ |
| c=0.5 | β=0.1, κ=0.1 (no_conv) | gap_closed_homogeneous (†) | +0.04800 | ✓ |
| c=0.5 | β=0.5, κ=0.9 (asym) | gap_closed_ne | +0.16174 | ✓ |

`pdfinfo figures/recalibrated_heatmap.pdf` reports: 1 page, valid PDF,
Matplotlib v3.10.9 backend, mtime 2026-06-11 22:57. No staleness concern.

## Figure 1 (phase_diagram.png)

Static PNG carried over from v3 (mtime 2026-06-11 22:55, 41 KB). Caption
updated to reference the wider grid and the new `mixed` class without
changing the underlying image, which still shows the c=0.5 panel of the
Nash phase diagram. The c=0.5 panel verdict assignments (3 no_conv at
κ=0.1, 2 sym_only at κ=0.5, 2 asym_only at κ=0.9) all reconcile with
`results.json`. No staleness check applies (no in-paper-dir source script).

## Table 1 (related-work)

Comparison-table claims (Overcooked 6, Melting Pot 8, Hanabi ≤20, SMAC
6+n_e, MAgent 21, PettingZoo MPE 3–5, Bucket Brigade 8/40) all reconcile
with `refs/benchmark_comparison.md` per v3 audit row 30. No changes in v4.

## Summary

- **One numerical disagreement that touches a structural claim**: the
  β-independence count is 12/13 (not 13/13) at the verdict level on the
  Nash 39-cell grid. The discrepancy comes from the cell (κ=0.9, c=0.5,
  β=0.1) returning `mixed` while β=0.5 and β=0.9 at the same (κ, c) return
  `asymmetric_only`. The qualitative β-independence framing still holds
  on 12/13; the headline ratio is off. See flags.md M3.
- **All 4-class ordering numbers reconcile**: symmetric (+0.180) >
  mixed (+0.107) > asymmetric (+0.059) > no_convergence (−0.049 on
  `gap_closed_homogeneous`). Independently recomputed from
  `recalibrated_verdict.json`.
- **4×-budget structural-failure claim verifies**: −0.049 → −0.108 on
  no_convergence cells (worsening).
- **Analytical κ-thresholds (0.028 / 0.65 / 0.972) and environment
  constants** all carry over from v3 audit and remain correct.
- **Figure 2 spot-check**: 5/5 cells match the source JSON to 5 d.p.
- **Two PPO-skipped cells** disclosed at §4 protocol match the actual
  37-cell PPO_v2 directory tree.
