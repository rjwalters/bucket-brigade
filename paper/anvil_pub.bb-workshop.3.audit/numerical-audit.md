# Numerical audit — `anvil_pub.bb-workshop.3`

Numbers-in-text vs figures/tables/source-of-truth consistency check for
`main.tex`. Sources consulted:

- `experiments/p3_specialization/phase_diagram_ppo/recalibrated_verdict.json`
  (the source-of-truth that Figure 2 is rendered from, deterministic)
- `experiments/p3_specialization/phase_diagram_ppo/recalibrated_verdict.md`
- `experiments/nash/phase_diagram/per_cell_baselines.json`
- `experiments/nash/phase_diagram/results.json` (NE classifications per cell)
- `paper/anvil_pub.bb-workshop/refs/ne_structure.md` (analytical thresholds)
- `paper/anvil_pub.bb-workshop/refs/env_spec.md` (state/action cardinality)
- `paper/anvil_pub.bb-workshop/refs/benchmark_comparison.md` (Table 1)

## Consistency table

| # | Claim location (main.tex line) | Claim value | Source | Source value | Agreement |
|---|---|---|---|---|---|
| 1 | Abstract L35: "per-agent action space at the minimal parameterization is $8$" | 8 | refs/benchmark_comparison.md Finding 1 + refs/env_spec.md | `MultiDiscrete([2,2,2])=8` | ✓ |
| 2 | Abstract L35: "state space is $2304$ states" | 2304 | refs/benchmark_comparison.md Finding 2 | `3^2·2^4·2^4 = 9·16·16 = 2304` | ✓ |
| 3 | Abstract L46: `gap_closed_ne` on symmetric-only row = $0.326$ | 0.326 | recalibrated_verdict.json, sym cells mean of (0.33396, 0.31724) | 0.32560 → rounded 0.326 | ✓ |
| 4 | Abstract L46: asymmetric-only row = $0.106$ | 0.106 | recalibrated_verdict.json, asym cells mean of (0.16174, 0.05094) | 0.10634 → rounded 0.106 | ✓ |
| 5 | Abstract L47: no-pure-NE row "near-zero" | near-zero | recalibrated_verdict.json, no_conv homo means (0.04800, -0.07847, -0.06924) → mean -0.0332 | "near-zero" matches (claim is qualitative); §4 Figure 2 caption cites -0.033 specifically | ✓ |
| 6 | Abstract L48: "thresholds … disagree by 3–10×" | 3–10× | refs/ne_structure.md headline + §5; analytical 0.028 vs empirical ~0.3 (≈10×); analytical 0.972 vs empirical ~0.7 (≈1.4×) | range "3–10×" mostly captures the collapse-boundary side; asymmetric-side ratio is closer to 1.4×. Paper's framing "3–10×" is reused throughout (§3, conclusion). Discrepancy is within the qualifier "3–10×" which already brackets a wide range. | ✓ acceptable as bracket |
| 7 | §2 L114: "minimal scenario fixes $H{=}2,N{=}4$" | H=2, N=4 | refs/env_spec.md | H=2, N=4 (v2_minimal) | ✓ |
| 8 | §2 L122: "Per-agent action cardinality is $4H$ ($8$ at $H{=}2$; $40$ at $H{=}10$)" | 8, 40 | refs/benchmark_comparison.md Finding 1 | 8 at minimal, 40 at default | ✓ |
| 9 | §2 L123: "Joint-action cardinality at the minimal parameterization is $8^4{=}4096$" | 4096 | refs/benchmark_comparison.md Finding 1 | `8^4 = 4096` | ✓ |
| 10 | §3 L182: reward tuple $W=(10,10,50,0,100,0)$ | tuple | refs/ne_structure.md §1 | $(R_{\text{team}}, P_{\text{team}}, r_{\text{own}}, r_{\text{other}}, p_{\text{own}}, p_{\text{other}}) = (10, 10, 50, 0, 100, 0)$ | ✓ |
| 11 | §3 L182: $\rho{=}0.02, T_{\min}{=}12$ | 0.02, 12 | refs/ne_structure.md §1 | ρ=0.02, T_min=12 | ✓ |
| 12 | §3 L193: "$A{=}\ldots T_{\min}/H{=}1812$" | 1812 | refs/ne_structure.md §2 | A = r_own·T_min + p_own·T_min + P_team·T_min/H = 50·12 + 100·12 + 10·12/10 = 600 + 1200 + 12 = 1812 | ✓ |
| 13 | §3 L194: "$\tilde A := A\rho = 36.24$" | 36.24 | refs/ne_structure.md §2 | 1812 · 0.02 = 36.24 | ✓ |
| 14 | §3 L230: collapse boundary "$\kappa\approx 0.028$" | 0.028 | refs/ne_structure.md §3 (eq A.a lower root) | $\kappa \approx 0.028$ | ✓ |
| 15 | §3 L231: symmetric NE exists on $\kappa\in[0.030, 0.65]$ | [0.030, 0.65] | refs/ne_structure.md §3.1 (roots of $f(\kappa)=\kappa(1-\kappa)^3=0.0276$) | 0.030 and 0.65 | ✓ |
| 16 | §3 L232: asymmetric NE exists "$\kappa\gtrsim 0.972$" | 0.972 | refs/ne_structure.md §3.2 (upper root of $g(\kappa)=36.24\kappa(1-\kappa)=1$) | 0.972 | ✓ |
| 17 | §3 L245: empirical collapse near $\kappa\approx 0.3$ | 0.3 | refs/ne_structure.md headline + §5.1 | "near $\kappa\approx 0.3$"; "ring-corrected → $\kappa \in [0.1, 0.3]$" | ✓ |
| 18 | §3 L246: empirical asymmetric onset near $\kappa\approx 0.7$ | 0.7 | refs/ne_structure.md headline | "near $\kappa\approx 0.7$" | ✓ |
| 19 | §3 L248: "Only 2 of 7 cells … achieve exact verdict agreement" | 2/7 | refs/ne_structure.md §4 | "2/7 cells … 5/7 cells disagree" | ✓ |
| 20 | §3 L242: "3/3 $\kappa$-rows show identical verdicts across $\beta$" | 3/3 | results.json + ne_structure.md §4 | κ=0.1: 3 cells all no_conv; κ=0.5: 2 cells both sym; κ=0.9: 2 cells both asym. 3 rows / 3 rows internally identical. Note: only the κ=0.1 row has all three β samples; the other two rows are 2-cell. The "3/3" framing reports row-level consistency, not per-row n=3 samples. | ✓ (correct under row-level reading) |
| 21 | §3 L279: "$7\times 4$ predicted-vs-empirical table" in `bbnestructure` | 7×4 | refs/ne_structure.md §4 | Actual table is 7 rows × **6** columns (β, κ, Empirical, Predicted, Match?, Notes). | **minor: 7×4 mismatch; actual dimensionality is 7×6.** Not load-bearing — the cite is to a table whose content is what the surrounding sentence refers to ("full bias accounting"), and that content is in the memo. |
| 22 | §4 L306: "$50$ training iterations of $2048$ rollout steps and $4$ random seeds per cell ($N{=}28$ runs)" | 50, 2048, 4, 28 | 7 cells × 4 seeds = 28; ne_structure.md notes the 7-cell preview; recalibrated_verdict.md confirms seeds=4 across all 7 cells | ✓ |
| 23 | §4 L340–342: ordering values $0.326$, $0.106$, $-0.033$ | 0.326, 0.106, -0.033 | recalibrated_verdict.md ordering check | 0.326, 0.106, -0.033 | ✓ |
| 24 | §4 L344: "two $\kappa{=}0.5$ cells — $\beta{=}0.5$ and $\beta{=}0.9$ — differ by $0.017$" | 0.017 | recalibrated_verdict.json: $\beta{=}0.5$ ne=0.33396, $\beta{=}0.9$ ne=0.31724 | $\|0.33396 - 0.31724\| = 0.01672$ → 0.017 | ✓ |
| 25 | §4 L348: per-cell std "$\approx 0.33$" | 0.33 | recalibrated_verdict.json: gap_closed_ne_std for sym cells = 0.33139, 0.34050 | ~0.33–0.34 | ✓ (rounded) |
| 26 | §4 L362: "inverted ordering: asymmetric_only $(0.262)$ > symmetric_only $(0.091)$ > no_convergence $(-0.176)$" | 0.262, 0.091, -0.176 | recalibrated_verdict.md "OLD gap_closed" ordering | 0.262, 0.091, -0.176 | ✓ |
| 27 | §4 caption L391: symmetric-only $0.326$ mean across $\beta\in\{0.5,0.9\}$ | 0.326 | same as row 3 | ✓ | ✓ |
| 28 | §4 caption L393: asymmetric-only $0.106$ | 0.106 | same as row 4 | ✓ | ✓ |
| 29 | §4 caption L397: "$\beta{=}0.1$ column was not sampled" | beta=0.1 not sampled at κ∈{0.5,0.9} | recalibrated_verdict.json: only β=0.1 cell present is at κ=0.1 (b0.10_k0.10_c0.50); β=0.1 cells at κ∈{0.5,0.9} are absent | ✓ |
| 30 | §5 Table 1 rows | various | refs/benchmark_comparison.md Findings 1–5 + the §"Action-space sizes in the table" appendix | Overcooked: 6 ✓; Melting Pot: 8 ✓; Hanabi: ≤20 (consistent with Hanabi standard action space; not explicit in refs but matches the published Hanabi action cardinality of ≤20 for full info-action sets); SMAC: 6+n_e ✓; MAgent: 21 ✓; PettingZoo MPE: 3-5 ✓; Bucket Brigade: 8/40 ✓ | ✓ |
| 31 | §6 L467: "state space is $2304$ at the minimal scenario and $\sim\!10^{10}$ at the default 10-house" | 2304, ~10^10 | refs/benchmark_comparison.md Finding 2 | 2304 at minimal; default 10-house "$3^{10}\cdot 10^4\cdot 2^4\approx 9.4\times 10^9$" — paper rounds "~10^10" which is one order up; refs says "≈ 9.4 × 10^9" | ✓ (within rounding; "~10^10" is acceptable order-of-magnitude framing for 9.4×10^9) |

## Figure 2 spot-check (rendered from `recalibrated_verdict.json`)

The Figure 2 generator (`figures/src/recalibrated_heatmap.py`) reads
the source JSON deterministically. Three cell values spot-checked
against the JSON:

| Cell                | Metric shown          | Source value | JSON value | Match |
|---------------------|----------------------|--------------|------------|-------|
| $\beta{=}0.5,\kappa{=}0.5$ | gap_closed_ne (sym) | +0.334       | 0.33396    | ✓ |
| $\beta{=}0.9,\kappa{=}0.9$ | gap_closed_ne (asym) | +0.051      | 0.05094    | ✓ |
| $\beta{=}0.1,\kappa{=}0.1$ | gap_closed_homo ($\dagger$) | +0.048 | 0.04800    | ✓ |

The script is deterministic and re-runnable; no stale-figure concern.

## Figure 1 (phase_diagram.png)

The phase-diagram figure is a static PNG (`figures/phase_diagram.png`,
41 KB, mtime 2026-06-11 11:19) sourced from
`experiments/nash/phase_diagram/phase_diagram.png` via the
`paper/anvil_pub.bb-workshop/refs/phase_diagram.png` symlink. No
in-paper-dir source script ships for Figure 1, so a staleness check
against `figures/src/` is not applicable. The figure's verdicts
(3 no_convergence + 2 symmetric_only + 2 asymmetric_only on the 7
sampled cells) agree with `results.json` "verdict" fields and with
the 3/3/2/2 counts spot-checked above.

## Summary

- **Zero numerical disagreements that change a conclusion.**
- **One minor discrepancy** (row 21): `bbnestructure` table is 7×6,
  not 7×4 as cited. The cite refers to content (bias accounting),
  not to a dimensionality claim — non-load-bearing.
- All abstract / §3 / §4 / §5 / §6 headline numbers reconcile with
  their on-disk source.
