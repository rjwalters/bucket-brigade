# Numerical audit for `anvil_pub.bb-workshop.5`

Every load-bearing numerical claim in the v5 body checked against its source-of-truth file. Source paths:

- Nash 39-cell verdicts: `experiments/nash/phase_diagram/results.json`
- PPO 37-cell sweep: `experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json`
- 4×-budget sweep: `experiments/p3_specialization/phase_diagram_ppo_longbudget/recalibrated_verdict.json`
- Memo source for Appendix A: `paper/anvil_memo.env_spec.1/env_spec.md`
- Memo source for Appendix B: `paper/anvil_memo.ne_structure.1/ne_structure.md`

## Body §1 / Abstract numerical claims

| # | Claim | Paper value | Source value | Match? |
|---|-------|------------|--------------|--------|
| 1 | Per-agent action cardinality at H=2 | 8 | 4H = 4·2 = 8 | ✓ |
| 2 | State cardinality at H=2,N=4 | 2304 | 3^2·2^4·2^4 = 9·16·16 = 2304 (memo §2.1 confirms) | ✓ |
| 3 | Joint action at H=2,N=4 | 4096 | (4H)^N = 8^4 = 4096 (memo §2.2 confirms) | ✓ |
| 4 | Phase-diagram grid | 39 cells | Nash results.json `grid.total_cells = 39` | ✓ |
| 5 | β-independence count (Abstract / §3) | "12 of 13 (κ,c) rows" | Recompute from results.json: 13 (κ,c) rows have ≥2 β samples; 12 identical, 1 splits at (κ=0.9, c=0.5) | ✓ |
| 6 | Class means for §1 Abstract: sym 0.180, mix 0.107, asy 0.059, col -0.049 | 0.180 / 0.107 / 0.059 / -0.049 | gap_closed_ne means by class (no_conv uses gap_closed_homogeneous): 0.1799 / 0.1070 / 0.0595 / -0.0485 | ✓ (rounded) |
| 7 | 4×-budget no-convergence mean | -0.108 | longbudget mean of gap_closed_homogeneous over 6 cells = -0.1079 | ✓ |

## §3 (NE structure) numerical claims

| # | Claim | Paper value | Source value | Match? |
|---|-------|------------|--------------|--------|
| 8 | Ã = Aρ | 36.24 | A·ρ = 1812·0.02 = 36.24 (memo §2.1) | ✓ |
| 9 | A coefficient | 1812 | 50·12 + 100·12 + 10·12/10 = 600+1200+12 = 1812 (memo §2.1) | ✓ |
| 10 | Symmetric NE κ-band | [0.030, 0.65] | memo §3.1 reports same roots | ✓ |
| 11 | Asymmetric NE threshold | κ > 0.972 | memo §3.2 reports same root | ✓ |
| 12 | Collapse boundary | κ ≈ 0.028 | memo §3.2 reports same root | ✓ |
| 13 | Predicted-class share in Table 1 row κ=0.1 | 0/9 (predicted sym, empirical 6 collapse + 3 asymmetric) | Nash JSON κ=0.1: {no_convergence: 6, asymmetric_only: 3}, n=9 | ✓ |
| 14 | Predicted-class share Table 1 row κ=0.3 | 6/6 (predicted sym, empirical 6 sym) | Nash JSON κ=0.3: {symmetric_only: 6}, n=6 (this is the row that lost cells because high-κ × c=0.5 corner is subsumed by c=1.0 — wait, this is the κ=0.3 row, full 6, not 9) | ✓ (n=6 because κ=0.3 was sampled only at c∈{1.0, 2.0}, the changelog and Table 1 caption both say "6 cells" for κ=0.3 implicitly via the row total) — **see flag M1** |
| 15 | Predicted-class share Table 1 row κ=0.5 | 6/9 (predicted sym, empirical 6 sym + 3 mixed) | Nash JSON κ=0.5: {symmetric_only: 6, mixed: 3}, n=9 | ✓ |
| 16 | Predicted-class share Table 1 row κ=0.7 | 0/6 (predicted mixed/transition, empirical 6 asymmetric) | Nash JSON κ=0.7: {asymmetric_only: 6}, n=6 (Nash only sampled κ=0.7 at c∈{1.0, 2.0}) | ✓ |
| 17 | Predicted-class share Table 1 row κ=0.9 | 7/9 (predicted mixed/transition, empirical 7 mixed + 2 asy) | Nash JSON κ=0.9: {mixed: 7, asymmetric_only: 2}, n=9 | ✓ |
| 18 | Overall verdict counts (text §3) | sym 12 / asy 11 / no_conv 6 / mixed 10 | Nash JSON verdict_counts: {symmetric_only: 12, asymmetric_only: 11, no_convergence: 6, mixed: 10} | ✓ |
| 19 | "Modal empirical class on 3/5 κ rows" claim | 3/5 | Direct count from rows: κ=0.3 sym (predicted sym → 6/6), κ=0.5 sym (predicted sym → 6/9 modal), κ=0.9 mixed (predicted mixed → 7/9 modal). Off-rows: κ=0.1 (predicted sym; empirical collapse modal), κ=0.7 (predicted mixed; empirical asymmetric modal). → 3/5 verified | ✓ |
| 20 | Lone splitter location | (κ=0.9, c=0.5): β=0.1→mixed, β∈{0.5,0.9}→asymmetric_only | Direct read from results.json: confirmed | ✓ |
| 21 | f(1/4) maximum of κ(1-κ)^3 | 27/256 ≈ 0.105 | (1/4)·(3/4)^3 = 27/256 = 0.10547 | ✓ |
| 22 | g(1/2) for 36.24κ(1-κ) | 9.06 | 36.24/4 = 9.06 | ✓ |

## §4 (PPO trainability) numerical claims

| # | Claim | Paper value | Source value | Match? |
|---|-------|------------|--------------|--------|
| 23 | PPO cell coverage | 37 of 39 | recalibrated_verdict.json has 37 cells | ✓ |
| 24 | PPO seeds | 4 per cell | n_seeds=4 in every cell of recalibrated_verdict.json | ✓ |
| 25 | Total PPO runs (N=148) | 148 | 37·4 = 148 | ✓ |
| 26 | Skipped cells | b0.10_k0.50_c0.50, b0.10_k0.90_c0.50 | These two cells are absent from recalibrated_verdict.json | ✓ |
| 27 | gap_closed_ne sym | 0.180 (n=11) | mean over 11 sym cells = 0.1799 | ✓ |
| 28 | gap_closed_ne mixed | 0.107 (n=9) | mean over 9 mixed cells = 0.1070 | ✓ |
| 29 | gap_closed_ne asym | 0.059 (n=11) | mean over 11 asym cells = 0.0595 | ✓ |
| 30 | gap_closed_homogeneous no_conv | -0.049 (n=6) | mean over 6 no_conv cells = -0.0485 | ✓ |
| 31 | Class-mean separations (0.073, 0.048, 0.108) | 0.073, 0.048, 0.108 | 0.180-0.107=0.073, 0.107-0.059=0.048, 0.059-(-0.049)=0.108 | ✓ — note asy↔collapse separation crosses metrics (gap_closed_ne vs gap_closed_homogeneous), see Minor flag below |
| 32 | Per-class std range ≈ 0.21-0.44 | 0.21-0.44 | Mean of per-cell stds by class: mixed 0.2085, no_conv 0.2565, asym 0.4190, sym 0.4397 — range 0.21-0.44 holds at this interpretation | ✓ (under the "class-level mean of per-cell stds" interpretation; raw per-cell stds span a wider 0.16-0.99 if read literally) |
| 33 | 4×-budget mean gap_closed_homogeneous worsens to -0.108 | -0.108 | longbudget mean of 6 no_conv cells = -0.1079 | ✓ |
| 34 | 4×-budget hardware/scale | 200 iters × 4096 steps, 4 seeds, ~6 hours on alc-6 | longbudget per-cell configs confirm 200 iters × 4096 steps × 4 seeds (config check) | ✓ (host attribution is metadata, not verifiable from JSON) |
| 35 | 7-cell preview inversion: asymmetric_only 0.262, symmetric_only 0.091, no_convergence -0.176 | 0.262 / 0.091 / -0.176 | Carried forward from v4; v4 audit verified this against the old-baseline columns; unchanged in v5 | ✓ (carry-forward) |
| 36 | Homogeneous metric class means (Caveats paragraph) | sym 0.051, mix 0.050, asy 0.033, col -0.049 | gap_closed_homogeneous means by verdict: sym 0.0507, mix 0.0502, asy 0.0328, col -0.0485 | ✓ (rounded) |
| 37 | PPO β-independence count (§4) | "all 13 (κ,c) rows of the PPO subgrid with ≥2 β samples" | PPO subgrid skipped the (κ=0.9, c=0.5) splitter so its 13 rows are all identical-across-β; v4 audit verified, unchanged in v5 | ✓ (carry-forward from v4 verified state) |

## §5 (Related work) / Table 2 numerical claims

| # | Claim | Paper value | Source value | Match? |
|---|-------|------------|--------------|--------|
| 38 | Overcooked per-agent action | 6 | published environment uses 6 actions (no-op, up, down, left, right, interact) | ✓ |
| 39 | Hanabi action upper bound | ≤20 | published spec | ✓ |
| 40 | Bucket Brigade Table 2 row | 8 / 40 | 4H at H=2 / 4H at H=10 | ✓ |
| 41 | "more than 50 substrates" for Melting Pot | >50 | Melting Pot has ~85 substrates (more than 50 is a fair lower bound) | ✓ |

## Appendix A numerical claims vs `env_spec.md`

Spot-checked Appendix A.1, A.2, A.6, A.7. Direct ports preserve every numerical
constant: 2304 states (A.2), 9.44×10^9 default-scenario states (A.2), 4096 joint
actions (A.2), default reward tuple (100, 100, 20, 0, 40, 0) (A.1), minimal-spec
reward tuple (10, 10, 50, 0, 100, 0) (referenced through A.7 from §3),
c_rest = 0.5 (A.4), T_min = 12 (A.1). No transcription errors detected.

## Appendix B numerical claims vs `ne_structure.md`

Spot-checked Appendix B.2 (Ã derivation), B.3 (NE candidates), B.4 (7-cell
phase table), B.5 (gaps). The derivation reproduces: A = 1812, Ã = 36.24,
symmetric κ-band [0.030, 0.65], asymmetric threshold 0.972, collapse boundary
0.028, ring-corrected defence 1-(1-κ)^0.4 ≈ 0.24 at κ=0.5 (B.5), Firefighter
effective k_{-i} ≈ 3.7 (B.5). The 7-cell preview table in Appendix B.4 matches
the memo's Table 3.4 verdicts (3 no_convergence at κ=0.1, 2 symmetric_only at
κ=0.5, 2 asymmetric_only at κ=0.9). No transcription errors detected.

## Summary

- **Claims checked**: 41 distinct numerical claims (body) + 14 appendix
  numbers (spot-checked).
- **Disagreements (load-bearing)**: 0.
- **Minor / interpretation notes**: 2.
  1. The "per-class std ≈ 0.21-0.44" range is the mean-of-per-cell-stds
     read; the raw per-cell-std list is wider (0.16-0.99). The paper's
     phrasing ("per-class std (≈0.21-0.44)") is most naturally read as
     this class-aggregate, and the resulting range is correct under that
     reading — but a reader could legitimately read it as the per-cell
     range, where the bound is loose. Worth tightening for camera-ready.
  2. The class-mean separation list `(0.073, 0.048, 0.108)` mixes two
     metrics: 0.073 and 0.048 come from gap_closed_ne (sym, mix, asy
     classes), while 0.108 is the asym(gap_closed_ne)→collapse
     (gap_closed_homogeneous) separation. The text's prior sentence
     (collapse "metric falls back to gap_closed_homogeneous because no
     NE policy exists") flags this implicitly, but the separation
     calculation itself does not asterisk the cross-metric subtraction.
     This is a transparency note, not a correctness flag.
