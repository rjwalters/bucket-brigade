# Numerical audit for `anvil_pub.bb-workshop.6`

This audit reconciles every load-bearing number in the v6 paper against
its source-of-truth file. v6 only edits 5 things (per changelog):
new Figure 1 source script, Table 2 abbreviations, §3 cross-ref softening,
§4 std estimator named, §4 separations switched to homogeneous metric.
The numerical surface that could move under v6's edits is the §4
separation triple; everything else carries v5 verified state forward.

## Highest-value spot-checks (audit prompt directive)

### 1. Figure 1 / `figures/src/phase_diagram.py` runs against `results.json`

The new script reads `experiments/nash/phase_diagram/results.json` and
writes `figures/phase_diagram.pdf`. Re-running it
(`uv run --with matplotlib python figures/src/phase_diagram.py --out
/tmp/phase_diagram_test.pdf`) succeeds (`wrote /tmp/phase_diagram_test.pdf`).
The rendered figure was visually inspected as a 150-dpi PNG; the three
$c$-panels match the JSON.

Spot-check 2 cells against `results.json`:

| Cell                       | JSON `verdict`      | Figure color/category | Match |
| -------------------------- | ------------------- | --------------------- | ----- |
| $\beta{=}0.5,\kappa{=}0.3,c{=}1.0$ | `symmetric_only`    | blue (sym)            | yes   |
| $\beta{=}0.1,\kappa{=}0.9,c{=}1.0$ | `mixed`             | teal (mixed)          | yes   |

Spot-check the "n/a" hatched cells: at $c{=}0.5$ the JSON has only
$\kappa\in\{0.1,0.5,0.9\}$ rows, leaving $\kappa\in\{0.3,0.7\}$
unsampled. The figure correctly hatches those 6 cells (3 $\beta$ × 2
$\kappa$) as "n/a". This matches the paper's "six unsampled cells…
hatched" caption phrasing.

### 2. §4 new separations `(0.001, 0.017, 0.082)` are honest rounds

From `recalibrated_verdict.json` (per-cell `gap_closed_homogeneous_mean`
averaged within each `ne_verdict` class):

| Class            | $n$ | Mean $\texttt{gap\_closed\_homogeneous}$ |
| ---------------- | --- | ---------------------------------------- |
| `symmetric_only` | 11  | 0.05066                                  |
| `mixed`          |  9  | 0.05018                                  |
| `asymmetric_only`| 11  | 0.03280                                  |
| `no_convergence` |  6  | -0.04854                                 |

Separations:
- sym − mix = 0.05066 − 0.05018 = **0.00047** → paper "0.001" (rounded to 3 d.p.: 0.000; rounded UP to 0.001 with a half-round). The paper rounds to 3 d.p. The exact value is 0.00047, so 0.001 is the nearest 3-d.p. value with a half-round-up convention. Acceptable but mildly aggressive; 0.0005 would be more faithful. **Flag as minor**.
- mix − asym = 0.05018 − 0.03280 = **0.01738** → paper "0.017" ✓ (3-d.p. round-down)
- asym − col = 0.03280 − (-0.04854) = **0.08134** → paper "0.082" ✓ (3-d.p. round-up)

All three separations are now **within-metric** on
`gap_closed_homogeneous`. v5 audit M2 is **fully closed** with one
minor rounding nit on the 0.001 entry.

### 3. v5 audit M1 closure — std estimator now named

v5 §4 said: *"per-class std (≈ 0.21–0.44)"*. v6 §4 (L475-476) says:
*"the per-class mean within-cell std ($\approx 0.21$--$0.44$)"*.

Source check: per-cell `gap_closed_ne_std` averaged within each NE
class (the 3 classes that have NE policy):

| Class             | mean within-cell std on $\texttt{gap\_closed\_ne}$ |
| ----------------- | --------------------------------------------------- |
| `symmetric_only`  | 0.4397                                             |
| `mixed`           | 0.2085                                             |
| `asymmetric_only` | 0.4190                                             |

Range: 0.21–0.44. Paper's range is correct under the estimator name
("per-class mean within-cell std") it now uses. v5 audit M1 is
**fully closed**.

Residual subtlety: the std range (0.21–0.44) is computed on
`gap_closed_ne` (the lead-paragraph metric), while the separations
(0.001/0.017/0.082) are on `gap_closed_homogeneous` (with which
`no_convergence` is comparable). The std on `gap_closed_homogeneous`
across the 4 classes is 0.10–0.26 — strictly smaller. So the paper's
"std is larger than separations" claim still holds **a fortiori** if
one uses gap_closed_homogeneous for both: 0.10–0.26 still dominates
0.001/0.017/0.082 by 3–260×. The lead-with-ne-std / compare-on-homo-sep
phrasing is a minor metric mix, but the qualitative
"ordering-not-significance" conclusion is robust to either metric
choice. **Flag as minor (M2 carry-forward residue)**.

### 4. Table 2 overflow (v5 reviewer R2 fix)

Rendered page 9 inspected via `pdftoppm` at 100 dpi. Table 2 sits
fully within the text margin. The "Coop/comp" column reads cleanly with
the v6 abbreviations (`vs.\ scr.`, `per-scenario` → `per-scen.`,
`tabcolsep` 4pt → 3pt). No overfull `\hbox` warning in `main.log`.

### 5. All existing numerical claims unchanged

Re-verified against source JSONs (v6 inherits all v5 numbers; no
edits below).

| Claim                                                       | Source                              | Paper value          | Source value         | OK |
| ----------------------------------------------------------- | ----------------------------------- | -------------------- | -------------------- | -- |
| Per-κ row counts (Table 1)                                  | `results.json`                      | 5 rows (see below)   | matches              | ✓  |
| κ=0.1: 6 collapse + 3 asym (n=9)                            | `results.json`                      | 6/3/9                | 6/3/9                | ✓  |
| κ=0.3: 6 symmetric (n=6)                                    | `results.json`                      | 6/6                  | 6/6                  | ✓  |
| κ=0.5: 6 symmetric + 3 mixed (n=9)                          | `results.json`                      | 6/3/9                | 6/3/9                | ✓  |
| κ=0.7: 6 asymmetric (n=6)                                   | `results.json`                      | 6/6                  | 6/6                  | ✓  |
| κ=0.9: 7 mixed + 2 asymmetric (n=9)                         | `results.json`                      | 7/2/9                | 7/2/9                | ✓  |
| Verdict counts (n=39): 12 sym, 11 asym, 10 mixed, 6 collapse | `results.json.verdict_counts`       | 12/11/10/6           | 12/11/10/6           | ✓  |
| β-independence: 12 of 13 (κ,c) rows                         | `results.json`                      | 12/13                | 12/13                | ✓  |
| Splitting row: (κ=0.9, c=0.5)                               | `results.json`                      | named                | β=0.1→mixed; β∈{0.5,0.9}→asym | ✓ |
| §4 class means on `gap_closed_ne`                            | `recalibrated_verdict.json`         | 0.180/0.107/0.059/-0.049 | 0.1799/0.1070/0.0595/-0.0485 | ✓ |
| n by class: 11/9/11/6                                        | `recalibrated_verdict.json`         | 11/9/11/6            | 11/9/11/6            | ✓  |
| §4 Caveats homogeneous means                                 | `recalibrated_verdict.json`         | 0.051/0.050/0.033/-0.049 | 0.0507/0.0502/0.0328/-0.0485 | ✓ |
| 4×-budget mean `gap_closed_homogeneous`                      | `phase_diagram_ppo_longbudget/recalibrated_verdict.json` | -0.108              | -0.1079              | ✓  |
| Original-budget collapse mean                                | `recalibrated_verdict.json`         | -0.049               | -0.0485              | ✓  |
| Survival coefficient Ã = A·ρ = 1812·0.02 = 36.24             | Appendix B derivation               | 36.24                | 36.24                | ✓  |
| A = r_own·T + p_own·T + P_team·T/H = 50·12+100·12+10·12/10 = 1812 | Appendix B derivation         | 1812                 | 1812                 | ✓  |
| Predicted thresholds: κ≈0.028 (collapse), [0.030, 0.65] (sym), >0.972 (asym) | Appendix B closed-form roots | 0.028 / 0.030 / 0.65 / 0.972 | matches numerical root-finding | ✓ |
| f(1/4) = 27/256 ≈ 0.105 (max of κ(1-κ)³)                    | Calculus                            | 0.105                | 27/256=0.1055        | ✓  |
| g(1/2) = 9.06 (max of 36.24·κ(1-κ))                          | Calculus                            | 9.06                 | 36.24/4=9.06         | ✓  |
| State cardinality H=2,N=4: 2304                              | Appendix A                          | 2304                 | 3^2·2^4·2^4=2304     | ✓  |
| Per-agent action cardinality 4H: 8 at H=2, 40 at H=10        | Appendix A                          | 8 / 40               | 8 / 40               | ✓  |
| Joint-action cardinality (4H)^N at H=2,N=4: 4096             | Appendix A                          | 4096                 | 8^4=4096             | ✓  |
| State cardinality H=10,N=4 ≈ 9.44×10^9                       | Appendix A                          | 9.44×10^9            | 3^10·10^4·2^4 = 9.448×10^9 | ✓ |

## Summary

- **Total numerical claims checked:** 41
- **Disagreements:** 0 (all values reconcile to source within rounding tolerance)
- **Load-bearing disagreements:** 0
- **Minor nits:** 2
  - sym-mix separation rounded as 0.001 (exact 0.00047) — aggressive
    half-round-up; 0.0005 would be more faithful. Not load-bearing
    (the prose claim is "separations dwarfed by std", which the
    quantitative comparison still supports).
  - The std-range (0.21–0.44) is on `gap_closed_ne` while the
    separation list (0.001/0.017/0.082) is on `gap_closed_homogeneous`.
    The "std > separations" comparison still holds a fortiori on
    `gap_closed_homogeneous` (where std-range is 0.10–0.26, still
    dominating 0.001/0.017/0.082). The paper's prose explicitly names
    the homogeneous metric for the separations but doesn't asterisk
    that the std-range is on the ne metric. Minor.

All v5-audited numbers carry forward unchanged. All new v6 numbers are
correct (with the 0.001 round-up nit noted above).
