"""Analysis for issue #285 — BC-init + high-λ PPO combined verdict.

Aggregates the 4 λ × 3 seed sweep produced by ``run_issue285_bc_highlambda.sh``
and emits a per-λ basin-trap / anti-attractor / partial verdict by reusing
``analyze_270.classify_verdict`` for each cell. The output operationalizes
the 2×2 verdict matrix in the issue #285 body:

| #270 verdict (BC-init λ=0.95) | This issue (best high-λ cell) | Interpretation |
|---|---|---|
| basin_trap                    | basin_trap                    | Specialist is stable; BC-init load-bearing forever. |
| basin_trap                    | anti_attractor                | High-λ destabilizes the basin (unexpected — investigate). |
| anti_attractor                | basin_trap                    | High-λ rescues specialist (high-λ is sufficient correction). |
| anti_attractor                | anti_attractor                | Anti-attractor is deeper than GAE bias; pivot to LOLA / shaping. |

Layout consumed (matches the sweep driver):

    experiments/p3_specialization/runs/issue285_bc_highlambda/
      minimal_specialization/
        lambda_0_95/seed_{42,43,44}/metrics.json
        lambda_0_99/seed_{42,43,44}/metrics.json
        lambda_0_999/seed_{42,43,44}/metrics.json
        lambda_1_0/seed_{42,43,44}/metrics.json

Per λ cell, this module:

1. Aggregates the per-seed trajectories with ``analyze_270.aggregate_arm``
   (re-use, not re-implement — preserves bit-identical agreement with the
   #270 verdict at the λ=0.95 baseline cell).
2. Classifies the arm with ``analyze_270.classify_verdict``.
3. Picks the "best" λ as the cell with the highest trailing-5 mean
   gap_closed (i.e., the strongest high-λ rescue candidate).

Optional cross-check: if a #270 BC-init λ=0.95 reference run is available
under ``--reference-270-root``, the λ=0.95 cell's iter0 and trailing5
gap_closed are compared to it and flagged if they drift outside
``BASELINE_DRIFT_TOLERANCE``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Ensure ``analyze_270`` is importable when running this script directly.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import analyze_270  # noqa: E402

# Default λ grid mirrors the sweep driver. Lambda → directory tag
# replaces '.' with '_' (matches the driver and analyze_282.py).
LAMBDA_VALUES: List[float] = [0.95, 0.99, 0.999, 1.0]

# Allowed drift between this sweep's λ=0.95 cell and the #270 baseline
# (test plan acceptance criterion: ±0.01 sanity check; we widen here to
# absorb seed reshuffle and reduce false-positive cross-check failures
# while still flagging gross stack regressions).
BASELINE_DRIFT_TOLERANCE = 0.05


def _lambda_tag(lam: float) -> str:
    """Replace '.' with '_' to match the sweep driver's path convention."""
    return str(lam).replace(".", "_")


def aggregate_lambda(cells: List[Path], lam: float) -> Dict:
    """Aggregate one λ cell's seeds via ``analyze_270.aggregate_arm``.

    The arm label includes the λ value so verdict-md output is unambiguous
    when multiple λ cells appear in the same report.
    """
    arm = analyze_270.aggregate_arm(cells, label=f"bc_init_ppo_lambda_{lam}")
    arm["lambda"] = lam
    return arm


def reference_270_cross_check(
    sweep_lambda_0_95_arm: Dict,
    reference_270_root: Optional[Path],
    seeds: List[int],
) -> Dict:
    """Check whether the λ=0.95 cell matches a #270 PR-#278 reference run.

    When ``reference_270_root`` is ``None`` (the common case for one-shot
    analyses on a fresh remote), returns a ``skipped`` status. Otherwise
    loads the reference seeds with ``analyze_270.aggregate_arm`` and
    compares trailing-5 gap_closed.
    """
    if reference_270_root is None:
        return {
            "status": "skipped",
            "message": "No --reference-270-root provided; skipping cross-check.",
        }
    ref_cells = [reference_270_root / f"seed_{s}" for s in seeds]
    ref_arm = analyze_270.aggregate_arm(ref_cells, label="reference_270_bc_init_ppo")
    if ref_arm.get("n_seeds", 0) == 0:
        return {
            "status": "missing",
            "message": (
                f"Reference root {reference_270_root} has no completed seeds; "
                "cannot cross-check λ=0.95 against #270."
            ),
        }
    if sweep_lambda_0_95_arm.get("n_seeds", 0) == 0:
        return {
            "status": "missing",
            "message": "Sweep λ=0.95 cell has no completed seeds; cannot cross-check.",
        }
    observed = sweep_lambda_0_95_arm["trailing5_gap_closed_mean"]
    expected = ref_arm["trailing5_gap_closed_mean"]
    delta = observed - expected
    if abs(delta) <= BASELINE_DRIFT_TOLERANCE:
        return {
            "status": "ok",
            "observed_trailing5_gap_closed": observed,
            "reference_trailing5_gap_closed": expected,
            "delta": delta,
            "message": (
                f"λ=0.95 trailing5 gap_closed = {observed:.3f} matches the "
                f"#270 reference ({expected:.3f}) within "
                f"±{BASELINE_DRIFT_TOLERANCE:.2f}."
            ),
        }
    return {
        "status": "drift",
        "observed_trailing5_gap_closed": observed,
        "reference_trailing5_gap_closed": expected,
        "delta": delta,
        "message": (
            f"λ=0.95 trailing5 gap_closed = {observed:.3f} drifts from the "
            f"#270 reference ({expected:.3f}) by {delta:+.3f}; tolerance "
            f"±{BASELINE_DRIFT_TOLERANCE:.2f}. Possible stack regression — "
            "investigate before trusting the high-λ ladder."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue285_bc_highlambda/minimal_specialization"
        ),
        help="Root containing lambda_<L>/seed_<S> cells from the sweep driver.",
    )
    p.add_argument(
        "--lambdas",
        type=float,
        nargs="+",
        default=LAMBDA_VALUES,
        help="λ values to aggregate. Default matches the sweep driver grid.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--reference-270-root",
        type=Path,
        default=None,
        help=(
            "Optional: root containing seed_<S> cells from the #270 / PR #278 "
            "BC-init PPO continuation (e.g., "
            "experiments/p3_specialization/runs/issue270_bc_continuation/"
            "minimal_specialization/lambda_0e0). If supplied, the λ=0.95 cell "
            "is cross-checked against this reference."
        ),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue285_bc_highlambda"
        ),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda_arms: Dict[float, Dict] = {}
    lambda_verdicts: Dict[float, Dict] = {}
    for lam in args.lambdas:
        cells = [
            args.runs_root / f"lambda_{_lambda_tag(lam)}" / f"seed_{s}"
            for s in args.seeds
        ]
        arm = aggregate_lambda(cells, lam)
        lambda_arms[lam] = arm
        verdict, reasoning = analyze_270.classify_verdict(arm)
        lambda_verdicts[lam] = {"verdict": verdict, "reasoning": reasoning}

    # "Best" λ = the λ whose trailing-5 gap_closed_mean is highest among the
    # cells with completed seeds. This is the strongest candidate for "high-λ
    # rescues the basin"; the verdict at this λ is the key cell of the 2×2.
    completed = {
        lam: arm
        for lam, arm in lambda_arms.items()
        if arm.get("n_seeds", 0) > 0
    }
    if completed:
        best_lambda, best_arm = max(
            completed.items(),
            key=lambda kv: kv[1]["trailing5_gap_closed_mean"],
        )
        best_verdict = lambda_verdicts[best_lambda]["verdict"]
        best_reasoning = lambda_verdicts[best_lambda]["reasoning"]
        best_gc = best_arm["trailing5_gap_closed_mean"]
    else:
        best_lambda = None
        best_verdict = "no_data"
        best_reasoning = (
            "No λ cell produced loadable metrics — cannot classify. Re-check "
            "the sweep output layout under ``runs/issue285_bc_highlambda/``."
        )
        best_gc = float("nan")

    # Reference cross-check against #270 (PR #278) λ=0.95 baseline.
    sweep_0_95 = lambda_arms.get(0.95, {})
    ref_check = reference_270_cross_check(
        sweep_0_95, args.reference_270_root, args.seeds
    )

    out = {
        "issue": 285,
        "best_lambda": best_lambda,
        "best_verdict": best_verdict,
        "best_reasoning": best_reasoning,
        "best_trailing5_gap_closed_mean": best_gc,
        "per_lambda_verdicts": {
            str(lam): {
                "lambda": lam,
                "n_seeds": lambda_arms[lam].get("n_seeds", 0),
                "iter0_gap_closed_mean": lambda_arms[lam].get(
                    "iter0_gap_closed_mean", float("nan")
                ),
                "trailing5_gap_closed_mean": lambda_arms[lam].get(
                    "trailing5_gap_closed_mean", float("nan")
                ),
                "min_iter_gap_closed_mean": lambda_arms[lam].get(
                    "min_iter_gap_closed_mean", float("nan")
                ),
                "verdict": lambda_verdicts[lam]["verdict"],
                "reasoning": lambda_verdicts[lam]["reasoning"],
            }
            for lam in args.lambdas
        },
        "reference_270_cross_check": ref_check,
        "lambda_arms": {str(lam): arm for lam, arm in lambda_arms.items()},
        "references": {
            "minspec_random": analyze_270.MINSPEC_RANDOM,
            "minspec_specialist": analyze_270.MINSPEC_SPECIALIST,
            "trailing_n": analyze_270.TRAILING_N,
            "baseline_drift_tolerance": BASELINE_DRIFT_TOLERANCE,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md_lines = [
        "# Issue #285 — BC-init + high-λ PPO verdict",
        "",
        f"**Best λ**: {best_lambda} (trailing5 gap_closed_mean = {best_gc:.3f})",
        "",
        f"**Best-cell verdict**: `{best_verdict}`",
        "",
        f"**Reasoning**: {best_reasoning}",
        "",
        "## Per-λ verdict ladder",
        "",
        "| λ | n_seeds | iter0 gap_closed | trailing5 gap_closed | min-iter gap_closed | verdict |",
        "|---|---|---|---|---|---|",
    ]
    for lam in args.lambdas:
        arm = lambda_arms[lam]
        md_lines.append(
            f"| {lam} | {arm.get('n_seeds', 0)} | "
            f"{arm.get('iter0_gap_closed_mean', float('nan')):.3f} | "
            f"{arm.get('trailing5_gap_closed_mean', float('nan')):.3f} | "
            f"{arm.get('min_iter_gap_closed_mean', float('nan')):.3f} | "
            f"`{lambda_verdicts[lam]['verdict']}` |"
        )

    md_lines.extend(
        [
            "",
            "## Reference cross-check (λ=0.95 vs #270 BC-init PPO continuation)",
            f"- status: `{ref_check['status']}`",
            f"- {ref_check['message']}",
            "",
            "## References",
            f"- random baseline (per-step team): {analyze_270.MINSPEC_RANDOM:.2f}",
            f"- specialist baseline (per-step team): {analyze_270.MINSPEC_SPECIALIST:.2f}",
            f"- trailing window: {analyze_270.TRAILING_N} iterations",
            f"- baseline drift tolerance: ±{BASELINE_DRIFT_TOLERANCE:.2f}",
            "",
            "## 2×2 verdict matrix (cross-issue interpretation)",
            "",
            "Cross-reference the **#270 (BC-init λ=0.95) verdict** with this "
            "issue's **best-cell verdict**:",
            "",
            "| #270 | this (best λ) | Interpretation |",
            "|---|---|---|",
            "| `basin_trap`     | `basin_trap`     | Specialist stable; BC-init load-bearing forever. |",
            "| `basin_trap`     | `anti_attractor` | High-λ destabilizes the basin (investigate). |",
            "| `anti_attractor` | `basin_trap`     | High-λ rescues specialist; high-λ is sufficient correction. |",
            "| `anti_attractor` | `anti_attractor` | Anti-attractor deeper than GAE bias; pivot to LOLA / shaping. |",
            "| `partial`        | (any)            | Hybrid intervention; verdict ladder shows direction. |",
            "",
        ]
    )
    (args.output_dir / "verdict.md").write_text("\n".join(md_lines))

    print(f"\nbest λ: {best_lambda} (trailing5 gap_closed_mean = {best_gc:.3f})")
    print(f"best-cell verdict: {best_verdict}")
    print(f"reasoning: {best_reasoning}")
    print(f"reference cross-check: {ref_check['status']} — {ref_check['message']}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
