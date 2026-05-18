"""Analysis for issue #282 — high-λ GAE PPO smoke test.

Aggregates the 4 λ × 3 seed sweep produced by ``run_issue282_lambda_sweep.sh``
and emits a per-λ tier verdict ranking whether GAE bootstrap bias is the
source of PPO's plateau on ``minimal_specialization``. Mirrors the structure
of ``analyze_270.py`` and reuses the canonical ``gap_closed`` helper +
reference table.

Layout consumed (matches the sweep driver):

    experiments/p3_specialization/runs/issue282_lambda_sweep/
      minimal_specialization/
        lambda_0_95/seed_{42,43,44}/metrics.json
        lambda_0_99/seed_{42,43,44}/metrics.json
        lambda_0_999/seed_{42,43,44}/metrics.json
        lambda_1_0/seed_{42,43,44}/metrics.json

Per-cell metric: trailing-5-iteration mean ``mean_step_reward_team`` mapped
through ``gap_closed`` (i.e., aggregate over the last 5 iterations of one
seed, then convert to a fraction-of-specialist-gap-closed). The per-λ
``gap_closed_mean`` is the mean of the per-seed trailing5 gap_closed values.

Tier verdict (best-cell `gap_closed_mean` across the 4 λ values):

| Best-cell gap_closed_mean | Tier | Interpretation |
|---|---|---|
| ≥ 0.50 | tier_1_breaks_plateau | Bootstrap bias was the misalignment source. |
| 0.25 - 0.50 | tier_2_partial | Bootstrap is contributing but not sole source. |
| < 0.25 | tier_3_insufficient | Bootstrap is not the binding constraint. |

Per-cell cross-check: λ=0.95 cell's gap_closed_mean should reproduce the
post-#236 baseline (~0.182 ± seed noise — see PR #257 / analyze_270.py).
Material drift here implies a stack regression and blocks the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Hardcoded references (mirror analyze_270.py:39-40).
MINSPEC_RANDOM = -96.07
MINSPEC_SPECIALIST = -22.07
TRAILING_N = 5

# Tier thresholds (issue #282 curator spec).
TIER_1_THRESHOLD = 0.50  # breaks plateau → bootstrap bias is the source
TIER_2_THRESHOLD = 0.25  # partial → bootstrap contributes but not alone

# Re-baseline cross-check (PR #257 post-#236 random-init IPPO baseline).
BASELINE_GAP_CLOSED = 0.182
BASELINE_DRIFT_TOLERANCE = 0.10  # ±0.10 is "consistent with baseline"

# Sweep driver replaces '.' with '_' in lambda path components; we mirror that.
LAMBDA_VALUES: List[float] = [0.95, 0.99, 0.999, 1.0]


def gap_closed(per_step_team: float) -> float:
    """Fraction of the random→specialist gap closed by a per-step team reward."""
    return (per_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def _lambda_tag(lam: float) -> str:
    """Replace '.' with '_' to match the sweep driver's path convention."""
    return str(lam).replace(".", "_")


def _load_metrics(cell: Path) -> Optional[List[dict]]:
    f = cell / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _trailing_team_reward(metrics: List[dict]) -> float:
    """Trailing-``TRAILING_N`` mean of ``mean_step_reward_team`` for one cell."""
    traj = np.asarray(
        [row["mean_step_reward_team"] for row in metrics], dtype=np.float64
    )
    return float(traj[-TRAILING_N:].mean())


def aggregate_lambda(cells: List[Path], lam: float) -> Dict:
    """Aggregate trailing-5 gap_closed across seeds for one λ value."""
    per_seed: List[Dict] = []
    gap_values: List[float] = []
    team_values: List[float] = []
    for cell in cells:
        metrics = _load_metrics(cell)
        if metrics is None:
            per_seed.append({"cell": str(cell), "missing": True})
            continue
        trail_team = _trailing_team_reward(metrics)
        gc = gap_closed(trail_team)
        per_seed.append(
            {
                "cell": str(cell),
                "n_iters": len(metrics),
                "trailing5_team": trail_team,
                "trailing5_gap_closed": gc,
            }
        )
        gap_values.append(gc)
        team_values.append(trail_team)

    if not gap_values:
        return {
            "lambda": lam,
            "n_seeds": 0,
            "per_seed": per_seed,
            "gap_closed_mean": float("nan"),
            "gap_closed_per_seed": [],
            "team_reward_trailing5_mean": float("nan"),
        }
    return {
        "lambda": lam,
        "n_seeds": len(gap_values),
        "per_seed": per_seed,
        "gap_closed_mean": float(np.mean(gap_values)),
        "gap_closed_std": float(np.std(gap_values, ddof=0)),
        "gap_closed_per_seed": gap_values,
        "team_reward_trailing5_mean": float(np.mean(team_values)),
    }


def classify_tier(best_gap_closed_mean: float) -> tuple[str, str]:
    """Tier verdict per the issue #282 success-criterion table."""
    if np.isnan(best_gap_closed_mean):
        return "no_data", (
            "No λ cell produced loadable metrics — cannot classify. Re-check "
            "the sweep output layout under ``runs/issue282_lambda_sweep/``."
        )
    if best_gap_closed_mean >= TIER_1_THRESHOLD:
        return "tier_1_breaks_plateau", (
            f"Best-cell gap_closed_mean = {best_gap_closed_mean:.3f} ≥ "
            f"{TIER_1_THRESHOLD:.2f}. Bootstrap bias was the misalignment "
            "source. PPO needs Monte Carlo credit on this game. Promotes "
            "issue #285 (BC-init + high-λ) and motivates a production "
            "training-config change."
        )
    if best_gap_closed_mean >= TIER_2_THRESHOLD:
        return "tier_2_partial", (
            f"Best-cell gap_closed_mean = {best_gap_closed_mean:.3f} in "
            f"[{TIER_2_THRESHOLD:.2f}, {TIER_1_THRESHOLD:.2f}). Bootstrap "
            "is contributing but not the sole source. Combine with #285 "
            "or other interventions."
        )
    return "tier_3_insufficient", (
        f"Best-cell gap_closed_mean = {best_gap_closed_mean:.3f} < "
        f"{TIER_2_THRESHOLD:.2f}. Bootstrap is not the binding constraint. "
        "Direct hypothesis pivot to LOLA / potential-based shaping / "
        "hierarchical RL."
    )


def baseline_cross_check(lambda_arms: Dict[float, Dict]) -> Dict:
    """Check whether λ=0.95 reproduces the PR #257 post-#236 baseline."""
    arm = lambda_arms.get(0.95)
    if arm is None or arm.get("n_seeds", 0) == 0:
        return {
            "status": "missing",
            "message": "λ=0.95 cell has no completed seeds; cannot cross-check baseline.",
        }
    observed = arm["gap_closed_mean"]
    delta = observed - BASELINE_GAP_CLOSED
    if abs(delta) <= BASELINE_DRIFT_TOLERANCE:
        return {
            "status": "ok",
            "observed_gap_closed_mean": observed,
            "baseline_gap_closed": BASELINE_GAP_CLOSED,
            "delta": delta,
            "message": (
                f"λ=0.95 gap_closed_mean = {observed:.3f} matches the post-#236 "
                f"baseline ({BASELINE_GAP_CLOSED:.3f}) within ±{BASELINE_DRIFT_TOLERANCE:.2f}."
            ),
        }
    return {
        "status": "drift",
        "observed_gap_closed_mean": observed,
        "baseline_gap_closed": BASELINE_GAP_CLOSED,
        "delta": delta,
        "message": (
            f"λ=0.95 gap_closed_mean = {observed:.3f} drifts from the post-#236 "
            f"baseline ({BASELINE_GAP_CLOSED:.3f}) by {delta:+.3f}; tolerance "
            f"±{BASELINE_DRIFT_TOLERANCE:.2f}. Possible stack regression — "
            "investigate before trusting the high-λ verdict."
        ),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue282_lambda_sweep/minimal_specialization"
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
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue282_lambda_sweep"
        ),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    lambda_arms: Dict[float, Dict] = {}
    for lam in args.lambdas:
        cells = [
            args.runs_root / f"lambda_{_lambda_tag(lam)}" / f"seed_{s}"
            for s in args.seeds
        ]
        lambda_arms[lam] = aggregate_lambda(cells, lam)

    # Best-cell gap_closed_mean over all λ arms with at least one completed seed.
    completed = {
        lam: arm
        for lam, arm in lambda_arms.items()
        if arm.get("n_seeds", 0) > 0
    }
    if completed:
        best_lambda, best_arm = max(
            completed.items(),
            key=lambda kv: kv[1]["gap_closed_mean"],
        )
        best_gc = best_arm["gap_closed_mean"]
    else:
        best_lambda = None
        best_gc = float("nan")

    tier, reasoning = classify_tier(best_gc)
    baseline = baseline_cross_check(lambda_arms)

    out = {
        "issue": 282,
        "tier": tier,
        "reasoning": reasoning,
        "best_lambda": best_lambda,
        "best_gap_closed_mean": best_gc,
        "baseline_cross_check": baseline,
        "lambda_arms": {str(lam): arm for lam, arm in lambda_arms.items()},
        "references": {
            "minspec_random": MINSPEC_RANDOM,
            "minspec_specialist": MINSPEC_SPECIALIST,
            "trailing_n": TRAILING_N,
            "tier_1_threshold": TIER_1_THRESHOLD,
            "tier_2_threshold": TIER_2_THRESHOLD,
            "baseline_gap_closed": BASELINE_GAP_CLOSED,
            "baseline_drift_tolerance": BASELINE_DRIFT_TOLERANCE,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md_lines = [
        "# Issue #282 — high-λ GAE PPO smoke verdict",
        "",
        f"**Tier**: `{tier}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        f"**Best λ**: {best_lambda} (gap_closed_mean = {best_gc:.3f})",
        "",
        "## Per-λ table",
        "",
        "| λ | n_seeds | gap_closed_mean | gap_closed_per_seed | team_reward_trailing5_mean |",
        "|---|---|---|---|---|",
    ]
    for lam in args.lambdas:
        arm = lambda_arms[lam]
        per_seed_str = ", ".join(
            f"{gc:.3f}" for gc in arm.get("gap_closed_per_seed", [])
        )
        md_lines.append(
            f"| {lam} | {arm.get('n_seeds', 0)} | "
            f"{arm.get('gap_closed_mean', float('nan')):.3f} | "
            f"[{per_seed_str}] | "
            f"{arm.get('team_reward_trailing5_mean', float('nan')):.3f} |"
        )

    md_lines.extend(
        [
            "",
            "## Baseline cross-check (λ=0.95 vs post-#236 reference)",
            f"- status: `{baseline['status']}`",
            f"- {baseline['message']}",
            "",
            "## References",
            f"- random baseline (per-step team): {MINSPEC_RANDOM:.2f}",
            f"- specialist baseline (per-step team): {MINSPEC_SPECIALIST:.2f}",
            f"- trailing window: {TRAILING_N} iterations",
            f"- tier 1 threshold (breaks plateau): {TIER_1_THRESHOLD:.2f}",
            f"- tier 2 threshold (partial): {TIER_2_THRESHOLD:.2f}",
            "",
        ]
    )
    (args.output_dir / "verdict.md").write_text("\n".join(md_lines))

    print(f"\ntier: {tier}")
    print(f"best λ: {best_lambda} (gap_closed_mean = {best_gc:.3f})")
    print(f"baseline cross-check: {baseline['status']} — {baseline['message']}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
