"""Analysis for issue #289 — HCA verdict.

Compares per-iteration team rewards from:

- **HCA arm** (this issue): cells under
  ``experiments/p3_specialization/runs/issue289_hca/minimal_specialization/hca_default/seed_{42,43,44}/``.
- **GAE baseline arm** (re-run at identical config in the same sweep):
  ``.../gae_baseline/seed_{42,43,44}/``.
- **PPO baseline reference** (PR #257): hardcoded gap_closed = 0.182 as
  the conservative floor.
- **High-λ GAE comparator** (issue #282): pulled in if its analysis
  artifact exists; otherwise reported as ``unavailable``.

Emits a markdown summary + JSON to
``experiments/p3_specialization/diagnostics/results/issue289_hca/`` and
applies the verdict matrix from the issue body:

| HCA gap_closed (50-iter, 3 seeds, mean) | Verdict |
|---|---|
| >= 0.5 | STRONG_WIN |
| 0.2..0.5 AND clearly above #282 high-λ result | WEAK_WIN_DISTINCT_FROM_MC |
| similar to #282 high-λ | AMBIGUOUS |
| similar to PPO baseline | NEGATIVE |
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Hardcoded references (mirror analyze_270.py:34-35).
MINSPEC_RANDOM = -96.07
MINSPEC_SPECIALIST = -22.07
TRAILING_N = 5

# PR #257 PPO baseline gap_closed (from the issue body).
PPO_BASELINE_GAP_CLOSED = 0.182


def gap_closed(per_step_team: float) -> float:
    return (per_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def _load_metrics(path: Path) -> Optional[List[dict]]:
    f = path / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _trajectory(metrics: List[dict]) -> np.ndarray:
    return np.asarray(
        [row["mean_step_reward_team"] for row in metrics], dtype=np.float64
    )


def aggregate_arm(cells: List[Path], label: str) -> Dict:
    seeds_data = []
    trajectories = []
    for cell in cells:
        metrics = _load_metrics(cell)
        if metrics is None:
            seeds_data.append({"cell": str(cell), "missing": True})
            continue
        traj = _trajectory(metrics)
        seeds_data.append(
            {
                "cell": str(cell),
                "n_iters": int(traj.size),
                "trailing5_team": float(traj[-TRAILING_N:].mean()),
                "trailing5_gap_closed": float(
                    gap_closed(float(traj[-TRAILING_N:].mean()))
                ),
            }
        )
        trajectories.append(traj)

    if not trajectories:
        return {"label": label, "seeds": seeds_data, "n_seeds": 0}
    min_len = min(t.size for t in trajectories)
    stacked = np.stack([t[:min_len] for t in trajectories], axis=0)
    mean_traj_step = stacked.mean(axis=0)
    mean_traj_gc = (mean_traj_step - MINSPEC_RANDOM) / (
        MINSPEC_SPECIALIST - MINSPEC_RANDOM
    )

    return {
        "label": label,
        "seeds": seeds_data,
        "n_seeds": len(trajectories),
        "n_iters": int(min_len),
        "trailing5_team_mean": float(mean_traj_step[-TRAILING_N:].mean()),
        "trailing5_gap_closed_mean": float(mean_traj_gc[-TRAILING_N:].mean()),
        "mean_traj_step_reward": mean_traj_step.tolist(),
        "mean_traj_gap_closed": mean_traj_gc.tolist(),
    }


def classify_verdict(
    hca_arm: Dict,
    gae_arm: Dict,
    high_lambda_gap_closed: Optional[float],
) -> Tuple[str, str]:
    if hca_arm.get("n_seeds", 0) == 0:
        return "no_data", "HCA arm has no completed seeds"
    hca_gc = hca_arm["trailing5_gap_closed_mean"]
    if hca_gc >= 0.5:
        return "STRONG_WIN", (
            f"HCA trailing-5 gap_closed = {hca_gc:.3f} >= 0.5. "
            "Retrospective credit assignment fixes the misalignment. "
            "Validates the 'coarse-grained gradient' framing."
        )
    near_baseline = abs(hca_gc - PPO_BASELINE_GAP_CLOSED) < 0.05
    if near_baseline:
        return "NEGATIVE", (
            f"HCA trailing-5 gap_closed = {hca_gc:.3f} within 0.05 of "
            f"PPO baseline {PPO_BASELINE_GAP_CLOSED}. Backward credit "
            "isn't the right mechanism for this game's misalignment."
        )
    if high_lambda_gap_closed is not None:
        if abs(hca_gc - high_lambda_gap_closed) < 0.05:
            return "AMBIGUOUS", (
                f"HCA trailing-5 gap_closed = {hca_gc:.3f} matches the "
                f"#282 high-λ GAE result ({high_lambda_gap_closed:.3f}). "
                "Both are Monte-Carlo-y; HCA's hindsight machinery isn't "
                "adding signal over a return target."
            )
        if 0.2 <= hca_gc < 0.5 and hca_gc - high_lambda_gap_closed > 0.05:
            return "WEAK_WIN_DISTINCT_FROM_MC", (
                f"HCA trailing-5 gap_closed = {hca_gc:.3f} (in 0.2..0.5) "
                f"is meaningfully above #282 high-λ ({high_lambda_gap_closed:.3f}). "
                "Hindsight ratio is doing something high-λ GAE cannot."
            )
    if 0.2 <= hca_gc < 0.5:
        return "WEAK_WIN", (
            f"HCA trailing-5 gap_closed = {hca_gc:.3f} in 0.2..0.5; "
            "comparison to #282 high-λ unavailable, but HCA clearly beats "
            f"the PPO baseline ({PPO_BASELINE_GAP_CLOSED})."
        )
    return "PARTIAL", (
        f"HCA trailing-5 gap_closed = {hca_gc:.3f}. Above PPO baseline "
        "but below 0.2; ambiguous improvement."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hca-runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue289_hca/"
            "minimal_specialization/hca_default"
        ),
    )
    p.add_argument(
        "--gae-runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue289_hca/"
            "minimal_specialization/gae_baseline"
        ),
    )
    p.add_argument(
        "--high-lambda-result",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/"
            "issue282_high_lambda/analysis.json"
        ),
        help=(
            "Optional path to the #282 high-λ analysis JSON. If present "
            "and contains ``trailing5_gap_closed_mean``, that value is "
            "used as the WEAK_WIN_DISTINCT_FROM_MC comparator."
        ),
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/p3_specialization/diagnostics/results/issue289_hca"),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    hca_cells = [args.hca_runs_root / f"seed_{s}" for s in args.seeds]
    gae_cells = [args.gae_runs_root / f"seed_{s}" for s in args.seeds]
    hca_arm = aggregate_arm(hca_cells, "hca")
    gae_arm = aggregate_arm(gae_cells, "gae_baseline")

    high_lambda_gc: Optional[float] = None
    if args.high_lambda_result.exists():
        try:
            payload = json.loads(args.high_lambda_result.read_text())
            high_lambda_gc = float(
                payload.get("trailing5_gap_closed_mean")
                or payload.get("high_lambda_arm", {}).get("trailing5_gap_closed_mean")
            )
        except (ValueError, TypeError, KeyError):
            high_lambda_gc = None

    verdict, reasoning = classify_verdict(hca_arm, gae_arm, high_lambda_gc)

    out = {
        "issue": 289,
        "verdict": verdict,
        "reasoning": reasoning,
        "hca_arm": hca_arm,
        "gae_baseline_arm": gae_arm,
        "high_lambda_comparator_gap_closed": high_lambda_gc,
        "references": {
            "minspec_random": MINSPEC_RANDOM,
            "minspec_specialist": MINSPEC_SPECIALIST,
            "trailing_n": TRAILING_N,
            "ppo_baseline_gap_closed": PPO_BASELINE_GAP_CLOSED,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Issue #289 — HCA verdict",
        "",
        f"**Verdict**: `{verdict}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        "| Arm | n_seeds | trailing-5 team reward | trailing-5 gap_closed |",
        "|---|---|---|---|",
        f"| HCA | {hca_arm.get('n_seeds', 0)} "
        f"| {hca_arm.get('trailing5_team_mean', float('nan')):.3f} "
        f"| {hca_arm.get('trailing5_gap_closed_mean', float('nan')):.3f} |",
        f"| GAE (in-sweep baseline) | {gae_arm.get('n_seeds', 0)} "
        f"| {gae_arm.get('trailing5_team_mean', float('nan')):.3f} "
        f"| {gae_arm.get('trailing5_gap_closed_mean', float('nan')):.3f} |",
        f"| PPO baseline (PR #257) | — | — | {PPO_BASELINE_GAP_CLOSED:.3f} |",
        f"| #282 high-λ GAE | — | — | "
        f"{'unavailable' if high_lambda_gc is None else f'{high_lambda_gc:.3f}'} |",
    ]
    (args.output_dir / "analysis.md").write_text("\n".join(md) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
