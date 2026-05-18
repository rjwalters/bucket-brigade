"""Analysis for issue #270 — basin trap vs anti-attractor verdict.

Compares per-iteration team rewards from:

- **BC-init PPO continuation** (this issue): cells under
  ``experiments/p3_specialization/runs/issue270_bc_continuation/minimal_specialization/lambda_0e0/seed_{42,43,44}/``.
- **Random-init IPPO baseline** (PR #257-ish equivalent): cells under
  ``experiments/p3_specialization/runs/issue231/ippo/minimal_specialization/seed_{42,43,44}/``.

Emits a markdown summary + JSON to
``experiments/p3_specialization/diagnostics/results/issue270_bc_continuation/`` and
applies the verdict matrix from the issue body:

| BC-init PPO trajectory | Verdict |
|---|---|
| gap_closed > 0.5 throughout the 50 iters | basin trap (specialist stable) |
| gap_closed decays back toward random (≤ 0.2 by iter 49) | anti-attractor |
| gap_closed sits above baseline but below 0.5 | partial basin |

Operationalized thresholds (see body of ``classify_verdict``):

- ``basin_trap`` ⟺ trailing-5 gap_closed (mean over seeds) > 0.5 AND
  min over iterations of mean gap_closed > 0.3 (i.e., no deep dip).
- ``anti_attractor`` ⟺ trailing-5 gap_closed < 0.2 AND iter-0 gap_closed > 0.5
  (BC took, then PPO erased it).
- ``partial`` ⟺ anything in between.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

# Canonical per-step references imported from ``bucket_brigade.baselines``
# (issue #293). See that module's docstring for derivation provenance.
TRAILING_N = 5


def gap_closed(per_step_team: float) -> float:
    return (per_step_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def _load_metrics(path: Path) -> Optional[List[dict]]:
    f = path / "metrics.json"
    if not f.exists():
        return None
    return json.loads(f.read_text())


def _trajectory(metrics: List[dict]) -> np.ndarray:
    """Per-iteration mean_step_reward_team as a 1-D array."""
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
                "iter0_team": float(traj[0]),
                "iter0_gap_closed": float(gap_closed(float(traj[0]))),
                "trailing5_team": float(traj[-TRAILING_N:].mean()),
                "trailing5_gap_closed": float(
                    gap_closed(float(traj[-TRAILING_N:].mean()))
                ),
                "min_iter_team": float(traj.min()),
            }
        )
        trajectories.append(traj)

    if not trajectories:
        return {"label": label, "seeds": seeds_data, "n_seeds": 0}
    # Align by min length so heterogeneous seed runs don't crash the analysis.
    min_len = min(t.size for t in trajectories)
    stacked = np.stack([t[:min_len] for t in trajectories], axis=0)  # [S, T]
    mean_traj_step = stacked.mean(axis=0)  # mean per-step team reward per iter
    mean_traj_gc = (mean_traj_step - MINSPEC_RANDOM) / (
        MINSPEC_SPECIALIST - MINSPEC_RANDOM
    )

    return {
        "label": label,
        "seeds": seeds_data,
        "n_seeds": len(trajectories),
        "n_iters": int(min_len),
        "iter0_team_mean": float(mean_traj_step[0]),
        "iter0_gap_closed_mean": float(mean_traj_gc[0]),
        "trailing5_team_mean": float(mean_traj_step[-TRAILING_N:].mean()),
        "trailing5_gap_closed_mean": float(mean_traj_gc[-TRAILING_N:].mean()),
        "min_iter_gap_closed_mean": float(mean_traj_gc.min()),
        "mean_traj_step_reward": mean_traj_step.tolist(),
        "mean_traj_gap_closed": mean_traj_gc.tolist(),
    }


def classify_verdict(bc_arm: Dict) -> Tuple[str, str]:
    """Apply the verdict matrix to the BC-init arm. Returns (label, reasoning)."""
    if bc_arm.get("n_seeds", 0) == 0:
        return "no_data", "BC-init arm has no completed seeds"
    iter0_gc = bc_arm["iter0_gap_closed_mean"]
    trail_gc = bc_arm["trailing5_gap_closed_mean"]
    min_gc = bc_arm["min_iter_gap_closed_mean"]
    if iter0_gc < 0.5:
        return "bc_did_not_take", (
            f"BC eval at iter 0 only closed {iter0_gc:.3f} of the gap "
            "(< 0.5). The probe is uninterpretable — BC pre-training failed "
            "to land in the specialist region. Re-run BC with more demos, "
            "more epochs, or a larger hidden size, or characterize "
            "architecture first (issue #272)."
        )
    if trail_gc > 0.5 and min_gc > 0.3:
        return "basin_trap", (
            f"trailing-5 gap_closed = {trail_gc:.3f} > 0.5 AND min over "
            f"iters {min_gc:.3f} > 0.3. PPO holds the specialist region. "
            "Specialist is stable under PPO updates and unreachable from "
            "random init — implies path-finding intervention (PBT, "
            "neuroevolution, BC-init as production technique)."
        )
    if trail_gc < 0.2 and iter0_gc > 0.5:
        return "anti_attractor", (
            f"trailing-5 gap_closed = {trail_gc:.3f} < 0.2 AND iter0 gap_closed = "
            f"{iter0_gc:.3f} > 0.5. BC took but PPO actively erased it. "
            "Something in the reward/dynamics repels specialist-like behavior — "
            "examine reward gradient structure near specialist."
        )
    return "partial", (
        f"trailing-5 gap_closed = {trail_gc:.3f} sits between 0.2 and 0.5 "
        f"(or showed a dip: min={min_gc:.3f}). PPO partially holds "
        "specialist; local minimum nearby. Hybrid intervention: BC-init + "
        "mild reward shaping."
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--bc-runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue270_bc_continuation/minimal_specialization/lambda_0e0"
        ),
        help="Root containing seed_{42,43,44} cells from the BC-init PPO continuation.",
    )
    p.add_argument(
        "--baseline-runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue231/ippo/minimal_specialization"
        ),
        help="Root containing seed_{42,43,44} cells from the random-init IPPO baseline.",
    )
    p.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "experiments/p3_specialization/diagnostics/results/issue270_bc_continuation"
        ),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    bc_cells = [args.bc_runs_root / f"seed_{s}" for s in args.seeds]
    baseline_cells = [args.baseline_runs_root / f"seed_{s}" for s in args.seeds]

    bc_arm = aggregate_arm(bc_cells, "bc_init_ppo")
    baseline_arm = aggregate_arm(baseline_cells, "random_init_ippo")

    verdict, reasoning = classify_verdict(bc_arm)

    out = {
        "issue": 270,
        "verdict": verdict,
        "reasoning": reasoning,
        "bc_init_arm": bc_arm,
        "random_init_arm": baseline_arm,
        "references": {
            "minspec_random": MINSPEC_RANDOM,
            "minspec_specialist": MINSPEC_SPECIALIST,
            "trailing_n": TRAILING_N,
        },
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Issue #270 — BC-init then PPO: basin trap vs anti-attractor verdict",
        "",
        f"**Verdict**: `{verdict}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        "## BC-init PPO arm",
        f"- n_seeds: {bc_arm.get('n_seeds', 0)}",
        f"- iter 0 gap_closed (BC-only): {bc_arm.get('iter0_gap_closed_mean', float('nan')):.3f}",
        f"- trailing-5 gap_closed (after PPO continuation): {bc_arm.get('trailing5_gap_closed_mean', float('nan')):.3f}",
        f"- min over iters gap_closed: {bc_arm.get('min_iter_gap_closed_mean', float('nan')):.3f}",
        "",
        "## Random-init IPPO baseline",
        f"- n_seeds: {baseline_arm.get('n_seeds', 0)}",
        f"- iter 0 gap_closed: {baseline_arm.get('iter0_gap_closed_mean', float('nan')):.3f}",
        f"- trailing-5 gap_closed: {baseline_arm.get('trailing5_gap_closed_mean', float('nan')):.3f}",
        "",
        "## References",
        f"- random baseline: {MINSPEC_RANDOM:.2f}",
        f"- specialist baseline: {MINSPEC_SPECIALIST:.2f}",
    ]
    (args.output_dir / "verdict.md").write_text("\n".join(md))

    print(f"\nverdict: {verdict}")
    print(f"reasoning: {reasoning}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
