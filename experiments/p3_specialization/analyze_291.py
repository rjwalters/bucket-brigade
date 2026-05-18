"""Analysis for issue #291 — multi-agent vs horizon basin-trap verdict.

Compares per-iteration team rewards from:

- **Joint-control single-controller PPO** (this issue): cells under
  ``experiments/p3_specialization/runs/issue291_joint_control/minimal_specialization/seed_{42,43,44}/``.
- **Random-init IPPO baseline** (#270/#271 reference): cells under
  ``experiments/p3_specialization/runs/issue231/ippo/minimal_specialization/seed_{42,43,44}/``.

Emits a markdown summary + JSON to
``experiments/p3_specialization/diagnostics/results/issue291_joint_control/``
and applies the H0/H1 verdict matrix from the curator-enhanced issue body:

| trailing-5 gap_closed (joint-control) | Verdict |
|---|---|
| >= 0.5 | H0 — basin trap is multi-agent-specific (credit assignment).
|        | Paper scope narrows to cooperative MARL.
| <= 0.25 | H1 — basin trap is horizon-based, survives single-controller
|         | reduction. Paper scope expands to long-horizon RL generally.
| in between | undecided / mixed — horizon contributes, multi-agent compounds.

NB: the curator-enhanced thresholds require K=200 re-evaluation of the
top checkpoint per seed, not the phase-1 training-time estimates. The
``--k200-reeval`` flag is a stability gate; this analyzer asserts it
either ran (presence of ``k200_gap_closed`` in the metrics.json) or is
explicitly disabled (``--allow-phase1-only``). Phase-1 numbers drifted
-46 points in #271 — do NOT report them as the verdict.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Hardcoded references (mirror analyze_270.py:39-40 — same scenario, so
# the same random/specialist baselines apply).
MINSPEC_RANDOM = -96.07
MINSPEC_SPECIALIST = -22.07
TRAILING_N = 5

# Verdict thresholds (curator-enhanced issue body):
#   gap_closed >= H0_THRESH        -> H0  (multi-agent-specific)
#   gap_closed <= H1_THRESH        -> H1  (horizon-based)
#   in between                     -> undecided
H0_THRESH = 0.5
H1_THRESH = 0.25


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
                "max_iter_team": float(traj.max()),
            }
        )
        trajectories.append(traj)

    if not trajectories:
        return {"label": label, "seeds": seeds_data, "n_seeds": 0}
    min_len = min(t.size for t in trajectories)
    stacked = np.stack([t[:min_len] for t in trajectories], axis=0)  # [S, T]
    mean_traj_step = stacked.mean(axis=0)
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
        "max_iter_gap_closed_mean": float(mean_traj_gc.max()),
        "mean_traj_step_reward": mean_traj_step.tolist(),
        "mean_traj_gap_closed": mean_traj_gc.tolist(),
    }


def classify_verdict(joint_arm: Dict) -> Tuple[str, str]:
    """Apply the curator-enhanced H0/H1 verdict matrix.

    Returns ``(label, reasoning)``. Labels:

    - ``H0_multi_agent_specific``: joint-control PPO succeeds on
      ``minimal_specialization`` (trailing-5 gap_closed >= 0.5). Paper scope
      narrows to cooperative MARL.
    - ``H1_horizon_based``: joint-control PPO also plateaus (trailing-5
      gap_closed <= 0.25). Paper scope expands.
    - ``undecided``: trailing-5 sits between the H0 and H1 thresholds. Mixed
      cause — horizon contributes, multi-agent compounds.
    """
    if joint_arm.get("n_seeds", 0) == 0:
        return "no_data", "joint-control arm has no completed seeds"
    trail_gc = joint_arm["trailing5_gap_closed_mean"]
    if trail_gc >= H0_THRESH:
        return "H0_multi_agent_specific", (
            f"trailing-5 gap_closed = {trail_gc:.3f} >= {H0_THRESH:.2f}. "
            "Single-controller joint-action PPO closes >= half of the "
            "random->specialist gap, while IPPO plateaus near 0.18 (#270/#271). "
            "The basin trap is therefore tied to multi-agent credit "
            "assignment, not to the long-horizon reward structure. Paper "
            "scope narrows to cooperative MARL."
        )
    if trail_gc <= H1_THRESH:
        return "H1_horizon_based", (
            f"trailing-5 gap_closed = {trail_gc:.3f} <= {H1_THRESH:.2f}. "
            "Single-controller joint-action PPO is also trapped in the "
            "greedy basin — removing the multi-agent decomposition does "
            "not rescue cooperation. The basin trap is therefore a "
            "general-RL long-horizon phenomenon, connecting directly to "
            "the specification-gaming literature. Paper scope expands."
        )
    return "undecided", (
        f"trailing-5 gap_closed = {trail_gc:.3f} sits between H1 "
        f"({H1_THRESH:.2f}) and H0 ({H0_THRESH:.2f}). Joint-control helps "
        "partially but does not solve the basin trap. Horizon contributes, "
        "multi-agent compounds. Document and discuss in the paper as "
        "'credit assignment is a partial cause.'"
    )


def _enforce_k200_gate(joint_arm: Dict, allow_phase1_only: bool) -> Optional[str]:
    """Verdict-gate check: require K=200 re-eval before reporting verdict.

    Phase-1 estimates drifted -46 points in #271. The curator-enhanced
    success criterion requires the top checkpoint per seed to be
    re-evaluated for K=200 episodes before the gap_closed verdict is
    counted. This analyzer cannot run the re-eval (it only consumes
    training-time metrics.json) — but it asserts that one of these holds:

    - Every per-seed entry has a ``k200_gap_closed`` field (provided by
      a separate re-eval pass that writes back into metrics.json).
    - The caller explicitly opted out via ``--allow-phase1-only``.

    Returns ``None`` if the gate passes; a string explaining the failure
    otherwise.
    """
    if allow_phase1_only:
        return None
    missing = [
        s.get("cell", "?") for s in joint_arm.get("seeds", [])
        if not s.get("missing", False) and "k200_gap_closed" not in s
    ]
    if missing:
        return (
            "K=200 re-eval gate failed: no `k200_gap_closed` field on "
            f"{len(missing)} seed cells. The curator-enhanced success "
            "criterion requires top-checkpoint K=200 re-evaluation before "
            "any verdict is reported (#271 showed phase-1 estimates drift "
            "-46 points). Run the K=200 re-eval (separate script) first, "
            "or pass --allow-phase1-only to explicitly bypass the gate "
            "(numbers should NOT be reported as the result in that mode)."
        )
    return None


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--joint-runs-root",
        type=Path,
        default=Path(
            "experiments/p3_specialization/runs/issue291_joint_control/minimal_specialization"
        ),
        help="Root containing seed_{42,43,44} cells from the joint-control PPO sweep.",
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
            "experiments/p3_specialization/diagnostics/results/issue291_joint_control"
        ),
    )
    p.add_argument(
        "--allow-phase1-only",
        action="store_true",
        help=(
            "Bypass the K=200 stability-gate check. The training-time "
            "phase-1 gap_closed should NOT be reported as the final "
            "verdict — #271 drifted -46 points between phase-1 and K=200. "
            "Provided for development/iteration only."
        ),
    )
    args = p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    joint_cells = [args.joint_runs_root / f"seed_{s}" for s in args.seeds]
    baseline_cells = [args.baseline_runs_root / f"seed_{s}" for s in args.seeds]

    joint_arm = aggregate_arm(joint_cells, "joint_control_ppo")
    baseline_arm = aggregate_arm(baseline_cells, "random_init_ippo")

    gate_failure = _enforce_k200_gate(joint_arm, args.allow_phase1_only)
    if gate_failure is not None:
        verdict, reasoning = "gate_failed", gate_failure
    else:
        verdict, reasoning = classify_verdict(joint_arm)

    out = {
        "issue": 291,
        "verdict": verdict,
        "reasoning": reasoning,
        "joint_control_arm": joint_arm,
        "random_init_arm": baseline_arm,
        "thresholds": {"H0": H0_THRESH, "H1": H1_THRESH},
        "references": {
            "minspec_random": MINSPEC_RANDOM,
            "minspec_specialist": MINSPEC_SPECIALIST,
            "trailing_n": TRAILING_N,
        },
        "k200_gate_bypassed": bool(args.allow_phase1_only),
    }
    (args.output_dir / "analysis.json").write_text(json.dumps(out, indent=2))

    md = [
        "# Issue #291 — single-controller (joint-action) basin-trap verdict",
        "",
        f"**Verdict**: `{verdict}`",
        "",
        f"**Reasoning**: {reasoning}",
        "",
        "## Joint-control PPO arm (this issue)",
        f"- n_seeds: {joint_arm.get('n_seeds', 0)}",
        f"- iter 0 gap_closed: {joint_arm.get('iter0_gap_closed_mean', float('nan')):.3f}",
        f"- trailing-5 gap_closed: {joint_arm.get('trailing5_gap_closed_mean', float('nan')):.3f}",
        f"- max-iter gap_closed: {joint_arm.get('max_iter_gap_closed_mean', float('nan')):.3f}",
        "",
        "## Random-init IPPO baseline (#270/#271)",
        f"- n_seeds: {baseline_arm.get('n_seeds', 0)}",
        f"- trailing-5 gap_closed: {baseline_arm.get('trailing5_gap_closed_mean', float('nan')):.3f}",
        "",
        "## Thresholds",
        f"- H0 (multi-agent-specific): trailing-5 gap_closed >= {H0_THRESH:.2f}",
        f"- H1 (horizon-based): trailing-5 gap_closed <= {H1_THRESH:.2f}",
        "",
        "## References",
        f"- random baseline: {MINSPEC_RANDOM:.2f}",
        f"- specialist baseline: {MINSPEC_SPECIALIST:.2f}",
        f"- K=200 stability gate bypassed: {bool(args.allow_phase1_only)}",
    ]
    (args.output_dir / "verdict.md").write_text("\n".join(md))

    print(f"\nverdict: {verdict}")
    print(f"reasoning: {reasoning}")
    print(f"\nartifacts written to {args.output_dir}/")


if __name__ == "__main__":
    main()
