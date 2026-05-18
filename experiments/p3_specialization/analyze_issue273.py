"""Verdict analyzer for issue #273 (REINFORCE off-PPO baseline).

Reads the ``metrics.json`` files produced by
``run_issue273_reinforce_sweep.sh`` and emits the 3-way verdict table:

    | Off-PPO outcome      | Interpretation                                |
    |----------------------|-----------------------------------------------|
    | Same plateau as PPO  | RL-general failure → env-side fixes the lever |
    | REINFORCE > PPO      | PPO clip / GAE is hurting → revisit knobs     |
    | REINFORCE < PPO      | PPO mitigates real variance → orthogonal axes |

Compares converged mean step-team reward (last 10% of iterations) against
the IPPO baseline (PR #257, ~0.182 gap_closed on
``minimal_specialization`` at 50 iterations × 2048 rollout_steps).

Usage::

    uv run python -m experiments.p3_specialization.analyze_issue273 \\
        --reinforce-root experiments/p3_specialization/runs_reinforce \\
        --ippo-baseline 0.182
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple


def _load_metrics(metrics_path: Path) -> List[Dict[str, float]]:
    with metrics_path.open() as fh:
        return json.load(fh)


def _converged_mean_reward(metrics: List[Dict[str, float]], tail_frac: float) -> float:
    """Mean team step reward across the last ``tail_frac`` of iterations."""
    if not metrics:
        return float("nan")
    n = len(metrics)
    tail = max(1, int(round(n * tail_frac)))
    rewards = [
        float(m.get("mean_step_reward_team", float("nan"))) for m in metrics[-tail:]
    ]
    rewards = [r for r in rewards if r == r]  # drop NaNs
    if not rewards:
        return float("nan")
    return sum(rewards) / len(rewards)


def collect_cells(root: Path, tail_frac: float) -> List[Dict[str, object]]:
    """Walk ``root/<scenario>/norm_<bool>/seed_<int>/metrics.json`` cells."""
    cells: List[Dict[str, object]] = []
    for scenario_dir in sorted(root.iterdir()) if root.exists() else []:
        if not scenario_dir.is_dir():
            continue
        for norm_dir in sorted(scenario_dir.iterdir()):
            if not norm_dir.is_dir() or not norm_dir.name.startswith("norm_"):
                continue
            normalize = norm_dir.name.split("_", 1)[1] == "true"
            for seed_dir in sorted(norm_dir.iterdir()):
                if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                    continue
                metrics_path = seed_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                try:
                    seed = int(seed_dir.name.split("_", 1)[1])
                except ValueError:
                    continue
                metrics = _load_metrics(metrics_path)
                cells.append(
                    dict(
                        scenario=scenario_dir.name,
                        normalize=normalize,
                        seed=seed,
                        converged_reward=_converged_mean_reward(metrics, tail_frac),
                        num_iterations=len(metrics),
                        path=str(seed_dir),
                    )
                )
    return cells


def _verdict(
    reinforce_mean: float, ippo_baseline: float, tolerance: float
) -> Tuple[str, str]:
    """Return a (verdict, interpretation) tuple."""
    delta = reinforce_mean - ippo_baseline
    if abs(delta) <= tolerance:
        return (
            "same_plateau",
            "Failure is RL-general, not PPO-specific. The plateau is a "
            "structural / representation / game-theoretic property of the "
            "env. Justifies doubling down on env-side fixes (CTDE / "
            "intrinsic-reward interventions are the right axis); "
            "deprioritize PPO-knob tuning.",
        )
    if delta > tolerance:
        return (
            "reinforce_better",
            "PPO's clip-ratio / GAE is hurting specifically. Re-examine "
            "clip-ratio annealing, GAE λ extremes, ppo_epochs=1, num "
            "minibatches. Possibly explains why MAPPO didn't help — same "
            "PPO core.",
        )
    return (
        "reinforce_worse",
        "PPO is mitigating something real (high variance, off-policy "
        "correction). The plateau is genuinely 'best PPO can do here,' "
        "but PPO is necessary. Search for orthogonal fixes (env, reward, "
        "init).",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--reinforce-root",
        type=Path,
        default=Path("experiments/p3_specialization/runs_reinforce"),
        help="Root directory containing the REINFORCE sweep output cells.",
    )
    p.add_argument(
        "--ippo-baseline",
        type=float,
        default=0.182,
        help=(
            "IPPO baseline mean step-team reward (PR #257 verdict on "
            "minimal_specialization, ~0.182). Override if comparing "
            "against a different scenario or run."
        ),
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=0.02,
        help=(
            "Reward delta within which REINFORCE and PPO are treated as "
            "'same plateau' for the verdict. Default 0.02 (≈10% of the "
            "IPPO baseline)."
        ),
    )
    p.add_argument(
        "--tail-frac",
        type=float,
        default=0.1,
        help="Fraction of trailing iterations used to compute converged mean.",
    )
    args = p.parse_args()

    cells = collect_cells(args.reinforce_root, args.tail_frac)
    if not cells:
        print(f"No cells found under {args.reinforce_root}.")
        print(
            "Run experiments/p3_specialization/run_issue273_reinforce_sweep.sh first."
        )
        return

    # Per-cell table.
    print(f"{'normalize':>10}  {'seed':>4}  {'iters':>6}  {'converged_reward':>18}")
    for cell in cells:
        print(
            f"{str(cell['normalize']):>10}  "
            f"{cell['seed']:>4}  "
            f"{cell['num_iterations']:>6}  "
            f"{cell['converged_reward']:>18.4f}"
        )
    print()

    # Per-normalize aggregates.
    print("Aggregates:")
    for normalize in (False, True):
        rewards = [
            float(c["converged_reward"])
            for c in cells
            if c["normalize"] == normalize
            and c["converged_reward"] == c["converged_reward"]
        ]
        if not rewards:
            continue
        mean = sum(rewards) / len(rewards)
        # Sample std.
        if len(rewards) > 1:
            mean_sq = sum((r - mean) ** 2 for r in rewards) / (len(rewards) - 1)
            std = mean_sq**0.5
        else:
            std = float("nan")
        print(
            f"  normalize={str(normalize):<5}  n={len(rewards):>2}  "
            f"mean={mean:>8.4f}  std={std:>8.4f}"
        )
    print()

    # Verdict (uses the best of the two normalize settings as the
    # REINFORCE mean — the sweep is meant to find the better of the
    # two; if one collapses we should not penalize REINFORCE for it).
    all_rewards = [
        float(c["converged_reward"])
        for c in cells
        if c["converged_reward"] == c["converged_reward"]
    ]
    if not all_rewards:
        print("No valid rewards to compute verdict.")
        return
    reinforce_best = max(all_rewards)
    reinforce_mean = sum(all_rewards) / len(all_rewards)

    print(f"IPPO baseline (PR #257):    {args.ippo_baseline:.4f}")
    print(f"REINFORCE mean (all cells): {reinforce_mean:.4f}")
    print(f"REINFORCE best:             {reinforce_best:.4f}")
    print()

    verdict, interpretation = _verdict(
        reinforce_mean, args.ippo_baseline, args.tolerance
    )
    print(f"VERDICT: {verdict}")
    print(f"  delta (REINFORCE - IPPO) = {reinforce_mean - args.ippo_baseline:+.4f}")
    print(f"  tolerance                = ±{args.tolerance:.4f}")
    print()
    print(f"  {interpretation}")


if __name__ == "__main__":
    main()
