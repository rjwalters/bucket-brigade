"""Grid driver for the P3 specialization sweep.

Iterates over ``scenario x lambda_red x seed`` and dispatches each cell to
:func:`train_one_cell`. Logs end up under
``experiments/p3_specialization/runs/{scenario}/{lambda}/seed{N}/``.

The full preregistered grid is:

    scenarios:  trivial_cooperation, default, chain_reaction
    lambda_red: 0.0, 1e-3, 1e-2, 1e-1
    seeds:      42..61 (20 seeds)

That is 240 cells; rough estimate on Mac Studio (50 iters of 2048 steps each,
hidden_size=64, 4 PPO epochs) is ~3 min/cell, so the full sweep is ~12 hours
single-threaded. Use ``--num-iterations`` to shorten for smoke tests.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from experiments.p3_specialization.train import CellConfig, train_one_cell


PREREG_SCENARIOS = ["trivial_cooperation", "default", "chain_reaction"]
PREREG_LAMBDAS = [0.0, 1e-3, 1e-2, 1e-1]
PREREG_SEEDS = list(range(42, 62))  # 20 seeds


def run_sweep(
    output_root: Path,
    scenarios: List[str],
    lambdas: List[float],
    seeds: List[int],
    num_iterations: int,
    rollout_steps: int,
    num_agents: int,
    device: str,
    skip_existing: bool,
    value_coef: float = CellConfig.__dataclass_fields__["value_coef"].default,
    entropy_coef: float = CellConfig.__dataclass_fields__["entropy_coef"].default,
    normalize_returns: bool = CellConfig.__dataclass_fields__[
        "normalize_returns"
    ].default,
) -> None:
    n_cells = len(scenarios) * len(lambdas) * len(seeds)
    print(
        f"P3 sweep: {n_cells} cells "
        f"({len(scenarios)} scenarios x {len(lambdas)} lambdas x {len(seeds)} seeds)"
    )
    print(f"Output root: {output_root}")
    print(
        f"PPO coefs: value_coef={value_coef}, entropy_coef={entropy_coef}, "
        f"normalize_returns={normalize_returns}"
    )

    t0 = time.time()
    done = 0
    for scenario in scenarios:
        for lam in lambdas:
            for seed in seeds:
                cell_dir = (
                    output_root
                    / scenario
                    / f"lambda_{lam:.0e}".replace("+0", "").replace("-0", "-")
                    / f"seed_{seed}"
                )
                if skip_existing and (cell_dir / "metrics.json").exists():
                    print(f"[{done + 1}/{n_cells}] skip (exists): {cell_dir}")
                    done += 1
                    continue

                cfg = CellConfig(
                    scenario=scenario,
                    lambda_red=lam,
                    seed=seed,
                    num_iterations=num_iterations,
                    rollout_steps=rollout_steps,
                    num_agents=num_agents,
                    value_coef=value_coef,
                    entropy_coef=entropy_coef,
                    normalize_returns=normalize_returns,
                    device=device,
                )

                print(f"[{done + 1}/{n_cells}] {cell_dir}")
                cell_t0 = time.time()
                train_one_cell(cfg, cell_dir)
                cell_elapsed = time.time() - cell_t0
                done += 1

                elapsed = time.time() - t0
                eta = elapsed / done * (n_cells - done)
                print(
                    f"  cell done in {cell_elapsed:.1f}s | sweep ETA {eta / 60:.1f} min"
                )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-root", type=Path, default=Path("experiments/p3_specialization/runs")
    )
    p.add_argument("--scenarios", nargs="+", default=PREREG_SCENARIOS)
    p.add_argument("--lambdas", nargs="+", type=float, default=PREREG_LAMBDAS)
    p.add_argument("--seeds", nargs="+", type=int, default=PREREG_SEEDS)
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument("--num-agents", type=int, default=4)
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip cells that already have metrics.json on disk.",
    )
    p.add_argument(
        "--value-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["value_coef"].default,
        help=(
            "PPO value-loss weight applied to every cell in the sweep "
            "(default matches CellConfig). Lowered in Phase 2 sweeps to test "
            "the value-loss-dominance hypothesis (issue #153)."
        ),
    )
    p.add_argument(
        "--entropy-coef",
        type=float,
        default=CellConfig.__dataclass_fields__["entropy_coef"].default,
        help=(
            "PPO entropy bonus weight applied to every cell in the sweep "
            "(default matches CellConfig). Raised in Phase 2 sweeps to "
            "prevent entropy collapse (issue #153)."
        ),
    )
    p.add_argument(
        "--normalize-returns",
        action="store_true",
        help=(
            "Issue #159: normalize PPO returns by running std before the "
            "value-loss MSE on every cell in the sweep. Default off preserves "
            "existing behavior; flip on for the 4-cell ablation."
        ),
    )
    args = p.parse_args()

    run_sweep(
        output_root=args.output_root,
        scenarios=args.scenarios,
        lambdas=args.lambdas,
        seeds=args.seeds,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_agents=args.num_agents,
        device=args.device,
        skip_existing=args.skip_existing,
        value_coef=args.value_coef,
        entropy_coef=args.entropy_coef,
        normalize_returns=args.normalize_returns,
    )


if __name__ == "__main__":
    main()
