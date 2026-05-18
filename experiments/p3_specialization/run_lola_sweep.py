"""LOLA-DiCE sweep driver (issue #287).

Trimmed grid over ``(lola_eta, lola_lookahead_steps, lr, lola_dice, seed)``
on the ``minimal_specialization`` scenario, mirroring the layout of
:mod:`experiments.p3_specialization.run_sweep` but with the LOLA-specific
axes.

The curator-spec full grid is 3 × 2 × 3 × 2 × 3 = **108 cells**; the
default grid below trims to **≤ 50** by spot-checking ``eta`` and ``lr``
sweeps independently rather than crossing them.

**Compute guideline**: do NOT run this locally on a laptop. LOLA's
second-order autograd is CPU-bound and the grid is parallelizable across
hosts. Per ``CLAUDE.md`` Compute Resource Guidelines, dispatch this
script to ``$COMPUTE_HOST_PRIMARY``. The harness simply enumerates cells
and forwards each one to :func:`train_one_cell` — running it locally
will melt your CPU.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List

from experiments.p3_specialization.train import CellConfig, train_one_cell


# --------------------------------------------------------------------------
# Default trimmed grid (≤ 50 cells per curator spec).
# --------------------------------------------------------------------------

# Anchor lr (matches IPPO baseline). The eta sweep is taken AT this lr;
# the lr sweep is taken AT eta = ANCHOR_ETA. This gives two crosses
# instead of one full grid.
ANCHOR_LR = 3e-4
ANCHOR_ETA = 1.0

# Eta sweep: 0.1, 1.0, 10.0 × the actor lr (paper's "η = α" with the
# 0.1× and 10× ablations).
ETA_VALUES = [0.1, 1.0, 10.0]
# LR sweep around the anchor (wider than IPPO baseline; LOLA correction
# scales the effective learning signal — see curator spec).
LR_VALUES = [1e-4, 3e-4, 1e-3]
# Lookahead steps. Paper main = 1; 2 is the standard ablation.
LOOKAHEAD_STEPS = [1, 2]
# DiCE on vs. an explicit-second-order baseline (currently DiCE-only
# implementation; second value present so the schema is preregistered
# for the follow-up if needed). Listed as a single value (True) by
# default to keep the grid trimmed.
DICE_VALUES = [True]
# Seed budget: matches all other p3 experiments (3 seeds per cell).
SEEDS = [42, 43, 44]

# Trimmed grid size = (|ETA| + |LR| - 1 [overlap at anchor]) × |LOOKAHEAD| × |DICE| × |SEEDS|
#                   = (3 + 3 - 1) × 2 × 1 × 3 = 30 cells (well under 50).


def trimmed_grid() -> List[dict]:
    """Build the trimmed (eta × lookahead) ∪ (lr × lookahead) cross.

    Both crosses share the (lookahead, dice, seed) inner loops. The eta
    cross holds lr at ``ANCHOR_LR``; the lr cross holds eta at
    ``ANCHOR_ETA``. The shared cell (eta=ANCHOR_ETA, lr=ANCHOR_LR) is
    emitted once.
    """
    cells: List[dict] = []
    seen: set[tuple] = set()
    for eta in ETA_VALUES:
        for lookahead in LOOKAHEAD_STEPS:
            for dice in DICE_VALUES:
                for seed in SEEDS:
                    key = (ANCHOR_LR, eta, lookahead, dice, seed)
                    if key not in seen:
                        seen.add(key)
                        cells.append(
                            dict(
                                lr=ANCHOR_LR,
                                eta=eta,
                                lookahead=lookahead,
                                dice=dice,
                                seed=seed,
                            )
                        )
    for lr in LR_VALUES:
        for lookahead in LOOKAHEAD_STEPS:
            for dice in DICE_VALUES:
                for seed in SEEDS:
                    key = (lr, ANCHOR_ETA, lookahead, dice, seed)
                    if key not in seen:
                        seen.add(key)
                        cells.append(
                            dict(
                                lr=lr,
                                eta=ANCHOR_ETA,
                                lookahead=lookahead,
                                dice=dice,
                                seed=seed,
                            )
                        )
    return cells


def run_lola_sweep(
    output_root: Path,
    scenario: str,
    cells: List[dict],
    num_iterations: int,
    rollout_steps: int,
    num_agents: int,
    device: str,
    skip_existing: bool,
) -> None:
    n_cells = len(cells)
    print(f"LOLA sweep: {n_cells} cells on scenario={scenario}")
    print(f"Output root: {output_root}")

    t0 = time.time()
    done = 0
    for cell in cells:
        cell_dir = (
            output_root
            / scenario
            / f"lr_{cell['lr']:.0e}"
            / f"eta_{cell['eta']:g}"
            / f"lookahead_{cell['lookahead']}"
            / f"dice_{int(cell['dice'])}"
            / f"seed_{cell['seed']}"
        )
        if skip_existing and (cell_dir / "metrics.json").exists():
            print(f"[{done + 1}/{n_cells}] skip (exists): {cell_dir}")
            done += 1
            continue

        cfg = CellConfig(
            scenario=scenario,
            lambda_red=0.0,
            seed=cell["seed"],
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
            num_agents=num_agents,
            lr=cell["lr"],
            lola_dice=True,  # full set is DiCE-on; explicit-second-order is follow-up.
            lola_eta=cell["eta"],
            lola_lookahead_steps=cell["lookahead"],
            device=device,
        )

        print(f"[{done + 1}/{n_cells}] {cell_dir}")
        cell_t0 = time.time()
        train_one_cell(cfg, cell_dir)
        cell_elapsed = time.time() - cell_t0
        done += 1

        elapsed = time.time() - t0
        eta_s = elapsed / done * (n_cells - done)
        print(f"  cell done in {cell_elapsed:.1f}s | sweep ETA {eta_s / 60:.1f} min")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-root",
        type=Path,
        default=Path("experiments/p3_specialization/runs_lola"),
    )
    p.add_argument(
        "--scenario",
        default="minimal_specialization",
        help="Scenario for the LOLA sweep. Curator spec: primary is "
        "minimal_specialization; default is the harder 10-house scenario.",
    )
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
        "--dry-run",
        action="store_true",
        help="Print the trimmed grid and exit without running any cells.",
    )
    args = p.parse_args()

    cells = trimmed_grid()
    print(f"Trimmed grid: {len(cells)} cells (target <= 50 per curator spec).")
    if args.dry_run:
        for c in cells:
            print(f"  {c}")
        return

    run_lola_sweep(
        output_root=args.output_root,
        scenario=args.scenario,
        cells=cells,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        num_agents=args.num_agents,
        device=args.device,
        skip_existing=args.skip_existing,
    )


if __name__ == "__main__":
    main()
