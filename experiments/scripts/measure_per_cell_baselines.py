"""Sweep driver for per-cell baseline calibration (issue #413).

Iterates the phase-diagram (β, κ, c) cells from a results.json (default:
``experiments/nash/phase_diagram/results.json``) and, for each cell,
measures three per-step team-reward baselines:

* ``random_baseline``: 4 uniform-random agents.
* ``specialist_homogeneous``: :class:`SpecialistPolicy` ×4 (the
  apples-to-apples per-cell drop-in for ``MINSPEC_SPECIALIST = -28.38``;
  this driver was the source of the n=10k re-derivation under issue #416).
* ``specialist_ne``: 1×Hero + 3×Firefighter heterogeneous-NE profile from
  the phase-diagram DO search (per-cell genomes under
  ``bucket_brigade/baselines/release/local/nash/phase_diagram/<tag>.json``).
  Cells with no NE-genome file (no-convergence cells in the DO search)
  emit ``null`` for ``specialist_ne`` and a top-level
  ``missing_ne_genomes`` note.

The canonical calibration cell — where the ``MINSPEC_RANDOM`` and
``MINSPEC_SPECIALIST`` constants in ``bucket_brigade.baselines`` were
measured — is the base ``minimal_specialization`` scenario at β=0.25,
κ=0.5, c=0.5 (see provenance comment in
``bucket_brigade/baselines/__init__.py:53-59``). That point is NOT one of
the 7 phase-diagram grid cells, so this driver explicitly measures it as
an 8th row tagged ``canonical_b0.25_k0.50_c0.50`` so the
``test_homogeneous_matches_minspec_at_canonical_cell`` acceptance check
has a row to assert against.

Output schema (per cell row):

    {
      "cell_tag": "b0.50_k0.90_c0.50",
      "params": {"beta": 0.5, "kappa": 0.9, "cost_to_work": 0.5},
      "random_baseline":        {"mean": ..., "ci95_lo": ..., "ci95_hi": ..., "n_episodes": 10000},
      "specialist_homogeneous": {"mean": ..., "ci95_lo": ..., "ci95_hi": ..., "n_episodes": 10000},
      "specialist_ne":          {"mean": ..., "ci95_lo": ..., "ci95_hi": ..., "n_episodes": 10000},
      "ne_genomes_path":        "bucket_brigade/baselines/release/local/nash/phase_diagram/b0.50_k0.90_c0.50.json",
      "ne_profile_label":       "hero | firefighter | firefighter | firefighter"
    }

Top-level:

    {"cells": [...], "metadata": {"n_episodes": 10000, "git_sha": "<sha>", "measured_at": "<iso>", "base_scenario": "minimal_specialization"}}

Compute placement
-----------------
7 phase-diagram cells + 1 canonical cell = 8 cells × 3 policies × 10k
episodes ≈ 40 min wall on a 32-core host (alc-6 / alc-9 / alc-2). DO NOT
run on a laptop.
"""

from __future__ import annotations

import argparse
import datetime
import json
import subprocess  # nosec B404
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CELLS_SOURCE = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "results.json"
)
DEFAULT_NE_GENOMES_DIR = (
    REPO_ROOT
    / "bucket_brigade"
    / "baselines"
    / "release"
    / "local"
    / "nash"
    / "phase_diagram"
)
DEFAULT_OUTPUT = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "per_cell_baselines.json"
)

# Canonical calibration point: the base minimal_specialization scenario,
# β=0.25 κ=0.5 c=0.5. Source: bucket_brigade/baselines/__init__.py:53-59
# provenance comment.
CANONICAL_BETA = 0.25
CANONICAL_KAPPA = 0.5
CANONICAL_COST = 0.5
CANONICAL_TAG = "canonical_b0.25_k0.50_c0.50"


def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.run(  # nosec B603 B607
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _load_phase_diagram_cells(source: Path) -> list[dict]:
    data = json.loads(source.read_text())
    cells = data.get("cells")
    if not isinstance(cells, list):
        raise SystemExit(f"{source} has no 'cells' array")
    out = []
    for raw in cells:
        beta = float(raw["beta"])
        kappa = float(raw["kappa"])
        c = float(raw["c"])
        tag = raw.get("tag") or f"b{beta:.2f}_k{kappa:.2f}_c{c:.2f}"
        out.append(
            {
                "tag": tag,
                "beta": beta,
                "kappa": kappa,
                "cost": c,
                "ne_verdict": raw.get("verdict"),
                "best_asymmetric_profile_label": raw.get(
                    "best_asymmetric_profile_label"
                ),
            }
        )
    return out


def _resolve_ne_genomes_path(tag: str, ne_dir: Path) -> Optional[Path]:
    """Find the NE-genome JSON for a cell tag. Returns None if absent."""
    candidate = ne_dir / f"{tag}.json"
    if candidate.exists():
        return candidate
    return None


def _measure_cell(
    *,
    tag: str,
    beta: float,
    kappa: float,
    cost: float,
    n_episodes: int,
    seed: int,
    num_workers: Optional[int],
    n_boot: int,
    ne_genomes_path: Optional[Path],
    base_scenario_name: str,
) -> dict:
    # Lazy import keeps --help cheap.
    from bucket_brigade.baselines.per_cell import (
        measure_random,
        measure_specialist_homogeneous,
        measure_specialist_ne,
    )

    print(
        f"  [random] β={beta:.2f} κ={kappa:.2f} c={cost:.2f}...",
        flush=True,
    )
    t0 = time.monotonic()
    rnd = measure_random(
        beta=beta,
        kappa=kappa,
        cost=cost,
        n_episodes=n_episodes,
        seed=seed,
        num_workers=num_workers,
        base_scenario_name=base_scenario_name,
        n_boot=n_boot,
    )
    print(
        f"    -> mean={rnd.mean:.3f} CI95=[{rnd.ci95_lo:.3f}, {rnd.ci95_hi:.3f}] "
        f"({time.monotonic() - t0:.1f}s)"
    )

    print("  [specialist_homogeneous]...", flush=True)
    t0 = time.monotonic()
    homo = measure_specialist_homogeneous(
        beta=beta,
        kappa=kappa,
        cost=cost,
        n_episodes=n_episodes,
        seed=seed,
        num_workers=num_workers,
        base_scenario_name=base_scenario_name,
        n_boot=n_boot,
    )
    print(
        f"    -> mean={homo.mean:.3f} CI95=[{homo.ci95_lo:.3f}, {homo.ci95_hi:.3f}] "
        f"({time.monotonic() - t0:.1f}s)"
    )

    if ne_genomes_path is not None:
        print(f"  [specialist_ne] genomes={ne_genomes_path.name}...", flush=True)
        t0 = time.monotonic()
        ne = measure_specialist_ne(
            beta=beta,
            kappa=kappa,
            cost=cost,
            ne_genomes_path=ne_genomes_path,
            n_episodes=n_episodes,
            seed=seed,
            num_workers=num_workers,
            base_scenario_name=base_scenario_name,
            n_boot=n_boot,
        )
        print(
            f"    -> mean={ne.mean:.3f} CI95=[{ne.ci95_lo:.3f}, {ne.ci95_hi:.3f}] "
            f"({time.monotonic() - t0:.1f}s)"
        )
        ne_dict = ne.to_dict()
        ne_path_str = str(ne_genomes_path.relative_to(REPO_ROOT))
    else:
        ne_dict = None
        ne_path_str = None
        print("  [specialist_ne] no NE-genomes file for this cell; emitting null.")

    return {
        "cell_tag": tag,
        "params": {"beta": beta, "kappa": kappa, "cost_to_work": cost},
        "random_baseline": rnd.to_dict(),
        "specialist_homogeneous": homo.to_dict(),
        "specialist_ne": ne_dict,
        "ne_genomes_path": ne_path_str,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cells-source", type=Path, default=DEFAULT_CELLS_SOURCE)
    p.add_argument("--ne-genomes-dir", type=Path, default=DEFAULT_NE_GENOMES_DIR)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument(
        "--base-scenario",
        default="minimal_specialization",
        help="Base scenario family.",
    )
    p.add_argument("--n-episodes", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="Pool size for parallel episodes; None -> cpu_count().",
    )
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument(
        "--skip-canonical",
        action="store_true",
        help="Don't measure the canonical β=0.25 κ=0.5 c=0.5 calibration row.",
    )
    args = p.parse_args(argv)

    cells = _load_phase_diagram_cells(args.cells_source)

    # Prepend the canonical row (so order matches the acceptance test's
    # natural ordering: canonical first, then grid).
    rows_to_measure: list[dict] = []
    if not args.skip_canonical:
        rows_to_measure.append(
            {
                "tag": CANONICAL_TAG,
                "beta": CANONICAL_BETA,
                "kappa": CANONICAL_KAPPA,
                "cost": CANONICAL_COST,
                "ne_verdict": "canonical_calibration_point",
                "best_asymmetric_profile_label": None,
            }
        )
    rows_to_measure.extend(cells)

    print(
        f"== per-cell baseline sweep: {len(rows_to_measure)} cells × 3 policies "
        f"× n_episodes={args.n_episodes} =="
    )
    print(f"   output: {args.output}")
    print(f"   ne_genomes_dir: {args.ne_genomes_dir}")
    print(f"   base_scenario: {args.base_scenario}")

    results: list[dict] = []
    sweep_start = time.monotonic()
    missing_ne: list[str] = []

    for idx, row in enumerate(rows_to_measure, start=1):
        print(
            f"\n[{idx}/{len(rows_to_measure)}] cell {row['tag']} "
            f"(NE={row.get('ne_verdict', '?')})"
        )
        if row["tag"] == CANONICAL_TAG:
            ne_path = None  # canonical cell has no DO NE genome file
        else:
            ne_path = _resolve_ne_genomes_path(row["tag"], args.ne_genomes_dir)
            if ne_path is None:
                missing_ne.append(row["tag"])

        cell_result = _measure_cell(
            tag=row["tag"],
            beta=row["beta"],
            kappa=row["kappa"],
            cost=row["cost"],
            n_episodes=args.n_episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            n_boot=args.n_boot,
            ne_genomes_path=ne_path,
            base_scenario_name=args.base_scenario,
        )
        cell_result["ne_verdict"] = row.get("ne_verdict")
        cell_result["ne_profile_label"] = row.get("best_asymmetric_profile_label")
        results.append(cell_result)

    wall = time.monotonic() - sweep_start
    payload = {
        "cells": results,
        "metadata": {
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "base_scenario": args.base_scenario,
            "git_sha": _git_sha(REPO_ROOT),
            "measured_at": datetime.datetime.utcnow().isoformat(timespec="seconds")
            + "Z",
            "wall_seconds": wall,
            "missing_ne_genomes": missing_ne,
            "cells_source": str(args.cells_source.relative_to(REPO_ROOT))
            if args.cells_source.is_relative_to(REPO_ROOT)
            else str(args.cells_source),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))
    print(
        f"\n== sweep complete in {wall:.1f}s. wrote {args.output} "
        f"({len(results)} cells, missing NE genomes: {missing_ne or 'none'}) =="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
