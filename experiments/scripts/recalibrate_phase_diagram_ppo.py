"""Re-aggregate #360 phase-diagram PPO results with #413's per-cell baselines.

Walks the existing
``experiments/p3_specialization/phase_diagram_ppo/cell_<tag>/seed_<seed>/summary.json``
files from the #360 sweep, recomputes ``gap_closed`` from
``trailing5_team_mean`` using the new ``per_cell_baselines.json`` table
(both ``gap_closed_homogeneous`` and ``gap_closed_ne``), and produces a
side-by-side diff table comparing OLD (single-MINSPEC) vs NEW per-cell
calibrations.

Outputs:

* ``experiments/p3_specialization/phase_diagram_ppo/recalibrated_verdict.md``
  — markdown diff table, sorted by NE verdict / gap_closed_homogeneous_mean.
* ``experiments/p3_specialization/phase_diagram_ppo/recalibrated_verdict.json``
  — machine-readable per-cell rows.

This is the load-bearing deliverable of issue #413: it disentangles
whether the cross-cell PPO ordering from #360 (asymmetric > symmetric >
no_convergence) survives proper per-cell calibration.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PPO_ROOT = REPO_ROOT / "experiments" / "p3_specialization" / "phase_diagram_ppo"
DEFAULT_PER_CELL = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "per_cell_baselines.json"
)


def _load_per_cell(path: Path) -> dict[str, dict]:
    if not path.exists():
        raise SystemExit(
            f"per_cell_baselines.json not found at {path}. Run "
            "experiments/scripts/measure_per_cell_baselines.py first."
        )
    data = json.loads(path.read_text())
    return {row["cell_tag"]: row for row in data.get("cells", []) if "cell_tag" in row}


def _gap(trailing5: float, low: float, high: float) -> Optional[float]:
    denom = high - low
    if denom == 0:
        return None
    return (trailing5 - low) / denom


def _recompute_cell(
    cell_dir: Path,
    per_cell_table: dict[str, dict],
) -> dict:
    """Aggregate per-seed trailing5 means and compute dual-column gap_closed."""
    # Lazy import for MINSPEC fallback.
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    tag = cell_dir.name.removeprefix("cell_")
    per_seed: list[dict] = []
    for sdir in sorted(cell_dir.glob("seed_*")):
        sf = sdir / "summary.json"
        if not sf.exists():
            continue
        try:
            sd = json.loads(sf.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if sd.get("trailing5_team_mean") is None:
            continue
        per_seed.append(sd)

    row_baseline = per_cell_table.get(tag)
    if row_baseline is None:
        print(
            f"WARN: cell '{tag}' missing from per_cell_baselines.json; "
            "using MINSPEC fallback for the recomputed gap_closed columns.",
            file=sys.stderr,
        )

    legacy_gaps = [
        float(sd["gap_closed"]) for sd in per_seed if sd.get("gap_closed") is not None
    ]
    trailings = [float(sd["trailing5_team_mean"]) for sd in per_seed]

    if row_baseline is not None:
        cell_random = float(row_baseline["random_baseline"]["mean"])
        cell_homo = float(row_baseline["specialist_homogeneous"]["mean"])
        ne_row = row_baseline.get("specialist_ne")
        cell_ne = float(ne_row["mean"]) if ne_row is not None else None
    else:
        cell_random = MINSPEC_RANDOM
        cell_homo = MINSPEC_SPECIALIST
        cell_ne = None

    new_homos = [_gap(t, cell_random, cell_homo) for t in trailings]
    new_homos = [g for g in new_homos if g is not None]
    if cell_ne is not None:
        new_nes = [_gap(t, cell_random, cell_ne) for t in trailings]
        new_nes = [g for g in new_nes if g is not None]
    else:
        new_nes = []

    def _mean(xs: list[float]) -> Optional[float]:
        return float(sum(xs) / len(xs)) if xs else None

    def _std(xs: list[float]) -> Optional[float]:
        return float(statistics.pstdev(xs)) if len(xs) > 1 else (0.0 if xs else None)

    return {
        "cell_tag": tag,
        "ne_verdict": per_seed[0]["cell"].get("ne_verdict") if per_seed else None,
        "n_seeds": len(per_seed),
        "trailing5_team_mean_mean": _mean(trailings),
        "old_gap_closed_mean": _mean(legacy_gaps),
        "old_gap_closed_std": _std(legacy_gaps),
        "gap_closed_homogeneous_mean": _mean(new_homos),
        "gap_closed_homogeneous_std": _std(new_homos),
        "gap_closed_ne_mean": _mean(new_nes),
        "gap_closed_ne_std": _std(new_nes) if new_nes else None,
        "cell_random_baseline": cell_random,
        "cell_specialist_homogeneous": cell_homo,
        "cell_specialist_ne": cell_ne,
        "old_baseline_random": MINSPEC_RANDOM,
        "old_baseline_specialist": MINSPEC_SPECIALIST,
        "baseline_source": "per_cell"
        if row_baseline is not None
        else "minspec_fallback",
    }


def _fmt(x: Optional[float], prec: int = 3) -> str:
    if x is None:
        return "—"
    return f"{x:.{prec}f}"


def _render_md(rows: list[dict], per_cell_path: Path) -> str:
    lines = [
        "# #360 phase-diagram PPO results — recalibrated per-cell (issue #413)\n",
        "Re-aggregation of the existing #360 PPO sweep results using the per-cell",
        "Random / Specialist baselines from",
        f"`{per_cell_path.relative_to(REPO_ROOT)}`. Columns:\n",
        "* `OLD gap_closed`: original #360 metric — every cell scored against the",
        "  single canonical MINSPEC_RANDOM = -87.72 / MINSPEC_SPECIALIST = -22.07.",
        "* `NEW gap_closed_homogeneous`: per-cell Random→SpecialistPolicy×4 (apples-",
        "  to-apples drop-in for the MINSPEC tradition).",
        "* `NEW gap_closed_ne`: per-cell Random→1×Hero+3×Firefighter (the heterogeneous",
        "  NE asymmetric profile from the DO search). The metric appropriate for the",
        "  paper §3/§4 NE-structure-vs-PPO-success hypothesis.\n",
        "Sort: by NE verdict, then by `gap_closed_homogeneous_mean` descending.\n",
        "| cell | NE verdict | seeds | OLD gap_closed | NEW gap_closed_homogeneous | NEW gap_closed_ne | cell random | cell homo | cell NE |",
        "|------|------------|------:|---------------:|---------------------------:|------------------:|------------:|----------:|--------:|",
    ]
    # Sort: asymmetric_only / symmetric_only / no_convergence, then by homo_mean desc.
    verdict_order = {
        "asymmetric_only": 0,
        "symmetric_only": 1,
        "no_convergence": 2,
        None: 3,
    }
    rows_sorted = sorted(
        rows,
        key=lambda r: (
            verdict_order.get(r.get("ne_verdict"), 99),
            -(r.get("gap_closed_homogeneous_mean") or float("-inf")),
        ),
    )
    for r in rows_sorted:
        old = _fmt(r["old_gap_closed_mean"])
        old_std = _fmt(r["old_gap_closed_std"])
        new_h = _fmt(r["gap_closed_homogeneous_mean"])
        new_h_std = _fmt(r["gap_closed_homogeneous_std"])
        new_n = _fmt(r["gap_closed_ne_mean"])
        new_n_std = (
            _fmt(r["gap_closed_ne_std"])
            if r.get("gap_closed_ne_std") is not None
            else "—"
        )
        lines.append(
            f"| {r['cell_tag']} | {r.get('ne_verdict', '?')} | {r['n_seeds']} "
            f"| {old}±{old_std} "
            f"| {new_h}±{new_h_std} "
            f"| {new_n}±{new_n_std} "
            f"| {_fmt(r['cell_random_baseline'], 2)} "
            f"| {_fmt(r['cell_specialist_homogeneous'], 2)} "
            f"| {_fmt(r['cell_specialist_ne'], 2)} |"
        )

    # Ordering commentary.
    lines.append("\n## Ordering check\n")
    by_v = {"asymmetric_only": [], "symmetric_only": [], "no_convergence": []}
    for r in rows:
        v = r.get("ne_verdict")
        if v in by_v:
            by_v[v].append(r)

    def _mean_of(rs: list[dict], key: str) -> Optional[float]:
        vs = [r[key] for r in rs if r.get(key) is not None]
        return sum(vs) / len(vs) if vs else None

    for col_name, col_key in [
        ("OLD gap_closed", "old_gap_closed_mean"),
        ("NEW gap_closed_homogeneous", "gap_closed_homogeneous_mean"),
        ("NEW gap_closed_ne", "gap_closed_ne_mean"),
    ]:
        means = {v: _mean_of(by_v[v], col_key) for v in by_v}
        ordering = sorted(
            [(v, m) for v, m in means.items() if m is not None],
            key=lambda x: -x[1],
        )
        if ordering:
            ranking_str = " > ".join(f"{v} ({_fmt(m)})" for v, m in ordering)
            lines.append(f"* **{col_name}**: {ranking_str}")
        else:
            lines.append(f"* **{col_name}**: no data")

    return "\n".join(lines) + "\n"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ppo-root", type=Path, default=DEFAULT_PPO_ROOT)
    p.add_argument("--per-cell-baselines", type=Path, default=DEFAULT_PER_CELL)
    p.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_PPO_ROOT / "recalibrated_verdict.md",
    )
    p.add_argument(
        "--output-json",
        type=Path,
        default=DEFAULT_PPO_ROOT / "recalibrated_verdict.json",
    )
    args = p.parse_args(argv)

    table = _load_per_cell(args.per_cell_baselines)
    cell_dirs = sorted(args.ppo_root.glob("cell_*"))
    if not cell_dirs:
        raise SystemExit(f"no cell_* dirs under {args.ppo_root}")

    rows = [_recompute_cell(cd, table) for cd in cell_dirs]

    args.output_json.write_text(
        json.dumps(
            {
                "per_cell_baselines_path": str(
                    args.per_cell_baselines.relative_to(REPO_ROOT)
                    if args.per_cell_baselines.is_relative_to(REPO_ROOT)
                    else args.per_cell_baselines
                ),
                "ppo_root": str(
                    args.ppo_root.relative_to(REPO_ROOT)
                    if args.ppo_root.is_relative_to(REPO_ROOT)
                    else args.ppo_root
                ),
                "cells": rows,
            },
            indent=2,
        )
    )
    args.output_md.write_text(_render_md(rows, args.per_cell_baselines))
    print(f"wrote {args.output_md} and {args.output_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
