#!/usr/bin/env python3
"""
Render the heterogeneous Nash phase diagram from an aggregate ``results.json``.

Consumes the aggregate ``results.json`` written by
``experiments/scripts/compute_nash_phase_diagram.py`` and produces:

  - ``phase_diagram.png``       — categorical heat-map figure, one subplot per
                                  ``c`` value, β on the y-axis and κ on the
                                  x-axis. Cell colour encodes the verdict;
                                  cell label shows the best team payoff.
  - ``phase_diagram_table.md``  — markdown table listing every cell sorted by
                                  ``(c, β, κ)`` with verdict, equilibrium
                                  payoff, and convergence rate. Paste-ready
                                  for the paper.

The plotting layer was deferred from PR #377 — quoting the builder report:

    Plotting (``phase_diagram.png``) explicitly deferred to a follow-up
    issue to keep PR script-only.

This script closes that follow-up (#380). It does NOT run any compute — it
only reads the aggregate ``results.json`` that the driver wrote on a remote
host (or the local 1×1×1 ``--smoke`` output).

Usage
-----

::

    # Default paths (matches the driver's default output dir)
    uv run python experiments/scripts/plot_phase_diagram.py \\
        --results experiments/nash/phase_diagram/results.json

    # Smoke-test against the driver's smoke output (1×1×1 degenerate grid)
    uv run python experiments/scripts/compute_nash_phase_diagram.py --smoke
    uv run python experiments/scripts/plot_phase_diagram.py \\
        --results experiments/nash/phase_diagram/smoke/results.json \\
        --out-png /tmp/test.png --out-md /tmp/test.md

Schema
------

The aggregate ``results.json`` produced by the driver has this shape::

    {
      "base_scenario": "...",
      "grid": {"beta_values": [...], "kappa_values": [...], "c_values": [...]},
      "cells": [
         {
           "beta": float, "kappa": float, "c": float,
           "verdict": "symmetric_only|asymmetric_only|mixed|no_convergence",
           "best_team_payoff": float | null,
           "converged": int, "total_restarts": int,
           ...
         }, ...
      ],
      ...
    }

Notes
-----

- Verdict → colour mapping uses a four-colour categorical palette chosen for
  print-greyscale separability (different lightnesses). A legend maps verdict
  to colour above the figure.
- For the degenerate 1×1×1 ``--smoke`` case the figure still renders cleanly:
  a single subplot with a single labelled cell.
- ``convergence_rate`` in the markdown table is ``converged / total_restarts``;
  ``equilibrium_payoff`` is the cell's ``best_team_payoff`` (null cells render
  as ``n/a``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Categorical verdict palette
# ---------------------------------------------------------------------------

# Order matches the driver's classifier in compute_nash_phase_diagram.py.
# Colours chosen for greyscale-print separability: monotonic lightness so a
# B/W printout still distinguishes verdicts by shade.
VERDICT_ORDER: tuple[str, ...] = (
    "symmetric_only",
    "mixed",
    "asymmetric_only",
    "no_convergence",
)

VERDICT_COLORS: dict[str, str] = {
    "symmetric_only": "#2c7fb8",  # dark blue
    "mixed": "#7fcdbb",  # teal
    "asymmetric_only": "#edf8b1",  # pale yellow
    "no_convergence": "#bdbdbd",  # neutral grey
}


def _verdict_color(verdict: str) -> str:
    """Return the palette colour for a verdict, with a safe fallback."""
    return VERDICT_COLORS.get(verdict, "#ffffff")


# ---------------------------------------------------------------------------
# Aggregate loader
# ---------------------------------------------------------------------------


def load_aggregate(results_path: Path) -> dict[str, Any]:
    """Load and lightly validate the aggregate ``results.json``."""
    if not results_path.exists():
        raise FileNotFoundError(f"Aggregate results not found: {results_path}")
    with open(results_path) as f:
        data = json.load(f)

    for key in ("grid", "cells"):
        if key not in data:
            raise ValueError(
                f"Aggregate results missing required key '{key}': {results_path}"
            )
    grid = data["grid"]
    for key in ("beta_values", "kappa_values", "c_values"):
        if key not in grid:
            raise ValueError(
                f"Aggregate results missing required grid axis '{key}': {results_path}"
            )
    return data


def _index_cells_by_coord(
    cells: list[dict[str, Any]],
) -> dict[tuple[float, float, float], dict[str, Any]]:
    """Index cell records by their (β, κ, c) coordinate triple."""
    return {(c["beta"], c["kappa"], c["c"]): c for c in cells}


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------


def render_png(data: dict[str, Any], out_png: Path) -> None:
    """Render the (β × κ) per-``c`` heat-map figure to ``out_png``.

    Layout:
      - One subplot per ``c`` value, horizontally arranged
      - β on y-axis (rows), κ on x-axis (columns)
      - Each cell colored by verdict; annotated with the best team payoff
      - Single shared legend above the subplots maps verdict → colour
    """
    grid = data["grid"]
    beta_values: list[float] = list(grid["beta_values"])
    kappa_values: list[float] = list(grid["kappa_values"])
    c_values: list[float] = list(grid["c_values"])
    base_scenario = data.get("base_scenario", "unknown")

    cells_by_coord = _index_cells_by_coord(data.get("cells", []))

    n_subplots = len(c_values)
    n_beta = len(beta_values)
    n_kappa = len(kappa_values)

    # Per-subplot size scales with grid dims; clamp to sane bounds so the
    # 1×1×1 smoke case stays legible and the full 5×5×3 case stays printable.
    # Add headroom (`+ 2.2`) so the suptitle + legend strip never overlap the
    # subplot titles ("c = ..."), even in the degenerate single-cell case.
    sub_w = max(3.5, 0.9 * n_kappa + 2.0)
    sub_h = max(3.5, 0.9 * n_beta + 1.5)
    fig_w = sub_w * n_subplots
    fig_h = sub_h + 1.2
    fig, axes = plt.subplots(
        1, n_subplots, figsize=(fig_w, fig_h), squeeze=False, constrained_layout=True
    )
    axes = axes[0]  # squeeze=False keeps the row dimension

    for ax_idx, (ax, c_val) in enumerate(zip(axes, c_values)):
        # Set up axis: rows = β (top→bottom = high→low so larger β is "up"),
        # cols = κ (left→right = low→high).
        ax.set_xlim(-0.5, n_kappa - 0.5)
        ax.set_ylim(-0.5, n_beta - 0.5)
        ax.set_aspect("equal")
        ax.set_xticks(range(n_kappa))
        ax.set_xticklabels([f"{k:.2f}" for k in kappa_values])
        ax.set_yticks(range(n_beta))
        ax.set_yticklabels([f"{b:.2f}" for b in beta_values])
        ax.set_xlabel("κ (solo extinguish prob)")
        if ax_idx == 0:
            ax.set_ylabel("β (spread prob)")
        ax.set_title(f"c = {c_val:.2f}")
        # Light grid lines between cells
        for x in range(n_kappa + 1):
            ax.axvline(x - 0.5, color="white", lw=0.5)
        for y in range(n_beta + 1):
            ax.axhline(y - 0.5, color="white", lw=0.5)

        for bi, beta in enumerate(beta_values):
            for ki, kappa in enumerate(kappa_values):
                cell = cells_by_coord.get((beta, kappa, c_val))
                if cell is None:
                    color = "#ffffff"
                    label = "—"
                else:
                    color = _verdict_color(cell.get("verdict", ""))
                    payoff = cell.get("best_team_payoff")
                    label = (
                        f"{payoff:.0f}" if isinstance(payoff, (int, float)) else "n/a"
                    )
                rect = plt.Rectangle(
                    (ki - 0.5, bi - 0.5),
                    1.0,
                    1.0,
                    facecolor=color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                ax.add_patch(rect)
                ax.text(
                    ki,
                    bi,
                    label,
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="black",
                )

    # Shared legend below the subplots — placing it below keeps the title
    # area clean and avoids overlap with the per-subplot "c = ..." titles.
    legend_handles = [
        mpatches.Patch(facecolor=_verdict_color(v), edgecolor="black", label=v)
        for v in VERDICT_ORDER
    ]
    fig.suptitle(
        f"Heterogeneous Nash Phase Diagram — {base_scenario}",
        fontsize=11,
    )
    fig.legend(
        handles=legend_handles,
        loc="outside lower center",
        ncol=len(VERDICT_ORDER),
        frameon=False,
        fontsize=9,
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------


def render_markdown(data: dict[str, Any], out_md: Path) -> None:
    """Render the per-cell markdown table to ``out_md``.

    Rows are sorted by ``(c, β, κ)``. Columns:

    - ``c``                 — cost-to-work-one-night
    - ``β``                 — spread probability
    - ``κ``                 — solo-extinguish probability
    - ``verdict``           — one of the four classifier verdicts
    - ``equilibrium_payoff``— ``best_team_payoff`` (``n/a`` if null)
    - ``convergence_rate``  — ``converged / total_restarts`` (``n/a`` if zero
                              restarts)
    """
    base_scenario = data.get("base_scenario", "unknown")
    grid = data.get("grid", {})
    params = data.get("parameters", {})
    cells = sorted(
        data.get("cells", []),
        key=lambda r: (r.get("c", 0.0), r.get("beta", 0.0), r.get("kappa", 0.0)),
    )

    lines: list[str] = []
    lines.append("# Heterogeneous Nash Phase Diagram — per-cell table\n")
    lines.append(f"Base scenario: `{base_scenario}`\n")
    lines.append(
        f"Grid: {len(grid.get('beta_values', []))}×"
        f"{len(grid.get('kappa_values', []))}×"
        f"{len(grid.get('c_values', []))} = {len(cells)} cells\n"
    )
    if params:
        lines.append(
            f"Per-cell budget: restarts={params.get('num_restarts')}, "
            f"max_iter={params.get('max_iterations')}, "
            f"simulations={params.get('num_simulations')}, "
            f"ε={params.get('epsilon')}\n"
        )
    lines.append("\n")
    lines.append("| c | β | κ | verdict | equilibrium_payoff | convergence_rate |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for cell in cells:
        payoff = cell.get("best_team_payoff")
        payoff_str = f"{payoff:.2f}" if isinstance(payoff, (int, float)) else "n/a"
        total = cell.get("total_restarts", 0) or 0
        converged = cell.get("converged", 0) or 0
        if total > 0:
            rate_str = f"{converged}/{total} ({converged / total:.0%})"
        else:
            rate_str = "n/a"
        lines.append(
            f"| {cell.get('c', float('nan')):.2f} "
            f"| {cell.get('beta', float('nan')):.2f} "
            f"| {cell.get('kappa', float('nan')):.2f} "
            f"| `{cell.get('verdict', 'unknown')}` "
            f"| {payoff_str} "
            f"| {rate_str} |\n"
        )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    with open(out_md, "w") as f:
        f.writelines(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Render the heterogeneous Nash phase diagram heat-map and per-cell "
            "table from an aggregate results.json."
        )
    )
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("experiments/nash/phase_diagram/results.json"),
        help=(
            "Path to the aggregate results.json written by "
            "compute_nash_phase_diagram.py (default: "
            "experiments/nash/phase_diagram/results.json)"
        ),
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=Path("experiments/nash/phase_diagram/phase_diagram.png"),
        help=(
            "Output PNG path (default: "
            "experiments/nash/phase_diagram/phase_diagram.png)"
        ),
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("experiments/nash/phase_diagram/phase_diagram_table.md"),
        help=(
            "Output markdown table path (default: "
            "experiments/nash/phase_diagram/phase_diagram_table.md)"
        ),
    )
    args = parser.parse_args()

    data = load_aggregate(args.results)
    render_png(data, args.out_png)
    render_markdown(data, args.out_md)

    print(f"Loaded {len(data.get('cells', []))} cells from {args.results}")
    print(f"PNG  → {args.out_png}")
    print(f"MD   → {args.out_md}")


if __name__ == "__main__":
    main()
