"""Render Figure 2 (per-cell PPO gap_closed_ne heatmap) for the v3 workshop paper.

Replaces the v2 TikZ heatmap that displayed std as inline text ("$\\pm 0.331$"
lines inside each cell). The v2 reviewer's D6 concern was that text-form std
obscures the signal-vs-noise ratio --- std is roughly comparable to the mean on
symmetric cells. Reviewer endorsed option (b): a matplotlib render with a
visual uncertainty overlay.

This script reads ground-truth values directly from
``experiments/p3_specialization/phase_diagram_ppo/recalibrated_verdict.json``,
so the figure stays in sync with the data automatically. Re-running this
script is deterministic and idempotent.

Output: ``figures/recalibrated_heatmap.pdf`` (vector PDF, ~3.4in wide, 1 page).

Run from the version dir::

    uv run --with matplotlib python figures/src/recalibrated_heatmap.py

Or from the repo root::

    uv run --with matplotlib python paper/anvil_pub.bb-workshop.3/figures/src/recalibrated_heatmap.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ----------------------------------------------------------------------------
# Paths: figure-relative for output, repo-relative for data
# ----------------------------------------------------------------------------

SCRIPT_PATH = Path(__file__).resolve()
FIG_DIR = SCRIPT_PATH.parent.parent  # .../figures/
VERSION_DIR = FIG_DIR.parent          # .../anvil_pub.bb-workshop.3/
PAPER_DIR = VERSION_DIR.parent        # .../paper/
REPO_ROOT = PAPER_DIR.parent          # .../bucket-brigade/

DATA_PATH = (
    REPO_ROOT
    / "experiments"
    / "p3_specialization"
    / "phase_diagram_ppo"
    / "recalibrated_verdict.json"
)
OUTPUT_PATH = FIG_DIR / "recalibrated_heatmap.pdf"


# ----------------------------------------------------------------------------
# Grid layout
# ----------------------------------------------------------------------------

BETAS = [0.1, 0.5, 0.9]   # columns left-to-right
KAPPAS = [0.1, 0.5, 0.9]  # rows bottom-to-top (kappa-low at bottom)


def load_cells() -> dict[tuple[float, float], dict]:
    """Load per-cell values keyed by (beta, kappa). Cells not present in the
    7-cell preview (beta=0.1 at kappa in {0.5, 0.9}) are absent from the data
    file and rendered as 'n/a'."""
    with DATA_PATH.open() as f:
        data = json.load(f)
    out: dict[tuple[float, float], dict] = {}
    for cell in data["cells"]:
        tag = cell["cell_tag"]  # e.g., "b0.50_k0.90_c0.50"
        parts = tag.split("_")
        beta = float(parts[0][1:])
        kappa = float(parts[1][1:])
        out[(beta, kappa)] = cell
    return out


def cell_metric(cell: dict) -> tuple[float, float, bool]:
    """Return (mean, std, is_homogeneous_fallback) for a cell.

    For NE-bearing cells (symmetric_only, asymmetric_only) use gap_closed_ne.
    For no_convergence cells, fall back to gap_closed_homogeneous and flag.
    """
    if cell["ne_verdict"] == "no_convergence":
        return (
            cell["gap_closed_homogeneous_mean"],
            cell["gap_closed_homogeneous_std"],
            True,
        )
    return (
        cell["gap_closed_ne_mean"],
        cell["gap_closed_ne_std"],
        False,
    )


def main() -> None:
    cells = load_cells()

    # Single-column workshop figure: ~3.4in wide is typical for two-column layouts.
    # The anvil-paper class is single-column so we go slightly wider.
    fig, ax = plt.subplots(figsize=(4.2, 3.6))

    # Diverging colormap centered at 0. RdYlGn is too saturated for a paper
    # figure; use RdYlGn but desaturate via norm limits.
    cmap = mpl.colormaps["RdYlGn"]
    norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.40)

    # Track the largest |std| seen, for scaling the error-bar overlay.
    # The std overlay is a horizontal bar of width proportional to std,
    # rendered inside each cell beneath the mean label.
    max_std = max(
        cell_metric(c)[1] or 0.0 for c in cells.values()
    )  # ~0.344

    # Cell geometry: each cell is 1.0 wide x 1.0 tall in axis coords.
    for j, beta in enumerate(BETAS):
        for i, kappa in enumerate(KAPPAS):
            x0, y0 = j, i
            cx, cy = x0 + 0.5, y0 + 0.5

            cell = cells.get((beta, kappa))
            if cell is None:
                # n/a cell (beta=0.1 at kappa in {0.5, 0.9} not sampled)
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0, y0), 1, 1,
                        facecolor="#ececec",
                        edgecolor="#999",
                        linewidth=0.5,
                        hatch="xxx",
                    )
                )
                ax.text(
                    cx, cy, "n/a",
                    ha="center", va="center",
                    fontsize=9, color="#666", style="italic",
                )
                continue

            mean, std, is_fallback = cell_metric(cell)

            # Heat color from mean
            color = cmap(norm(mean))
            ax.add_patch(
                mpatches.Rectangle(
                    (x0, y0), 1, 1,
                    facecolor=color,
                    edgecolor="#444",
                    linewidth=0.5,
                )
            )

            # Hatching on no_convergence row to flag metric switch
            if is_fallback:
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0, y0), 1, 1,
                        facecolor="none",
                        edgecolor="#a33",
                        linewidth=0.0,
                        hatch="///",
                        alpha=0.7,
                    )
                )

            # Mean value, centered, prominent.
            # Add dagger superscript for fallback metric.
            label = f"{mean:+.3f}"
            if is_fallback:
                label = label + r"$^{\dagger}$"
            ax.text(
                cx, cy + 0.18, label,
                ha="center", va="center",
                fontsize=11, fontweight="bold",
                color="#1a1a1a",
            )

            # --- Uncertainty overlay: Tufte-style horizontal error bar ---
            # Width is proportional to std; centered at the mean's x-position.
            # Max overlay width fits inside the cell with margin.
            #
            # Convention: bar half-width = (std / max_std) * 0.40 cell-units.
            # So a cell with std == max_std spans 80% of cell width.
            # A "tick" mark at center indicates the mean point.
            bar_y = cy - 0.15
            half = (std / max_std) * 0.40
            ax.plot(
                [cx - half, cx + half], [bar_y, bar_y],
                color="#1a1a1a", linewidth=1.2, solid_capstyle="butt",
            )
            # End caps
            cap_h = 0.04
            ax.plot(
                [cx - half, cx - half], [bar_y - cap_h, bar_y + cap_h],
                color="#1a1a1a", linewidth=1.0,
            )
            ax.plot(
                [cx + half, cx + half], [bar_y - cap_h, bar_y + cap_h],
                color="#1a1a1a", linewidth=1.0,
            )
            # Mean tick (filled dot at center)
            ax.plot(
                [cx], [bar_y],
                marker="o", markersize=3.5,
                color="#1a1a1a", markeredgecolor="white", markeredgewidth=0.5,
            )
            # Numeric std label below the bar
            ax.text(
                cx, cy - 0.32, f"$\\pm${std:.2f}",
                ha="center", va="center",
                fontsize=7, color="#444",
            )

    # Axes setup
    ax.set_xlim(0, 3)
    ax.set_ylim(0, 3)
    ax.set_aspect("equal")
    ax.set_xticks([0.5, 1.5, 2.5])
    ax.set_xticklabels([rf"$\beta{{=}}{b}$" for b in BETAS], fontsize=9)
    ax.set_yticks([0.5, 1.5, 2.5])
    ax.set_yticklabels([rf"$\kappa{{=}}{k}$" for k in KAPPAS], fontsize=9)
    ax.set_xlabel(r"Fire-spread $\beta$", fontsize=10)
    ax.set_ylabel(r"Extinguish $\kappa$", fontsize=10)
    # Remove tick marks but keep labels
    ax.tick_params(axis="both", length=0)
    # Remove default spines (the cell borders provide the frame)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # ---- Legend / key in figure margin ----
    # Three small swatches: NE-anchored, fallback-metric (hatched), n/a (xxx).
    legend_handles = [
        mpatches.Patch(
            facecolor=cmap(norm(0.30)), edgecolor="#444", linewidth=0.5,
            label=r"NE-anchored $\mathtt{gap\_closed\_ne}$",
        ),
        mpatches.Patch(
            facecolor=cmap(norm(-0.05)), edgecolor="#a33", linewidth=0.5,
            hatch="///",
            label=(
                r"fallback $\mathtt{gap\_closed\_homogeneous}$ "
                r"($\dagger$, no NE)"
            ),
        ),
        mpatches.Patch(
            facecolor="#ececec", edgecolor="#999", linewidth=0.5,
            hatch="xxx",
            label=r"cell not sampled (n/a)",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.20),
        ncol=1,
        fontsize=7,
        frameon=False,
        handlelength=2.0,
        handleheight=1.0,
        borderpad=0.4,
        title=(
            r"Horizontal bar in each cell: $\pm 1$ std (4 seeds), "
            r"scaled so widest std $\approx 0.34$ spans 80% of the cell."
        ),
        title_fontsize=6.5,
    )

    plt.savefig(
        OUTPUT_PATH,
        bbox_inches="tight",
        pad_inches=0.05,
    )
    plt.close(fig)
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
