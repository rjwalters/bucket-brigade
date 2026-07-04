"""Render Figure 2 (per-cell PPO gap_closed_ne heatmap) for the v4 workshop paper.

v4 expands the v3 3x3 grid (7 cells, beta x kappa at c=0.5) to a 3-panel
3x5 grid (37 cells, beta x kappa across c in {0.5, 1.0, 2.0}). Three panels
sit side by side, one per c value; rows are kappa (low at bottom), columns
are beta. Two cells (b=0.10, k=0.50, c=0.50 and b=0.10, k=0.90, c=0.50) are
not present in the PPO sweep and render as "n/a".

Reads ground truth from
``experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.json``.

Output: ``figures/recalibrated_heatmap.pdf`` (vector PDF, ~6.0in wide).

Run from the version dir::

    uv run --with matplotlib python figures/src/recalibrated_heatmap.py

Or from the repo root::

    uv run --with matplotlib python paper/anvil_pub.bb-workshop.4/figures/src/recalibrated_heatmap.py
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
VERSION_DIR = FIG_DIR.parent  # .../anvil_pub.bb-workshop.4/
PAPER_DIR = VERSION_DIR.parent  # .../paper/
REPO_ROOT = PAPER_DIR.parent  # .../bucket-brigade/

DATA_PATH = (
    REPO_ROOT
    / "experiments"
    / "p3_specialization"
    / "phase_diagram_ppo_v2"
    / "recalibrated_verdict.json"
)
OUTPUT_PATH = FIG_DIR / "recalibrated_heatmap.pdf"


# ----------------------------------------------------------------------------
# Grid layout
# ----------------------------------------------------------------------------

BETAS = [0.1, 0.5, 0.9]  # columns left-to-right
KAPPAS = [0.1, 0.3, 0.5, 0.7, 0.9]  # rows bottom-to-top (kappa-low at bottom)
CS = [0.5, 1.0, 2.0]  # three panels


def load_cells() -> dict[tuple[float, float, float], dict]:
    """Load per-cell values keyed by (beta, kappa, c)."""
    with DATA_PATH.open() as f:
        data = json.load(f)
    out: dict[tuple[float, float, float], dict] = {}
    for cell in data["cells"]:
        tag = cell["cell_tag"]  # e.g., "b0.50_k0.90_c0.50"
        parts = tag.split("_")
        beta = float(parts[0][1:])
        kappa = float(parts[1][1:])
        c = float(parts[2][1:])
        out[(beta, kappa, c)] = cell
    return out


def cell_metric(cell: dict) -> tuple[float, float, bool]:
    """Return (mean, std, is_homogeneous_fallback) for a cell.

    For NE-bearing cells (symmetric_only, asymmetric_only, mixed) use
    gap_closed_ne. For no_convergence cells, fall back to
    gap_closed_homogeneous and flag.
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

    # 3 panels (one per c), each 3x5 cells. Anvil-paper is single-column;
    # ~6.5in wide accommodates three panels at a readable cell pitch.
    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.9), gridspec_kw={"wspace": 0.18})

    cmap = mpl.colormaps["RdYlGn"]
    norm = TwoSlopeNorm(vmin=-0.40, vcenter=0.0, vmax=0.40)

    max_std = max((cell_metric(c)[1] or 0.0) for c in cells.values())

    for panel_idx, c_val in enumerate(CS):
        ax = axes[panel_idx]
        for j, beta in enumerate(BETAS):
            for i, kappa in enumerate(KAPPAS):
                x0, y0 = j, i
                cx, cy = x0 + 0.5, y0 + 0.5

                cell = cells.get((beta, kappa, c_val))
                if cell is None:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (x0, y0),
                            1,
                            1,
                            facecolor="#ececec",
                            edgecolor="#999",
                            linewidth=0.5,
                            hatch="xxx",
                        )
                    )
                    ax.text(
                        cx,
                        cy,
                        "n/a",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="#666",
                        style="italic",
                    )
                    continue

                mean, std, is_fallback = cell_metric(cell)

                color = cmap(norm(mean))
                ax.add_patch(
                    mpatches.Rectangle(
                        (x0, y0),
                        1,
                        1,
                        facecolor=color,
                        edgecolor="#444",
                        linewidth=0.5,
                    )
                )

                if is_fallback:
                    ax.add_patch(
                        mpatches.Rectangle(
                            (x0, y0),
                            1,
                            1,
                            facecolor="none",
                            edgecolor="#a33",
                            linewidth=0.0,
                            hatch="///",
                            alpha=0.7,
                        )
                    )

                label = f"{mean:+.2f}"
                if is_fallback:
                    label = label + r"$^{\dagger}$"
                ax.text(
                    cx,
                    cy + 0.18,
                    label,
                    ha="center",
                    va="center",
                    fontsize=7,
                    fontweight="bold",
                    color="#1a1a1a",
                )

                # Tufte-style horizontal error bar
                bar_y = cy - 0.18
                half = (std / max_std) * 0.40 if max_std > 0 else 0.0
                ax.plot(
                    [cx - half, cx + half],
                    [bar_y, bar_y],
                    color="#1a1a1a",
                    linewidth=0.9,
                    solid_capstyle="butt",
                )
                cap_h = 0.04
                ax.plot(
                    [cx - half, cx - half],
                    [bar_y - cap_h, bar_y + cap_h],
                    color="#1a1a1a",
                    linewidth=0.8,
                )
                ax.plot(
                    [cx + half, cx + half],
                    [bar_y - cap_h, bar_y + cap_h],
                    color="#1a1a1a",
                    linewidth=0.8,
                )
                ax.plot(
                    [cx],
                    [bar_y],
                    marker="o",
                    markersize=2.5,
                    color="#1a1a1a",
                    markeredgecolor="white",
                    markeredgewidth=0.3,
                )

        ax.set_xlim(0, 3)
        ax.set_ylim(0, 5)
        ax.set_aspect("equal")
        ax.set_xticks([0.5, 1.5, 2.5])
        ax.set_xticklabels([f"{b}" for b in BETAS], fontsize=7)
        ax.set_yticks([0.5, 1.5, 2.5, 3.5, 4.5])
        if panel_idx == 0:
            ax.set_yticklabels([f"{k}" for k in KAPPAS], fontsize=7)
            ax.set_ylabel(r"Extinguish $\kappa$", fontsize=8)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel(r"Fire-spread $\beta$", fontsize=8)
        ax.tick_params(axis="both", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(rf"$c{{=}}{c_val}$", fontsize=9, pad=4)

    legend_handles = [
        mpatches.Patch(
            facecolor=cmap(norm(0.30)),
            edgecolor="#444",
            linewidth=0.5,
            label=r"NE-anchored $\mathtt{gap\_closed\_ne}$",
        ),
        mpatches.Patch(
            facecolor=cmap(norm(-0.10)),
            edgecolor="#a33",
            linewidth=0.5,
            hatch="///",
            label=(
                r"fallback $\mathtt{gap\_closed\_homogeneous}$ "
                r"($\dagger$, no NE)"
            ),
        ),
        mpatches.Patch(
            facecolor="#ececec",
            edgecolor="#999",
            linewidth=0.5,
            hatch="xxx",
            label=r"cell not sampled (n/a)",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=3,
        fontsize=6.5,
        frameon=False,
        handlelength=1.5,
        handleheight=0.9,
        borderpad=0.4,
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
