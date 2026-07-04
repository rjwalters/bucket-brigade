"""Render the Bucket Brigade Nash phase diagram (Figure 1).

Reads experiments/nash/phase_diagram/results.json and produces a
categorical 3-panel heatmap (one panel per c in {0.5, 1.0, 2.0}),
each panel a 3x5 beta-by-kappa grid colored by Nash verdict.

Replaces the v4/v5 carry-forward PNG that showed raw payoff numerics
instead of the four regime classes. The figure is the visual anchor
for the body section 3 headline claim about four empirical NE
regimes; this version makes the four regimes visible at a glance.

Usage:
    python figures/src/phase_diagram.py
        --in  experiments/nash/phase_diagram/results.json
        --out figures/phase_diagram.pdf
"""

import argparse
import json
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle


# Verdict -> ordinal index (controls colormap slot and legend order)
VERDICT_ORDER = [
    "no_convergence",
    "symmetric_only",
    "mixed",
    "asymmetric_only",
]

# Colorblind-safe categorical palette (matches the v5 legend conventions
# the body prose already names: gray for collapse, blue for symmetric,
# teal for mixed, pale yellow for asymmetric).
VERDICT_COLORS = {
    "no_convergence":  "#bdbdbd",
    "symmetric_only":  "#2c7bb6",
    "mixed":           "#7fcdbb",
    "asymmetric_only": "#f4e285",
}

# Pretty labels for the legend (typeset names match the prose)
VERDICT_LABELS = {
    "no_convergence":  "no_convergence",
    "symmetric_only":  "symmetric_only",
    "mixed":           "mixed",
    "asymmetric_only": "asymmetric_only",
}

# Grid axes (must match the Nash sweep schedule)
KAPPAS = [0.1, 0.3, 0.5, 0.7, 0.9]
BETAS = [0.1, 0.5, 0.9]
COSTS = [0.5, 1.0, 2.0]


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[4]
    default_in = repo_root / "experiments/nash/phase_diagram/results.json"
    default_out = (
        Path(__file__).resolve().parent.parent / "phase_diagram.pdf"
    )
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", type=Path, default=default_in)
    p.add_argument("--out", dest="out_path", type=Path, default=default_out)
    return p.parse_args()


def load_grid(in_path: Path) -> dict:
    """Return {(beta, kappa, c): verdict_string}; unsampled cells absent."""
    with in_path.open() as f:
        data = json.load(f)
    return {(c["beta"], c["kappa"], c["c"]): c["verdict"] for c in data["cells"]}


def render(grid: dict, out_path: Path) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(7.5, 2.6), sharey=True, constrained_layout=True
    )
    # Map verdicts to small integers for the colormap; -1 = unsampled.
    verdict_to_idx = {v: i for i, v in enumerate(VERDICT_ORDER)}
    cmap = ListedColormap([VERDICT_COLORS[v] for v in VERDICT_ORDER])

    for ax, cost in zip(axes, COSTS):
        # Build a 3 (beta) x 5 (kappa) grid of verdict indices; rows are
        # beta values ascending bottom-to-top so beta=0.1 is at the bottom
        # like a conventional heatmap.
        rows = []
        for beta in BETAS:
            row = []
            for kappa in KAPPAS:
                v = grid.get((beta, kappa, cost))
                row.append(verdict_to_idx[v] if v is not None else -1)
            rows.append(row)

        # Draw each cell as a colored Rectangle. Unsampled cells get a
        # diagonal-hatched gray with an "n/a" label.
        for i, beta in enumerate(BETAS):
            for j, kappa in enumerate(KAPPAS):
                idx = rows[i][j]
                if idx == -1:
                    ax.add_patch(Rectangle(
                        (j, i), 1, 1,
                        facecolor="#f0f0f0",
                        edgecolor="white",
                        linewidth=1.5,
                        hatch="//",
                    ))
                    ax.text(j + 0.5, i + 0.5, "n/a",
                            ha="center", va="center",
                            fontsize=7, color="#666666")
                else:
                    verdict = VERDICT_ORDER[idx]
                    ax.add_patch(Rectangle(
                        (j, i), 1, 1,
                        facecolor=VERDICT_COLORS[verdict],
                        edgecolor="white",
                        linewidth=1.5,
                    ))

        ax.set_xlim(0, len(KAPPAS))
        ax.set_ylim(0, len(BETAS))
        ax.set_xticks([k + 0.5 for k in range(len(KAPPAS))])
        ax.set_xticklabels([f"{k:g}" for k in KAPPAS], fontsize=8)
        ax.set_yticks([b + 0.5 for b in range(len(BETAS))])
        ax.set_yticklabels([f"{b:g}" for b in BETAS], fontsize=8)
        ax.set_xlabel(r"$\kappa$ (extinguish prob)", fontsize=9)
        ax.set_title(f"$c={cost:g}$", fontsize=10)
        ax.set_aspect("equal")
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)

    axes[0].set_ylabel(r"$\beta$ (spread prob)", fontsize=9)

    # Shared categorical legend at the bottom.
    handles = [
        mpatches.Patch(facecolor=VERDICT_COLORS[v], edgecolor="white",
                       label=VERDICT_LABELS[v])
        for v in VERDICT_ORDER
    ]
    handles.append(mpatches.Patch(
        facecolor="#f0f0f0", edgecolor="white", hatch="//", label="n/a (unsampled)"
    ))
    fig.legend(
        handles=handles, loc="lower center", ncol=5,
        frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.05),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.05)
    print(f"wrote {out_path}")


def main() -> None:
    args = parse_args()
    grid = load_grid(args.in_path)
    render(grid, args.out_path)


if __name__ == "__main__":
    main()
