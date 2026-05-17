"""Analysis for issue #261/#262 — action-shaping calibration sweep.

Reads per-cell ``metrics.json`` and ``config.json`` files under

    experiments/p3_specialization/runs/issue261_calibration/
        alpha_{ALPHA}/beta_{BETA}/seed_{SEED}/

For each ``(alpha, beta)`` cell aggregated over seeds, computes:

* ``trailing5_team_reward`` — mean of last-5 ``mean_step_reward_team`` rows
  (matches ``analyze_231._trailing_team``).
* ``gap_closed`` — ``(trailing5 - random) / (specialist - random)`` using the
  ``minimal_specialization`` references from ``analyze_231``
  (random=-87.72, specialist=-22.07; denominator=65.65).
* ``mean_action_entropy_final`` — from ``metrics[-1]['action_entropy/mean']``
  (already logged by ``train.py``).
* ``entropy_collapse_multiple`` — ``baseline_entropy / cell_entropy`` where
  baseline is the in-sweep ``(alpha=0, beta=0)`` cell. Flagged if > 100×
  (MAPPO's collapse was 1874× per #257 — a 100× threshold is conservative).

The best cell is the one with the largest ``gap_closed_mean``. Verdict
ladder is then applied per the issue body:

| Outcome | tier label |
|---|---|
| best gap_closed ≥ 0.50 | ``tier_1_breaks_plateau`` |
| best gap_closed in [0.25, 0.50) | ``tier_2_partial`` |
| best gap_closed < 0.25 | ``tier_3_insufficient`` |
| over-shaping flag fires | annotated alongside the tier |

Writes summary.{json,md} under
``experiments/p3_specialization/diagnostics/results/issue261_calibration/``.

Usage::

    uv run python experiments/p3_specialization/analyze_261_calibration.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


# Pre-registered references (per-step mean team reward) from
# ``analyze_231.BASELINES['minimal_specialization']``. Sources:
#   random:     PR #244 (issue #237 post-#236 re-derivation)
#   specialist: PR #243 (issue #238 post-#236 re-derivation)
# Denominator: specialist - random = 65.65
MINSPEC_RANDOM = -87.72
MINSPEC_SPECIALIST = -22.07

SEEDS = [42, 43, 44]
TRAILING_N = 5

# In-sweep baseline cell coordinates. The baseline gap_closed lands near
# the PR #257 IPPO obs-fix verdict (~0.182) when alpha=beta=0.
BASELINE_ALPHA = 0.0
BASELINE_BETA = 0.0

# Over-shaping threshold: entropy collapse multiple over the baseline.
# MAPPO collapse was 1874× per #257; 100× is the curator-recommended
# conservative flag.
OVER_SHAPING_ENTROPY_MULTIPLE = 100.0


def _fmt_alpha(alpha: float) -> str:
    """Match the run-dir naming used by ``run_issue262_sweep.sh``."""
    return f"{alpha:g}"


def _fmt_beta(beta: float) -> str:
    return f"{beta:g}"


def _cell_dir(root: Path, alpha: float, beta: float, seed: int) -> Path:
    return (
        root
        / "issue261_calibration"
        / f"alpha_{_fmt_alpha(alpha)}"
        / f"beta_{_fmt_beta(beta)}"
        / f"seed_{seed}"
    )


def _trailing_team(metrics: List[dict], n: int = TRAILING_N) -> float:
    rewards = [row["mean_step_reward_team"] for row in metrics]
    tail = rewards[-n:] if len(rewards) >= n else rewards
    return float(np.mean(tail))


def gap_closed(trailing5_team: float) -> float:
    return (trailing5_team - MINSPEC_RANDOM) / (MINSPEC_SPECIALIST - MINSPEC_RANDOM)


def load_cell(cell: Path) -> Optional[dict]:
    mfile = cell / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    if not metrics:
        return None
    last = metrics[-1]
    return {
        "cell": str(cell),
        "trailing5_team_reward": _trailing_team(metrics),
        "final_iter": len(metrics) - 1,
        "mean_action_entropy_final": float(
            last.get("action_entropy/mean", float("nan"))
        ),
    }


def aggregate_cell(root: Path, alpha: float, beta: float) -> Dict[str, object]:
    seeds_data: List[Dict] = []
    for s in SEEDS:
        cell = _cell_dir(root, alpha, beta, s)
        d = load_cell(cell)
        if d is None:
            seeds_data.append({"seed": s, "missing": True})
            continue
        d["seed"] = s
        seeds_data.append(d)
    present = [d for d in seeds_data if not d.get("missing")]
    out: Dict[str, object] = {
        "alpha": alpha,
        "beta": beta,
        "seeds": seeds_data,
        "n_seeds": len(present),
    }
    if not present:
        return out
    team = np.array([d["trailing5_team_reward"] for d in present])
    ents = np.array([d["mean_action_entropy_final"] for d in present])
    out["team_reward_mean"] = float(team.mean())
    out["team_reward_std"] = float(team.std(ddof=1)) if len(team) > 1 else 0.0
    out["team_reward_per_seed"] = team.tolist()
    out["mean_entropy_final_mean"] = float(np.nanmean(ents))
    out["gap_closed_mean"] = float(gap_closed(team.mean()))
    out["gap_closed_per_seed"] = [float(gap_closed(x)) for x in team]
    return out


def _verdict_tier(best_gap: float) -> str:
    if best_gap >= 0.50:
        return "tier_1_breaks_plateau"
    if best_gap >= 0.25:
        return "tier_2_partial"
    return "tier_3_insufficient"


def compute_verdict(
    cells: List[Dict[str, object]],
) -> Dict[str, object]:
    """Apply the pre-registered ladder + over-shaping check.

    Picks the best ``(alpha, beta)`` cell by ``gap_closed_mean``. Flags
    any cell whose mean action entropy is < 1/100th of the baseline
    cell's. The verdict is computed from the best gap; over-shaping is
    reported alongside (does not change the tier itself).
    """
    # Locate the in-sweep baseline (alpha=0, beta=0).
    baseline = next(
        (
            c
            for c in cells
            if c.get("alpha") == BASELINE_ALPHA and c.get("beta") == BASELINE_BETA
        ),
        None,
    )
    baseline_entropy: Optional[float] = None
    if baseline is not None and baseline.get("n_seeds", 0) > 0:
        baseline_entropy = float(baseline.get("mean_entropy_final_mean", float("nan")))

    # Annotate each cell with entropy collapse multiple vs baseline.
    over_shaping_flagged: List[Tuple[float, float, float]] = []
    for c in cells:
        if c.get("n_seeds", 0) == 0:
            c["entropy_collapse_multiple"] = None
            continue
        cell_ent = float(c.get("mean_entropy_final_mean", float("nan")))
        if (
            baseline_entropy is None
            or not np.isfinite(baseline_entropy)
            or baseline_entropy <= 0.0
            or not np.isfinite(cell_ent)
            or cell_ent <= 0.0
        ):
            c["entropy_collapse_multiple"] = None
            continue
        multiple = baseline_entropy / cell_ent
        c["entropy_collapse_multiple"] = float(multiple)
        if multiple > OVER_SHAPING_ENTROPY_MULTIPLE:
            over_shaping_flagged.append(
                (float(c["alpha"]), float(c["beta"]), float(multiple))
            )

    # Pick the best cell by gap_closed_mean (among cells with data).
    cells_with_data = [c for c in cells if c.get("n_seeds", 0) > 0]
    best_cell: Optional[Dict[str, object]] = None
    if cells_with_data:
        best_cell = max(
            cells_with_data, key=lambda c: c.get("gap_closed_mean", float("-inf"))
        )

    if best_cell is None:
        return {
            "best_cell": None,
            "tier": "no_data",
            "headline": "NO_DATA",
            "baseline_entropy": baseline_entropy,
            "over_shaping_flagged": [],
        }

    best_gap = float(best_cell["gap_closed_mean"])
    tier = _verdict_tier(best_gap)

    if tier == "tier_1_breaks_plateau":
        headline = "ACTION_SHAPING_BREAKS_PLATEAU"
    elif tier == "tier_2_partial":
        headline = "ACTION_SHAPING_PARTIAL"
    else:
        headline = "ACTION_SHAPING_INSUFFICIENT"
    if over_shaping_flagged:
        headline = f"{headline}__OVER_SHAPING_FLAGGED"

    return {
        "best_cell": {
            "alpha": best_cell["alpha"],
            "beta": best_cell["beta"],
            "gap_closed_mean": best_gap,
            "team_reward_mean": best_cell.get("team_reward_mean"),
            "mean_entropy_final_mean": best_cell.get("mean_entropy_final_mean"),
            "entropy_collapse_multiple": best_cell.get("entropy_collapse_multiple"),
            "n_seeds": best_cell.get("n_seeds"),
        },
        "tier": tier,
        "headline": headline,
        "baseline_entropy": baseline_entropy,
        "over_shaping_flagged": [
            {"alpha": a, "beta": b, "entropy_collapse_multiple": m}
            for (a, b, m) in over_shaping_flagged
        ],
    }


def render_markdown(cells: List[Dict[str, object]], verdict: Dict[str, object]) -> str:
    lines = [
        "# Issue #261/#262 — Action-shaping calibration sweep",
        "",
        "Scenario: ``minimal_specialization``  ",
        f"References (per-step mean team reward): "
        f"random={MINSPEC_RANDOM:+.2f}, specialist={MINSPEC_SPECIALIST:+.2f} "
        f"(denominator={MINSPEC_SPECIALIST - MINSPEC_RANDOM:+.2f}).",
        "",
        "## Per-cell results",
        "",
        "| alpha | beta | n | team_reward (mean±std) | gap_closed_mean | "
        "mean_action_entropy_final | entropy_collapse_x |",
        "|---|---|---|---|---|---|---|",
    ]
    for c in cells:
        alpha = c.get("alpha")
        beta = c.get("beta")
        n = c.get("n_seeds", 0)
        if n == 0:
            lines.append(f"| {alpha} | {beta} | 0 | missing | — | — | — |")
            continue
        team_mean = c["team_reward_mean"]
        team_std = c["team_reward_std"]
        gap = c["gap_closed_mean"]
        ent = c["mean_entropy_final_mean"]
        coll = c.get("entropy_collapse_multiple")
        coll_s = f"{coll:.1f}" if isinstance(coll, (int, float)) else "—"
        lines.append(
            f"| {alpha} | {beta} | {n} | {team_mean:+.2f} ± {team_std:.2f} | "
            f"{gap:+.3f} | {ent:.3f} | {coll_s} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        f"**Headline**: ``{verdict['headline']}``",
        f"**Tier**: ``{verdict['tier']}``",
    ]
    best = verdict.get("best_cell")
    if best is not None:
        lines += [
            "",
            "Best cell:",
            f"- alpha = ``{best['alpha']}``, beta = ``{best['beta']}``",
            f"- gap_closed_mean = ``{best['gap_closed_mean']:+.3f}``",
            f"- team_reward_mean = ``{best.get('team_reward_mean'):+.3f}``",
            f"- mean_action_entropy_final = ``{best.get('mean_entropy_final_mean'):.3f}``",
            f"- n_seeds = ``{best.get('n_seeds')}``",
        ]
    over = verdict.get("over_shaping_flagged") or []
    if over:
        lines += [
            "",
            f"**Over-shaping flagged** (entropy collapse > "
            f"{OVER_SHAPING_ENTROPY_MULTIPLE:g}× baseline):",
            "",
            "| alpha | beta | entropy_collapse_multiple |",
            "|---|---|---|",
        ]
        for entry in over:
            lines.append(
                f"| {entry['alpha']} | {entry['beta']} | "
                f"{entry['entropy_collapse_multiple']:.1f} |"
            )
    else:
        lines += [
            "",
            "_No over-shaping flagged at the "
            f"{OVER_SHAPING_ENTROPY_MULTIPLE:g}× threshold._",
        ]
    lines.append("")
    return "\n".join(lines) + "\n"


def _discover_cells(root: Path) -> List[Tuple[float, float]]:
    """Discover (alpha, beta) pairs by listing the run-dir layout.

    Returns a sorted list of (alpha, beta) pairs found on disk. Falls
    back to an empty list if the calibration root doesn't exist yet.
    """
    cal_root = root / "issue261_calibration"
    if not cal_root.is_dir():
        return []
    pairs: List[Tuple[float, float]] = []
    for alpha_dir in sorted(cal_root.iterdir()):
        if not alpha_dir.is_dir() or not alpha_dir.name.startswith("alpha_"):
            continue
        try:
            alpha = float(alpha_dir.name[len("alpha_") :])
        except ValueError:
            continue
        for beta_dir in sorted(alpha_dir.iterdir()):
            if not beta_dir.is_dir() or not beta_dir.name.startswith("beta_"):
                continue
            try:
                beta = float(beta_dir.name[len("beta_") :])
            except ValueError:
                continue
            pairs.append((alpha, beta))
    return sorted(set(pairs))


def main() -> None:
    runs_root = Path("experiments/p3_specialization/runs")
    out_dir = Path(
        "experiments/p3_specialization/diagnostics/results/issue261_calibration"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    pairs = _discover_cells(runs_root)
    if not pairs:
        # Fallback: pre-registered grid from the issue body.
        alphas = [0.0, 0.1, 0.5, 2.0]
        betas = [0.0, 0.1, 0.5]
        pairs = sorted({(a, b) for a in alphas for b in betas})

    cells = [aggregate_cell(runs_root, a, b) for (a, b) in pairs]
    verdict = compute_verdict(cells)
    payload = {
        "cells": cells,
        "verdict": verdict,
        "baselines": {
            "random": MINSPEC_RANDOM,
            "specialist": MINSPEC_SPECIALIST,
            "denominator": MINSPEC_SPECIALIST - MINSPEC_RANDOM,
        },
        "thresholds": {
            "tier_1": 0.50,
            "tier_2": 0.25,
            "over_shaping_entropy_multiple": OVER_SHAPING_ENTROPY_MULTIPLE,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(cells, verdict))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"headline verdict: {verdict['headline']}")
    if verdict.get("best_cell"):
        b = verdict["best_cell"]
        print(
            f"best cell: alpha={b['alpha']}, beta={b['beta']}, "
            f"gap_closed_mean={b['gap_closed_mean']:+.3f}"
        )


if __name__ == "__main__":
    main()
