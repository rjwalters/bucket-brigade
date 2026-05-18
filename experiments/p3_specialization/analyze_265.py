"""Analysis for issue #265 — dense Δsafe progress shaping smoke test.

Reads per-cell ``metrics.json`` files under

    experiments/p3_specialization/runs/issue265_progress_signal/coef_<COEF>/seed_<SEED>/

where ``<COEF>`` in ``{0.0, 1.0, 5.0, 25.0}`` and ``<SEED>`` in ``{42, 43, 44}``.
Emits a markdown + JSON summary under

    experiments/p3_specialization/diagnostics/results/issue265_progress_signal/

Pre-registered 3-tier verdict ladder (per the curator enrichment on #265):

* ``gap_closed`` reported for each non-zero ``coef`` (mean across seeds):
    - **Tier 1 (success)**: ``gap_closed >= 0.25`` at any ``coef > 0``.
      Promote to fuller sweep / additional scenarios.
    - **Tier 2 (partial)**: ``0.10 <= gap_closed < 0.25`` (best
      non-zero coef).  Layer with action-shaping (#262) or curriculum
      and re-test.
    - **Tier 3 (failed)**: ``gap_closed < 0.10`` for every cell.
      File follow-up: "dense Δsafe shaping unhelpful — escalate to
      potential-based shaping (#283) for policy-invariant aligned signal".

``gap_closed = (mean_team_reward - random_ref) / (specialist_ref - random_ref)``,
using the published ``minimal_specialization`` references from #260's verdict:
random=-92.921, specialist=-22.072, gap=+70.850.

Usage::

    uv run python experiments/p3_specialization/analyze_265.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Pre-registered coefficient grid (mirrors run_issue265_progress_sweep.sh).
COEFS = [0.0, 1.0, 5.0, 25.0]
SEEDS = [42, 43, 44]
TRAILING_N = 5
SCENARIO = "minimal_specialization"

# Per-step (random, specialist) team-reward references for
# ``minimal_specialization`` (sourced from
# ``experiments/p3_specialization/diagnostics/results/issue260_curriculum/summary.md``).
RANDOM_REF = -92.92147252747253
SPECIALIST_REF = -22.07174358974359
SPEC_RAND_GAP = SPECIALIST_REF - RANDOM_REF  # ≈ +70.85

# Verdict thresholds (per #265 issue body).
TIER_1_THRESHOLD = 0.25  # >= => clear win (any non-zero coef)
TIER_2_THRESHOLD = 0.10  # >= => partial (best non-zero coef)


def _coef_dir_name(coef: float) -> str:
    # Match the directory layout emitted by ``run_issue265_progress_sweep.sh``,
    # which interpolates the bash array value (e.g. ``0.0`` or ``25.0``).
    return f"coef_{coef}"


def _cell_dir(root: Path, coef: float, seed: int) -> Path:
    return root / "issue265_progress_signal" / _coef_dir_name(coef) / f"seed_{seed}"


def _trailing_team(metrics: List[dict], n: int = TRAILING_N) -> float:
    rewards = [row["mean_step_reward_team"] for row in metrics]
    tail = rewards[-n:] if len(rewards) >= n else rewards
    return float(np.mean(tail))


def load_cell(cell: Path) -> Optional[dict]:
    mfile = cell / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    return {
        "cell": str(cell),
        "trailing5_team_reward": _trailing_team(metrics),
        "final_iter": len(metrics) - 1,
        "final_team_reward": float(metrics[-1]["mean_step_reward_team"]),
    }


def gap_closed(per_step: float) -> float:
    return (per_step - RANDOM_REF) / SPEC_RAND_GAP


def aggregate_coef(root: Path, coef: float) -> Dict[str, object]:
    seeds_data: List[Dict] = []
    for s in SEEDS:
        d = load_cell(_cell_dir(root, coef, s))
        if d is None:
            seeds_data.append({"seed": s, "missing": True})
            continue
        d["seed"] = s
        seeds_data.append(d)
    present = [d for d in seeds_data if not d.get("missing")]
    if not present:
        return {"coef": coef, "seeds": seeds_data, "n_seeds": 0}
    team = np.array([d["trailing5_team_reward"] for d in present])
    return {
        "coef": coef,
        "seeds": seeds_data,
        "n_seeds": len(present),
        "team_reward_mean": float(team.mean()),
        "team_reward_std": float(team.std(ddof=1)) if len(team) > 1 else 0.0,
        "team_reward_per_seed": team.tolist(),
        "gap_closed_mean": float(gap_closed(team.mean())),
        "gap_closed_per_seed": [float(gap_closed(x)) for x in team],
    }


def _tier(best_gap_nonzero: Optional[float]) -> str:
    if best_gap_nonzero is None:
        return "missing_treatment"
    if best_gap_nonzero >= TIER_1_THRESHOLD:
        return "tier_1_clear_win"
    if best_gap_nonzero >= TIER_2_THRESHOLD:
        return "tier_2_partial_layer_with_other_shaping"
    return "tier_3_progress_signal_unhelpful_escalate_to_283"


def compute_verdict(results: Dict[float, Dict[str, object]]) -> Dict[str, object]:
    """Verdict based on the best gap_closed across *non-zero* coefficients.

    The baseline (coef=0.0) gap is reported for context but the tiering
    follows the #265 success criterion which is "any coef>0 hits the
    threshold", consistent with the calibration-style readout.
    """
    baseline = results.get(0.0, {})
    baseline_gap = baseline.get("gap_closed_mean") if baseline.get("n_seeds") else None

    treatment_entries = [
        (c, r) for c, r in results.items() if c != 0.0 and r.get("n_seeds")
    ]
    if not treatment_entries:
        return {
            "status": "missing_treatment",
            "baseline_gap": baseline_gap,
            "tier": _tier(None),
        }

    # Track each treatment cell's gap and the best of them.
    per_coef_gap = {c: float(r["gap_closed_mean"]) for c, r in treatment_entries}  # type: ignore[index]
    best_coef = max(per_coef_gap, key=per_coef_gap.get)
    best_gap = per_coef_gap[best_coef]

    delta_vs_baseline: Optional[float] = None
    if baseline_gap is not None:
        delta_vs_baseline = best_gap - float(baseline_gap)

    return {
        "baseline_gap": baseline_gap,
        "per_coef_gap_closed": per_coef_gap,
        "best_coef": best_coef,
        "best_gap_closed": best_gap,
        "delta_vs_baseline": delta_vs_baseline,
        "tier": _tier(best_gap),
    }


def render_markdown(
    results: Dict[float, Dict[str, object]], verdict: Dict[str, object]
) -> str:
    lines = [
        f"# Issue #265 — dense Δsafe progress shaping on `{SCENARIO}`",
        "",
        f"Pre-registered references (per-step mean team reward) on `{SCENARIO}`:",
        "",
        f"- random: `{RANDOM_REF:+.3f}`",
        f"- specialist: `{SPECIALIST_REF:+.3f}`",
        f"- spec-rand gap: `{SPEC_RAND_GAP:+.3f}`",
        "",
        "## Team reward (trailing-5 mean of `mean_step_reward_team`)",
        "",
        "| coef | n | mean ± std | gap_closed | per-seed |",
        "|---|---|---|---|---|",
    ]
    for coef in COEFS:
        r = results.get(coef, {})
        if not r.get("n_seeds"):
            lines.append(f"| {coef} | 0 | — | — | missing |")
            continue
        per_seed = [f"{x:+.3f}" for x in r["team_reward_per_seed"]]  # type: ignore[index]
        lines.append(
            f"| {coef} | {r['n_seeds']} | "
            f"{r['team_reward_mean']:+.3f} ± {r['team_reward_std']:.3f} | "
            f"{r['gap_closed_mean']:+.3f} | "
            f"{per_seed} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if verdict.get("status") == "missing_treatment":
        lines.append("**MISSING_TREATMENT** — no non-zero coef cells found.")
    else:
        baseline_gap = verdict.get("baseline_gap")
        if baseline_gap is not None:
            lines.append(f"- `baseline (coef=0) gap_closed = {baseline_gap:+.3f}`")
        per_coef = verdict.get("per_coef_gap_closed") or {}
        for c in sorted(per_coef.keys()):
            lines.append(f"- `coef={c} gap_closed = {per_coef[c]:+.3f}`")
        lines.append(f"- **best coef**: `{verdict['best_coef']}`")
        lines.append(f"- **best gap_closed**: `{verdict['best_gap_closed']:+.3f}`")
        if verdict.get("delta_vs_baseline") is not None:
            lines.append(
                f"- `delta vs baseline = {verdict['delta_vs_baseline']:+.3f}`"
            )
        lines.append(f"- **Tier: `{verdict['tier']}`**")
    lines.append("")

    lines += [
        "## Tier thresholds",
        "",
        f"- Tier 1 (clear win, escalate): `gap_closed >= {TIER_1_THRESHOLD}` at any `coef>0`",
        f"- Tier 2 (partial, layer): `{TIER_2_THRESHOLD} <= gap_closed < {TIER_1_THRESHOLD}`",
        f"- Tier 3 (failed, escalate to #283): `gap_closed < {TIER_2_THRESHOLD}` for every `coef>0`",
        "",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    runs_root = Path("experiments/p3_specialization/runs")
    out_dir = Path(
        "experiments/p3_specialization/diagnostics/results/issue265_progress_signal"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {coef: aggregate_coef(runs_root, coef) for coef in COEFS}
    verdict = compute_verdict(results)
    payload = {
        "results": {str(k): v for k, v in results.items()},
        "verdict": verdict,
        "baselines": {
            "scenario": SCENARIO,
            "random": RANDOM_REF,
            "specialist": SPECIALIST_REF,
            "spec_rand_gap": SPEC_RAND_GAP,
        },
        "thresholds": {
            "tier_1": TIER_1_THRESHOLD,
            "tier_2": TIER_2_THRESHOLD,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(results, verdict))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    if verdict.get("status") != "missing_treatment":
        print(
            f"verdict: {verdict['tier']} "
            f"(best gap_closed={verdict['best_gap_closed']:+.3f} "
            f"at coef={verdict['best_coef']})"
        )


if __name__ == "__main__":
    main()
