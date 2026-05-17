"""Analysis for issue #260 — episode-length curriculum smoke test.

Reads per-cell ``metrics.json`` files under

    experiments/p3_specialization/runs/issue260_<arm>/seed_<N>/

where ``<arm>`` is ``baseline`` or ``curriculum`` and ``N`` in ``{0, 1, 2}``.
Emits a markdown + JSON summary under

    experiments/p3_specialization/diagnostics/results/issue260_curriculum/

Pre-registered 3-tier verdict ladder (per the curator enrichment on #260):

* gap-closed delta ``(curriculum − baseline)``:
    - tier 1 ``>= 0.50`` — clear win; production sweep follow-up
    - tier 2 ``[0.25, 0.50)`` — combine with intervention #2 (action shaping #261)
    - tier 3 ``< 0.25`` — curriculum unhelpful; promote intervention #4
      (dense progress signal)

The final-iter team reward is what the verdict is computed on. Because the
curriculum's last phase here uses the canonical ``min_nights=12`` (the
native floor of ``minimal_specialization``), the final-iter measurement
is on the same episode-length regime as the baseline — apples-to-apples.

Usage::

    uv run python experiments/p3_specialization/analyze_260.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

ARMS = ["baseline", "curriculum"]
SEEDS = [0, 1, 2]
NUM_AGENTS = 4
TRAILING_N = 5
SCENARIO = "minimal_specialization"

# Per-step (random, specialist) team-reward references for
# ``minimal_specialization`` (issue199_minspec baselines, the canonical
# 4-agent specialist policy with honest signaling, seed=42, n=50 episodes).
# Sourced from
# ``experiments/p3_specialization/diagnostics/results/issue199_minspec/baselines.json``.
RANDOM_REF = -92.92147252747253
SPECIALIST_REF = -22.07174358974359
SPEC_RAND_GAP = SPECIALIST_REF - RANDOM_REF  # ≈ +70.85


def _cell_dir(root: Path, arm: str, seed: int) -> Path:
    return root / f"issue260_{arm}" / f"seed_{seed}"


def _trailing_team(metrics: List[dict], n: int = TRAILING_N) -> float:
    rewards = [row["mean_step_reward_team"] for row in metrics]
    tail = rewards[-n:] if len(rewards) >= n else rewards
    return float(np.mean(tail))


def load_cell(cell: Path) -> Optional[dict]:
    mfile = cell / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    floors = [int(row.get("min_nights_floor", -1)) for row in metrics]
    return {
        "cell": str(cell),
        "trailing5_team_reward": _trailing_team(metrics),
        "final_iter": len(metrics) - 1,
        "final_team_reward": float(metrics[-1]["mean_step_reward_team"]),
        "min_nights_floor_first": floors[0] if floors else None,
        "min_nights_floor_last": floors[-1] if floors else None,
        "min_nights_floor_unique": sorted(set(floors)),
    }


def gap_closed(per_step: float) -> float:
    return (per_step - RANDOM_REF) / SPEC_RAND_GAP


def aggregate_arm(root: Path, arm: str) -> Dict[str, object]:
    seeds_data: List[Dict] = []
    for s in SEEDS:
        d = load_cell(_cell_dir(root, arm, s))
        if d is None:
            seeds_data.append({"seed": s, "missing": True})
            continue
        d["seed"] = s
        seeds_data.append(d)
    present = [d for d in seeds_data if not d.get("missing")]
    if not present:
        return {"seeds": seeds_data, "n_seeds": 0}
    team = np.array([d["trailing5_team_reward"] for d in present])
    return {
        "seeds": seeds_data,
        "n_seeds": len(present),
        "team_reward_mean": float(team.mean()),
        "team_reward_std": float(team.std(ddof=1)) if len(team) > 1 else 0.0,
        "team_reward_per_seed": team.tolist(),
        "gap_closed_mean": float(gap_closed(team.mean())),
        "gap_closed_per_seed": [float(gap_closed(x)) for x in team],
    }


def _tier(delta: float) -> str:
    if delta >= 0.50:
        return "tier_1_clear_win"
    if delta >= 0.25:
        return "tier_2_combine_with_action_shaping"
    return "tier_3_curriculum_unhelpful"


def compute_verdict(results: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    baseline = results["baseline"]
    curriculum = results["curriculum"]
    if not (baseline.get("n_seeds") and curriculum.get("n_seeds")):
        return {"status": "missing_arm"}
    gap_baseline = baseline["gap_closed_mean"]  # type: ignore[index]
    gap_curriculum = curriculum["gap_closed_mean"]  # type: ignore[index]
    delta = float(gap_curriculum) - float(gap_baseline)
    return {
        "gap_baseline": float(gap_baseline),
        "gap_curriculum": float(gap_curriculum),
        "delta": delta,
        "tier": _tier(delta),
    }


def render_markdown(
    results: Dict[str, Dict[str, object]], verdict: Dict[str, object]
) -> str:
    lines = [
        f"# Issue #260 — episode-length curriculum on `{SCENARIO}`",
        "",
        f"Pre-registered references (per-step mean team reward) on `{SCENARIO}`:",
        "",
        f"- random: `{RANDOM_REF:+.3f}`",
        f"- specialist: `{SPECIALIST_REF:+.3f}`",
        f"- spec-rand gap: `{SPEC_RAND_GAP:+.3f}`",
        "",
        "## Team reward (trailing-5 mean of `mean_step_reward_team`)",
        "",
        "| arm | n | mean ± std | per-seed |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        r = results.get(arm, {})
        if not r.get("n_seeds"):
            lines.append(f"| {arm} | 0 | — | missing |")
            continue
        per_seed = [f"{x:+.3f}" for x in r["team_reward_per_seed"]]  # type: ignore[index]
        lines.append(
            f"| {arm} | {r['n_seeds']} | "
            f"{r['team_reward_mean']:+.3f} ± {r['team_reward_std']:.3f} | "
            f"{per_seed} |"
        )

    lines += [
        "",
        "## Curriculum floors observed",
        "",
        "| arm | seed | floors_unique | first | last |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        for d in results.get(arm, {}).get("seeds", []):  # type: ignore[union-attr]
            if d.get("missing"):
                lines.append(f"| {arm} | {d['seed']} | — | — | — |")
                continue
            lines.append(
                f"| {arm} | {d['seed']} | "
                f"{d['min_nights_floor_unique']} | "
                f"{d['min_nights_floor_first']} | "
                f"{d['min_nights_floor_last']} |"
            )

    lines += [
        "",
        "## Verdict",
        "",
    ]
    if verdict.get("status") == "missing_arm":
        lines.append("**MISSING_ARM** — cannot compute verdict.")
    else:
        lines.append(f"- `gap_baseline = {verdict['gap_baseline']:+.3f}`")
        lines.append(f"- `gap_curriculum = {verdict['gap_curriculum']:+.3f}`")
        lines.append(f"- `delta = {verdict['delta']:+.3f}`")
        lines.append(f"- **Tier: `{verdict['tier']}`**")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    runs_root = Path("experiments/p3_specialization/runs")
    out_dir = Path(
        "experiments/p3_specialization/diagnostics/results/issue260_curriculum"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {arm: aggregate_arm(runs_root, arm) for arm in ARMS}
    verdict = compute_verdict(results)
    payload = {
        "results": results,
        "verdict": verdict,
        "baselines": {
            "scenario": SCENARIO,
            "random": RANDOM_REF,
            "specialist": SPECIALIST_REF,
            "spec_rand_gap": SPEC_RAND_GAP,
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(results, verdict))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    if verdict.get("status") != "missing_arm":
        print(f"verdict: {verdict['tier']} (delta={verdict['delta']:+.3f})")


if __name__ == "__main__":
    main()
