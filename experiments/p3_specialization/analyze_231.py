"""Analysis for issue #231 — paired IPPO vs MAPPO comparison across three
scenarios.

Reads per-cell ``metrics.json`` (and optional ``pairwise_action_kl.json``)
files under

    experiments/p3_specialization/runs/issue231/<arm>/<scenario>/seed_<N>/

where ``<arm>`` is ``ippo`` or ``mappo`` and ``<scenario>`` is one of
``default``, ``minimal_specialization``, ``positional_default``. Emits a
markdown + JSON summary under

    experiments/p3_specialization/diagnostics/results/issue231_mappo/

Mirrors the shape of ``analyze_220.py`` but generalised to three scenarios
and computes gap-closure on all three. The arm naming is ``ippo`` /
``mappo`` (vs ``baseline`` / ``treatment`` in #220) and the run-dir layout
omits the ``lambda_0e0`` subdir (issue #231 fixes ``lambda_red=0`` for
every cell — see the curator enrichment on the issue).

Verdict ladder (pre-registered, applied per scenario then aggregated):

* per-scenario gap-closed delta ``(mappo - ippo)``:
    - tier 1 ``≥0.50`` — MAPPO succeeds for that scenario;
    - tier 2 ``[0.25, 0.50)`` — partial / helps but does not solve;
    - tier 3 ``<0.25`` — insufficient;
    - tier ``< -0.10`` — MAPPO harmful (regression).
* headline verdict:
    - ``MAPPO_SUCCEEDS`` iff ≥2 of 3 scenarios at tier 1
    - ``MAPPO_HELPS_GLOBALLY_NOT_DECISIVE`` iff all 3 scenarios at tier 2
    - ``PARTIAL_SCENARIO_DEPENDENT`` iff some pass and some fail (any tier 1
      and any tier 3) but headline thresholds not met
    - ``MAPPO_HARMFUL_ON_<SCEN>`` if any scenario regresses (overrides
      everything else; we flag rather than declare success)
    - ``INSUFFICIENT`` otherwise

Usage::

    uv run python experiments/p3_specialization/analyze_231.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


ARMS = ["ippo", "mappo"]
SCENARIOS = ["default", "minimal_specialization", "positional_default"]
SEEDS = [42, 43, 44]
NUM_AGENTS = 4
TRAILING_N = 5

# Per-scenario (random, specialist) per-step references. Sources:
#   random: experiments/p3_specialization/diagnostics/results/issue237_postmerge/
#           (post-#236 re-derivation at commit dffe1060, n=1000 each;
#           issue #237). Replaces the pre-#236 n=50 figures from
#           issue199_minspec/baselines.json and issue221_positional/baselines.json.
#   specialist: experiments/p3_specialization/diagnostics/results/issue199_minspec/baselines.json
#               and issue221_positional/baselines.json (specialist provenance is
#               separate from random; specialist baselines tracked by sibling
#               issue #238 family, NOT updated here).
BASELINES: Dict[str, Tuple[float, float]] = {
    "default": (251.23, 320.94),
    "minimal_specialization": (-87.72, -22.07),
    "positional_default": (250.73, 320.89),
}


def _cell_dir(root: Path, arm: str, scenario: str, seed: int) -> Path:
    return root / "issue231" / arm / scenario / f"seed_{seed}"


def _trailing_team(metrics: List[dict], n: int = TRAILING_N) -> float:
    rewards = [row["mean_step_reward_team"] for row in metrics]
    tail = rewards[-n:] if len(rewards) >= n else rewards
    return float(np.mean(tail))


def _per_agent_entropy_final(metrics: List[dict]) -> Tuple[List[float], float]:
    last = metrics[-1]
    per = [float(last[f"action_entropy/agent_{i}"]) for i in range(NUM_AGENTS)]
    mean = float(last["action_entropy/mean"])
    return per, mean


def _load_kl(cell: Path) -> Optional[dict]:
    p = cell / "pairwise_action_kl.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_cell(cell: Path) -> Optional[dict]:
    mfile = cell / "metrics.json"
    if not mfile.exists():
        return None
    metrics = json.loads(mfile.read_text())
    per_ent, mean_ent = _per_agent_entropy_final(metrics)
    kl = _load_kl(cell)
    return {
        "cell": str(cell),
        "trailing5_team_reward": _trailing_team(metrics),
        "final_iter": len(metrics) - 1,
        "per_agent_entropy_final": per_ent,
        "mean_entropy_final": mean_ent,
        "entropy_spread": float(np.max(per_ent) - np.min(per_ent)),
        "kl_off_diag_mean": (kl["kl_off_diag_mean"] if kl else None),
        "kl_off_diag_max": (kl["kl_off_diag_max"] if kl else None),
    }


def gap_closed(scenario: str, ppo_trailing5: float) -> float:
    rand, spec = BASELINES[scenario]
    return (ppo_trailing5 - rand) / (spec - rand)


def aggregate_arm(root: Path, arm: str) -> Dict[str, Dict]:
    per_scenario: Dict[str, Dict] = {}
    for scenario in SCENARIOS:
        seeds_data: List[Dict] = []
        for s in SEEDS:
            cell = _cell_dir(root, arm, scenario, s)
            d = load_cell(cell)
            if d is None:
                seeds_data.append({"seed": s, "missing": True})
                continue
            d["seed"] = s
            seeds_data.append(d)
        present = [d for d in seeds_data if not d.get("missing")]
        if not present:
            per_scenario[scenario] = {"seeds": seeds_data, "n_seeds": 0}
            continue
        team = np.array([d["trailing5_team_reward"] for d in present])
        ents = np.array([d["mean_entropy_final"] for d in present])
        spreads = np.array([d["entropy_spread"] for d in present])
        kl_means = [
            d["kl_off_diag_mean"] for d in present if d["kl_off_diag_mean"] is not None
        ]
        per_scenario[scenario] = {
            "seeds": seeds_data,
            "n_seeds": len(present),
            "team_reward_mean": float(team.mean()),
            "team_reward_std": float(team.std(ddof=1)) if len(team) > 1 else 0.0,
            "team_reward_per_seed": team.tolist(),
            "mean_entropy_mean": float(ents.mean()),
            "entropy_spread_mean": float(spreads.mean()),
            "kl_off_diag_mean": (float(np.mean(kl_means)) if kl_means else None),
            "gap_closed_mean": float(gap_closed(scenario, team.mean())),
            "gap_closed_per_seed": [float(gap_closed(scenario, x)) for x in team],
        }
    return per_scenario


def _per_scenario_tier(delta: float) -> str:
    if delta < -0.10:
        return "harmful"
    if delta >= 0.50:
        return "tier_1_succeeds"
    if delta >= 0.25:
        return "tier_2_partial"
    return "tier_3_insufficient"


def compute_verdict(results: Dict[str, Dict[str, Dict]]) -> Dict[str, object]:
    """Per-scenario gap-deltas + headline verdict per the pre-reg ladder."""
    per_scen: Dict[str, Dict[str, float]] = {}
    harmful_scens: List[str] = []
    tier_counts = {"tier_1_succeeds": 0, "tier_2_partial": 0, "tier_3_insufficient": 0}
    for scenario in SCENARIOS:
        ippo = results["ippo"].get(scenario, {})
        mappo = results["mappo"].get(scenario, {})
        if not (ippo.get("n_seeds") and mappo.get("n_seeds")):
            per_scen[scenario] = {"status": "missing"}
            continue
        gap_ippo = ippo["gap_closed_mean"]
        gap_mappo = mappo["gap_closed_mean"]
        delta = gap_mappo - gap_ippo
        tier = _per_scenario_tier(delta)
        if tier == "harmful":
            harmful_scens.append(scenario)
        else:
            tier_counts[tier] += 1
        per_scen[scenario] = {
            "gap_ippo": gap_ippo,
            "gap_mappo": gap_mappo,
            "delta": delta,
            "tier": tier,
        }

    n_tier_1 = tier_counts["tier_1_succeeds"]
    n_tier_2 = tier_counts["tier_2_partial"]
    n_tier_3 = tier_counts["tier_3_insufficient"]

    if harmful_scens:
        headline = "MAPPO_HARMFUL_ON_" + ",".join(s.upper() for s in harmful_scens)
    elif n_tier_1 >= 2:
        headline = "MAPPO_SUCCEEDS"
    elif n_tier_2 == 3:
        headline = "MAPPO_HELPS_GLOBALLY_NOT_DECISIVE"
    elif n_tier_1 >= 1 and n_tier_3 >= 1:
        headline = "PARTIAL_SCENARIO_DEPENDENT"
    elif n_tier_3 == 3:
        headline = "INSUFFICIENT"
    else:
        headline = "MIXED_SUB_THRESHOLD"

    return {
        "per_scenario": per_scen,
        "harmful_scenarios": harmful_scens,
        "tier_counts": tier_counts,
        "headline_verdict": headline,
    }


def render_markdown(
    results: Dict[str, Dict[str, Dict]], verdict: Dict[str, object]
) -> str:
    lines = [
        "# Issue #231 — IPPO vs MAPPO across three scenarios",
        "",
        "Pre-registered references (per-step mean team reward):",
        "",
        "| scenario | random | specialist | spec-rand gap |",
        "|---|---|---|---|",
    ]
    for s in SCENARIOS:
        rand, spec = BASELINES[s]
        lines.append(f"| {s} | {rand:+.2f} | {spec:+.2f} | {spec - rand:+.2f} |")
    lines += [
        "",
        "## Team reward (trailing-5 mean of `mean_step_reward_team`)",
        "",
        "| scenario | arm | n | mean ± std | per-seed |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for arm in ARMS:
            r = results[arm].get(scenario, {})
            if not r.get("n_seeds"):
                lines.append(f"| {scenario} | {arm} | 0 | — | missing |")
                continue
            per_seed = [f"{x:+.2f}" for x in r["team_reward_per_seed"]]
            lines.append(
                f"| {scenario} | {arm} | {r['n_seeds']} | "
                f"{r['team_reward_mean']:+.2f} ± {r['team_reward_std']:.2f} | "
                f"{per_seed} |"
            )

    lines += [
        "",
        "## Policy divergence (final-iter entropy spread + pairwise action KL)",
        "",
        "| scenario | arm | mean_entropy | entropy_spread | KL off-diag mean |",
        "|---|---|---|---|---|",
    ]
    for scenario in SCENARIOS:
        for arm in ARMS:
            r = results[arm].get(scenario, {})
            if not r.get("n_seeds"):
                continue
            kl = r.get("kl_off_diag_mean")
            kl_s = f"{kl:.4f}" if kl is not None else "n/a"
            lines.append(
                f"| {scenario} | {arm} | {r['mean_entropy_mean']:.3f} | "
                f"{r['entropy_spread_mean']:.3f} | {kl_s} |"
            )

    lines += [
        "",
        "## Gap-closed by scenario (per-arm and delta)",
        "",
        "| scenario | gap_ippo | gap_mappo | delta (mappo−ippo) | per-scenario tier |",
        "|---|---|---|---|---|",
    ]
    per_scen_verdict = verdict["per_scenario"]  # type: ignore[index]
    for scenario in SCENARIOS:
        v = per_scen_verdict[scenario]
        if v.get("status") == "missing":
            lines.append(f"| {scenario} | — | — | — | missing |")
            continue
        lines.append(
            f"| {scenario} | {v['gap_ippo']:+.3f} | {v['gap_mappo']:+.3f} | "
            f"{v['delta']:+.3f} | `{v['tier']}` |"
        )

    headline = verdict["headline_verdict"]
    tier_counts = verdict["tier_counts"]  # type: ignore[index]
    lines += [
        "",
        "## Headline verdict",
        "",
        f"**`{headline}`**",
        "",
        f"Tier counts: tier_1_succeeds={tier_counts['tier_1_succeeds']}, "
        f"tier_2_partial={tier_counts['tier_2_partial']}, "
        f"tier_3_insufficient={tier_counts['tier_3_insufficient']}.",
    ]
    harmful = verdict["harmful_scenarios"]  # type: ignore[index]
    if harmful:
        lines.append(f"Harmful regression on: `{', '.join(harmful)}`.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main() -> None:
    runs_root = Path("experiments/p3_specialization/runs")
    out_dir = Path("experiments/p3_specialization/diagnostics/results/issue231_mappo")
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {arm: aggregate_arm(runs_root, arm) for arm in ARMS}
    verdict = compute_verdict(results)
    payload = {"results": results, "verdict": verdict, "baselines": BASELINES}
    (out_dir / "summary.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "summary.md").write_text(render_markdown(results, verdict))
    print(f"wrote {out_dir / 'summary.json'}")
    print(f"wrote {out_dir / 'summary.md'}")
    print(f"headline verdict: {verdict['headline_verdict']}")


if __name__ == "__main__":
    main()
