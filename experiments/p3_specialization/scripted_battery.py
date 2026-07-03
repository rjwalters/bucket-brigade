#!/usr/bin/env python3
"""All-scripted team battery for a named scenario (issue #436, Part A).

Measures the best *scripted team* per-step team reward on a named scenario
(``rest_trap`` first) so the Tier-1 trap-escape verdict rule has a measured
``scripted_best`` upper anchor. This mirrors the k = 1 improvability oracle
(``experiments/nash/phase_diagram/improvability_oracle.py``, PR #432) but at
the **team** level: ALL agents play scripted policies (no frozen uniform
opponents, no NN, no PPO, no training).

Battery (team profiles over 4 agents):

- ``uniform``       -- all 4 agents uniform-random (baseline; bit-identical
  to ``per_cell._run_random_episode`` given the same seed, i.e. the same
  convention behind ``SCENARIO_RANDOM_BASELINES``).
- ``always_rest``   -- all 4 agents rest every night (the social-trap
  temptation profile).
- ``specialist``    -- :func:`bucket_brigade.baselines.specialist.specialist_action`
  x4 (owned-house firefighters).
- ``firefighter[owned|any, work=1.00] x4`` -- the deterministic firefighter
  family from the improvability oracle, played homogeneously.
- ``{k}xfirefighter[any]+{4-k}xrest`` for k = 1..3 -- asymmetric profiles
  shaped like rest_trap's frozen NE (3xfree_rider + 1xfirefighter).
- ``--random-search N`` additionally samples N homogeneous
  ``(scope, work_prob)`` firefighter teams.

Two-stage measurement:

1. **Screen**: every battery member at ``--n-episodes`` (default 2000)
   with identical episode seeds across members (paired comparison) and
   episode-bootstrap 95% CIs (``per_cell._episode_bootstrap_ci``).
2. **Final**: the screen winner (best team mean) and ``uniform`` are
   re-measured on fresh paired seeds at ``--n-episodes-final`` (default
   10000, matching the #413/#416 measurement conventions). The final-stage
   numbers are what gets committed as ``scripted_best`` in
   ``bucket_brigade.baselines.SCENARIO_GAP_REFERENCES``.

Conventions shared with :mod:`bucket_brigade.baselines.per_cell`:
``per_step_team`` = total episode team reward / ``env.night``; per-episode
seeds via ``per_cell._seeds_for``; episode-bootstrap 95% CIs via
``per_cell._episode_bootstrap_ci``.

Compute: scripted policies only -- minutes of CPU, safe locally (the merged
improvability oracle ran 69 policies x 400 episodes x 3 cells in ~18 s).

    uv run python experiments/p3_specialization/scripted_battery.py

Outputs (written under ``experiments/p3_specialization/scripted_battery/``,
deterministic given ``--seed`` up to the host provenance field):
    - ``<scenario>.json`` -- machine-readable, includes the ``scripted_best``
      block ready to transcribe into ``SCENARIO_GAP_REFERENCES``.
    - ``<scenario>.md``   -- human-readable table + verdict note.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess  # nosec B404 (git rev-parse for provenance only)
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from bucket_brigade.baselines import SCENARIO_RANDOM_BASELINES
from bucket_brigade.baselines.per_cell import (
    _episode_bootstrap_ci,
    _parallel_map,
    _seeds_for,
)
from bucket_brigade.baselines.specialist import _owned_houses, specialist_action
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario, get_scenario_by_name

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "experiments" / "p3_specialization" / "scripted_battery"

NUM_AGENTS = 4

# House-state / action-mode codes mirrored from BucketBrigadeEnv (module
# constants for the same import-cost reason as baselines/specialist.py).
_BURNING = 1
_REST = 0
_WORK = 1

# Seed offset between the screen stage and the final stage so the winner is
# confirmed on fresh episodes (guards against winner's-curse selection bias).
_FINAL_STAGE_SEED_OFFSET = 500_009


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def _ff(scope_owned_only: bool, work_prob: float) -> dict:
    return {
        "kind": "firefighter",
        "scope_owned_only": scope_owned_only,
        "work_prob": float(work_prob),
    }


def battery_specs(random_search: int, seed: int) -> list[dict]:
    """The team battery. Each spec is ``{"name", "members": [4 agent specs]}``.

    Agent-spec ``kind`` is one of ``uniform | always_rest | specialist |
    firefighter`` (firefighter specs carry ``scope_owned_only`` and
    ``work_prob``), mirroring the improvability-oracle family.
    """
    uniform = {"kind": "uniform"}
    rest = {"kind": "always_rest"}
    spec_ = {"kind": "specialist"}
    ff_any = _ff(scope_owned_only=False, work_prob=1.0)
    ff_owned = _ff(scope_owned_only=True, work_prob=1.0)

    specs: list[dict] = [
        {"name": "uniform", "members": [uniform] * NUM_AGENTS},
        {"name": "always_rest", "members": [rest] * NUM_AGENTS},
        {"name": "specialist", "members": [spec_] * NUM_AGENTS},
        {
            "name": "firefighter[owned,work=1.00]x4",
            "members": [ff_owned] * NUM_AGENTS,
        },
        {
            "name": "firefighter[any,work=1.00]x4",
            "members": [ff_any] * NUM_AGENTS,
        },
    ]
    # NE-shaped asymmetric profiles: k any-house firefighters + (4-k) resters.
    # rest_trap's frozen NE is 3xfree_rider + 1xfirefighter, so k = 1 is the
    # scripted analogue of the equilibrium profile shape.
    for k in (1, 2, 3):
        specs.append(
            {
                "name": f"{k}xfirefighter[any]+{NUM_AGENTS - k}xrest",
                "members": [ff_any] * k + [rest] * (NUM_AGENTS - k),
            }
        )
    if random_search > 0:
        rng = np.random.default_rng(seed + 1723)
        for _ in range(random_search):
            owned = bool(rng.integers(0, 2))
            wp = float(rng.random())
            specs.append(
                {
                    "name": f"search[{'owned' if owned else 'any'},work={wp:.3f}]x4",
                    "members": [_ff(owned, wp)] * NUM_AGENTS,
                    "from_random_search": True,
                }
            )
    return specs


def _member_action(
    member: dict,
    agent_id: int,
    obs: dict,
    rng: np.random.Generator,
    num_agents: int,
    num_houses: int,
    sampled_uniform: np.ndarray,
) -> np.ndarray:
    """One agent's action under an agent spec. Signals honestly (signal == mode)."""
    kind = member["kind"]
    if kind == "uniform":
        return sampled_uniform
    if kind == "always_rest":
        return np.array([0, _REST, _REST], dtype=np.int64)
    if kind == "specialist":
        return specialist_action(obs, agent_id, num_agents, num_houses)
    if kind == "firefighter":
        houses = np.asarray(obs["houses"])
        if member["scope_owned_only"]:
            candidates = _owned_houses(agent_id, num_agents, num_houses)
        else:
            candidates = np.arange(num_houses)
        burning = candidates[houses[candidates] == _BURNING]
        work_prob = float(member["work_prob"])
        if burning.size > 0 and (work_prob >= 1.0 or float(rng.random()) < work_prob):
            return np.array([int(burning[0]), _WORK, _WORK], dtype=np.int64)
        return np.array([0, _REST, _REST], dtype=np.int64)
    raise ValueError(f"unknown agent-spec kind: {kind!r}")


# ---------------------------------------------------------------------------
# Episode worker
# ---------------------------------------------------------------------------


def _run_team_episode(args: tuple[Scenario, int, dict]) -> tuple[float, float]:
    """One episode with all agents scripted per ``spec["members"]``.

    Returns ``(per_step_team, team_work_rate)`` where ``team_work_rate`` is
    the fraction of (agent x night) slots spent working.

    The full uniform-random joint action block is drawn from the seeded rng
    each night *before* overwriting scripted rows, so the ``uniform`` team is
    bit-identical to ``per_cell._run_random_episode`` for the same seed.
    """
    scenario, seed, spec = args
    members = spec["members"]
    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_team = 0.0
    work_slots = 0
    slots = 0
    while not env.done:
        actions = np.stack(
            [
                rng.integers(0, env.num_houses, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
            ],
            axis=-1,
        ).astype(np.int64)
        for i, member in enumerate(members):
            if member["kind"] != "uniform":
                actions[i] = _member_action(
                    member, i, obs, rng, env.num_agents, env.num_houses, actions[i]
                )
        work_slots += int((actions[:, 1] == _WORK).sum())
        slots += env.num_agents
        obs, rewards, _, _ = env.step(actions)
        total_team += float(rewards.sum())
    nights = max(1, int(env.night))
    return total_team / nights, work_slots / max(1, slots)


# ---------------------------------------------------------------------------
# Battery evaluation
# ---------------------------------------------------------------------------


def _measure_policy(
    scenario: Scenario,
    spec: dict,
    seeds: list[int],
    *,
    n_boot: int,
    boot_seed: int,
    num_workers: Optional[int],
) -> tuple[np.ndarray, dict]:
    """Measure one team spec over the given episode seeds.

    Returns ``(per_episode_team_values, record)`` where the record carries
    the mean/CI/work-rate summary.
    """
    args = [(scenario, s, spec) for s in seeds]
    results = _parallel_map(_run_team_episode, args, num_workers)
    team = np.asarray([r[0] for r in results], dtype=np.float64)
    work_rate = float(np.mean([r[1] for r in results]))
    mean, lo, hi = _episode_bootstrap_ci(
        team, n_boot=n_boot, rng=np.random.default_rng(boot_seed)
    )
    record = {
        "team": {"mean": mean, "ci95_lo": lo, "ci95_hi": hi},
        "team_work_rate": work_rate,
        "n_episodes": len(seeds),
    }
    return team, record


def evaluate_battery(
    scenario_name: str,
    specs: list[dict],
    *,
    n_episodes: int,
    seed: int,
    num_workers: Optional[int],
    n_boot: int,
) -> dict:
    """Screen the full battery on identical episode seeds (paired vs uniform)."""
    scenario = get_scenario_by_name(scenario_name, num_agents=NUM_AGENTS)
    seeds = _seeds_for(seed, n_episodes)

    all_args = [(scenario, s, spec) for spec in specs for s in seeds]
    all_results = _parallel_map(_run_team_episode, all_args, num_workers)

    assert specs[0]["name"] == "uniform", "battery_specs must put uniform first"
    uniform_team = np.asarray(
        [r[0] for r in all_results[:n_episodes]], dtype=np.float64
    )

    policies: dict[str, dict] = {}
    search_samples: list[dict] = []
    for i, spec in enumerate(specs):
        results = all_results[i * n_episodes : (i + 1) * n_episodes]
        team = np.asarray([r[0] for r in results], dtype=np.float64)
        work_rate = float(np.mean([r[1] for r in results]))
        t_mean, t_lo, t_hi = _episode_bootstrap_ci(
            team, n_boot=n_boot, rng=np.random.default_rng(seed + 7 + 1000 * i)
        )
        record = {
            "team": {"mean": t_mean, "ci95_lo": t_lo, "ci95_hi": t_hi},
            "team_work_rate": work_rate,
            "n_episodes": n_episodes,
        }
        if spec["name"] != "uniform":
            d_mean, d_lo, d_hi = _episode_bootstrap_ci(
                team - uniform_team,
                n_boot=n_boot,
                rng=np.random.default_rng(seed + 13 + 1000 * i),
            )
            record["team_delta_vs_uniform_paired"] = {
                "mean": d_mean,
                "ci95_lo": d_lo,
                "ci95_hi": d_hi,
            }
        if spec.get("from_random_search"):
            search_samples.append({"name": spec["name"], **record})
        else:
            policies[spec["name"]] = record

    all_named = list(policies.items()) + [(s["name"], s) for s in search_samples]
    best_name, best_rec = max(all_named, key=lambda kv: kv[1]["team"]["mean"])

    result = {
        "scenario": scenario_name,
        "policies": policies,
        "screen_best": {
            "name": best_name,
            "team_mean": best_rec["team"]["mean"],
            "team_delta_vs_uniform": best_rec["team"]["mean"]
            - policies["uniform"]["team"]["mean"],
            "team_delta_vs_uniform_paired_ci": best_rec.get(
                "team_delta_vs_uniform_paired"
            ),
        },
    }
    if search_samples:
        best_sample = max(search_samples, key=lambda s: s["team"]["mean"])
        result["random_search"] = {
            "n_samples": len(search_samples),
            "best_by_team": best_sample["name"],
            "samples": search_samples,
        }
    return result


def measure_final(
    scenario_name: str,
    winner_spec: dict,
    *,
    n_episodes: int,
    seed: int,
    num_workers: Optional[int],
    n_boot: int,
) -> dict:
    """Re-measure the screen winner + uniform on fresh paired seeds (n=10k)."""
    scenario = get_scenario_by_name(scenario_name, num_agents=NUM_AGENTS)
    final_seed = seed + _FINAL_STAGE_SEED_OFFSET
    seeds = _seeds_for(final_seed, n_episodes)

    uniform_spec = {"name": "uniform", "members": [{"kind": "uniform"}] * NUM_AGENTS}
    uniform_team, uniform_rec = _measure_policy(
        scenario,
        uniform_spec,
        seeds,
        n_boot=n_boot,
        boot_seed=final_seed + 7,
        num_workers=num_workers,
    )
    winner_team, winner_rec = _measure_policy(
        scenario,
        winner_spec,
        seeds,
        n_boot=n_boot,
        boot_seed=final_seed + 11,
        num_workers=num_workers,
    )
    d_mean, d_lo, d_hi = _episode_bootstrap_ci(
        winner_team - uniform_team,
        n_boot=n_boot,
        rng=np.random.default_rng(final_seed + 13),
    )
    return {
        "stage_seed": final_seed,
        "n_episodes": n_episodes,
        "uniform": uniform_rec,
        "winner": {"name": winner_spec["name"], **winner_rec},
        "winner_delta_vs_uniform_paired": {
            "mean": d_mean,
            "ci95_lo": d_lo,
            "ci95_hi": d_hi,
        },
    }


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _git_sha() -> str:
    try:
        out = subprocess.run(  # nosec B603 B607 (hardcoded argv)
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    scenario = result["scenario"]
    screen = result["screen"]
    final = result["final"]
    prov = result["provenance"]
    random_baseline = result.get("cited_random_baseline")
    sb = result["scripted_best"]

    beats = sb["beats_random"]
    if beats:
        title = (
            f"# Scripted team battery on `{scenario}`: `{sb['name']}` beats"
            " the uniform-random baseline (measured `scripted_best` upper"
            " anchor)"
        )
        verdict = (
            f"**Verdict: `scripted_best` = {sb['value']:.3f}/step"
            f" [{sb['ci95_lo']:.3f}, {sb['ci95_hi']:.3f}]"
            f" (`{sb['name']}`, n={sb['n_episodes']}), decisively above the"
            f" cited uniform-random baseline"
            f" {random_baseline:.2f}/step.** This value is committed as the"
            " `scripted_best` trap-verdict anchor in"
            " `bucket_brigade.baselines.SCENARIO_GAP_REFERENCES` (issue"
            " #436). It is an *anchor for the categorical trap-escape rule*,"
            " not a `reference` for the gap_closed fraction ladder — the"
            " scenario stays on the degenerate-reference path because its"
            " frozen NE sits below random (social trap)."
        )
    else:
        title = (
            f"# Scripted team battery on `{scenario}`: no battery member"
            " beats the uniform-random baseline"
        )
        verdict = (
            "**Verdict: `scripted_best` <= random.** No scripted team"
            " profile in the battery beats the uniform-random baseline"
            f" ({random_baseline:.2f}/step) on team return — itself a"
            " publishable characterization. The trap-escape verdict rule"
            " (issue #436 Part B) operates from the NE + random anchors"
            " alone; `scripted_best` is recorded as absent with this"
            " measurement as provenance."
        )

    lines = [
        title,
        "",
        "Generated by `experiments/p3_specialization/scripted_battery.py`"
        " (issue #436, Part A). Method: ALL agents play **scripted** policies"
        " (team-level battery — no NN, no PPO, no training), mirroring the"
        " k = 1 improvability oracle (PR #432) conventions:"
        " `per_step_team` = total episode team reward / `env.night`,"
        " per-episode seeding via `per_cell._seeds_for`, episode-bootstrap"
        " 95% CIs, identical episode seeds across battery members (paired"
        " deltas vs `uniform`).",
        "",
        f"Config: scenario `{scenario}` (4 agents), screen"
        f" `n_episodes={cfg['n_episodes']}` per policy, final"
        f" `n_episodes_final={cfg['n_episodes_final']}`,"
        f" `seed={cfg['seed']}`, `n_boot={cfg['n_boot']}`,"
        f" `random_search={cfg['random_search']}`.",
        "",
        f"Provenance: host `{prov['host']}`, commit `{prov['git_sha']}`.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
        f"## Screen stage (n={cfg['n_episodes']} episodes/policy, paired seeds)",
        "",
        f"Cited uniform-random baseline (`SCENARIO_RANDOM_BASELINES`):"
        f" **{random_baseline:.2f}**/step.",
        "",
        "| Team profile | per-step team return [95% CI] | Δ team vs uniform"
        " [paired 95% CI] | team work rate |",
        "|---|---|---|---|",
    ]
    for name, rec in screen["policies"].items():
        t = rec["team"]
        d = rec.get("team_delta_vs_uniform_paired")
        delta_str = (
            "—"
            if d is None
            else f"{d['mean']:+.3f} [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}]"
        )
        marker = " **(screen best)**" if name == screen["screen_best"]["name"] else ""
        lines.append(
            f"| `{name}`{marker} | {t['mean']:.3f}"
            f" [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}] | {delta_str}"
            f" | {rec['team_work_rate']:.3f} |"
        )
    if "random_search" in screen:
        rs = screen["random_search"]
        lines += [
            "",
            f"Random search over the homogeneous firefighter family"
            f" ({rs['n_samples']} samples): best by team return is"
            f" `{rs['best_by_team']}` (full samples in the JSON).",
        ]
    fw = final["winner"]
    fu = final["uniform"]
    fd = final["winner_delta_vs_uniform_paired"]
    lines += [
        "",
        f"## Final stage (n={final['n_episodes']} fresh paired episodes,"
        " winner vs uniform)",
        "",
        "| Team profile | per-step team return [95% CI] |",
        "|---|---|",
        f"| `{fw['name']}` | {fw['team']['mean']:.3f}"
        f" [{fw['team']['ci95_lo']:.3f}, {fw['team']['ci95_hi']:.3f}] |",
        f"| `uniform` | {fu['team']['mean']:.3f}"
        f" [{fu['team']['ci95_lo']:.3f}, {fu['team']['ci95_hi']:.3f}] |",
        "",
        f"Paired Δ (winner − uniform): {fd['mean']:+.3f}"
        f" [{fd['ci95_lo']:+.3f}, {fd['ci95_hi']:+.3f}] per step.",
        "",
        "Reproduce with:",
        "",
        "```bash",
        f"uv run python experiments/p3_specialization/scripted_battery.py"
        f" --scenario {scenario}"
        + (f" --random-search {cfg['random_search']}" if cfg["random_search"] else ""),
        "```",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scenario",
        default="rest_trap",
        help="named scenario to measure (default: rest_trap)",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=2000,
        help="screen-stage episodes per battery member",
    )
    parser.add_argument(
        "--n-episodes-final",
        type=int,
        default=10_000,
        help="final-stage episodes for the winner + uniform re-measurement",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--random-search",
        type=int,
        default=16,
        metavar="N",
        help="additionally sample N homogeneous (scope, work_prob) firefighter teams",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="multiprocessing workers (default: cpu_count; 1 = sequential)",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args(argv)

    specs = battery_specs(args.random_search, args.seed)
    print(
        f"Screening {args.scenario}: {len(specs)} team profiles x"
        f" {args.n_episodes} episodes...",
        flush=True,
    )
    screen = evaluate_battery(
        args.scenario,
        specs,
        n_episodes=args.n_episodes,
        seed=args.seed,
        num_workers=args.num_workers,
        n_boot=args.n_boot,
    )

    winner_name = screen["screen_best"]["name"]
    winner_spec = next(s for s in specs if s["name"] == winner_name)
    print(
        f"Screen best: {winner_name}; re-measuring winner + uniform at"
        f" n={args.n_episodes_final} fresh paired episodes...",
        flush=True,
    )
    final = measure_final(
        args.scenario,
        winner_spec,
        n_episodes=args.n_episodes_final,
        seed=args.seed,
        num_workers=args.num_workers,
        n_boot=args.n_boot,
    )

    cited_random = SCENARIO_RANDOM_BASELINES.get(args.scenario)
    fw = final["winner"]["team"]
    # "Beats random": the winner's final-stage 95% CI lower bound must clear
    # BOTH the cited random baseline and this run's own uniform CI upper
    # bound (guards against a drifted cited value).
    beats_random = (
        cited_random is not None
        and fw["ci95_lo"] > cited_random
        and fw["ci95_lo"] > final["uniform"]["team"]["ci95_hi"]
    )
    scripted_best = {
        "name": final["winner"]["name"],
        "value": fw["mean"],
        "ci95_lo": fw["ci95_lo"],
        "ci95_hi": fw["ci95_hi"],
        "n_episodes": final["n_episodes"],
        "beats_random": bool(beats_random),
    }

    result = {
        "generated_by": "experiments/p3_specialization/scripted_battery.py",
        "issue": 436,
        "scenario": args.scenario,
        "config": {
            "n_episodes": args.n_episodes,
            "n_episodes_final": args.n_episodes_final,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "random_search": args.random_search,
            "num_agents": NUM_AGENTS,
        },
        "provenance": {
            "host": platform.node(),
            "git_sha": _git_sha(),
        },
        "cited_random_baseline": cited_random,
        "screen": screen,
        "final": final,
        "scripted_best": scripted_best,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.out_dir / f"{args.scenario}.json"
    out_md = args.out_dir / f"{args.scenario}.md"
    with open(out_json, "w") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    print(f"Wrote {out_json}")
    with open(out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {out_md}")

    print(
        f"{args.scenario}: scripted_best = {scripted_best['name']} at"
        f" {scripted_best['value']:.3f}/step"
        f" [{scripted_best['ci95_lo']:.3f}, {scripted_best['ci95_hi']:.3f}]"
        f" (cited random = {cited_random!r}, beats_random={beats_random})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
