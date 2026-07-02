#!/usr/bin/env python3
"""k = 1 improvability oracle for the phase-diagram no-convergence cells
(issue #428).

Freezes N-1 = 3 uniform-random opponents and scores a battery of *scripted*
policies for the single best-response agent (agent 0) — no NN, no PPO, no
training. If no battery member meaningfully beats the all-uniform baseline
on **team** return, the cell is degenerate for single-best-response
learning: a well-fit critic correctly reports ~0 advantage and flat PPO
curves are the correct answer on that reward surface, not an optimizer
failure.

Battery (agent 0; agents 1-3 uniform-random):

- ``uniform``       — agent 0 also uniform-random (baseline; bit-identical
  to ``per_cell._run_random_episode`` given the same seed).
- ``always_rest``   — agent 0 rests every night (zero work cost).
- ``specialist``    — :func:`bucket_brigade.baselines.specialist.specialist_action`
  (the ``gap_closed == 1.0`` endpoint policy).
- ``firefighter[owned|any, work=1.00]`` — deterministic firefighter family
  ``{scope: owned_only | any_house, work_prob}`` at the ``work_prob = 1.0``
  endpoints. An optional ``--random-search N`` samples N additional
  ``(scope, work_prob)`` members of the same family (which contains the
  specialist-like behaviours).

Conventions are shared with :mod:`bucket_brigade.baselines.per_cell` so the
numbers are bit-comparable with the committed ``per_cell_baselines.json``:

- Scenario construction via ``make_phase_diagram_scenario(beta, kappa, cost)``.
- ``per_step_team`` = total episode team reward / ``env.night``; additionally
  the BR agent's own per-step return (``rewards[0]`` summed / ``env.night``).
- Per-episode seeds via ``per_cell._seeds_for``; the *same* episode seeds are
  used for every battery member within a cell (paired comparison).
- Episode-bootstrap 95% CIs via ``per_cell._episode_bootstrap_ci``.

Compute: scripted policies only. The default run (400 episodes/cell x 3
cells x 5 battery members) is minutes of CPU and safe locally.

    uv run python experiments/nash/phase_diagram/improvability_oracle.py

Outputs (written next to this script, deterministic given ``--seed``):
    - ``improvability_oracle.json`` — machine-readable, keyed by ``cell_tag``
    - ``improvability_oracle.md``   — human-readable table + mechanism note
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from bucket_brigade.baselines.per_cell import (
    _episode_bootstrap_ci,
    _parallel_map,
    _seeds_for,
    make_phase_diagram_scenario,
)
from bucket_brigade.baselines.specialist import _owned_houses, specialist_action
from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
from bucket_brigade.envs.scenarios_generated import Scenario

REPO_ROOT = Path(__file__).resolve().parents[3]
PHASE_DIAGRAM_ROOT = REPO_ROOT / "experiments" / "nash" / "phase_diagram"

DEFAULT_OUT_JSON = PHASE_DIAGRAM_ROOT / "improvability_oracle.json"
DEFAULT_OUT_MD = PHASE_DIAGRAM_ROOT / "improvability_oracle.md"

# The three no_convergence cells at c = 0.5 reported in issue #428. The other
# three no_convergence cells (k0.10_c2.00 — see
# experiments/p3_specialization/phase_diagram_ppo_v2/recalibrated_verdict.md)
# can be characterized in a follow-up via --cells.
DEFAULT_CELLS = (
    "b0.10_k0.10_c0.50",
    "b0.50_k0.10_c0.50",
    "b0.90_k0.10_c0.50",
)

# House-state / action-mode codes mirrored from BucketBrigadeEnv (kept as
# module constants for the same import-cost reason as baselines/specialist.py).
_BURNING = 1
_REST = 0
_WORK = 1

_CELL_TAG_RE = re.compile(r"^b(\d+\.\d+)_k(\d+\.\d+)_c(\d+\.\d+)$")


def parse_cell_tag(tag: str) -> tuple[float, float, float]:
    """Parse ``b{beta}_k{kappa}_c{cost}`` into (beta, kappa, cost).

    Matches the cell_tag convention of ``conditional_entropy.json`` /
    ``recalibrated_verdict.json``.
    """
    m = _CELL_TAG_RE.match(tag)
    if m is None:
        raise ValueError(
            f"cell tag {tag!r} does not match 'b<beta>_k<kappa>_c<cost>' "
            "(e.g. b0.10_k0.10_c0.50)"
        )
    return float(m.group(1)), float(m.group(2)), float(m.group(3))


# ---------------------------------------------------------------------------
# Battery
# ---------------------------------------------------------------------------


def battery_specs(random_search: int, seed: int) -> list[dict]:
    """The BR battery for agent 0. Each spec is a plain picklable dict.

    ``kind`` is one of ``uniform | always_rest | specialist | firefighter``;
    firefighter specs carry ``scope_owned_only`` and ``work_prob``.
    """
    specs: list[dict] = [
        {"name": "uniform", "kind": "uniform"},
        {"name": "always_rest", "kind": "always_rest"},
        {"name": "specialist", "kind": "specialist"},
        {
            "name": "firefighter[owned,work=1.00]",
            "kind": "firefighter",
            "scope_owned_only": True,
            "work_prob": 1.0,
        },
        {
            "name": "firefighter[any,work=1.00]",
            "kind": "firefighter",
            "scope_owned_only": False,
            "work_prob": 1.0,
        },
    ]
    if random_search > 0:
        rng = np.random.default_rng(seed + 1723)
        for _ in range(random_search):
            owned = bool(rng.integers(0, 2))
            wp = float(rng.random())
            specs.append(
                {
                    "name": f"search[{'owned' if owned else 'any'},work={wp:.3f}]",
                    "kind": "firefighter",
                    "scope_owned_only": owned,
                    "work_prob": wp,
                    "from_random_search": True,
                }
            )
    return specs


def _br_action(
    spec: dict,
    obs: dict,
    rng: np.random.Generator,
    num_agents: int,
    num_houses: int,
    sampled_uniform: np.ndarray,
) -> np.ndarray:
    """Agent 0's action under a battery spec. Signals honestly (signal == mode)."""
    kind = spec["kind"]
    if kind == "uniform":
        return sampled_uniform
    if kind == "always_rest":
        return np.array([0, _REST, _REST], dtype=np.int64)
    if kind == "specialist":
        return specialist_action(obs, 0, num_agents, num_houses)
    if kind == "firefighter":
        houses = np.asarray(obs["houses"])
        if spec["scope_owned_only"]:
            candidates = _owned_houses(0, num_agents, num_houses)
        else:
            candidates = np.arange(num_houses)
        burning = candidates[houses[candidates] == _BURNING]
        work_prob = float(spec["work_prob"])
        if burning.size > 0 and (work_prob >= 1.0 or float(rng.random()) < work_prob):
            return np.array([int(burning[0]), _WORK, _WORK], dtype=np.int64)
        return np.array([0, _REST, _REST], dtype=np.int64)
    raise ValueError(f"unknown battery kind: {kind!r}")


# ---------------------------------------------------------------------------
# Episode worker
# ---------------------------------------------------------------------------


def _run_oracle_episode(args: tuple[Scenario, int, dict]) -> tuple[float, float, float]:
    """One episode: agent 0 plays ``spec``, agents 1..N-1 uniform-random.

    Returns ``(per_step_team, per_step_br_agent, br_work_rate)``.

    The full uniform-random joint action block is drawn from the seeded rng
    each night *before* overwriting agent 0's row, so the opponents' action
    stream is identical across battery members for a given episode seed —
    and the ``uniform`` member is bit-identical to
    ``per_cell._run_random_episode``.
    """
    scenario, seed, spec = args
    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=seed)
    rng = np.random.default_rng(seed)
    total_team = 0.0
    total_br = 0.0
    br_work_steps = 0
    steps = 0
    while not env.done:
        actions = np.stack(
            [
                rng.integers(0, env.num_houses, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
                rng.integers(0, 2, size=env.num_agents),
            ],
            axis=-1,
        ).astype(np.int64)
        actions[0] = _br_action(
            spec, obs, rng, env.num_agents, env.num_houses, actions[0]
        )
        if actions[0][1] == _WORK:
            br_work_steps += 1
        obs, rewards, _, _ = env.step(actions)
        total_team += float(rewards.sum())
        total_br += float(rewards[0])
        steps += 1
    nights = max(1, int(env.night))
    work_rate = br_work_steps / max(1, steps)
    return total_team / nights, total_br / nights, work_rate


# ---------------------------------------------------------------------------
# Per-cell evaluation
# ---------------------------------------------------------------------------


def evaluate_cell(
    cell_tag: str,
    specs: list[dict],
    n_episodes: int,
    seed: int,
    num_workers: Optional[int],
    n_boot: int,
    base_scenario_name: str = "minimal_specialization",
) -> dict:
    """Evaluate the full battery on one cell. Returns the per-cell record.

    All (policy x episode) rollouts for the cell go through a single
    ``_parallel_map`` call so multiprocessing pool-spawn overhead is
    amortized. Because every battery member replays the *same* episode
    seeds, each non-uniform policy additionally gets a **paired**
    episode-bootstrap CI on its per-episode team-return delta vs
    ``uniform`` — a much tighter interval for the headroom claim than
    comparing two independent CIs. (Pairing is partial: trajectories
    share the initial ignitions and the opponents' RNG stream, but
    diverge once agent 0 acts differently.)
    """
    beta, kappa, cost = parse_cell_tag(cell_tag)
    scenario = make_phase_diagram_scenario(beta, kappa, cost, base_scenario_name)
    seeds = _seeds_for(seed, n_episodes)

    all_args = [(scenario, s, spec) for spec in specs for s in seeds]
    all_results = _parallel_map(_run_oracle_episode, all_args, num_workers)

    assert specs[0]["kind"] == "uniform", "battery_specs must put uniform first"
    uniform_team = np.asarray(
        [r[0] for r in all_results[:n_episodes]], dtype=np.float64
    )

    policies: dict[str, dict] = {}
    search_samples: list[dict] = []
    for i, spec in enumerate(specs):
        results = all_results[i * n_episodes : (i + 1) * n_episodes]
        team = np.asarray([r[0] for r in results], dtype=np.float64)
        br = np.asarray([r[1] for r in results], dtype=np.float64)
        work_rate = float(np.mean([r[2] for r in results]))
        t_mean, t_lo, t_hi = _episode_bootstrap_ci(
            team, n_boot=n_boot, rng=np.random.default_rng(seed + 7 + 1000 * i)
        )
        b_mean, b_lo, b_hi = _episode_bootstrap_ci(
            br, n_boot=n_boot, rng=np.random.default_rng(seed + 11 + 1000 * i)
        )
        record = {
            "team": {"mean": t_mean, "ci95_lo": t_lo, "ci95_hi": t_hi},
            "br_agent": {"mean": b_mean, "ci95_lo": b_lo, "ci95_hi": b_hi},
            "br_work_rate": work_rate,
            "n_episodes": n_episodes,
        }
        if spec["kind"] != "uniform":
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
            search_samples.append(
                {
                    "name": spec["name"],
                    "scope_owned_only": spec["scope_owned_only"],
                    "work_prob": spec["work_prob"],
                    **record,
                }
            )
        else:
            policies[spec["name"]] = record

    uniform_team_mean = policies["uniform"]["team"]["mean"]
    uniform_br_mean = policies["uniform"]["br_agent"]["mean"]

    all_named = [(name, rec) for name, rec in policies.items()] + [
        (s["name"], s) for s in search_samples
    ]
    best_team_name, best_team_rec = max(all_named, key=lambda kv: kv[1]["team"]["mean"])
    best_br_name, best_br_rec = max(all_named, key=lambda kv: kv[1]["br_agent"]["mean"])

    def _pct_over(value: float, baseline: float) -> float:
        # Returns are negative at these cells; "headroom %" is the improvement
        # relative to the magnitude of the uniform baseline.
        return 100.0 * (value - baseline) / abs(baseline)

    record = {
        "cell_tag": cell_tag,
        "beta": beta,
        "kappa": kappa,
        "c": cost,
        "policies": policies,
        "headroom": {
            "best_team_policy": best_team_name,
            "best_team_mean": best_team_rec["team"]["mean"],
            "team_delta_vs_uniform": best_team_rec["team"]["mean"] - uniform_team_mean,
            "team_delta_vs_uniform_paired_ci": best_team_rec.get(
                "team_delta_vs_uniform_paired"
            ),
            "team_pct_over_uniform": _pct_over(
                best_team_rec["team"]["mean"], uniform_team_mean
            ),
            "best_br_agent_policy": best_br_name,
            "best_br_agent_mean": best_br_rec["br_agent"]["mean"],
            "br_agent_delta_vs_uniform": best_br_rec["br_agent"]["mean"]
            - uniform_br_mean,
            "br_agent_pct_over_uniform": _pct_over(
                best_br_rec["br_agent"]["mean"], uniform_br_mean
            ),
        },
    }
    if search_samples:
        best_sample = max(search_samples, key=lambda s: s["team"]["mean"])
        record["random_search"] = {
            "n_samples": len(search_samples),
            "best_by_team": best_sample["name"],
            "samples": search_samples,
        }
    return record


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _cell_is_degenerate(cell: dict) -> bool:
    """Degenerate for single-BR learning = the best battery member's team
    headroom is within noise of uniform (paired 95% CI includes 0) or
    negligible (< 0.5% of the uniform magnitude)."""
    head = cell["headroom"]
    paired = head.get("team_delta_vs_uniform_paired_ci")
    ci_includes_zero = paired is not None and paired["ci95_lo"] <= 0.0
    return ci_includes_zero or head["team_pct_over_uniform"] < 0.5


def render_markdown(result: dict) -> str:
    cfg = result["config"]
    cells = result["cells"]
    all_degenerate = all(_cell_is_degenerate(c) for c in cells.values())
    if all_degenerate:
        title = (
            "# k = 1 improvability oracle: no single-agent improvable gap"
            " on the evaluated no-convergence cells"
        )
        verdict = (
            "**Verdict: degenerate for single-best-response learning.** No"
            " battery member meaningfully beats the all-uniform baseline on"
            " team return — this reproduces the finding reported in issue"
            " #428."
        )
    else:
        title = (
            "# k = 1 improvability oracle: significant single-agent headroom"
            " on the κ=0.1 / c=0.5 no-convergence cells (non-reproduction of"
            " the #428 report)"
        )
        verdict = (
            "**Verdict: NOT degenerate in the repo-native configuration.**"
            " The degeneracy reported in issue #428 (best team return within"
            " ~+0.01% of uniform, achieved by resting; specialist *worse*"
            " than uniform) does **not** reproduce with this repo's"
            " phase-diagram conventions: the deterministic any-house"
            " firefighter beats uniform on team return by a margin whose"
            " paired 95% CI clearly excludes zero, and the specialist is"
            " significantly *better* than uniform, not worse. Per the issue's"
            " acceptance criteria, the numbers are committed as-is (no"
            " tuning). See the non-reproduction note below for the likely"
            " cause."
        )
    lines = [
        title,
        "",
        "Generated by `improvability_oracle.py` (issue #428). Method: freeze"
        " N−1 = 3 **uniform-random** opponents and score a battery of"
        " **scripted** policies for the single best-response agent (agent 0)"
        " — no NN, no PPO, no training. Scenario construction"
        " (`make_phase_diagram_scenario`, base family"
        f" `{cfg['base_scenario_name']}`), per-step-team-return convention,"
        " per-episode seeding, and episode-bootstrap CIs are shared with"
        " `bucket_brigade/baselines/per_cell.py`, so numbers are"
        " bit-comparable with `per_cell_baselines.json`.",
        "",
        f"Config: `n_episodes={cfg['n_episodes']}` per (cell × policy),"
        f" `seed={cfg['seed']}`, `n_boot={cfg['n_boot']}`,"
        f" `random_search={cfg['random_search']}`. Identical episode seeds"
        " across battery members within a cell, so each non-uniform policy"
        " carries a **paired** episode-bootstrap CI on its team-return delta"
        " vs uniform.",
        "",
        "## Verdict",
        "",
        verdict,
        "",
    ]

    for tag, cell in result["cells"].items():
        head = cell["headroom"]
        lines += [
            f"## `{tag}` (β={cell['beta']:.2f}, κ={cell['kappa']:.2f},"
            f" c={cell['c']:.2f})",
            "",
            "| BR policy (agent 0; other 3 uniform) | per-step team return"
            " [95% CI] | Δ team vs uniform [paired 95% CI]"
            " | per-step BR-agent return [95% CI] | BR work rate |",
            "|---|---|---|---|---|",
        ]
        for name, rec in cell["policies"].items():
            t = rec["team"]
            b = rec["br_agent"]
            d = rec.get("team_delta_vs_uniform_paired")
            delta_str = (
                "—"
                if d is None
                else f"{d['mean']:+.3f} [{d['ci95_lo']:+.3f}, {d['ci95_hi']:+.3f}]"
            )
            marker = " **(best team)**" if name == head["best_team_policy"] else ""
            lines.append(
                f"| `{name}`{marker} | {t['mean']:.3f}"
                f" [{t['ci95_lo']:.3f}, {t['ci95_hi']:.3f}] | {delta_str}"
                f" | {b['mean']:.3f}"
                f" [{b['ci95_lo']:.3f}, {b['ci95_hi']:.3f}] |"
                f" {rec['br_work_rate']:.3f} |"
            )
        paired = head.get("team_delta_vs_uniform_paired_ci")
        paired_str = (
            ""
            if paired is None
            else (
                f" Paired Δ CI: [{paired['ci95_lo']:+.3f}, {paired['ci95_hi']:+.3f}]."
            )
        )
        lines += [
            "",
            f"- Best **team** return: `{head['best_team_policy']}` at"
            f" {head['best_team_mean']:.3f} — "
            f"**{head['team_pct_over_uniform']:+.3f}%** vs uniform"
            f" (Δ = {head['team_delta_vs_uniform']:+.3f} per step).{paired_str}",
            f"- Best **BR-agent** return: `{head['best_br_agent_policy']}` at"
            f" {head['best_br_agent_mean']:.3f} — "
            f"**{head['br_agent_pct_over_uniform']:+.3f}%** vs uniform"
            f" (Δ = {head['br_agent_delta_vs_uniform']:+.3f} per step).",
        ]
        if "random_search" in cell:
            rs = cell["random_search"]
            lines.append(
                f"- Random search over the firefighter family"
                f" ({rs['n_samples']} samples): best by team return is"
                f" `{rs['best_by_team']}` (full samples in the JSON)."
            )
        lines.append("")

    if all_degenerate:
        mechanism = [
            "## Mechanism",
            "",
            "At κ = 0.1 a solo worker extinguishes a fire only ~10% of the"
            " time; at c = 0.5 working is net-negative whenever teammates are"
            " random. Once a few fires ignite, Bernoulli burn-out ruins the"
            " ring within a few nights regardless of any single agent's"
            " effort, and the per-step ruined-house penalty swamps a lone"
            " firefighter's marginal effect. A well-fit critic correctly"
            " reports ~0 advantage and the policy correctly stays"
            " near-uniform — flat PPO curves on these cells"
            " (`docs/PAPER_RESULTS.md` §6b) are *correct behavior on this"
            " reward surface*, not an optimizer bug. This is the k = 1 arm"
            " of the coordination-threshold story tracked in #430 (k\\* > 1"
            " on the no-convergence cells).",
        ]
    else:
        mechanism = [
            "## Non-reproduction note: why the #428 report likely differs",
            "",
            "The #428 report (from the downstream `rjwalters/thrust` harness)"
            " quotes per-step team returns of ≈ −674 and per-step BR-agent"
            " returns of ≈ −205 for the uniform baseline; the repo-native"
            " oracle measures ≈ −93 and ≈ −32 on the same cells — a ~7×"
            " scale difference. The repo's phase-diagram convention"
            " (`per_cell.py`, the §6 PPO sweep, the NE search) builds cells"
            " on the `minimal_specialization` base family, in which"
            " per-agent ownership rewards dominate (own-house survives +50 /"
            " own-house burns −100 per agent vs a team signal of ±10)."
            " Under that reward surface a lone any-house firefighter has a"
            " real marginal effect even at κ = 0.1 / c = 0.5: every"
            " extinguished house spares its owner the −100 burn penalty and"
            " the team the per-night ruin penalty. The thrust harness"
            " evidently evaluated a differently-scaled reward configuration"
            " (magnitudes consistent with the `default`-family team weights"
            " of ±100), where the shared ruin penalty swamps the lone"
            " firefighter's contribution.",
            "",
            "## Implication for the paper (§6)",
            "",
            "Flat PPO curves on the c = 0.5 no-convergence cells **cannot**"
            " be attributed to a missing single-agent improvable gap: a"
            " scripted k = 1 best response against frozen uniform opponents"
            " picks up a statistically decisive team-return improvement that"
            " PPO fails to find. The trainability failure on these cells"
            " remains unexplained by this oracle — the coordination-threshold"
            " account (k\\* > 1, issue #430 / thrust#259) and plain"
            " exploration failure both remain live hypotheses.",
        ]
    lines += mechanism
    cells_identical = (
        len(cells) > 1
        and len({json.dumps(c["policies"], sort_keys=True) for c in cells.values()})
        == 1
    )
    if cells_identical:
        lines += [
            "",
            "## β-invariance",
            "",
            "All evaluated cells produce byte-identical aggregate statistics"
            " under identical episode seeds — consistent with the"
            " byte-identical rows for these cells in the committed"
            " `per_cell_baselines.json` (at this (κ, c) the sampled"
            " trajectories do not depend on β).",
        ]
    lines += [
        "",
        "Reproduce with:",
        "",
        "```bash",
        "uv run python experiments/nash/phase_diagram/improvability_oracle.py"
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
        "--cells",
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="cell tags to evaluate (b<beta>_k<kappa>_c<cost>)",
    )
    parser.add_argument("--n-episodes", type=int, default=400)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument(
        "--random-search",
        type=int,
        default=0,
        metavar="N",
        help="additionally sample N random (scope, work_prob) firefighters",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=None,
        help="multiprocessing workers (default: cpu_count; 1 = sequential)",
    )
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    args = parser.parse_args(argv)

    specs = battery_specs(args.random_search, args.seed)
    cells: dict[str, dict] = {}
    for tag in args.cells:
        print(
            f"Evaluating {tag}: {len(specs)} policies x {args.n_episodes} episodes...",
            flush=True,
        )
        cells[tag] = evaluate_cell(
            tag,
            specs,
            n_episodes=args.n_episodes,
            seed=args.seed,
            num_workers=args.num_workers,
            n_boot=args.n_boot,
        )

    result = {
        "generated_by": "experiments/nash/phase_diagram/improvability_oracle.py",
        "issue": 428,
        "config": {
            "n_episodes": args.n_episodes,
            "seed": args.seed,
            "n_boot": args.n_boot,
            "random_search": args.random_search,
            "base_scenario_name": "minimal_specialization",
        },
        "cells": cells,
    }

    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2, sort_keys=False)
        f.write("\n")
    print(f"Wrote {args.out_json}")

    with open(args.out_md, "w") as f:
        f.write(render_markdown(result))
    print(f"Wrote {args.out_md}")

    for tag, cell in cells.items():
        head = cell["headroom"]
        print(
            f"{tag}: best team = {head['best_team_policy']}"
            f" ({head['team_pct_over_uniform']:+.3f}% vs uniform);"
            f" best BR-agent = {head['best_br_agent_policy']}"
            f" ({head['br_agent_pct_over_uniform']:+.3f}% vs uniform)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
