"""Phase-diagram PPO sweep driver (issue #360).

Trains ``JointPPOTrainer`` (IPPO) across the same (β, κ, c) grid the
heterogeneous Nash phase-diagram search (#358) used, and writes
per-(cell × seed) ``summary.json`` plus per-cell ``cell_summary.json``
artifacts that the NE-vs-PPO side-by-side figure (parent issue #357 / M2.1)
consumes.

Cell source
-----------
By default we read the cells from
``experiments/nash/phase_diagram/results.json``. Each entry has a ``beta``,
``kappa``, ``c`` triple plus a tag and the NE-search verdict — we use only
the triple here; the verdict is what we cross-tabulate against post-hoc.

Per-cell training
-----------------
For each (β, κ, c) cell:
    for each seed in --seeds:
        python -m experiments.p3_specialization.train \\
            --scenario <--scenario>           # default minimal_specialization
            --algorithm ppo                   # JointPPOTrainer (IPPO)
            --lambda-red 0.0
            --seed <seed>
            --output-dir <out>/cell_<tag>/seed_<seed>
            --num-iterations <N>
            --rollout-steps <R>
            --prob-fire-spreads-to-neighbor <β>
            --prob-solo-agent-extinguishes-fire <κ>
            --cost-to-work-one-night <c>

The three new ``--prob-*`` / ``--cost-*`` flags on ``train.py`` were added
in #360 alongside this driver: they mutate the loaded ``Scenario`` instance
the same way ``--action-shaping-alpha`` does, mirroring the
``compute_nash_phase_diagram._make_scenario`` wiring so the NE-search and
PPO-training rewards are bit-comparable.

Per-cell aggregation
--------------------
After every (cell × seed) finishes we read ``metrics.json`` from each seed
dir, compute ``gap_closed`` against the
``bucket_brigade.baselines.MINSPEC_RANDOM`` / ``MINSPEC_SPECIALIST``
baselines (same formula as ``analyze_270.py:45-46`` and ``run_tier1_cell``),
and write:

    <output-root>/cell_<tag>/seed_<seed>/summary.json
        {cell, seed, gap_closed, verdict, final_episode_return, wall_seconds}

    <output-root>/cell_<tag>/cell_summary.json
        per-cell aggregate using run_tier1_cell.build_cell_summary

The cell-level aggregate is the same Tier-1 schema (gap_closed_mean ± std,
verdict tier, per-seed gaps, mean trajectory) so the existing
``aggregate_tier1.py`` works on the output root without modification.

Per-seed ``summary.json`` is a small superset of the Tier-1 schema that
includes the NE-search cell context (β, κ, c, NE verdict if known) — that
is what the cross-tab figure / Spearman ρ correlation in #360's acceptance
criteria consume.

CLAUDE.md note
--------------
This driver is meant to be run on a remote host (alc-* / Mac Studio) via
``experiments/scripts/launch_phase_diagram_ppo.sh``. The launcher does the
ssh + tmux + bootstrap. This file knows nothing about ssh; running it
locally on the full grid will melt the CPU.

Smoke-test (safe locally, validates wiring only):

    uv run python experiments/p3_specialization/run_phase_diagram_ppo.py \\
        --cells-source experiments/nash/phase_diagram/results.json \\
        --seeds 42 \\
        --num-iterations 1 \\
        --rollout-steps 64 \\
        --output-root /tmp/phase_diagram_ppo_smoke
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess  # nosec B404 (orchestrator spawns train.py with fixed argv)
import sys
import time
from pathlib import Path
from typing import Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
P3_DIR = REPO_ROOT / "experiments" / "p3_specialization"
TRAIN_PY_MODULE = "experiments.p3_specialization.train"

# Default NE phase-diagram results file (7 cells as of #358's partial
# aggregate). The driver also accepts the freqtest variant or any other
# results.json with the same schema.
DEFAULT_CELLS_SOURCE = (
    REPO_ROOT / "experiments" / "nash" / "phase_diagram" / "results.json"
)


# ---------------------------------------------------------------------------
# Cell loading
# ---------------------------------------------------------------------------


def _cell_tag(beta: float, kappa: float, c: float) -> str:
    """Filesystem-safe tag — matches ``compute_nash_phase_diagram._cell_tag``."""
    return f"b{beta:.2f}_k{kappa:.2f}_c{c:.2f}"


def load_cells(source: Path) -> list[dict]:
    """Load the phase-diagram cells from a results.json.

    Returns a list of dicts each containing at least ``beta``, ``kappa``,
    ``c``, ``tag``. Any extra keys (e.g. NE ``verdict``) are passed through
    untouched so the per-seed ``summary.json`` can include the NE-search
    context the cross-tab figure needs.
    """
    if not source.exists():
        raise SystemExit(
            f"cells-source not found: {source}. Did #358 land its results.json?"
        )
    try:
        data = json.loads(source.read_text())
    except json.JSONDecodeError as exc:
        raise SystemExit(f"cells-source {source} is not valid JSON: {exc}") from exc

    cells = data.get("cells")
    if not isinstance(cells, list) or not cells:
        raise SystemExit(
            f"cells-source {source} has no 'cells' array. "
            f"Expected the schema from compute_nash_phase_diagram.py."
        )

    normalized: list[dict] = []
    for raw in cells:
        try:
            beta = float(raw["beta"])
            kappa = float(raw["kappa"])
            c = float(raw["c"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"cell entry missing required (β,κ,c): {raw}. ({exc})"
            ) from exc
        normalized.append(
            {
                "beta": beta,
                "kappa": kappa,
                "c": c,
                "tag": raw.get("tag") or _cell_tag(beta, kappa, c),
                "ne_verdict": raw.get("verdict"),
                "ne_verdict_detail": raw.get("verdict_detail"),
            }
        )
    return normalized


# ---------------------------------------------------------------------------
# Per-seed train.py dispatch
# ---------------------------------------------------------------------------


def _python() -> str:
    return sys.executable


def build_train_argv(
    *,
    scenario: str,
    seed: int,
    output_dir: Path,
    num_iterations: int,
    rollout_steps: int,
    beta: float,
    kappa: float,
    c: float,
) -> list[str]:
    """Build the ``train.py`` argv for one seed inside one cell.

    Uses the IPPO baseline (``--algorithm ppo``) — that is ``JointPPOTrainer``
    in its default decentralized-critic configuration. Issue #360's
    hypothesis is about PPO trainability per NE structure, so we want the
    baseline trainer here, not MAPPO / LOLA / etc.
    """
    return [
        _python(),
        "-m",
        TRAIN_PY_MODULE,
        "--scenario",
        scenario,
        "--algorithm",
        "ppo",
        "--lambda-red",
        "0.0",
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
        "--num-iterations",
        str(num_iterations),
        "--rollout-steps",
        str(rollout_steps),
        "--prob-fire-spreads-to-neighbor",
        str(beta),
        "--prob-solo-agent-extinguishes-fire",
        str(kappa),
        "--cost-to-work-one-night",
        str(c),
    ]


def _run_subprocess(argv: Sequence[str], *, cwd: Path) -> int:
    print(f"\n$ {' '.join(argv)}", flush=True)
    completed = subprocess.run(list(argv), cwd=str(cwd))  # nosec B603 (cmd is list, no shell)
    return int(completed.returncode)


def _git_sha(cwd: Path) -> str:
    try:
        out = subprocess.run(  # nosec B603 B607 (git rev-parse — argv is hardcoded, not user input)
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


# ---------------------------------------------------------------------------
# Per-seed summary
# ---------------------------------------------------------------------------


def _load_metrics(seed_dir: Path) -> Optional[list[dict]]:
    f = seed_dir / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _safe_json_dump(obj: object) -> str:
    """``json.dumps`` with NaN -> null so output is portable JSON.

    Mirrors ``run_tier1_cell._safe_json_dump`` so downstream consumers can
    treat phase-diagram and Tier-1 summaries identically.
    """

    def _replace(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _replace(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_replace(x) for x in o]
        return o

    return json.dumps(_replace(obj), indent=2)


def _verdict_for(gap_closed_value: float) -> tuple[str, str]:
    """Same verdict ladder as ``run_tier1_cell._verdict_for``.

    Re-implemented inline (rather than imported) so this driver is a
    self-contained read of the per-seed metrics — no import-time coupling
    to the Tier-1 thresholds beyond the constants we copy below.
    """
    # Mirror run_tier1_cell.VERDICT_THRESHOLDS — re-declared so a change
    # over there is a visible diff here, not a silent drift.
    low, mid, high = 0.20, 0.49, 0.88
    if gap_closed_value >= high:
        return "closed", f"gap_closed = {gap_closed_value:.3f} >= {high}"
    if gap_closed_value >= mid:
        return "partial_upper", f"{mid} <= gap_closed < {high}"
    if gap_closed_value >= low:
        return "partial_lower", f"{low} <= gap_closed < {mid}"
    return "insufficient", f"gap_closed = {gap_closed_value:.3f} < {low}"


def write_seed_summary(
    *,
    cell: dict,
    scenario: str,
    seed: int,
    seed_dir: Path,
    wall_seconds: float,
    command_invoked: str,
    git_sha: str,
) -> dict:
    """Compute the per-seed ``summary.json`` payload and write it to disk.

    Schema (matches issue #360's "at minimum" list, plus NE-search context):
        cell:                   {beta, kappa, c, tag, ne_verdict?, ne_verdict_detail?}
        scenario:               base scenario name (defaults to minimal_specialization)
        seed:                   int
        gap_closed:             trailing-5 mean_step_reward_team mapped via MINSPEC baselines
        verdict:                tier name from the run_tier1_cell ladder
        verdict_reason:         human-readable threshold string
        final_episode_return:   last iteration's mean episode return (per-agent average)
        trailing5_team_mean:    trailing-5 mean_step_reward_team (raw, pre-gap-closed)
        wall_seconds:           subprocess wall time
        command_invoked, git_sha
        n_iterations_completed: how many iterations metrics.json actually has
    """
    # Lazy import — keeps the driver importable in environments without the
    # Rust extension built (e.g. for CLI introspection / --help).
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    metrics = _load_metrics(seed_dir)
    if metrics is None or len(metrics) == 0:
        payload = {
            "cell": cell,
            "scenario": scenario,
            "seed": seed,
            "gap_closed": None,
            "verdict": "no_data",
            "verdict_reason": "metrics.json missing or empty",
            "final_episode_return": None,
            "trailing5_team_mean": None,
            "wall_seconds": wall_seconds,
            "command_invoked": command_invoked,
            "git_sha": git_sha,
            "n_iterations_completed": 0,
        }
    else:
        trajectory = [float(row["mean_step_reward_team"]) for row in metrics]
        trail = trajectory[-5:] if len(trajectory) >= 5 else trajectory
        trailing5 = sum(trail) / len(trail)
        denom = MINSPEC_SPECIALIST - MINSPEC_RANDOM
        gap_closed_val = (trailing5 - MINSPEC_RANDOM) / denom if denom != 0 else 0.0
        verdict, reason = _verdict_for(gap_closed_val)
        # ``mean_episode_return`` is the canonical per-iteration team-return
        # metric that ``train.py`` writes; we just expose the final one here.
        final_ret_key = (
            "mean_episode_return"
            if "mean_episode_return" in metrics[-1]
            else "mean_step_reward_team"
        )
        final_ret = float(metrics[-1].get(final_ret_key, float("nan")))
        payload = {
            "cell": cell,
            "scenario": scenario,
            "seed": seed,
            "gap_closed": gap_closed_val,
            "verdict": verdict,
            "verdict_reason": reason,
            "final_episode_return": final_ret,
            "trailing5_team_mean": trailing5,
            "wall_seconds": wall_seconds,
            "command_invoked": command_invoked,
            "git_sha": git_sha,
            "n_iterations_completed": len(metrics),
        }

    (seed_dir / "summary.json").write_text(_safe_json_dump(payload))
    return payload


# ---------------------------------------------------------------------------
# Per-cell aggregation (Tier-1 schema reuse)
# ---------------------------------------------------------------------------


def build_cell_summary_for_phase_diagram(
    *,
    cell: dict,
    scenario: str,
    seeds: Sequence[int],
    seed_dirs: Sequence[Path],
    num_iterations: int,
    rollout_steps: int,
    command_invoked: str,
    git_sha: str,
    wall_clock_seconds: float,
) -> dict:
    """Aggregate per-seed metrics for one (β, κ, c) cell.

    Delegates the gap_closed / verdict ladder math to
    ``run_tier1_cell.build_cell_summary`` so the output is schema-compatible
    with ``aggregate_tier1.py``. Adds the (β, κ, c) cell context as an
    extra top-level field so the cross-tab figure / Spearman correlation
    in #360's acceptance criteria can pivot on it without re-reading the
    NE-search results.json.
    """
    # Lazy import — run_tier1_cell imports the baselines and other modules
    # that need the Rust extension built; lifting that to driver import-time
    # would break ``--help`` on a freshly-cloned host.
    from experiments.p3_specialization.run_tier1_cell import build_cell_summary

    tier1_like = build_cell_summary(
        trainer="ippo",
        scenario=scenario,
        seeds=seeds,
        seed_dirs=seed_dirs,
        num_iterations=num_iterations,
        command_invoked=command_invoked,
        git_sha=git_sha,
        wall_clock_seconds=wall_clock_seconds,
    )
    # Stitch the phase-diagram-specific context on top of the Tier-1 schema.
    tier1_like["cell"] = cell
    tier1_like["rollout_steps"] = rollout_steps
    tier1_like["sweep"] = "phase_diagram_ppo"
    return tier1_like


# ---------------------------------------------------------------------------
# Main per-cell loop
# ---------------------------------------------------------------------------


def run_cell(
    *,
    cell: dict,
    scenario: str,
    seeds: Sequence[int],
    num_iterations: int,
    rollout_steps: int,
    output_root: Path,
    cwd: Path = REPO_ROOT,
) -> dict:
    """Run all seeds for one (β, κ, c) cell and write summaries."""
    cell_dir = output_root / f"cell_{cell['tag']}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    git_sha = _git_sha(cwd)

    seed_dirs: list[Path] = []
    invoked_commands: list[str] = []
    cell_start = time.monotonic()

    for seed in seeds:
        seed_dir = cell_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        argv = build_train_argv(
            scenario=scenario,
            seed=seed,
            output_dir=seed_dir,
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
            beta=cell["beta"],
            kappa=cell["kappa"],
            c=cell["c"],
        )
        cmd_str = " ".join(argv)
        invoked_commands.append(cmd_str)

        seed_start = time.monotonic()
        code = _run_subprocess(argv, cwd=cwd)
        seed_wall = time.monotonic() - seed_start

        if code != 0:
            print(
                f"WARN: cell {cell['tag']} seed {seed} exited {code}; "
                "writing whatever metrics.json the run produced.",
                file=sys.stderr,
            )

        write_seed_summary(
            cell=cell,
            scenario=scenario,
            seed=seed,
            seed_dir=seed_dir,
            wall_seconds=seed_wall,
            command_invoked=cmd_str,
            git_sha=git_sha,
        )
        seed_dirs.append(seed_dir)

    cell_wall = time.monotonic() - cell_start
    summary = build_cell_summary_for_phase_diagram(
        cell=cell,
        scenario=scenario,
        seeds=seeds,
        seed_dirs=seed_dirs,
        num_iterations=num_iterations,
        rollout_steps=rollout_steps,
        command_invoked=" && ".join(invoked_commands),
        git_sha=git_sha,
        wall_clock_seconds=cell_wall,
    )
    out_path = cell_dir / "cell_summary.json"
    out_path.write_text(_safe_json_dump(summary))
    print(
        f"\n== cell {cell['tag']} complete: verdict={summary['verdict_tier']} "
        f"gap_closed_mean={summary['gap_closed_mean']!r} "
        f"(NE: {cell.get('ne_verdict')}) -> {out_path}",
        flush=True,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--cells-source",
        type=Path,
        default=DEFAULT_CELLS_SOURCE,
        help=(
            "Path to the NE phase-diagram results.json (default: "
            "experiments/nash/phase_diagram/results.json). Must follow the "
            "compute_nash_phase_diagram.py schema with a 'cells' array."
        ),
    )
    p.add_argument(
        "--scenario",
        default="minimal_specialization",
        help=(
            "Base scenario whose (β, κ, c) get overridden per-cell. Default "
            "minimal_specialization, matching compute_nash_phase_diagram's "
            "BASE_SCENARIO_NAME. NOTE: scenario-level reward overrides "
            "(action_shaping_*, team_welfare_*, etc.) compose freely with "
            "the per-cell (β, κ, c) override — they touch different "
            "Scenario fields. Only the three swept fields are touched here."
        ),
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44, 45],
        help="Seeds per cell (default: 42 43 44 45 — 4 seeds, matching #360).",
    )
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument(
        "--output-root",
        type=Path,
        default=P3_DIR / "phase_diagram_ppo",
        help=(
            "Root for per-cell artifacts (default: "
            "experiments/p3_specialization/phase_diagram_ppo/)."
        ),
    )
    p.add_argument(
        "--limit-cells",
        type=int,
        default=None,
        help=(
            "Stop after running this many cells (debug aid). Default None "
            "runs every cell in cells-source."
        ),
    )
    p.add_argument(
        "--list-cells",
        action="store_true",
        help=(
            "Print the (β, κ, c) cells loaded from --cells-source (one per "
            "line) and exit. Used by the launcher to render the pre-launch "
            "plan summary."
        ),
    )
    args = p.parse_args(argv)

    cells = load_cells(args.cells_source)
    if args.limit_cells is not None:
        cells = cells[: args.limit_cells]

    if args.list_cells:
        for cell in cells:
            print(
                f"{cell['tag']}\tβ={cell['beta']:.2f}\tκ={cell['kappa']:.2f}\t"
                f"c={cell['c']:.2f}\tNE={cell.get('ne_verdict', '?')}"
            )
        return 0

    args.output_root.mkdir(parents=True, exist_ok=True)
    print(
        f"== phase-diagram PPO sweep: {len(cells)} cells × {len(args.seeds)} seeds "
        f"× scenario={args.scenario} =="
    )
    print(f"   output-root: {args.output_root}")
    print(f"   cells-source: {args.cells_source}")

    n_failed_cells = 0
    for idx, cell in enumerate(cells, start=1):
        print(
            f"\n[{idx}/{len(cells)}] cell {cell['tag']} "
            f"(β={cell['beta']:.2f} κ={cell['kappa']:.2f} c={cell['c']:.2f}, "
            f"NE={cell.get('ne_verdict', '?')})"
        )
        summary = run_cell(
            cell=cell,
            scenario=args.scenario,
            seeds=args.seeds,
            num_iterations=args.num_iterations,
            rollout_steps=args.rollout_steps,
            output_root=args.output_root,
        )
        if summary["n_seeds_completed"] == 0:
            n_failed_cells += 1

    print(
        f"\n== phase-diagram PPO sweep complete: "
        f"{len(cells) - n_failed_cells}/{len(cells)} cells produced metrics =="
    )
    # Non-zero exit only if *every* cell failed — partial success should let
    # the launcher's post-rsync aggregate step still run.
    return 0 if n_failed_cells < len(cells) else 2


if __name__ == "__main__":
    raise SystemExit(main())
