"""Tier-1 sweep driver for the #343 trainer matrix.

Takes a trainer name + scenario + seeds and produces a uniform cell artifact
tree under ``--output-root/<trainer>_<scenario>/`` containing per-seed
``metrics.json`` + ``config.json`` (as written by ``train.py``) and a
post-run ``cell_summary.json`` with the canonical Tier-1 schema.

The dispatcher knows about 14 trainer names (see :data:`TRAINERS`):

* 12 go through ``experiments/p3_specialization/train.py`` with different
  selector flags (``--algorithm``, ``--centralized-critic``, ``--use-coma``,
  ``--use-hca``, ``--lola-dice``, ``--influence-coef``, ``--gae-lambda``,
  ``--macro-actions``, ``--progress-shaping-coef``, ``--team-welfare-*``).
* Two BC-init variants run a two-step flow:
  ``bc_init.py`` first, then ``train.py --bc-init-checkpoint-dir``.
* ``pbt`` is the only trainer that doesn't go through ``train.py``; it
  shells out to ``experiments/p3_specialization/run_issue288_pbt.py``.

``gap_closed`` is computed via ``bucket_brigade.baselines.MINSPEC_RANDOM`` /
``MINSPEC_SPECIALIST`` (same formula as ``analyze_270.py:45-46``).

Usage:

    uv run python experiments/p3_specialization/run_tier1_cell.py \\
        --trainer lola \\
        --scenario minimal_specialization \\
        --seeds 42 43 44 \\
        --num-iterations 50

See issue #345 / parent #343 for the trainer matrix and the verdict ladder.

CLAUDE.md note: real sweeps run remotely on COMPUTE_HOST_PRIMARY. The
``--num-iterations 1 --rollout-steps 64`` smoke test is the only safe
local invocation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess  # nosec B404 (orchestrator spawns train.py with fixed argv)
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
P3_DIR = REPO_ROOT / "experiments" / "p3_specialization"
TRAIN_PY_MODULE = "experiments.p3_specialization.train"
BC_INIT_PY_MODULE = "experiments.p3_specialization.bc_init"
PBT_SCRIPT = P3_DIR / "run_issue288_pbt.py"

# Number of trailing iterations used for the "trailing-5" trajectory summary
# (matches ``analyze_270.py:TRAILING_N``).
TRAILING_N = 5

# Verdict ladder (#343 / curator's schema, mirroring the historical 4-tier
# ladder from ``experiments/p3_specialization/diagnostics/random_mlp_search.py``
# used in #271/#277 verdict notebooks).
#
# Tiers (in descending order of gap_closed_mean):
#   gap_closed >= 0.88           -> ``closed``         (stunning_near_specialist)
#   0.49 <= gap_closed < 0.88    -> ``partial_upper``  (anti-attractor confirmed)
#   0.20 <= gap_closed < 0.49    -> ``partial_lower``  (basin-trap consistent)
#   gap_closed < 0.20            -> ``insufficient``   (random-play basin)
VERDICT_THRESHOLDS = (0.20, 0.49, 0.88)


# ---------------------------------------------------------------------------
# Trainer dispatch table
# ---------------------------------------------------------------------------


def _python() -> str:
    """Return the Python interpreter used for child invocations.

    Falls back to ``sys.executable`` so the same venv is used both in CI
    (where ``uv run`` re-resolves to a different cache) and in local smoke
    tests.
    """
    return sys.executable


def _train_argv(extra: Sequence[str] = ()) -> List[str]:
    """Base ``python -m train`` argv. Per-seed flags appended by caller."""
    return [_python(), "-m", TRAIN_PY_MODULE, *extra]


def _bc_init_argv(extra: Sequence[str] = ()) -> List[str]:
    return [_python(), "-m", BC_INIT_PY_MODULE, *extra]


@dataclass(frozen=True)
class TrainerSpec:
    """A single trainer's dispatch description.

    Most trainers are single-step (``train.py`` with extra flags). BC-init
    variants are two-step and override ``run_seed``. PBT is a one-shot
    shell-out to its own script over all seeds at once.
    """

    name: str
    description: str
    # Extra argv tokens appended to the ``train.py`` invocation. Ignored when
    # ``run_seed`` is overridden.
    train_extra: tuple[str, ...] = ()
    # Per-seed runner. Default: build argv via ``_build_train_argv`` and
    # invoke ``train.py`` once. Two-step / non-``train.py`` trainers override.
    run_seed: Optional[Callable[..., List[List[str]]]] = None
    # If True, the trainer is dispatched once for all seeds (PBT). The
    # per-seed loop is skipped and ``run_pbt`` is invoked instead.
    is_pbt: bool = False


def _build_train_argv(
    spec: TrainerSpec,
    *,
    scenario: str,
    seed: int,
    output_dir: Path,
    num_iterations: int,
    rollout_steps: int,
) -> List[str]:
    """Construct the ``train.py`` argv for a single seed."""
    argv = _train_argv(
        [
            "--scenario",
            scenario,
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
            *spec.train_extra,
        ]
    )
    return argv


def _bc_init_then_train(
    extra_train: Sequence[str],
) -> Callable[..., List[List[str]]]:
    """Build a two-step runner: BC fit then ``train.py --bc-init-checkpoint-dir``."""

    def runner(
        spec: TrainerSpec,
        *,
        scenario: str,
        seed: int,
        output_dir: Path,
        num_iterations: int,
        rollout_steps: int,
    ) -> List[List[str]]:
        bc_dir = output_dir / "bc_init"
        bc_argv = _bc_init_argv(
            [
                "--scenario",
                scenario,
                "--seed",
                str(seed),
                "--output-dir",
                str(bc_dir),
            ]
        )
        train_argv = _train_argv(
            [
                "--scenario",
                scenario,
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
                "--bc-init-checkpoint-dir",
                str(bc_dir),
                *extra_train,
            ]
        )
        return [bc_argv, train_argv]

    return runner


# Module-level dispatch table. Keys are the user-facing trainer names from
# the #343 matrix. Values describe the dispatch (argv + optional two-step
# or PBT shell-out).
TRAINERS: dict[str, TrainerSpec] = {
    "ippo": TrainerSpec(
        name="ippo",
        description="Independent PPO baseline (--algorithm ppo, no extras).",
        train_extra=("--algorithm", "ppo"),
    ),
    "het_ppo": TrainerSpec(
        name="het_ppo",
        description=(
            "HetGPPO-style asymmetry-aware PPO (issue #386): IPPO with "
            "--per-agent-init-seed-offset 1000 so each per-position policy "
            "is initialized from a maximally-distinct RNG stream. Designed "
            "for asymmetric_only phase-diagram cells (e.g. rest_trap) where "
            "the Nash equilibrium is per-position-distinct and shared-stream "
            "init can trap SGD in a symmetric basin."
        ),
        train_extra=("--algorithm", "ppo", "--per-agent-init-seed-offset", "1000"),
    ),
    "mappo": TrainerSpec(
        name="mappo",
        description="MAPPO: centralized critic, decentralized actors.",
        train_extra=("--centralized-critic",),
    ),
    "high_lambda": TrainerSpec(
        name="high_lambda",
        description="High-lambda GAE PPO (lambda=0.99).",
        train_extra=("--gae-lambda", "0.99"),
    ),
    "bc_init_continuation": TrainerSpec(
        name="bc_init_continuation",
        description="BC fit then PPO continuation (--bc-init-checkpoint-dir).",
        run_seed=_bc_init_then_train(()),
    ),
    "bc_init_high_lambda": TrainerSpec(
        name="bc_init_high_lambda",
        description="BC fit then high-lambda PPO continuation.",
        run_seed=_bc_init_then_train(("--gae-lambda", "0.99")),
    ),
    "lola": TrainerSpec(
        name="lola",
        description="LOLA-DiCE opponent shaping.",
        train_extra=("--lola-dice",),
    ),
    "coma": TrainerSpec(
        name="coma",
        description="COMA counterfactual multi-agent policy gradient.",
        train_extra=("--use-coma",),
    ),
    "hca": TrainerSpec(
        name="hca",
        description="Hindsight credit assignment.",
        train_extra=("--use-hca",),
    ),
    "influence": TrainerSpec(
        name="influence",
        description="Social influence intrinsic motivation (alpha=0.5).",
        train_extra=("--influence-coef", "0.5"),
    ),
    "nhr": TrainerSpec(
        name="nhr",
        description="Potential-based team-welfare shaping (NHR).",
        train_extra=(
            "--team-welfare-lambda",
            "0.5",
            "--team-welfare-kind",
            "team_welfare_closed_form",
        ),
    ),
    "progress": TrainerSpec(
        name="progress",
        description="Dense Δsafe progress shaping (coef=1.0).",
        train_extra=("--progress-shaping-coef", "1.0"),
    ),
    "macro_actions": TrainerSpec(
        name="macro_actions",
        description="MacroActionEnv wrapper (--macro-actions).",
        train_extra=("--macro-actions",),
    ),
    "reinforce": TrainerSpec(
        name="reinforce",
        description="Vanilla REINFORCE (--algorithm reinforce).",
        train_extra=("--algorithm", "reinforce"),
    ),
    "pbt": TrainerSpec(
        name="pbt",
        description="Population-based training (separate entry script).",
        is_pbt=True,
    ),
}


# ---------------------------------------------------------------------------
# gap_closed helpers (mirrors analyze_270.py for the Tier-1 schema)
# ---------------------------------------------------------------------------


def _import_baselines() -> tuple[float, float]:
    """Return ``(MINSPEC_RANDOM, MINSPEC_SPECIALIST)`` from
    ``bucket_brigade.baselines``. Imported lazily so the dispatcher can be
    invoked from a stripped environment in unit tests.
    """
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    return MINSPEC_RANDOM, MINSPEC_SPECIALIST


def gap_closed(per_step_team: float) -> float:
    """Canonical Tier-1 gap_closed (same formula as analyze_270.py:45-46)."""
    random_, specialist_ = _import_baselines()
    denom = specialist_ - random_
    if denom == 0:
        return 0.0
    return (per_step_team - random_) / denom


def _trajectory(metrics: list[dict]) -> list[float]:
    """Per-iteration ``mean_step_reward_team`` as a Python list."""
    return [float(row["mean_step_reward_team"]) for row in metrics]


def _verdict_for(mean: float) -> tuple[str, str]:
    low, mid, high = VERDICT_THRESHOLDS
    if mean >= high:
        return "closed", f"gap_closed_mean = {mean:.3f} >= {high}"
    if mean >= mid:
        return "partial_upper", f"{mid} <= gap_closed_mean < {high}"
    if mean >= low:
        return "partial_lower", f"{low} <= gap_closed_mean < {mid}"
    return "insufficient", f"gap_closed_mean = {mean:.3f} < {low}"


# ---------------------------------------------------------------------------
# Cell-summary aggregation (post-run, single trainer × single scenario)
# ---------------------------------------------------------------------------


def _safe_json_dump(obj: object) -> str:
    """``json.dumps`` with NaN -> ``null`` so output is portable JSON."""

    def _replace(o):
        if isinstance(o, float) and math.isnan(o):
            return None
        if isinstance(o, dict):
            return {k: _replace(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_replace(x) for x in o]
        return o

    return json.dumps(_replace(obj), indent=2)


def _load_seed_metrics(seed_dir: Path) -> Optional[list[dict]]:
    f = seed_dir / "metrics.json"
    if not f.exists():
        return None
    try:
        return json.loads(f.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def build_cell_summary(
    *,
    trainer: str,
    scenario: str,
    seeds: Sequence[int],
    seed_dirs: Sequence[Path],
    num_iterations: int,
    command_invoked: str,
    git_sha: str,
    wall_clock_seconds: float,
) -> dict:
    """Aggregate per-seed metrics into the curator-defined schema."""
    import statistics

    completed: list[tuple[int, list[float]]] = []
    failed: list[int] = []
    for seed, sdir in zip(seeds, seed_dirs):
        m = _load_seed_metrics(sdir)
        if m is None or len(m) == 0:
            failed.append(seed)
            continue
        completed.append((seed, _trajectory(m)))

    per_seed_gap: list[float] = []
    iter0_gaps: list[float] = []
    min_gaps: list[float] = []
    trailing5_teams: list[float] = []
    aligned_trajs: list[list[float]] = []
    for _seed, traj in completed:
        trail = traj[-TRAILING_N:] if len(traj) >= TRAILING_N else traj
        trailing5_team = sum(trail) / len(trail)
        trailing5_teams.append(trailing5_team)
        per_seed_gap.append(gap_closed(trailing5_team))
        iter0_gaps.append(gap_closed(traj[0]))
        min_gaps.append(min(gap_closed(x) for x in traj))
        aligned_trajs.append(traj)

    if completed:
        min_len = min(len(t) for _, t in completed)
        # Mean trajectory in gap_closed space, aligned to the shortest run.
        mean_traj_step = [
            sum(t[i] for _, t in completed) / len(completed) for i in range(min_len)
        ]
        mean_traj_gc = [gap_closed(x) for x in mean_traj_step]
        gap_closed_mean = sum(per_seed_gap) / len(per_seed_gap)
        gap_closed_std = (
            statistics.pstdev(per_seed_gap) if len(per_seed_gap) > 1 else 0.0
        )
        trailing5_team_mean = sum(trailing5_teams) / len(trailing5_teams)
        iter0_gap_closed_mean = sum(iter0_gaps) / len(iter0_gaps)
        min_iter_gap_closed_mean = sum(min_gaps) / len(min_gaps)
    else:
        mean_traj_gc = []
        gap_closed_mean = float("nan")
        gap_closed_std = 0.0
        trailing5_team_mean = float("nan")
        iter0_gap_closed_mean = float("nan")
        min_iter_gap_closed_mean = float("nan")

    if completed:
        verdict, reason = _verdict_for(gap_closed_mean)
    else:
        verdict, reason = "no_data", "no seeds produced metrics.json"

    return {
        "trainer": trainer,
        "scenario": scenario,
        "seeds": list(seeds),
        "num_iterations": num_iterations,
        "n_seeds_completed": len(completed),
        "n_seeds_failed": len(failed),
        "failed_seeds": failed,
        "gap_closed_mean": gap_closed_mean,
        "gap_closed_std": gap_closed_std,
        "gap_closed_per_seed": per_seed_gap,
        "trailing5_team_mean": trailing5_team_mean,
        "iter0_gap_closed_mean": iter0_gap_closed_mean,
        "min_iter_gap_closed_mean": min_iter_gap_closed_mean,
        "mean_traj_gap_closed": mean_traj_gc,
        "verdict_tier": verdict,
        "verdict_reason": reason,
        "command_invoked": command_invoked,
        "git_sha": git_sha,
        "wall_clock_seconds": wall_clock_seconds,
    }


# ---------------------------------------------------------------------------
# Subprocess plumbing
# ---------------------------------------------------------------------------


def _run_subprocess(argv: Sequence[str], *, cwd: Path) -> int:
    """Run a subprocess and stream its output. Returns the exit code."""
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


def _disk_precheck(output_dir: Path) -> None:
    """Lazy-load the disk precheck so unit tests can stub it cleanly."""
    spec = importlib.util.spec_from_file_location(
        "_disk_precheck",
        REPO_ROOT / "experiments" / "scripts" / "_disk_precheck.py",
    )
    if spec is None or spec.loader is None:
        return  # precheck is best-effort; skip silently if absent
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.check_free_space(output_dir)


# ---------------------------------------------------------------------------
# PBT handling (one shell-out for all seeds)
# ---------------------------------------------------------------------------


def _pbt_argv(
    *,
    scenario: str,
    seeds: Sequence[int],
    output_dir: Path,
    num_iterations: int,
    rollout_steps: int,
) -> List[str]:
    return [
        _python(),
        str(PBT_SCRIPT),
        "--scenario",
        scenario,
        "--seeds",
        *[str(s) for s in seeds],
        "--output-dir",
        str(output_dir),
        "--iters-per-gen",
        str(num_iterations),
        "--rollout-steps",
        str(rollout_steps),
    ]


# ---------------------------------------------------------------------------
# Main per-cell entry point
# ---------------------------------------------------------------------------


def build_argvs_for_seed(
    trainer: str,
    *,
    scenario: str,
    seed: int,
    output_dir: Path,
    num_iterations: int,
    rollout_steps: int,
) -> List[List[str]]:
    """Return the list of argvs to dispatch for a single seed.

    For single-step trainers, this is one argv. For BC-init variants, two.
    PBT is dispatched separately (not via this function).
    """
    spec = TRAINERS[trainer]
    if spec.is_pbt:
        raise ValueError("PBT is dispatched via build_pbt_argv(); not per-seed.")
    if spec.run_seed is not None:
        return spec.run_seed(
            spec,
            scenario=scenario,
            seed=seed,
            output_dir=output_dir,
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
        )
    return [
        _build_train_argv(
            spec,
            scenario=scenario,
            seed=seed,
            output_dir=output_dir,
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
        )
    ]


def build_pbt_argv(
    *,
    scenario: str,
    seeds: Sequence[int],
    output_dir: Path,
    num_iterations: int,
    rollout_steps: int,
) -> List[str]:
    """Return the one-shot PBT argv covering all seeds."""
    return _pbt_argv(
        scenario=scenario,
        seeds=seeds,
        output_dir=output_dir,
        num_iterations=num_iterations,
        rollout_steps=rollout_steps,
    )


def run_cell(
    *,
    trainer: str,
    scenario: str,
    seeds: Sequence[int],
    num_iterations: int,
    rollout_steps: int,
    output_root: Path,
    cwd: Path = REPO_ROOT,
    skip_precheck: bool = False,
) -> dict:
    """Run a full Tier-1 cell (trainer × scenario × seeds) and write
    ``cell_summary.json``. Returns the summary dict.

    All subprocesses run with ``cwd=cwd`` (default: repo root) so that
    ``-m experiments.p3_specialization.train`` resolves.
    """
    if trainer not in TRAINERS:
        raise SystemExit(f"unknown trainer: {trainer!r}. Known: {sorted(TRAINERS)}")

    cell_dir = output_root / f"{trainer}_{scenario}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    if not skip_precheck:
        _disk_precheck(cell_dir)

    seed_dirs: list[Path] = []
    invoked_commands: list[str] = []

    spec = TRAINERS[trainer]
    start = time.monotonic()

    if spec.is_pbt:
        # PBT writes its own ``seed_<S>/metrics.json`` under ``output-dir``.
        argv = build_pbt_argv(
            scenario=scenario,
            seeds=seeds,
            output_dir=cell_dir,
            num_iterations=num_iterations,
            rollout_steps=rollout_steps,
        )
        invoked_commands.append(" ".join(argv))
        code = _run_subprocess(argv, cwd=cwd)
        if code != 0:
            print(
                f"WARN: PBT dispatch exited {code}; "
                "continuing to aggregation with whatever seeds finished.",
                file=sys.stderr,
            )
        for s in seeds:
            seed_dirs.append(cell_dir / f"seed_{s}")
    else:
        for seed in seeds:
            seed_dir = cell_dir / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            argvs = build_argvs_for_seed(
                trainer,
                scenario=scenario,
                seed=seed,
                output_dir=seed_dir,
                num_iterations=num_iterations,
                rollout_steps=rollout_steps,
            )
            for argv in argvs:
                invoked_commands.append(" ".join(argv))
                code = _run_subprocess(argv, cwd=cwd)
                if code != 0:
                    print(
                        f"WARN: seed {seed} step exited {code}; "
                        "moving on to the next seed.",
                        file=sys.stderr,
                    )
                    break
            seed_dirs.append(seed_dir)

    wall_clock = time.monotonic() - start

    summary = build_cell_summary(
        trainer=trainer,
        scenario=scenario,
        seeds=seeds,
        seed_dirs=seed_dirs,
        num_iterations=num_iterations,
        command_invoked=" && ".join(invoked_commands),
        git_sha=_git_sha(cwd),
        wall_clock_seconds=wall_clock,
    )

    out_path = cell_dir / "cell_summary.json"
    out_path.write_text(_safe_json_dump(summary))
    print(
        f"\n== cell {trainer}_{scenario} complete: verdict={summary['verdict_tier']} "
        f"gap_closed_mean={summary['gap_closed_mean']!r} -> {out_path}",
        flush=True,
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--trainer",
        required=True,
        choices=sorted(TRAINERS),
        help="Trainer name (one of the #343 matrix arms).",
    )
    p.add_argument(
        "--scenario",
        default="minimal_specialization",
        help="Scenario name (default: minimal_specialization, the Tier-1 baseline).",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 43, 44],
        help="Seeds to sweep (default: 42 43 44).",
    )
    p.add_argument("--num-iterations", type=int, default=50)
    p.add_argument("--rollout-steps", type=int, default=2048)
    p.add_argument(
        "--output-root",
        type=Path,
        default=P3_DIR / "tier1_runs",
        help="Root for tier-1 cell outputs (default: experiments/p3_specialization/tier1_runs/).",
    )
    p.add_argument(
        "--skip-precheck",
        action="store_true",
        help="Skip the disk-space precheck (use for tests).",
    )
    args = p.parse_args(argv)

    summary = run_cell(
        trainer=args.trainer,
        scenario=args.scenario,
        seeds=args.seeds,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        output_root=args.output_root,
        skip_precheck=args.skip_precheck,
    )
    # Exit non-zero if we got no seeds completed; otherwise success even if
    # partial (so a remote sweep harness can still aggregate).
    return 0 if summary["n_seeds_completed"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
