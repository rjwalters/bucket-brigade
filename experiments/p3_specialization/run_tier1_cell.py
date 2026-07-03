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

``gap_closed`` is **scenario-aware** (issue #434): each scenario resolves a
``(random, reference)`` pair from
``bucket_brigade.baselines.SCENARIO_GAP_REFERENCES`` with three outcomes:

* valid pair (``reference > random``) -> ``gap = (x - random) / (reference -
  random)`` and the #343 verdict ladder applies (``gap_source = "scenario"``).
  For ``minimal_specialization`` this is numerically identical to the
  historical MINSPEC formula (``analyze_270.py:45-46``).
* degenerate reference (no upper reference beats random, e.g. ``rest_trap``'s
  NE-below-random social trap) -> ``gap_closed = null``, the ladder is NOT
  applied (``verdict_tier = "not_scored_degenerate_reference"``), and the
  scenario-scale ``uplift_over_random`` is reported instead. Additionally
  (issue #436) such rows carry a categorical **trap-escape verdict**
  (:func:`classify_trap_verdict`): the seed-bootstrap 95% CI of the
  trailing-5 per-step team reward is classified against up to three anchors
  from the reference table -- ``ne_per_step_bound`` (upper bound on the
  frozen NE's per-step payoff), ``random``, and the measured
  ``scripted_best`` -- into ``trapped_at_ne`` / ``at_random`` /
  ``escaped_trap`` / ``above_scripted_best``.
* unknown scenario (no table entry) -> ``gap_closed = null``, loud stderr
  warning, ``verdict_tier = "not_scored"``. There is deliberately no MINSPEC
  fallback — that silent fallback produced the vacuous ``gap ~ 6.6`` on
  rest_trap (#434).

The MINSPEC-scale value is always kept as the clearly-labeled
``gap_closed_minspec_legacy_mean`` audit column for continuity with
pre-recalibration summaries.

``--summarize-only`` rebuilds ``cell_summary.json`` from existing
``seed_*/metrics.json`` without dispatching any training subprocess (used to
regenerate committed artifacts after metric changes).

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
import random
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


# Schema version for ``cell_summary.json``. Version 2 (issue #434) added
# scenario-aware gap references: ``gap_source``, ``scenario_random_baseline``,
# ``scenario_reference``, ``uplift_over_random_*`` and
# ``gap_closed_minspec_legacy_mean``. Version-1 summaries (implicit, no
# ``schema_version`` key) scored every scenario against the MINSPEC constants.
# Version 3 (issue #436) added the trap-escape verdict fields for
# degenerate-reference rows: ``trap_verdict``, ``trap_verdict_reason``,
# ``trap_anchors`` and ``trailing5_team_ci95``.
SUMMARY_SCHEMA_VERSION = 3

# Trap-verdict bootstrap convention (issue #436): percentile bootstrap over
# SEEDS of the mean trailing-5 per-step team reward, resampled
# ``TRAP_N_BOOT`` times with a fixed stdlib RNG seed so re-summarization is
# deterministic. Documented in the reference-table provenance and
# docs/PAPER_RESULTS.md.
TRAP_N_BOOT = 10_000
TRAP_BOOT_SEED = 436


# ---------------------------------------------------------------------------
# gap_closed helpers (scenario-aware since #434; minspec branch mirrors
# analyze_270.py for the Tier-1 schema)
# ---------------------------------------------------------------------------


def _import_baselines() -> tuple[float, float]:
    """Return ``(MINSPEC_RANDOM, MINSPEC_SPECIALIST)`` from
    ``bucket_brigade.baselines``. Imported lazily so the dispatcher can be
    invoked from a stripped environment in unit tests.
    """
    from bucket_brigade.baselines import MINSPEC_RANDOM, MINSPEC_SPECIALIST

    return MINSPEC_RANDOM, MINSPEC_SPECIALIST


def resolve_scenario_references(scenario: str, *, warn: bool = True) -> dict:
    """Resolve the per-scenario ``(random, reference)`` gap pair (issue #434).

    Returns a dict with keys:

    * ``source`` -- ``"scenario"`` (valid pair, ladder applies),
      ``"degenerate_reference"`` (no upper reference beats random; ladder
      must NOT be applied), or ``"missing_reference"`` (scenario absent from
      ``SCENARIO_GAP_REFERENCES``; not scorable).
    * ``random`` -- per-step uniform-random team reward, or ``None`` if the
      scenario is absent from ``SCENARIO_RANDOM_BASELINES`` too.
    * ``reference`` -- per-step upper-reference team reward, or ``None``.
    * ``reference_kind`` / ``reason`` / ``provenance`` -- audit metadata.
    * ``ne_per_step_bound`` / ``scripted_best`` -- trap-verdict anchors
      (issue #436), passed through from the table entry when present
      (``None`` otherwise). Only consumed on the degenerate-reference path.

    A degenerate pair includes ``reference <= random`` (zero or negative
    denominator), not just a missing upper reference.

    There is deliberately **no MINSPEC fallback** for unknown scenarios:
    named scenarios differ by ~400 per-step reward units, so a MINSPEC-scale
    fraction is never meaningful off ``minimal_specialization``. That silent
    fallback was exactly the #434 bug (vacuous ``gap ~ 6.6`` on rest_trap).
    """
    from bucket_brigade.baselines import (
        SCENARIO_GAP_REFERENCES,
        SCENARIO_RANDOM_BASELINES,
    )

    entry = SCENARIO_GAP_REFERENCES.get(scenario)
    if entry is None:
        if warn:
            print(
                f"WARNING: scenario {scenario!r} has no entry in "
                "bucket_brigade.baselines.SCENARIO_GAP_REFERENCES; "
                "gap_closed will be null and the cell will be reported as "
                "'not_scored'. Measure a (random, reference) pair and add it "
                "to the table before applying the verdict ladder. "
                "(No MINSPEC fallback — see issue #434.)",
                file=sys.stderr,
                flush=True,
            )
        return {
            "source": "missing_reference",
            "random": SCENARIO_RANDOM_BASELINES.get(scenario),
            "reference": None,
            "reference_kind": None,
            "reason": "missing_reference",
            "provenance": None,
            "ne_per_step_bound": None,
            "scripted_best": None,
        }

    random_ = entry["random"]
    reference = entry.get("reference")
    ne_bound = entry.get("ne_per_step_bound")
    scripted_best = entry.get("scripted_best")
    if reference is None or float(reference) - float(random_) <= 0:
        return {
            "source": "degenerate_reference",
            "random": float(random_),
            "reference": float(reference) if reference is not None else None,
            "reference_kind": entry.get("reference_kind"),
            "reason": entry.get("degenerate_reason", "reference_not_above_random"),
            "provenance": entry.get("provenance"),
            "ne_per_step_bound": float(ne_bound) if ne_bound is not None else None,
            "scripted_best": scripted_best,
        }

    return {
        "source": "scenario",
        "random": float(random_),
        "reference": float(reference),
        "reference_kind": entry.get("reference_kind"),
        "reason": None,
        "provenance": entry.get("provenance"),
        "ne_per_step_bound": float(ne_bound) if ne_bound is not None else None,
        "scripted_best": scripted_best,
    }


def gap_closed(per_step_team: float, scenario: str) -> Optional[float]:
    """Scenario-aware Tier-1 gap_closed (issue #434).

    Returns ``None`` when the scenario's reference pair is degenerate or
    missing — the fraction ladder must not be applied to such cells. For
    ``minimal_specialization`` this reproduces the historical MINSPEC
    formula (``analyze_270.py:45-46``) bit-for-bit.
    """
    refs = resolve_scenario_references(scenario, warn=False)
    if refs["source"] != "scenario":
        return None
    return (per_step_team - refs["random"]) / (refs["reference"] - refs["random"])


def gap_closed_minspec_legacy(per_step_team: float) -> float:
    """The pre-#434 MINSPEC-scale gap (audit column only).

    Kept so recalibrated summaries stay comparable with historical
    version-1 summaries that scored every scenario against the MINSPEC
    constants. Never feed this into the verdict ladder off
    ``minimal_specialization``.
    """
    random_, specialist_ = _import_baselines()
    denom = specialist_ - random_
    if denom == 0:
        return 0.0
    return (per_step_team - random_) / denom


def _seed_bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int = TRAP_N_BOOT,
    confidence: float = 0.95,
    boot_seed: int = TRAP_BOOT_SEED,
) -> tuple[float, float, float]:
    """Percentile bootstrap 95% CI of the mean over per-seed values.

    Seeds are the independent replication unit for a trained cell (each
    seed's trailing-5 mean is one draw), so the bootstrap resamples seeds
    with replacement. Stdlib-only (deliberately no numpy import: this
    dispatcher stays importable from a stripped environment in unit tests)
    and deterministic given ``boot_seed``. With a single seed the CI
    degenerates to the point estimate — honest, since one seed carries no
    between-seed uncertainty information.
    """
    vals = [float(v) for v in values]
    if not vals:
        raise ValueError("values must be non-empty")
    n = len(vals)
    point = sum(vals) / n
    if n == 1:
        return point, point, point
    rng = random.Random(boot_seed)
    boots = sorted(
        sum(vals[rng.randrange(n)] for _ in range(n)) / n for _ in range(n_boot)
    )
    alpha = (1.0 - confidence) / 2.0
    lo = boots[max(0, min(n_boot - 1, int(alpha * n_boot)))]
    hi = boots[max(0, min(n_boot - 1, int((1.0 - alpha) * n_boot)))]
    return point, lo, hi


def classify_trap_verdict(
    trailing5_teams: Sequence[float],
    *,
    random_baseline: float,
    ne_per_step_bound: Optional[float],
    scripted_best: Optional[dict],
) -> tuple[str, str, dict]:
    """Four-way trap-escape verdict for degenerate-reference cells (#436).

    Classifies the seed-bootstrap 95% CI ``[lo, hi]`` of the mean trailing-5
    per-step team reward against up to three per-step anchors, as a nested
    one-sided ladder on the CI **lower bound** ("significantly above X" =
    ``lo > X``):

    * ``above_scripted_best`` -- ``lo`` clears the scripted_best anchor
      (its ``ci95_hi`` when available — conservative — else its value).
      Requires a usable scripted_best (present AND value > random; a
      battery that fails to beat random is recorded but never anchors
      this rung — issue #436's documented failure mode).
    * ``escaped_trap``        -- ``lo > random`` (significantly above the
      uniform-random baseline) but not above scripted_best.
    * ``at_random``           -- not significantly above random, but ``lo``
      clears the NE per-step bound (when recorded).
    * ``trapped_at_ne``       -- cannot be ruled significantly above the NE
      bound (CI overlaps or lies below it).

    ``ne_per_step_bound`` is an UPPER bound on the NE per-step payoff
    (per-episode payoff / min_nights), which is conservative in the right
    direction for the "significantly above the NE" claim. Returns
    ``(verdict, reason, details)`` where ``details`` carries the CI and the
    anchors used.
    """
    mean, lo, hi = _seed_bootstrap_ci(trailing5_teams)

    scripted_anchor: Optional[float] = None
    scripted_note = "no scripted_best recorded"
    if scripted_best is not None:
        sb_value = float(scripted_best["value"])
        if sb_value > random_baseline:
            scripted_anchor = float(scripted_best.get("ci95_hi", sb_value))
        else:
            scripted_note = (
                f"scripted_best = {sb_value:.2f} <= random "
                f"{random_baseline:.2f}: not usable as an upper anchor "
                "(#436 failure mode); classifying on NE + random anchors only"
            )

    details = {
        "trailing5_ci95": [lo, hi],
        "trailing5_mean": mean,
        "n_seeds": len(trailing5_teams),
        "n_boot": TRAP_N_BOOT,
        "boot_seed": TRAP_BOOT_SEED,
        "anchors": {
            "ne_per_step_bound": ne_per_step_bound,
            "random": random_baseline,
            "scripted_best_anchor": scripted_anchor,
        },
    }

    ci_str = (
        f"trailing-5 mean {mean:.2f}/step, seed-bootstrap 95% CI [{lo:.2f}, {hi:.2f}]"
    )
    if scripted_anchor is not None and lo > scripted_anchor:
        return (
            "above_scripted_best",
            f"{ci_str}: CI lower bound clears the scripted_best anchor "
            f"{scripted_anchor:.2f}",
            details,
        )
    if lo > random_baseline:
        upper_note = (
            f"below the scripted_best anchor {scripted_anchor:.2f}"
            if scripted_anchor is not None
            else scripted_note
        )
        return (
            "escaped_trap",
            f"{ci_str}: significantly above random {random_baseline:.2f}, {upper_note}",
            details,
        )
    if ne_per_step_bound is not None and lo > ne_per_step_bound:
        return (
            "at_random",
            f"{ci_str}: CI overlaps random {random_baseline:.2f} but clears "
            f"the NE per-step bound {ne_per_step_bound:.2f}",
            details,
        )
    if ne_per_step_bound is None:
        return (
            "at_random",
            f"{ci_str}: CI overlaps random {random_baseline:.2f}; no NE "
            "per-step bound recorded for this scenario",
            details,
        )
    return (
        "trapped_at_ne",
        f"{ci_str}: CI does not clear the NE per-step bound {ne_per_step_bound:.2f}",
        details,
    )


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
    """Aggregate per-seed metrics into the curator-defined schema.

    Scenario-aware since #434 (``SUMMARY_SCHEMA_VERSION = 2``): the gap
    columns are only populated when the scenario resolves a valid
    ``(random, reference)`` pair. Degenerate/missing references produce
    null gap columns, a ``not_scored*`` verdict tier and (when the random
    baseline is known) the scenario-scale ``uplift_over_random`` columns.
    The MINSPEC-scale value is always kept as the
    ``gap_closed_minspec_legacy_mean`` audit column.

    Degenerate-reference cells additionally carry the four-way trap-escape
    verdict (issue #436, schema v3): ``trap_verdict``,
    ``trap_verdict_reason``, ``trap_anchors`` and ``trailing5_team_ci95``
    (seed-bootstrap). These fields are ``None`` on every other path.
    """
    import statistics

    refs = resolve_scenario_references(scenario, warn=True)
    scored = refs["source"] == "scenario"

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
    legacy_gaps: list[float] = []
    for _seed, traj in completed:
        trail = traj[-TRAILING_N:] if len(traj) >= TRAILING_N else traj
        trailing5_team = sum(trail) / len(trail)
        trailing5_teams.append(trailing5_team)
        legacy_gaps.append(gap_closed_minspec_legacy(trailing5_team))
        if scored:
            per_seed_gap.append(gap_closed(trailing5_team, scenario))
            iter0_gaps.append(gap_closed(traj[0], scenario))
            min_gaps.append(min(gap_closed(x, scenario) for x in traj))

    # Gap columns: populated only for scored scenarios; null otherwise so the
    # fraction ladder can never be applied to a mismatched reward scale.
    gap_closed_mean: Optional[float] = None
    gap_closed_std: Optional[float] = None
    gap_closed_per_seed: Optional[list[float]] = None
    iter0_gap_closed_mean: Optional[float] = None
    min_iter_gap_closed_mean: Optional[float] = None
    mean_traj_gc: Optional[list[float]] = None

    if completed:
        trailing5_team_mean = sum(trailing5_teams) / len(trailing5_teams)
        gap_closed_minspec_legacy_mean = sum(legacy_gaps) / len(legacy_gaps)
        if scored:
            min_len = min(len(t) for _, t in completed)
            # Mean trajectory in gap_closed space, aligned to the shortest run.
            mean_traj_step = [
                sum(t[i] for _, t in completed) / len(completed) for i in range(min_len)
            ]
            mean_traj_gc = [gap_closed(x, scenario) for x in mean_traj_step]
            gap_closed_per_seed = per_seed_gap
            gap_closed_mean = sum(per_seed_gap) / len(per_seed_gap)
            gap_closed_std = (
                statistics.pstdev(per_seed_gap) if len(per_seed_gap) > 1 else 0.0
            )
            iter0_gap_closed_mean = sum(iter0_gaps) / len(iter0_gaps)
            min_iter_gap_closed_mean = sum(min_gaps) / len(min_gaps)
    else:
        trailing5_team_mean = float("nan")
        gap_closed_minspec_legacy_mean = float("nan")
        if scored:
            # Preserve the historical no-data encoding for scored scenarios
            # (NaN -> null via _safe_json_dump, empty lists).
            gap_closed_mean = float("nan")
            gap_closed_std = 0.0
            gap_closed_per_seed = []
            iter0_gap_closed_mean = float("nan")
            min_iter_gap_closed_mean = float("nan")
            mean_traj_gc = []

    # Scenario-scale uplift over the uniform-random baseline (per-step). This
    # is the honest headline for degenerate-reference scenarios (rest_trap).
    uplift_mean: Optional[float] = None
    uplift_std: Optional[float] = None
    uplift_per_seed: Optional[list[float]] = None
    if refs["random"] is not None and completed:
        uplift_per_seed = [t - refs["random"] for t in trailing5_teams]
        uplift_mean = sum(uplift_per_seed) / len(uplift_per_seed)
        uplift_std = (
            statistics.pstdev(uplift_per_seed) if len(uplift_per_seed) > 1 else 0.0
        )

    # Trap-escape verdict (issue #436): only on the degenerate-reference
    # path with completed seeds and a known random baseline.
    trap_verdict: Optional[str] = None
    trap_verdict_reason: Optional[str] = None
    trap_anchors: Optional[dict] = None
    trailing5_team_ci95: Optional[list[float]] = None
    if (
        completed
        and refs["source"] == "degenerate_reference"
        and refs["random"] is not None
    ):
        trap_verdict, trap_verdict_reason, trap_details = classify_trap_verdict(
            trailing5_teams,
            random_baseline=refs["random"],
            ne_per_step_bound=refs.get("ne_per_step_bound"),
            scripted_best=refs.get("scripted_best"),
        )
        sb = refs.get("scripted_best")
        trap_anchors = {
            **trap_details["anchors"],
            "scripted_best_value": float(sb["value"]) if sb else None,
            "scripted_best_kind": sb.get("kind") if sb else None,
        }
        trailing5_team_ci95 = trap_details["trailing5_ci95"]

    if not completed:
        verdict, reason = "no_data", "no seeds produced metrics.json"
    elif scored:
        verdict, reason = _verdict_for(gap_closed_mean)
    elif refs["source"] == "degenerate_reference":
        verdict = "not_scored_degenerate_reference"
        reason = (
            f"reference pair degenerate ({refs['reason']}): fraction ladder "
            f"not applicable; uplift_over_random = "
            f"{uplift_mean:+.3f}/step vs random = {refs['random']:.2f}/step"
        )
        if trap_verdict is not None:
            reason += f"; trap_verdict = {trap_verdict} ({trap_verdict_reason})"
    else:
        verdict = "not_scored"
        reason = (
            f"scenario {scenario!r} has no entry in SCENARIO_GAP_REFERENCES; "
            "fraction ladder not applicable (no MINSPEC fallback, #434)"
        )

    return {
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "trainer": trainer,
        "scenario": scenario,
        "seeds": list(seeds),
        "num_iterations": num_iterations,
        "n_seeds_completed": len(completed),
        "n_seeds_failed": len(failed),
        "failed_seeds": failed,
        "gap_closed_mean": gap_closed_mean,
        "gap_closed_std": gap_closed_std,
        "gap_closed_per_seed": gap_closed_per_seed,
        "gap_source": refs["source"],
        "scenario_random_baseline": refs["random"],
        "scenario_reference": {
            "value": refs["reference"],
            "kind": refs["reference_kind"],
            "reason": refs["reason"],
            "provenance": refs["provenance"],
        },
        "uplift_over_random_mean": uplift_mean,
        "uplift_over_random_std": uplift_std,
        "uplift_over_random_per_seed": uplift_per_seed,
        "trap_verdict": trap_verdict,
        "trap_verdict_reason": trap_verdict_reason,
        "trap_anchors": trap_anchors,
        "trailing5_team_ci95": trailing5_team_ci95,
        "gap_closed_minspec_legacy_mean": gap_closed_minspec_legacy_mean,
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
    summarize_only: bool = False,
) -> dict:
    """Run a full Tier-1 cell (trainer × scenario × seeds) and write
    ``cell_summary.json``. Returns the summary dict.

    All subprocesses run with ``cwd=cwd`` (default: repo root) so that
    ``-m experiments.p3_specialization.train`` resolves.

    With ``summarize_only=True`` (issue #434) no training subprocess is
    dispatched and the disk-space precheck is skipped: ``cell_summary.json``
    is rebuilt from the existing ``seed_*/metrics.json`` files via the same
    :func:`build_cell_summary`. The training-run provenance fields
    (``command_invoked``, ``git_sha``, ``wall_clock_seconds``) are carried
    forward from a pre-existing ``cell_summary.json`` when present, so a
    recompute is an additive schema refresh rather than a provenance rewrite.
    """
    if trainer not in TRAINERS:
        raise SystemExit(f"unknown trainer: {trainer!r}. Known: {sorted(TRAINERS)}")

    cell_dir = output_root / f"{trainer}_{scenario}"
    cell_dir.mkdir(parents=True, exist_ok=True)

    if summarize_only:
        seed_dirs = [cell_dir / f"seed_{s}" for s in seeds]
        command_invoked = f"summarize-only recompute (schema v{SUMMARY_SCHEMA_VERSION})"
        git_sha = _git_sha(cwd)
        wall_clock = 0.0
        prior_path = cell_dir / "cell_summary.json"
        if prior_path.exists():
            try:
                prior = json.loads(prior_path.read_text())
                command_invoked = prior.get("command_invoked", command_invoked)
                git_sha = prior.get("git_sha", git_sha)
                wall_clock = prior.get("wall_clock_seconds", wall_clock)
            except (OSError, json.JSONDecodeError):
                pass  # fall back to the recompute provenance defaults

        summary = build_cell_summary(
            trainer=trainer,
            scenario=scenario,
            seeds=seeds,
            seed_dirs=seed_dirs,
            num_iterations=num_iterations,
            command_invoked=command_invoked,
            git_sha=git_sha,
            wall_clock_seconds=wall_clock,
        )
        out_path = cell_dir / "cell_summary.json"
        out_path.write_text(_safe_json_dump(summary))
        print(
            f"\n== cell {trainer}_{scenario} re-summarized (no training): "
            f"verdict={summary['verdict_tier']} "
            f"gap_closed_mean={summary['gap_closed_mean']!r} -> {out_path}",
            flush=True,
        )
        return summary

    if not skip_precheck:
        _disk_precheck(cell_dir)

    seed_dirs = []
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
    # ``--trainer`` is required for the normal dispatch path, but the
    # ``--list-trainers`` introspection short-circuit (used by operator-launch
    # shell wrappers to source the trainer set without duplicating it) does
    # not need it. We therefore mark ``--trainer`` as not-required at the
    # argparse layer and enforce its presence below, after the short-circuit.
    p.add_argument(
        "--trainer",
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
    p.add_argument(
        "--summarize-only",
        action="store_true",
        help=(
            "Do not dispatch any training: rebuild cell_summary.json from "
            "the existing seed_*/metrics.json files via build_cell_summary "
            "(issue #434 recompute mode). Also skips the disk-space "
            "precheck. Training provenance fields are carried forward from "
            "a pre-existing cell_summary.json when present."
        ),
    )
    p.add_argument(
        "--list-trainers",
        action="store_true",
        help=(
            "Print the known trainer names (one per line) to stdout and exit. "
            "Used by the operator-launch shell wrappers (e.g. "
            "launch_tier1_sweep.sh) to validate operator-supplied trainer "
            "names against this dispatch table without duplicating it."
        ),
    )
    args = p.parse_args(argv)

    # Introspection short-circuit: print the known trainer names and exit.
    # Consumed by the operator-launch shell wrappers so they can fail-fast on
    # a typo against the canonical TRAINERS dict instead of a hand-maintained
    # duplicate (issue #405).
    if args.list_trainers:
        print("\n".join(sorted(TRAINERS)))
        return 0

    if args.trainer is None:
        p.error("the following arguments are required: --trainer")

    summary = run_cell(
        trainer=args.trainer,
        scenario=args.scenario,
        seeds=args.seeds,
        num_iterations=args.num_iterations,
        rollout_steps=args.rollout_steps,
        output_root=args.output_root,
        skip_precheck=args.skip_precheck,
        summarize_only=args.summarize_only,
    )
    # Exit non-zero if we got no seeds completed; otherwise success even if
    # partial (so a remote sweep harness can still aggregate).
    return 0 if summary["n_seeds_completed"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
