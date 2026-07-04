"""Versioned scenario registry for Bucket Brigade (issue #369).

This module exposes a *frozen-by-ID* registry of scenarios so paper results
are reproducible by ID. Each entry maps a versioned ID (e.g.
``"minimal_specialization-v1"``) to a frozen scenario factory that returns a
:class:`~bucket_brigade.envs.scenarios_generated.Scenario`.

The registry is the **single source of truth** for both:

1. The Python-side :func:`bucket_brigade.make` Gym/Gymnasium adapter
   (see :mod:`bucket_brigade.envs.gym_adapter`).
2. Any downstream consumer that needs a reproducible scenario lookup
   (e.g. an external Thrust-side binding — see the issue #369 scope
   directive).

Version-bump policy
-------------------

**Any change to the following requires a NEW ``-vN`` ID; the old ID stays
frozen** (and the old entry MUST NOT be mutated or removed without a
deprecation cycle):

- The observation space (shape, dtype, channel ordering, scenario_info
  feature vector, identity-tail layout).
- The action space (per-agent ``[house, mode, signal]`` layout,
  ``MultiDiscrete`` dimensions).
- The reward function (per-agent or team components, sign convention,
  cost-to-work formula, distance cost).
- Any scenario parameter consumed by the underlying
  :class:`~bucket_brigade.envs.bucket_brigade_env.BucketBrigadeEnv`
  (fire dynamics, ownership vectors, ring size, num_agents, commitment
  mode, extinguish mode, suppression coefficient, etc.).
- Anything that changes the random sequence produced by ``env.reset(seed)``
  for an otherwise identical scenario, given identical actions
  (e.g. the order in which the engine calls ``self.rng``).

Things that DO NOT require a version bump:

- Internal refactors that are bit-exact for every existing seed
  (verified by ``tests/test_environment.py`` + the round-trip tests in
  ``tests/test_env_registry.py``).
- Adding NEW registry entries (``-v1`` of a new scenario name).
- Docstring fixes, type annotations, internal helper renames.

When in doubt, bump. Frozen IDs are cheap; reproducibility is not.

Naming conventions
------------------

- Scenario base name in ``snake_case``, e.g. ``minimal_specialization``,
  ``rest_trap``.
- Version suffix ``-vN`` with ``N`` starting at 1, monotonically
  increasing.
- Full ID example: ``"minimal_specialization-v1"``.

Promoting a phase-diagram cell to a named scenario (issue #435)
---------------------------------------------------------------

The NE phase diagram (#358) sweeps (β, κ, c) overrides on the
``minimal_specialization`` base via
:func:`bucket_brigade.baselines.per_cell.make_phase_diagram_scenario`.
To promote a cell to a first-class named scenario:

1. **Name**: ``asym_bBB_kKK_cCC`` where ``BB``/``KK``/``CC`` are the two
   digits after the decimal point, e.g. cell (β=0.50, κ=0.90, c=0.50)
   becomes ``asym_b05_k09_c05``. Read the (β, κ, c) values from the
   per-cell artifacts (``experiments/nash/phase_diagram/results.json``
   ``cells[*].{beta,kappa,c}`` or the ``b*_k*_c*`` cell tag), NOT from a
   rendered markdown table — the ``phase_diagram_table.md`` column order
   is ``c | β | κ``, which has already caused one transposition mishap
   (issue #435 was filed against non-existent c=0.90 cells).
2. **Definition**: add the scenario to ``definitions/scenarios.json``,
   copying the ``minimal_specialization`` parameters with ONLY
   ``prob_fire_spreads_to_neighbor`` (β),
   ``prob_solo_agent_extinguishes_fire`` (κ), and
   ``cost_to_work_one_night`` (c) overridden; then run
   ``scripts/generate_python.py`` and ``scripts/generate_typescript.py``,
   and mirror the entry in ``bucket-brigade-core/src/scenarios.rs``
   (bump its ``test_scenario_count``).
3. **Parity test**: extend the bit-parity test in
   ``tests/test_env_registry.py`` asserting the registered scenario is
   field-for-field identical to ``make_phase_diagram_scenario(β, κ, c)``
   — this is what keeps the cell's NE artifacts (payoff, convergence,
   genome files) citable for the named scenario.
4. **Freeze**: add the ``<name>-v1`` entry to :data:`SCENARIO_VERSIONS`
   below (append-only, per the version-bump policy above).
5. **Baselines**: give the scenario a canonical uniform-random baseline
   under the issue #237 convention (n=1000; see
   ``experiments/p3_specialization/diagnostics/random_baseline.py``) and
   wire it through ``SCENARIO_RANDOM_BASELINES``,
   ``SCENARIO_GAP_REFERENCES`` (reference side from the n=10k #413
   per-cell calibration in
   ``experiments/nash/phase_diagram/per_cell_baselines.json`` when the
   cell is covered), and the parity manifest
   (``bucket_brigade/baselines/parity.py``: ``REFERENCE_CI95`` +
   ``SCENARIO_FINGERPRINTS``), each with measurement provenance.

The registry intentionally ships a *small* curated set of frozen IDs (the
ones used in our own paper results + diagnostics). External users who
want every scenario name from ``definitions/scenarios.json`` can still go
through :func:`~bucket_brigade.envs.scenarios_generated.get_scenario_by_name`;
the versioned registry is the **reproducibility surface**, not an
auto-generated mirror.
"""

from __future__ import annotations

from typing import Callable, Dict, List

from .scenarios_generated import (
    Scenario,
    asym_b05_k09_c05_scenario,
    asym_b09_k09_c05_scenario,
    chain_reaction_scenario,
    default_scenario,
    deceptive_calm_scenario,
    early_containment_scenario,
    easy_scenario,
    greedy_neighbor_scenario,
    hard_scenario,
    minimal_specialization_scenario,
    mixed_motivation_scenario,
    overcrowding_scenario,
    positional_default_scenario,
    rest_trap_scenario,
    sparse_heroics_scenario,
    trivial_cooperation_scenario,
    v2_minimal_scenario,
)


__all__ = [
    "SCENARIO_VERSIONS",
    "DEFAULT_NUM_AGENTS",
    "list_versioned_scenarios",
    "get_scenario_by_id",
    "parse_scenario_id",
]


# Default ``num_agents`` for frozen scenario IDs. The Bucket Brigade env
# is parameterized by num_agents but every published result in this repo
# uses 4. Pinning this default into the registry is part of what makes
# an ID *frozen*: ``make("minimal_specialization-v1")`` always yields the
# same 4-agent scenario. Callers can still override via the ``make()``
# kwarg, but doing so produces an env that is NOT covered by the frozen
# ID's reproducibility guarantee (the override is reflected in the
# returned env's metadata for traceability).
DEFAULT_NUM_AGENTS: int = 4


# Type alias for a frozen scenario factory: takes num_agents, returns
# a fully-constructed Scenario. Each entry in SCENARIO_VERSIONS is one
# of these.
_ScenarioFactory = Callable[[int], Scenario]


# Frozen versioned scenario registry. **APPEND-ONLY** in the version
# direction (see module docstring). Each ID maps to a zero-argument-ish
# factory that produces a Scenario for the given num_agents.
#
# v1 of each ID delegates to the existing ``*_scenario`` factory in
# ``scenarios_generated`` (which is itself generated from
# ``definitions/scenarios.json``). If we ever need to change a scenario's
# parameters without breaking reproducibility, we'll add e.g.
# ``"minimal_specialization-v2"`` pointing to a NEW factory and leave
# ``-v1`` untouched.
SCENARIO_VERSIONS: Dict[str, _ScenarioFactory] = {
    # Default-family scenarios (covered by most diagnostic suites).
    "default-v1": default_scenario,
    # ``easy-v1`` added by issue #437 so every scenario with a canonical
    # random baseline (``SCENARIO_RANDOM_BASELINES``) has a frozen ID the
    # parity manifest can key on. New ``-v1`` entries never require a bump.
    "easy-v1": easy_scenario,
    "hard-v1": hard_scenario,
    # Named test scenarios from ``definitions/scenarios.json``.
    "trivial_cooperation-v1": trivial_cooperation_scenario,
    "early_containment-v1": early_containment_scenario,
    "greedy_neighbor-v1": greedy_neighbor_scenario,
    "sparse_heroics-v1": sparse_heroics_scenario,
    "rest_trap-v1": rest_trap_scenario,
    "chain_reaction-v1": chain_reaction_scenario,
    "deceptive_calm-v1": deceptive_calm_scenario,
    "overcrowding-v1": overcrowding_scenario,
    "mixed_motivation-v1": mixed_motivation_scenario,
    # P3 / specialization diagnostics (issue #199 and follow-ups).
    "minimal_specialization-v1": minimal_specialization_scenario,
    # 2-house topology for PPO learnability diagnostics (#254).
    "v2_minimal-v1": v2_minimal_scenario,
    # Positional-reward variant of default — frozen baseline for PPO
    # training (#384) and frozen-baseline release manifest (#371). See #403.
    "positional_default-v1": positional_default_scenario,
    # asymmetric_only NE phase-diagram cells promoted to named scenarios
    # (issue #435; see the "Promoting a phase-diagram cell" section in the
    # module docstring). Bit-identical to make_phase_diagram_scenario(β, κ, c)
    # on the minimal_specialization base; targeted by het_ppo Phase 2 (#429).
    "asym_b05_k09_c05-v1": asym_b05_k09_c05_scenario,
    "asym_b09_k09_c05-v1": asym_b09_k09_c05_scenario,
}


def list_versioned_scenarios() -> List[str]:
    """Return a sorted list of frozen scenario IDs.

    Returns:
        Sorted list of versioned IDs, e.g.
        ``["chain_reaction-v1", "default-v1", ...]``.

    Example:
        >>> ids = list_versioned_scenarios()
        >>> "minimal_specialization-v1" in ids
        True
    """
    return sorted(SCENARIO_VERSIONS.keys())


def parse_scenario_id(scenario_id: str) -> tuple[str, int]:
    """Split a frozen scenario ID into ``(base_name, version)``.

    Args:
        scenario_id: A versioned ID like ``"minimal_specialization-v1"``.

    Returns:
        Tuple ``(base_name, version_int)``, e.g.
        ``("minimal_specialization", 1)``.

    Raises:
        ValueError: If the ID does not match the ``<name>-v<int>`` shape.

    Example:
        >>> parse_scenario_id("minimal_specialization-v1")
        ('minimal_specialization', 1)
    """
    if "-v" not in scenario_id:
        raise ValueError(
            f"Invalid scenario ID {scenario_id!r}: expected '<name>-v<int>' "
            f"shape (e.g. 'minimal_specialization-v1')."
        )
    base, _, version_str = scenario_id.rpartition("-v")
    if not base or not version_str.isdigit():
        raise ValueError(
            f"Invalid scenario ID {scenario_id!r}: expected '<name>-v<int>' "
            f"shape with a non-empty base name and integer version."
        )
    return base, int(version_str)


def get_scenario_by_id(
    scenario_id: str, num_agents: int = DEFAULT_NUM_AGENTS
) -> Scenario:
    """Look up a scenario by its frozen versioned ID.

    Args:
        scenario_id: A versioned ID registered in
            :data:`SCENARIO_VERSIONS`, e.g. ``"minimal_specialization-v1"``.
        num_agents: Number of agents to instantiate the scenario with.
            Defaults to :data:`DEFAULT_NUM_AGENTS` (4). Overriding this
            value produces a scenario that is **not** covered by the
            frozen ID's reproducibility guarantee — callers who do so
            should record the override in their experiment metadata.

    Returns:
        A fully-constructed :class:`Scenario`.

    Raises:
        KeyError: If ``scenario_id`` is not registered. The error message
            includes the full list of available IDs.

    Example:
        >>> sc = get_scenario_by_id("minimal_specialization-v1")
        >>> sc.num_agents
        4
    """
    if scenario_id not in SCENARIO_VERSIONS:
        available = ", ".join(list_versioned_scenarios())
        raise KeyError(
            f"Unknown scenario ID {scenario_id!r}. Available IDs: {available}"
        )
    factory = SCENARIO_VERSIONS[scenario_id]
    return factory(num_agents)
