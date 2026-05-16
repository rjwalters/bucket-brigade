"""Regression tests for issue #240: heuristic-agent signal channel honors honesty_bias.

Pre-#240 the Rust-backed Nash evaluator (``payoff_evaluator_rust.py``) hardcoded
``signal == mode`` in both the Python helper ``_heuristic_action`` and the Rust
``HeuristicAgent::select_action``. This silently made the Liar archetype
(``honesty_bias = 0.1``) behave identically to a non-deceptive agent through
the Rust path, while the pure-Python ``HeuristicAgent.act`` path correctly
emitted deceptive signals. The two evaluators therefore disagreed for any
strategy with ``honesty_bias < 1.0``.

This test pins the fix: with the new logic, both paths emit deceptive signals
at a rate determined by ``honesty_bias``.
"""

import numpy as np

from bucket_brigade.agents.archetypes import (
    FIREFIGHTER_PARAMS,
    LIAR_PARAMS,
)
from bucket_brigade.equilibrium.payoff_evaluator_rust import _heuristic_action


def _signal_disagreement_rate(theta: np.ndarray, n: int = 4000, seed: int = 0) -> float:
    """Return P(signal != mode) for the Python helper given ``theta``."""
    rng = np.random.RandomState(seed)
    # Force a non-trivial environment so both work and rest paths get exercised.
    obs = {
        "houses": np.array([1, 1, 0, 1, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "signals": np.zeros(4, dtype=np.int8),
        "locations": np.zeros(4, dtype=np.int8),
    }
    disagreements = 0
    for _ in range(n):
        action = _heuristic_action(theta, obs, agent_id=0, rng=rng)
        assert len(action) == 3, "action must be [house, mode, signal] post-#235"
        if action[1] != action[2]:
            disagreements += 1
    return disagreements / n


def test_firefighter_archetype_signals_honestly():
    """honesty_bias=1.0 must produce signal == mode in every draw."""
    rate = _signal_disagreement_rate(FIREFIGHTER_PARAMS, n=2000, seed=11)
    assert rate == 0.0, f"Firefighter should never lie; got disagreement rate {rate:.4f}"


def test_liar_archetype_signals_deceptively():
    """honesty_bias=0.1 must produce signal != mode roughly 90% of the time."""
    rate = _signal_disagreement_rate(LIAR_PARAMS, n=4000, seed=22)
    # Expected ~0.90 (honesty_bias=0.1), allow ±0.04 for RNG noise on n=4000.
    assert 0.86 <= rate <= 0.94, (
        f"Liar should lie ~90% of the time; got disagreement rate {rate:.4f}"
    )


def test_python_helper_action_shape():
    """All archetypes must emit a length-3 action under #235's signaling shape."""
    rng = np.random.RandomState(7)
    obs = {
        "houses": np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=np.int8),
        "signals": np.zeros(4, dtype=np.int8),
        "locations": np.zeros(4, dtype=np.int8),
    }
    for params in (FIREFIGHTER_PARAMS, LIAR_PARAMS):
        action = _heuristic_action(params, obs, agent_id=0, rng=rng)
        assert len(action) == 3
        assert action[1] in (0, 1)
        assert action[2] in (0, 1)


def test_rust_heuristic_episode_runs_with_liar():
    """The full Rust path must execute end-to-end for the Liar archetype.

    Pre-#240 this would run but silently honest-signal. This test just
    verifies the path doesn't error and returns a finite reward — the
    aggregate behavioral verification lives in the Rust unit test
    ``test_heuristic_agent_liar_archetype_lies_most_of_time``.
    """
    import bucket_brigade_core as core
    from bucket_brigade.equilibrium.payoff_evaluator_rust import (
        _convert_scenario_to_rust,
    )
    from bucket_brigade.envs import trivial_cooperation_scenario

    scenario = trivial_cooperation_scenario(num_agents=4)
    rust_scenario = _convert_scenario_to_rust(scenario)

    reward = core.run_heuristic_episode_focal(
        rust_scenario,
        4,
        LIAR_PARAMS.tolist(),
        LIAR_PARAMS.tolist(),
        seed=12345,
    )
    assert np.isfinite(reward)
