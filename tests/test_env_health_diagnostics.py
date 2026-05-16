"""Env-health regression suite (issue #201).

Wraps the three diagnostic scripts under
``experiments/p3_specialization/diagnostics/`` (H1/H2/H3) as numerical
regression assertions. The intent is to catch silent breakage of the
reward signal when env-side changes land (e.g. team-vs-ownership rebalance
in #197, per-agent ownership vectors in #198, new scenarios in #199).

Marked ``@pytest.mark.slow`` so the CI fast lane (``-m "not slow and not
integration"``) skips it. Run locally with::

    uv run pytest tests/test_env_health_diagnostics.py --run-slow -v

Thresholds are derived from the rubric in issue #201:

H1 (reward signal not degenerate)
    For at least one agent: CV > 0.05 OR action-reward R² > 0.01. If both
    are below those bounds for every agent the gradient signal is
    effectively flat — see #190.

H2 (team and ownership rewards plausibly trainable)
    Median ``|team| / |ownership|`` ratio < 5× AND min pairwise agent-reward
    correlation < 0.95. Above those bounds independent PPO is dominated by
    a shared team signal that no single agent can move — see #183/#197.

H3 (uniform-random baseline holds the post-#197/#198 scale on ``default``)
    Mean uniform-random per-step team reward ∈ [220, 290] on the
    ``default`` scenario (current baseline ~250). Issue #145 cited 308,
    but the team-vs-ownership rebalance in #197/#198 shifted the absolute
    scale; a regression *outside* the new window means the env reward
    magnitudes have shifted again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

# Ensure ``experiments/p3_specialization/diagnostics/`` is importable so we
# can re-use the H2/H3 driver functions directly (rather than reimplementing
# them here). H1 lives in ``bucket_brigade.diagnostics`` so no path hack is
# needed for it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_DIAG_DIR = _REPO_ROOT / "experiments" / "p3_specialization" / "diagnostics"
if str(_DIAG_DIR) not in sys.path:
    sys.path.insert(0, str(_DIAG_DIR))


# ---------------------------------------------------------------------------
# Thresholds (single source of truth — keep these aligned with the rubric
# in the issue body and ``experiments/p3_specialization/diagnostics/README``).
# ---------------------------------------------------------------------------

H1_MIN_CV_OR_R2 = {
    # (cv_threshold, r2_threshold) — H1 passes if for any agent
    # cv > cv_threshold OR r2_packed > r2_threshold. The default scenario
    # has high team-share, so the per-agent CV easily clears 0.05 even
    # though R² stays low (#190 finding). On minimal_specialization the
    # ownership signal dominates so R² is the more informative side.
    "default": (0.05, 0.01),
    "minimal_specialization": (0.05, 0.01),
}

H2_MAX_TEAM_TO_OWN_RATIO = {
    # Median |team|/|ownership| ratio per scenario. Both committed JSONs
    # under ``experiments/p3_specialization/diagnostics/results/`` show
    # values well below 5x (default: 3.0, minimal_specialization: 0.12)
    # after the #197/#198 rebalance.
    "default": 5.0,
    "minimal_specialization": 5.0,
}

H2_MAX_MIN_PAIRWISE_CORR = {
    # Min off-diagonal pairwise agent-reward correlation. Above 0.95 means
    # every agent's reward signal is essentially the team term; PPO has
    # nothing to specialize on. Committed JSONs show default=0.71,
    # minimal_specialization=0.04.
    "default": 0.95,
    "minimal_specialization": 0.95,
}

H3_RANDOM_PER_STEP_RANGE_DEFAULT = (220.0, 290.0)
# Window centered on the post-#197/#198 uniform-random baseline (~250).
# Issue #145 cited 308, but the team-vs-ownership rebalance in #197 and the
# per-agent ownership vectors in #198 shifted the absolute scale downward —
# the H3 script run against current main reports per-step ~250 (250.39 mean,
# 95% CI [237, 263] with n=250 episodes). We pick ±~15% around that to
# catch order-of-magnitude regressions without coupling to seed noise.
# The cited 308 in random_baseline.py's docstring is preserved as historical
# context but is no longer the current baseline.


# ---------------------------------------------------------------------------
# Shared rollout fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def h2_rollouts():
    """One uniform-random rollout per scenario for the H2 audit.

    Computed once per test module to keep the slow suite under the issue's
    5-minute budget. Returns ``{scenario_name: summary_dict}`` where each
    summary mirrors ``audit_reward_attribution.summarize`` output.
    """
    from audit_reward_attribution import run_rollouts, summarize  # type: ignore

    out = {}
    for scenario in ("default", "minimal_specialization"):
        team, own, work, reward = run_rollouts(
            scenario_name=scenario, seeds=tuple(range(10))
        )
        out[scenario] = summarize(team, own, work, reward)
    return out


@pytest.fixture(scope="module")
def h1_random_init():
    """Hermetic H1 random-init rollout per scenario.

    Uses ``bucket_brigade.diagnostics.random_init_rollout_stats`` so the
    test does not depend on any on-disk trained cell.
    """
    torch = pytest.importorskip("torch")  # noqa: F841 (skip if no RL extras)
    from bucket_brigade.diagnostics import random_init_rollout_stats

    out = {}
    for scenario in ("default", "minimal_specialization"):
        # 1024 steps is enough to get stable CV/R² for the regression bar
        # while keeping the test fast (~1-2s per scenario).
        _R, _A, per_agent = random_init_rollout_stats(
            scenario_name=scenario, rollout_steps=1024, seed=42
        )
        out[scenario] = per_agent
    return out


# ---------------------------------------------------------------------------
# H1: reward signal is not degenerate (#190)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["default", "minimal_specialization"])
def test_h1_reward_signal_not_degenerate(h1_random_init, scenario) -> None:
    """At least one agent has CV > threshold OR action-R² > threshold.

    Inverts the #190 rubric: "CV < 0.05 AND R² < 0.01 for every agent"
    means H1 (degenerate signal) is confirmed. We want H1 to remain
    *ruled out* for both scenarios.
    """
    per_agent = h1_random_init[scenario]
    cv_thresh, r2_thresh = H1_MIN_CV_OR_R2[scenario]

    passes = []
    for stats in per_agent:
        # R² can be NaN if the agent's rewards have zero variance. Treat
        # NaN as "no signal" (False) so the degeneracy check is strict.
        r2 = stats["r2_packed"]
        r2_ok = not np.isnan(r2) and r2 > r2_thresh
        cv_ok = stats["cv"] > cv_thresh
        passes.append(cv_ok or r2_ok)

    summary = [
        f"agent_{i}: cv={s['cv']:.4f}, r2_packed={s['r2_packed']:.4f}"
        for i, s in enumerate(per_agent)
    ]
    assert any(passes), (
        f"H1 degeneracy regression on scenario={scenario}: "
        f"every agent has cv <= {cv_thresh} AND r2_packed <= {r2_thresh}. "
        f"Per-agent stats: {summary}"
    )


# ---------------------------------------------------------------------------
# H2: team and ownership rewards are plausibly trainable (#183, #197)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["default", "minimal_specialization"])
def test_h2_team_to_ownership_ratio(h2_rollouts, scenario) -> None:
    """Median |team|/|ownership| ratio must stay below the per-scenario cap.

    A regression above 5x means the team component dominates per-agent
    rewards so strongly that independent PPO has nothing to specialize
    on (failure mode tracked in #183).
    """
    summary = h2_rollouts[scenario]
    median = summary["ratios"]["team_over_ownership"]["median"]
    cap = H2_MAX_TEAM_TO_OWN_RATIO[scenario]
    assert median is not None and median < cap, (
        f"H2 regression on scenario={scenario}: "
        f"median |team|/|ownership| = {median} (cap {cap}). "
        f"Full ratio summary: {summary['ratios']['team_over_ownership']}"
    )


@pytest.mark.slow
@pytest.mark.parametrize("scenario", ["default", "minimal_specialization"])
def test_h2_min_pairwise_reward_correlation(h2_rollouts, scenario) -> None:
    """Min off-diagonal pairwise reward correlation must stay below cap.

    If every pair of agents has correlation > 0.95, the per-agent rewards
    are essentially the same scalar and PPO can't differentiate roles.
    """
    summary = h2_rollouts[scenario]
    min_corr = summary["min_off_diag_corr"]
    cap = H2_MAX_MIN_PAIRWISE_CORR[scenario]
    assert min_corr < cap, (
        f"H2 regression on scenario={scenario}: "
        f"min pairwise reward correlation = {min_corr:.4f} (cap {cap}). "
        f"Full pairwise matrix: {summary['pairwise_corr']}"
    )


# ---------------------------------------------------------------------------
# H3: uniform-random per-step baseline tracks the cited 308 (#145, #196)
# ---------------------------------------------------------------------------


def _random_baseline_per_step_default(
    *, episodes_per_seed: int, seeds: int
) -> Tuple[float, int]:
    """Re-derive the uniform-random per-step team reward on ``default``.

    Inlined uniform-random rollout — equivalent to
    ``random_baseline.run_random_episode`` but does not import the H3
    script (whose transitive imports pull in torch via the training
    package; we want this test to remain runnable in the no-RL CI install).
    """
    from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv
    from bucket_brigade.envs.scenarios_generated import get_scenario_by_name

    scenario = get_scenario_by_name("default", num_agents=4)
    # Anti-regression guard: PR #236 (issue #235) made signal a first-class
    # action dimension, widening MultiDiscrete([10, 2]) → MultiDiscrete([10, 2, 2]).
    # PR #236 missed updating this helper, so it silently kept measuring a
    # 2-dim sampling policy (a strictly different stochastic process than the
    # one exercised by the env post-#236), and the H3 verdict became
    # meaningless until issue #237 caught it.
    #
    # The env does NOT validate input action shape (a 2-dim action returns
    # successfully), so we cross-check against the canonical source-of-truth
    # in random_baseline.py. We grep the source file rather than importing,
    # because random_baseline transitively imports torch via the training
    # package and this test must remain runnable in the no-RL CI install.
    from pathlib import Path as _Path

    _rb_src = (
        _Path(__file__).resolve().parent.parent
        / "experiments"
        / "p3_specialization"
        / "diagnostics"
        / "random_baseline.py"
    ).read_text()
    assert "ACTION_DIMS = [10, 2, 2]" in _rb_src, (
        "random_baseline.ACTION_DIMS no longer equals [10, 2, 2]. "
        "Update this helper's sampling loop AND the action_dims grep above to "
        "match the new layout (otherwise H3 silently measures the wrong policy, "
        "the bug PR #236 introduced and issue #237 fixed)."
    )
    per_step = []
    for seed in range(42, 42 + seeds):
        rng = np.random.default_rng(seed)
        env = BucketBrigadeEnv(scenario=scenario)
        for _ in range(episodes_per_seed):
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            total_reward = 0.0
            while not env.done:
                # MultiDiscrete([10, 2, 2]) per agent post-#236 (issue #235).
                # Third column is the broadcast signal channel, sampled
                # independently here to fully exercise the action manifold —
                # otherwise this helper silently measures a pre-#236 policy
                # (the bug that motivated issue #237).
                actions = np.stack(
                    [
                        rng.integers(0, 10, size=env.num_agents),
                        rng.integers(0, 2, size=env.num_agents),
                        rng.integers(0, 2, size=env.num_agents),
                    ],
                    axis=-1,
                ).astype(np.int64)
                _, rewards, _, _ = env.step(actions)
                total_reward += float(rewards.sum())
            nights = int(env.night)
            per_step.append(total_reward / nights)
    arr = np.asarray(per_step)
    return float(arr.mean()), int(arr.size)


@pytest.mark.slow
def test_h3_random_baseline_default_in_range() -> None:
    """Mean uniform-random per-step team reward stays in the #145 window.

    Window is intentionally wide (~±5%): this test catches order-of-
    magnitude regressions in reward scaling, not Monte-Carlo jitter.
    """
    mean_per_step, n = _random_baseline_per_step_default(episodes_per_seed=25, seeds=2)
    lo, hi = H3_RANDOM_PER_STEP_RANGE_DEFAULT
    assert lo <= mean_per_step <= hi, (
        f"H3 regression on scenario=default: "
        f"uniform-random per-step mean = {mean_per_step:.2f} "
        f"outside [{lo}, {hi}] (n={n} episodes). "
        f"Tracks issue #145 / PR #196."
    )


# ---------------------------------------------------------------------------
# H2: spot-check that committed result JSONs still parse (anti-rot)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize(
    "filename",
    [
        "h2_reward_attribution.json",
        "h2_reward_attribution_minimal_specialization.json",
    ],
)
def test_committed_h2_jsons_have_expected_schema(filename) -> None:
    """The fixture JSONs we compare future runs against must keep schema.

    Anti-rot guard: if someone changes ``audit_reward_attribution.py``'s
    output shape, this test breaks before the regression assertions above
    silently start matching nonsense.
    """
    path = (
        _REPO_ROOT
        / "experiments"
        / "p3_specialization"
        / "diagnostics"
        / "results"
        / filename
    )
    if not path.exists():
        pytest.skip(f"committed result file not present: {path}")

    data = json.loads(path.read_text())
    required_keys = {
        "n_steps",
        "n_agents",
        "magnitudes",
        "ratios",
        "pairwise_corr",
        "mean_off_diag_corr",
        "min_off_diag_corr",
        "team_var",
        "mean_per_agent_var",
        "team_share_lower_bound_on_corr",
    }
    missing = required_keys - data.keys()
    assert not missing, f"{filename} missing keys: {missing}"
    assert "team_over_ownership" in data["ratios"]
    assert "team_over_work_cost" in data["ratios"]
