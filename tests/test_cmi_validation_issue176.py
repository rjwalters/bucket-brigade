"""Tests for the issue #176 Phase 1 CMI conditioner validation harness.

Coverage:

- Verdict ladder (``informative`` / ``weak`` / ``vacuous``) on the
  ``verdict_cmi_informative`` helper.
- Refusal of 2-head (pre-#236) checkpoints in the validation entry point
  (no ``allow_legacy_2head`` opt-in is allowed in this harness).
- End-to-end smoke run against a synthetic mini-checkpoint with two
  freshly-initialized :class:`PolicyNetwork` instances saved to disk; the
  rollout is short, the network is small, and the test runs in <1s.
- JSON schema round-trip (the result dict written by ``validate`` is
  re-loadable as a :class:`ValidationResult`).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from bucket_brigade.envs.bucket_brigade_env import BucketBrigadeEnv  # noqa: E402
from bucket_brigade.envs.scenarios_generated import (  # noqa: E402
    get_scenario_by_name,
)
from bucket_brigade.training.joint_trainer import flatten_dict_obs  # noqa: E402
from bucket_brigade.training.networks import PolicyNetwork  # noqa: E402


def _scenario_obs_dim(scenario_name: str, num_agents: int) -> int:
    """Discover the real flattened-obs dim by resetting the scenario."""
    scenario = get_scenario_by_name(scenario_name, num_agents=num_agents)
    env = BucketBrigadeEnv(scenario=scenario)
    obs = env.reset(seed=0)
    return flatten_dict_obs(obs, agent_id=0, num_agents=num_agents).shape[0]


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = (
    REPO_ROOT / "experiments" / "p3_specialization" / "validate_cmi_conditioner.py"
)


def _load_validation_module():
    """Import validate_cmi_conditioner.py as a standalone module."""
    spec = importlib.util.spec_from_file_location(
        "validate_cmi_conditioner_issue176", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["validate_cmi_conditioner_issue176"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def vmod():
    return _load_validation_module()


# ---------------------------------------------------------------------------
# Verdict ladder
# ---------------------------------------------------------------------------


def test_verdict_informative_when_cmi_far_below_mi(vmod):
    """CMI < 0.5 * MI -> informative."""
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=0.4) == "informative"
    # Boundary case: ratio == 0.5 should be 'weak', not 'informative'.
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=0.5) == "weak"


def test_verdict_weak_when_cmi_between_half_and_ninety_percent(vmod):
    """0.5 * MI <= CMI < 0.9 * MI -> weak."""
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=0.6) == "weak"
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=0.89) == "weak"


def test_verdict_vacuous_when_cmi_tracks_mi(vmod):
    """CMI >= 0.9 * MI -> vacuous."""
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=0.9) == "vacuous"
    assert vmod.verdict_cmi_informative(mi=1.0, cmi=1.0) == "vacuous"


def test_verdict_vacuous_when_mi_is_zero(vmod):
    """No MI -> conditioner has nothing to remove -> vacuous."""
    assert vmod.verdict_cmi_informative(mi=0.0, cmi=0.0) == "vacuous"


# ---------------------------------------------------------------------------
# Refusal of 2-head (pre-#236) checkpoints
# ---------------------------------------------------------------------------


def _make_dummy_obs(num_agents: int = 4, num_houses: int = 10) -> dict:
    return {
        "houses": np.zeros(num_houses, dtype=np.int8),
        "signals": np.zeros(num_agents, dtype=np.int8),
        "locations": np.zeros(num_agents, dtype=np.int8),
        "last_actions": np.zeros((num_agents, 2), dtype=np.int8),
        "scenario_info": np.zeros(10, dtype=np.float32),
    }


def _save_fake_checkpoint(
    out_dir: Path,
    num_agents: int,
    obs_dim: int,
    action_dims: list,
    hidden_size: int = 16,
    seed: int = 0,
) -> Path:
    """Materialize per-agent state-dict files in ``out_dir``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    for i in range(num_agents):
        net = PolicyNetwork(
            obs_dim=obs_dim, action_dims=action_dims, hidden_size=hidden_size
        )
        torch.save(net.state_dict(), out_dir / f"agent_{i}.pt")
    return out_dir


def test_load_refuses_legacy_2head_checkpoint(tmp_path, vmod):
    """A 2-head checkpoint in the dir must raise without the legacy opt-in."""
    num_agents = 4
    obs = _make_dummy_obs(num_agents=num_agents)
    obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=num_agents).shape[0]
    ckpt_dir = tmp_path / "legacy_2head"
    _save_fake_checkpoint(
        ckpt_dir,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dims=[10, 2],  # Pre-#236 layout.
    )
    # ``_load_archetypes`` wraps the per-agent constructor, which itself
    # raises for 2-head when ``allow_legacy_2head`` is False (the default
    # in our harness).
    with pytest.raises(ValueError, match="allow_legacy_2head"):
        vmod._load_archetypes(ckpt_dir, num_agents=num_agents)


def test_load_accepts_modern_3head_checkpoint(tmp_path, vmod):
    """A modern 3-head checkpoint must load cleanly without warnings."""
    num_agents = 4
    obs = _make_dummy_obs(num_agents=num_agents)
    obs_dim = flatten_dict_obs(obs, agent_id=0, num_agents=num_agents).shape[0]
    ckpt_dir = tmp_path / "modern_3head"
    _save_fake_checkpoint(
        ckpt_dir,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dims=[10, 2, 2],
    )
    archetypes = vmod._load_archetypes(ckpt_dir, num_agents=num_agents)
    assert len(archetypes) == num_agents
    assert archetypes[0].action_dims == [10, 2, 2]


def test_load_missing_dir_emits_helpful_error(tmp_path, vmod):
    """Missing checkpoint dir must surface the expected absolute path."""
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError) as excinfo:
        vmod._load_archetypes(missing, num_agents=4)
    msg = str(excinfo.value)
    assert "does_not_exist" in msg
    # Helpful error names the canonical remote-rsync recipe.
    assert "rsync" in msg or "COMPUTE_HOST" in msg


# ---------------------------------------------------------------------------
# End-to-end smoke run + JSON schema round-trip
# ---------------------------------------------------------------------------


def test_validate_end_to_end_smoke(tmp_path, vmod):
    """A short rollout against a synthetic checkpoint produces a result.

    The result need not be ``informative`` --- on a freshly-initialized
    policy with no learning signal, the CMI is likely to track MI closely
    (verdict 'vacuous' or 'weak'). The test only asserts that the harness
    runs end-to-end and emits a well-formed result.
    """
    num_agents = 4
    # The checkpoint must be sized for the real ``minimal_specialization``
    # flattened obs dim (post-#252 ``round1_signals``, post-#283 wider
    # ``scenario_info``); a synthetic dummy obs shape would mismatch the
    # env's actual obs and raise inside the rollout.
    obs_dim = _scenario_obs_dim("minimal_specialization", num_agents)
    ckpt_dir = tmp_path / "synth"
    _save_fake_checkpoint(
        ckpt_dir,
        num_agents=num_agents,
        obs_dim=obs_dim,
        action_dims=[10, 2, 2],
        hidden_size=16,
    )

    out_dir = tmp_path / "out"
    result = vmod.validate(
        checkpoint_dir=ckpt_dir,
        output_dir=out_dir,
        scenario="minimal_specialization",
        num_agents=num_agents,
        rollout_steps=128,  # Tiny so the test is fast.
        seed=42,
        n_bins=4,
        mi_proj_dims=3,
    )

    # Result has the documented shape.
    assert result.verdict in {"informative", "weak", "vacuous"}
    assert result.mi_mean_pair >= 0.0
    assert result.cmi_mean_pair >= 0.0
    assert result.num_agents == num_agents
    assert result.rollout_steps == 128
    assert result.action_dims == [10, 2, 2]
    assert isinstance(result.is_degenerate_conditioner, bool)
    # Per-pair entries cover all i<j pairs.
    expected_keys = {
        f"agent_{i}_{j}" for i in range(num_agents) for j in range(i + 1, num_agents)
    }
    assert set(result.mi_per_pair.keys()) == expected_keys
    assert set(result.cmi_per_pair.keys()) == expected_keys

    # On-disk artifacts exist.
    assert (out_dir / "results.json").exists()
    assert (out_dir / "verdict.md").exists()

    # JSON schema round-trip.
    with (out_dir / "results.json").open() as f:
        round_tripped = json.load(f)
    assert round_tripped["verdict"] == result.verdict
    assert round_tripped["mi_mean_pair"] == pytest.approx(result.mi_mean_pair)
    assert round_tripped["cmi_mean_pair"] == pytest.approx(result.cmi_mean_pair)
    assert round_tripped["checkpoint_hash"] == result.checkpoint_hash
    # All required top-level schema keys present.
    required_keys = {
        "checkpoint_dir",
        "checkpoint_hash",
        "scenario",
        "num_agents",
        "rollout_steps",
        "seed",
        "n_bins",
        "mi_proj_dims",
        "hidden_size",
        "obs_dim",
        "action_dims",
        "mi_mean_pair",
        "cmi_mean_pair",
        "cmi_over_mi_ratio",
        "is_degenerate_conditioner",
        "conditioner_n_distinct",
        "conditioner_modal_fraction",
        "conditioner_entropy_bits",
        "verdict",
        "mi_per_pair",
        "cmi_per_pair",
    }
    assert required_keys.issubset(round_tripped.keys())

    # Verdict markdown contains the expected anchor strings.
    md = (out_dir / "verdict.md").read_text()
    assert "CMI Conditioner Validation" in md
    assert result.verdict in md
    assert "issue #176" in md


def test_verdict_md_includes_next_steps_for_each_verdict(vmod):
    """The Markdown report must include a 'Next steps' section per verdict."""
    base = dict(
        checkpoint_dir="/tmp/x",
        checkpoint_hash="abc",
        scenario="minimal_specialization",
        num_agents=4,
        rollout_steps=128,
        seed=42,
        n_bins=4,
        mi_proj_dims=3,
        hidden_size=16,
        obs_dim=64,
        action_dims=[10, 2, 2],
        is_degenerate_conditioner=False,
        conditioner_n_distinct=8,
        conditioner_modal_fraction=0.3,
        conditioner_entropy_bits=2.4,
        mi_per_pair={},
        cmi_per_pair={},
    )
    for verdict, mi, cmi in [
        ("informative", 1.0, 0.3),
        ("weak", 1.0, 0.7),
        ("vacuous", 1.0, 0.95),
    ]:
        result = vmod.ValidationResult(
            mi_mean_pair=mi,
            cmi_mean_pair=cmi,
            cmi_over_mi_ratio=cmi / mi,
            verdict=verdict,
            **base,
        )
        md = vmod._format_verdict_md(result)
        assert "## Next steps" in md
        assert verdict in md
