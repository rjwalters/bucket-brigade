"""Unit tests for ``softmax_packed`` in the pairwise action-KL diagnostic.

Tests verify that the joint-distribution helper works for both:
- Pre-#236 2-head action space (10 houses x 2 modes = 20-class joint)
- Post-#236 3-head action space (10 houses x 2 modes x 2 signals = 40-class joint)

Regression guard for issue #239: the post-#236 substrate exposes the signal as
a real action head; the KL diagnostic must include it in the joint or it will
silently understate per-agent divergence on the new game.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")  # skip module when RL extras absent (issue #484)

# Load ``softmax_packed`` via importlib so we don't trigger the diagnostic
# module's transitive imports (JointPPOTrainer -> bucket_brigade_core). Those
# require the Rust extension to be built, which is unrelated to the helper
# we're testing here.
_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "p3_specialization"
    / "diagnostics"
    / "pairwise_action_kl.py"
)


def _load_softmax_packed():
    """Load just ``softmax_packed`` from the diagnostic source without imports.

    We compile the source and pull out only the function we need. This avoids
    importing the rest of the module (which pulls in the Rust core extension).
    """
    import ast

    source = _MODULE_PATH.read_text()
    tree = ast.parse(source)
    func_node = next(
        (
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name == "softmax_packed"
        ),
        None,
    )
    if func_node is None:
        raise RuntimeError("softmax_packed not found in source")
    module = ast.Module(body=[func_node], type_ignores=[])
    code = compile(module, str(_MODULE_PATH), "exec")
    namespace: dict = {"torch": torch}
    exec(code, namespace)
    return namespace["softmax_packed"]


softmax_packed = _load_softmax_packed()


def test_softmax_packed_two_heads_pre236() -> None:
    """Two-head case yields the pre-#236 20-class joint with correct values."""
    torch.manual_seed(0)
    B = 5
    logits_house = torch.randn(B, 10)
    logits_mode = torch.randn(B, 2)
    joint = softmax_packed([logits_house, logits_mode])
    assert joint.shape == (B, 20), f"expected (B,20), got {joint.shape}"
    # Each row must be a probability distribution.
    sums = joint.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    # Verify outer-product structure: p[h,m] == p_h * p_m, with flattening
    # consistent with the original implementation (reshape(B, -1) on the
    # outer product of [B, H, 1] * [B, 1, M]).
    p_house = torch.softmax(logits_house, dim=-1)
    p_mode = torch.softmax(logits_mode, dim=-1)
    expected = (p_house.unsqueeze(-1) * p_mode.unsqueeze(-2)).reshape(B, -1)
    assert torch.allclose(joint, expected, atol=1e-6)


def test_softmax_packed_three_heads_post236() -> None:
    """Three-head case yields the post-#236 40-class joint with correct values."""
    torch.manual_seed(1)
    B = 7
    logits_house = torch.randn(B, 10)
    logits_mode = torch.randn(B, 2)
    logits_signal = torch.randn(B, 2)
    joint = softmax_packed([logits_house, logits_mode, logits_signal])
    assert joint.shape == (B, 40), f"expected (B,40), got {joint.shape}"
    # Each row must be a probability distribution.
    sums = joint.sum(dim=-1)
    assert torch.allclose(sums, torch.ones(B), atol=1e-5)

    # Cross-check against an explicit 3D outer product.
    p_h = torch.softmax(logits_house, dim=-1)
    p_m = torch.softmax(logits_mode, dim=-1)
    p_s = torch.softmax(logits_signal, dim=-1)
    expected_3d = (
        p_h.unsqueeze(-1).unsqueeze(-1)
        * p_m.unsqueeze(1).unsqueeze(-1)
        * p_s.unsqueeze(1).unsqueeze(1)
    )  # [B, 10, 2, 2]
    assert expected_3d.shape == (B, 10, 2, 2)
    expected = expected_3d.reshape(B, -1)
    assert torch.allclose(joint, expected, atol=1e-6)


def test_softmax_packed_kl_includes_signal_dim() -> None:
    """KL between two agents that differ only on the signal head must be > 0.

    This is the regression case: the original 2-head implementation would
    collapse identical house/mode logits to KL == 0 even if the signal
    distributions disagreed.
    """
    torch.manual_seed(2)
    B = 16
    # Both agents share house/mode logits.
    shared_house = torch.randn(B, 10)
    shared_mode = torch.randn(B, 2)
    # But disagree on signal.
    sig_a = torch.tensor([[2.0, -2.0]] * B)
    sig_b = torch.tensor([[-2.0, 2.0]] * B)

    joint_a = softmax_packed([shared_house, shared_mode, sig_a]).numpy()
    joint_b = softmax_packed([shared_house, shared_mode, sig_b]).numpy()

    eps = 1e-10
    pa = joint_a + eps
    pa = pa / pa.sum(axis=-1, keepdims=True)
    pb = joint_b + eps
    pb = pb / pb.sum(axis=-1, keepdims=True)
    kl_ab = float((pa * np.log(pa / pb)).sum(axis=-1).mean())
    assert kl_ab > 0.5, (
        f"signal-only divergence should produce substantial KL; got {kl_ab}. "
        "If this is ~0, softmax_packed is ignoring the signal head."
    )


def test_softmax_packed_single_head_degenerate() -> None:
    """Single-head input returns the bare softmax (defensive against edge cases)."""
    B = 3
    logits = torch.randn(B, 5)
    joint = softmax_packed([logits])
    assert joint.shape == (B, 5)
    assert torch.allclose(joint, torch.softmax(logits, dim=-1), atol=1e-6)


def test_softmax_packed_empty_raises() -> None:
    """Empty logits list raises a clear error rather than returning silently."""
    import pytest

    with pytest.raises(ValueError):
        softmax_packed([])
