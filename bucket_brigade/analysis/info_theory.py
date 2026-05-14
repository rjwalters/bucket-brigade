"""Plug-in entropy and mutual-information estimators with Miller-Madow correction.

These estimators are the measurement layer for the Slepian-Wolf MARL protocol
(see ``slepian-wolf-marl/paper/slepian-wolf-marl.4/paper.tex`` Section 7). They
report point estimates of discrete information quantities and bootstrap CIs;
they do **not** carry gradients. The conditional-MI regularizer used at training
time is a separate object (see ``bucket_brigade.training.joint_trainer``).

Sample format:
    Each variable is a 1-D ``numpy.ndarray`` of length ``N`` whose entries are
    hashable discrete values (integers, tuples, strings). Joint variables are
    built internally by row-wise tupling. The caller is responsible for any
    quantization of continuous quantities (see :func:`quantize_uniform`).

Bias correction:
    Plug-in entropy of a finite sample is negatively biased; the Miller-Madow
    correction ``+(K - 1) / (2 * N * ln(2))`` (in bits) removes the leading bias
    term, where ``K`` is the number of observed categories. We apply MM by
    default to every primitive entropy, so derived quantities (conditional
    entropy, MI, CMI) are corrected as differences of MM-corrected entropies.
    For very small samples consider Panzeri-Treves or NSB instead.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Callable, Sequence

import numpy as np

__all__ = [
    "entropy_discrete",
    "joint_entropy",
    "conditional_entropy",
    "mutual_information",
    "conditional_mutual_information",
    "bootstrap_ci",
    "quantize_uniform",
    "conditioner_diagnostics",
    "is_degenerate_conditioner",
]


def _as_tuples(*arrays: np.ndarray) -> np.ndarray:
    """Stack 1-D arrays into a 1-D array of tuples (one row per joint sample).

    Returns a plain object array; suitable for Counter and unique-counting.
    """
    if len(arrays) == 0:
        raise ValueError("at least one array required")
    n = len(arrays[0])
    for a in arrays[1:]:
        if len(a) != n:
            raise ValueError(
                f"all arrays must have the same length; got {[len(a) for a in arrays]}"
            )
    if len(arrays) == 1:
        return np.asarray(arrays[0])
    # Use tuples so any hashable scalar type works (int, str, etc.).
    # Assigning into a pre-allocated object array is required: passing a list
    # of equal-length tuples to ``np.array(..., dtype=object)`` gets
    # reinterpreted as a 2-D array and the inner tuples become ndarrays.
    result = np.empty(n, dtype=object)
    for i, row in enumerate(zip(*arrays)):
        result[i] = tuple(int(v) if isinstance(v, np.integer) else v for v in row)
    return result


def entropy_discrete(
    samples: np.ndarray,
    bias_correction: str = "miller-madow",
) -> float:
    """Estimate the entropy of a discrete distribution from samples, in bits.

    Args:
        samples: 1-D array of hashable discrete values.
        bias_correction:
            ``"miller-madow"`` adds ``(K - 1) / (2 N ln 2)`` bits to the plug-in
            estimate, where ``K`` is the number of observed categories.
            ``"none"`` returns the raw plug-in estimate.

    Returns:
        Entropy estimate in bits. Returns 0.0 for empty samples.
    """
    samples = _as_tuples(samples)
    n = len(samples)
    if n == 0:
        return 0.0

    # For object arrays (tuples) iterate directly to preserve hashability;
    # numpy's tolist() converts tuples to lists which break Counter.
    if samples.dtype == object:
        counts = Counter(samples)
    else:
        counts = Counter(samples.tolist())
    k = len(counts)

    # Plug-in entropy (in bits).
    h_plugin = 0.0
    for c in counts.values():
        p = c / n
        if p > 0:
            h_plugin -= p * math.log2(p)

    if bias_correction == "miller-madow":
        # The MM correction is (K-1)/(2N) in nats; convert to bits.
        return h_plugin + (k - 1) / (2.0 * n * math.log(2.0))
    if bias_correction == "none":
        return h_plugin
    raise ValueError(f"unknown bias_correction: {bias_correction!r}")


def joint_entropy(
    *arrays: np.ndarray,
    bias_correction: str = "miller-madow",
) -> float:
    """Joint entropy ``H(X_1, ..., X_k)`` of several discrete variables, in bits."""
    return entropy_discrete(_as_tuples(*arrays), bias_correction=bias_correction)


def conditional_entropy(
    x: np.ndarray,
    y: np.ndarray,
    bias_correction: str = "miller-madow",
) -> float:
    """Conditional entropy ``H(X | Y) = H(X, Y) - H(Y)`` in bits."""
    h_xy = joint_entropy(x, y, bias_correction=bias_correction)
    h_y = entropy_discrete(y, bias_correction=bias_correction)
    # Differences of MM-corrected entropies may dip slightly negative due to
    # the correction term; floor at 0.
    return max(0.0, h_xy - h_y)


def mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bias_correction: str = "miller-madow",
) -> float:
    """Mutual information ``I(X; Y) = H(X) + H(Y) - H(X, Y)`` in bits.

    Floored at 0 (MI is non-negative; small samples can produce slightly
    negative MM-corrected estimates).
    """
    h_x = entropy_discrete(x, bias_correction=bias_correction)
    h_y = entropy_discrete(y, bias_correction=bias_correction)
    h_xy = joint_entropy(x, y, bias_correction=bias_correction)
    return max(0.0, h_x + h_y - h_xy)


def conditional_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bias_correction: str = "miller-madow",
) -> float:
    """Conditional MI ``I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(X, Y, Z) - H(Z)`` in bits.

    Floored at 0.

    This is the central quantity used by the paper's redundancy penalty
    (``Section~\\ref{sec:algorithm}``): for two agents' encoder outputs
    :math:`\\hat{Z}_i, \\hat{Z}_j` and a shared state ``S``, ``I(Ẑ_i; Ẑ_j | S)``
    measures redundancy beyond what the state itself induces.
    """
    h_xz = joint_entropy(x, z, bias_correction=bias_correction)
    h_yz = joint_entropy(y, z, bias_correction=bias_correction)
    h_xyz = joint_entropy(x, y, z, bias_correction=bias_correction)
    h_z = entropy_discrete(z, bias_correction=bias_correction)
    return max(0.0, h_xz + h_yz - h_xyz - h_z)


def bootstrap_ci(
    estimator: Callable[..., float],
    samples: Sequence[np.ndarray],
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: np.random.Generator | None = None,
) -> tuple[float, float, float]:
    """Bootstrap CI for an information-theoretic estimator.

    Args:
        estimator: Function taking the unpacked sample arrays and returning a
            float (e.g., :func:`entropy_discrete`, :func:`mutual_information`).
        samples: Tuple/list of equal-length 1-D arrays passed to ``estimator``.
        n_boot: Number of bootstrap resamples.
        confidence: Coverage level for the CI (default 0.95).
        rng: Optional NumPy random Generator for reproducibility.

    Returns:
        ``(point_estimate, lower, upper)``: the estimate on the full sample
        and the bootstrap quantile CI.

    Notes:
        Resamples with replacement at the sample level. For trajectory data
        where i.i.d. is violated, the caller should bootstrap by *episode*
        rather than by step --- compose with an episode-aware sampler before
        passing in here.
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(samples[0])
    for a in samples:
        if len(a) != n:
            raise ValueError("all sample arrays must have the same length")

    point = estimator(*samples)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[b] = estimator(*(a[idx] for a in samples))

    alpha = (1.0 - confidence) / 2.0
    lower = float(np.quantile(boots, alpha))
    upper = float(np.quantile(boots, 1.0 - alpha))
    return point, lower, upper


def quantize_uniform(
    values: np.ndarray,
    n_bins: int = 16,
    feature_range: tuple[float, float] | None = None,
) -> np.ndarray:
    """Quantize continuous values into ``n_bins`` equal-width bins.

    Args:
        values: ``(N,)`` or ``(N, D)`` array of continuous values.
        n_bins: Number of bins per dimension.
        feature_range: Optional ``(min, max)`` range. If ``None``, uses the
            empirical min/max from ``values``.

    Returns:
        Integer codes:
            - 1-D input → 1-D ``(N,)`` array of bin indices in ``[0, n_bins)``.
            - 2-D input → 1-D ``(N,)`` array of integer codes packing
              per-dimension bin indices into a single discrete label (handy for
              joint entropy on vector encoders).

    This is a coarse helper for the encoder-output case in P3 — the encoder
    output ``Ẑ_i`` is a continuous vector, and the plug-in estimator above
    needs a discrete label. For ``hidden_size`` ≈ 64 and ``n_bins`` = 16, the
    raw bin tuple would have ``16^64`` possible values, so we pack by first
    PCA-projecting to a small number of components externally if needed.
    The 2-D path here is fine for small ``D`` (≲ 4); for larger ``D`` consider
    KMeans or VQ.
    """
    values = np.asarray(values)
    if values.ndim == 1:
        v_min, v_max = (
            feature_range if feature_range is not None else (values.min(), values.max())
        )
        if v_max <= v_min:
            return np.zeros(len(values), dtype=np.int64)
        edges = np.linspace(v_min, v_max, n_bins + 1)
        # ``np.digitize`` returns 1..n_bins; map to 0..n_bins-1 and clip.
        bins = np.clip(np.digitize(values, edges[1:-1]), 0, n_bins - 1)
        return bins.astype(np.int64)

    if values.ndim != 2:
        raise ValueError(f"values must be 1-D or 2-D, got shape {values.shape}")

    # 2-D path: per-column quantize, then pack into a base-n_bins integer.
    n, d = values.shape
    codes = np.zeros(n, dtype=np.int64)
    multiplier = 1
    for col in range(d):
        per_col = quantize_uniform(values[:, col], n_bins=n_bins, feature_range=None)
        codes += per_col.astype(np.int64) * multiplier
        multiplier *= n_bins
    return codes


def conditioner_diagnostics(z: np.ndarray) -> dict:
    """Summarise the empirical distribution of a CMI conditioning variable.

    Many conditional-MI failure modes (see issue #146) reduce to "the
    conditioner is effectively constant on the sample." When the support of
    ``Z`` collapses to a single value, ``I(X; Y | Z)`` is mathematically equal
    to ``I(X; Y)`` and the conditional quantity adds no information beyond the
    unconditional one. This helper returns a handful of numbers a caller can
    use to gate a warning at measurement time.

    Args:
        z: ``(N,)`` array of discrete (or hashable) conditioning codes — the
            same shape/format ``conditional_mutual_information`` consumes.

    Returns:
        ``{
            "n_samples": N,
            "n_distinct": number of distinct values observed,
            "modal_fraction": fraction of mass concentrated in the most common
                bin (1.0 means constant),
            "entropy_bits": plug-in entropy in bits (no bias correction here —
                this is a *diagnostic*, not an estimator),
        }``

        For the empty input ``N == 0`` we return zeros for everything and
        ``modal_fraction = 1.0`` so the "degenerate" check still fires.
    """
    z = np.asarray(z)
    n = int(z.size)
    if n == 0:
        return {
            "n_samples": 0,
            "n_distinct": 0,
            "modal_fraction": 1.0,
            "entropy_bits": 0.0,
        }

    counts = Counter(z.ravel().tolist())
    n_distinct = len(counts)
    modal_count = max(counts.values())
    modal_fraction = modal_count / n

    if n_distinct <= 1:
        entropy_bits = 0.0
    else:
        probs = np.array(list(counts.values()), dtype=np.float64) / n
        # Drop zeros defensively (Counter values are always > 0, but be safe).
        probs = probs[probs > 0]
        entropy_bits = float(-np.sum(probs * np.log2(probs)))

    return {
        "n_samples": n,
        "n_distinct": n_distinct,
        "modal_fraction": float(modal_fraction),
        "entropy_bits": entropy_bits,
    }


def is_degenerate_conditioner(
    z: np.ndarray,
    *,
    min_distinct: int = 2,
    max_modal_fraction: float = 0.99,
) -> tuple[bool, dict]:
    """Detect when a CMI conditioner is mathematically guaranteed to be vacuous.

    A conditioner ``Z`` is "degenerate" for the purposes of ``I(X; Y | Z)`` if
    it is effectively constant on the sample. In that case the plug-in
    estimator collapses to ``I(X; Y)`` and the conditional quantity carries no
    additional information beyond the unconditional one. See issue #146 for
    the motivating failure mode (team-reward conditioning on
    ``trivial_cooperation``, where the team reward is essentially constant).

    This is a *measurement-time* check. It detects the problem; it does not
    fix it. The fix (choosing a different conditioner) is a research call and
    lives in the experiment design, not here.

    Args:
        z: Conditioning codes (1-D array of hashable values).
        min_distinct: Minimum number of distinct values required to *not* be
            flagged degenerate. Defaults to 2 (a literal single-value
            conditioner is unambiguously degenerate).
        max_modal_fraction: If a single value carries more than this fraction
            of the sample mass, the conditioner is flagged degenerate even if
            other values appear. Defaults to 0.99 (gives near-constant
            conditioners — e.g., a near-deterministic reward — a fail).

    Returns:
        ``(is_degenerate, diagnostics)``: a boolean and the dict returned by
        :func:`conditioner_diagnostics`.
    """
    diag = conditioner_diagnostics(z)
    degenerate = (
        diag["n_distinct"] < min_distinct or diag["modal_fraction"] > max_modal_fraction
    )
    return degenerate, diag
