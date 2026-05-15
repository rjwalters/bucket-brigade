"""Running statistics helpers for RL training.

Currently exposes :class:`RunningMeanStd`, an online (Welford / Chan)
mean+variance estimator used by :class:`JointPPOTrainer` to optionally
normalize returns before the value-loss MSE. See issue #159 for the
motivating Phase 1 diagnostics (value loss six to seven orders of
magnitude larger than the policy term).

The implementation follows the parallel/Chan update used in OpenAI
baselines / Stable-Baselines3 ``VecNormalize`` so that a whole batch can
be folded in with one update rather than one sample at a time.
"""

from __future__ import annotations

import numpy as np


__all__ = ["RunningMeanStd"]


class RunningMeanStd:
    """Online mean / variance over a stream of batches.

    Tracks ``mean``, ``var``, and ``count`` using Chan's parallel
    variance update (numerically stable, ``O(1)`` memory). Intended for
    scalar streams (e.g., flattened PPO returns) but works for any
    fixed-shape ``ndarray`` per sample.

    Args:
        epsilon: Initial pseudo-count to avoid division by zero on the
            first ``update``. The reported ``var`` is biased toward
            ``1.0`` until enough real samples have arrived.
        shape: Shape of a single sample. ``()`` for a scalar stream.
    """

    def __init__(self, epsilon: float = 1e-4, shape: tuple = ()) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        """Fold a batch of samples into the running statistics.

        ``x`` is treated as ``(batch, *shape)``; the leading axis is
        reduced.
        """
        x = np.asarray(x, dtype=np.float64)
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        self._update_from_moments(batch_mean, batch_var, batch_count)

    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        # Chan / parallel-algorithm combine of two (mean, M2, count) blocks.
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = m2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self) -> np.ndarray:
        """Standard deviation, computed lazily from ``var``."""
        return np.sqrt(self.var)
