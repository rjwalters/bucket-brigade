"""Tests for the plug-in information-theoretic estimators.

These tests pin down the estimators against analytically known values on
synthetic distributions. They are *not* expected to be tight (bias correction
gives only the leading term), but they should land within 0.05 bits of the
truth at the sample sizes used here.
"""

import math

import numpy as np
import pytest

from bucket_brigade.analysis.info_theory import (
    bootstrap_ci,
    conditional_entropy,
    conditional_mutual_information,
    entropy_discrete,
    joint_entropy,
    mutual_information,
    quantize_uniform,
)


TOL_LARGE = 0.05  # bits; OK at N >= 10000
TOL_SMALL = 0.20  # bits; OK at N >= 1000


class TestEntropy:
    def test_zero_for_constant(self):
        x = np.zeros(1000, dtype=int)
        assert entropy_discrete(x) == pytest.approx(0.0, abs=1e-9)

    def test_uniform_on_k(self):
        # H(uniform_k) = log2(k); MM correction is small at large N.
        rng = np.random.default_rng(0)
        for k in (2, 4, 8):
            x = rng.integers(0, k, size=10_000)
            est = entropy_discrete(x)
            assert est == pytest.approx(math.log2(k), abs=TOL_LARGE)

    def test_fair_coin(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 2, size=100_000)
        assert entropy_discrete(x) == pytest.approx(1.0, abs=TOL_LARGE)

    def test_biased_coin(self):
        # H(Bernoulli(0.1)) = -0.1 log2 0.1 - 0.9 log2 0.9 ≈ 0.469
        rng = np.random.default_rng(2)
        x = (rng.random(100_000) < 0.1).astype(int)
        truth = -0.1 * math.log2(0.1) - 0.9 * math.log2(0.9)
        assert entropy_discrete(x) == pytest.approx(truth, abs=TOL_LARGE)

    def test_no_correction_is_lower(self):
        # MM correction should be > 0 when categories observed.
        rng = np.random.default_rng(3)
        x = rng.integers(0, 8, size=200)  # small N to make MM matter
        h_plug = entropy_discrete(x, bias_correction="none")
        h_mm = entropy_discrete(x, bias_correction="miller-madow")
        assert h_mm > h_plug
        # The correction (K-1)/(2N ln 2) should be the gap.
        # With k <= 8 observed and N = 200: (7) / (400 * 0.693) ≈ 0.0252 bits
        assert (h_mm - h_plug) == pytest.approx(7 / (2 * 200 * math.log(2)), abs=1e-6)

    def test_empty_returns_zero(self):
        assert entropy_discrete(np.array([], dtype=int)) == 0.0

    def test_unknown_correction_raises(self):
        with pytest.raises(ValueError):
            entropy_discrete(np.array([0, 1, 2]), bias_correction="bogus")


class TestJointEntropy:
    def test_joint_of_independent(self):
        # X, Y independent uniform on {0, 1, 2, 3}: H(X, Y) = 4 bits.
        rng = np.random.default_rng(10)
        x = rng.integers(0, 4, size=20_000)
        y = rng.integers(0, 4, size=20_000)
        assert joint_entropy(x, y) == pytest.approx(4.0, abs=TOL_LARGE)

    def test_joint_of_identical(self):
        # X = Y: H(X, Y) = H(X) = log2 k.
        rng = np.random.default_rng(11)
        x = rng.integers(0, 4, size=20_000)
        assert joint_entropy(x, x) == pytest.approx(2.0, abs=TOL_LARGE)


class TestMutualInformation:
    def test_zero_for_independent(self):
        rng = np.random.default_rng(20)
        x = rng.integers(0, 4, size=20_000)
        y = rng.integers(0, 4, size=20_000)
        assert mutual_information(x, y) == pytest.approx(0.0, abs=TOL_LARGE)

    def test_full_for_identical(self):
        # I(X; X) = H(X)
        rng = np.random.default_rng(21)
        x = rng.integers(0, 4, size=20_000)
        truth = 2.0  # log2(4)
        assert mutual_information(x, x) == pytest.approx(truth, abs=TOL_LARGE)

    def test_symmetry(self):
        # I(X; Y) == I(Y; X)
        rng = np.random.default_rng(22)
        x = rng.integers(0, 3, size=5_000)
        y = (x + rng.integers(0, 2, size=5_000)) % 3  # correlated
        assert mutual_information(x, y) == pytest.approx(
            mutual_information(y, x), abs=1e-9
        )

    def test_partial_correlation(self):
        # Y = X xor B where B ~ Bernoulli(0.1). I(X; Y) ≈ 1 - H(B) ≈ 0.531.
        rng = np.random.default_rng(23)
        x = rng.integers(0, 2, size=20_000)
        flip = (rng.random(20_000) < 0.1).astype(int)
        y = x ^ flip
        h_flip = -0.1 * math.log2(0.1) - 0.9 * math.log2(0.9)
        truth = 1.0 - h_flip
        assert mutual_information(x, y) == pytest.approx(truth, abs=TOL_LARGE)


class TestConditionalEntropy:
    def test_zero_for_function_of_y(self):
        # If X is a function of Y, H(X | Y) = 0.
        rng = np.random.default_rng(30)
        y = rng.integers(0, 4, size=10_000)
        x = (y * 2) % 4
        assert conditional_entropy(x, y) == pytest.approx(0.0, abs=TOL_LARGE)

    def test_full_for_independent(self):
        # X _||_ Y: H(X | Y) = H(X).
        rng = np.random.default_rng(31)
        x = rng.integers(0, 4, size=20_000)
        y = rng.integers(0, 2, size=20_000)
        assert conditional_entropy(x, y) == pytest.approx(2.0, abs=TOL_LARGE)


class TestConditionalMutualInformation:
    def test_zero_when_z_d_separates(self):
        # X = Z, Y = Z, so X _||_ Y | Z. I(X; Y | Z) should be 0.
        rng = np.random.default_rng(40)
        z = rng.integers(0, 4, size=20_000)
        x = z.copy()
        y = z.copy()
        assert conditional_mutual_information(x, y, z) == pytest.approx(
            0.0, abs=TOL_LARGE
        )

    def test_full_when_independent_of_z(self):
        # X = Y (fully correlated) and Z _||_ (X, Y): I(X; Y | Z) ≈ I(X; Y) = H(X).
        rng = np.random.default_rng(41)
        x = rng.integers(0, 2, size=20_000)
        y = x.copy()
        z = rng.integers(0, 2, size=20_000)
        assert conditional_mutual_information(x, y, z) == pytest.approx(
            1.0, abs=TOL_LARGE
        )

    def test_pmic_synchrony_scenario(self):
        # Two "agents" both copy a shared task signal S (e.g., synchronized actions).
        # Unconditional MI is large; conditional MI given S should be near 0.
        # This is the failure mode PMIC documents: penalize unconditional MI and
        # you fight the task. Conditional MI is the correct quantity.
        rng = np.random.default_rng(42)
        s = rng.integers(0, 4, size=20_000)
        noise_x = (rng.random(20_000) < 0.05).astype(int)
        noise_y = (rng.random(20_000) < 0.05).astype(int)
        # Agents' encoder outputs: mostly s, with small independent noise.
        x = (s + noise_x) % 4
        y = (s + noise_y) % 4

        i_unconditional = mutual_information(x, y)
        i_conditional = conditional_mutual_information(x, y, s)

        # Unconditional should be near full action entropy (~2 bits minus small noise).
        assert i_unconditional > 1.0
        # Conditional should be near zero — the agents share noise only.
        assert i_conditional < 0.1


class TestBootstrap:
    def test_point_estimate_returned(self):
        rng = np.random.default_rng(50)
        x = rng.integers(0, 4, size=2_000)
        est, lo, hi = bootstrap_ci(
            entropy_discrete, (x,), n_boot=200, confidence=0.95,
            rng=np.random.default_rng(51),
        )
        # Point estimate is the full-sample estimator.
        assert est == entropy_discrete(x)
        # CI should bracket the point estimate.
        assert lo <= est <= hi

    def test_mi_ci_brackets_truth(self):
        rng = np.random.default_rng(52)
        x = rng.integers(0, 4, size=2_000)
        y = x.copy()  # I(X; X) = log2 4 = 2 bits.
        est, lo, hi = bootstrap_ci(
            mutual_information, (x, y), n_boot=200,
            rng=np.random.default_rng(53),
        )
        assert lo <= 2.0 <= hi or abs(est - 2.0) < TOL_LARGE


class TestQuantize:
    def test_uniform_1d(self):
        x = np.linspace(0, 1, 1000)
        bins = quantize_uniform(x, n_bins=8)
        assert bins.min() == 0
        assert bins.max() == 7
        # Roughly equal bin populations.
        counts = np.bincount(bins, minlength=8)
        assert counts.std() < counts.mean()

    def test_constant_input(self):
        x = np.zeros(100)
        bins = quantize_uniform(x, n_bins=8)
        assert (bins == 0).all()

    def test_2d_packing(self):
        # Two independent uniform columns → packed code has up to n_bins**2 values.
        rng = np.random.default_rng(60)
        x = rng.random((5_000, 2))
        codes = quantize_uniform(x, n_bins=4)
        assert codes.shape == (5_000,)
        # Should see roughly all 16 combinations.
        assert len(np.unique(codes)) >= 10
