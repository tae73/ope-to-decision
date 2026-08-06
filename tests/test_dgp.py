import numpy as np
import pytest

from ope.dgp import (
    DGPConfig,
    _support_mask,
    make_synthetic_bandit_data,
    true_policy_value,
)
from conftest import BASE
from ope.policies import softmax_policy


def test_shapes_and_ranges(data):
    n, k = BASE.n, BASE.n_actions
    assert data.context.shape == (n, BASE.dim_context)
    assert data.action.shape == (n,) and data.action.min() >= 0 and data.action.max() < k
    assert data.pi_e_dist.shape == (n, k) and np.allclose(data.pi_e_dist.sum(axis=1), 1.0)
    assert np.all(data.pscore_logged > 0) and np.all(data.pscore_logged <= 1.0)
    assert np.all(data.pscore_true > 0) and np.all(data.pscore_true <= 1.0)
    assert np.isfinite(data.reward).all()
    assert 0.0 < data.v_true < 1.0  # q = sigmoid 이므로


def test_deterministic_given_seed():
    d1 = make_synthetic_bandit_data(BASE)
    d2 = make_synthetic_bandit_data(BASE)
    np.testing.assert_array_equal(d1.action, d2.action)
    np.testing.assert_allclose(d1.reward, d2.reward)


def test_unconfounded_means_logged_equals_true(data):
    assert BASE.confounding_strength == 0.0
    np.testing.assert_allclose(data.pscore_logged, data.pscore_true, rtol=1e-14)


def test_confounded_means_logged_differs():
    cfg = BASE._replace(confounding_strength=1.0)
    d = make_synthetic_bandit_data(cfg)
    assert not np.allclose(d.pscore_logged, d.pscore_true)


def test_support_mask_structural():
    rng = np.random.default_rng(5)
    q = rng.uniform(size=(100, 5))
    mask = _support_mask(q, deficiency=0.2)  # ⌊0.2·5⌋ = 1 per row
    assert mask.shape == q.shape
    assert np.all((~mask).sum(axis=1) == 1)
    removed_q = q[~mask]
    assert np.all(removed_q <= q.min(axis=1) + 1e-12)  # 제거된 것은 row 최솟값(하위-q)


def test_support_deficiency_zero_is_full_support():
    q = np.random.default_rng(6).uniform(size=(10, 4))
    assert _support_mask(q, 0.0).all()
    with pytest.raises(ValueError):
        _support_mask(q, 1.0)


def test_row_weight_identity_exact():
    """행별 Σ_a π_0(a|x)·(π_e(a|x)/π_0(a|x)) = 1 — 정확 항등식."""
    rng = np.random.default_rng(7)
    q = rng.normal(size=(50, 6))
    pi0 = softmax_policy(q, 1.0)
    pi_e = softmax_policy(q, 3.0)
    np.testing.assert_allclose((pi0 * (pi_e / pi0)).sum(axis=1), 1.0, rtol=1e-12)


def test_v_true_mc_stable():
    v1 = true_policy_value(BASE, n_mc=100_000)
    v2 = true_policy_value(BASE, n_mc=400_000)
    assert v1 == pytest.approx(v2, abs=0.005)


def test_v_true_invariant_to_confounding_and_noise():
    """설계 정리: v_true 는 γ·σ 와 무관 (연속형 reward — dgp.py 설계 결정)."""
    v0 = true_policy_value(BASE)
    v1 = true_policy_value(BASE._replace(confounding_strength=2.0, reward_noise=1.5))
    assert v0 == v1
