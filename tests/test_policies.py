import numpy as np
import pytest

from ope.policies import epsilon_greedy_policy, softmax_policy


def test_softmax_rows_are_distributions():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(50, 7))
    pi = softmax_policy(q, beta=2.5)
    assert pi.shape == q.shape
    assert np.all(pi > 0)
    assert np.allclose(pi.sum(axis=1), 1.0)


def test_softmax_beta_zero_is_uniform():
    q = np.random.default_rng(1).normal(size=(10, 4))
    assert np.allclose(softmax_policy(q, beta=0.0), 0.25)


def test_softmax_monotone_in_q():
    q = np.array([[0.1, 0.9]])
    pi = softmax_policy(q, beta=3.0)
    assert pi[0, 1] > pi[0, 0]
    pi_neg = softmax_policy(q, beta=-3.0)  # worse-than-uniform 방향
    assert pi_neg[0, 1] < pi_neg[0, 0]


def test_softmax_extreme_beta_stable():
    q = np.array([[0.0, 1.0, 0.5]])
    pi = softmax_policy(q, beta=1e4)
    assert np.isfinite(pi).all()
    assert pi[0, 1] == pytest.approx(1.0)


def test_epsilon_greedy_values():
    q = np.array([[0.1, 0.9, 0.5]])
    pi = epsilon_greedy_policy(q, eps=0.3)
    assert pi[0, 1] == pytest.approx(0.7)
    assert pi[0, 0] == pytest.approx(0.15)
    assert np.allclose(pi.sum(axis=1), 1.0)


def test_epsilon_greedy_validation():
    q = np.zeros((2, 3))
    with pytest.raises(ValueError):
        epsilon_greedy_policy(q, eps=1.5)
    with pytest.raises(ValueError):
        epsilon_greedy_policy(np.zeros((2, 1)), eps=0.1)
