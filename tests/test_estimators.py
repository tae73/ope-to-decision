"""정확 항등식 + 손으로 계산한 리터럴 예제 — 통계적 성질은 test_statistical.py."""

import numpy as np
import pytest

from ope.estimators import (
    bootstrap_ci,
    estimate_clipped_ips,
    estimate_dm,
    estimate_dr,
    estimate_dros,
    estimate_ips,
    estimate_snips,
    estimate_switch_dr,
)

# ── 손 계산 리터럴 예제 (n=3, K=2) ────────────────────────────────────────────
R = np.array([1.0, 0.0, 2.0])
A = np.array([0, 1, 0])
PS = np.array([0.5, 0.4, 0.8])
PE = np.array([[0.9, 0.1], [0.3, 0.7], [0.6, 0.4]])
QH = np.array([[0.5, 0.2], [0.1, 0.3], [1.0, 0.5]])
# w = [0.9/0.5, 0.7/0.4, 0.6/0.8] = [1.8, 1.75, 0.75]


def test_hand_example_exact():
    assert estimate_ips(R, A, PS, PE).value == pytest.approx(1.1)                      # (1.8+0+1.5)/3
    assert estimate_snips(R, A, PS, PE).value == pytest.approx(3.3 / 4.3)
    assert estimate_dm(PE, QH).value == pytest.approx(1.51 / 3)                        # (.47+.24+.8)/3
    assert estimate_dr(R, A, PS, PE, QH).value == pytest.approx(1.51 / 3 + 0.375)
    assert estimate_clipped_ips(R, A, PS, PE, lam=1.0).value == pytest.approx(2.5 / 3)  # w_c=[1,1,.75]
    assert estimate_switch_dr(R, A, PS, PE, QH, tau=1.0).value == pytest.approx(1.51 / 3 + 0.25)
    assert estimate_dros(R, A, PS, PE, QH, lam=1.0).value == pytest.approx(0.6910111272375423)


def test_weights_field_semantics():
    assert estimate_dm(PE, QH).weights.size == 0
    np.testing.assert_allclose(estimate_ips(R, A, PS, PE).weights, [1.8, 1.75, 0.75])
    np.testing.assert_allclose(estimate_switch_dr(R, A, PS, PE, QH, tau=1.0).weights, [0.0, 0.0, 0.75])


# ── 항등식 (합성 데이터 fixture 위에서 정확 성립) ─────────────────────────────
def test_clipped_lam_inf_equals_ips(data):
    ips = estimate_ips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).value
    clip = estimate_clipped_ips(data.reward, data.action, data.pscore_logged, data.pi_e_dist,
                                lam=1e9).value
    assert clip == pytest.approx(ips, rel=1e-12)


def test_switch_tau_inf_equals_dr(data):
    q_hat = 0.5 * data.q_true + 0.25
    dr = estimate_dr(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q_hat).value
    sw = estimate_switch_dr(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q_hat,
                            tau=1e9).value
    assert sw == pytest.approx(dr, rel=1e-12)


def test_dros_limits(data):
    q_hat = 0.5 * data.q_true + 0.25
    dm = estimate_dm(data.pi_e_dist, q_hat).value
    dr = estimate_dr(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q_hat).value
    lo = estimate_dros(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q_hat,
                       lam=1e-12).value
    hi = estimate_dros(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q_hat,
                       lam=1e12).value
    assert lo == pytest.approx(dm, abs=1e-4)   # λ→0⁺ 에서 O(λ) 근사로 DM
    assert hi == pytest.approx(dr, rel=1e-6)   # λ→∞ 에서 DR


def test_dr_with_zero_qhat_equals_ips(data):
    q0 = np.zeros_like(data.q_true)
    ips = estimate_ips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).value
    dr = estimate_dr(data.reward, data.action, data.pscore_logged, data.pi_e_dist, q0).value
    assert dr == pytest.approx(ips, rel=1e-12)


def test_snips_invariant_to_weight_scale(data):
    v1 = estimate_snips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).value
    v2 = estimate_snips(data.reward, data.action, 0.5 * data.pscore_logged, data.pi_e_dist).value
    assert v2 == pytest.approx(v1, rel=1e-12)


def test_hyperparam_validation():
    with pytest.raises(ValueError):
        estimate_clipped_ips(R, A, PS, PE, lam=0.0)
    with pytest.raises(ValueError):
        estimate_switch_dr(R, A, PS, PE, QH, tau=-1.0)
    with pytest.raises(ValueError):
        estimate_dros(R, A, PS, PE, QH, lam=0.0)
    with pytest.raises(ValueError):
        estimate_ips(R, A, np.array([0.5, 0.0, 0.8]), PE)  # pscore=0 금지


# ── bootstrap CI ──────────────────────────────────────────────────────────────
def test_bootstrap_ci_basic(data):
    sl = slice(0, 500)
    args = (data.reward[sl], data.action[sl], data.pscore_logged[sl], data.pi_e_dist[sl])
    point = estimate_ips(*args).value
    lo, hi = bootstrap_ci(*args, q_hat=None, estimator="ips", n_boot=400, alpha=0.05, seed=3)
    lo2, hi2 = bootstrap_ci(*args, q_hat=None, estimator="ips", n_boot=400, alpha=0.05, seed=3)
    assert lo < hi
    assert lo <= point <= hi
    assert (lo, hi) == (lo2, hi2)  # 고정 seed 재현


def test_bootstrap_ci_validation(data):
    args = (data.reward[:100], data.action[:100], data.pscore_logged[:100], data.pi_e_dist[:100])
    with pytest.raises(ValueError):
        bootstrap_ci(*args, q_hat=None, estimator="nope", n_boot=10, alpha=0.05, seed=0)
    with pytest.raises(ValueError):
        bootstrap_ci(*args, q_hat=None, estimator="dr", n_boot=10, alpha=0.05, seed=0)  # q_hat 필요
    with pytest.raises(ValueError):
        bootstrap_ci(*args, q_hat=data.q_true[:100], estimator="dros", n_boot=10, alpha=0.05,
                     seed=0)  # hyperparam 필요
    for bad_alpha in (0.0, 1.0, 1.2):  # alpha=유의수준 — 범위 밖은 fail-silent 방지 위해 즉사
        with pytest.raises(ValueError):
            bootstrap_ci(*args, q_hat=None, estimator="ips", n_boot=10, alpha=bad_alpha, seed=0)
