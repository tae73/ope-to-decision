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


def test_log_invariant_to_beta_eval():
    """축 10 전제(회귀 고정): β_eval 은 rng 를 소비하지 않는다 — 같은 seed ⇒ 같은 로그.
    (같은 로그 위에서 후보 정책들을 평가하는 factorial 설계의 성립 조건.)"""
    d1 = make_synthetic_bandit_data(BASE._replace(beta_eval=3.0))
    d2 = make_synthetic_bandit_data(BASE._replace(beta_eval=10.0))
    np.testing.assert_array_equal(d1.action, d2.action)
    np.testing.assert_allclose(d1.reward, d2.reward)
    np.testing.assert_allclose(d1.pscore_logged, d2.pscore_logged)
    assert not np.allclose(d1.pi_e_dist, d2.pi_e_dist)  # 평가 정책만 달라짐


def test_v_true_invariant_to_confounding_and_noise():
    """설계 정리: v_true 는 γ·σ 와 무관 (연속형 reward — dgp.py 설계 결정)."""
    v0 = true_policy_value(BASE)
    v1 = true_policy_value(BASE._replace(confounding_strength=2.0, reward_noise=1.5))
    assert v0 == v1


# ── M8 추가분 (PLAN §3.5) ──────────────────────────────────────────────────────


def test_pi_log_dist_is_logged_layer_matrix(data):
    """M8 필드 계약: pi_log_dist 는 기록 로깅 분포의 행렬판 — 행합 1,
    pscore_logged = pi_log_dist[i, a_i] 정확 항등(로그 층 자산, oracle 아님)."""
    n = BASE.n
    assert data.pi_log_dist.shape == (n, BASE.n_actions)
    np.testing.assert_allclose(data.pi_log_dist.sum(axis=1), 1.0, rtol=1e-12)
    np.testing.assert_array_equal(
        data.pscore_logged, data.pi_log_dist[np.arange(n), data.action])


def test_dgp_output_frozen_checksums():
    """DGP 동결 보호(M8 — PLAN §3.5-4): 대표 config 2종의 산출 checksum 을 리터럴로 고정.
    marginal_logging_dist 등 사후 함수 추가·리팩터가 생성기 rng draw 순서를 건드리면
    여기서 즉시 깨진다 — 축 01–16 committed 결과의 재현성 배리어."""
    cfg = DGPConfig(n=10_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                    support_deficiency=0.0, reward_noise=0.5, confounding_strength=0.0,
                    seed=12345, struct_seed=7)
    d = make_synthetic_bandit_data(cfg)
    assert int(d.action.sum()) == 45243
    assert float(d.reward.sum()) == pytest.approx(5985.649106063735, rel=1e-12)
    assert float(d.pscore_logged.sum()) == pytest.approx(1079.8742437535548, rel=1e-12)
    assert float(d.v_true) == pytest.approx(0.7153918288961995, rel=1e-12)
    cfg2 = cfg._replace(support_deficiency=0.4, confounding_strength=2.5, seed=777)
    d2 = make_synthetic_bandit_data(cfg2)
    assert int(d2.action.sum()) == 44640
    assert float(d2.reward.sum()) == pytest.approx(7371.39925462877, rel=1e-12)
    assert float(d2.pscore_logged.sum()) == pytest.approx(1708.5045560528035, rel=1e-12)
    assert float(d2.pscore_true.sum()) == pytest.approx(2997.9107306683977, rel=1e-12)


def test_marginal_logging_dist_gamma0_identity(data):
    """γ=0 이면 U-주변화 분포 == 기록 분포 정확 항등(mask 유무 모두) — probe M8-B 재현."""
    from ope.dgp import marginal_logging_dist
    p = marginal_logging_dist(BASE, data.context, n_nodes=200)
    np.testing.assert_allclose(p, data.pi_log_dist, atol=1e-12)
    cfg4 = BASE._replace(support_deficiency=0.4)
    d4 = make_synthetic_bandit_data(cfg4)
    p4 = marginal_logging_dist(cfg4, d4.context, n_nodes=200)
    np.testing.assert_allclose(p4, d4.pi_log_dist, atol=1e-12)


def test_marginal_logging_dist_calibrated_world(data):
    """γ>0 관측 동등성 세계: 주변화 pscore 를 기록으로 쓰면 E[w]≈1 (HT 항등 — k·SE 검정)
    + 행합 1 + 구적 수렴(200 vs 400 노드)."""
    from ope.dgp import marginal_logging_dist
    cfg = BASE._replace(confounding_strength=1.5)
    d = make_synthetic_bandit_data(cfg)
    p = marginal_logging_dist(cfg, d.context, n_nodes=400)
    np.testing.assert_allclose(p.sum(axis=1), 1.0, rtol=1e-12)
    p200 = marginal_logging_dist(cfg, d.context, n_nodes=200)
    assert float(np.max(np.abs(p - p200) / p)) < 1e-8
    idx = np.arange(BASE.n)
    w = d.pi_e_dist[idx, d.action] / p[idx, d.action]
    se = float(w.std(ddof=1) / np.sqrt(BASE.n))
    assert abs(float(w.mean()) - 1.0) < 4 * se
