"""통계적 성질 — 고정 seed MC. 판정은 SE 배수 기준(확률적이나 seed 고정으로 결정적)."""

import numpy as np
import pytest

from ope.dgp import CONF_REWARD_SCALE, make_synthetic_bandit_data, true_policy_value
from ope.estimators import estimate_dm, estimate_dr, estimate_ips, estimate_snips
from conftest import BASE
from ope.policies import softmax_policy

S = 30
SEEDS = range(1000, 1000 + S)


def _collect(config):
    out = {"ips": [], "snips": [], "dm_wrong": [], "dr_wrong": [], "w_mean": []}
    for s in SEEDS:
        d = make_synthetic_bandit_data(config._replace(seed=s))
        q_wrong = 0.5 * d.q_true + 0.25
        ips = estimate_ips(d.reward, d.action, d.pscore_logged, d.pi_e_dist)
        out["ips"].append(ips.value)
        out["w_mean"].append(float(ips.weights.mean()))
        out["snips"].append(estimate_snips(d.reward, d.action, d.pscore_logged, d.pi_e_dist).value)
        out["dm_wrong"].append(estimate_dm(d.pi_e_dist, q_wrong).value)
        out["dr_wrong"].append(
            estimate_dr(d.reward, d.action, d.pscore_logged, d.pi_e_dist, q_wrong).value)
    return {k: np.asarray(v) for k, v in out.items()}


@pytest.fixture(scope="module")
def mc():
    return _collect(BASE)


@pytest.fixture(scope="module")
def v_true():
    return true_policy_value(BASE)


def test_ips_unbiased(mc, v_true):
    se = mc["ips"].std(ddof=1) / np.sqrt(S)
    assert abs(mc["ips"].mean() - v_true) <= 4 * se


def test_snips_lower_sd_than_ips(mc):
    assert mc["snips"].std(ddof=1) < mc["ips"].std(ddof=1)


def test_dm_wrong_model_biased(mc, v_true):
    se = mc["dm_wrong"].std(ddof=1) / np.sqrt(S)
    assert abs(mc["dm_wrong"].mean() - v_true) > 5 * max(se, 1e-12)


def test_dr_survives_wrong_model(mc, v_true):
    se = mc["dr_wrong"].std(ddof=1) / np.sqrt(S)
    assert abs(mc["dr_wrong"].mean() - v_true) <= 4 * se


def test_sample_mean_weight_near_one(mc):
    assert abs(mc["w_mean"].mean() - 1.0) < 0.01


def test_on_policy_end_to_end():
    """종단 검산: π_e 를 U 채널 포함 전체 reward 로 실배포 시뮬 → 평균보상 ≈ v_true.

    B/C 의미론(연속형 reward·U 개입·v_true 불변)의 버그를 한 번에 잡는 테스트 (Plan 검증 추가).
    """
    cfg = BASE._replace(confounding_strength=1.0, n=200_000, seed=777)
    from ope.dgp import _make_structure, _q_true  # 구조 재사용 (테스트 전용 내부 접근)

    theta, b, _ = _make_structure(cfg)
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.n, cfg.dim_context))
    q = _q_true(x, theta, b)
    pi_e = softmax_policy(q, cfg.beta_eval)
    a = (pi_e.cumsum(axis=1) > rng.random((cfg.n, 1))).argmax(axis=1)
    u = rng.normal(size=cfg.n)
    eps = rng.normal(size=cfg.n)
    r = (q[np.arange(cfg.n), a] + CONF_REWARD_SCALE * cfg.confounding_strength * u
         + cfg.reward_noise * eps)
    se = r.std(ddof=1) / np.sqrt(cfg.n)
    assert abs(r.mean() - true_policy_value(cfg)) <= 4 * se


def test_confounding_contrast():
    """축 09 전제: γ>0 에서 IPS(pscore_true) 는 불편, IPS(pscore_logged) 는 유의 편향."""
    # γ=2.5·n=30k·S=25: 고정 seed 에서 bias_logged ≈ +0.023 > 1.6×(6·SE) — 결정적 여유 마진.
    # (γ↑ 는 oracle-IPS 분산도 키운다 — pscore_true 가 U 틸트로 작아지는 표본 때문. 4·SE 로 흡수.)
    cfg = BASE._replace(confounding_strength=2.5, n=30_000)
    v = true_policy_value(cfg)
    est_true, est_logged = [], []
    for s in range(2000, 2025):
        d = make_synthetic_bandit_data(cfg._replace(seed=s))
        est_true.append(estimate_ips(d.reward, d.action, d.pscore_true, d.pi_e_dist).value)
        est_logged.append(estimate_ips(d.reward, d.action, d.pscore_logged, d.pi_e_dist).value)
    est_true, est_logged = np.asarray(est_true), np.asarray(est_logged)
    se_t = est_true.std(ddof=1) / np.sqrt(len(est_true))
    se_l = est_logged.std(ddof=1) / np.sqrt(len(est_logged))
    assert abs(est_true.mean() - v) <= 4 * se_t          # oracle propensity → 불편
    assert abs(est_logged.mean() - v) > 6 * se_l          # 기록 propensity → 편향
