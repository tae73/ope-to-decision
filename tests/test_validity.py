"""validity battery(M8, PLAN §3.5-1) property tests — 5장르.

battery 는 필요조건 검사(falsifier)다: 여기서 고정하는 성질은 "임계 그대로의 fail 규칙이
잡으라는 것을 잡고(방향 포함), 잡지 말라는 것에 오알람을 내지 않으며, joint bootstrap 이
per-estimator 재계산과 산술 동일하고, Λ*_flip 이 grid 스캔과 정합"이다. 사전등록 임계의
사후 조정을 막는 것이 테스트의 존재 이유 — 임계값 리터럴은 ValidityConfig 기본값과의
일치로 고정한다.
"""

import numpy as np
import pytest

from conftest import BASE
from ope.dgp import make_synthetic_bandit_data
from ope.estimators import (
    estimate_clipped_ips, estimate_dm, estimate_dr, estimate_dros,
    estimate_ips, estimate_snips, estimate_switch_dr, msm_snips_bounds,
)
from ope.validity import (
    LambdaStar, ValidityConfig, bootstrap_joint, lambda_star_vs_anchor,
    make_placebo_reward, run_validity_checks,
)


def test_preregistered_defaults_locked():
    """ValidityConfig 기본값 = PLAN §3.5-1 사전등록값 — 침묵 변경 시 여기서 깨진다."""
    cfg = ValidityConfig()
    assert (cfg.n_boot, cfg.alpha) == (500, 0.05)
    assert (cfg.mean_w_tol, cfg.harmonic_tol, cfg.disagreement_tol) == (0.10, 0.25, 0.50)
    assert cfg.min_action_count == 30 and cfg.disagreement_floor == 0.01


def test_exact_uniform_identities():
    """정확 항등: π_e=π₀=uniform·pscore 정확 기록 ⇒ w≡1 → mean_w=1, T(a)=n_a·K/n 정확."""
    reward = np.array([1.0, 0.0, 1.0, 0.0])
    action = np.array([0, 0, 1, 1])
    pscore = np.full(4, 0.5)
    pi_e = np.full((4, 2), 0.5)
    cfg = ValidityConfig(n_boot=50, min_action_count=1, seed=3)
    rep = run_validity_checks(reward, action, pscore, pi_e, cfg=cfg)
    assert rep.mean_w.value == 1.0
    for (_, t, _, _, n_a, fail) in rep.harmonic_by_action:
        assert t == pytest.approx(n_a * 2 / 4, abs=1e-15) and not fail
    assert rep.mean_w.state == "pass" and rep.harmonic.state == "pass"


def test_config_contract_and_validation(data):
    """계약: raw dict cfg 는 TypeError(오타 키 침묵 통과 방지 — GateThresholds 선례),
    pscore≤0 은 estimators 검증이 전파."""
    with pytest.raises(TypeError):
        run_validity_checks(data.reward, data.action, data.pscore_logged,
                            data.pi_e_dist, cfg={"n_boot": 10})
    bad = data.pscore_logged.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError):
        run_validity_checks(data.reward, data.action, bad, data.pi_e_dist,
                            cfg=ValidityConfig(n_boot=10))


def test_statistical_clean_all_pass(data):
    """clean 로그(BASE)에서 전 arm 무발화 — 오알람 없음 (probe M8-A clean 재현)."""
    rep = run_validity_checks(data.reward, data.action, data.pscore_logged,
                              data.pi_e_dist, context=data.context,
                              cfg=ValidityConfig(seed=11))
    assert rep.checks_failed == ()
    assert abs(rep.mean_w.value - 1.0) < 0.05


def test_statistical_noised_fires_up(data):
    """곱셈 log-normal 오염(s=1.0) ⇒ mean_w 가 e^{s²/2}≈1.65 방향으로 발화 (축 05 기전)."""
    z = np.random.default_rng(500_101).normal(size=BASE.n)
    ps = np.clip(data.pscore_logged * np.exp(1.0 * z), 1e-6, 1.0)
    rep = run_validity_checks(data.reward, data.action, ps, data.pi_e_dist,
                              cfg=ValidityConfig(seed=11))
    assert rep.mean_w.state == "fail" and rep.mean_w.value > 1.0
    assert "mean_w" in rep.checks_failed


def test_statistical_support_fires_down():
    """구조적 support 결핍(δ=0.4) ⇒ mean_w = 1 − 미지지 π_e 질량 < 1 로 발화 —
    전역 proxy 가 0 인 지점의 기대값 회복(probe M8-A 1순위 확인 항목의 회귀 고정)."""
    d = make_synthetic_bandit_data(BASE._replace(support_deficiency=0.4))
    rep = run_validity_checks(d.reward, d.action, d.pscore_logged, d.pi_e_dist,
                              cfg=ValidityConfig(seed=11))
    assert rep.mean_w.state == "fail" and rep.mean_w.value < 1.0


def test_placebo_constructed_zero_truth(data):
    """placebo ε 는 구성상 참값 0 — 점추정 = mean(w·ε) 산술 일치·재현성(cfg.seed 고정)."""
    cfg = ValidityConfig(n_boot=100, seed=42)
    eps1 = make_placebo_reward(data.reward, cfg)
    eps2 = make_placebo_reward(data.reward, cfg)
    np.testing.assert_array_equal(eps1, eps2)
    w = estimate_ips(data.reward, data.action, data.pscore_logged,
                     data.pi_e_dist).weights
    rep = run_validity_checks(data.reward, data.action, data.pscore_logged,
                              data.pi_e_dist, cfg=cfg)
    assert rep.placebo.value == pytest.approx(float((w * eps1).mean()), rel=1e-12)


def test_disagreement_floor_inconclusive(data):
    """|SNIPS| 극소 스케일 ⇒ disagreement 는 fail 이 아니라 inconclusive (§3.5-1 사전등록)."""
    rep = run_validity_checks(np.zeros(BASE.n), data.action, data.pscore_logged,
                              data.pi_e_dist, cfg=ValidityConfig(n_boot=20, seed=1))
    assert rep.disagreement.state == "inconclusive"
    assert "disagreement" not in rep.checks_failed


def test_bootstrap_joint_bruteforce_vs_estimators(data):
    """brute-force: joint bootstrap 의 replicate = 같은 인덱스로 재표집한 배열에 estimator
    함수를 직접 호출한 값과 산술 동일 — per-row 기여 벡터 요약이 지름길이 아님을 증명."""
    q_hat = 0.9 * data.q_true + 0.05
    cfg = ValidityConfig(n_boot=5, seed=7)
    w = estimate_ips(data.reward, data.action, data.pscore_logged,
                     data.pi_e_dist).weights
    hypers = {"tau": float(np.quantile(w, 0.95)),
              "lam_clip": float(np.quantile(w, 0.90)),
              "lam_dros": float(np.quantile(w, 0.90)) ** 2}
    boot = bootstrap_joint(data.reward, data.action, data.pscore_logged,
                           data.pi_e_dist, q_hat=q_hat, hypers=hypers, cfg=cfg)
    # 같은 rng 스트림 재구성 → 같은 인덱스 시퀀스
    kids = np.random.SeedSequence(cfg.seed).spawn(2)
    rng = np.random.default_rng(kids[1])
    n = BASE.n
    for b in range(cfg.n_boot):
        ii = rng.integers(0, n, n)
        r, a = data.reward[ii], data.action[ii]
        ps, pe = data.pscore_logged[ii], data.pi_e_dist[ii]
        qh = q_hat[ii]
        assert boot.estimates["ips"][b] == pytest.approx(
            estimate_ips(r, a, ps, pe).value, rel=1e-12)
        assert boot.estimates["snips"][b] == pytest.approx(
            estimate_snips(r, a, ps, pe).value, rel=1e-12)
        assert boot.estimates["clipped_ips"][b] == pytest.approx(
            estimate_clipped_ips(r, a, ps, pe, hypers["lam_clip"]).value, rel=1e-12)
        assert boot.estimates["dm"][b] == pytest.approx(
            estimate_dm(pe, qh).value, rel=1e-12)
        assert boot.estimates["dr"][b] == pytest.approx(
            estimate_dr(r, a, ps, pe, qh).value, rel=1e-12)
        assert boot.estimates["switch_dr"][b] == pytest.approx(
            estimate_switch_dr(r, a, ps, pe, qh, hypers["tau"]).value, rel=1e-12)
        assert boot.estimates["dros"][b] == pytest.approx(
            estimate_dros(r, a, ps, pe, qh, hypers["lam_dros"]).value, rel=1e-12)
        assert boot.mean_w[b] == pytest.approx(
            float((pe[np.arange(n), a] / ps).mean()), rel=1e-12)


def test_lambda_star_grid_consistency(data):
    """Λ*_flip 의 log-bisection 은 0.01 간격 grid 스캔의 최초 교차점과 정합 (축 14 패턴)."""
    anchor = float(data.reward.mean())
    got = lambda_star_vs_anchor(data.reward, data.action, data.pscore_logged,
                                data.pi_e_dist, anchor)
    if got.censored:
        pytest.skip("이 fixture 에선 censored — grid 대조 불가")
    grid = np.arange(1.0, 8.0 + 1e-9, 0.01)
    snips = estimate_snips(data.reward, data.action, data.pscore_logged,
                           data.pi_e_dist).value
    for lam in grid:
        lo, hi = msm_snips_bounds(data.reward, data.action, data.pscore_logged,
                                  data.pi_e_dist, float(lam))
        crossed = lo <= anchor if snips > anchor else hi >= anchor
        if crossed:
            assert abs(got.lam_star - lam) <= 0.011
            break


def test_lambda_star_censored_and_at_anchor(data):
    """경계: 도달 불가 anchor ⇒ censored=True·lam_star=lam_max / anchor=SNIPS ⇒ at_anchor."""
    snips = estimate_snips(data.reward, data.action, data.pscore_logged,
                           data.pi_e_dist).value
    far = lambda_star_vs_anchor(data.reward, data.action, data.pscore_logged,
                                data.pi_e_dist, anchor=snips + 10.0, lam_max=8.0)
    assert far == LambdaStar(8.0, True, "below")
    at = lambda_star_vs_anchor(data.reward, data.action, data.pscore_logged,
                               data.pi_e_dist, anchor=snips)
    assert at.direction == "at_anchor" and at.lam_star == 1.0


def test_report_deterministic(data):
    """같은 cfg ⇒ 같은 ValidityReport (placebo·bootstrap 이 cfg.seed 파생 — 재현성)."""
    cfg = ValidityConfig(n_boot=50, seed=99)
    r1 = run_validity_checks(data.reward, data.action, data.pscore_logged,
                             data.pi_e_dist, cfg=cfg)
    r2 = run_validity_checks(data.reward, data.action, data.pscore_logged,
                             data.pi_e_dist, cfg=cfg)
    assert r1.mean_w == r2.mean_w and r1.placebo == r2.placebo
    assert r1.harmonic_by_action == r2.harmonic_by_action
