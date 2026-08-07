"""축 14 MSM 정규화 bound — Λ=1 항등·단조 확장·n=8 brute-force 전수 대조·λ<1 검증.

알고리즘 출처: probe M5-14 (experiments/probes/probe_lambda_msm.py, GO) 의 정렬-임계 cumsum
정확해를 estimators.py 규약으로 이식한 것 — 여기 brute-force 는 그 "정확해" 주장 자체를
극점 전수(각 표본 u∈{1/Λ,Λ}, 2^8=256 조합)와 대조해 검증한다.
"""

import itertools

import numpy as np
import pytest

from ope.estimators import estimate_snips, msm_snips_bounds


def test_lam1_is_snips_identity(data):
    """① Λ=1 ⇒ lo=hi=SNIPS (배율 구간이 점 {1} 로 붕괴 — worst case = 점추정)."""
    snips = estimate_snips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).value
    lo, hi = msm_snips_bounds(data.reward, data.action, data.pscore_logged, data.pi_e_dist, 1.0)
    assert lo == pytest.approx(snips, abs=1e-11)
    assert hi == pytest.approx(snips, abs=1e-11)


def test_monotone_expansion(data):
    """② Λ 증가 ⇒ [lo, hi] 단조 확장 (feasible set [1/Λ,Λ]^n 이 중첩 확대되므로 정확 성질)."""
    snips = estimate_snips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).value
    prev_lo, prev_hi = snips, snips
    for lam in (1.2, 1.5, 2.0, 3.0, 5.0, 8.0):
        lo, hi = msm_snips_bounds(data.reward, data.action, data.pscore_logged,
                                  data.pi_e_dist, lam)
        assert lo <= prev_lo + 1e-12
        assert hi >= prev_hi - 1e-12
        assert lo <= snips + 1e-11 and hi >= snips - 1e-11  # 점추정은 항상 구간 안
        prev_lo, prev_hi = lo, hi


def test_bruteforce_n8_exact():
    """③ n=8 전수 대조: u∈{1/Λ,Λ}^8 극점 256 조합의 min/max 와 정확 일치.

    목적함수 Σu_i w_i r_i / Σu_i w_i 는 각 u_i 에 단조(fractional-linear — 좌표별 Möbius 변환)
    이므로 극값은 box 의 꼭짓점에서 달성된다 → 극점 전수 = 정확 기준값.
    """
    rng = np.random.default_rng(42)
    n, k = 8, 3
    reward = rng.normal(0.5, 1.0, size=n)
    action = rng.integers(0, k, size=n)
    pscore = rng.uniform(0.2, 0.8, size=n)
    pi_e = rng.uniform(0.05, 1.0, size=(n, k))
    pi_e /= pi_e.sum(axis=1, keepdims=True)
    w = pi_e[np.arange(n), action] / pscore

    for lam in (1.0, 1.3, 2.0, 4.0):
        lo, hi = msm_snips_bounds(reward, action, pscore, pi_e, lam)
        vals = []
        for combo in itertools.product((1.0 / lam, lam), repeat=n):
            u = np.asarray(combo)
            vals.append(float((u * w * reward).sum() / (u * w).sum()))
        assert lo == pytest.approx(min(vals), rel=1e-12, abs=1e-12)
        assert hi == pytest.approx(max(vals), rel=1e-12, abs=1e-12)


def test_lam_below_one_raises(data):
    """④ λ<1 은 감도 모델 정의 위반 — 즉시 ValueError (fail-silent 방지)."""
    sl = slice(0, 100)
    args = (data.reward[sl], data.action[sl], data.pscore_logged[sl], data.pi_e_dist[sl])
    for bad in (0.99, 0.5, 0.0, -2.0):
        with pytest.raises(ValueError):
            msm_snips_bounds(*args, bad)
