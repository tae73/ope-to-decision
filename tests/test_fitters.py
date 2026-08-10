"""fitters.py property tests — cross-fit 계약·OOF 성질·fallback·pscore 회복·oracle-leak 금지.

테스트 데이터는 numpy 만으로 인라인 생성한다(ope.dgp 미사용) — fitters 는 GT-미상 트랙,
즉 로그 (x, a, r)만 주어진 상황을 재현해야 하고, src 쪽 oracle 격리 계약과도 정합해야 한다.
"""

from pathlib import Path

import numpy as np
import pytest

from ope import fitters
from ope.fitters import (
    PS_CLIP_MIN,
    CrossFitConfig,
    _fold_indices,
    fit_pscore_crossfit,
    fit_q_hat_crossfit,
    make_log_derived_candidate,
)

CFG = CrossFitConfig()  # n_folds=2, seed=0
N, K, D = 200, 4, 3


@pytest.fixture(scope="module")
def log():
    """미니 로그 (x, a, r): 선형 스코어 softmax 로깅 정책 + 선형 보상 — Ridge/LR 이 둘 다
    well-specified 라 contract·OOF 성질이 모델 실패와 얽히지 않는다."""
    rng = np.random.default_rng(42)
    x = rng.normal(size=(N, D))
    theta = rng.normal(size=(D, K))
    scores = x @ theta
    z = scores - scores.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    a = (p.cumsum(axis=1) > rng.random((N, 1))).argmax(axis=1)
    r = rng.normal(loc=scores[np.arange(N), a], scale=0.5)
    return x, a, r


# ---------------------------------------------------------------- 1. contract


def test_q_hat_shape_and_finite(log):
    """q̂ 계약: (n, K) 전 셀 유한 — downstream(DR·softmax 후보)이 무조건 의존하는 최소 계약."""
    x, a, r = log
    q = fit_q_hat_crossfit(x, a, r, K, CFG)
    assert q.shape == (N, K)
    assert np.all(np.isfinite(q))


def test_candidate_rows_are_distributions(log):
    """로그-유래 후보는 정책 분포여야 한다: 행합 1·비음 — estimator 의 pi_e_dist 계약."""
    x, a, r = log
    pi = make_log_derived_candidate(x, a, r, K, beta=2.0, cfg=CFG)
    assert pi.shape == (N, K)
    np.testing.assert_allclose(pi.sum(axis=1), 1.0)
    assert np.all(pi >= 0.0)


def test_candidate_beta_zero_uniform(log):
    """beta=0 ⇒ 정확히 uniform — softmax 온도 노브의 앵커(축 02·03 정책쌍 관례와 정합)."""
    x, a, r = log
    pi = make_log_derived_candidate(x, a, r, K, beta=0.0, cfg=CFG)
    np.testing.assert_allclose(pi, 1.0 / K)


def test_n_folds_lt_two_raises(log):
    """n_folds=1 은 cross-fit 이 아니라 in-sample — 준-null 함정으로 되돌아가므로 금지."""
    x, a, r = log
    with pytest.raises(ValueError):
        fit_q_hat_crossfit(x, a, r, K, CFG._replace(n_folds=1))
    with pytest.raises(ValueError):
        fit_pscore_crossfit(x, a, K, CFG._replace(n_folds=1))


def test_length_mismatch_raises(log):
    """길이 불일치는 조용한 브로드캐스트 대신 즉시 ValueError — 로그 정렬 실수 조기 검출."""
    x, a, r = log
    with pytest.raises(ValueError):
        fit_q_hat_crossfit(x, a[:-1], r, K, CFG)
    with pytest.raises(ValueError):
        fit_q_hat_crossfit(x, a, r[:-1], K, CFG)
    with pytest.raises(ValueError):
        fit_pscore_crossfit(x, a[:-1], K, CFG)


# ------------------------------------------------------------ 2. OOF property


def test_oof_predictions_independent_of_own_fold_rewards(log):
    """OOF 핵심 성질: fold i 행의 q̂ 은 fold i 의 reward 에 의존하면 안 된다 — 이것이 깨지면
    in-sample 자동보정(축 05 estimated 모드의 준-null 함정)이 되살아난다. fold 1 의 reward 만
    크게 교란하면 fold 1 행의 예측(fold 0 으로만 학습)은 불변이어야 하고, 반대로 fold 0 행의
    예측은 변해야 교란 자체가 유효했음이 확인된다(같은 seed ⇒ fold 분할 결정적)."""
    x, a, r = log
    folds = _fold_indices(N, CFG.n_folds, CFG.seed)
    q_base = fit_q_hat_crossfit(x, a, r, K, CFG)
    r_pert = r.copy()
    r_pert[folds[1]] += 100.0
    q_pert = fit_q_hat_crossfit(x, a, r_pert, K, CFG)
    np.testing.assert_allclose(q_pert[folds[1]], q_base[folds[1]])  # 자기 fold reward 와 무관
    assert not np.allclose(q_pert[folds[0]], q_base[folds[0]])      # 교란이 실제로 전파됨


# --------------------------------------------------- 3. missing-action fallback


def test_missing_action_fallback_no_crash():
    """한 fold 에만 존재하는 action: 반대 fold 학습 시 표본 0개 → 상수 fallback 이 작동해
    crash 없이 유한한 q̂ 을 내야 한다(실로그의 희소 action 은 흔하다 — 축 12 fallback 관례).
    pscore 쪽은 결측 클래스 열이 정확히 0 이 되고 p̂ 은 PS_CLIP_MIN 하한으로 방어돼야 한다."""
    rng = np.random.default_rng(7)
    n, k = 60, 3
    x = rng.normal(size=(n, 2))
    a = rng.integers(0, 2, size=n)  # 기본은 {0, 1}
    cfg = CrossFitConfig(n_folds=2, seed=0)
    folds = _fold_indices(n, cfg.n_folds, cfg.seed)
    a[folds[0][:5]] = 2             # action 2 는 fold 0 에만 존재
    r = rng.normal(size=n)

    q = fit_q_hat_crossfit(x, a, r, k, cfg)
    assert q.shape == (n, k)
    assert np.all(np.isfinite(q))
    assert np.all(np.isfinite(q[:, 2]))

    p_hat, pi0 = fit_pscore_crossfit(x, a, k, cfg)
    np.testing.assert_allclose(pi0.sum(axis=1), 1.0, atol=1e-8)
    # fold 0 행의 예측 모델(train=fold 1)은 class 2 를 못 봤다 → 확률 0 열 + 하한 클립 방어
    assert np.all(pi0[folds[0], 2] == 0.0)
    assert np.all(p_hat[folds[0][:5]] == PS_CLIP_MIN)


# ------------------------------------------------------------ 4. pscore contract


def test_pscore_rows_sum_and_clip_identity(log):
    """π̂0 행합 ~1(결측 0 열 포함), p̂ ≥ PS_CLIP_MIN, 그리고 p̂ 은 정확히
    clip(π̂0[i, a_i], PS_CLIP_MIN) 이어야 한다 — 진단(ESS·max-weight) 입력과 dist 의 정합."""
    x, a, _ = log
    p_hat, pi0 = fit_pscore_crossfit(x, a, K, CFG)
    assert p_hat.shape == (N,) and pi0.shape == (N, K)
    np.testing.assert_allclose(pi0.sum(axis=1), 1.0, atol=1e-8)
    assert np.all(p_hat >= PS_CLIP_MIN)
    np.testing.assert_array_equal(p_hat, np.clip(pi0[np.arange(N), a], PS_CLIP_MIN, None))


# --------------------------------------------------------- 5. statistical sanity


def test_pscore_recovers_known_softmax_propensity():
    """참 로깅 정책이 x 의 선형 스코어 softmax 이면(= multinomial logistic 이 well-specified)
    cross-fitted LR 이 관측행동 propensity 를 높은 상관으로 회복해야 한다 — 이것이 깨지면
    M8 트랙의 π̂0 기반 IPS/DR 전체를 신뢰할 수 없다. 느슨한 하한 corr > 0.7 (n=5000)."""
    rng = np.random.default_rng(2026)
    n, k, d = 5000, 4, 3
    x = rng.normal(size=(n, d))
    theta = rng.normal(size=(d, k))
    z = x @ theta
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    a = (p.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    p_true_obs = p[np.arange(n), a]

    p_hat, _ = fit_pscore_crossfit(x, a, k, CrossFitConfig(n_folds=2, seed=0, max_iter=1000))
    corr = np.corrcoef(p_hat, p_true_obs)[0, 1]
    assert corr > 0.7, f"corr(p̂, p_true) = {corr:.3f}"


# ------------------------------------------------------------- 6. ban encoding


def test_no_oracle_leak_in_source():
    """oracle-leak 금지 명문화(test_business.test_no_confounding_knob 정신): fitters.py 는
    로그 (x, a, r)만 입력으로 받아야 하며, DGP 내부 참값에 접근하는 import·식별자가 소스
    텍스트에 나타나면 안 된다 — GT-미상 트랙의 격리 계약을 소스 수준에서 고정한다."""
    src = Path(fitters.__file__).read_text(encoding="utf-8")
    banned = [
        "from ope.dgp", "import ope.dgp",
        "q_true", "pscore_true", "v_true", "gt_value",
        # 모듈 스펙의 금지 의존 전체도 소스 수준에서 함께 고정
        "from ope.datasets", "from ope.business", "from ope.diagnostics", "from ope.estimators",
    ]
    for s in banned:
        assert s not in src, f"banned string in fitters.py source: {s!r}"
