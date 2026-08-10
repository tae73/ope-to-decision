"""c2b 변환 속성 테스트 — optdigits 만 사용(최초 1회 네트워크 후 data/openml 캐시로 오프라인).

나머지 3종(satimage·pendigits·letter)은 축 11 실행이 커버한다(여기서 4종 전부 fetch 하면
테스트가 네트워크에 과의존).
"""

import numpy as np
import pytest

from ope.datasets import classification_to_bandit


@pytest.fixture(scope="module")
def c2b():
    return classification_to_bandit("optdigits", seed=1000)


def test_shapes_and_exact_propensity(c2b):
    n, k = c2b.pi_e_dist.shape
    assert len(c2b.action) == len(c2b.reward) == len(c2b.pscore) == n
    assert c2b.n_actions == k == 10
    assert np.all(c2b.pscore > 0) and np.all(c2b.pscore <= 1)
    assert np.allclose(c2b.pi_e_dist.sum(axis=1), 1.0)
    assert set(np.unique(c2b.reward)) <= {0.0, 1.0}  # 결정적 reward


def test_gt_exact_and_in_range(c2b):
    assert 0.0 < c2b.gt_value < 1.0
    assert c2b.gt_ci is None  # c2b 참값은 정확 — CI 없음(의미 분리)


def test_structure_fixed_log_varies():
    d1 = classification_to_bandit("optdigits", seed=1000)
    d2 = classification_to_bandit("optdigits", seed=1001)
    np.testing.assert_allclose(d1.pi_e_dist, d2.pi_e_dist)  # 구조(정책) 동일
    assert d1.gt_value == d2.gt_value
    assert not np.array_equal(d1.action, d2.action)          # 로그만 다름


def test_degraded_qhat_is_worse_model(c2b):
    """모델 품질은 보상 예측 Brier score 로 비교한다 — 로깅 행동의 log-likelihood 는 정확도가
    아니라 확신도(good LR 의 near-0/1 확률)에 지배돼 잘못된 척도(테스트 자체의 초기 버그)."""
    idx = np.arange(len(c2b.action))
    good_brier = ((c2b.q_scores[idx, c2b.action] - c2b.reward) ** 2).mean()
    bad_brier = ((c2b.q_scores_degraded[idx, c2b.action] - c2b.reward) ** 2).mean()
    assert bad_brier > good_brier


# ── M9 추가분 (PLAN §3.6-2 — support_deficiency 주입) ─────────────────────────


def test_c2b_delta_zero_bit_identity_checksums():
    """c2b 동결 배리어(M9): δ keyword 추가 후에도 δ=0 산출은 M9 이전과 bit-identical —
    대표 config 의 checksum 리터럴 고정(dgp checksum 배리어 선례). δ=0 경로는 renormalize
    분기 자체를 건너뛴다(부동소수 재정규화 회피 — datasets.py 주석)."""
    d = classification_to_bandit("optdigits", seed=5)
    d0 = classification_to_bandit("optdigits", seed=5, support_deficiency=0.0)
    assert int(d.action.sum()) == 12510
    assert float(d.reward.sum()) == pytest.approx(1184.0, rel=1e-12)
    assert float(d.pscore.sum()) == pytest.approx(631.9309024110061, rel=1e-12)
    assert float(d.gt_value) == pytest.approx(0.9379271525435664, rel=1e-12)
    np.testing.assert_array_equal(d.action, d0.action)
    np.testing.assert_array_equal(d.pscore, d0.pscore)


def test_c2b_support_mask_helper():
    """mask 순수 헬퍼: 하위-s ⌊δK⌋ 제거·행당 정확 개수·경계 검증(0 항등·전제거 거부)."""
    from ope.datasets import _c2b_support_mask

    rng = np.random.default_rng(3)
    s = rng.uniform(size=(50, 5))
    mask = _c2b_support_mask(s, 0.4)  # ⌊0.4·5⌋ = 2 제거
    assert (~mask).sum(axis=1).tolist() == [2] * 50
    removed = np.where(~mask, s, np.inf)
    kept_min = np.where(mask, s, np.inf).min(axis=1)
    assert np.all(removed.min(axis=1) <= kept_min)  # 제거된 것이 하위-s
    assert _c2b_support_mask(s, 0.0).all()
    with pytest.raises(ValueError):
        _c2b_support_mask(s, 1.0)


def test_c2b_injection_mechanics(c2b):
    """주입 역학(δ=0.4): masked 행동 로그 무출현 · π₀ 행합 1 · pscore>0 ·
    gt_value·π_e 불변(오염은 로깅측뿐 — probe M9-A ① 의 production 로더 판)."""
    d4 = classification_to_bandit("optdigits", seed=1000, support_deficiency=0.4)
    n, k = d4.pi_e_dist.shape
    idx = np.arange(n)
    # mask 는 반환되지 않지만 q_scores(=s)로 재구성 가능 — 하위-s 4개가 masked
    order = np.argsort(d4.q_scores, axis=1)
    masked_cols = order[:, : int(0.4 * k)]
    is_masked = np.zeros((n, k), dtype=bool)
    np.put_along_axis(is_masked, masked_cols, True, axis=1)
    assert not is_masked[idx, d4.action].any()          # masked 행동 무출현
    assert np.all(d4.pscore > 0)
    assert d4.gt_value == c2b.gt_value                   # 참값 불변
    np.testing.assert_allclose(d4.pi_e_dist, c2b.pi_e_dist)  # π_e 불변
    w = d4.pi_e_dist[idx, d4.action] / d4.pscore
    assert float(np.where(is_masked, d4.pi_e_dist, 0.0).sum(axis=1).mean()) > 0.0
    assert np.isfinite(w).all()
