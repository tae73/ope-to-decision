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
