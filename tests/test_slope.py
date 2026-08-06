import numpy as np
import pytest

from conftest import BASE
from ope.dgp import make_synthetic_bandit_data, true_policy_value
from ope.estimators import (
    _lepski_select,
    estimate_switch_dr,
    slope_select,
)


# ── 선택 규칙 단독 (crafted 배열 — 구현 복제 아닌 규칙 자체 검증) ─────────────
def test_lepski_all_overlap_picks_last():
    values = np.array([1.0, 1.01, 0.99, 1.02])
    ses = np.array([0.5, 0.3, 0.2, 0.1])
    assert _lepski_select(values, ses) == 3


def test_lepski_jump_blocks_later_indices():
    # j=2 가 i=0 과 크게 어긋나면 j=2·3 모두 (i=0 과의 pairwise 조건으로) 탈락 → ĵ=1
    values = np.array([1.0, 1.05, 9.0, 9.05])
    ses = np.array([0.10, 0.08, 0.05, 0.04])
    assert _lepski_select(values, ses) == 1


def test_lepski_requires_all_predecessors_not_only_adjacent():
    # 인접쌍(1↔2)은 교차하지만 0↔2 는 어긋남 — 인접-only 버그 구현이면 2 를 고르는 함정
    values = np.array([0.0, 1.0, 1.9])
    ses = np.array([0.20, 0.40, 0.20])
    # j=1: |1-0|=1.0 ≤ 2(0.4+0.2)=1.2 → OK. j=2: 인접 |1.9-1|=0.9 ≤ 1.2 OK 지만
    # 0↔2: |1.9| > 2(0.2+0.2)=0.8 → 탈락. 따라서 ĵ=1.
    assert _lepski_select(values, ses) == 1


def test_lepski_single_candidate():
    assert _lepski_select(np.array([1.0]), np.array([0.1])) == 0


# ── slope_select 통합 ─────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def harsh_data():
    # 축 07 과 같은 가혹 config: β_log=8 → heavy weight tail
    cfg = BASE._replace(beta_log=8.0, n=10_000, seed=42)
    return cfg, make_synthetic_bandit_data(cfg)


def test_slope_validation(harsh_data):
    _, d = harsh_data
    args = (d.reward, d.action, d.pscore_logged, d.pi_e_dist)
    with pytest.raises(ValueError):
        slope_select(*args, q_hat=None, estimator="ips", ladder=np.array([1.0, 2.0]))
    with pytest.raises(ValueError):
        slope_select(*args, q_hat=None, estimator="clipped_ips", ladder=np.array([2.0, 1.0]))
    with pytest.raises(ValueError):
        slope_select(*args, q_hat=None, estimator="switch_dr", ladder=np.array([1.0, 2.0]))


def test_slope_selected_value_matches_ladder_point(harsh_data):
    cfg, d = harsh_data
    q_hat = 0.9 * d.q_true + 0.05
    ladder = np.geomspace(0.5, 50.0, 8)
    res = slope_select(d.reward, d.action, d.pscore_logged, d.pi_e_dist, q_hat,
                       estimator="switch_dr", ladder=ladder)
    direct = estimate_switch_dr(d.reward, d.action, d.pscore_logged, d.pi_e_dist, q_hat,
                                tau=res.hyperparam).value
    assert res.value == pytest.approx(direct, rel=1e-12)
    assert 0 <= res.index < len(ladder)
    assert res.values.shape == res.ses.shape == ladder.shape


def test_slope_clipped_not_anchored_to_most_biased_rung(harsh_data):
    """축 07 실증 회귀 — 방향 반전 버그(초기 구현): 오름차순으로 Lepski 를 걸으면 최강 클리핑
    rung(고편향·협폭 CI)이 전부를 거부해 30/30 seed 가 index 0 으로 붕괴, 상대오차 ~15배.
    수정 후: 광폭(미규제) 끝에서 출발 → 최소 rung 대비 평균 오차가 확실히 작아야 한다."""
    from ope.estimators import estimate_ips
    cfg, _ = harsh_data
    v = true_policy_value(cfg)
    errs_slope, errs_minrung, idxs = [], [], []
    for s in range(400, 415):
        d = make_synthetic_bandit_data(cfg._replace(seed=s))
        w = estimate_ips(d.reward, d.action, d.pscore_logged, d.pi_e_dist).weights
        ladder = np.geomspace(np.median(w), 2 * w.max(), 8)
        res = slope_select(d.reward, d.action, d.pscore_logged, d.pi_e_dist, None,
                           estimator="clipped_ips", ladder=ladder)
        errs_slope.append(abs(res.value - v))
        errs_minrung.append(abs(res.values[0] - v))
        idxs.append(res.index)
    assert np.mean(errs_slope) < 0.5 * np.mean(errs_minrung)
    assert np.mean(idxs) > 1.0  # 전 seed 가 최소 rung 에 앵커되는 붕괴 재발 방지


def test_slope_beats_worst_ladder_point_on_average(harsh_data):
    """약한 sanity: S seed 평균에서 SLOPE 선택의 |오차| ≤ ladder 최악 점의 |오차|."""
    cfg, _ = harsh_data
    v = true_policy_value(cfg)
    ladder = np.geomspace(0.5, 200.0, 8)
    err_slope, err_worst = [], []
    for s in range(300, 320):
        d = make_synthetic_bandit_data(cfg._replace(seed=s))
        q_hat = 0.9 * d.q_true + 0.05
        res = slope_select(d.reward, d.action, d.pscore_logged, d.pi_e_dist, q_hat,
                           estimator="switch_dr", ladder=ladder)
        err_slope.append(abs(res.value - v))
        err_worst.append(np.max(np.abs(res.values - v)))
    assert np.mean(err_slope) <= np.mean(err_worst)
