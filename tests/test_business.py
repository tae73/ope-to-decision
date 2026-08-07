"""M6 business.py property tests — 지표별 종단 검산·GT 해석성·가드·HHI 정확성."""

import numpy as np
import pytest

from ope.business import (
    METRICS, FunnelConfig, exposure_shares, funnel_ground_truth, hhi,
    make_funnel_bandit_data, subgroup_revenue_ips,
)
from ope.estimators import estimate_ips

CFG = FunnelConfig(n=20_000, n_actions=10, dim_context=5, beta_log=2.0, beta_eval=6.0, seed=11)


@pytest.fixture(scope="module")
def d():
    return make_funnel_bandit_data(CFG)


def test_shapes_and_funnel_nesting(d):
    n = CFG.n
    assert all(d.rewards[m].shape == (n,) for m in METRICS)
    assert set(np.unique(d.rewards["ctr"])) <= {0.0, 1.0}
    # funnel 중첩: conv ≤ click, rev>0 ⇒ conv=1
    assert np.all(d.rewards["cvr"] <= d.rewards["ctr"])
    assert np.all((d.rewards["rev"] > 0) <= (d.rewards["cvr"] == 1.0))


def test_gt_ordering_and_ranges(d):
    assert 0.0 < d.gt["cvr"] < d.gt["ctr"] < 1.0  # 세션 기준 CVR < CTR (funnel)
    assert d.gt["rev"] > 0.0


def test_on_policy_end_to_end_per_metric():
    """지표별 종단 검산: π_e 실배포 시뮬 평균 ≈ GT (4·SE)."""
    cfg = CFG._replace(n=200_000, seed=77)
    from ope.business import _rates, _structure
    from ope.policies import softmax_policy
    th_c, b_c, th_v, b_v, price, _ = _structure(cfg)
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.n, cfg.dim_context))
    s, p_c, p_v = _rates(x, th_c, b_c, th_v, b_v, cfg)
    pi_e = softmax_policy(s, cfg.beta_eval)
    a = (pi_e.cumsum(1) > rng.random((cfg.n, 1))).argmax(1)
    i = np.arange(cfg.n)
    click = rng.binomial(1, p_c[i, a]).astype(float)
    conv = click * rng.binomial(1, p_v[i, a])
    rev = conv * price[a]
    gt = funnel_ground_truth(cfg)
    for name, vec in [("ctr", click), ("cvr", conv.astype(float)), ("rev", rev)]:
        se = vec.std(ddof=1) / np.sqrt(cfg.n)
        assert abs(vec.mean() - gt[name]) <= 4 * se, name


def test_ips_recovers_gt_per_metric(d):
    for m in METRICS:
        est = estimate_ips(d.rewards[m], d.action, d.pscore, d.pi_e_dist).value
        # 단일 로그 — 느슨한 sanity(상대 40% 이내; 정밀 검정은 축 15 ensemble 소관)
        assert abs(est - d.gt[m]) / d.gt[m] < 0.4, m


def test_exposure_and_hhi_exact():
    sh_e = exposure_shares(CFG, CFG.beta_eval)
    sh_0 = exposure_shares(CFG, CFG.beta_log)
    assert sh_e.shape == (CFG.n_advertisers,)
    assert np.isclose(sh_e.sum(), 1.0) and np.isclose(sh_0.sum(), 1.0)
    assert 1.0 / CFG.n_advertisers <= hhi(sh_e) <= 1.0  # HHI 하한 = 1/G
    assert hhi(np.array([1.0, 0.0])) == 1.0


def test_subgroup_honest_return(d):
    res = subgroup_revenue_ips(d, CFG.n_advertisers)
    assert len(res) == CFG.n_advertisers
    for r in res:
        if not r.estimable:
            assert np.isnan(r.estimate)
        else:
            assert np.isfinite(r.estimate) and r.weighted_events >= 10


def test_no_confounding_knob():
    """γ 영구 금지 명문화 — FunnelConfig 에 confounding 필드가 없어야 한다."""
    assert "confounding_strength" not in FunnelConfig._fields
