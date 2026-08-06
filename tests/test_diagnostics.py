import numpy as np
import pytest

from ope.diagnostics import PROVISIONAL_THRESHOLDS, compute_diagnostics, decision_gate
from ope.estimators import estimate_clipped_ips, estimate_ips


def test_uniform_weights_ess_ratio_one(data):
    n = len(data.action)
    rep = compute_diagnostics(data.pscore_logged, data.action, data.pi_e_dist,
                              weights=np.ones(n))
    assert rep.ess == pytest.approx(n)
    assert rep.ess_ratio == pytest.approx(1.0)


def test_ess_bounds_and_raw_recompute(data):
    rep_none = compute_diagnostics(data.pscore_logged, data.action, data.pi_e_dist)
    rep_empty = compute_diagnostics(data.pscore_logged, data.action, data.pi_e_dist,
                                    weights=np.empty(0))
    assert rep_none == rep_empty  # 빈 배열 = None 취급 (DM weights 가드)
    assert 0.0 < rep_none.ess <= len(data.action)
    assert rep_none.max_weight >= rep_none.weight_tail_p99


def test_clipping_raises_ess(data):
    raw = estimate_ips(data.reward, data.action, data.pscore_logged, data.pi_e_dist).weights
    clipped = estimate_clipped_ips(data.reward, data.action, data.pscore_logged,
                                   data.pi_e_dist, lam=np.quantile(raw, 0.9)).weights
    ess_raw = compute_diagnostics(data.pscore_logged, data.action, data.pi_e_dist, raw).ess
    ess_clip = compute_diagnostics(data.pscore_logged, data.action, data.pi_e_dist, clipped).ess
    assert ess_clip >= ess_raw


def test_support_proxy_detects_never_logged_action():
    n, k = 200, 4
    rng = np.random.default_rng(11)
    action = rng.integers(0, 3, size=n)          # 액션 3 은 로그에 0회
    pscore = np.full(n, 1.0 / 3.0)
    pi_e = np.full((n, k), 0.25)                  # π_e 는 액션 3 에 질량 0.25
    rep = compute_diagnostics(pscore, action, pi_e)
    assert rep.support_deficiency == pytest.approx(0.25)


def test_gate_three_way():
    th = PROVISIONAL_THRESHOLDS
    base = dict(ess=1000.0, ess_ratio=0.5, max_weight=3.0, weight_tail_p99=2.0,
                support_deficiency=0.0)
    from ope.diagnostics import DiagnosticsReport
    assert decision_gate(DiagnosticsReport(**base), th).decision == "trust"
    v = decision_gate(DiagnosticsReport(**{**base, "ess_ratio": 0.005}), th)
    assert v.decision == "ab_fallback" and v.reasons
    v = decision_gate(DiagnosticsReport(**{**base, "max_weight": 1e4}), th)
    assert v.decision == "distrust"
    v = decision_gate(DiagnosticsReport(**{**base, "support_deficiency": 0.3}), th)
    assert v.decision == "distrust"


def test_gate_thresholds_contract():
    """raw dict 는 오타 키를 침묵 무시했었다(M1 적대 리뷰 확정 결함) — NamedTuple 강제로 즉사시킨다."""
    from ope.diagnostics import DiagnosticsReport, GateThresholds
    rep = DiagnosticsReport(ess=1.0, ess_ratio=1.0, max_weight=1.0, weight_tail_p99=1.0,
                            support_deficiency=0.0)
    with pytest.raises(TypeError):
        decision_gate(rep, {"ess_ratio_hard": 0.01})  # dict 금지
    with pytest.raises(TypeError):
        decision_gate(rep, dict(PROVISIONAL_THRESHOLDS._asdict()))  # 전체 키여도 dict 금지
    with pytest.raises(TypeError):
        GateThresholds(ess_ratio_hard=0.01)  # 누락 키 즉사
    assert isinstance(PROVISIONAL_THRESHOLDS, GateThresholds)
