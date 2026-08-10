"""M8 frontstage/backstage 분리 계약 tests — 스키마 ban·소스 ban·blindness·규칙 고정.

frontstage 가 oracle 을 만질 수 없음을 **네 겹**으로 고정한다(PLAN §3.5 M8 설계 결정):
① DECISION_COLUMNS 에 oracle 컬럼 부재(스키마 ban) ② validity/fitters 소스 텍스트에 oracle
식별자 부재(소스 ban) ③ oracle 필드를 NaN 으로 오염시켜도 run_protocol 산출 불변(blindness —
실행 수준 증명) ④ reveal 은 파일 경유 + oracle 컬럼 혼입 시 거부.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))

from conftest import BASE  # noqa: E402
from ope.dgp import make_synthetic_bandit_data  # noqa: E402
import ope.dgp as dgp_module  # noqa: E402
from ope.validity import ValidityConfig  # noqa: E402
import _practitioner as pr  # noqa: E402

ORACLE_TOKENS = ("v_true", "q_true", "pscore_true", "gt_value")


def _make_log(d) -> pr.PractitionerLog:
    return pr.PractitionerLog(context=d.context, action=d.action, reward=d.reward,
                              pscore=d.pscore_logged, pi_log_dist=d.pi_log_dist)


def test_decision_columns_ban_oracle():
    """스키마 ban: frontstage CSV 스키마에 oracle 컬럼이 하나라도 있으면 계약 위반."""
    for tok in ORACLE_TOKENS:
        assert tok not in pr.DECISION_COLUMNS
    # reveal(백스테이지)에는 truth 가 있어야 정상이다 — 역방향 확인
    assert "v_true" in pr.REVEAL_COLUMNS and "truth_kind" in pr.REVEAL_COLUMNS


def test_source_ban_validity_and_fitters():
    """소스 ban: 로그-층 전용 모듈(validity·fitters)의 소스 텍스트에 oracle 식별자·dgp import
    가 존재하지 않는다 (test_no_confounding_knob 의 ban-encoding 패턴)."""
    for mod in ("validity.py", "fitters.py"):
        text = (ROOT / "src" / "ope" / mod).read_text(encoding="utf-8")
        for tok in ORACLE_TOKENS + ("from ope.dgp", "import ope.dgp"):
            assert tok not in text, f"{mod} 에 금지 토큰 {tok!r}"


def test_blindness_protocol_ignores_oracle(data, monkeypatch):
    """blindness: oracle 필드를 전부 NaN 으로 오염 + true_policy_value 를 폭파시켜도
    run_protocol 산출이 동일 — frontstage 는 실행 수준에서 참값 비접촉."""
    cfg = ValidityConfig(n_boot=30, seed=5)
    rows_clean = pr.run_protocol(_make_log(data), data.pi_e_dist, axis_id="00",
                                 scenario="t", run_id="r0", seed=5, cfg=cfg)

    poisoned = data._replace(v_true=float("nan"),
                             q_true=np.full_like(data.q_true, np.nan),
                             pscore_true=np.full_like(data.pscore_true, np.nan))

    def _boom(*a, **k):
        raise AssertionError("frontstage 가 true_policy_value 를 호출했다 — blindness 위반")

    monkeypatch.setattr(dgp_module, "true_policy_value", _boom)
    rows_poisoned = pr.run_protocol(_make_log(poisoned), poisoned.pi_e_dist,
                                    axis_id="00", scenario="t", run_id="r0",
                                    seed=5, cfg=cfg)
    # dict 동등 비교는 NaN(hyperparam_value 등)에서 항상 깨진다 — NaN-aware 비교
    assert len(rows_clean) == len(rows_poisoned)
    for a, b in zip(rows_clean, rows_poisoned):
        assert a.keys() == b.keys()
        for k in a:
            va, vb = a[k], b[k]
            both_nan = (isinstance(va, float) and isinstance(vb, float)
                        and np.isnan(va) and np.isnan(vb))
            assert both_nan or va == vb, f"blindness 위반 필드 {k}: {va!r} != {vb!r}"


def test_protocol_verdict_and_decide_rules():
    """§3.5-2 결합·결정 규칙의 진리표 고정 — ab_fallback 우선 포함."""
    class _Arm:
        def __init__(self, state):
            self.state = state

    class _Rep:
        def __init__(self, harmonic="pass", mean_w="pass", placebo="pass",
                     disagreement="pass"):
            self.harmonic, self.mean_w = _Arm(harmonic), _Arm(mean_w)
            self.placebo, self.disagreement = _Arm(placebo), _Arm(disagreement)

    assert pr.protocol_verdict("trust", _Rep()) == "trust"
    assert pr.protocol_verdict("ab_fallback", _Rep()) == "ab_fallback"
    assert pr.protocol_verdict("trust", _Rep(harmonic="fail")) == "ab_fallback"
    assert pr.protocol_verdict("distrust", _Rep()) == "distrust"
    for arm in ("mean_w", "placebo", "disagreement"):
        assert pr.protocol_verdict("trust", _Rep(**{arm: "fail"})) == "distrust"
    # 동시 성립 → ab_fallback 우선 (§3.5-2)
    assert pr.protocol_verdict("distrust", _Rep(harmonic="fail")) == "ab_fallback"
    # inconclusive 는 fail 이 아니다 (§3.5-1 스케일 바닥)
    assert pr.protocol_verdict("trust", _Rep(disagreement="inconclusive")) == "trust"

    assert pr.decide("trust", 0.6, 0.8, 0.5) == "go"
    assert pr.decide("trust", 0.2, 0.4, 0.5) == "no_go"
    assert pr.decide("trust", 0.4, 0.6, 0.5) == "ab_test"
    assert pr.decide("distrust", 0.6, 0.8, 0.5) == "ab_test"
    assert pr.decide("ab_fallback", 0.6, 0.8, 0.5) == "ab_test"


def test_split_log_partition():
    """split_log 는 무결 분할: 서로소·합집합 완전·비율 정확·seed 재현."""
    a1, b1 = pr.split_log(100, 0.3, seed=1)
    a2, b2 = pr.split_log(100, 0.3, seed=1)
    np.testing.assert_array_equal(a1, a2)
    np.testing.assert_array_equal(b1, b2)
    assert len(a1) == 30 and len(b1) == 70
    assert len(np.intersect1d(a1, b1)) == 0
    assert len(np.union1d(a1, b1)) == 100
    with pytest.raises(ValueError):
        pr.split_log(100, 1.0, seed=1)


def test_run_protocol_schema(data):
    """run_protocol 행의 key 집합 == DECISION_COLUMNS 정확 일치(스키마 드리프트 방지) —
    q̂ 미제공 시 weighting 3종, 제공 시 7종."""
    cfg = ValidityConfig(n_boot=20, seed=2)
    log = _make_log(data)
    rows3 = pr.run_protocol(log, data.pi_e_dist, axis_id="00", scenario="t",
                            run_id="r", seed=2, cfg=cfg)
    assert len(rows3) == 3
    assert set(rows3[0].keys()) == set(pr.DECISION_COLUMNS)
    q_hat = 0.9 * data.q_true + 0.05  # 테스트 전용 oracle 파생 — frontstage 밖
    rows7 = pr.run_protocol(log, data.pi_e_dist, axis_id="00", scenario="t",
                            run_id="r", seed=2, q_hat=q_hat, cfg=cfg)
    assert len(rows7) == 7
    assert {r["estimator"] for r in rows7} == {
        "ips", "snips", "clipped_ips", "dm", "dr", "switch_dr", "dros"}


def test_reveal_file_path_enforced_and_scored(tmp_path, data):
    """reveal 은 ① 파일 경유만 ② oracle 혼입 frontstage 거부 ③ truth_kind='none' 거부
    ④ rel_err·large_err·ci_covers 채점 정확."""
    cfg = ValidityConfig(n_boot=20, seed=2)
    rows = pr.run_protocol(_make_log(data), data.pi_e_dist, axis_id="00",
                           scenario="t", run_id="r", seed=2, cfg=cfg)
    csv = tmp_path / "00_test_decision.csv"
    pd.DataFrame(rows)[pr.DECISION_COLUMNS].to_csv(csv, index=False)

    truth = pd.DataFrame({"run_id": ["r"], "v_true": [data.v_true]})
    out = pr.reveal(csv, truth, "exact_synthetic")
    assert out.name == "00_test_reveal.csv" and out.exists()
    rev = pd.read_csv(out)
    assert list(rev.columns) == pr.REVEAL_COLUMNS
    row = rev[rev.estimator == "snips"].iloc[0]
    expect = abs(row.estimate - data.v_true) / abs(data.v_true)
    assert row.rel_err == pytest.approx(expect, rel=1e-9)
    assert bool(row.large_err) == (expect > pr.LARGE_ERR)
    assert bool(row.ci_covers_truth) == (row.ci_lo <= data.v_true <= row.ci_hi)

    with pytest.raises(ValueError):
        pr.reveal(csv, truth, "none")
    bad = pd.read_csv(csv)
    bad["v_true"] = 0.0
    bad_csv = tmp_path / "00_bad_decision.csv"
    bad.to_csv(bad_csv, index=False)
    with pytest.raises(ValueError):
        pr.reveal(bad_csv, truth, "exact_synthetic")
