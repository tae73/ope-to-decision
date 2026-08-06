"""PROBE M0-B — obp 교차검증 GO/NO-GO (pinned Python 3.9 env 안에서 실행)

WHAT GENERALIZES:
    GO 면 "자기 numpy 구현을 표준 라이브러리(obp)로 적대 교차검증한다"는 M1 게이트 전략이
    실행 가능하다. 데이터 생성과 두 구현(자기 구현 vs obp)이 *같은 env 안에서 같은 배열*을
    받으므로 순수 산술 비교다(난수 스트림 cross-version 문제 없음).

THE RESULT (boxed — 수치 정본은 results/tables/probe_obp_crossval.json):
    ┌────────────────────────────────────────────────────────────┐
    │ 코어 4종(DM·IPS·SNIPS·DR): 상대차 ≤ 1e-8 이면 GO           │
    │ Switch-DR·DRos·Clipped: 일치/불일치·API 차이를 기록(참고)  │
    └────────────────────────────────────────────────────────────┘

HONEST reduces_check:
    - obp 설치 자체가 실패하면 이 probe 는 NO-GO 가 아니라 "INSTALL-FAIL"이며, PLAN.md 의
      폴백(소스 설치 → sb-obp → 논문 수치 재현·property test 대체)으로 넘어간다.
    - Switch-DR·DRos 는 obp 구현의 하이퍼파라미터 정의가 다를 수 있어 코어 게이트에서 제외,
      불일치는 공식 차이 규명 대상으로만 기록한다.

VERDICT: stdout 및 JSON "verdict" — GO / NO-GO / INSTALL-FAIL(별도 기록).

주의: 이 파일은 Python 3.9 호환 문법만 사용한다(obp pinned env 전용).
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
_SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""  # sb-obp 트랙은 "_sbobp" 로 별도 JSON
OUT = ROOT / "results" / "tables" / "probe_obp_crossval{}.json".format(_SUFFIX)

D, K, N = 5, 10, 50_000
BETA_LOG, BETA_EVAL = 2.0, 6.0
SEED = 20260806
TAU_SWITCH = 2.0  # 이 DGP 의 w_max≈2.97, P(w>2)≈7.5% — switch 분기가 실제로 발동하도록 설정
LAM_DROS = 5.0
CORE = ["dm", "ips", "snips", "dr"]
REL_TOL = 1e-8


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits, beta):
    z = beta * logits
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def make_data():
    rng = np.random.default_rng(SEED)
    theta = rng.normal(0.0, 1.0, size=(K, D))
    b = rng.normal(0.0, 0.5, size=K)
    x = rng.normal(size=(N, D))
    q = sigmoid(x @ theta.T + b)
    pi0 = softmax(q, BETA_LOG)
    pi_e = softmax(q, BETA_EVAL)
    a = (pi0.cumsum(axis=1) > rng.random((N, 1))).argmax(axis=1)
    idx = np.arange(N)
    r = rng.binomial(1, q[idx, a]).astype(float)
    ps = pi0[idx, a]
    q_hat = 0.5 * q + 0.25  # 고의 오지정 reward 모델(비교는 산술 일치가 목적)
    return x, a, r, ps, pi_e, q_hat


def my_estimates(a, r, ps, pi_e, q_hat):
    idx = np.arange(len(a))
    w = pi_e[idx, a] / ps
    dm = float((pi_e * q_hat).sum(axis=1).mean())
    resid = r - q_hat[idx, a]
    w_switch = w * (w <= TAU_SWITCH)
    w_shrink = (LAM_DROS * w) / (w**2 + LAM_DROS)
    return {
        "dm": dm,
        "ips": float((w * r).mean()),
        "snips": float((w * r).sum() / w.sum()),
        "dr": float(dm + (w * resid).mean()),
        "switch_dr": float(dm + (w_switch * resid).mean()),
        "dros": float(dm + (w_shrink * resid).mean()),
    }


def obp_estimates(a, r, ps, pi_e, q_hat):
    """obp 0.5.x API. 실패한 estimator 는 값 대신 오류 문자열 기록."""
    import obp  # noqa: F401  (버전 기록용)
    from obp.ope import (
        DirectMethod,
        DoublyRobust,
        DoublyRobustWithShrinkage,
        InverseProbabilityWeighting,
        SelfNormalizedInverseProbabilityWeighting,
        SwitchDoublyRobust,
    )

    n = len(a)
    position = np.zeros(n, dtype=int)
    action_dist = pi_e[:, :, np.newaxis]
    erm = q_hat[:, :, np.newaxis]
    common = dict(reward=r, action=a, pscore=ps, action_dist=action_dist, position=position)

    def build(name):
        # 생성자도 try 안에서: 버전별 API 차이(예: switch 임계값 tau vs lambda_)를 기록으로 남긴다
        if name == "dm":
            return DirectMethod(), dict(action_dist=action_dist, position=position,
                                        estimated_rewards_by_reg_model=erm)
        if name == "ips":
            return InverseProbabilityWeighting(), dict(common)
        if name == "snips":
            return SelfNormalizedInverseProbabilityWeighting(), dict(common)
        if name == "dr":
            return DoublyRobust(), dict(common, estimated_rewards_by_reg_model=erm)
        if name == "switch_dr":
            try:
                est = SwitchDoublyRobust(lambda_=TAU_SWITCH)  # obp 0.5.x 표기
            except TypeError:
                est = SwitchDoublyRobust(tau=TAU_SWITCH)  # 구버전 표기
            return est, dict(common, estimated_rewards_by_reg_model=erm)
        if name == "dros":
            return DoublyRobustWithShrinkage(lambda_=LAM_DROS), dict(common, estimated_rewards_by_reg_model=erm)
        raise ValueError(name)

    out = {}
    for name in ["dm", "ips", "snips", "dr", "switch_dr", "dros"]:
        try:
            estimator, kwargs = build(name)
            out[name] = float(estimator.estimate_policy_value(**kwargs))
        except Exception as exc:  # noqa: BLE001 — API 차이를 그대로 기록
            out[name] = "ERROR: {}: {}".format(type(exc).__name__, exc)
    return out, obp.__version__


def main():
    x, a, r, ps, pi_e, q_hat = make_data()
    mine = my_estimates(a, r, ps, pi_e, q_hat)

    payload = {
        "probe": "M0-B obp_crossval",
        "env": {"python": sys.version.split()[0], "platform": platform.platform(),
                "numpy": np.__version__},
        "config": {"d": D, "k": K, "n": N, "beta_log": BETA_LOG, "beta_eval": BETA_EVAL,
                   "seed": SEED, "tau_switch": TAU_SWITCH, "lam_dros": LAM_DROS,
                   "core_estimators": CORE, "rel_tol": REL_TOL},
        "mine": mine,
    }
    try:
        theirs, obp_version = obp_estimates(a, r, ps, pi_e, q_hat)
        payload["obp_version"] = obp_version
        payload["obp"] = theirs
        comparison = {}
        for name, v_mine in mine.items():
            v_obp = theirs.get(name)
            if isinstance(v_obp, float):
                rel = abs(v_mine - v_obp) / max(abs(v_mine), 1e-12)
                comparison[name] = {"mine": v_mine, "obp": v_obp, "rel_diff": rel,
                                    "match": bool(rel <= REL_TOL)}
            else:
                comparison[name] = {"mine": v_mine, "obp": v_obp, "rel_diff": None, "match": False}
        payload["comparison"] = comparison
        core_ok = all(comparison[nm]["match"] for nm in CORE)
        payload["verdict"] = "GO" if core_ok else "NO-GO"
    except ImportError as exc:
        payload["verdict"] = "INSTALL-FAIL"
        payload["error"] = str(exc)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print("┌" + "─" * 60 + "┐")
    for name, v in mine.items():
        line = "│ {:10s} mine={:.10f}".format(name, v)
        if "comparison" in payload:
            comp = payload["comparison"][name]
            line += "  obp={}  match={}".format(comp["obp"], comp["match"])
        print(line)
    print("│ VERDICT: {}".format(payload["verdict"]))
    print("└" + "─" * 60 + "┘")


if __name__ == "__main__":
    main()
