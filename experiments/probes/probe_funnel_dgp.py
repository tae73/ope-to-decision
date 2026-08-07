"""PROBE M6 — funnel DGP 의 기저율-분리력 상충 GO/NO-GO.

WHAT GENERALIZES:
    비즈니스 층(M6)의 make-or-break: CTR 을 현실적 저기저율로 만들면 softmax(β·q) 정책 분리가
    압축된다(Plan 검증 지적). 해소 설계 = **정책은 relevance 스코어 s∈(0,1) 위에서 작동**(코어와
    동일 span·β), CTR 은 p_c = base + amp·s 로 캘리브레이션 — 기저율과 분리력의 구조적 분리.
    GO 면 이 설계로 business.py 를 구현한다.

THE RESULT (boxed — 정본 results/tables/probe_funnel_dgp.json):
    ┌──────────────────────────────────────────────────────────────┐
    │ (a) CTR 기저율 ∈ [0.02, 0.06]  (b) 같은-로그 Δ̂_CTR 판별:      │
    │ |Δ_true| > 2.8·SE(Δ̂) @ n=10k  (c) ESS ratio ∈ [0.1, 0.95]    │
    │ (d) conversion 이벤트 ≥ 30 @ n=10k (REV 사다리 재료 존재)      │
    └──────────────────────────────────────────────────────────────┘

HONEST reduces_check: 판정은 base=0.01·amp=0.06·conv(0.05+0.25s')·β=(2,6)·struct_seed=7 조건부.
REV 판별 '가능성'은 probe 기준이 아니다 — 그 급감 자체가 축 15 의 측정 대상.

VERDICT: stdout + JSON.
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "probe_funnel_dgp.json"
D, K, N, S = 5, 10, 10_000, 20
BASE_CTR, AMP_CTR = 0.01, 0.06
BASE_CVR, AMP_CVR = 0.05, 0.25
B_LOG, B_EVAL = 2.0, 6.0
STRUCT = 7


def sigmoid(z):
    return np.exp(-np.logaddexp(0.0, -z))


def softmax(v, b):
    z = b * v - (b * v).max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    rs = np.random.default_rng(STRUCT)
    th_c, b_c = rs.normal(0, 1, (K, D)), rs.normal(0, 0.5, K)
    th_v, b_v = rs.normal(0, 1, (K, D)), rs.normal(0, 0.5, K)

    # ground truth (200k MC)
    xm = np.random.default_rng(1_000_003 + STRUCT).normal(size=(200_000, D))
    s = sigmoid(xm @ th_c.T + b_c)
    pc = BASE_CTR + AMP_CTR * s
    pe, p0 = softmax(s, B_EVAL), softmax(s, B_LOG)
    v_ctr_e, v_ctr_0 = float((pe * pc).sum(1).mean()), float((p0 * pc).sum(1).mean())
    delta_true = v_ctr_e - v_ctr_0

    deltas, esss, convs = [], [], []
    for seed in range(700, 700 + S):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(N, D))
        sv = sigmoid(x @ th_c.T + b_c)
        pcv = BASE_CTR + AMP_CTR * sv
        pvv = BASE_CVR + AMP_CVR * sigmoid(x @ th_v.T + b_v)
        pi0, pie = softmax(sv, B_LOG), softmax(sv, B_EVAL)
        a = (pi0.cumsum(1) > rng.random((N, 1))).argmax(1)
        i = np.arange(N)
        click = rng.binomial(1, pcv[i, a]).astype(float)
        conv = click * rng.binomial(1, pvv[i, a])
        w = pie[i, a] / pi0[i, a]
        # 같은-로그 비교형 Δ̂ = IPS(π_e) − mean(r) — 공통 오차 상쇄(축 10 원리)
        deltas.append(float((w * click).mean() - click.mean()))
        esss.append(float(w.sum() ** 2 / (w**2).sum() / N))
        convs.append(int(conv.sum()))

    se_d = float(np.std(deltas, ddof=1) / np.sqrt(S)) * np.sqrt(S)  # 단일 로그 SE ≈ sd over seeds
    checks = {
        "ctr_base_in_range": bool(0.02 <= v_ctr_0 <= 0.06),
        "delta_detectable": bool(abs(delta_true) > 2.8 * se_d),
        "ess_healthy": bool(0.1 <= float(np.mean(esss)) <= 0.95),
        "conv_events_ge_30": bool(min(convs) >= 30),
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"
    payload = {"probe": "M6 funnel_dgp",
               "config": {"base_ctr": BASE_CTR, "amp_ctr": AMP_CTR, "base_cvr": BASE_CVR,
                          "amp_cvr": AMP_CVR, "beta": [B_LOG, B_EVAL], "n": N, "s_seeds": S},
               "v_ctr_pi0": v_ctr_0, "v_ctr_pie": v_ctr_e, "delta_true": delta_true,
               "delta_hat_sd_over_seeds": float(np.std(deltas, ddof=1)),
               "ess_ratio_mean": float(np.mean(esss)), "conv_events_min": min(convs),
               "checks": checks, "verdict": verdict}
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"[probe M6] V_ctr(π0)={v_ctr_0:.4f} V_ctr(πe)={v_ctr_e:.4f} Δ={delta_true:.4f} "
          f"sd(Δ̂)={np.std(deltas, ddof=1):.4f} ESS={np.mean(esss):.3f} conv_min={min(convs)}")
    print(f"[probe M6] checks={checks} VERDICT: {verdict}")


if __name__ == "__main__":
    main()
