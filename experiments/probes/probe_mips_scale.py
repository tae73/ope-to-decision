"""PROBE M5-13 — 대규모 액션 공간에서 cluster-MIPS 의 성립 (GO/NO-GO).

WHAT GENERALIZES: GO 면 축 13(액션 수 스케일 + MIPS) 착수. 액션→클러스터 결정적 매핑(임베딩의
최소형)에서 marginal weight w=p_e(c|x)/p_0(c|x) 가 K 폭발에서도 유계인지, 임베딩 위반(η — 클러스터
내 잔차 보상)이 만드는 bias 가 분산 절감 대비 수용 가능한지. (Saito-Joachims 2022 의 요지 재현.)

THE RESULT (boxed → results/tables/probe_mips_scale.json):
  (a) K≥500 에서 MSE(MIPS) ≤ MSE(IPS)/5  (b) K 전 구간 MSE(MIPS) < MSE(IPS)
  (c) η>0 에서 MIPS bias 존재 기록(경계 정직 — bias² < IPS 분산 절감분)

HONEST reduces_check: 클러스터=참 임베딩(oracle)·η 고정 — 실무의 학습 임베딩 오지정은 더 나쁠 수
있다(축 13 에서 η 스윕으로 전시). VERDICT: stdout + JSON.
"""

import json
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "probe_mips_scale.json"
D, C, N, S = 5, 10, 20_000, 15
ETA = 0.05
B0, BE = 1.0, 3.0


def sig(z):
    return np.exp(-np.logaddexp(0.0, -z))


def sm(v, b):
    z = b * v - (b * v).max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def run_k(k, rs):
    cl = np.arange(k) % C
    th = rs.normal(0, 1, (C, D))
    b = rs.normal(0, 0.5, C)
    da = rs.normal(0, 1, k)  # per-action 잔차(임베딩 위반 채널)
    xm = np.random.default_rng(99).normal(size=(100_000, D))
    qm = sig(xm @ th[cl].T + b[cl] + ETA * da)
    v_true = float((sm(qm, BE) * qm).sum(1).mean())
    e_ips, e_mips = [], []
    for s in range(S):
        rng = np.random.default_rng(3000 + s)
        x = rng.normal(size=(N, D))
        q = sig(x @ th[cl].T + b[cl] + ETA * da)
        p0, pe = sm(q, B0), sm(q, BE)
        a = (p0.cumsum(1) > rng.random((N, 1))).argmax(1)
        i = np.arange(N)
        r = q[i, a] + 0.3 * rng.normal(size=N)
        w = pe[i, a] / p0[i, a]
        # cluster-marginal weight
        p0c = np.zeros((N, C)); pec = np.zeros((N, C))
        for c in range(C):
            m = cl == c
            p0c[:, c] = p0[:, m].sum(1); pec[:, c] = pe[:, m].sum(1)
        wm = pec[i, cl[a]] / p0c[i, cl[a]]
        e_ips.append(float((w * r).mean())); e_mips.append(float((wm * r).mean()))
    mse = lambda v: float(np.mean((np.array(v) - v_true) ** 2))
    return {"k": k, "v_true": v_true, "mse_ips": mse(e_ips), "mse_mips": mse(e_mips),
            "bias_mips": float(np.mean(e_mips) - v_true), "max_w_ips": float(w.max()),
            "max_w_mips": float(wm.max())}


def main():
    rs = np.random.default_rng(7)
    rows = [run_k(k, rs) for k in (50, 500, 2000)]
    big = [r for r in rows if r["k"] >= 500]
    checks = {
        "mips_5x_at_large_k": all(r["mse_mips"] <= r["mse_ips"] / 5 for r in big),
        "mips_wins_everywhere": all(r["mse_mips"] < r["mse_ips"] for r in rows),
        "bias_within_var_savings": all(r["bias_mips"] ** 2 < (r["mse_ips"] - r["mse_mips"]) for r in big),
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"
    OUT.write_text(json.dumps({"probe": "M5-13 mips_scale", "eta": ETA, "rows": rows,
                               "checks": checks, "verdict": verdict}, indent=2))
    for r in rows:
        print(f"K={r['k']:5d} mse_ips={r['mse_ips']:.2e} mse_mips={r['mse_mips']:.2e} "
              f"bias_mips={r['bias_mips']:+.4f} maxw {r['max_w_ips']:.0f}->{r['max_w_mips']:.1f}")
    print(f"checks={checks} VERDICT: {verdict}")


if __name__ == "__main__":
    main()
