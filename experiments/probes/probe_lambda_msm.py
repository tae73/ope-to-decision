"""PROBE M5-14 — MSM Λ-sweep(정규화 bound)의 multi-action 수치 안정성 (GO/NO-GO).

WHAT GENERALIZES: GO 면 축 14 착수. per-sample 배율 u_i∈[1/Λ, Λ] 의 정규화 worst-case
V±(Λ) = extremize Σ u_i w_i r_i / Σ u_i w_i 는 fractional-linear — r 정렬 후 임계 스캔(O(n log n),
cumsum 벡터화)으로 정확해. 검증 4종: ① Λ=1 ⇒ SNIPS 정확 재현 ② Λ 단조 확장 ③ n=30k 안정
④ **커버리지**: 우리 DGP 는 pscore_true 를 알므로 참 odds-보정 Λ_emp = max(w_true/w_logged 비율
왜곡) 에서 [lo,hi] 가 v_true 를 덮어야 함(합성만 가능한 정직 검증).

THE RESULT → results/tables/probe_lambda_msm.json. VERDICT: stdout + JSON.
HONEST reduces_check: 정규화(SNIPS형) bound 하나만 검증 — 비정규화·CI 동반 bound 는 범위 밖.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ope.dgp import DGPConfig, make_synthetic_bandit_data, true_policy_value  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "probe_lambda_msm.json"
CFG = DGPConfig(n=30_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                support_deficiency=0.0, reward_noise=0.3, confounding_strength=1.5,
                seed=0, struct_seed=7)


def msm_bounds(w, r, lam):
    """정렬-임계 정확해: u_i=Λ (r_i>t)·1/Λ(else) 꼴이 최적 — 전 임계 cumsum 스캔."""
    o = np.argsort(-r)
    wr, rr = w[o], r[o]
    cw_hi, cwr_hi = np.cumsum(wr), np.cumsum(wr * rr)
    tw, twr = wr.sum(), (wr * rr).sum()
    num = lam * cwr_hi + (twr - cwr_hi) / lam
    den = lam * cw_hi + (tw - cw_hi) / lam
    vals = np.concatenate([[twr / lam / (tw / lam)], num / den])
    hi = float(vals.max())
    num2 = cwr_hi / lam + (twr - cwr_hi) * lam
    den2 = cw_hi / lam + (tw - cw_hi) * lam
    lo = float(np.concatenate([[twr * lam / (tw * lam)], num2 / den2]).min())
    return lo, hi


def main():
    checks = {"lam1_is_snips": True, "monotone": True, "stable": True, "coverage": True}
    v = true_policy_value(CFG)
    cover, lam_emps = [], []
    for s in range(500, 505):
        d = make_synthetic_bandit_data(CFG._replace(seed=s))
        i = np.arange(CFG.n)
        w = d.pi_e_dist[i, d.action] / d.pscore_logged
        snips = float((w * d.reward).sum() / w.sum())
        lo1, hi1 = msm_bounds(w, d.reward, 1.0)
        if not (abs(lo1 - snips) < 1e-10 and abs(hi1 - snips) < 1e-10):
            checks["lam1_is_snips"] = False
        prev = (lo1, hi1)
        for lam in (1.2, 1.5, 2.0, 3.0):
            lo, hi = msm_bounds(w, d.reward, lam)
            if not (lo <= prev[0] + 1e-12 and hi >= prev[1] - 1e-12):
                checks["monotone"] = False
            if not (np.isfinite(lo) and np.isfinite(hi)):
                checks["stable"] = False
            prev = (lo, hi)
        ratio = (d.pscore_true / d.pscore_logged)
        lam_emp = float(max(ratio.max(), (1 / ratio).max()))
        lam_emps.append(lam_emp)
        lo, hi = msm_bounds(w, d.reward, lam_emp)
        cover.append(bool(lo <= v <= hi))
    checks["coverage"] = all(cover)
    verdict = "GO" if all(checks.values()) else "NO-GO"
    OUT.write_text(json.dumps({"probe": "M5-14 lambda_msm", "v_true": v,
                               "lam_emp_range": [min(lam_emps), max(lam_emps)],
                               "coverage_per_seed": cover, "checks": checks,
                               "verdict": verdict}, indent=2))
    print(f"v_true={v:.4f} lam_emp={min(lam_emps):.2f}–{max(lam_emps):.2f} "
          f"coverage={cover} checks={checks} VERDICT: {verdict}")


if __name__ == "__main__":
    main()
