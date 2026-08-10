"""PROBE M8-A — GT-free validity battery 의 방향성 발화 + 런타임 (GO/NO-GO — 축 17·19·20 착수 게이트).

WHAT GENERALIZES: GO 면 PLAN §3.5-1 **사전등록 정의·임계 그대로**의 battery 가
① pscore-noised(s=1.0)에서 mean_w 가 log-normal 예측 방향(e^{s²/2}>1)으로 발화하고
② 구조적 support 결핍(δ=0.4)에서 mean_w<1 방향으로 발화하며 — **전역 support proxy 가 0 인
   바로 그 지점**(축 04 `m2-04-proxy-blind` 의 GT-미상 대응) —
③ clean 에서 placebo 오알람이 없고 ④ joint bootstrap(B=500)이 실용 런타임(≤3s/battery)임이
확인된다 → 축 17(blind-then-reveal)·19·20 착수. 여기 inline 구현이 battery 의 원형이고
Stage 2 에서 `src/ope/validity.py` 로 승격된다(M5 의 msm_bounds → estimators.py 승격 선례).

THE RESULT → results/tables/probe_validity_battery.json. VERDICT: stdout + JSON.
HONEST reduces_check: 발화 여부의 5-seed 스크린(기준별 ≥4/5)일 뿐 — 예보력(발화↔대오차
confusion·family 분리)은 축 17 본실험의 몫이고, confounded family 는 M8-B 소관.
임계값(0.10/0.25/0.50·B=500·n_a≥30)은 PLAN §3.5-1 사전등록값 그대로 — 본 probe 에서 조정 금지.
disagreement 의 λ 는 관측 pscore 의 raw weight p90(실무자에겐 이것이 유일한 weight —
`hyperparams_from_weights` 정책 계승).
"""

import json
import time
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ope.dgp import DGPConfig, make_synthetic_bandit_data  # noqa: E402

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "probe_validity_battery.json"

BASE = DGPConfig(n=10_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                 support_deficiency=0.0, reward_noise=0.5, confounding_strength=0.0,
                 seed=0, struct_seed=7)
SEEDS = range(700, 705)          # 5-seed 스크린
NOISE_S = 1.0                     # §3.5-4 ①: s=1.0 (축 05 noised_1.0 기전 동일)
DELTA = 0.4                       # §3.5-4 ②: δ=0.4
PS_CLIP = (1e-6, 1.0)             # 축 05 와 동일
NOISE_RNG_OFFSET = 500_000        # 축 05 와 동일 (오염은 시스템측 — seed 파생 고정)
PLACEBO_RNG_OFFSET = 9_000_000    # placebo ε 는 분석가측 — 별도 고정 rng
BOOT_RNG_OFFSET = 17_000
B, ALPHA = 500, 0.05              # §3.5-1 joint bootstrap 사전등록값
TOL_MEANW, TOL_HARM, TOL_DISAG = 0.10, 0.25, 0.50
MIN_NA = 30                       # harmonic 대상 action 최소 출현 수
DISAG_FLOOR = 0.01                # |SNIPS| < 0.01·max(|mean r|, eps) → inconclusive
NEED = 4                          # 기준별 5-seed 중 ≥4 성립


def _ci(samples: np.ndarray) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [ALPHA / 2, 1 - ALPHA / 2])
    return float(lo), float(hi)


def run_battery(reward, action, pscore, pi_e_dist, boot_seed):
    """PLAN §3.5-1 gate arm 4종 — 로그 층 입력만. joint bootstrap: 같은 재표집 인덱스 공유."""
    n, k = pi_e_dist.shape
    idx = np.arange(n)
    w = pi_e_dist[idx, action] / pscore
    inv_ps = 1.0 / pscore
    eps = np.random.default_rng(PLACEBO_RNG_OFFSET + boot_seed).normal(
        0.0, float(np.std(reward)), size=n)

    t0 = time.perf_counter()
    rng = np.random.default_rng(BOOT_RNG_OFFSET + boot_seed)
    mw_b = np.empty(B)
    pl_b = np.empty(B)
    t_b = np.empty((B, k))
    for b in range(B):
        ii = rng.integers(0, n, n)
        mw_b[b] = w[ii].mean()
        pl_b[b] = (w[ii] * eps[ii]).mean()
        t_b[b] = np.bincount(action[ii], weights=inv_ps[ii], minlength=k) / n

    mean_w = float(w.mean())
    mw_ci = _ci(mw_b)
    mw_fail = (mw_ci[0] > 1.0 or mw_ci[1] < 1.0) and abs(mean_w - 1.0) > TOL_MEANW

    n_a = np.bincount(action, minlength=k)
    t_point = np.bincount(action, weights=inv_ps, minlength=k) / n
    harm, harm_fail = {}, False
    for a in range(k):
        if n_a[a] < MIN_NA:
            continue
        lo, hi = _ci(t_b[:, a])
        fail = (lo > 1.0 or hi < 1.0) and abs(t_point[a] - 1.0) > TOL_HARM
        harm[str(a)] = {"T": float(t_point[a]), "ci": [lo, hi], "n_a": int(n_a[a]), "fail": bool(fail)}
        harm_fail = harm_fail or fail

    pl_val = float((w * eps).mean())
    pl_ci = _ci(pl_b)
    pl_fail = pl_ci[0] > 0.0 or pl_ci[1] < 0.0

    ips = float((w * reward).mean())
    snips = float((w * reward).sum() / w.sum())
    lam = float(np.quantile(w, 0.90))
    clip = float((np.minimum(w, lam) * reward).mean())
    floor = DISAG_FLOOR * max(abs(float(reward.mean())), 1e-12)
    if abs(snips) < floor:
        disag, disag_state = None, "inconclusive"
    else:
        vals = [ips, snips, clip]
        disag = float((max(vals) - min(vals)) / max(abs(snips), 1e-12))
        disag_state = "fail" if disag > TOL_DISAG else "pass"
    runtime = time.perf_counter() - t0

    return {"mean_w": mean_w, "mean_w_ci": list(mw_ci), "mean_w_fail": bool(mw_fail),
            "harmonic": harm, "harmonic_fail": bool(harm_fail),
            "placebo": pl_val, "placebo_ci": list(pl_ci), "placebo_fail": bool(pl_fail),
            "disagreement": disag, "disagreement_state": disag_state,
            "lam_clip_p90": lam, "runtime_sec": runtime}


def _support_proxy(action: np.ndarray, pi_e_dist: np.ndarray) -> float:
    """진단의 전역 support proxy(로그 무출현 액션의 π̄_e 질량) — 대조 기록용(발화 기대 0)."""
    k = pi_e_dist.shape[1]
    appeared = np.bincount(action, minlength=k) > 0
    return float(pi_e_dist[:, ~appeared].sum(axis=1).mean()) if (~appeared).any() else 0.0


def main() -> None:
    cases: dict[str, list] = {"clean": [], "noised": [], "support": []}
    for seed in SEEDS:
        d = make_synthetic_bandit_data(BASE._replace(seed=seed))
        cases["clean"].append(
            run_battery(d.reward, d.action, d.pscore_logged, d.pi_e_dist, seed))

        z = np.random.default_rng(NOISE_RNG_OFFSET + seed).normal(size=BASE.n)
        ps_noised = np.clip(d.pscore_logged * np.exp(NOISE_S * z), *PS_CLIP)
        cases["noised"].append(
            run_battery(d.reward, d.action, ps_noised, d.pi_e_dist, seed))

        d4 = make_synthetic_bandit_data(BASE._replace(seed=seed, support_deficiency=DELTA))
        rec = run_battery(d4.reward, d4.action, d4.pscore_logged, d4.pi_e_dist, seed)
        rec["global_support_proxy"] = _support_proxy(d4.action, d4.pi_e_dist)
        cases["support"].append(rec)

    fires_up = sum(r["mean_w_fail"] and r["mean_w"] > 1.0 for r in cases["noised"])
    fires_dn = sum(r["mean_w_fail"] and r["mean_w"] < 1.0 for r in cases["support"])
    placebo_ok = sum(not r["placebo_fail"] for r in cases["clean"])
    runtimes = [r["runtime_sec"] for c in cases.values() for r in c]
    checks = {
        "a1_noised_meanw_fires_up": fires_up >= NEED,
        "a2_support_meanw_fires_down": fires_dn >= NEED,
        "a3_clean_placebo_no_alarm": placebo_ok >= NEED,
        "a4_runtime_le_3s": float(np.median(runtimes)) <= 3.0,
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"
    OUT.write_text(json.dumps({
        "probe": "M8-A validity_battery",
        "prereg": {"thresholds": {"mean_w": TOL_MEANW, "harmonic": TOL_HARM,
                                  "disagreement": TOL_DISAG},
                   "B": B, "alpha": ALPHA, "min_n_a": MIN_NA, "noise_s": NOISE_S,
                   "delta": DELTA, "seeds": list(SEEDS), "need": NEED,
                   "source": "PLAN.md §3.5-1/§3.5-4 (커밋 a1bfa4c 사전등록)"},
        "counts": {"noised_fires_up": fires_up, "support_fires_down": fires_dn,
                   "clean_placebo_ok": placebo_ok},
        "runtime_median_sec": float(np.median(runtimes)),
        "cases": cases, "checks": checks, "verdict": verdict}, indent=2))
    mw_noised = [round(r["mean_w"], 4) for r in cases["noised"]]
    mw_support = [round(r["mean_w"], 4) for r in cases["support"]]
    print(f"noised mean_w={mw_noised} (예측 e^{{s²/2}}≈{np.exp(NOISE_S**2 / 2):.3f}) "
          f"fires_up={fires_up}/5 · support mean_w={mw_support} fires_down={fires_dn}/5 · "
          f"clean placebo ok={placebo_ok}/5 · runtime_med={np.median(runtimes):.2f}s "
          f"→ VERDICT: {verdict}")


if __name__ == "__main__":
    main()
