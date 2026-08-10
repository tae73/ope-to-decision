"""PROBE M8-B — calibrated(U-주변화) 기록 pscore 구성의 안정성 + battery null 정합 (GO/NO-GO — 축 18 게이트).

WHAT GENERALIZES: GO 면 "기록 pscore = 참 marginal P(a|x)" 인 **관측 동등성 세계**를 DGP 생성기
본체 무수정으로 만들 수 있고(사후 순수함수 — rng 무접촉·기존 축 01–16 산출 불변), 그 세계에서
① 주변화 계산이 수치 안정(node 400 vs 800 상대차 < 1% ∧ γ=0 항등)하고
② battery 의 calibration arm(mean_w·harmonic — PLAN §3.5-1 임계 그대로)이 **비발화**(null 정합 —
   이 세계에선 어떤 로그 통계도 confounding 을 원리적으로 구별 불가)하며
③ 그럼에도 IPS bias 는 γ 와 함께 성장 잔존함(백스테이지 채점 — probe 한정 허용)이 확인된다
→ 축 18("battery 는 blind spot 을 줄이지만 없애지 못한다" 경계 전시) 착수.
④ 대조: 같은 γ 의 **as-recorded**(의도값 기록 — 축 09 장치) 로그에서 같은 통계의 발화 여부를
   실측 기록한다(방향 무관 — miscalibration 부분 검출 서사의 분기 데이터).

구현 노트: U-주변화는 MC 가 아니라 **결정적 구적**으로 계산한다 — rng 자체가 불필요해 DGP
생성기 rng 무접촉이 자명하고, 수치 안정 기준이 기계 판독된다. 구적은 Gauss–Legendre ×
Gaussian weight(절단 [-8,8] — φ 질량 손실 ~1.2e-15, 행 정규화로 흡수)를 쓴다: numpy
`hermegauss` 는 degree ≥ ~400 에서 overflow(실측 — 첫 실행 NaN 전파로 NO-GO, 본 파일에서
교체) 하는 반면 `leggauss` 는 고차에서 안정하다. 여기 inline 구현이 원형이고 Stage 2 에서
dgp.py 사후 함수로 승격된다.

THE RESULT → results/tables/probe_calibrated_confounding.json. VERDICT: stdout + JSON.
HONEST reduces_check: β_log=1·δ=0·K=10 한 구조(struct_seed=7)에서의 5-seed 스크린이다 —
축 18 본실험이 γ 그리드·S=40 으로 확장한다. ② 의 "비발화"는 사전등록 임계 기준이지 통계량이
정확히 0 이라는 주장이 아니다(quadrature·표본 오차 존재). ③ 은 v_true 를 쓰는 백스테이지
채점으로 probe 에서만 허용(frontstage 규약은 축 18 본실험에서 스키마로 강제).
"""

import json
from pathlib import Path

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from ope.dgp import (  # noqa: E402
    DGPConfig, make_synthetic_bandit_data, true_policy_value,
    _make_structure, _q_true,  # probe 전용 private 재사용 (tests 선례) — 구조 재현에 필요
)

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "probe_calibrated_confounding.json"

BASE = DGPConfig(n=10_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                 support_deficiency=0.0, reward_noise=0.5, confounding_strength=0.0,
                 seed=0, struct_seed=7)
GAMMAS = (1.0, 2.5)               # §3.5-3 confounded family 와 동일 그리드
SEEDS = range(700, 705)
N_NODES, N_NODES_CHECK = 800, 400  # §3.5-4 ①: 400 vs 800 상대차 < 1%
B, ALPHA = 500, 0.05
TOL_MEANW, TOL_HARM = 0.10, 0.25   # §3.5-1 사전등록 임계
MIN_NA = 30
BOOT_RNG_OFFSET = 18_000
NEED = 4                           # 5-seed 중 ≥4


U_TRUNC = 8.0  # ∫_{|u|>8} φ(u)du ≈ 1.2e-15 — 절단 오차는 행 정규화가 흡수


def marginal_logging_dist(x: np.ndarray, cfg: DGPConfig, gamma: float,
                          n_nodes: int) -> np.ndarray:
    """U-주변화 로깅 분포 P(a|x) = E_U[softmax(β_log·q(x) + γ·U·d)] — 결정적 구적.

    Gauss–Legendre 노드를 [-U_TRUNC, U_TRUNC] 로 스케일하고 Gaussian 밀도 φ(u) 를 weight 로
    곱한다(rng 불요). dgp.py 생성기와 같은 구조(θ·b·d — _make_structure)·같은 logits 정의를
    쓰되 생성기 rng 는 일절 건드리지 않는 사후 순수함수다(δ=0 전제 — mask 결합은 축 18
    본구현 범위 밖).
    """
    theta, bvec, dvec = _make_structure(cfg)
    q = _q_true(x, theta, bvec)
    logits = cfg.beta_log * q
    nodes, wts = np.polynomial.legendre.leggauss(n_nodes)
    u_nodes = nodes * U_TRUNC
    wts = wts * U_TRUNC * np.exp(-0.5 * u_nodes ** 2) / np.sqrt(2.0 * np.pi)
    acc = np.zeros_like(q)
    for u, wt in zip(u_nodes, wts):
        z = logits + gamma * u * dvec[None, :]
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        acc += wt * (e / e.sum(axis=1, keepdims=True))
    return acc / acc.sum(axis=1, keepdims=True)


def calibration_arms(action, pscore, pi_e_dist, boot_seed):
    """battery 중 calibration arm 2종(mean_w·harmonic)만 — PLAN §3.5-1 임계 그대로."""
    n, k = pi_e_dist.shape
    idx = np.arange(n)
    w = pi_e_dist[idx, action] / pscore
    inv_ps = 1.0 / pscore
    rng = np.random.default_rng(BOOT_RNG_OFFSET + boot_seed)
    mw_b = np.empty(B)
    t_b = np.empty((B, k))
    for b in range(B):
        ii = rng.integers(0, n, n)
        mw_b[b] = w[ii].mean()
        t_b[b] = np.bincount(action[ii], weights=inv_ps[ii], minlength=k) / n
    mean_w = float(w.mean())
    lo, hi = np.quantile(mw_b, [ALPHA / 2, 1 - ALPHA / 2])
    mw_fail = (lo > 1.0 or hi < 1.0) and abs(mean_w - 1.0) > TOL_MEANW
    n_a = np.bincount(action, minlength=k)
    t_point = np.bincount(action, weights=inv_ps, minlength=k) / n
    harm_fail, t_range = False, []
    for a in range(k):
        if n_a[a] < MIN_NA:
            continue
        alo, ahi = np.quantile(t_b[:, a], [ALPHA / 2, 1 - ALPHA / 2])
        if (alo > 1.0 or ahi < 1.0) and abs(t_point[a] - 1.0) > TOL_HARM:
            harm_fail = True
        t_range.append(float(t_point[a]))
    return {"mean_w": mean_w, "mean_w_ci": [float(lo), float(hi)], "mean_w_fail": bool(mw_fail),
            "harmonic_fail": bool(harm_fail),
            "harmonic_T_range": [min(t_range), max(t_range)] if t_range else None}


def main() -> None:
    v_true = true_policy_value(BASE)
    idx = np.arange(BASE.n)
    results: dict[str, dict] = {}
    stability: dict[str, dict] = {}
    checks = {"s1_quadrature_stable": True, "s1_gamma0_identity": True,
              "s2_calibrated_null": True, "s3_bias_grows": True}

    # ① γ=0 항등: 주변화 분포 == 의도 정책 분포 (mask 없음)
    d0 = make_synthetic_bandit_data(BASE._replace(seed=700))
    p0 = marginal_logging_dist(d0.context, BASE, 0.0, N_NODES)
    id_diff = float(np.max(np.abs(p0[idx, d0.action] - d0.pscore_logged)))
    checks["s1_gamma0_identity"] = id_diff < 1e-10

    for gamma in GAMMAS:
        cfg = BASE._replace(confounding_strength=gamma)
        per_seed = []
        ips_cal = []
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            p_marg = marginal_logging_dist(d.context, cfg, gamma, N_NODES)
            ps_cal = p_marg[idx, d.action]
            cal = calibration_arms(d.action, ps_cal, d.pi_e_dist, seed)
            rec_arms = calibration_arms(d.action, d.pscore_logged, d.pi_e_dist, seed)
            w_cal = d.pi_e_dist[idx, d.action] / ps_cal
            ips_cal.append(float((w_cal * d.reward).mean()))
            per_seed.append({"seed": seed, "calibrated": cal, "as_recorded": rec_arms,
                             "ips_calibrated": ips_cal[-1]})
        # ① 수치 안정 (구조 수준 — 첫 seed 로그에서 400 vs 800 노드 대조)
        d1 = make_synthetic_bandit_data(cfg._replace(seed=700))
        pa = marginal_logging_dist(d1.context, cfg, gamma, N_NODES)[idx, d1.action]
        pb = marginal_logging_dist(d1.context, cfg, gamma, N_NODES_CHECK)[idx, d1.action]
        rel = float(np.max(np.abs(pa - pb) / pa))
        stability[str(gamma)] = {"max_rel_diff_400v800": rel}
        if rel >= 0.01:
            checks["s1_quadrature_stable"] = False
        # ② calibrated null: mean_w·harmonic 비발화 ≥4/5
        null_ok = sum((not r["calibrated"]["mean_w_fail"]) and
                      (not r["calibrated"]["harmonic_fail"]) for r in per_seed)
        if null_ok < NEED:
            checks["s2_calibrated_null"] = False
        rec_fires = sum(r["as_recorded"]["mean_w_fail"] or r["as_recorded"]["harmonic_fail"]
                        for r in per_seed)
        bias = float(np.mean(ips_cal) - v_true)
        se = float(np.std(ips_cal, ddof=1) / np.sqrt(len(ips_cal)))
        results[str(gamma)] = {"per_seed": per_seed, "null_ok": null_ok,
                               "as_recorded_fires": rec_fires,
                               "ips_calibrated_bias_mean": bias, "ips_se": se}

    b1, b2 = results[str(GAMMAS[0])], results[str(GAMMAS[1])]
    checks["s3_bias_grows"] = (abs(b2["ips_calibrated_bias_mean"]) > abs(b1["ips_calibrated_bias_mean"])
                               and abs(b2["ips_calibrated_bias_mean"]) > 3.0 * b2["ips_se"])
    verdict = "GO" if all(checks.values()) else "NO-GO"
    OUT.write_text(json.dumps({
        "probe": "M8-B calibrated_confounding",
        "prereg": {"gammas": list(GAMMAS), "n_nodes": [N_NODES_CHECK, N_NODES],
                   "thresholds": {"mean_w": TOL_MEANW, "harmonic": TOL_HARM},
                   "B": B, "alpha": ALPHA, "seeds": list(SEEDS), "need": NEED,
                   "quadrature": "Gauss-Legendre × φ(u), 절단 [-8,8] — rng 불요 사후 순수함수 "
                                 "(hermegauss 는 degree≥~400 overflow 실측으로 교체)",
                   "source": "PLAN.md §3.5-4 (커밋 a1bfa4c 사전등록)"},
        "v_true_backstage": v_true, "gamma0_identity_max_diff": id_diff,
        "stability": stability, "results": results,
        "checks": checks, "verdict": verdict}, indent=2))
    print(f"γ=0 identity diff={id_diff:.2e} · 400v800 rel="
          f"{[round(stability[str(g)]['max_rel_diff_400v800'], 8) for g in GAMMAS]} · "
          f"calibrated null_ok={[results[str(g)]['null_ok'] for g in GAMMAS]}/5 · "
          f"as-recorded fires={[results[str(g)]['as_recorded_fires'] for g in GAMMAS]}/5 · "
          f"bias(cal)={[round(results[str(g)]['ips_calibrated_bias_mean'], 5) for g in GAMMAS]} "
          f"(v_true 백스테이지 채점) → VERDICT: {verdict}")


if __name__ == "__main__":
    main()
