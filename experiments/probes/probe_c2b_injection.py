"""PROBE M9-A — c2b 주입 기전의 역학·방향·런타임 (GO/NO-GO — 축 21 착수 게이트, PLAN §3.6-5).

WHAT GENERALIZES: GO 면 ① 구조 mask 주입(하위-s ⌊δK⌋ 제거 → π₀ renormalize → 재표집)이
역학적으로 무결하고(masked 행동 로그 무출현·기록 pscore = renormalized π₀ verbatim·gt_value
불변 — 오염은 로깅측뿐) ② support 주입이 mean_w < 1 방향으로 움직이며(**발화(임계 통과)는
GO 조건이 아니다** — 발화를 요구하면 주입 튜닝이 된다, §3.6-5) ③ noised 주입의 e^{s²/2} 기전이
실데이터에서도 데이터-불가지(mean_w > 1)이고 ④ 최악 케이스(letter: n=10k·K=26)의 battery+joint
bootstrap 이 예산(≤3s/run) 내임이 확인된다 → 축 21 착수.

여기 인라인 구현(c2b 변환·mask·battery)이 검증 대상의 독립 재구현이고, production 로더와의
δ=0 bit-항등 대조는 Stage 2 테스트 소관(probe 는 src 미의존 — §3.6-5 분업).

THE RESULT → results/tables/probe_c2b_injection.json. VERDICT: stdout + JSON.
HONEST reduces_check: 데이터셋 2종(satimage=K6 최조악 양자화·letter=K26 런타임 최악)·5-seed
스크린이다 — 4 datasets × S=20 전량과 harmonic/placebo/disagreement arm 채점은 축 21 본실험의
몫. 부수 실측(masked π_e 질량 — §3.6-4 예상의 정량 앵커)은 게이트가 아니라 기록이다.
"""

import json
import time
from pathlib import Path

import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

OUT = Path(__file__).resolve().parents[2] / "results" / "tables" / "probe_c2b_injection.json"
DATA_HOME = Path(__file__).resolve().parents[2] / "data" / "openml"

DATASETS = {"satimage": 1, "letter": 1}   # K=6 최조악 양자화 · K=26 런타임 최악
SPLIT_SEED = 0                            # C2B_SPLIT_SEED 관례 (인라인 재구현)
BETA_LOG, BETA_EVAL = 2.0, 6.0
SEEDS = range(700, 705)
DELTAS = (0.2, 0.4)
NOISE_S = 1.0
PS_CLIP = (1e-6, 1.0)
B = 500                                   # §3.5-1 joint bootstrap 사전등록값 재사용
RUNTIME_BUDGET = 3.0                      # §3.6-5 ④ (M8-A 예산 규칙)


def convert(name: str, version: int):
    """c2b 변환 인라인 재구현(구조 1회 — seed 는 로깅 재표집 전용)."""
    bunch = fetch_openml(name=name, version=version, data_home=str(DATA_HOME),
                         as_frame=False, parser="auto")
    x = np.asarray(bunch.data, dtype=float)
    y = LabelEncoder().fit_transform(bunch.target)
    k = int(y.max()) + 1
    x_a, x_b, y_a, y_b = train_test_split(x, y, test_size=0.5, stratify=y,
                                          random_state=SPLIT_SEED)
    scaler = StandardScaler().fit(x_a)
    s = LogisticRegression(max_iter=1000).fit(scaler.transform(x_a), y_a) \
        .predict_proba(scaler.transform(x_b))
    z0 = np.exp(BETA_LOG * (s - s.max(axis=1, keepdims=True)))
    pi0 = z0 / z0.sum(axis=1, keepdims=True)
    ze = np.exp(BETA_EVAL * (s - s.max(axis=1, keepdims=True)))
    pi_e = ze / ze.sum(axis=1, keepdims=True)
    return s, pi0, pi_e, y_b, k


def mask_bottom(s: np.ndarray, delta: float) -> np.ndarray:
    n, k = s.shape
    m = int(np.floor(delta * k))
    keep = np.ones((n, k), dtype=bool)
    if m > 0:
        np.put_along_axis(keep, np.argsort(s, axis=1)[:, :m], False, axis=1)
    return keep


def sample_log(pi0: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (pi0.cumsum(axis=1) > rng.random((len(pi0), 1))).argmax(axis=1)


def battery_runtime(reward, action, pscore, pi_e, seed: int) -> float:
    """mean_w·harmonic·placebo(joint bootstrap B=500) + disagreement — 런타임 실측용 인라인."""
    n, k = pi_e.shape
    idx = np.arange(n)
    w = pi_e[idx, action] / pscore
    inv_ps = 1.0 / pscore
    eps = np.random.default_rng(9_000_000 + seed).normal(0.0, float(np.std(reward)), n)
    t0 = time.perf_counter()
    rng = np.random.default_rng(17_000 + seed)
    mw_b, pl_b, t_b = np.empty(B), np.empty(B), np.empty((B, k))
    for b in range(B):
        ii = rng.integers(0, n, n)
        mw_b[b] = w[ii].mean()
        pl_b[b] = (w[ii] * eps[ii]).mean()
        t_b[b] = np.bincount(action[ii], weights=inv_ps[ii], minlength=k) / n
    ips = float((w * reward).mean())
    snips = float((w * reward).sum() / w.sum())
    lam = float(np.quantile(w, 0.90))
    _ = float((np.minimum(w, lam) * reward).mean())
    _ = (max(ips, snips) - min(ips, snips)) / max(abs(snips), 1e-12)
    return time.perf_counter() - t0


def main() -> None:
    checks = {"c1_mask_mechanics": True, "c2_support_direction": True,
              "c3_noised_direction": True, "c4_runtime_le_3s": True}
    results: dict[str, dict] = {}
    for name, ver in DATASETS.items():
        s, pi0, pi_e, y_b, k = convert(name, ver)
        n = len(y_b)
        idx = np.arange(n)
        gt_clean = float(pi_e[idx, y_b].mean())
        rec: dict = {"n": n, "K": k, "gt_value_clean": gt_clean,
                     "masked_pi_e_mass": {}, "mean_w": {}}
        for delta in DELTAS:
            keep = mask_bottom(s, delta)
            pi0_m = np.where(keep, pi0, 0.0)
            pi0_m = pi0_m / pi0_m.sum(axis=1, keepdims=True)
            rec["masked_pi_e_mass"][str(delta)] = float(
                np.where(~keep, pi_e, 0.0).sum(axis=1).mean())
            mws = []
            for seed in SEEDS:
                a = sample_log(pi0_m, seed)
                # ① 역학: masked 행동 무출현 · pscore verbatim · gt 불변(π_e 무접촉)
                if not keep[idx, a].all():
                    checks["c1_mask_mechanics"] = False
                ps = pi0_m[idx, a]
                if not (ps > 0).all():
                    checks["c1_mask_mechanics"] = False
                if float(pi_e[idx, y_b].mean()) != gt_clean:
                    checks["c1_mask_mechanics"] = False
                mws.append(float((pi_e[idx, a] / ps).mean()))
            rec["mean_w"][f"support_d{delta:g}"] = mws
            if delta == 0.4 and not all(v < 1.0 for v in mws):
                checks["c2_support_direction"] = False
        # ③ noised 방향 (letter 지정 — §3.6-5) + ④ 런타임 (letter)
        if name == "letter":
            mws_noised, runtimes = [], []
            for seed in SEEDS:
                a = sample_log(pi0, seed)
                ps_true = pi0[idx, a]
                z = np.random.default_rng(900_000 + seed).normal(size=n)
                ps_noised = np.clip(ps_true * np.exp(NOISE_S * z), *PS_CLIP)
                mws_noised.append(float((pi_e[idx, a] / ps_noised).mean()))
                r = (a == y_b).astype(float)
                runtimes.append(battery_runtime(r, a, ps_noised, pi_e, seed))
            rec["mean_w"]["noised_s10"] = mws_noised
            rec["battery_runtime_sec"] = runtimes
            if not all(v > 1.0 for v in mws_noised):
                checks["c3_noised_direction"] = False
            if float(np.median(runtimes)) > RUNTIME_BUDGET:
                checks["c4_runtime_le_3s"] = False
        results[name] = rec
    verdict = "GO" if all(checks.values()) else "NO-GO"
    OUT.write_text(json.dumps({
        "probe": "M9-A c2b_injection",
        "prereg": {"datasets": list(DATASETS), "deltas": list(DELTAS), "noise_s": NOISE_S,
                   "seeds": list(SEEDS), "B": B, "runtime_budget_sec": RUNTIME_BUDGET,
                   "source": "PLAN.md §3.6-5 (커밋 5bda384 사전등록)"},
        "results": results, "checks": checks, "verdict": verdict}, indent=2))
    sat, let = results["satimage"], results["letter"]
    print(f"masked π_e 질량: satimage {sat['masked_pi_e_mass']} · letter {let['masked_pi_e_mass']}")
    print(f"support_d0.4 mean_w: satimage {[round(v, 4) for v in sat['mean_w']['support_d0.4']]} · "
          f"letter {[round(v, 4) for v in let['mean_w']['support_d0.4']]}")
    print(f"noised_s10 mean_w(letter): {[round(v, 3) for v in let['mean_w']['noised_s10']]} · "
          f"runtime_med={np.median(let['battery_runtime_sec']):.3f}s")
    print(f"checks={checks} → VERDICT: {verdict}")


if __name__ == "__main__":
    main()
