"""축 17 — GT-free validity battery 의 blind-then-reveal 채점 (M8 GT-미상 본편).

사전등록(PLAN §3.5 — 커밋 a1bfa4c, 실험 수치 이전): battery 정의·[제안] 임계·실패 family
목록·채점 규칙이 실험 전에 고정되었다. **frontstage**(`17_*_decision.csv` — v_true 류 컬럼
부재, 계약 테스트)와 **reveal**(`17_*_reveal.csv` — 백스테이지 채점)은 코드 경로로 분리되며,
pooled 단독 보고는 금지된다(family×arm 분리 — §3.5-3).

프레임 주의(정직성): battery 는 **필요조건 검사** — calibrated confounding 은 관측 동등성으로
원리적 무검출이며(impossible family — 축 18 이 경계 전시), 이 축의 detection matrix 는 그
빈칸을 그대로 전시한다. 후보 분포는 외생 입력(벤치마크 지정 β_eval=3) — 후보 *구축* 의
GT-미상 판은 축 19. q̂ 는 로그 유래(fitters crossfit — 실무자 가용 자원만).

실측 예고(probe M8-A/M8-B — PLAN §3.5-4 판정 기록): noised·support 는 mean_w 방향 발화,
calibrated 는 비발화(원리), as-recorded 도 비발화(실측 — "부분 검출" 예상의 반증 가능성).
불발 family 도 matrix 에 그대로 보고한다 — 예상 반증도 발견이다.

산출: results/tables/17_validity_battery_decision.csv · _reveal.csv · _matrix.csv ·
      _confusion.csv ↔ results/figures/17_validity_battery.png
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from _common import BASE_M2, SEEDS_DEFAULT, TAB_DIR, cached_v_true
from _practitioner import (
    DECISION_ESTIMATOR, PractitionerLog, reveal, run_protocol, write_decision_csv,
)
from _style import apply_style, save_figure
from ope.dgp import make_synthetic_bandit_data, marginal_logging_dist
from ope.fitters import CrossFitConfig, fit_q_hat_crossfit
from ope.validity import ValidityConfig

AXIS_ID, SLUG = "17", "validity_battery"
SEEDS = SEEDS_DEFAULT  # S=40 (§3.5-3)
PS_CLIP = (1e-6, 1.0)
NOISE_RNG_OFFSET = 500_000  # 축 05 관례 계승

# 실패 family 사전등록 목록(§3.5-3 — 사후 추가·삭제 금지). (scenario, family, cfg 변형, pscore 모드)
SCENARIOS = [
    ("clean",               "control",    {},                            "recorded"),
    ("small_n_500",         "control",    {"n": 500},                    "recorded"),
    ("small_n_2000",        "control",    {"n": 2000},                   "recorded"),
    ("low_overlap_b8",      "detectable", {"beta_log": 8.0},             "recorded"),
    ("low_overlap_b16",     "detectable", {"beta_log": 16.0},            "recorded"),
    ("support_d02",         "detectable", {"support_deficiency": 0.2},   "recorded"),
    ("support_d04",         "detectable", {"support_deficiency": 0.4},   "recorded"),
    ("noised_s05",          "detectable", {},                            "noised_0.5"),
    ("noised_s10",          "detectable", {},                            "noised_1.0"),
    ("estimated_insample",  "partial",    {},                            "estimated"),
    ("conf_recorded_g10",   "partial",    {"confounding_strength": 1.0}, "recorded"),
    ("conf_recorded_g25",   "partial",    {"confounding_strength": 2.5}, "recorded"),
    ("conf_calibrated_g10", "impossible", {"confounding_strength": 1.0}, "calibrated"),
    ("conf_calibrated_g25", "impossible", {"confounding_strength": 2.5}, "calibrated"),
]
FAMILY_ORDER = ["control", "detectable", "partial", "impossible"]
ARMS = ["mean_w", "harmonic", "placebo", "disagreement"]


def _pscore_for(mode: str, d, cfg, seed: int) -> np.ndarray:
    """시나리오별 '시스템이 기록한' propensity — 오염은 시스템측(seed 파생 고정 rng)."""
    if mode == "recorded":
        return d.pscore_logged
    if mode.startswith("noised_"):
        s = float(mode.split("_")[1])
        z = np.random.default_rng(NOISE_RNG_OFFSET + seed).normal(size=len(d.action))
        return np.clip(d.pscore_logged * np.exp(s * z), *PS_CLIP)
    if mode == "estimated":  # 축 05 estimated 모드 계승 — in-sample MLE 준-null 함정 사전 문서화
        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(d.context, d.action)
        proba = clf.predict_proba(d.context)
        col = np.searchsorted(clf.classes_, d.action)
        return np.clip(proba[np.arange(len(d.action)), col], *PS_CLIP)
    if mode == "calibrated":  # 관측 동등성 세계 — 기록 = 참 marginal (축 18 소관, 여기선 참조 전시)
        p = marginal_logging_dist(cfg, d.context)
        return p[np.arange(len(d.action)), d.action]
    raise ValueError(mode)


def main() -> None:
    t0 = time.perf_counter()
    all_rows = []
    for scenario, family, over, mode in SCENARIOS:
        cfg = BASE_M2._replace(**over)
        for seed in SEEDS:
            c = cfg._replace(seed=seed)
            d = make_synthetic_bandit_data(c)
            ps = _pscore_for(mode, d, c, seed)
            q_hat = fit_q_hat_crossfit(d.context, d.action, d.reward, c.n_actions,
                                       CrossFitConfig(seed=seed))
            log = PractitionerLog(context=d.context, action=d.action, reward=d.reward,
                                  pscore=ps,
                                  pi_log_dist=d.pi_log_dist if mode == "recorded" else None)
            rows = run_protocol(log, d.pi_e_dist, axis_id=AXIS_ID, scenario=scenario,
                                run_id=f"{scenario}-{seed}", seed=seed, q_hat=q_hat,
                                cfg=ValidityConfig(seed=seed))
            all_rows += rows
    dec_path = write_decision_csv(AXIS_ID, SLUG, all_rows)

    # ── reveal (백스테이지 — committed frontstage CSV 를 파일로 읽어 채점) ─────────
    v_true = cached_v_true(BASE_M2)  # β_eval·K·d 불변 ⇒ 전 시나리오 공통 참값
    truth = pd.DataFrame({"run_id": [f"{s}-{seed}" for s, _, _, _ in SCENARIOS
                                     for seed in SEEDS],
                          "v_true": v_true})
    rev_path = reveal(dec_path, truth, "exact_synthetic")

    # ── family×arm matrix (run 수준 — battery 필드는 estimator 행에 중복 기록됨) ──
    dec = pd.read_csv(dec_path)
    runs = dec[dec["estimator"] == DECISION_ESTIMATOR].copy()
    rev = pd.read_csv(rev_path)
    rev_runs = rev[rev["estimator"] == DECISION_ESTIMATOR][["run_id", "rel_err", "large_err"]]
    runs = runs.merge(rev_runs, on="run_id", validate="one_to_one")
    fam_map = {s: f for s, f, _, _ in SCENARIOS}
    runs["family"] = runs["scenario"].map(fam_map)

    def _rates(g: pd.DataFrame) -> pd.Series:
        out = {f"{arm}_fire": (g[f"{arm}_state"] == "fail").mean() for arm in ARMS}
        out["disagreement_inconclusive"] = (g["disagreement_state"] == "inconclusive").mean()
        out["gate_v1_nontrust"] = (g["gate_v1"] != "trust").mean()
        out["verdict_nontrust"] = (g["protocol_verdict"] != "trust").mean()
        out["large_err_rate"] = g["large_err"].mean()
        out["median_rel_err"] = g["rel_err"].median()
        out["runs"] = len(g)
        return pd.Series(out)

    matrix = (runs.groupby(["family", "scenario"], sort=False).apply(_rates, include_groups=False)
              .reset_index())
    matrix["family"] = pd.Categorical(matrix["family"], FAMILY_ORDER, ordered=True)
    order = {s: i for i, (s, _, _, _) in enumerate(SCENARIOS)}
    matrix = matrix.sort_values(by="scenario", key=lambda s: s.map(order)).reset_index(drop=True)
    matrix_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    # ── coverage-by-verdict (축 08 confusion·축 11 커버리지 교훈의 일반화) ─────────
    rev_s = rev[rev["estimator"] == DECISION_ESTIMATOR]
    conf = (rev_s.groupby("protocol_verdict")
            .agg(runs=("rel_err", "size"), share_large_err=("large_err", "mean"),
                 median_rel_err=("rel_err", "median"),
                 ci_covers_rate=("ci_covers_truth", "mean")).reset_index())
    conf_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_confusion.csv"
    conf.to_csv(conf_path, index=False)

    # ── figure: detection matrix 히트맵 + 백스테이지 대오차 패널 ──────────────────
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("fire", ["#fcfcfb", "#44433e"])
    cols = [f"{arm}_fire" for arm in ARMS] + ["gate_v1_nontrust", "verdict_nontrust"]
    col_labels = ["E[w]", "harmonic", "placebo", "disagree", "gate v1\n(≠trust)",
                  "verdict\n(≠trust)"]
    m = matrix[cols].to_numpy(dtype=float)
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(11.5, 6.2), gridspec_kw={"width_ratios": [3.2, 1.0]}, sharey=True)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                    color="#fcfcfb" if m[i, j] > 0.55 else "#33322e")
    ax.set_xticks(range(len(cols)), col_labels, fontsize=8)
    ax.set_yticks(range(len(matrix)),
                  [f"[{f[:4]}] {s}" for f, s in zip(matrix['family'], matrix['scenario'])],
                  fontsize=8)
    prev = None
    for i, f in enumerate(matrix["family"]):
        if prev is not None and f != prev:
            ax.axhline(i - 0.5, color="#6f6e66", lw=1.2)
        prev = f
    ax.set_title("A. Battery fire rate by pre-registered failure family\n"
                 "(frontstage — computed from logs alone; thresholds pre-registered, "
                 "no tuning)", fontsize=10)
    ax.grid(False)

    axr.barh(range(len(matrix)), matrix["large_err_rate"], color="#e34948", height=0.62)
    axr.set_xlim(0, 1.02)
    axr.set_xlabel(f"P(rel err > 0.10) — {DECISION_ESTIMATOR.upper()}")
    axr.set_title("B. What was actually broken\n(backstage reveal — invisible in practice)",
                  fontsize=10)
    axr.grid(axis="x")
    fig.suptitle("Axis 17 — GT-free validity battery: what it catches, and the blank it "
                 "cannot see\n(calibrated confounding rows stay dark-free by construction — "
                 "observational equivalence; the exit is Λ-sensitivity, axis 14/18)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── PATTERN (사전등록 예상 대조 — 불발도 그대로 출력) ─────────────────────────
    def _rate(scn, col):
        return float(matrix.loc[matrix["scenario"] == scn, col].iloc[0])

    patterns = {
        "noised_s10 mean_w fire ≥ 0.9": _rate("noised_s10", "mean_w_fire") >= 0.9,
        "support_d04 mean_w fire ≥ 0.9": _rate("support_d04", "mean_w_fire") >= 0.9,
        "calibrated g25 calibration-arm fire ≤ 0.05":
            max(_rate("conf_calibrated_g25", "mean_w_fire"),
                _rate("conf_calibrated_g25", "harmonic_fire")) <= 0.05,
        "clean any-arm false alarm ≤ 0.2":
            max(_rate("clean", f"{a}_fire") for a in ARMS) <= 0.2,
    }
    print(f"[17] → {dec_path.name}, {rev_path.name}, {matrix_path.name}, {conf_path.name}, "
          f"{fig_path.name} ({time.perf_counter() - t0:.0f}s)")
    for k, ok in patterns.items():
        print(f"[17] PATTERN {'PASS' if ok else 'FAIL'}: {k}")
    print("[17] 실측(예상 없음 — 기록): as-recorded g25 mean_w fire = "
          f"{_rate('conf_recorded_g25', 'mean_w_fire'):.2f} · estimated_insample mean_w fire = "
          f"{_rate('estimated_insample', 'mean_w_fire'):.2f} (in-sample 준-null 함정 문서화)")
    print(conf.to_string(index=False))


if __name__ == "__main__":
    main()
