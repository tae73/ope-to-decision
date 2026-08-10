"""축 21 — c2b 주입-실패 채점: battery 예보력의 실데이터 replication (M9 GT-미상 본편).

사전등록(PLAN §3.6 — 커밋 5bda384, 실험 수치 이전): 시나리오 7종·주입 기전·예상 라벨·PATTERN
이 실험 전에 고정되었다. 주장 = **replication(외적 타당성)**: 축 17 이 합성 기하에서 실측한
battery 예보력이 실제 공변량(표준화 OpenML 피처)·실제 정책 기하(LR-score softmax·K∈{6,10,26}·
n 2.8k–10k)·결정적 보상(1[a=y]) 위에서 재현되는가 + q̂-품질 채널(good vs degraded — gate arm
무반응 예측·`dr_correction` 상승 예측).

probe M9-A 판정: **NO-GO(② satimage support 방향 한정 — §3.6-5 폴백 ② 분기)** — 예상 라벨
무수정으로 실행하되 satimage 반증을 그대로 등재한다. probe 의 확정 발견("주입 실효 강도는
δ 가 아니라 π_e 첨도 × 하위-s 질량이 결정")이 본 축 해석의 렌즈다(LEDGER `m9-probe-a`).

**impossible family(confounded-calibrated)는 실데이터에서 구성 불가**(§3.6-3 — 참 P(y|x) 미지):
figure 에 hatched placeholder 로 부재를 전시하고 matrix CSV 에는 가짜 행을 넣지 않는다.
co-exhibit(CLAUDE.md §5): 이 축의 battery 주장 전부에 축 18(`m8-18-boundary`) 경계가 붙는다.

오염은 전부 로깅측 — `gt_value` 는 전 시나리오·전 seed 불변(스크립트가 assert). 로더는 라벨을
은닉하므로 시나리오 로그는 `classification_to_bandit` 호출로만 생성한다(라벨 무접촉 유지).

산출: results/tables/21_c2b_injection_decision.csv · _reveal.csv · _matrix.csv(28행 — pooled
행 금지) · _confusion.csv ↔ results/figures/21_c2b_injection.png
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from _common import TAB_DIR
from _practitioner import (
    DECISION_ESTIMATOR, PractitionerLog, reveal, run_protocol, write_decision_csv,
)
from _style import apply_style, save_figure
from ope.datasets import C2B_DATASETS, classification_to_bandit
from ope.validity import ValidityConfig

AXIS_ID, SLUG = "21", "c2b_injection"
SEEDS = range(1000, 1020)          # S=20 (축 11 정본 계승 — §3.6-1)
PS_CLIP = (1e-6, 1.0)
NOISE_RNG_BASE = 900_000           # §3.6-1: z ~ rng(900_000 + 10_000·dataset_idx + seed)

# 시나리오 7종 (§3.6-1 고정): (scenario, family, δ-variant, pscore 모드, q̂ 선택)
SCENARIOS = [
    ("clean",              "control",    0.0, "recorded",  "good"),
    ("clean_qdeg",         "control",    0.0, "recorded",  "degraded"),
    ("noised_s05",         "detectable", 0.0, "noised_0.5", "good"),
    ("noised_s10",         "detectable", 0.0, "noised_1.0", "good"),
    ("support_d02",        "detectable", 0.2, "recorded",  "good"),
    ("support_d04",        "detectable", 0.4, "recorded",  "good"),
    ("estimated_insample", "partial",    0.0, "estimated", "good"),
]
FAMILY_ORDER = ["control", "detectable", "partial"]
ARMS = ["mean_w", "harmonic", "placebo", "disagreement"]


def _pscore_for(mode: str, d, dataset_idx: int, seed: int) -> np.ndarray:
    if mode == "recorded":
        return d.pscore
    if mode.startswith("noised_"):
        s = float(mode.split("_")[1])
        z = np.random.default_rng(NOISE_RNG_BASE + 10_000 * dataset_idx + seed).normal(
            size=len(d.action))
        return np.clip(d.pscore * np.exp(s * z), *PS_CLIP)
    if mode == "estimated":  # in-sample LR — c2b 는 기록 자체가 LR-softmax 라 준-null 예상(§3.6-4)
        clf = LogisticRegression(C=1.0, max_iter=300)
        clf.fit(d.context, d.action)
        proba = clf.predict_proba(d.context)
        col = np.searchsorted(clf.classes_, d.action)
        return np.clip(proba[np.arange(len(d.action)), col], *PS_CLIP)
    raise ValueError(mode)


def main() -> None:
    t0 = time.perf_counter()
    all_rows, truth_rows = [], []
    for dataset_idx, name in enumerate(C2B_DATASETS):
        gt_seen: set[float] = set()
        for seed in SEEDS:
            logs = {delta: classification_to_bandit(name, seed, support_deficiency=delta)
                    for delta in (0.0, 0.2, 0.4)}
            for delta, d in logs.items():
                gt_seen.add(float(d.gt_value))
            for scenario, family, delta, mode, q_sel in SCENARIOS:
                d = logs[delta]
                ps = _pscore_for(mode, d, dataset_idx, seed)
                q_hat = d.q_scores if q_sel == "good" else d.q_scores_degraded
                run_id = f"{name}-{scenario}-{seed}"
                log = PractitionerLog(context=d.context, action=d.action,
                                      reward=d.reward, pscore=ps)
                all_rows += run_protocol(
                    log, d.pi_e_dist, axis_id=AXIS_ID, scenario=f"{name}/{scenario}",
                    run_id=run_id, seed=seed, q_hat=q_hat, cfg=ValidityConfig(seed=seed))
                truth_rows.append({"run_id": run_id, "v_true": float(d.gt_value),
                                   "dataset": name, "scenario": scenario, "family": family})
        assert len(gt_seen) == 1, f"{name}: gt_value 가 시나리오/seed 에 불변이어야 한다"
        print(f"[21] {name} 완료 ({time.perf_counter() - t0:.0f}s)")
    dec_path = write_decision_csv(AXIS_ID, SLUG, all_rows)
    truth = pd.DataFrame(truth_rows)
    rev_path = reveal(dec_path, truth[["run_id", "v_true"]], "exact_c2b")

    # ── matrix (28행 = 4 dataset × 7 scenario — pooled 행 금지, §3.6-6) ───────────
    dec = pd.read_csv(dec_path)
    runs = dec[dec["estimator"] == DECISION_ESTIMATOR].copy()
    rev = pd.read_csv(rev_path)
    rev_runs = rev[rev["estimator"] == DECISION_ESTIMATOR][["run_id", "rel_err", "large_err"]]
    runs = runs.merge(rev_runs, on="run_id", validate="one_to_one")
    meta = truth[["run_id", "dataset", "scenario", "family"]].rename(
        columns={"scenario": "scenario_id"})
    runs = runs.merge(meta, on="run_id", validate="one_to_one")

    def _rates(g: pd.DataFrame) -> pd.Series:
        out = {f"{arm}_fire": (g[f"{arm}_state"] == "fail").mean() for arm in ARMS}
        out["disagreement_inconclusive"] = (g["disagreement_state"] == "inconclusive").mean()
        out["gate_v1_nontrust"] = (g["gate_v1"] != "trust").mean()
        out["verdict_nontrust"] = (g["protocol_verdict"] != "trust").mean()
        out["mean_w_median"] = g["mean_w"].median()
        out["dr_correction_median"] = g["dr_correction"].median()
        out["large_err_rate"] = g["large_err"].mean()
        out["median_rel_err"] = g["rel_err"].median()
        out["runs"] = len(g)
        return pd.Series(out)

    matrix = (runs.groupby(["scenario_id", "family", "dataset"], sort=False)
              .apply(_rates, include_groups=False).reset_index())
    s_order = {s: i for i, (s, *_rest) in enumerate(SCENARIOS)}
    ds_order = {ds: i for i, ds in enumerate(C2B_DATASETS)}
    matrix = matrix.sort_values(
        by=["scenario_id", "dataset"],
        key=lambda col: col.map(s_order if col.name == "scenario_id" else ds_order),
    ).reset_index(drop=True)
    matrix_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    conf = (rev[rev["estimator"] == DECISION_ESTIMATOR]
            .merge(meta, on="run_id", validate="one_to_one")
            .groupby(["dataset", "protocol_verdict"])
            .agg(runs=("rel_err", "size"), share_large_err=("large_err", "mean"),
                 median_rel_err=("rel_err", "median"),
                 ci_covers_rate=("ci_covers_truth", "mean")).reset_index())
    conf_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_confusion.csv"
    conf.to_csv(conf_path, index=False)

    # ── figure: scenario-major heatmap + impossible hatched 밴드 + backstage bars ──
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("fire", ["#fcfcfb", "#44433e"])
    cols = [f"{arm}_fire" for arm in ARMS] + ["gate_v1_nontrust", "verdict_nontrust"]
    col_labels = ["E[w]", "harmonic", "placebo", "disagree", "gate v1\n(≠trust)",
                  "verdict\n(≠trust)"]
    m = matrix[cols].to_numpy(dtype=float)
    n_rows = len(matrix)
    fig, (ax, axr) = plt.subplots(
        1, 2, figsize=(12.0, 10.2), gridspec_kw={"width_ratios": [3.2, 1.0]}, sharey=True)
    ax.imshow(m, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n_rows):
        for j in range(m.shape[1]):
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="#fcfcfb" if m[i, j] > 0.55 else "#33322e")
    ax.set_xticks(range(len(cols)), col_labels, fontsize=8)
    ax.set_yticks(range(n_rows),
                  [f"[{f[:4]}] {s} · {ds}" for s, f, ds in
                   zip(matrix["scenario_id"], matrix["family"], matrix["dataset"])],
                  fontsize=7)
    for i in range(4, n_rows, 4):  # scenario 블록 경계
        ax.axhline(i - 0.5, color="#b3b2a9", lw=0.8)
    prev_family = None
    for i, f in enumerate(matrix["family"]):
        if prev_family is not None and f != prev_family:
            ax.axhline(i - 0.5, color="#6f6e66", lw=1.6)
        prev_family = f
    ax.set_title("A. Battery fire rate — real covariates, pre-registered failure "
                 "scenarios\n(frontstage — logs only; thresholds unchanged from axis 17)",
                 fontsize=10)
    ax.grid(False)
    # impossible family 부재 전시 (§3.6-3 — matrix CSV 에는 가짜 행 없음)
    ax.text(0.5, -0.075,
            "[impo] confounded-calibrated — NOT CONSTRUCTIBLE on real data "
            "(true P(y|x) unknown; the boundary exhibit lives in axis 18, synthetic only)",
            transform=ax.transAxes, ha="center", fontsize=8, style="italic",
            color="#6f6e66",
            bbox={"facecolor": "#fdf3f7", "edgecolor": "#e87ba4", "hatch": "///",
                  "boxstyle": "round,pad=0.45"})
    axr.barh(range(n_rows), matrix["large_err_rate"], color="#e34948", height=0.62)
    axr.set_xlim(0, 1.02)
    axr.set_xlabel(f"P(rel err > 0.10) — {DECISION_ESTIMATOR.upper()}")
    axr.set_title("B. What actually broke\n(backstage reveal — exact c2b truth)",
                  fontsize=10)
    axr.grid(axis="x")
    fig.suptitle("Axis 21 — Does the battery's forecast power replicate on real "
                 "covariates?\n(pre-registered PLAN §3.6; probe M9-A fallback branch: "
                 "satimage support direction refuted — injection strength is set by "
                 "π_e sharpness × masked mass, not δ)", fontsize=11)
    fig.tight_layout(rect=(0, 0.02, 1, 0.91))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── PATTERN 4종 (§3.6-6 사전등록) ────────────────────────────────────────────
    def _cell(scn, ds, col):
        r = matrix[(matrix["scenario_id"] == scn) & (matrix["dataset"] == ds)]
        return float(r[col].iloc[0])

    datasets = list(C2B_DATASETS)
    patterns = {
        "① noised_s10 mean_w fire ≥ 0.9 (전 dataset)":
            all(_cell("noised_s10", ds, "mean_w_fire") >= 0.9 for ds in datasets),
        "② clean any-arm fire ≤ 0.2 (dataset 별)":
            all(max(_cell("clean", ds, f"{a}_fire") for a in ARMS) <= 0.2
                for ds in datasets),
        "③ support_d04 mean_w 중앙값 < 1 (전 dataset — 방향)":
            all(_cell("support_d04", ds, "mean_w_median") < 1.0 for ds in datasets),
        "④ q-채널: |Δfire| ≤ 0.1 ∧ dr_correction(qdeg) > (clean) (전 dataset)":
            all(all(abs(_cell("clean_qdeg", ds, f"{a}_fire")
                        - _cell("clean", ds, f"{a}_fire")) <= 0.1 for a in ARMS)
                and _cell("clean_qdeg", ds, "dr_correction_median")
                > _cell("clean", ds, "dr_correction_median") for ds in datasets),
    }
    print(f"[21] → {dec_path.name}, {rev_path.name}, {matrix_path.name}, {conf_path.name}, "
          f"{fig_path.name} ({time.perf_counter() - t0:.0f}s)")
    for k, ok in patterns.items():
        print(f"[21] PATTERN {'PASS' if ok else 'FAIL'}: {k}")
    print("[21] 실측(방향·질량 렌즈 — probe M9-A): support_d04 mean_w 중앙값 = "
          f"{ {ds: round(_cell('support_d04', ds, 'mean_w_median'), 4) for ds in datasets} }")
    print("[21] estimated_insample 발화(준-null 예상): "
          f"{ {ds: round(_cell('estimated_insample', ds, 'mean_w_fire'), 2) for ds in datasets} }")


if __name__ == "__main__":
    main()
