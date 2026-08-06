"""축 09 — confounding blind spot: 표준 진단은 평평한데 bias 만 성장 (hero ③ 후보).

노브: DGPConfig.confounding_strength γ ∈ {0, 0.5, 1, 1.5, 2, 2.5}, n=30,000 고정. S=40.
q̂ = 0.9·q + 0.05 (축 01 과 동일 mild misspecification). γ 마다 oracle-IPS 를 병기 —
pscore=`pscore_true`(U-조건부 진짜 propensity)로 IPS 만 재평가, variant="oracle_ps"
(dgp.py 정의: oracle-IPS 가 참값을 복원하는 propensity — U-주변화 아님).

기대 패턴: (A) logged-pscore 기준 ESS/n·max-weight 는 γ 에 거의 불변 — 로그만 보는 진단이
원리적으로 못 보는 실패 모드. (B) logged-pscore 계열 bias 는 γ 와 함께 성장(부호 유지),
oracle-IPS 는 전 구간 0 근방. 등식 주장 금지(범위 주장) — A 패널에 ESS 원수치를 주석한다.
soft 임계선 0.10 은 본 레포의 [제안](문헌 표준 아님 — CLAUDE.md §5 프레임). 불발 시 그대로
보고(은폐 금지).

산출: results/tables/09_confounding_blindspot.csv (long, _common.COLUMNS)
    + results/figures/09_confounding_blindspot.png (2패널: A 진단 / B bias).
"""

import numpy as np
import pandas as pd

from _common import (BASE_M2, RunRecord, cached_v_true, evaluate_run, summarize,
                     write_axis_csv)
from _style import (ESTIMATOR_ORDER, ORACLE_COLOR, apply_style, end_label,
                    save_figure, series_kwargs)
from ope.dgp import make_synthetic_bandit_data
from ope.diagnostics import compute_diagnostics
from ope.estimators import estimate_ips

AXIS_ID, SLUG = "09", "confounding_blindspot"
KNOB = "confounding_strength"
GRID = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5]
N_FIXED = 30_000
SEEDS = range(1000, 1040)  # S=40
ALPHA_QHAT = 0.9
EMPHASIZED = ("ips", "dr", "snips")
ESS_SOFT_THRESHOLD = 0.10  # [제안] soft 임계 — 본 레포 제안값, 문헌 표준 아님
DIAG_COLOR = "#3f3e39"     # 진단 곡선(estimator 아님 — entity 팔레트 밖 중립색)


def _oracle_ips_record(gamma: float, seed: int, d, v_true: float) -> RunRecord:
    """pscore_true 로 IPS 만 재평가한 한 행 (variant="oracle_ps", RunRecord 스키마 준수).

    진단(raw)도 pscore_true 기준 — oracle 이 실제 쓴 weight 의 진단을 CSV 에 남긴다.
    """
    res = estimate_ips(d.reward, d.action, d.pscore_true, d.pi_e_dist)
    diag = compute_diagnostics(d.pscore_true, d.action, d.pi_e_dist)
    w = res.weights
    s, q = float(w.sum() ** 2), float((w**2).sum())
    ess_used = (s / q / len(d.action)) if q > 0 else 0.0
    return RunRecord(
        axis_id=AXIS_ID, knob_name=KNOB, knob_value=gamma, seed=seed,
        estimator="ips", variant="oracle_ps", hyperparam_value=np.nan,
        estimate=res.value, v_true=v_true,
        ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
        support_proxy=diag.support_deficiency, ess_ratio_used=ess_used)


def main() -> None:
    v_true = cached_v_true(BASE_M2)
    records = []
    for gamma in GRID:
        cfg = BASE_M2._replace(n=N_FIXED, confounding_strength=gamma)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            records += evaluate_run(AXIS_ID, KNOB, gamma, seed, d, q_hat, v_true)
            records.append(_oracle_ips_record(gamma, seed, d, v_true))
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    # 진단(logged-pscore 기준 raw)은 estimator 공통 — ips 행에서 seed-ensemble mean±SE 재도출
    df = pd.read_csv(csv_path)
    base_ips = df[df["variant"].isna() & (df["estimator"] == "ips")]
    diag = base_ips.groupby("knob_value").agg(
        ess_mean=("ess_ratio_raw", "mean"),
        ess_se=("ess_ratio_raw", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
        mw_mean=("max_weight_raw", "mean"),
        mw_se=("max_weight_raw", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
    ).reset_index().sort_values("knob_value")
    base_sum = summary[summary["variant"].fillna("") == ""]
    oracle_sum = summary[summary["variant"] == "oracle_ps"].sort_values("knob_value")

    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11.8, 5.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.15], height_ratios=[2.1, 1.0],
                          hspace=0.5, wspace=0.3)
    ax_ess = fig.add_subplot(gs[0, 0])
    ax_mw = fig.add_subplot(gs[1, 0])
    ax_bias = fig.add_subplot(gs[:, 1])

    # 패널 A(상): ESS/n — 거의 평평(원수치 주석 = 범위 주장, 등식 주장 아님)
    ax_ess.plot(diag["knob_value"], diag["ess_mean"], marker="o", color=DIAG_COLOR,
                lw=2.0, label="ESS/n of raw logged-pscore weights")
    ax_ess.fill_between(diag["knob_value"], diag["ess_mean"] - 2 * diag["ess_se"],
                        diag["ess_mean"] + 2 * diag["ess_se"],
                        color=DIAG_COLOR, alpha=0.15, lw=0)
    for xv, yv in zip(diag["knob_value"], diag["ess_mean"]):
        ax_ess.annotate(f"{yv:.3f}", xy=(xv, yv), xytext=(0, 7), ha="center",
                        textcoords="offset points", fontsize=7, color=DIAG_COLOR)
    ax_ess.axhline(ESS_SOFT_THRESHOLD, color="#a33b3b", ls=":", lw=1.2)
    ax_ess.annotate("soft threshold 0.10 (proposed in this repo — not a literature standard)",
                    xy=(GRID[0], ESS_SOFT_THRESHOLD), xytext=(2, 4),
                    textcoords="offset points", fontsize=7.5, color="#a33b3b")
    ax_ess.set_ylim(0, 1)
    ax_ess.set_xticks(GRID)
    ax_ess.set_ylabel("ESS / n")
    ax_ess.set_title("A. Standard diagnostics stay flat as confounding grows", loc="left")
    ax_ess.legend(loc="center right")

    # 패널 A(하): max raw weight 소패널 (보조 y축 금지 규약 — 별도 소패널로 병기)
    ax_mw.plot(diag["knob_value"], diag["mw_mean"], marker="o", color=DIAG_COLOR, lw=1.6,
               label="max raw weight (same logged-pscore weights)")
    ax_mw.fill_between(diag["knob_value"], diag["mw_mean"] - 2 * diag["mw_se"],
                       diag["mw_mean"] + 2 * diag["mw_se"],
                       color=DIAG_COLOR, alpha=0.15, lw=0)
    for xv, yv in zip(diag["knob_value"].iloc[[0, -1]], diag["mw_mean"].iloc[[0, -1]]):
        ax_mw.annotate(f"{yv:.1f}", xy=(xv, yv), xytext=(0, 6), ha="center",
                       textcoords="offset points", fontsize=7, color=DIAG_COLOR)
    ax_mw.set_ylim(bottom=0)
    ax_mw.set_xticks(GRID)
    ax_mw.set_ylabel("max weight")
    ax_mw.set_xlabel(r"Confounding strength $\gamma$")
    ax_mw.legend(loc="lower right")

    # 패널 B: bias(부호 유지) — logged-pscore 계열 성장 vs oracle-IPS 0 근방
    ax_bias.axhline(0.0, color="#8a897f", lw=0.8, zorder=1)
    for est in ESTIMATOR_ORDER:
        s = base_sum[base_sum["estimator"] == est].sort_values("knob_value")
        emph = est in EMPHASIZED
        ax_bias.plot(s["knob_value"], s["bias"], marker="o", **series_kwargs(est, emph))
        if emph:
            ax_bias.fill_between(s["knob_value"], s["bias"] - 2 * s["est_se"],
                                 s["bias"] + 2 * s["est_se"],
                                 color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
        end_label(ax_bias, s["knob_value"].iloc[-1], s["bias"].iloc[-1], est, emph)
    o = oracle_sum
    ax_bias.plot(o["knob_value"], o["bias"], marker="o", ls="--", color=ORACLE_COLOR,
                 lw=2.0, zorder=4, label="Oracle-IPS (true pscore)")
    ax_bias.fill_between(o["knob_value"], o["bias"] - 2 * o["est_se"],
                         o["bias"] + 2 * o["est_se"], color=ORACLE_COLOR, alpha=0.15, lw=0)
    ax_bias.annotate("Oracle-IPS", xy=(o["knob_value"].iloc[-1], o["bias"].iloc[-1]),
                     xytext=(4, 0), textcoords="offset points", va="center",
                     fontsize=7.5, color=ORACLE_COLOR)
    ax_bias.set_xticks(GRID)
    ax_bias.set_xlabel(r"Confounding strength $\gamma$")
    ax_bias.set_ylabel(r"Bias (mean estimate $-\ V(\pi_e)$, sign kept)")
    ax_bias.set_title(r"B. Bias grows with $\gamma$; oracle-IPS stays near zero", loc="left")
    ax_bias.legend(ncol=2, loc="upper left")

    fig.suptitle("Axis 09 — Confounding blind spot: what standard OPE diagnostics cannot see\n"
                 r"$\hat q = 0.9q+0.05$ · $n$=30,000 · shaded: ensemble $\pm$2SE over "
                 f"{len(SEEDS)} seeds", y=1.04)
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    ess = dict(zip(diag["knob_value"], diag["ess_mean"]))
    mw = dict(zip(diag["knob_value"], diag["mw_mean"]))
    g0, g1 = GRID[0], GRID[-1]
    ess_drift = max(abs(ess[g] - ess[g0]) for g in GRID)
    mw_rel_drift = max(abs(mw[g] - mw[g0]) / mw[g0] for g in GRID)
    piv_b = base_sum.pivot_table(index="knob_value", columns="estimator", values="bias")
    piv_se = base_sum.pivot_table(index="knob_value", columns="estimator", values="est_se")
    grows = {est: (abs(piv_b.loc[g1, est]) >
                   abs(piv_b.loc[g0, est]) + 2 * (piv_se.loc[g0, est] + piv_se.loc[g1, est]))
             for est in EMPHASIZED}
    oracle_max_abs = float(o["bias"].abs().max())
    oracle_within = bool((o["bias"].abs() <= 2 * o["est_se"]).all())

    print(f"[09] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[09] PATTERN diag_flat: ess/n {ess[g0]:.4f}@γ=0 → {ess[g1]:.4f}@γ=2.5, "
          f"max|Δ|={ess_drift:.4f}  flat(<0.02)={ess_drift < 0.02}")
    print(f"[09] PATTERN maxw_flat: {mw[g0]:.2f}@γ=0 → {mw[g1]:.2f}@γ=2.5, "
          f"max relΔ={mw_rel_drift:.3f}  flat(<0.10)={mw_rel_drift < 0.10}")
    print(f"[09] PATTERN bias_grows(beyond 2SE): "
          + "  ".join(f"{est}: {piv_b.loc[g0, est]:+.4f}@0 → {piv_b.loc[g1, est]:+.4f}@2.5 "
                      f"grows={grows[est]} neg={piv_b.loc[g1, est] < 0}"
                      for est in EMPHASIZED))
    print(f"[09] PATTERN oracle_clean: max|bias|={oracle_max_abs:.4f}, "
          f"all_within_2SE={oracle_within}")


if __name__ == "__main__":
    main()
