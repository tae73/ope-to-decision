"""축 04 — deficient support: 미지지(unsupported) π_e 질량의 식별 불능과 진단 proxy 의 한계.

노브: DGPConfig.support_deficiency ∈ {0, 0.1, 0.2, 0.3, 0.4} — 컨텍스트별 하위-q ⌊δK⌋개
액션(K=10 → 0~4개)의 로깅 지지를 구조적으로 제거(per-row 랜덤 아님 — dgp.py 설계 결정).
S=40. q̂ = 0.9·q + 0.05 (mild misspecification — 축 01 과 동일 정책).

기대 패턴(불발 시 그대로 보고 — 은폐 금지): ① IPS 계열 bias 가 δ 와 함께 단조 악화
(π_e 질량이 미지지 영역 — 식별 불능이라 표본을 늘려도 안 낫는다), ② DM 은 q̂ 외삽으로
상대 생존(δ 불감), ③ DR 도 미지지 질량 위의 q̂ 오차는 보정항이 닿지 못해 bias 잔존
("DR 만능" 신화 반박 자리), ④ support proxy(전역 미등장 액션의 π̄_e 질량)는 구조적 mask
에서도 컨텍스트-국소 결핍을 과소 신호 — panel B 에서 oracle 실측치와 대조(그 gap 이 교훈).

실측 기록(2026-08-06 1차 실행): ④ 는 기대보다 강하게 불발 — proxy 가 전 δ 에서 정확히 0
(부분 신호가 아니라 **전면 blind**). K=10·n=10k·β_log=1 에서는 컨텍스트-국소 mask 로는
전역 미등장 액션이 생기지 않는다(모든 액션이 로그 어딘가에 등장). gap 은 '과소'가 아니라
'무신호'다 — 진단 한계 교훈은 오히려 강화되나, "0 은 아니다" 하위 주장은 기각.

산출: results/tables/04_deficient_support.csv (long, _common.COLUMNS)
    + results/tables/04_deficient_support_oracle.csv (companion — panel B oracle 곡선의 정본 수치)
    + results/figures/04_deficient_support.png (2-panel: A bias 부호 유지 / B 진단 vs oracle).
"""

import numpy as np
import pandas as pd

from _common import BASE_M2, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import (ESTIMATOR_ORDER, ORACLE_COLOR, apply_style, end_label, save_figure,
                    series_kwargs)
from ope.dgp import _support_mask, make_synthetic_bandit_data

AXIS_ID, SLUG = "04", "deficient_support"
KNOB = "support_deficiency"
GRID = [0.0, 0.1, 0.2, 0.3, 0.4]
SEEDS = range(1000, 1040)  # S=40
ALPHA_QHAT = 0.9
EMPHASIZED = ("ips", "dm", "dr")
PROXY_COLOR = "#1f1e1b"  # panel B 진단선 — estimator 팔레트와 무관한 중립 근흑(oracle 회색 점선과 명도 대비)


def _spread_labels(ys: dict[str, float], min_gap: float) -> dict[str, float]:
    """끝단 직접 라벨의 y 충돌 회피 — 값(데이터)은 불변, 라벨 anchor 만 최소 간격으로 벌린다."""
    adj: dict[str, float] = {}
    prev = -np.inf
    for k in sorted(ys, key=ys.get):
        adj[k] = max(ys[k], prev + min_gap)
        prev = adj[k]
    return adj


def main() -> None:
    v_true = cached_v_true(BASE_M2)  # π_e·q 는 mask 불변이므로 v_true 는 δ 와 무관
    records = []
    oracle_mass: dict[float, list[float]] = {g: [] for g in GRID}
    for delta in GRID:
        cfg = BASE_M2._replace(support_deficiency=delta)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            records += evaluate_run(AXIS_ID, KNOB, delta, seed, d, q_hat, v_true)
            # oracle 기준선: DGP 의 mask 정의를 읽기 전용 재사용(재구현 불일치 위험 제거).
            # 컨텍스트-국소 진짜 미지지 π_e 질량 = 진단 proxy 가 측정하고 싶은 것의 참값.
            unsupported = ~_support_mask(d.q_true, delta)
            oracle_mass[delta].append(float((d.pi_e_dist * unsupported).sum(axis=1).mean()))
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)
    base = summary[summary["variant"].fillna("") == ""]

    # 진단 proxy 는 (knob, seed)당 동일값이 estimator 7행에 복제 저장 — ips 행만 취해 seed 집계
    df = pd.read_csv(csv_path)
    prox = (df[df["estimator"] == "ips"].groupby("knob_value")["support_proxy"]
            .agg(["mean", "std", "count"]).reindex(GRID))
    prox["se"] = prox["std"] / np.sqrt(prox["count"])
    om_mean = np.array([np.mean(oracle_mass[g]) for g in GRID])
    om_se = np.array([np.std(oracle_mass[g], ddof=1) / np.sqrt(len(SEEDS)) for g in GRID])

    # oracle 곡선의 committed 정본 — LEDGER 인용 가능하도록 companion CSV 로 저장
    # (10_decision_metrics_metrics.csv 관례와 동일 패턴; figure 는 아래 값을 그대로 사용).
    oracle_df = pd.DataFrame({"knob_value": GRID,
                              "oracle_unsupported_mass_mean": om_mean,
                              "oracle_unsupported_mass_se": om_se,
                              "n_seeds": len(SEEDS)})
    oracle_path = csv_path.parent / f"{AXIS_ID}_{SLUG}_oracle.csv"
    oracle_df.to_csv(oracle_path, index=False)

    apply_style()
    import matplotlib.pyplot as plt
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.8, 4.8), constrained_layout=True)

    # Panel A — bias(부호 유지): systematic bias 의 방향이 정보다 (log-MSE 로 뭉개지 않는다)
    ax_a.axhline(0.0, color="#c9c8c0", lw=1.0, zorder=1)
    end_ys: dict[str, float] = {}
    for est in ESTIMATOR_ORDER:
        s = base[base["estimator"] == est].sort_values("knob_value")
        emph = est in EMPHASIZED
        ax_a.plot(s["knob_value"], s["bias"], marker="o", **series_kwargs(est, emph))
        if emph:
            ax_a.fill_between(s["knob_value"], s["bias"] - 2 * s["est_se"],
                              s["bias"] + 2 * s["est_se"],
                              color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
        end_ys[est] = float(s["bias"].iloc[-1])
    ylo, yhi = ax_a.get_ylim()
    adj = _spread_labels(end_ys, 0.05 * (yhi - ylo))
    for est in ESTIMATOR_ORDER:
        end_label(ax_a, GRID[-1], adj[est], est, est in EMPHASIZED)
    ax_a.set_xlim(-0.015, 0.475)  # 우측 직접 라벨 여백
    ax_a.set_xlabel(r"Support deficiency $\delta$ (bottom-$\lfloor\delta K\rfloor$ of $K=10$ actions masked per context)")
    ax_a.set_ylabel(r"Bias $\;\mathrm{E}[\hat V]-V(\pi_e)$ (sign preserved)")
    ax_a.set_title("A — Systematic bias under deficient support")
    ax_a.legend(ncol=2, loc="lower left")

    # Panel B — 진단 신호 vs oracle 참값: proxy 가 얼마나 과소 신호하는지 그 gap 자체를 전시
    ax_b.plot(GRID, om_mean, color=ORACLE_COLOR, ls="--", marker="o",
              label=r"True unsupported $\pi_e$ mass (oracle)")
    ax_b.fill_between(GRID, om_mean - 2 * om_se, om_mean + 2 * om_se,
                      color=ORACLE_COLOR, alpha=0.12, lw=0)
    ax_b.plot(GRID, prox["mean"], color=PROXY_COLOR, marker="o",
              label="Support proxy (log-computable diagnostic)")
    ax_b.fill_between(GRID, prox["mean"] - 2 * prox["se"], prox["mean"] + 2 * prox["se"],
                      color=PROXY_COLOR, alpha=0.12, lw=0)
    for y, txt, c in ((om_mean[-1], "oracle", ORACLE_COLOR),
                      (float(prox["mean"].iloc[-1]), "proxy", PROXY_COLOR)):
        ax_b.annotate(txt, xy=(GRID[-1], y), xytext=(4, 0), textcoords="offset points",
                      va="center", fontsize=7.5, color=c)
    if float(prox["mean"].max()) == 0.0:  # 실측 발견의 정직 전시 — proxy 전면 blind 일 때만
        ax_b.annotate("proxy stays at 0: every action still appears\n"
                      "somewhere in the log (context-local masking)",
                      xy=(0.25, om_mean[-1] * 0.02), xytext=(0.145, om_mean[-1] * 0.20),
                      fontsize=8, color=PROXY_COLOR,
                      arrowprops={"arrowstyle": "->", "color": PROXY_COLOR, "lw": 0.8})
    ax_b.set_xlim(-0.015, 0.44)
    ax_b.set_xlabel(r"Support deficiency $\delta$")
    ax_b.set_ylabel(r"$\pi_e$ mass on unsupported actions")
    ax_b.set_title("B — Diagnostic signal vs ground truth")
    ax_b.legend(loc="upper left")

    fig.suptitle("Axis 04 — Deficient support: unsupported $\\pi_e$ mass is unidentifiable from the log\n"
                 r"structural mask (bottom-$q$ actions per context) · $\hat q = 0.9q+0.05$ · "
                 f"shaded: ensemble $\\pm$2SE over {len(SEEDS)} seeds")
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    piv_b = base.pivot_table(index="knob_value", columns="estimator", values="bias")
    piv_se = base.pivot_table(index="knob_value", columns="estimator", values="est_se")
    ips_mono = bool(np.all(np.diff(np.abs(piv_b["ips"].to_numpy())) > 0))
    dm_shift = abs(piv_b["dm"].loc[GRID[-1]] - piv_b["dm"].loc[GRID[0]])
    ips_shift = abs(piv_b["ips"].loc[GRID[-1]] - piv_b["ips"].loc[GRID[0]])
    dr_resid = bool(abs(piv_b["dr"].loc[GRID[-1]]) > 2 * piv_se["dr"].loc[GRID[-1]])
    proxy_max, oracle_max = float(prox["mean"].loc[GRID[-1]]), float(om_mean[-1])
    proxy_nonzero = bool(proxy_max > 0.0)
    proxy_undersignals = bool(proxy_max < oracle_max)

    print(f"[04] v_true={v_true:.6f}  → {csv_path.name}, {oracle_path.name}, {fig_path.name}")
    for est in EMPHASIZED:
        print(f"[04] BIAS {est}: " + "  ".join(
            f"δ={g:.1f}:{piv_b[est].loc[g]:+.5f}(se {piv_se[est].loc[g]:.5f})" for g in GRID))
    print("[04] SUPPORT proxy:  " + "  ".join(
        f"δ={g:.1f}:{prox['mean'].loc[g]:.5f}" for g in GRID))
    print("[04] SUPPORT oracle: " + "  ".join(
        f"δ={g:.1f}:{m:.5f}" for g, m in zip(GRID, om_mean)))
    print(f"[04] PATTERN ips_bias_monotone_worse={ips_mono}  "
          f"dm_survives_rel_ips={bool(dm_shift < ips_shift)} (|Δdm|={dm_shift:.5f} vs |Δips|={ips_shift:.5f})  "
          f"dr_bias_residual_at_max={dr_resid}  "
          f"proxy_nonzero={proxy_nonzero} proxy_undersignals={proxy_undersignals} "
          f"(proxy {proxy_max:.5f} vs oracle {oracle_max:.5f})")


if __name__ == "__main__":
    main()
