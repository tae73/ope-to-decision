"""축 15 — funnel 신뢰도 사다리: 같은 로그가 CTR→CVR→REV 로 갈수록 말할 수 있는 것이 급감.

설계(플랜 M6 §2 정본): FunnelConfig(probe GO 파라미터 — base_ctr=0.01·amp_ctr=0.06·base_cvr=0.05·
amp_cvr=0.25·β_log=2·β_eval=6·K=10·d=5) × n∈{2000, 10000, 50000} × S=40(seed 1000–1039).
같은 로그에서 지표 3종(CTR · CVR **세션 기준**=click·conv · REV)을 각각 IPS·SNIPS·DR 로 평가.
DR 의 q̂_m 은 oracle 확률(p_c · p_c·p_v — business.py 필드) + price 기지(플랫폼 기지값 규약).
오지정 variant 는 **확률에 0.9-shrink 1종만**(p_c→0.9·p_c, p_v→0.9·p_v — price 불변),
DR 행에만 variant="m:<metric>|q:shrink0.9" 로 추가(IPS·SNIPS 는 q̂ 미사용).

판별한계(MDE 명칭 회피 — "relative discrimination limit"):
    rel_limit = 2.8·SE(Δ̂) / V_m(π_0),  Δ̂ = V̂_m(π_e) − mean(r_m)
(incumbent 추정 = on-policy 로그 평균 — 축 10 결정측 규약, 2.8 ≈ z_{0.975}+z_{0.80}).
SE(Δ̂)는 seed-ensemble SD(단일 로그 추정기의 표집 SE 를 S=40 replicate 로 직접 추정 —
합성 MC 축이므로 bootstrap 금지 규약). band 는 SD 의 정규근사 SE(=SD/√(2(S−1)))·heavy tail 에서
과소 가능 → **p90 실현 상대오차 |Δ̂−Δ_true|/V_m(π_0) 병기**(계약: band+p90).

스키마: 주 CSV 는 _common.COLUMNS 그대로 — variant="m:ctr" 식 metric 인코딩, estimate 는
**지표 원단위 V̂_m(π_e)**(상대 오차 비교용), v_true=V_m(π_e), hyperparam_value=NaN(세 estimator
모두 무 hyperparam). **companion CSV 예외(문서화)**: 15_funnel_reliability_events.csv —
[knob, seed, metric, n_events, onpolicy_mean, v_true_pi0]. n_events 는 지표별 원시 이벤트 수
(ctr=클릭 수, cvr=전환 수, rev=양의 매출 행 수 — 0-이벤트 seed 퇴화의 정직 기록, 축 12 계보).
onpolicy_mean·v_true_pi0 두 컬럼은 계약 목록 외 추가분 — figure 의 Δ̂·Δ_true·분모가 committed
산출물에서 재도출 가능하도록 넣는다(수치 주장은 committed 산출물에서만 규약).

진단 지표-불변은 **구성적 사실**(진단은 weight 만 보고 reward 를 안 본다 — 로그당 1회 계산해
세 지표 행에 동일 기입): 패널 B 의 수평선은 정의상 평평하고, 경험적 내용은 지표별 상대 오차
분포가 몇 배 벌어진다는 쪽이다(축 08/09 계보 확장).

리텐션 사다리 제외(figure 각주): 세션 간 지표는 single-step bandit OPE 로 식별 불가 —
RL OPE 소관(PLAYBOOK 경계 선언).

기대 패턴: CTR→CVR→REV 판별한계 단조 악화(이벤트 희소 + price heavy tail). 불발 시 그대로
보고(은폐 금지).

산출: results/tables/15_funnel_reliability.csv (+ _events.csv)
    + results/figures/15_funnel_reliability.png.
"""

import csv

import numpy as np
import pandas as pd

from _common import TAB_DIR, RunRecord, write_axis_csv
from _style import CONTEXT_GRAY, ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style, save_figure
from ope.business import METRICS, FunnelConfig, funnel_ground_truth, make_funnel_bandit_data
from ope.diagnostics import (PROVISIONAL_THRESHOLDS, DiagnosticsReport, compute_diagnostics,
                             decision_gate)
from ope.estimators import estimate_dr, estimate_ips, estimate_snips

AXIS_ID, SLUG = "15", "funnel_reliability"
KNOB = "n"
GRID = [2000, 10000, 50000]
SEEDS = range(1000, 1040)  # S=40
BASE = FunnelConfig(n=10_000, n_actions=10, dim_context=5, beta_log=2.0, beta_eval=6.0)
SHRINK = 0.9               # DR q̂ 오지정 1종: 확률에만 0.9-shrink (price 는 플랫폼 기지)
MDE_MULT = 2.8             # z_{0.975}+z_{0.80} ≈ 1.96+0.84 — 관례 배수 ("판별한계" 명칭)
ESTS = ("ips", "snips", "dr")
METRIC_TITLES = {"ctr": "CTR (click)", "cvr": "CVR (session-level: click·conv)",
                 "rev": "REV (conv · price)"}
N_PANEL_B = 10_000         # 패널 B 는 base n 고정(중앙 rung — probe 와 동일)


def _q_hat(metric: str, p_c: np.ndarray, p_v: np.ndarray, price: np.ndarray,
           shrink: float = 1.0) -> np.ndarray:
    """지표별 q̂_m — 확률만 shrink 오지정 가능, price 는 항상 기지(플랫폼 기지값 규약)."""
    pc, pv = shrink * p_c, shrink * p_v
    if metric == "ctr":
        return pc
    if metric == "cvr":
        return pc * pv
    return pc * pv * price[None, :]


def _ess_used(weights: np.ndarray, n: int) -> float:
    s, q = float(weights.sum() ** 2), float((weights**2).sum())
    return (s / q / n) if q > 0 else 0.0


def main() -> None:
    gt_pie = funnel_ground_truth(BASE)                       # V_m(π_e) — 200k MC
    gt_pi0 = funnel_ground_truth(BASE, beta=BASE.beta_log)   # V_m(π_0) — Δ_true·상대화 분모

    records: list[RunRecord] = []
    events: list[list] = []
    for n in GRID:
        for seed in SEEDS:
            d = make_funnel_bandit_data(BASE._replace(n=n, seed=seed))
            diag = compute_diagnostics(d.pscore, d.action, d.pi_e_dist)  # 로그당 1회 — 지표 불변
            for metric in METRICS:
                r = d.rewards[metric]
                args = (r, d.action, d.pscore, d.pi_e_dist)
                runs = {
                    ("ips", f"m:{metric}"): estimate_ips(*args),
                    ("snips", f"m:{metric}"): estimate_snips(*args),
                    ("dr", f"m:{metric}"): estimate_dr(*args, _q_hat(metric, d.p_c, d.p_v,
                                                                     d.price)),
                    ("dr", f"m:{metric}|q:shrink0.9"): estimate_dr(
                        *args, _q_hat(metric, d.p_c, d.p_v, d.price, SHRINK)),
                }
                for (est, variant), res in runs.items():
                    records.append(RunRecord(
                        axis_id=AXIS_ID, knob_name=KNOB, knob_value=n, seed=seed,
                        estimator=est, variant=variant, hyperparam_value=np.nan,
                        estimate=res.value, v_true=gt_pie[metric],
                        ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
                        support_proxy=diag.support_deficiency,
                        ess_ratio_used=_ess_used(res.weights, n)))
                n_events = int((r > 0).sum())  # ctr=클릭, cvr=전환, rev=양의 매출 행(=전환)
                events.append([n, seed, metric, n_events, float(r.mean()), gt_pi0[metric]])

    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    ev_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_events.csv"
    with ev_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["knob", "seed", "metric", "n_events", "onpolicy_mean", "v_true_pi0"])
        w.writerows(events)

    # ── 분석: committed CSV 에서 재도출(figure 수치의 출처 고정) ────────────────
    df = pd.read_csv(csv_path)
    ev = pd.read_csv(ev_path)
    df["metric"] = df["variant"].str.split("|").str[0].str.replace("m:", "", regex=False)
    df["series"] = np.where(df["variant"].str.contains("|q:", regex=False),
                            df["estimator"] + "_shrink", df["estimator"])
    df = df.merge(ev, left_on=["knob_value", "seed", "metric"],
                  right_on=["knob", "seed", "metric"], how="left")
    df["delta_hat"] = df["estimate"] - df["onpolicy_mean"]
    df["delta_true"] = df["v_true"] - df["v_true_pi0"]
    df["rel_err_delta"] = (df["delta_hat"] - df["delta_true"]).abs() / df["v_true_pi0"]
    df["rel_err_v"] = (df["estimate"] - df["v_true"]).abs() / df["v_true"]

    S = len(list(SEEDS))
    lad = (df.groupby(["knob_value", "metric", "series"])
             .agg(sd_delta=("delta_hat", lambda s: s.std(ddof=1)),
                  p90_rel=("rel_err_delta", lambda s: s.quantile(0.9)),
                  v_pi0=("v_true_pi0", "first"))
             .reset_index())
    lad["rel_limit"] = MDE_MULT * lad["sd_delta"] / lad["v_pi0"]
    lad["rel_limit_se"] = lad["rel_limit"] / np.sqrt(2 * (S - 1))  # SD 정규근사 SE — 과소 가능

    # 게이트 verdict — 지표 무관(weight 만 봄): (n, seed)당 1회
    one = df[(df["series"] == "ips") & (df["metric"] == "ctr")]
    verdicts = {}
    for n in GRID:
        g = one[one["knob_value"] == n]
        vs = [decision_gate(DiagnosticsReport(ess=np.nan, ess_ratio=r.ess_ratio_raw,
                                              max_weight=r.max_weight_raw,
                                              weight_tail_p99=np.nan,
                                              support_deficiency=r.support_proxy),
                            PROVISIONAL_THRESHOLDS).decision
              for r in g.itertuples()]
        verdicts[n] = pd.Series(vs).value_counts().to_dict()

    # ── figure (1 PNG · 2패널: A=사다리 3 small multiples / B=진단 불변 vs 오차 분포) ──
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter
    fig = plt.figure(figsize=(12.8, 9.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.62, wspace=0.10)

    ax0 = fig.add_subplot(gs[0, 0])
    axes_a = [ax0, fig.add_subplot(gs[0, 1], sharey=ax0), fig.add_subplot(gs[0, 2], sharey=ax0)]
    for ax, metric in zip(axes_a, METRICS):
        sub = lad[lad["metric"] == metric]
        for est in ESTS:
            s = sub[sub["series"] == est].sort_values("knob_value")
            c = ESTIMATOR_COLORS[est]
            ax.plot(s["knob_value"], s["rel_limit"], marker="o", color=c, lw=2.0,
                    label=f"{ESTIMATOR_LABELS[est]} — 2.8·SE(Δ̂)/V(π₀)")
            ax.fill_between(s["knob_value"], s["rel_limit"] - 2 * s["rel_limit_se"],
                            s["rel_limit"] + 2 * s["rel_limit_se"], color=c, alpha=0.15, lw=0)
            ax.plot(s["knob_value"], s["p90_rel"], marker="v", mfc="none", color=c, lw=1.1,
                    ls=":", label=f"{ESTIMATOR_LABELS[est]} — p90 |Δ̂−Δ|/V(π₀)")
        s = sub[sub["series"] == "dr_shrink"].sort_values("knob_value")
        ax.plot(s["knob_value"], s["rel_limit"], marker="o", color=CONTEXT_GRAY, lw=1.2,
                label="DR, q̂ 0.9-shrink — 2.8·SE(Δ̂)/V(π₀)")
        true_lift = (gt_pie[metric] - gt_pi0[metric]) / gt_pi0[metric]
        ax.axhline(true_lift, color="#a33b3b", ls="-.", lw=1.2)
        ax.annotate(f"true relative lift {true_lift:.1%}", xy=(GRID[0], true_lift),
                    xytext=(2, 4), textcoords="offset points", fontsize=7.5, color="#a33b3b")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(GRID, ["2k", "10k", "50k"])
        ax.xaxis.set_minor_formatter(NullFormatter())   # log 보조 눈금 라벨 억제(라벨 충돌)
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("log size n")
        ax.set_title(METRIC_TITLES[metric], loc="left", fontsize=10)
    axes_a[0].set_ylabel("relative discrimination limit (log)")
    axes_a[0].set_yticks([0.02, 0.05, 0.10, 0.20, 0.50],
                         ["2%", "5%", "10%", "20%", "50%"])
    for ax in axes_a[1:]:
        ax.tick_params(which="both", labelleft=False)
    axes_a[0].set_title("A. Reliability ladder — the deeper the metric, the less the same log can say\n"
                        + METRIC_TITLES["ctr"], loc="left", fontsize=10)
    handles_a, labels_a = axes_a[0].get_legend_handles_labels()
    fig.legend(handles_a, labels_a, loc="center", bbox_to_anchor=(0.5, 0.475), ncol=4,
               fontsize=7.2)

    # 패널 B: 진단은 지표 불변(수평선·구성적) vs 지표별 |상대 오차| 분포는 몇 배 벌어짐
    axb = fig.add_subplot(gs[1, :])
    db = df[(df["knob_value"] == N_PANEL_B) & (~df["series"].str.endswith("_shrink"))]
    positions, box_data, box_colors = [], [], []
    for mi, metric in enumerate(METRICS):
        for ei, est in enumerate(ESTS):
            vals = db[(db["metric"] == metric) & (db["series"] == est)]["rel_err_v"]
            positions.append(mi + (ei - 1) * 0.24)
            box_data.append(vals.clip(lower=1e-6).values)
            box_colors.append(ESTIMATOR_COLORS[est])
    bp = axb.boxplot(box_data, positions=positions, widths=0.18, patch_artist=True,
                     medianprops={"color": "#33322e", "lw": 1.4},
                     whiskerprops={"color": "#8a897f"}, capprops={"color": "#8a897f"},
                     flierprops={"marker": ".", "markersize": 3, "alpha": 0.6})
    for patch, c in zip(bp["boxes"], box_colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.45)
        patch.set_edgecolor(c)
    ess_flat = float(db[db["series"] == "ips"].groupby("seed")["ess_ratio_raw"].first().mean())
    axb.axhline(ess_flat, color="#3f3e39", lw=1.8, ls="--")
    axb.annotate(f"ESS/n = {ess_flat:.3f} — identical for CTR·CVR·REV (diagnostics see only "
                 "weights, never rewards)", xy=(2.42, ess_flat), xytext=(0, 5), ha="right",
                 textcoords="offset points", fontsize=8, color="#3f3e39")
    vd = verdicts[N_PANEL_B]
    vd_txt = ", ".join(f"{k} {v}/{S}" for k, v in sorted(vd.items()))
    axb.annotate(f"gate verdict (pre-registered thresholds): {vd_txt} — same verdict for every "
                 "metric (by construction)", xy=(0.01, 0.05), xycoords="axes fraction",
                 fontsize=8, color="#44433e")
    axb.set_yscale("log")
    axb.set_xticks(range(len(METRICS)),
                   [METRIC_TITLES[m] for m in METRICS])
    axb.set_ylabel(r"|relative error| of $\hat V_m(\pi_e)$ (log)")
    axb.set_title(f"B. Diagnostics are metric-invariant, error distributions are not "
                  f"(n={N_PANEL_B:,}, {S} seeds — axes 08/09 lineage)", loc="left", fontsize=10)
    handles = [plt.Line2D([], [], color=ESTIMATOR_COLORS[e], lw=5, alpha=0.45,
                          label=ESTIMATOR_LABELS[e]) for e in ESTS]
    axb.legend(handles=handles, loc="lower right", fontsize=7.5, ncol=3)

    fig.suptitle("Axis 15 — Funnel reliability ladder: same log, three metrics, shrinking say\n"
                 f"funnel DGP (probe GO): base CTR 1%·CVR|click 5–30%·β=({BASE.beta_log:g},"
                 f"{BASE.beta_eval:g})·K={BASE.n_actions} · Δ̂ vs on-policy incumbent mean · "
                 f"band: ±2·SE of seed-ensemble SD (S={S}) · p90 realized error alongside "
                 "(heavy-tail check)", y=1.00)
    fig.text(0.01, 0.005,
             "Footnote — retention is excluded from this ladder: cross-session metrics are not "
             "identifiable from single-step logged bandit feedback (RL OPE territory; see "
             "PLAYBOOK boundary declaration).", fontsize=7.5, color="#6f6e66")
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ────────────────
    print(f"[15] GT π_e={ {m: round(gt_pie[m], 6) for m in METRICS} } "
          f"π_0={ {m: round(gt_pi0[m], 6) for m in METRICS} } → {csv_path.name}, "
          f"{ev_path.name}, {fig_path.name}")
    lim = {(r.knob_value, r.metric, r.series): r.rel_limit for r in lad.itertuples()}
    for n in GRID:
        mono = {est: lim[(n, "ctr", est)] < lim[(n, "cvr", est)] < lim[(n, "rev", est)]
                for est in ESTS}
        vals = "  ".join(f"{est}: ctr={lim[(n, 'ctr', est)]:.3f} cvr={lim[(n, 'cvr', est)]:.3f} "
                         f"rev={lim[(n, 'rev', est)]:.3f} mono={mono[est]}" for est in ESTS)
        print(f"[15] PATTERN ladder n={n}: {vals}")
    ev_min = ev.groupby(["knob", "metric"])["n_events"].min().unstack()
    zero_rows = ev[ev["n_events"] == 0]
    print(f"[15] EVENTS min per (n, metric):\n{ev_min}")
    print(f"[15] EVENTS zero-event (knob, seed, metric) rows: {len(zero_rows)}"
          + ("" if zero_rows.empty else f" → {zero_rows[['knob', 'seed', 'metric']].values.tolist()}"))
    med = db.groupby(["metric", "series"])["rel_err_v"].median().unstack()
    spread = {est: float(med.loc["rev", est] / med.loc["ctr", est]) for est in ESTS}
    print(f"[15] PATTERN error fan-out @n={N_PANEL_B} (median |rel err| rev/ctr): "
          + "  ".join(f"{e}×{spread[e]:.1f}" for e in ESTS))
    print(f"[15] PATTERN diag invariant: by construction (1 diagnostics per log, stamped on all "
          f"metric rows) — ESS/n@10k={ess_flat:.4f}; gate verdicts per n: {verdicts}")


if __name__ == "__main__":
    main()
