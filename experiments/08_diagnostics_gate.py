"""축 08 — 진단 예보력 + 사전등록 decision gate 평가 (축 01–07 pooling).

설계(순환 회피): PROVISIONAL_THRESHOLDS 는 M1 에서 folklore 로 **사전 등록**된 값 — 본 축은 그
값을 평가만 하고 **교정하지 않는다**(교정하려면 seed-split 필수, M3+ 소관). 축 09 는 pooling 에서
제외하고 별도 패널(blind spot). 축 10 은 결정 metric 축이라 제외. DM 은 weight 미사용이라
negative control 로 분리(ESS 와 무관해야 정상).

대오차 정의 [제안 — 표준 아님]: 상대오차 |V̂−v_true|/v_true > 0.10.
실측 예고: support_proxy 는 전 축에서 0(축 04 이 실증한 전면 blind) — gate 의 support arm 은
pooled 데이터에서 한 번도 발화하지 않으며, 실효 arm 은 ESS·max-weight 다(figure 에 명시).

산출: results/tables/08_diagnostics_gate.csv (pooled per-run + gate verdict) +
      results/tables/08_diagnostics_gate_confusion.csv + results/figures/08_diagnostics_gate.png.
"""

import numpy as np
import pandas as pd

from _common import TAB_DIR, write_axis_csv  # noqa: F401  (TAB_DIR 경로 재사용)
from _style import ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style, save_figure
from ope.diagnostics import PROVISIONAL_THRESHOLDS, DiagnosticsReport, decision_gate

AXIS_ID, SLUG = "08", "diagnostics_gate"
POOL = ["01_sample_size", "02_logging_beta", "03_policy_gap", "04_deficient_support",
        "05_propensity_misspec", "06_reward_misspec", "07_hyperparam_ieoe"]
BLIND = "09_confounding_blindspot"
LARGE_ERR = 0.10  # [제안] 대오차 임계
WEIGHTING = ["ips", "snips", "clipped_ips", "dr", "switch_dr", "dros"]


def _gate_row(r) -> str:
    rep = DiagnosticsReport(ess=np.nan, ess_ratio=r["ess_ratio_raw"],
                            max_weight=r["max_weight_raw"], weight_tail_p99=np.nan,
                            support_deficiency=r["support_proxy"])
    return decision_gate(rep, PROVISIONAL_THRESHOLDS).decision


def main() -> None:
    frames = []
    for slug in POOL:
        df = pd.read_csv(TAB_DIR / f"{slug.split('_')[0]}_{'_'.join(slug.split('_')[1:])}.csv")
        df["source_axis"] = slug[:2]
        frames.append(df)
    pool = pd.concat(frames, ignore_index=True)
    pool = pool[pool["estimator"].isin(WEIGHTING + ["dm"])].copy()
    pool["rel_err"] = (pool["estimate"] - pool["v_true"]).abs() / pool["v_true"]
    pool["gate"] = pool.apply(_gate_row, axis=1)
    pool["large_err"] = pool["rel_err"] > LARGE_ERR

    out_cols = ["source_axis", "knob_name", "knob_value", "seed", "estimator", "variant",
                "ess_ratio_raw", "max_weight_raw", "support_proxy", "rel_err", "gate",
                "large_err"]
    csv_path = TAB_DIR / f"{AXIS_ID}_{SLUG}.csv"
    pool[out_cols].to_csv(csv_path, index=False)

    w = pool[pool["estimator"].isin(WEIGHTING)]
    dmn = pool[pool["estimator"] == "dm"]
    conf = (w.groupby("gate")
             .agg(runs=("rel_err", "size"), share_large_err=("large_err", "mean"),
                  median_rel_err=("rel_err", "median"), p90_rel_err=("rel_err", lambda s: s.quantile(0.9)))
             .reset_index())
    conf_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_confusion.csv"
    conf.to_csv(conf_path, index=False)

    blind = pd.read_csv(TAB_DIR / f"{BLIND[:2]}_{BLIND[3:]}.csv")
    blind = blind[(blind["estimator"].isin(["ips", "dr", "snips"])) & (blind["variant"].fillna("") == "")]
    blind["rel_err"] = (blind["estimate"] - blind["v_true"]).abs() / blind["v_true"]
    blind["gate"] = blind.apply(_gate_row, axis=1)
    bg = (blind.groupby("knob_value")
          .agg(trust_share=("gate", lambda s: (s == "trust").mean()),
               mean_rel_err=("rel_err", "mean")).reset_index())

    # ── figure ───────────────────────────────────────────────────────────────
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1.4, 1.0, 1.0]})

    ax = axes[0]  # A: 예보력 산점 (weighting family)
    for est in WEIGHTING:
        g = w[w["estimator"] == est]
        ax.scatter(g["ess_ratio_raw"], g["rel_err"].clip(lower=1e-5), s=3, alpha=0.12,
                   color=ESTIMATOR_COLORS[est], edgecolors="none")
    bins = np.geomspace(w["ess_ratio_raw"].clip(lower=1e-4).min(), 1.0, 18)
    binned = w.groupby(pd.cut(w["ess_ratio_raw"], bins), observed=True)["rel_err"].median()
    centers = [iv.mid for iv in binned.index]
    ax.plot(centers, binned.values, color="#33322e", lw=2.2, label="binned median")
    for x, style, lab in [(PROVISIONAL_THRESHOLDS.ess_ratio_hard, ":", "hard 0.01 (pre-registered)"),
                          (PROVISIONAL_THRESHOLDS.ess_ratio_soft, "--", "soft 0.10 (pre-registered)")]:
        ax.axvline(x, color="#6f6e66", linestyle=style, lw=1.2, label=lab)
    ax.axhline(LARGE_ERR, color="#e34948", lw=1.0, ls="-.", label=f"large error {LARGE_ERR}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("raw ESS ratio (log)"); ax.set_ylabel("|relative error| (log)")
    ax.set_title("A. Diagnostics forecast error — weighting family\n(axes 01–07 pooled; "
                 "support arm never fires: proxy ≡ 0)")
    ax.legend(loc="lower left", fontsize=7)

    ax = axes[1]  # B: gate confusion + DM negative control
    # bar 색은 팔레트 밖 중립색 — verdict 는 estimator entity 가 아니므로 고정 색 매핑 재사용 금지
    xs = np.arange(len(conf))
    ax.bar(xs, conf["share_large_err"], width=0.55, color="#44433e")
    for i, row in conf.iterrows():
        ax.annotate(f"n={row['runs']}", xy=(i, row["share_large_err"]), xytext=(0, 4),
                    textcoords="offset points", ha="center", fontsize=7.5)
    ax.set_xticks(xs, conf["gate"])
    dm_rate = dmn["large_err"].mean()
    ax.axhline(dm_rate, color=ESTIMATOR_COLORS["dm"], ls="--", lw=1.2,
               label=f"DM negative control ({dm_rate:.2f})")
    ax.set_ylabel(f"P(rel err > {LARGE_ERR})")
    ax.set_title("B. Pre-registered gate verdicts vs large error\n(proposal, not a standard)")
    ax.legend(fontsize=7.5)

    ax = axes[2]  # C: 축 09 blind spot — 진단 계열(estimator 아님) → 팔레트 밖 중립색
    ax.plot(bg["knob_value"], bg["trust_share"], marker="o", color="#44433e",
            label="share gated 'trust'")
    ax.plot(bg["knob_value"], bg["mean_rel_err"], marker="s", color="#e34948",
            label="mean |rel err| (ips·dr·snips)")
    ax.set_xlabel("confounding strength γ (axis 09 — excluded from pooling)")
    ax.set_ylim(-0.03, 1.05)
    ax.set_title("C. The blind spot — gate trusts while error grows\n(recorded-propensity diagnostics)")
    ax.legend(fontsize=7.5)

    fig.suptitle("Axis 08 — What ESS·max-weight diagnostics can and cannot forecast "
                 "(thresholds pre-registered at M1; evaluation only, no tuning)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    trust_row = conf[conf["gate"] == "trust"]
    fb_row = conf[conf["gate"] == "ab_fallback"]
    print(f"[08] → {csv_path.name}, {conf_path.name}, {fig_path.name}")
    print(f"[08] PATTERN P(large|trust)={float(trust_row['share_large_err'].iloc[0]):.4f} "
          f"vs P(large|ab_fallback)={float(fb_row['share_large_err'].iloc[0]) if len(fb_row) else float('nan'):.4f} "
          f"| blind spot: trust_share@γ=2.5={bg['trust_share'].iloc[-1]:.2f} "
          f"mean_rel_err@γ=2.5={bg['mean_rel_err'].iloc[-1]:.4f} "
          f"| support arm fired={int((w['support_proxy'] > PROVISIONAL_THRESHOLDS.support_deficiency_max).sum())}회")


if __name__ == "__main__":
    main()
