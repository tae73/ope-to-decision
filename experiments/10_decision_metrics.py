"""축 10 — 의사결정 metric: MSE 가 아니라 "잘못 배포할 확률"과 순위 상관.

설계(자체 factorial — 축 03 CSV 재사용 아님): regime β_log ∈ {0.5, 1, 4} × 후보 β_eval ∈
{0.25, 0.5, 0.8, 1.25, 1.5, 2.0, 3.0, 5.0} (근접 8후보 — 최초 광간격 {0.25,0.5,1,2,3,5,8} 에서
난이도 보정으로 교체, 아래 이력 참조). 같은 seed = 같은 로그(β_eval 은 rng 미소비 — tests/test_dgp.py 회귀 고정)
위에서 모든 후보를 평가한다. incumbent = 로깅 정책 π_0: 결정측 기준은 **추정치** mean(r)(로그 on-policy
평균), 판정측 참값은 V(π_0)=true_policy_value(β_eval=β_log) (γ=δ=0 에서 유효한 트릭).

wrong-deploy(false-go): V̂(cand) > mean(r)+ε 인데 true V(cand) ≤ V(π_0). false-stop 은 반대.
margin ε grid {0, 0.005, 0.01, 0.02} + per-seed Spearman(V̂ 순위 vs 참 순위) + top-1 regret.

발견(정직 기록 — 설계 이력 포함): 최초 n=10k 광간격에서 전 estimator 만점 → n=2000·근접 grid 로
난이도 보정(사전 보정, 결과 조정 아님). 그 결과 세 메커니즘이 분리됐다:
① **같은 estimator·같은 로그 비교는 공통 오차가 상쇄**(Δ-OPE 가 정식화한 효과 — Jeunen+
   RecSys'24, arXiv:2405.10024): IPS·SNIPS 계열 comparative false-go = 0, Spearman 1.0
   (Spearman 만점엔 q̂=αq+(1−α)/2 가 rank-보존 오지정이라는 구조적 이유도 있음 — 한계 명시).
② **혼합 비교는 bias 를 부활시킨다**: DM(후보) vs mean(r)(incumbent) 처럼 다른 추정기끼리
   비교하면 상쇄가 깨져 DM 의 수축 bias 가 그대로 결정 오류가 된다(false-go·false-stop 모두).
③ **절대 임계 게이트**(V̂ > T)는 상쇄가 아예 없어 bias·variance 를 전부 상속한다.
플레이북 함의: 게이트는 가능하면 같은 로그 위·같은 estimator 의 **비교형**으로 설계하고, 경계
근방(참값 차 ≈ 0)의 판정은 margin 이 아니라 불확실성 구간으로 다뤄라. 전제: γ=0 — 이 전제가
깨지면 상쇄 논리도 깨진다(축 09 소관).

산출: results/tables/10_decision_metrics.csv + results/figures/10_decision_metrics.png.
비고: incumbent 추정행은 estimator="_incumbent_mean_r" 로 기록(코어 7종 아님 — 스키마 예외, 문서화).
"""

import numpy as np
import pandas as pd

from _common import BASE_M2, cached_v_true, evaluate_run, write_axis_csv
from _style import (
    ESTIMATOR_COLORS, ESTIMATOR_LABELS, ESTIMATOR_ORDER, apply_style, save_figure,
)
from ope.dgp import make_synthetic_bandit_data
from ope.policies import softmax_policy

AXIS_ID, SLUG = "10", "decision_metrics"
REGIMES = [0.5, 1.0, 4.0]
CANDIDATES = [0.25, 0.5, 0.8, 1.25, 1.5, 2.0, 3.0, 5.0]  # 근접 grid — 난이도 보정(docstring)
N_LOG = 2000
SEEDS = range(1000, 1040)
MARGINS = [0.0, 0.005, 0.01, 0.02]
ALPHA_QHAT = 0.9
FOCUS = ("ips", "snips", "dm", "dr")  # figure 에 표시할 4종 (7종 전부는 과밀 — CSV 엔 전부)


def main() -> None:
    records = []
    for blog in REGIMES:
        for seed in SEEDS:
            cfg = BASE_M2._replace(n=N_LOG, beta_log=blog, seed=seed)
            d = make_synthetic_bandit_data(cfg)
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            # incumbent 추정(결정측 기준): 로그 on-policy 평균
            rec = evaluate_run(AXIS_ID, "candidate_beta_eval", "incumbent", seed, d,
                               q_hat, cached_v_true(cfg._replace(beta_eval=blog)))
            records.append(rec[0]._replace(estimator="_incumbent_mean_r",
                                           variant=f"blog{blog}",
                                           estimate=float(d.reward.mean()),
                                           hyperparam_value=np.nan,
                                           ess_ratio_used=np.nan))
            for cand in CANDIDATES:
                pi_e = softmax_policy(d.q_true, cand)
                d_cand = d._replace(pi_e_dist=pi_e)
                v_cand = cached_v_true(cfg._replace(beta_eval=cand))
                records += [r._replace(variant=f"blog{blog}")
                            for r in evaluate_run(AXIS_ID, "candidate_beta_eval", cand,
                                                  seed, d_cand, q_hat, v_cand)]
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)

    # ── metric 계산 ──────────────────────────────────────────────────────────
    df = pd.read_csv(csv_path)
    inc = (df[df["estimator"] == "_incumbent_mean_r"]
           .set_index(["variant", "seed"])["estimate"])
    est_df = df[df["estimator"] != "_incumbent_mean_r"].copy()
    est_df["knob_value"] = est_df["knob_value"].astype(float)

    rows = []
    for blog in REGIMES:
        variant = f"blog{blog}"
        v_inc = cached_v_true(BASE_M2._replace(beta_eval=blog))
        sub = est_df[est_df["variant"] == variant]
        for est in ESTIMATOR_ORDER:
            g = sub[sub["estimator"] == est]
            piv_est = g.pivot_table(index="seed", columns="knob_value", values="estimate")
            piv_true = g.pivot_table(index="seed", columns="knob_value", values="v_true")
            truly_worse = piv_true.iloc[0] <= v_inc  # 후보별 참값은 seed 무관
            spearman = piv_est.rank(axis=1).corrwith(
                piv_true.rank(axis=1).iloc[0], axis=1, method="pearson").mean()
            best_true = piv_true.iloc[0].max()
            picked = piv_est.idxmax(axis=1)
            regret = np.array([best_true - piv_true.iloc[0][p] for p in picked])
            for eps in MARGINS:
                deploy = piv_est.gt(inc[variant] + eps, axis=0)
                fg = (deploy.loc[:, truly_worse].to_numpy().mean()
                      if truly_worse.any() else np.nan)
                fs = ((~deploy).loc[:, ~truly_worse].to_numpy().mean()
                      if (~truly_worse).any() else np.nan)
                rows.append({"regime_beta_log": blog, "estimator": est, "margin": eps,
                             "false_go": fg, "false_stop": fs,
                             "spearman_mean": spearman,
                             "regret_mean": regret.mean(),
                             "regret_p90": float(np.quantile(regret, 0.90))})
    met = pd.DataFrame(rows)

    # 발견 ② — 절대 임계 게이트: deploy iff V̂(cand) > T (외부 목표치). 공통 오차 상쇄 없음.
    thr_rows = []
    t_grid = np.round(np.arange(0.50, 0.86, 0.02), 3)
    for blog in REGIMES:
        sub = est_df[est_df["variant"] == f"blog{blog}"]
        for est in ESTIMATOR_ORDER:
            g = sub[sub["estimator"] == est]
            for t in t_grid:
                deploy = g["estimate"] > t
                worse = g["v_true"] <= t
                thr_rows.append({"regime_beta_log": blog, "estimator": est, "threshold": t,
                                 "false_go_abs": float((deploy & worse).mean()),
                                 "false_stop_abs": float((~deploy & ~worse).mean())})
    thr = pd.DataFrame(thr_rows)
    met_path = csv_path.parent / f"{AXIS_ID}_{SLUG}_metrics.csv"
    met.to_csv(met_path, index=False)
    thr_path = csv_path.parent / f"{AXIS_ID}_{SLUG}_thresholds.csv"
    thr.to_csv(thr_path, index=False)

    # ── figure: Row1 절대 임계 게이트(오차 상속) vs Row2 비교형 게이트(상쇄 — null) ──
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.6), sharex="row")
    for j, blog in enumerate(REGIMES):
        ax = axes[0, j]
        for est in FOCUS:
            m = thr[(thr["regime_beta_log"] == blog) & (thr["estimator"] == est)]
            ax.plot(m["threshold"], m["false_go_abs"], marker="o", markersize=3,
                    color=ESTIMATOR_COLORS[est], label=ESTIMATOR_LABELS[est])
            ax.plot(m["threshold"], m["false_stop_abs"], marker="s", markersize=3,
                    linestyle="--", color=ESTIMATOR_COLORS[est], alpha=0.65)
        ax.set_title(f"logging β={blog}")
        ax.set_xlabel("absolute deploy threshold T")
        ax.set_ylim(-0.02, 1.02)
        if j == 0:
            ax.set_ylabel("ABSOLUTE gate errors\nsolid: false-go · dashed: false-stop")
            ax.legend(ncol=2, loc="upper right")
        ax = axes[1, j]
        for est in FOCUS:
            m = met[(met["regime_beta_log"] == blog) & (met["estimator"] == est)]
            ax.plot(m["margin"], m["false_go"], marker="o", markersize=3,
                    color=ESTIMATOR_COLORS[est], label=ESTIMATOR_LABELS[est])
            ax.plot(m["margin"], m["false_stop"], marker="s", markersize=3, linestyle="--",
                    color=ESTIMATOR_COLORS[est], alpha=0.65)
        ax.set_xlabel("comparative margin ε (vs on-log mean reward)")
        ax.set_ylim(-0.02, 1.02)  # DM false-stop 0.925@blog=4 까지 — 잘림 금지 (verify 지적)
        if j == 0:
            ax.set_ylabel("COMPARATIVE gate errors\nsolid: false-go · dashed: false-stop")
        m0r = met[(met["regime_beta_log"] == blog) & (met["margin"] == 0.0)]
        note = (f"IPS-fam fg={m0r[m0r.estimator.isin(['ips', 'snips'])]['false_go'].max():.2f}\n"
                f"DM fg={m0r[m0r.estimator == 'dm']['false_go'].iloc[0]:.2f} "
                f"fs={m0r[m0r.estimator == 'dm']['false_stop'].iloc[0]:.2f}")
        ax.annotate(note, xy=(0.04, 0.96), xycoords="axes fraction", ha="left", va="top",
                    fontsize=6.8, color="#6f6e66")
    fig.suptitle("Axis 10 — Decision safety is a property of the comparison design, not the estimator alone\n"
                 "Top: absolute thresholds inherit all error. Bottom: same-log comparisons cancel it — "
                 "except (i) mixed comparisons (DM vs mean-reward incumbent) resurrect bias, "
                 "(ii) DR-family cancellation is partial at the value boundary (coin-flip on V≈V(π₀))\n"
                 f"n={N_LOG} logs · {len(CANDIDATES)} candidates × {len(SEEDS)} seeds · "
                 "caveats: rank-preserving q̂ misspec, γ=0 (axis 09 breaks this), boundary candidate counted as worse",
                 fontsize=9.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    m0_all = met[met["margin"] == 0.0]
    comp_fg_max = m0_all["false_go"].max()
    sp_min = m0_all["spearman_mean"].min()
    dm_worst = thr[(thr["estimator"] == "dm")]["false_go_abs"].max()
    ips_worst = thr[(thr["estimator"] == "ips")]["false_go_abs"].max()
    weighting_fg = m0_all[m0_all["estimator"].isin(["ips", "snips", "clipped_ips"])]["false_go"].max()
    dm_fs = m0_all[m0_all["estimator"] == "dm"]["false_stop"].max()
    drfam_fg = m0_all[m0_all["estimator"].isin(["dr", "switch_dr", "dros"])]["false_go"].max()
    print(f"[10] → {csv_path.name}, {met_path.name}, {thr_path.name}, {fig_path.name}")
    print(f"[10] PATTERN ①weighting-계열 comparative fg={weighting_fg:.3f}(상쇄) "
          f"②DM 혼합비교 fg_max={m0_all[m0_all['estimator'] == 'dm']['false_go'].max():.3f}/"
          f"fs_max={dm_fs:.3f}(bias 부활) ②' DR-계열 boundary fg_max={drfam_fg:.3f}"
          f"(경계 coin-flip — 부분 상쇄) ③절대게이트 worst ips={ips_worst:.3f}(상속) "
          f"spearman_min={sp_min:.3f}(rank-보존 오지정 한계 병기)")


if __name__ == "__main__":
    main()
