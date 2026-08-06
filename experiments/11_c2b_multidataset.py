"""축 11 — 실데이터 c2b 멀티데이터셋: synthetic 결론(DR 강건·게이트 작동)의 실데이터 재현 확인.

4 OpenML 데이터셋(optdigits·satimage·pendigits·letter — `datasets.C2B_DATASETS`) × S=20 로깅
재표집(seed 1000–1019) × q̂ 2종(variant="q_good"=A-절반 적합 LR proba / "q_degraded"=피처 앞
절반만 본 LR — 비-affine 오지정). 정책 쌍 softmax(β·s), β: 2(로깅)→6(평가) — datasets.py 기본값.
propensity 는 **정확히 계산·기록**되고 참값도 **정확**(V(π_e)=mean_B[π_e(y|x)], gt_ci=None) —
근사 GT·bootstrap 참값 규약은 축 12(OBD) 소관.

정직 문서화 — 이 축은 판별 축이 아니다: c2b reward 는 결정적 1[a=y] (노이즈 채널 없음)이라
DR 잔차 = 순수 모델오차이고, A-절반 적합 LR 은 정확도가 높아 good-q̂ 계열(DM·DR·Switch-DR·
DRos)의 압승이 **예상값**이다. 본 축의 목적은 "깨끗한 GT 를 가진 실데이터에서 synthetic
결론 — DR 의 q̂-오지정 생존(축 06)·decision gate 작동(축 08) — 이 재현되는가" 확인이며,
예상 밖 결과(예: degraded q̂ 에서 DM 실패·DR 생존 실패, 게이트 오판)가 진짜 볼거리다.
불발 시 그대로 보고(은폐 금지). q̂-무관 3종(IPS·SNIPS·Clipped-IPS)은 variant 간 산술 동일
(같은 로그·같은 pscore) — CSV 에는 두 variant 모두 기록하되 figure 는 q_good 만 그린다.

실측 결과(2026-08-06 — 수치 정본은 committed CSV·stdout 패턴 체크): ① DR 의 q̂-오지정 생존은
4/4 데이터셋 재현 — degraded q̂ 에서 DM bias −0.026(satimage) ~ −0.387(letter):
optdigits·pendigits·letter 는 −0.16 ~ −0.39 파국, satimage 는 −0.026 으로 온건하나
DR 의 |bias| ≤ 0.0032 대비 여전히 열세 — 생존 판정 4/4 성립.
② 게이트는 4데이터셋 × 20 seeds = 80/80 로그 전부 trust — 실현 오차(가중 계열 |bias| 소폭)와
정합. ③ 단 "good-q̂ 계열 압승" 예상은 **부분 불발**: pendigits·letter 에서 DM(q_good)이 IPS 에
MSE 로 패배(정확 propensity 의 IPS 는 불편인데 DM 은 good q̂ 로도 잔존 bias −0.02 ~ −0.04) —
DR 은 전 데이터셋에서 IPS 이하 MSE 로 생존. ④ 대표 seed bootstrap CI 는 28행 중 19행만 gt 커버,
미커버 9행 전부 구조적 bias 보유 조합(모델-bias: dm×3·switch_dr×3·dros×2 / clipping-bias:
letter clipped_ips×1 — clipped_ips 는 q̂ 미사용이므로 모델-bias 아님) — percentile bootstrap 은
표본 분산만 잡고 bias 는 못 잡는다는 교훈의 실데이터 실증.

실데이터 불확실성 규약(CLAUDE.md §2): 단일 로그는 seed 반복이 불가하므로 bootstrap_ci 전용 —
대표 seed 1000 로그에 estimator 별 percentile bootstrap(n_boot=1000, alpha=0.05, q̂=good;
hyperparam 은 evaluate_run 과 동일 정책 = raw w 분위수 — 해당 로그의 RunRecord 값 verbatim 재사용,
재표집 seed 는 estimator 간 공통 고정) → companion CSV. S=20 로깅 재표집 ensemble(±2SE)과
단일 로그 bootstrap CI 를 병기해 두 불확실성 축을 한 figure 에서 대조한다.

산출: results/tables/11_c2b_multidataset.csv (long, _common.COLUMNS)
    + results/tables/11_c2b_multidataset_ci.csv (**스키마 예외 — companion**: 단일 로그 bootstrap
      이라 COLUMNS 의 seed-ensemble 진단 열 대신 CI 열. 컬럼 = axis_id, knob_name, knob_value,
      seed, estimator, variant, hyperparam_value, estimate, ci_lo, ci_hi, gt_value, n_boot, alpha.
      본 docstring 이 스키마 정본.)
    + results/figures/11_c2b_multidataset.png (4패널 — estimator 별 (estimate−gt) 오차:
      S=20 ensemble 점+±2SE, 대표 seed bootstrap CI whisker, 게이트 verdict 주석 —
      PROVISIONAL_THRESHOLDS·raw 진단, gt=0 수평선).

주의(비용): 최초 실행은 satimage·pendigits·letter 의 OpenML fetch 에 네트워크가 필요하다
(data/openml 캐시, 커밋 금지 — data/README.md). optdigits 는 캐시 완비.
"""

import csv

import numpy as np
import pandas as pd

from _common import TAB_DIR, evaluate_run, summarize, write_axis_csv
from _style import (ESTIMATOR_COLORS, ESTIMATOR_LABELS, ESTIMATOR_ORDER, apply_style,
                    save_figure)
from ope.datasets import C2B_DATASETS, classification_to_bandit
from ope.diagnostics import PROVISIONAL_THRESHOLDS, compute_diagnostics, decision_gate
from ope.estimators import bootstrap_ci

AXIS_ID, SLUG = "11", "c2b_multidataset"
KNOB = "dataset"
DATASETS = list(C2B_DATASETS)  # optdigits · satimage · pendigits · letter (순서 고정)
SEEDS = range(1000, 1020)      # S=20 로깅 재표집 (구조는 C2B_SPLIT_SEED 로 고정)
REP_SEED = 1000                # 대표 seed — 단일 로그 bootstrap CI 전용
N_BOOT, ALPHA, BOOT_SEED = 1000, 0.05, 2026  # 재표집 seed 는 estimator 간 공통(비교 가능성)
Q_DEPENDENT = ("dm", "dr", "switch_dr", "dros")  # q̂ 를 실제로 쓰는 4종
VARIANTS = ("q_good", "q_degraded")
CI_COLUMNS = ["axis_id", "knob_name", "knob_value", "seed", "estimator", "variant",
              "hyperparam_value", "estimate", "ci_lo", "ci_hi", "gt_value", "n_boot", "alpha"]
GATE_COLOR = "#3f3e39"  # 진단 주석(estimator 아님 — entity 팔레트 밖 중립색)


def main() -> None:
    records, ci_rows = [], []
    meta: dict[str, dict] = {}          # 데이터셋별 n·K·gt·대표 seed raw 진단
    verdicts: dict[str, list] = {}      # 데이터셋별 per-seed gate verdict (raw 진단 기준)
    for name in DATASETS:
        verdicts[name] = []
        for seed in SEEDS:
            d = classification_to_bandit(name, seed)
            rep_recs = None
            for variant, q_hat in (("q_good", d.q_scores), ("q_degraded", d.q_scores_degraded)):
                recs = evaluate_run(AXIS_ID, KNOB, name, seed, d, q_hat, d.gt_value,
                                    pscore=d.pscore, variant=variant)
                records += recs
                if seed == REP_SEED and variant == "q_good":
                    rep_recs = {r.estimator: r for r in recs}
            diag = compute_diagnostics(d.pscore, d.action, d.pi_e_dist)
            verdicts[name].append(decision_gate(diag, PROVISIONAL_THRESHOLDS).decision)
            if seed == REP_SEED:
                meta[name] = {"n": len(d.action), "k": d.n_actions, "gt": d.gt_value,
                              "ess": diag.ess_ratio, "mw": diag.max_weight,
                              "sp": diag.support_deficiency}
                # 대표 seed 단일 로그 bootstrap CI — hyperparam 은 evaluate_run 이 실제 쓴 값 verbatim
                for est in ESTIMATOR_ORDER:
                    h = rep_recs[est].hyperparam_value
                    lo, hi = bootstrap_ci(d.reward, d.action, d.pscore, d.pi_e_dist,
                                          d.q_scores, est, n_boot=N_BOOT, alpha=ALPHA,
                                          seed=BOOT_SEED,
                                          hyperparam=None if np.isnan(h) else float(h))
                    ci_rows.append([AXIS_ID, KNOB, name, seed, est, "q_good", h,
                                    rep_recs[est].estimate, lo, hi, d.gt_value, N_BOOT, ALPHA])
        print(f"[11] {name}: n={meta[name]['n']:,} K={meta[name]['k']} "
              f"gt={meta[name]['gt']:.6f} done")

    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    ci_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_ci.csv"
    with ci_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(CI_COLUMNS)
        w.writerows(ci_rows)
    summary = summarize(csv_path)
    ci_df = pd.read_csv(ci_path)
    df = pd.read_csv(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6),
                             gridspec_kw={"hspace": 0.48, "wspace": 0.22})
    rng_j = np.random.default_rng(11)  # 점 jitter 전용(수치 무관)
    for ax, name in zip(axes.flat, DATASETS):
        m = meta[name]
        sub = df[df["knob_value"] == name]
        ssum = summary[summary["knob_value"] == name]
        ax.axhline(0.0, color="#8a897f", lw=0.9, zorder=1)
        for j, est in enumerate(ESTIMATOR_ORDER):
            color = ESTIMATOR_COLORS[est]
            for variant, dx, filled in (("q_good", -0.28, True), ("q_degraded", 0.28, False)):
                if variant == "q_degraded" and est not in Q_DEPENDENT:
                    continue  # q̂-무관 3종은 variant 간 산술 동일 — figure 는 q_good 만
                pts = sub[(sub["estimator"] == est) & (sub["variant"] == variant)]
                errs = (pts["estimate"] - pts["v_true"]).to_numpy()
                xj = j + dx + rng_j.uniform(-0.07, 0.07, len(errs))
                if filled:
                    ax.scatter(xj, errs, s=9, color=color, alpha=0.30, lw=0, zorder=2)
                else:
                    ax.scatter(xj, errs, s=12, facecolors="none", edgecolors=color,
                               alpha=0.55, lw=0.7, marker="D", zorder=2)
                row = ssum[(ssum["estimator"] == est) & (ssum["variant"] == variant)].iloc[0]
                mean_err = row["mean_estimate"] - row["v_true"]
                ax.errorbar(j + dx, mean_err, yerr=2.0 * row["est_se"],
                            fmt="o" if filled else "D", color=color, ms=4.5, capsize=3,
                            lw=1.6, mfc=color if filled else "white", zorder=4)
            # 대표 seed bootstrap CI whisker (중앙, q_good) — gt 기준 오차 좌표로 변환
            c = ci_df[(ci_df["knob_value"] == name) & (ci_df["estimator"] == est)].iloc[0]
            e0 = c["estimate"] - m["gt"]
            ax.errorbar(j, e0,
                        yerr=[[max(e0 - (c["ci_lo"] - m["gt"]), 0.0)],
                              [max((c["ci_hi"] - m["gt"]) - e0, 0.0)]],
                        fmt="s", color=color, ms=4.0, mfc="white", capsize=2.5,
                        lw=1.1, alpha=0.9, zorder=3)
        lo, hi = ax.get_ylim()  # 주석·클러스터 충돌 방지 headroom (DM-degraded 최하단 잘림 방지 포함)
        span = hi - lo
        ax.set_ylim(lo - 0.04 * span, hi + 0.16 * span)
        n_trust = verdicts[name].count("trust")
        ax.text(0.98, 0.965,
                f"gate: {verdicts[name][0]} — {n_trust}/{len(SEEDS)} seeds trust "
                f"(provisional thresholds)\n"
                f"raw diag @seed {REP_SEED}: ESS/n={m['ess']:.3f} · "
                f"max w={m['mw']:.1f} · support proxy={m['sp']:.3f}",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5, color=GATE_COLOR)
        ax.set_xticks(range(len(ESTIMATOR_ORDER)))
        ax.set_xticklabels([ESTIMATOR_LABELS[e] for e in ESTIMATOR_ORDER],
                           rotation=25, ha="right", fontsize=8)
        ax.set_title(f"{name} — n={m['n']:,} · K={m['k']} · exact gt "
                     f"$V(\\pi_e)$={m['gt']:.4f}", loc="left")
    for ax in axes[:, 0]:
        ax.set_ylabel("estimate − ground truth")
    handles = [
        Line2D([0], [0], marker="o", color=GATE_COLOR, lw=0, ms=5,
               label="good q̂ (A-half LR): S=20 seeds + mean ±2SE"),
        Line2D([0], [0], marker="D", color=GATE_COLOR, lw=0, ms=5, mfc="white",
               label="degraded q̂ (half features): q̂-dependent estimators only"),
        Line2D([0], [0], marker="s", color=GATE_COLOR, lw=1.1, ms=5, mfc="white",
               label=f"rep-seed bootstrap 95% CI (n_boot={N_BOOT}, good q̂)"),
    ]
    fig.legend(handles=handles, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Axis 11 — Real-data c2b, 4 datasets: do the synthetic conclusions "
                 "(DR robustness, working gate) reproduce under exact propensity and exact GT?\n"
                 "softmax policy pair β: 2→6 · deterministic reward 1[a=y] (DR residual = pure "
                 "model error — good-q̂ family expected to win; this is a replication axis, "
                 "not a discriminating one)", y=1.055)
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    mean_err = summary.pivot_table(index=["knob_value", "variant"], columns="estimator",
                                   values="mean_estimate").sub(
        summary.pivot_table(index=["knob_value", "variant"], columns="estimator",
                            values="v_true")["dm"], axis=0)
    mse = summary.pivot_table(index=["knob_value", "variant"], columns="estimator",
                              values="mse")
    print(f"[11] → {csv_path.name}, {ci_path.name}, {fig_path.name}")
    # 1) variant 무관 sanity: q̂-무관 3종은 두 variant 산술 동일이어야 한다
    qfree = [e for e in ESTIMATOR_ORDER if e not in Q_DEPENDENT]
    piv_est = df.pivot_table(index=["knob_value", "seed", "estimator"], columns="variant",
                             values="estimate")
    qfree_dev = float(piv_est.loc[(slice(None), slice(None), qfree), :]
                      .diff(axis=1).iloc[:, -1].abs().max())
    print(f"[11] SANITY qfree_variant_identical(max|Δ|={qfree_dev:.2e})={qfree_dev == 0.0}")
    # 2) DR 의 q̂-오지정 생존 재현(축 06 결론): degraded 에서 |bias| DR < DM?
    for name in DATASETS:
        dm_d = mean_err.loc[(name, "q_degraded"), "dm"]
        dr_d = mean_err.loc[(name, "q_degraded"), "dr"]
        dm_g = mean_err.loc[(name, "q_good"), "dm"]
        print(f"[11] PATTERN [{name}] dr_survives_degraded(|{dr_d:+.4f}| < |{dm_d:+.4f}|)="
              f"{abs(dr_d) < abs(dm_d)}  dm_good_bias={dm_g:+.4f}")
    # 3) good-q̂ 계열 압승(예상값 — 판별 아님): MSE(dm|q_good) < MSE(ips)?
    for name in DATASETS:
        dm_mse = mse.loc[(name, "q_good"), "dm"]
        dr_mse = mse.loc[(name, "q_good"), "dr"]
        ips_mse = mse.loc[(name, "q_good"), "ips"]
        print(f"[11] PATTERN [{name}] goodq_dm_beats_ips({dm_mse:.2e} < {ips_mse:.2e})="
              f"{dm_mse < ips_mse}  dr_leq_ips={dr_mse <= ips_mse * 1.05}")
    # 4) gate + bootstrap CI 커버리지 (대표 seed·q_good)
    all_trust = all(v == "trust" for name in DATASETS for v in verdicts[name])
    cover = ((ci_df["ci_lo"] <= ci_df["gt_value"]) &
             (ci_df["gt_value"] <= ci_df["ci_hi"]))
    print(f"[11] PATTERN gate_all_trust={all_trust}  "
          f"boot_ci_covers_gt={int(cover.sum())}/{len(ci_df)}")


if __name__ == "__main__":
    main()
