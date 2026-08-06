"""축 05 — propensity 오지정: 기록 p̂ 를 true→estimated→noised 로 교체 (estimator 입력측 노브).

노브(모드형): pscore_mode ∈ {true, estimated, noised_s}. S=40, BASE_M2 고정(γ=0 ⇒ pscore_logged
= pscore_true — "true" 모드가 곧 clean 기준). DGP 는 모드와 무관하므로 로그는 seed 당 1회 생성.
- estimated: sklearn LogisticRegression(x→a, C=1.0, max_iter=300) 을 로그마다 적합, 적합확률의
  관측행동 성분을 p̂ 로 사용, clip(1e-6, 1). lbfgs 는 multiclass 에서 기본이 multinomial
  (multi_class 인자는 sklearn 1.5 deprecated·1.7 제거 — 1.9.0 환경이므로 인자 생략).
- noised_s (s ∈ {0.1, 0.25, 0.5, 1.0}): p̂ = ps_logged·exp(s·N(0,1)), clip(1e-6, 1).
  noise z 는 seed 파생 고정 rng 로 뽑아 **스케일 간 공유**(nested 오염 — s 만의 효과가 분리됨).
- variant 축: q̂ ∈ {q_oracle(=d.q_true), q_shrink09(0.9q+0.05)} — 두 variant 모두 기록.
- **hyperparam(τ·λ)은 clean(true) pscore 의 raw weight 로 seed 당 1회 계산해 hypers 로 전달**
  (_common.hyperparams_from_weights) — 오염 weight 로 재계산 금지(축 08 pooling 보호).

기대 패턴: 오염↑ → IPS |bias|↑ (곱셈 log-normal 노이즈는 E[exp(-s·z)]=e^{s²/2}>1 로 가치 인플레이션);
DR@q_oracle 은 pscore 오염에도 생존(잔차 평균 0 — 한쪽 모델만 맞아도 됨); DR@q_shrink09 는 둘 다
틀리면 악화. estimated 모드는 흥미 지점 — 적합이 좋으면 true 와 유사할 수 있음. 불발 시 그대로 보고.

산출: results/tables/05_propensity_misspec.csv (long, _common.COLUMNS)
     + results/figures/05_propensity_misspec.png (variant 2패널 × 모드 라인).
"""

import numpy as np
from sklearn.linear_model import LogisticRegression

from _common import (BASE_M2, cached_v_true, evaluate_run, hyperparams_from_weights,
                     summarize, write_axis_csv)
from _style import ESTIMATOR_ORDER, apply_style, end_label, save_figure, series_kwargs
from ope.dgp import SyntheticBanditData, make_synthetic_bandit_data
from ope.estimators import estimate_ips

AXIS_ID, SLUG = "05", "propensity_misspec"
KNOB = "pscore_mode"
MODES = ["true", "estimated", "noised_0.1", "noised_0.25", "noised_0.5", "noised_1.0"]
NOISE_SCALES = {"noised_0.1": 0.1, "noised_0.25": 0.25, "noised_0.5": 0.5, "noised_1.0": 1.0}
SEEDS = range(1000, 1040)  # S=40
VARIANTS = ("q_oracle", "q_shrink09")
EMPHASIZED = ("ips", "dr")
PS_CLIP = (1e-6, 1.0)
NOISE_RNG_OFFSET = 500_000  # noise rng = default_rng(OFFSET + seed) — seed 파생 고정


def _estimated_pscore(d: SyntheticBanditData) -> np.ndarray:
    """(x → a) multinomial logistic 적합확률의 관측행동 성분. 로그에 없는 행동 클래스가 있어도
    classes_ 매핑이 어긋나지 않도록 searchsorted 로 열 인덱스를 찾는다."""
    clf = LogisticRegression(C=1.0, max_iter=300)
    clf.fit(d.context, d.action)
    proba = clf.predict_proba(d.context)
    col = np.searchsorted(clf.classes_, d.action)
    ps = proba[np.arange(len(d.action)), col]
    return np.clip(ps, *PS_CLIP)


def _spread_log(ys: list[float], min_gap: float = 0.15) -> list[float]:
    """끝단 직접 라벨 y 좌표를 log10 공간 최소 간격으로 벌린다(라벨 충돌 방지, 입력 순서 유지 반환)."""
    order = np.argsort(ys)
    logs = np.log10(np.asarray(ys, dtype=float)[order])
    for i in range(1, len(logs)):
        logs[i] = max(logs[i], logs[i - 1] + min_gap)
    out = np.empty_like(logs)
    out[order] = 10.0 ** logs
    return out.tolist()


def main() -> None:
    v_true = cached_v_true(BASE_M2)
    records = []
    for seed in SEEDS:
        d = make_synthetic_bandit_data(BASE_M2._replace(seed=seed))
        # hyperparam 은 clean pscore 의 raw weight 로 고정 (모든 모드·variant 에 동일 전달)
        w_clean = estimate_ips(d.reward, d.action, d.pscore_logged, d.pi_e_dist).weights
        hypers = hyperparams_from_weights(w_clean)
        q_hats = {"q_oracle": d.q_true, "q_shrink09": 0.9 * d.q_true + 0.05}
        z = np.random.default_rng(NOISE_RNG_OFFSET + seed).normal(size=len(d.action))
        ps_by_mode = {"true": d.pscore_logged, "estimated": _estimated_pscore(d)}
        for mode, s in NOISE_SCALES.items():
            ps_by_mode[mode] = np.clip(d.pscore_logged * np.exp(s * z), *PS_CLIP)
        for mode in MODES:
            for variant in VARIANTS:
                records += evaluate_run(AXIS_ID, KNOB, mode, seed, d, q_hats[variant],
                                        v_true, pscore=ps_by_mode[mode], variant=variant,
                                        hypers=hypers)
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), sharey=True)
    xs = np.arange(len(MODES))
    y_floor = 1e-5  # log 축 하한 — band 하한이 0 을 지나면 여기서 잘라 "0 과 구분 불가"로 읽힘
    panel_titles = {"q_oracle": r"$\hat q$ = oracle ($q_{true}$)",
                    "q_shrink09": r"$\hat q$ = shrink ($0.9q+0.05$, mild misspec.)"}
    for ax, variant in zip(axes, VARIANTS):
        sv = summary[summary["variant"] == variant]
        ends = []
        for est in ESTIMATOR_ORDER:
            s = sv[sv["estimator"] == est].set_index("knob_value").reindex(MODES)
            y = s["bias"].abs().to_numpy()
            se = s["est_se"].to_numpy()
            emph = est in EMPHASIZED
            ax.plot(xs, y, marker="o", **series_kwargs(est, emph))
            if emph:
                ax.fill_between(xs, np.clip(y - 2 * se, y_floor, None), y + 2 * se,
                                color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
            ends.append((float(y[-1]), est, emph))
        for y_adj, (_, est, emph) in zip(_spread_log([e[0] for e in ends]), ends):
            end_label(ax, xs[-1], y_adj, est, emph)
        ax.set_yscale("log")
        ax.set_ylim(y_floor, 1.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(["true", "estimated", "noise 0.1", "noise 0.25",
                            "noise 0.5", "noise 1.0"], rotation=18, ha="right")
        ax.set_xlim(-0.4, len(MODES) - 1 + 1.25)  # 우측 여백 = 끝단 직접 라벨 자리
        ax.set_title(panel_titles[variant], fontsize=10)
        ax.set_xlabel("Logged propensity mode (clean → contaminated)")
    axes[0].set_ylabel(r"Absolute ensemble bias $|\,\mathrm{mean}(\hat V) - V\,|$ (log scale)")
    axes[0].legend(ncol=2, loc="upper left")
    fig.suptitle("Axis 05 — Propensity misspecification: IPS inherits pscore contamination; "
                 "DR survives iff one model is right\n"
                 rf"shaded: ensemble $\pm$2SE over {len(SEEDS)} seeds · "
                 r"$\tau,\lambda$ from clean weights (fixed across modes)", fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    ab = summary.assign(abs_bias=summary["bias"].abs()).pivot_table(
        index=["variant", "estimator"], columns="knob_value", values="abs_bias")
    noised = list(NOISE_SCALES)
    ips = ab.loc[("q_oracle", "ips")]      # IPS 는 q̂ 미사용 — variant 간 동일
    dr_o = ab.loc[("q_oracle", "dr")]
    dr_s = ab.loc[("q_shrink09", "dr")]
    ips_mono = bool(np.all(np.diff([ips[m] for m in noised]) > 0)
                    and ips["noised_1.0"] > ips["true"])
    dr_oracle_survives = bool(dr_o["noised_1.0"] < 0.1 * ips["noised_1.0"])
    dr_shrink_degrades = bool(dr_s["noised_1.0"] > dr_s["true"])
    print(f"[05] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[05] PATTERN ips_bias_monotone_in_noise={ips_mono}  "
          f"dr_oracle_survives_worst={dr_oracle_survives}  "
          f"dr_shrink09_degrades={dr_shrink_degrades}")
    print(f"[05] |bias| ips: true={ips['true']:.5f} estimated={ips['estimated']:.5f} "
          + " ".join(f"{m.split('_')[1]}={ips[m]:.5f}" for m in noised))
    print(f"[05] |bias| dr@q_oracle: true={dr_o['true']:.5f} noised_1.0={dr_o['noised_1.0']:.5f}  "
          f"dr@q_shrink09: true={dr_s['true']:.5f} estimated={dr_s['estimated']:.5f} "
          f"noised_1.0={dr_s['noised_1.0']:.5f}")


if __name__ == "__main__":
    main()
