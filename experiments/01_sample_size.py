"""축 01 — 표본 크기: 소표본 DM 우세 ↔ 대표본 IPS/DR 우세 regime 교차.

노브: DGPConfig.n ∈ {250, 800, 2500, 8000, 32000} (log grid). S=100 (교차점 해상 — MC 오차 완화).
q̂ = 0.9·q + 0.05 (shrink α=0.9 — Plan 검증 실측: 교차점 n*≈1.3k 가 grid 정중앙에 오는 유일 α;
α=0.5 는 DM bias² 가 커서 교차 불발). 기대 패턴: 작은 n 에서 MSE(DM) < MSE(IPS), 큰 n 에서 역전,
DR 은 전 구간 IPS 이하. 불발 시 그대로 보고(은폐 금지).

산출: results/tables/01_sample_size.csv (long, _common.COLUMNS) + results/figures/01_sample_size.png.
"""

import numpy as np

from _common import BASE_M2, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import ESTIMATOR_ORDER, apply_style, end_label, save_figure, series_kwargs
from ope.dgp import make_synthetic_bandit_data

AXIS_ID, SLUG = "01", "sample_size"
KNOB = "n"
GRID = [250, 800, 2500, 8000, 32000]
SEEDS = range(1000, 1100)  # S=100
ALPHA_QHAT = 0.9
EMPHASIZED = ("dm", "ips", "dr")


def main() -> None:
    v_true = cached_v_true(BASE_M2)
    records = []
    for n in GRID:
        cfg = BASE_M2._replace(n=n)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            records += evaluate_run(AXIS_ID, KNOB, n, seed, d, q_hat, v_true)
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    for est in ESTIMATOR_ORDER:
        s = summary[summary["estimator"] == est].sort_values("knob_value")
        emph = est in EMPHASIZED
        ax.plot(s["knob_value"], s["mse"], marker="o", **series_kwargs(est, emph))
        if emph:
            ax.fill_between(s["knob_value"], s["mse"] - 2 * s["mse_se"],
                            s["mse"] + 2 * s["mse_se"],
                            color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
        end_label(ax, s["knob_value"].iloc[-1], s["mse"].iloc[-1], est, emph)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Sample size n (log scale)")
    ax.set_ylabel("MSE vs true value (log scale)")
    ax.set_title("Axis 01 — Sample size: model-bias (DM) vs weighting-variance (IPS/DR) regimes\n"
                 r"$\hat q = 0.9q+0.05$ (mild misspecification) · shaded: ensemble $\pm$2SE over "
                 f"{len(SEEDS)} seeds")
    ax.legend(ncol=4, loc="upper right")
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    piv = summary[summary["variant"].fillna("") == ""].pivot_table(
        index="knob_value", columns="estimator", values="mse")
    small, large = piv.loc[GRID[0]], piv.loc[GRID[-1]]
    print(f"[01] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[01] PATTERN dm_beats_ips_small_n={small['dm'] < small['ips']}  "
          f"ips_beats_dm_large_n={large['ips'] < large['dm']}  "
          f"dr_leq_ips_all={(piv['dr'] <= piv['ips'] * 1.05).all()}")


if __name__ == "__main__":
    main()
