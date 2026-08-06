"""축 06 — reward model 오지정: DM bias 의 표본 불감성 vs DR 생존 vs IPS negative control.

노브: q̂ = α·q_true + (1−α)/2 (α=1 oracle → α=0 uninformative 상수 0.5), knob_name="alpha",
α ∈ {1, 0.9, 0.75, 0.5, 0.25, 0} × variant ∈ {n2500, n10000, n40000} (BASE_M2.n 교체). S=40.
기대 패턴 ([MRDR 계보](https://arxiv.org/pdf/1802.03493)): DM bias ≈ (1−α)·|0.5−v_true| 로 선형 성장하고
**n 에 불감**(3패널에서 DM 곡선 동일 — 표본을 늘려도 model bias 는 못 고친다는 것이 핵심 교훈),
DR 은 pscore 가 정확하므로 전 α 에서 생존, IPS 는 q̂ 를 아예 안 쓰므로 평평(negative control).
α=1 에서는 DM 이 최강(oracle q̂ 의 저분산)임도 정직하게 전시. 불발 시 그대로 보고(은폐 금지).

산출: results/tables/06_reward_misspec.csv (long, _common.COLUMNS) + results/figures/06_reward_misspec.png.
"""

import numpy as np

from _common import BASE_M2, SEEDS_DEFAULT, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import ESTIMATOR_ORDER, ORACLE_COLOR, apply_style, end_label, save_figure, series_kwargs
from ope.dgp import make_synthetic_bandit_data

AXIS_ID, SLUG = "06", "reward_misspec"
KNOB = "alpha"
GRID = [1.0, 0.9, 0.75, 0.5, 0.25, 0.0]
VARIANTS = [(2_500, "n2500"), (10_000, "n10000"), (40_000, "n40000")]
SEEDS = SEEDS_DEFAULT  # S=40
EMPHASIZED = ("dm", "dr", "ips")


def _spread_end_labels(ax, entries, min_frac: float = 0.04) -> None:
    """α=0 끝단 직접 라벨 — near-zero 군집(ips·snips·dr 등)의 겹침을 최소 세로 간격으로 해소."""
    y0, y1 = ax.get_ylim()
    gap = min_frac * (y1 - y0)
    placed = []
    for x, y, est, emph in sorted(entries, key=lambda e: e[1]):
        y_adj = y if not placed else max(y, placed[-1] + gap)
        placed.append(y_adj)
        end_label(ax, x, y_adj, est, emph)


def main() -> None:
    v_true = cached_v_true(BASE_M2)
    records = []
    for n, var in VARIANTS:
        cfg = BASE_M2._replace(n=n)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            for alpha in GRID:
                q_hat = alpha * d.q_true + (1.0 - alpha) / 2.0
                records += evaluate_run(AXIS_ID, KNOB, alpha, seed, d, q_hat, v_true,
                                        variant=var)
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.4), sharey=True,
                             gridspec_kw={"wspace": 0.30})
    alphas = np.array(GRID)
    theory = (1.0 - alphas) * abs(0.5 - v_true)
    label_entries = []  # (ax, entries) — sharey autoscale 확정 후 일괄 라벨
    for k, (ax, (n, var)) in enumerate(zip(axes, VARIANTS)):
        sub = summary[summary["variant"] == var]
        ax.plot(alphas, theory, color=ORACLE_COLOR, ls=":", lw=1.4, zorder=1,
                label=r"DM theory $(1-\alpha)\,|0.5-v_{\mathrm{true}}|$" if k == 0 else None)
        entries = []
        for est in ESTIMATOR_ORDER:
            s = sub[sub["estimator"] == est].sort_values("knob_value", ascending=False)
            y = s["bias"].abs().to_numpy()
            emph = est in EMPHASIZED
            ax.plot(s["knob_value"], y, marker="o", **series_kwargs(est, emph))
            if emph:
                se = s["est_se"].to_numpy()
                ax.fill_between(s["knob_value"], np.maximum(y - 2 * se, 0.0), y + 2 * se,
                                color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
            entries.append((s["knob_value"].iloc[-1], y[-1], est, emph))
        label_entries.append((ax, entries))
        ax.set_xlim(1.06, -0.06)  # α 감소 방향(오른쪽=오지정 심화) — α=1 이 oracle
        ax.set_xticks(GRID)
        ax.set_xticklabels(["1", ".9", ".75", ".5", ".25", "0"])
        ax.set_title(f"n = {n:,}")
        ax.set_xlabel(r"$\alpha$  (1 = oracle $\hat q$ $\rightarrow$ 0 = uninformative)")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_ylabel(r"|bias| vs true value  $|\mathrm{mean}(\hat V) - v_{\mathrm{true}}|$")
    for ax, entries in label_entries:
        _spread_end_labels(ax, entries)
    # 범례는 figure-level 상단 스트립 — 패널 내 배치는 DM 곡선·끝단 라벨과 충돌(육안 검수 반영)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("Axis 06 — Reward-model misspecification: DM inherits q̂ bias at every n; "
                 "correct-propensity weighting survives\n"
                 r"$\hat q = \alpha q + (1-\alpha)/2$ · DM curve is identical across panels "
                 f"(sample size cannot fix model bias) · shaded: ensemble ±2SE over "
                 f"{len(SEEDS)} seeds", y=1.17, fontsize=11)
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    bias = summary.pivot_table(index=["variant", "knob_value"], columns="estimator",
                               values="bias")
    mse = summary.pivot_table(index=["variant", "knob_value"], columns="estimator",
                              values="mse")
    dm_abs = bias["dm"].abs()
    th = {a: (1.0 - a) * abs(0.5 - v_true) for a in GRID}
    dm_dev = float(max(abs(dm_abs.loc[(var, a)] - th[a]) for _, var in VARIANTS for a in GRID))
    xs = np.array([1.0 - a for _, var in VARIANTS for a in GRID])
    ys = np.array([dm_abs.loc[(var, a)] for _, var in VARIANTS for a in GRID])
    slope_ratio = float((xs @ ys) / (xs @ xs) / abs(0.5 - v_true))
    dm_spread = float(dm_abs.groupby(level="knob_value").apply(lambda s: s.max() - s.min()).max())
    dr_max = float(bias["dr"].abs().max())
    ips_range = float(bias["ips"].groupby(level="variant").apply(lambda s: s.max() - s.min()).max())
    mse_a1 = mse.xs(1.0, level="knob_value")
    dm_oracle = bool((mse_a1["dm"] <= mse_a1.min(axis=1) + 1e-15).all())
    runner = float((mse_a1.drop(columns="dm").min(axis=1) / mse_a1["dm"]).min())

    print(f"[06] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[06] PATTERN dm_linear_theory(max_dev={dm_dev:.4f}, slope_ratio={slope_ratio:.3f})="
          f"{dm_dev < 0.01}  dm_n_insensitive(max_spread_over_n={dm_spread:.4f})="
          f"{dm_spread < 0.01}")
    print(f"[06] PATTERN dr_survives_all_alpha(max|bias_dr|={dr_max:.4f} vs "
          f"dm_bias_alpha0={th[0.0]:.4f})={dr_max < 0.1 * th[0.0]}  "
          f"ips_flat(range={ips_range:.2e})={ips_range < 1e-12}  "
          f"dm_oracle_at_alpha1(runnerup_mse_ratio={runner:.2f})={dm_oracle}")


if __name__ == "__main__":
    main()
