"""축 03 — 타깃-로깅 괴리: 평가정책 온도 스윕(로깅 β_log=1 고정) → 분산 폭발·순위 역전.

노브: DGPConfig.beta_eval ∈ {0.5, 1, 3, 6, 10, 15} (β_log=1 고정). S=40.
q̂ = 0.9·q + 0.05 (축 01 과 동일한 mild misspecification).
**v_true 가 β_eval 마다 다르다** — cached_v_true(cfg) 를 knob 별로 호출(캐시 키에 beta_eval 포함).
기대 패턴: 괴리 없음(β_eval≈β_log)에선 IPS/DR 이 DM 을 이기고, 괴리↑ 에서 weight 분산이
폭발해 DM/저분산 계열이 우세해지는 순위 역전. 불발 시 그대로 보고(은폐 금지).

관측(정직 기록, 초회 실행): 역전 **불발** — β_log=1 로깅이 near-uniform(logits=β_log·q, q∈(0,1))
이라 raw weight 가 유계(mean max ≈6.7 at β_eval=15)여서 분산은 ×6 증가에 그치고, n=10,000 에서
β_eval≥1 구간에서 IPS/DR 이 DM 우세(DR 최강); β_eval<β_log 인 0.5 에선 DRos 최강·DM≈IPS(점추정).
실제로 관측된 역전은 다른 곳에 있다: Clipped-IPS 가
λ=p90(w) clipping bias 폭발로 β_eval≥6 에서 최하위(심지어 DM 이하)로 추락하고, SNIPS/Switch-DR 이
plain IPS 를 괴리 증가와 함께 추월한다. figure 제목·패턴 체크는 관측 결과를 그대로 기술한다.

산출: results/tables/03_policy_gap.csv (long, _common.COLUMNS) + results/figures/03_policy_gap.png.
"""

import numpy as np

from _common import BASE_M2, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import (ESTIMATOR_ORDER, ORACLE_COLOR, apply_style, save_figure,
                    series_kwargs)
from ope.dgp import make_synthetic_bandit_data

AXIS_ID, SLUG = "03", "policy_gap"
KNOB = "beta_eval"
GRID = [0.5, 1.0, 3.0, 6.0, 10.0, 15.0]
SEEDS = range(1000, 1040)  # S=40
ALPHA_QHAT = 0.9
EMPHASIZED = ("ips", "dr", "dm")


def _dodged_end_labels(ax, items: list, min_gap_px: float = 11.0) -> None:
    """끝단 직접 라벨 — _style.end_label 규약(색·7.5pt·(4,0)pt 오프셋)에 세로 dodge 만 추가.

    본 축은 계열 끝값이 근접(DR≈SNIPS, IPS≈DRos)해 겹침이 생기므로, display 좌표에서
    min_gap_px 간격을 강제한다(_style 수정 금지 제약의 로컬 보완 — 색·라벨 규약은 동일).
    items: (estimator, x, y, emphasized) 목록.
    """
    from _style import ESTIMATOR_COLORS, ESTIMATOR_LABELS
    disp = ax.transData.transform([(x, y) for _, x, y, _ in items])
    order = np.argsort(disp[:, 1])
    adj = disp[:, 1].copy()
    for prev, cur in zip(order[:-1], order[1:]):
        adj[cur] = max(adj[cur], adj[prev] + min_gap_px)
    inv = ax.transData.inverted()
    for i, (est, x, _, emph) in enumerate(items):
        y_adj = inv.transform((disp[i, 0], adj[i]))[1]
        color = ESTIMATOR_COLORS[est] if emph else "#6f6e66"
        ax.annotate(ESTIMATOR_LABELS[est], xy=(x, y_adj), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=7.5, color=color)


def main() -> None:
    records = []
    v_true_by_knob = {}
    for beta_eval in GRID:
        cfg = BASE_M2._replace(beta_eval=beta_eval)
        v_true = cached_v_true(cfg)  # β_eval 마다 다름 — knob 별 호출 필수
        v_true_by_knob[beta_eval] = v_true
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            records += evaluate_run(AXIS_ID, KNOB, beta_eval, seed, d, q_hat, v_true)
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11.6, 4.6))
    for ax, metric in ((ax_a, "mse"), (ax_b, "abs_bias")):
        ends = []
        for est in ESTIMATOR_ORDER:
            s = summary[summary["estimator"] == est].sort_values("knob_value")
            emph = est in EMPHASIZED
            if metric == "mse":
                y, se = s["mse"], s["mse_se"]
            else:
                y, se = s["bias"].abs(), s["est_se"]
            ax.plot(s["knob_value"], y, marker="o", **series_kwargs(est, emph))
            if emph:
                lo = (y - 2 * se).clip(lower=y.min() * 1e-2 if metric == "mse" else 0.0)
                ax.fill_between(s["knob_value"], lo, y + 2 * se,
                                color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
            ends.append((est, s["knob_value"].iloc[-1], y.iloc[-1], emph))
        ax.axvline(1.0, color=ORACLE_COLOR, linestyle=":", linewidth=1.2, zorder=1)
        ax.set_xscale("log")
        ax.set_xticks(GRID, labels=["0.5", "1", "3", "6", "10", "15"])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.set_xlabel(r"Evaluation-policy temperature $\beta_{eval}$ ($\beta_{log}=1$ fixed, log scale)")
        if metric == "mse":
            ax.set_yscale("log")
        _dodged_end_labels(ax, ends)
    ax_a.set_ylabel("MSE vs true value (log scale)")
    ax_a.set_title(r"A — MSE: DR best across the gap sweep ($\beta_{eval}\geq 1$); Clipped-IPS collapses")
    ax_b.set_ylabel("|bias| vs true value")
    ax_b.set_title("B — |bias|: clipping/model bias grow with the gap; IPS/DR stay centered")
    ax_b.annotate(r"no gap ($\beta_{eval}=\beta_{log}$)", xy=(1.0, 0.97),
                  xycoords=("data", "axes fraction"), xytext=(4, 0),
                  textcoords="offset points", fontsize=7.5, color=ORACLE_COLOR)
    ax_a.legend(ncol=4, loc="upper left")
    fig.suptitle("Axis 03 — Target-logging policy gap: weights stay bounded (near-uniform logging) —\n"
                 "variance grows only mildly and DM is never rescued; the real casualty is naive clipping\n"
                 r"$\hat q = 0.9q+0.05$ · shaded: ensemble $\pm$2SE over "
                 f"{len(SEEDS)} seeds · note: $v_{{true}}$ differs per $\\beta_{{eval}}$",
                 fontsize=10.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    base = summary[summary["variant"].fillna("") == ""]
    piv_mse = base.pivot_table(index="knob_value", columns="estimator", values="mse")
    piv_var = base.pivot_table(index="knob_value", columns="estimator", values="var")
    small, large = piv_mse.loc[1.0], piv_mse.loc[GRID[-1]]
    var_ratio = piv_var.loc[GRID[-1], "ips"] / piv_var.loc[1.0, "ips"]
    ess = base[base["estimator"] == "ips"].set_index("knob_value")["ess_ratio_raw_mean"]
    maxw = base[base["estimator"] == "ips"].set_index("knob_value")["max_weight_raw_mean"]
    print(f"[03] v_true by beta_eval: "
          + "  ".join(f"{b:g}:{v:.4f}" for b, v in v_true_by_knob.items())
          + f"  → {csv_path.name}, {fig_path.name}")
    print(f"[03] PATTERN ips_beats_dm_no_gap={small['ips'] < small['dm']}  "
          f"dr_beats_dm_no_gap={small['dr'] < small['dm']}  "
          f"dm_beats_ips_large_gap={large['dm'] < large['ips']}  "
          f"dm_beats_dr_large_gap={large['dm'] < large['dr']}  "
          f"ips_var_explodes={var_ratio > 100}(x{var_ratio:.1f})  "
          f"best_at_large_gap={large.idxmin()}")
    print(f"[03] PATTERN(observed inversions) snips_overtakes_ips_large_gap="
          f"{large['snips'] < large['ips']}  clipped_worst_large_gap="
          f"{large.idxmax() == 'clipped_ips'}  clipped_below_dm_large_gap="
          f"{large['clipped_ips'] > large['dm']}  "
          f"weights_bounded: mean_max_w {maxw.loc[1.0]:.2f}->{maxw.loc[GRID[-1]]:.2f}, "
          f"ess_ratio {ess.loc[1.0]:.3f}->{ess.loc[GRID[-1]]:.3f}")


if __name__ == "__main__":
    main()
