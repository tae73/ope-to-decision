"""축 02 — 로깅 stochasticity: softmax inverse-temperature β_log 스윕 (worse-than-uniform → 준결정).

노브: DGPConfig.beta_log ∈ {-1, 0, 0.5, 1, 2, 4, 8, 16}. S=40 (range(1000,1040)).
q̂ = 0.9·q + 0.05 (shrink α=0.9 — 축 01 과 동일 정책). 기대 패턴: β_log≥8(준결정 로깅)에서
IPS/DR 계열 MSE 폭발, DM 은 불변 — 같은 seed 의 x 표본이 β_log 와 무관하게 동일하므로
DM 추정치는 **정확히** 같다(rng draw 순서 x→U→action→noise 참조). raw ESS ratio 는 β_log 의
**비단조** 함수: β_eval=3 과 정렬되는 중간 구간(β_log≈2–4)에서 개선된 뒤 붕괴
(Plan 검증 실측 0.74→0.95→0.26→0.005) — B 패널이 그 자체로 교훈("ESS 는 β_log 의 단조 함수가
아니다"). 불발 시 그대로 보고(은폐 금지).

본 실행 실측(정직 기록): MSE 폭발은 β_log=16 에서만 성립(IPS ×143·DR ×202 vs 각자 min) —
β_log=8 은 ×2.2–2.5 열화에 그치고 **여전히 DM 보다 낮다**(cliff 는 8→16 사이). ESS 실측 시퀀스는
0.44→0.64→0.73→0.82→0.95→0.95→0.28→0.04 (S=40·n=10k — Plan 인용 수치와 모양 동일, 끝값 상이).
adaptive τ=p95(w) 의 Switch-DR·DRos 는 고 β_log 에서 오히려 개선(강한 truncation → DM-like).
단 Clipped-IPS 는 예외 — **양끝 모두 열화**(β_log=-1: 1.05e-2 ×146 vs 자체 min · @8: 5.34e-3
= DM 의 ×11 열세 · @16: 4.29e-2), 즉 "@8 에서 DM 을 이긴다"는 IPS/SNIPS/DR 한정.

산출: results/tables/02_logging_beta.csv (long, _common.COLUMNS)
    + results/figures/02_logging_beta.png (2패널 단일 PNG — A: MSE vs β_log(log y),
      B: mean raw ESS ratio vs β_log).
"""

import numpy as np

from _common import BASE_M2, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import (ESTIMATOR_ORDER, ORACLE_COLOR, apply_style, end_label,
                    save_figure, series_kwargs)
from ope.dgp import make_synthetic_bandit_data

AXIS_ID, SLUG = "02", "logging_beta"
KNOB = "beta_log"
GRID = [-1, 0, 0.5, 1, 2, 4, 8, 16]
SEEDS = range(1000, 1040)  # S=40
ALPHA_QHAT = 0.9
EMPHASIZED = ("ips", "snips", "dm")
BETA_EVAL = BASE_M2.beta_eval  # 3.0 — ESS 비단조의 정렬점(참조선)
ESS_COLOR = "#44433e"  # 진단 계열(estimator 아님) — entity 팔레트 밖 중립 dark


def _spread_log(ys: list[float], min_ratio: float = 1.5) -> list[float]:
    """끝단 직접 라벨의 y 위치만 log-공간에서 최소 간격으로 벌린다(데이터 불변 — 라벨 배치 전용)."""
    ys_arr = np.asarray(ys, dtype=float)
    order = np.argsort(ys_arr)
    vals = ys_arr[order].copy()
    for i in range(1, len(vals)):
        vals[i] = max(vals[i], vals[i - 1] * min_ratio)
    out = np.empty_like(ys_arr)
    out[order] = vals
    return out.tolist()


def main() -> None:
    v_true = cached_v_true(BASE_M2)
    records = []
    for beta_log in GRID:
        cfg = BASE_M2._replace(beta_log=beta_log)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
            records += evaluate_run(AXIS_ID, KNOB, beta_log, seed, d, q_hat, v_true)
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    summary = summarize(csv_path)

    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullLocator

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(12.0, 4.8))

    # ── 패널 A: MSE vs β_log (log y) ────────────────────────────────────────
    end_ys, end_meta = [], []
    for est in ESTIMATOR_ORDER:
        s = summary[summary["estimator"] == est].sort_values("knob_value")
        emph = est in EMPHASIZED
        ax_a.plot(s["knob_value"], s["mse"], marker="o", **series_kwargs(est, emph))
        if emph:
            ax_a.fill_between(s["knob_value"], s["mse"] - 2 * s["mse_se"],
                              s["mse"] + 2 * s["mse_se"],
                              color=series_kwargs(est, True)["color"], alpha=0.15, lw=0)
        end_ys.append(float(s["mse"].iloc[-1]))
        end_meta.append((float(s["knob_value"].iloc[-1]), est, emph))
    for y_lab, (x_end, est, emph) in zip(_spread_log(end_ys), end_meta):
        end_label(ax_a, x_end, y_lab, est, emph)
    ax_a.axvline(BETA_EVAL, color=ORACLE_COLOR, ls="--", lw=1.0, zorder=1)
    ax_a.set_xscale("symlog", linthresh=1)
    ax_a.xaxis.set_minor_locator(NullLocator())
    ax_a.set_xticks(GRID, [str(g) for g in GRID])
    ax_a.set_yscale("log")
    ax_a.set_xlabel(r"Logging inverse-temperature $\beta_{\mathrm{log}}$ (symlog scale)")
    ax_a.set_ylabel("MSE vs true value (log scale)")
    ax_a.set_title(r"A — IPS/SNIPS/DR blow up at $\beta_{\mathrm{log}}=16$ (at 8 they still beat DM);"
                   "\nadaptive Switch-DR/DRos survive; DM exactly invariant")
    ax_a.legend(ncol=2, loc="upper left")

    # ── 패널 B: mean raw ESS ratio vs β_log (진단은 estimator 공통 — IPS 행에서 취득) ──
    ess = (summary[summary["estimator"] == "ips"]
           .sort_values("knob_value")[["knob_value", "ess_ratio_raw_mean"]])
    ax_b.plot(ess["knob_value"], ess["ess_ratio_raw_mean"], marker="o",
              color=ESS_COLOR, linewidth=2.0, zorder=3,
              label="mean raw ESS ratio (diagnostic)")
    for _, row in ess.iterrows():
        v = row["ess_ratio_raw_mean"]
        ax_b.annotate(f"{v:.2f}" if v >= 0.01 else f"{v:.3f}",
                      xy=(row["knob_value"], v), xytext=(0, 7),
                      textcoords="offset points", ha="center",
                      fontsize=7.5, color=ESS_COLOR)
    ax_b.axvline(BETA_EVAL, color=ORACLE_COLOR, ls="--", lw=1.0, zorder=1,
                 label=r"$\beta_{\mathrm{eval}}=3$ (alignment)")
    ax_b.set_xscale("symlog", linthresh=1)
    ax_b.xaxis.set_minor_locator(NullLocator())
    ax_b.set_xticks(GRID, [str(g) for g in GRID])
    ax_b.set_ylim(-0.04, 1.12)
    ax_b.set_xlabel(r"Logging inverse-temperature $\beta_{\mathrm{log}}$ (symlog scale)")
    ax_b.set_ylabel("Mean ESS ratio of raw weights")
    ax_b.set_title(r"B — ESS is NOT monotone in $\beta_{\mathrm{log}}$:"
                   "\nit peaks where logging aligns with the target, then collapses")
    ax_b.legend(loc="lower left")

    fig.suptitle(r"Axis 02 — Logging stochasticity: worse-than-uniform ($\beta_{\mathrm{log}}<0$) "
                 r"$\to$ near-deterministic ($\beta_{\mathrm{log}}=16$)"
                 "\n" + r"$\hat q = 0.9q+0.05$ (mild misspecification) · shaded: ensemble "
                 r"$\pm$2SE over " + f"{len(SEEDS)} seeds", y=1.02)
    fig.tight_layout()
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출)
    piv = summary[summary["variant"].fillna("") == ""].pivot_table(
        index="knob_value", columns="estimator", values="mse")
    ess_seq = ess["ess_ratio_raw_mean"].to_numpy()
    peak = int(np.argmax(ess_seq))
    dm_ratio = piv["dm"].max() / piv["dm"].min()
    ips_r8, ips_r16 = piv.loc[8, "ips"] / piv["ips"].min(), piv.loc[16, "ips"] / piv["ips"].min()
    dr_r8, dr_r16 = piv.loc[8, "dr"] / piv["dr"].min(), piv.loc[16, "dr"] / piv["dr"].min()
    print(f"[02] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[02] PATTERN ips_dr_explode_at_16={(ips_r16 > 100) and (dr_r16 > 100)}  "
          f"ips_dr_explode_at_8={(ips_r8 > 10) and (dr_r8 > 10)}  "
          f"(mse/min — ips: x{ips_r8:.1f}@8, x{ips_r16:.1f}@16; dr: x{dr_r8:.1f}@8, x{dr_r16:.1f}@16; "
          f"@8 both still below DM: ips={piv.loc[8, 'ips']:.2e}, dr={piv.loc[8, 'dr']:.2e} "
          f"vs dm={piv.loc[8, 'dm']:.2e})")
    print(f"[02] PATTERN dm_invariant={bool(dm_ratio < 1.05)}  (max/min={dm_ratio:.6f})")
    print(f"[02] PATTERN ess_nonmonotone="
          f"{0 < peak < len(ess_seq) - 1 and ess_seq[peak] > ess_seq[0] and ess_seq[-1] < ess_seq[peak]}  "
          f"(peak at beta_log={ess['knob_value'].iloc[peak]:g}; seq="
          f"{np.array2string(ess_seq, precision=3, separator=', ')})")


if __name__ == "__main__":
    main()
