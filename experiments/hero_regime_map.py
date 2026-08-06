"""hero ② — regime map: 어떤 estimator 가 어디서 이기고, 게이트는 언제 "믿지 말라"는가.

M3 hero 후보 ② (PLAN §1 M3 "hero 3장 확정"). 축 01(표본 n)과 축 02(로깅 β_log)를 2D factorial 로
합쳐 한 장의 지도로 만든다 — README hero 후보이므로 figure 캡션은 self-contained 로 저작한다.

설계 (사전 정의 — 사후 재튜닝 금지):
- factorial n ∈ {500, 2000, 8000, 32000} × β_log ∈ {-1, 0.5, 1, 2, 4, 8, 16}, S=30(seed 1000–1029),
  BASE_M2 기반(β_eval=3 고정), q̂ = 0.9·q_true + 0.05 (축 01 과 동일한 mild misspecification).
- cell 승자 = S-seed MSE 최저 estimator. **동률 규칙(사전 정의)**: 경쟁자 MSE ≤ 승자 MSE +
  2·(승자 mse_se) 이면 tie 표기(≈) — MC 오차 범위 내 승자 단정 금지.
- 게이트 verdict = run(=seed)별 decision_gate(PROVISIONAL_THRESHOLDS, **raw 진단**)의 S=30
  **다수결**(plurality; 동수 시 보수 순서 ab_fallback > distrust > trust). 임계값 재튜닝 금지.

산출:
- results/tables/hero_regime_map.csv — long, _common.COLUMNS 스키마 그대로.
  knob_name="n_x_beta_log", knob_value="{n}x{beta}" 문자열 (예: "500x-1", "32000x0.5").
  게이트 다수결은 이 CSV 의 raw 진단 열(ess_ratio_raw·max_weight_raw·support_proxy)에서 재도출 가능.
- results/tables/hero_regime_map_summary.csv — **_common 스키마 예외(companion 요약)**: cell 별
  n, beta_log, knob_value, winner, winner_mse, winner_mse_se, runner_up, runner_mse, tie,
  tie_partners(';' 연결), gate_majority, n_trust, n_distrust, n_ab_fallback.
  long CSV 에서 전량 재도출 가능한 파생 캐시다(정본은 long CSV).
- results/figures/hero_regime_map.png — 히트맵: cell 색=승자 estimator entity 색(_style),
  cell 텍스트=승자 라벨(+tie ≈ 병기)+게이트 verdict 약자(T/D/F), 축=n(세로, log-spaced)×β_log(가로).

정직 각주: β_log=16 열은 준결정적 로깅이라 단일 seed 의 weight 폭발이 MSE 를 지배할 수 있다 —
그대로 보고(figure 각주 병기·수치는 long CSV 재도출)하고 스윕·임계 재조정은 하지 않는다.
기대 불발 시에도 은폐 없이 그대로 보고한다.
"""

from collections import Counter

import numpy as np
import pandas as pd

from _common import BASE_M2, TAB_DIR, cached_v_true, evaluate_run, summarize, write_axis_csv
from _style import (ESTIMATOR_COLORS, ESTIMATOR_LABELS, ESTIMATOR_ORDER, SURFACE,
                    apply_style, save_figure)
from ope.dgp import make_synthetic_bandit_data
from ope.diagnostics import PROVISIONAL_THRESHOLDS, compute_diagnostics, decision_gate

AXIS_ID, SLUG = "hero", "regime_map"
KNOB = "n_x_beta_log"
N_GRID = [500, 2000, 8000, 32000]
BETA_GRID = [-1, 0.5, 1, 2, 4, 8, 16]
SEEDS = range(1000, 1030)  # S=30
ALPHA_QHAT = 0.9
VERDICT_ABBREV = {"trust": "T", "distrust": "D", "ab_fallback": "F"}
SEVERITY = {"ab_fallback": 2, "distrust": 1, "trust": 0}  # 다수결 동수 시 보수(높은 쪽) 우선
WEIGHTING = {"ips", "snips", "clipped_ips", "dr", "switch_dr", "dros"}


def _cell_key(n: int, beta: float) -> str:
    return f"{n}x{beta:g}"


def _cell_ink(hex_color: str) -> str:
    """cell 텍스트 잉크 — WCAG 상대휘도 기반으로 white/dark 중 대비 큰 쪽 선택."""
    def lin(c: float) -> float:
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(hex_color[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    lum = 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)
    contrast_white = 1.05 / (lum + 0.05)
    contrast_dark = (lum + 0.05) / (0.0104 + 0.05)  # dark ink #1a1a18 의 상대휘도 ≈ 0.0104
    return "#ffffff" if contrast_white >= contrast_dark else "#1a1a18"


def main() -> None:
    v_true = cached_v_true(BASE_M2)  # β_eval=3 고정 — 전 cell 공통 참값
    records = []
    gate_counts: dict[str, Counter] = {}
    for n in N_GRID:
        for beta in BETA_GRID:
            cell = _cell_key(n, beta)
            cfg = BASE_M2._replace(n=n, beta_log=float(beta))
            cnt: Counter = Counter()
            for seed in SEEDS:
                d = make_synthetic_bandit_data(cfg._replace(seed=seed))
                q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
                records += evaluate_run(AXIS_ID, KNOB, cell, seed, d, q_hat, v_true)
                diag = compute_diagnostics(d.pscore_logged, d.action, d.pi_e_dist)
                cnt[decision_gate(diag, PROVISIONAL_THRESHOLDS).decision] += 1
            gate_counts[cell] = cnt
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)

    # cell 별 승자·동률·게이트 다수결 (규칙은 docstring 의 사전 정의 그대로)
    summary = summarize(csv_path)
    summary = summary[summary["variant"].fillna("") == ""]
    rows = []
    for n in N_GRID:
        for beta in BETA_GRID:
            cell = _cell_key(n, beta)
            s = summary[summary["knob_value"] == cell].set_index("estimator")
            order = s["mse"].sort_values()
            winner, runner = order.index[0], order.index[1]
            w_mse, w_se = float(order.iloc[0]), float(s.loc[winner, "mse_se"])
            partners = [e for e in order.index[1:] if order[e] <= w_mse + 2.0 * w_se]
            cnt = gate_counts[cell]
            top = max(cnt.values())
            majority = max((k for k, v in cnt.items() if v == top), key=SEVERITY.get)
            rows.append({
                "n": n, "beta_log": beta, "knob_value": cell,
                "winner": winner, "winner_mse": w_mse, "winner_mse_se": w_se,
                "runner_up": runner, "runner_mse": float(order.iloc[1]),
                "tie": bool(partners), "tie_partners": ";".join(partners),
                "gate_majority": majority,
                "n_trust": cnt.get("trust", 0), "n_distrust": cnt.get("distrust", 0),
                "n_ab_fallback": cnt.get("ab_fallback", 0),
            })
    cells = pd.DataFrame(rows)
    summary_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_summary.csv"
    cells.to_csv(summary_path, index=False)

    # β=16 정직 각주 수치 — 단일 seed 의 squared-error 지배 비중 (long CSV 에서 재도출 가능)
    df = pd.read_csv(csv_path)
    df = df[df["variant"].fillna("") == ""].assign(sq=lambda t: (t["estimate"] - t["v_true"]) ** 2)
    ips16 = df[(df["estimator"] == "ips") & (df["knob_value"].str.endswith("x16"))]
    share16 = float((ips16.groupby("knob_value")["sq"].apply(lambda g: g.max() / g.sum())).max())

    # ---- figure: 승자 히트맵 -------------------------------------------------
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    fig, ax = plt.subplots(figsize=(11.0, 5.4))
    lut = cells.set_index("knob_value")
    for i, n in enumerate(N_GRID):
        for j, beta in enumerate(BETA_GRID):
            row = lut.loc[_cell_key(n, beta)]
            color = ESTIMATOR_COLORS[row["winner"]]
            ink = _cell_ink(color)
            ax.add_patch(Rectangle((j, i), 1.0, 1.0, facecolor=color,
                                   edgecolor=SURFACE, linewidth=2.0))
            label = ESTIMATOR_LABELS[row["winner"]] + (" ≈" if row["tie"] else "")
            ax.text(j + 0.5, i + 0.60, label, ha="center", va="center",
                    fontsize=8.5, fontweight="bold", color=ink)
            ax.text(j + 0.5, i + 0.28, VERDICT_ABBREV[row["gate_majority"]],
                    ha="center", va="center", fontsize=8.0, color=ink, alpha=0.9)

    ax.set_xlim(0, len(BETA_GRID))
    ax.set_ylim(0, len(N_GRID))
    ax.set_xticks([j + 0.5 for j in range(len(BETA_GRID))])
    ax.set_xticklabels([f"{b:g}" for b in BETA_GRID])
    ax.set_yticks([i + 0.5 for i in range(len(N_GRID))])
    ax.set_yticklabels([f"{n:,}" for n in N_GRID])
    ax.grid(False)
    ax.tick_params(length=0)
    for side in ("left", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r"Logging inverse-temperature $\beta_{log}$  (overlap shrinks $\rightarrow$;"
                  r" $-1$ = worse-than-uniform logging)")
    ax.set_ylabel("Sample size n (log-spaced grid)")
    ax.set_title("Hero — OPE regime map: which estimator wins where (lowest MSE), "
                 "and when the gate says stop")

    # 범례 정직성: 승자 아닌 estimator 를 tie-band 등장/완전 부재로 구분해 병기 (데이터 파생)
    winners_set = set(cells["winner"])
    partners_union = {p for ps in cells["tie_partners"] for p in ps.split(";") if p}
    tie_only = [e for e in ESTIMATOR_ORDER if e in partners_union - winners_set]
    absent = [e for e in ESTIMATOR_ORDER if e not in winners_set | partners_union]
    handles = [Patch(facecolor=ESTIMATOR_COLORS[e], edgecolor="none",
                     label=ESTIMATOR_LABELS[e]) for e in ESTIMATOR_ORDER]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.02, 1.0),
              frameon=False, title="Winner (entity colors)", alignment="left")
    tie_only_txt = " · ".join(ESTIMATOR_LABELS[e] for e in tie_only) or "—"
    absent_txt = " · ".join(ESTIMATOR_LABELS[e] for e in absent) or "—"
    ax.text(1.02, 0.46, f"Never wins a cell outright:\n{tie_only_txt}\n"
                        f"(tie bands only) ·\n{absent_txt} (nowhere)",
            transform=ax.transAxes, fontsize=8.0, va="top", color="#55554f")
    ax.text(1.02, 0.24, "Cell letter — gate majority\nover S=30 runs:\n"
                        "T = trust\nD = distrust\nF = ab_fallback\n"
                        "(provisional thresholds —\nthis repo's proposal,\nnot a literature standard)",
            transform=ax.transAxes, fontsize=8.0, va="top", color="#55554f")

    # 게이트 소표본 저감도(β=16 열) — non-trust run 수의 n-단조 증가 (summary CSV 재도출 가능)
    nontrust16 = [len(SEEDS) - int(lut.loc[_cell_key(n, 16), "n_trust"]) for n in N_GRID]
    nt16_txt = " → ".join(f"{c}/{len(SEEDS)}" for c in nontrust16)
    cell16 = _cell_key(N_GRID[0], 16)
    ips16_small = df[(df["estimator"] == "ips") & (df["knob_value"] == cell16)]["sq"].mean()
    ips_vs_winner16 = float(ips16_small / lut.loc[cell16, "winner_mse"])
    caption = (
        "Winner-take-all regime map over 7 OPE estimators, synthetic DGP (BASE_M2 environment, struct_seed=7): "
        "factorial n × $\\beta_{log}$, evaluation policy fixed ($\\beta_{eval}$=3),\n"
        "$\\hat q$ = 0.9·q_true + 0.05 (mild misspecification), S=30 seeds (1000–1029). "
        "Cell color/text = estimator with lowest MSE across seeds; '≈' = at least one competitor\n"
        "within winner MSE + 2·SE(MSE) (pre-declared tie rule — winner not separated from runner-up at MC "
        "resolution). Letter = majority decision_gate verdict over the 30 runs\n"
        "(raw-weight diagnostics; thresholds fixed a priori, no re-tuning). "
        f"Footnote: at $\\beta_{{log}}$=16 logging is near-deterministic and single-seed weight explosions "
        f"dominate MSE (one seed carries up to {share16:.0%} of the\n"
        f"IPS squared-error sum in that column; reported as-is). In that column the gate's warning rate grows "
        f"with n (non-trust runs {nt16_txt} from n=500 to 32,000):\n"
        f"with few near-deterministic draws the explosive weights are rarely sampled, so the n=500 cell still "
        f"reads T even though raw-IPS MSE there is ≈{ips_vs_winner16:.0f}× the winner's —\n"
        "the diagnostic loses power exactly where the log is least informative. "
        "Data: results/tables/hero_regime_map.csv (per-run) · hero_regime_map_summary.csv (per-cell)."
    )
    fig.text(0.045, 0.015, caption, fontsize=7.6, color="#55554f", va="top", linespacing=1.45)

    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ---- 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ---------------
    win = dict(zip(cells["knob_value"], cells["winner"]))
    gate = dict(zip(cells["knob_value"], cells["gate_majority"]))
    p_regime_boundary = len(set(win.values())) >= 2
    p_dm_low_overlap_small_n = win["500x8"] == "dm" and win["500x16"] == "dm"
    p_weighting_large_n_mid_beta = all(
        win[_cell_key(32000, b)] in WEIGHTING for b in (0.5, 1, 2))
    p_gate_flags_beta16 = all(gate[_cell_key(n, 16)] != "trust" for n in N_GRID)
    print(f"[hero] v_true={v_true:.6f}  → {csv_path.name}, {summary_path.name}, {fig_path.name}")
    print(f"[hero] winners (rows n={N_GRID} bottom→top, cols beta={BETA_GRID}):")
    for n in reversed(N_GRID):
        row = "  ".join(f"{win[_cell_key(n, b)]:>11s}/{VERDICT_ABBREV[gate[_cell_key(n, b)]]}"
                        for b in BETA_GRID)
        print(f"[hero]   n={n:>6d}: {row}")
    print(f"[hero] PATTERN regime_boundary_exists={p_regime_boundary}  "
          f"dm_wins_small_n_low_overlap={p_dm_low_overlap_small_n}  "
          f"weighting_wins_large_n_mid_beta={p_weighting_large_n_mid_beta}  "
          f"gate_nontrust_all_beta16={p_gate_flags_beta16}")
    print(f"[hero] FOOTNOTE beta16 max single-seed share of IPS sq-error sum = {share16:.4f}")
    print(f"[hero] FOOTNOTE beta16 non-trust runs (n={N_GRID}): {nontrust16}  "
          f"ips_mse/winner_mse at {cell16} = {ips_vs_winner16:.1f}")
    n_tie = int(cells['tie'].sum())
    print(f"[hero] ties={n_tie}/28 cells  gate majorities: "
          f"{dict(Counter(cells['gate_majority']))}")


if __name__ == "__main__":
    main()
