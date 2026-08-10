"""축 20 — OBD decision card: reveal 없는 완전 실전 1-page 판정 (M8 GT-미상 본편).

ZOZO OBD small BTS 실로그(축 12 프로토콜 계승 — campaign=all·uniform 타깃 단방향)에
frontstage 프로토콜 전체를 돌려 **한 장의 판정 카드**를 만든다: 추정+CI(weighting 3종 —
컨텍스트 피처 부재로 q̂ 모델 없음·DM/DR 트랙은 축 12 참조) · 진단·gate v1 · validity battery ·
Λ-부채꼴(SNIPS MSM band vs incumbent anchor) · protocol verdict · decision.

**reveal 파일은 존재하지 않는다** — 이것이 실전이다. 근사 GT 와의 대조는 축 12(LEDGER
`m3-12-gate-demo`) 소관이며 여기서 재서술하지 않는다(구간 언어 규약 §3.4 계승). 시연
프레임: 이 카드는 프로토콜이 실로그에서 *작동함* 을 보이는 것이지 판정의 *정확성* 을
검증하는 것이 아니다. inconclusive/abstention 은 정당한 출력이다(GLOSSARY §8).

time_split(보고 전용 §3.5-1): OBD 는 시간순 로그 — 전/후반 SNIPS gap 을 카드에 병기.

산출: results/tables/20_obd_decision_card_decision.csv (+ _card.csv 카드 수치 정본)
      ↔ results/figures/20_obd_decision_card.png — reveal CSV 없음(계약).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from _common import TAB_DIR
from _practitioner import (
    DECISION_ESTIMATOR, LAM_MAX, PractitionerLog, run_protocol, write_decision_csv,
)
from _style import ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style, save_figure
from ope.datasets import load_obd_small
from ope.estimators import msm_snips_bounds
from ope.validity import ValidityConfig, time_split_gap

AXIS_ID, SLUG = "20", "obd_decision_card"
DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "obd"
N_ACTIONS = 80
SEED = 0
LAM_GRID = np.geomspace(1.0, LAM_MAX, 24)


def main() -> None:
    t0 = time.perf_counter()
    df = load_obd_small("bts", data_dir=str(DATA_DIR))
    reward = df["click"].to_numpy(dtype=float)
    action = df["item_id"].to_numpy(dtype=int)
    pscore = df["propensity_score"].to_numpy(dtype=float)
    n = len(reward)
    pi_e = np.full((n, N_ACTIONS), 1.0 / N_ACTIONS)

    # frontstage — 컨텍스트 피처 부재: context 없음(nc_covariate n/a)·q̂ 없음(weighting 3종)
    log = PractitionerLog(context=None, action=action, reward=reward, pscore=pscore)
    rows = run_protocol(log, pi_e, axis_id=AXIS_ID, scenario="bts_uniform_target",
                        run_id="obd-bts-all", seed=SEED, cfg=ValidityConfig(seed=SEED))
    dec_path = write_decision_csv(AXIS_ID, SLUG, rows)
    assert not (TAB_DIR / f"{AXIS_ID}_{SLUG}_reveal.csv").exists(), "축 20 은 reveal 없음(계약)"

    r0 = rows[0]  # battery·진단·verdict 는 estimator 행에 공통
    incumbent = r0["incumbent_mean_r"]
    tsplit = time_split_gap(reward, action, pscore, pi_e, np.arange(n))  # 로그가 시간순
    bands = [msm_snips_bounds(reward, action, pscore, pi_e, float(lam))
             for lam in LAM_GRID]

    card = {
        "n_rounds": n, "n_actions": N_ACTIONS, "incumbent_mean_r": incumbent,
        "time_split_gap_snips": tsplit,
        **{f"{k}": r0[k] for k in
           ("ess_ratio_raw", "max_weight_raw", "support_proxy", "gate_v1",
            "mean_w", "mean_w_state", "harmonic_worst_t", "harmonic_state",
            "placebo_value", "placebo_state", "disagreement", "disagreement_state",
            "lam_star_flip", "lam_star_censored", "lam_direction",
            "protocol_verdict", "decision", "fragile")},
    }
    card_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_card.csv"
    pd.DataFrame([card]).to_csv(card_path, index=False)

    # ── 1-page decision card figure ──────────────────────────────────────────────
    apply_style()
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(11.5, 7.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.28)

    ax = fig.add_subplot(gs[0, 0])  # 추정 + CI
    ests = [r for r in rows]
    ys = np.arange(len(ests))
    for i, r in enumerate(ests):
        c = ESTIMATOR_COLORS[r["estimator"]]
        ax.errorbar(r["estimate"], i,
                    xerr=[[r["estimate"] - r["ci_lo"]], [r["ci_hi"] - r["estimate"]]],
                    fmt="o", color=c, capsize=3)
    ax.axvline(incumbent, color="#33322e", lw=1.4, ls="--",
               label=f"incumbent mean(r) = {incumbent:.4f}")
    ax.set_yticks(ys, [ESTIMATOR_LABELS[r["estimator"]] for r in ests])
    ax.set_xlabel("V̂(π_e)  (joint bootstrap 95% CI)")
    ax.set_title("Estimates — weighting family only\n(no context features ⇒ no q̂ model; "
                 "DM/DR track: axis 12)", fontsize=9.5)
    ax.legend(fontsize=7.5)

    bx = fig.add_subplot(gs[0, 1])  # Λ-부채꼴
    lo = np.array([b[0] for b in bands])
    hi = np.array([b[1] for b in bands])
    bx.fill_between(LAM_GRID, lo, hi, color="#b3b2a9", alpha=0.45,
                    label="SNIPS MSM band [V−(Λ), V+(Λ)]")
    bx.plot(LAM_GRID, lo, color="#6f6e66", lw=1.0)
    bx.plot(LAM_GRID, hi, color="#6f6e66", lw=1.0)
    bx.axhline(incumbent, color="#33322e", lw=1.4, ls="--", label="incumbent mean(r)")
    bx.axvline(card["lam_star_flip"], color="#e34948", lw=1.4,
               label=f"Λ*_flip = {card['lam_star_flip']:.3f}")
    bx.set_xscale("log")
    bx.set_xticks([1, 2, 4, 8], ["1", "2", "4", "8"])
    bx.set_xlabel("sensitivity Λ (log — assumption scale, not identified from data)")
    bx.set_ylabel("V̂(π_e) worst-case band")
    bx.set_title("Λ-fan — how much recorded-propensity distortion\nwould flip the "
                 "comparison (Kallus & Zhou tool, axis 14)", fontsize=9.5)
    bx.legend(fontsize=7.5, loc="upper left")

    cx = fig.add_subplot(gs[1, 0])  # 진단 + battery 표
    cx.axis("off")
    tbl = [
        ["gate v1 (pre-registered M1)", "", ""],
        ["  ESS/n", f"{card['ess_ratio_raw']:.4f}", "soft 0.10 / hard 0.01"],
        ["  max weight", f"{card['max_weight_raw']:.1f}", "cap 100"],
        ["  support proxy", f"{card['support_proxy']:.4f}", "max 0.02"],
        ["  → verdict", card["gate_v1"].upper(), ""],
        ["validity battery [proposal]", "", ""],
        ["  E[w] (HT identity)", f"{card['mean_w']:.4f}", card["mean_w_state"]],
        ["  harmonic worst T(a*)", f"{card['harmonic_worst_t']:.3f}", card["harmonic_state"]],
        ["  placebo (truth 0)", f"{card['placebo_value']:.2e}", card["placebo_state"]],
        ["  disagreement", (f"{card['disagreement']:.3f}"
                            if np.isfinite(card["disagreement"]) else "—"),
         card["disagreement_state"]],
        ["  time-split SNIPS gap", f"{card['time_split_gap_snips']:.2e}", "report-only"],
    ]
    t = cx.table(cellText=tbl, colLabels=["signal (logs only)", "value", "state/threshold"],
                 loc="center", cellLoc="left", colWidths=[0.48, 0.26, 0.26])
    t.auto_set_font_size(False)
    t.set_fontsize(8)
    t.scale(1.0, 1.32)
    cx.set_title("Diagnostics + battery — all computable without ground truth",
                 fontsize=9.5, pad=18)

    dx = fig.add_subplot(gs[1, 1])  # verdict box
    dx.axis("off")
    verdict_color = {"trust": "#1baf7a", "distrust": "#eb6834",
                     "ab_fallback": "#e34948"}[card["protocol_verdict"]]
    dx.add_patch(plt.Rectangle((0.02, 0.52), 0.96, 0.40, transform=dx.transAxes,
                               facecolor=verdict_color, alpha=0.14,
                               edgecolor=verdict_color, lw=2))
    dx.text(0.5, 0.80, f"PROTOCOL VERDICT: {card['protocol_verdict'].upper()}",
            transform=dx.transAxes, ha="center", fontsize=14, color=verdict_color,
            fontweight="bold")
    dx.text(0.5, 0.62, f"DECISION: {card['decision'].upper().replace('_', ' ')}"
            + ("   ·   FRAGILE (Λ*_flip < 1.5)" if card["fragile"] else ""),
            transform=dx.transAxes, ha="center", fontsize=11, color="#33322e")
    dx.text(0.02, 0.40, "No reveal file exists for this axis — this is what the\n"
            "protocol looks like in real life: a verdict you must act on\n"
            "without ever learning the truth. Backstage cross-check with the\n"
            "approximate ground truth (±32% CI band) lives in axis 12 only.",
            transform=dx.transAxes, fontsize=9, va="top", color="#44433e")
    dx.text(0.02, 0.02, "Frame: demonstration of protocol operation — NOT accuracy "
            "evidence.\nRules pre-registered (PLAN §3.5); thresholds untuned.",
            transform=dx.transAxes, fontsize=8, va="bottom", color="#6f6e66",
            style="italic")

    fig.suptitle("Axis 20 — Decision card on a production log (ZOZO OBD small · BTS → "
                 "uniform target)\nfrontstage only: every number on this page is "
                 "computed from the log and the candidate policy alone", fontsize=11.5)
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    print(f"[20] → {dec_path.name}, {card_path.name}, {fig_path.name} "
          f"({time.perf_counter() - t0:.0f}s)")
    print(f"[20] PATTERN(기대 — 축 12 선례): gate v1 = {card['gate_v1']} "
          f"(distrust 예상) · decision = {card['decision']} (ab_test 예상)")
    print(f"[20] 실측: E[w]={card['mean_w']:.4f}({card['mean_w_state']}) · "
          f"harmonic worst T={card['harmonic_worst_t']:.3f}({card['harmonic_state']}) · "
          f"placebo {card['placebo_state']} · disagreement="
          f"{card['disagreement']}({card['disagreement_state']}) · "
          f"Λ*_flip={card['lam_star_flip']:.3f}({card['lam_direction']}"
          f"{', censored' if card['lam_star_censored'] else ''}) · "
          f"time-split gap={card['time_split_gap_snips']:.2e}")


if __name__ == "__main__":
    main()
