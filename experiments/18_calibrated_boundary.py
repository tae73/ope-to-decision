"""축 18 — calibrated-confounding 경계 전시: battery 가 원리적으로 못 보는 것 (M8 GT-미상 본편).

그림 3(축 09 — "진단이 못 보는 것")의 GT-미상 세대교체: 축 09 는 ESS 의 blind 를 참값으로
갈라 보였다면, 본 축은 **validity battery 전체**가 못 보는 경계를 관측 동등성 구성으로
전시한다. 기록 방식 2종을 나란히:
- `as_recorded` — 의도값 기록(축 09 장치·pscore_logged). probe M8-B 실측: 이 miscalibration
  조차 사전등록 임계에 원거리 미달(발화 0/5) — "부분 검출" 예상 반증.
- `calibrated` — U-주변화 marginal 기록(`marginal_logging_dist` — 결정적 구적). 이 세계에선
  **어떤 로그 통계도 confounding 을 구별할 수 없다**(관측 동등성 — 정리). battery null 은
  구현의 성공이 아니라 원리의 전시다.

frontstage(battery·Λ*_flip — 로그만) 는 양 모드에서 평평하고, backstage(reveal — bias) 만
γ 와 함께 자란다. **출구는 점추정이 아니라 Λ-감도 구간**(축 14 도구·`lambda_star_vs_anchor`) —
구간의 폭과 abstention 이 정직한 답이다(PLAN §3.5 원칙·CLAUDE.md §5 co-exhibit).

산출: results/tables/18_calibrated_boundary_decision.csv · _reveal.csv · _summary.csv
      ↔ results/figures/18_calibrated_boundary.png
"""

import time

import numpy as np
import pandas as pd

from _common import BASE_M2, SEEDS_DEFAULT, TAB_DIR, cached_v_true
from _practitioner import (
    DECISION_ESTIMATOR, PractitionerLog, reveal, run_protocol, write_decision_csv,
)
from _style import ESTIMATOR_COLORS, apply_style, save_figure
from ope.dgp import make_synthetic_bandit_data, marginal_logging_dist
from ope.fitters import CrossFitConfig, fit_q_hat_crossfit
from ope.validity import ValidityConfig

AXIS_ID, SLUG = "18", "calibrated_boundary"
GAMMAS = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)   # 축 09 grid 계승(0→2.5)
MODES = ("as_recorded", "calibrated")
SEEDS = SEEDS_DEFAULT                       # S=40
CAL_ARMS = ("mean_w", "harmonic")           # calibration arm — 관측 동등성 정리의 대상


def main() -> None:
    t0 = time.perf_counter()
    all_rows = []
    for gamma in GAMMAS:
        cfg = BASE_M2._replace(confounding_strength=gamma)
        for seed in SEEDS:
            c = cfg._replace(seed=seed)
            d = make_synthetic_bandit_data(c)
            idx = np.arange(c.n)
            p_marg = marginal_logging_dist(c, d.context)
            q_hat = fit_q_hat_crossfit(d.context, d.action, d.reward, c.n_actions,
                                       CrossFitConfig(seed=seed))
            for mode in MODES:
                ps = d.pscore_logged if mode == "as_recorded" else p_marg[idx, d.action]
                log = PractitionerLog(context=d.context, action=d.action, reward=d.reward,
                                      pscore=ps,
                                      pi_log_dist=d.pi_log_dist if mode == "as_recorded"
                                      else p_marg)
                all_rows += run_protocol(
                    log, d.pi_e_dist, axis_id=AXIS_ID, scenario=f"{mode}_g{gamma}",
                    run_id=f"{mode}-g{gamma}-{seed}", seed=seed, q_hat=q_hat,
                    cfg=ValidityConfig(seed=seed))
    dec_path = write_decision_csv(AXIS_ID, SLUG, all_rows)

    v_true = cached_v_true(BASE_M2)  # γ 와 무관(설계 정리 — tests/test_dgp.py 고정)
    truth = pd.DataFrame({"run_id": [f"{m}-g{g}-{s}" for g in GAMMAS for s in SEEDS
                                     for m in MODES],
                          "v_true": v_true})
    rev_path = reveal(dec_path, truth, "exact_synthetic")

    dec = pd.read_csv(dec_path)
    rev = pd.read_csv(rev_path)
    runs = dec[dec["estimator"] == DECISION_ESTIMATOR].copy()
    runs["gamma"] = runs["scenario"].str.rsplit("_g", n=1).str[1].astype(float)
    runs["mode"] = runs["scenario"].str.rsplit("_g", n=1).str[0]
    r_ips = rev[rev["estimator"] == "ips"][["run_id", "estimate"]].rename(
        columns={"estimate": "est_ips"})
    r_sn = rev[rev["estimator"] == DECISION_ESTIMATOR][["run_id", "estimate"]].rename(
        columns={"estimate": "est_snips"})
    runs = runs.merge(r_ips, on="run_id").merge(r_sn, on="run_id")

    summary = (runs.groupby(["mode", "gamma"])
               .agg(cal_fire=("run_id", "size"),  # placeholder — 아래서 대체
                    mean_w_mean=("mean_w", "mean"), mean_w_sd=("mean_w", "std"),
                    bias_ips=("est_ips", lambda s: s.mean() - v_true),
                    bias_snips=("est_snips", lambda s: s.mean() - v_true),
                    se_snips=("est_snips", lambda s: s.std(ddof=1) / np.sqrt(len(s))),
                    lam_star_med=("lam_star_flip", "median"),
                    verdict_nontrust=("protocol_verdict",
                                      lambda s: (s != "trust").mean()))
               .reset_index())
    fire = (runs.assign(cal_fired=(runs["mean_w_state"] == "fail")
                        | (runs["harmonic_state"] == "fail"))
            .groupby(["mode", "gamma"])["cal_fired"].mean().reset_index()
            .rename(columns={"cal_fired": "cal_arm_fire_rate"}))
    summary = summary.drop(columns=["cal_fire"]).merge(fire, on=["mode", "gamma"])
    sum_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_summary.csv"
    summary.to_csv(sum_path, index=False)

    # ── figure: A frontstage(평평) / B backstage(성장) ────────────────────────────
    apply_style()
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    mode_style = {"calibrated": {"ls": "-", "marker": "o"},
                  "as_recorded": {"ls": "--", "marker": "s"}}
    for mode in MODES:
        g = summary[summary["mode"] == mode].sort_values("gamma")
        band = 2 * g["mean_w_sd"] / np.sqrt(len(SEEDS))
        ax.plot(g["gamma"], g["mean_w_mean"], color="#44433e", **mode_style[mode],
                label=f"E[w] — {mode}")
        ax.fill_between(g["gamma"], g["mean_w_mean"] - band, g["mean_w_mean"] + band,
                        color="#44433e", alpha=0.12, linewidth=0)
    ax.axhspan(0.90, 1.10, color="#e6e5df", alpha=0.5, zorder=0)
    ax.annotate("no-fire zone (pre-registered tol ±0.10)", xy=(0.05, 1.083), fontsize=7.5,
                color="#6f6e66")
    total = len(SEEDS) * len(GAMMAS)
    fired = {m: int(round(float(summary[summary["mode"] == m]["cal_arm_fire_rate"]
                                .mean()) * total)) for m in MODES}
    ax.set_xlabel("confounding strength γ")
    ax.set_ylabel("mean importance weight E[w]")
    ax.set_ylim(0.85, 1.15)
    ax.set_title(f"A. Frontstage — the battery sees nothing\n(calibration-arm fires: "
                 f"calibrated {fired['calibrated']}/{total} · as-recorded "
                 f"{fired['as_recorded']}/{total})", fontsize=10)
    ax.legend(fontsize=8)

    for mode, alpha in (("calibrated", 1.0), ("as_recorded", 0.45)):
        g = summary[summary["mode"] == mode].sort_values("gamma")
        bx.plot(g["gamma"], (g["bias_snips"]).abs(), color=ESTIMATOR_COLORS["snips"],
                alpha=alpha, **mode_style[mode], label=f"|bias| SNIPS — {mode}")
        bx.plot(g["gamma"], (g["bias_ips"]).abs(), color=ESTIMATOR_COLORS["ips"],
                alpha=alpha, **mode_style[mode], label=f"|bias| IPS — {mode}")
    bx.set_xlabel("confounding strength γ")
    bx.set_ylabel("|mean estimate − V(π_e)|  (backstage only)")
    bx.set_title("B. Backstage — bias grows anyway\n(needs v_true — invisible in practice; "
                 "exit = Λ-band, axis 14)", fontsize=10)
    bx.legend(fontsize=7.5)
    fig.suptitle("Axis 18 — Observational equivalence: a confounded world the whole "
                 "GT-free battery cannot distinguish\n(recorded pscore = true marginal ⇒ "
                 "every log statistic is blind by construction, not by weakness)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── PATTERN ──────────────────────────────────────────────────────────────────
    cal = summary[summary["mode"] == "calibrated"].sort_values("gamma")
    b0 = abs(float(cal["bias_snips"].iloc[0]))
    b25 = abs(float(cal["bias_snips"].iloc[-1]))
    se25 = float(cal["se_snips"].iloc[-1])
    patterns = {
        "calibrated cal-arm fire ≤ 0.05 (전 γ)":
            float(cal["cal_arm_fire_rate"].max()) <= 0.05,
        "|bias snips| 성장: γ=2.5 > γ=0 + 3·SE": b25 > b0 + 3 * se25,
    }
    print(f"[18] → {dec_path.name}, {rev_path.name}, {sum_path.name}, {fig_path.name} "
          f"({time.perf_counter() - t0:.0f}s)")
    for k, ok in patterns.items():
        print(f"[18] PATTERN {'PASS' if ok else 'FAIL'}: {k}")
    print("[18] 실측: as-recorded cal-arm fire (γ별) = "
          f"{summary[summary['mode'] == 'as_recorded'].sort_values('gamma')['cal_arm_fire_rate'].tolist()}")
    print(f"[18] bias(snips, calibrated) γ=0→2.5: "
          f"{[round(float(v), 5) for v in cal['bias_snips']]} · Λ*_flip 중앙값: "
          f"{[round(float(v), 3) for v in cal['lam_star_med']]}")


if __name__ == "__main__":
    main()
