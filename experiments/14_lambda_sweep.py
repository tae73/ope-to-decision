"""축 14 [스트레치] — MSM Λ-sweep + breakdown Λ*: 순위 단정이 무너지는 감도 수준.

probe M5-14 GO 후 착수(`results/tables/probe_lambda_msm.json`). MSM 정규화(SNIPS형) bound 는
**기존 published 방법**(Kallus & Zhou 2018, arXiv:1805.08593)이다 — 본 레포는 도구 시연이며
제안이 아니다. confounding 하 식별 본류는 연구 트랙 소관(범위 밖, CLAUDE.md §1).

설계: 축 09 DGP 재사용 — cfg = BASE_M2._replace(n=30_000, confounding_strength=γ),
γ ∈ {0.5, 1.5} 두 패널. 각 γ 에서 정책 후보 2개(β_eval=3 vs 5 — v_true 로 참 순위 확인;
β_eval 은 로그 rng 미소비 → 같은 seed=같은 로그, 축 10 트릭)의 SNIPS 점추정과
Λ grid(1→8 기하 25점)의 [lo, hi] 부채꼴을 같은 로그에서 계산. S=20 seed 전부 CSV,
figure 는 대표 seed 1000 (+ 전 seed Λ* rug).

**breakdown Λ*** = 두 후보의 bound 가 겹치기 시작하는(순위 단정 불능) 최소 Λ.
[lo,hi] 는 Λ 에 단조 확장(tests/test_msm.py)이므로 겹침 지표도 Λ 에 단조 → [1, 8]
log-bisection 으로 grid 해상도와 무관하게 산출, seed 별 breakdown CSV 기록.
Λ=8 에서도 안 겹치면 censored(lam_star=NaN).

참 Λ 스케일 참조: DGP 가 아는 per-sample 진짜 배율 왜곡
max(w_true/w_logged, w_logged/w_true) = max(ρ, 1/ρ), ρ = pscore_true/pscore_logged 의
분위 p50/p90/p99/max 를 breakdown CSV 에 기록한다(figure 에는 대표 seed 값 주석).
각주: probe M5-14 실측 max 왜곡은 125.55–562.43 (γ=1.5, seeds 500–504,
`results/tables/probe_lambda_msm.json`) — **극단 tail** 로, p99 스케일과 자릿수가 다르다.

산출 (자체 스키마 — _common.COLUMNS 아님, 축 10 선례 따라 docstring 문서화):
- `results/tables/14_lambda_sweep.csv` (long): seed·gamma·candidate(=β_eval)·lam·lo·hi.
  lam=1.0 행의 lo=hi 가 해당 후보의 SNIPS 점추정이다.
- `results/tables/14_lambda_sweep_breakdown.csv` (companion): seed·gamma·lam_star·
  true_rank_gap(= v_true(β5) − v_true(β3), γ·seed 무관 상수 — self-contained 병기)·
  lam_star_censored·snips_gap(= SNIPS(β5) − SNIPS(β3))·snips_rank_correct·
  v_true_beta3·v_true_beta5·true_viol_p50/p90/p99/max.
- `results/figures/14_lambda_sweep.png` (2패널): γ 별 Λ vs [lo,hi] 부채꼴(후보 2)·Λ* 수직선·
  SNIPS 마커·v_true 참조선. 후보는 estimator 가 아니므로 entity 팔레트 밖 중립 2색 + 주석.
"""

import numpy as np
import pandas as pd

from _common import BASE_M2, TAB_DIR, cached_v_true
from _style import ORACLE_COLOR, apply_style, save_figure
from ope.dgp import make_synthetic_bandit_data
from ope.estimators import estimate_snips, msm_snips_bounds
from ope.policies import softmax_policy

AXIS_ID, SLUG = "14", "lambda_sweep"
GAMMAS = [0.5, 1.5]
CAND_BETAS = [3.0, 5.0]            # 정책 후보 (β_eval) — estimator 아님
N_FIXED = 30_000
SEEDS = range(1000, 1020)          # S=20
LAM_GRID = np.geomspace(1.0, 8.0, 25)
LAM_MAX = 8.0
REP_SEED = 1000
# 후보 = 정책(estimator 아님) — _style entity 팔레트 밖 중립 2색 (dataviz 규율)
CAND_COLORS = {3.0: "#2f6f6a", 5.0: "#8a5a2f"}
BREAK_COLOR = "#a33b3b"


def _bounds_pair(d, pi_e_by_cand, lam: float) -> dict:
    return {c: msm_snips_bounds(d.reward, d.action, d.pscore_logged, pi_e_by_cand[c], lam)
            for c in CAND_BETAS}


def _overlaps(b: dict) -> bool:
    (lo_a, hi_a), (lo_b, hi_b) = b[CAND_BETAS[0]], b[CAND_BETAS[1]]
    return bool(lo_a <= hi_b and lo_b <= hi_a)


def _lam_star(d, pi_e_by_cand) -> tuple[float, bool]:
    """겹침 시작 최소 Λ — bound 단조 확장 ⇒ 겹침 지표 단조 ⇒ log-scale bisection 유효."""
    if _overlaps(_bounds_pair(d, pi_e_by_cand, 1.0)):
        return 1.0, False
    if not _overlaps(_bounds_pair(d, pi_e_by_cand, LAM_MAX)):
        return float("nan"), True
    lo_l, hi_l = np.log(1.0), np.log(LAM_MAX)
    for _ in range(40):
        mid = 0.5 * (lo_l + hi_l)
        if _overlaps(_bounds_pair(d, pi_e_by_cand, float(np.exp(mid)))):
            hi_l = mid
        else:
            lo_l = mid
    return float(np.exp(hi_l)), False


def main() -> None:
    v_true = {c: cached_v_true(BASE_M2._replace(beta_eval=c)) for c in CAND_BETAS}
    v3, v5 = v_true[3.0], v_true[5.0]
    rows, bd_rows = [], []
    for gamma in GAMMAS:
        cfg = BASE_M2._replace(n=N_FIXED, confounding_strength=gamma)
        for seed in SEEDS:
            d = make_synthetic_bandit_data(cfg._replace(seed=seed))
            pi_e = {c: softmax_policy(d.q_true, c) for c in CAND_BETAS}
            snips = {c: estimate_snips(d.reward, d.action, d.pscore_logged, pi_e[c]).value
                     for c in CAND_BETAS}
            for c in CAND_BETAS:
                for lam in LAM_GRID:
                    lo, hi = msm_snips_bounds(d.reward, d.action, d.pscore_logged,
                                              pi_e[c], float(lam))
                    rows.append({"seed": seed, "gamma": gamma, "candidate": c,
                                 "lam": float(lam), "lo": lo, "hi": hi})
            lam_star, censored = _lam_star(d, pi_e)
            ratio = d.pscore_true / d.pscore_logged
            viol = np.maximum(ratio, 1.0 / ratio)
            bd_rows.append({
                "seed": seed, "gamma": gamma, "lam_star": lam_star,
                "true_rank_gap": v5 - v3, "lam_star_censored": censored,
                "snips_gap": snips[5.0] - snips[3.0],
                "snips_rank_correct": bool(np.sign(snips[5.0] - snips[3.0])
                                           == np.sign(v5 - v3)),
                "v_true_beta3": v3, "v_true_beta5": v5,
                "true_viol_p50": float(np.quantile(viol, 0.50)),
                "true_viol_p90": float(np.quantile(viol, 0.90)),
                "true_viol_p99": float(np.quantile(viol, 0.99)),
                "true_viol_max": float(viol.max()),
            })
    long_df = pd.DataFrame(rows)
    bd_df = pd.DataFrame(bd_rows)
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = TAB_DIR / f"{AXIS_ID}_{SLUG}.csv"
    long_df.to_csv(csv_path, index=False)
    bd_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_breakdown.csv"
    bd_df.to_csv(bd_path, index=False)

    # ── figure: γ 별 Λ-부채꼴 + breakdown Λ* ────────────────────────────────
    apply_style()
    import matplotlib.pyplot as plt
    from matplotlib.ticker import NullFormatter, ScalarFormatter
    from _style import SURFACE
    label_box = dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 5.0), sharey=True)
    for j, gamma in enumerate(GAMMAS):
        ax = axes[j]
        rep = long_df[(long_df["gamma"] == gamma) & (long_df["seed"] == REP_SEED)]
        bd_g = bd_df[bd_df["gamma"] == gamma]
        bd = bd_g[bd_g["seed"] == REP_SEED].iloc[0]
        for c in CAND_BETAS:
            s = rep[rep["candidate"] == c].sort_values("lam")
            col = CAND_COLORS[c]
            ax.fill_between(s["lam"], s["lo"], s["hi"], color=col, alpha=0.20, lw=0)
            ax.plot(s["lam"], s["lo"], color=col, lw=1.6)
            ax.plot(s["lam"], s["hi"], color=col, lw=1.6,
                    label=rf"policy candidate $\beta_{{eval}}$={c:g} — MSM band")
            snips_pt = float(s[s["lam"] == 1.0]["lo"].iloc[0])
            ax.plot([1.0], [snips_pt], marker="D", ms=6, color=col, zorder=5, clip_on=False)
            ax.annotate(f"SNIPS {snips_pt:.3f}", xy=(1.0, snips_pt),
                        xytext=(7, -20 if c == CAND_BETAS[0] else 14),
                        textcoords="offset points", fontsize=7.5, color=col,
                        bbox=label_box, zorder=6)
            ax.axhline(v_true[c], color=ORACLE_COLOR, ls=":", lw=1.0, zorder=1)
            ax.annotate(rf"$V(\pi_{{\beta={c:g}}})$={v_true[c]:.3f}",
                        xy=(LAM_MAX, v_true[c]), xytext=(-4, 5 if c == CAND_BETAS[1] else -12),
                        textcoords="offset points", ha="right", fontsize=7,
                        color=ORACLE_COLOR, bbox=label_box, zorder=6)
        ls_val = float(bd["lam_star"])
        ax.axvline(ls_val, color=BREAK_COLOR, ls="--", lw=1.4)
        ax.annotate(f"breakdown $\\Lambda^*$={ls_val:.3f}\n(bands overlap → ranking\n"
                    "no longer decidable)", xy=(ls_val, 0.86), xycoords=("data", "axes fraction"),
                    xytext=(8, 0), textcoords="offset points", va="top",
                    fontsize=7.5, color=BREAK_COLOR, bbox=label_box, zorder=6)
        # 전 seed Λ* rug (상단) — 분포 감각, 수치는 breakdown CSV
        finite_ls = bd_g["lam_star"].dropna()
        ax.plot(finite_ls, np.full(len(finite_ls), 0.985), transform=ax.get_xaxis_transform(),
                marker="|", ms=9, lw=0, color=BREAK_COLOR, alpha=0.55, clip_on=False)
        ax.annotate(rf"per-seed $\Lambda^*$ (S={len(bd_g)})",
                    xy=(float(finite_ls.max()), 0.975), xycoords=("data", "axes fraction"),
                    xytext=(8, 0), textcoords="offset points", va="center",
                    fontsize=6.8, color=BREAK_COLOR, alpha=0.9)
        ax.annotate(rf"true distortion this seed (oracle): p99={bd['true_viol_p99']:.1f}, "
                    rf"max={bd['true_viol_max']:.0f} $\gg\Lambda^*$",
                    xy=(0.02, 0.03), xycoords="axes fraction", fontsize=7, color="#6f6e66")
        ax.set_xscale("log")
        ax.set_xlim(1.0, LAM_MAX)
        ax.set_xticks([1, 2, 4, 8])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(r"sensitivity level $\Lambda$ (log scale)")
        ax.set_title(rf"$\gamma$ = {gamma:g}  (confounding strength)", loc="left")
        if j == 0:
            ax.set_ylabel("policy value — normalized (SNIPS-type) MSM bound")
            ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.10), fontsize=8)
    fig.suptitle(
        "Axis 14 [stretch] — MSM $\\Lambda$-sweep: worst-case bands per candidate policy and the "
        "breakdown $\\Lambda^*$ where ranking becomes undecidable\n"
        "MSM normalized bounds are an existing published method (Kallus & Zhou 2018, "
        "arXiv:1805.08593) — tool demonstration only, NOT a proposal of this repo; "
        "identification under confounding is out of scope (research track)\n"
        f"n={N_FIXED:,} · representative seed {REP_SEED} (all S={len(list(SEEDS))} seeds in CSV; "
        "top rug: per-seed $\\Lambda^*$) · candidates are policies, not estimators → neutral "
        "colors · true distortion max is an extreme tail (see breakdown CSV p50–p99 vs max)",
        fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ─────────────
    print(f"[14] v_true: beta3={v3:.6f} beta5={v5:.6f} "
          f"true_better={'beta5' if v5 > v3 else 'beta3'} gap={v5 - v3:+.6f}")
    print(f"[14] → {csv_path.name}, {bd_path.name}, {fig_path.name}")
    mono_ok = True
    for _, grp in long_df.groupby(["gamma", "seed", "candidate"]):
        g = grp.sort_values("lam")
        if not (np.all(np.diff(g["lo"]) <= 1e-12) and np.all(np.diff(g["hi"]) >= -1e-12)):
            mono_ok = False
    for gamma in GAMMAS:
        b = bd_df[bd_df["gamma"] == gamma]
        ls = b["lam_star"]
        print(f"[14] PATTERN γ={gamma:g}: lam_star min={ls.min():.4f} "
              f"median={ls.median():.4f} max={ls.max():.4f} "
              f"censored={int(b['lam_star_censored'].sum())}/{len(b)} "
              f"snips_rank_correct={int(b['snips_rank_correct'].sum())}/{len(b)} "
              f"true_viol_p99=[{b['true_viol_p99'].min():.2f}, {b['true_viol_p99'].max():.2f}] "
              f"viol_max=[{b['true_viol_max'].min():.0f}, {b['true_viol_max'].max():.0f}]")
    at8 = long_df[long_df["lam"] == LAM_MAX]
    cov = {(g, c): float((((s := at8[(at8["gamma"] == g) & (at8["candidate"] == c)])["lo"]
                          <= v_true[c]) & (v_true[c] <= s["hi"])).mean())
           for g in GAMMAS for c in CAND_BETAS}
    print(f"[14] PATTERN monotone_bands={mono_ok}  coverage@Λ=8(v_true in [lo,hi]): "
          + "  ".join(f"γ={g:g}·β{c:g}={cov[(g, c)]:.2f}" for g in GAMMAS for c in CAND_BETAS))
    # grid-consistency: bisection Λ* 이 long CSV grid 겹침 패턴과 정합해야 함
    grid_ok = True
    for _, r in bd_df.iterrows():
        sub = long_df[(long_df["gamma"] == r["gamma"]) & (long_df["seed"] == r["seed"])]
        piv = sub.pivot_table(index="lam", columns="candidate", values=["lo", "hi"])
        ov = ((piv[("lo", CAND_BETAS[0])] <= piv[("hi", CAND_BETAS[1])])
              & (piv[("lo", CAND_BETAS[1])] <= piv[("hi", CAND_BETAS[0])]))
        for lam, o in ov.items():
            if np.isnan(r["lam_star"]):
                if o:
                    grid_ok = False
            elif (lam < r["lam_star"] - 1e-9 and o) or (lam > r["lam_star"] + 1e-9 and not o):
                grid_ok = False
    print(f"[14] PATTERN lam_star_grid_consistent={grid_ok}")


if __name__ == "__main__":
    main()
