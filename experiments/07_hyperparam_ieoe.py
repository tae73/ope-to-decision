"""축 07 — hyperparameter 민감도(IEOE error-CDF) + SLOPE 회복.

노브: estimator 하이퍼 층(DGP 필드 아님). **가혹 config** BASE_M2._replace(beta_log=8.0)
— committed CSV seed-ensemble 실측: mean max_w≈61(seed 별 41–108)·mean ESS ratio≈0.28.
(base config 는 max_w≈2.8 이라 draw 범위 대부분이 no-op — 가혹 config 필수. 계획 단계의
참고치 max_w≈92 는 특정 단일 seed 측정값이었음 — 정본은 CSV ensemble.) S=30.
q̂ = 0.9·q + 0.05 (축 01 과 동일 shrink).

seed 마다 raw w 에서 med=median(w)·mx=max(w)를 잡고 family(clipped_ips·switch_dr·dros)별:
- random 60 draws: τ, λ_clip ~ log-U[med, 2·mx]; **λ_dros ~ log-U[med², (2·mx)²]** (DRos 의
  λ 는 w² 스케일 — Plan 검증 교정, 범주 혼동 금지). variant="random".
- fixed 기준선: ips·snips·dr (hyperparam 없음) 1회. variant="fixed".
- SLOPE: 고정 기하 ladder np.geomspace(med, 2·mx, 8) (dros 는 제곱 스케일) 위 slope_select.
  variant="slope".

기대 패턴(IEOE, arXiv:2108.13703): 미튜닝 random CDF 는 snips 보다 열위(오른쪽 처짐),
SLOPE 가 격차 대부분 회복. per-sample SE 의 heavy-tail 과소평가로 SLOPE 선택이 공격적일 수
있음(λ/τ ≥ max w → 사실상 무변형) — 관찰되면 그대로 보고. 불발 시 그대로 보고(은폐 금지).

산출: results/tables/07_hyperparam_ieoe.csv (long, _common.COLUMNS)
     + results/figures/07_hyperparam_ieoe.png (family 별 3-panel error-CDF).
"""

import numpy as np
import pandas as pd

from _common import BASE_M2, RunRecord, cached_v_true, write_axis_csv
from _style import (CONTEXT_GRAY, ESTIMATOR_COLORS, ESTIMATOR_LABELS, apply_style,
                    save_figure)
from ope.dgp import make_synthetic_bandit_data
from ope.diagnostics import compute_diagnostics
from ope.estimators import (estimate_clipped_ips, estimate_dr, estimate_dros,
                            estimate_ips, estimate_snips, estimate_switch_dr,
                            slope_select)

AXIS_ID, SLUG = "07", "hyperparam_ieoe"
KNOB = "hyperparam_draw"
SEEDS = range(1000, 1030)  # S=30
N_DRAWS = 60               # random draws / family / seed → 3×60×30 = 5400 rows
LADDER_SIZE = 8
ALPHA_QHAT = 0.9
BETA_LOG_HARSH = 8.0       # 가혹 config — base(β_log=1)에선 draw 대부분이 무효(no-op) 구간
FAMILIES = ("clipped_ips", "switch_dr", "dros")
FIXED_BASELINES = ("ips", "snips", "dr")


def _hyper_range(family: str, med: float, mx: float) -> tuple[float, float]:
    """random draw·ladder 공용 범위 — dros 만 w² 스케일(λ_dros ~ [med², (2mx)²])."""
    if family == "dros":
        return med**2, (2.0 * mx) ** 2
    return med, 2.0 * mx


def _estimate_family(family: str, d, ps, q_hat, h: float):
    args = (d.reward, d.action, ps, d.pi_e_dist)
    if family == "clipped_ips":
        return estimate_clipped_ips(*args, lam=h)
    if family == "switch_dr":
        return estimate_switch_dr(*args, q_hat, tau=h)
    return estimate_dros(*args, q_hat, lam=h)


def _ess_used(w_used: np.ndarray, n: int) -> float:
    if not w_used.size:
        return float("nan")
    q = float((w_used**2).sum())
    return float(w_used.sum() ** 2 / q / n) if q > 0 else 0.0


def _transform_weights(family: str, w: np.ndarray, h: float) -> np.ndarray:
    """estimator 정의와 동일한 weight 변형 (SLOPE 선택 row 의 ess_ratio_used 산출용)."""
    if family == "clipped_ips":
        return np.minimum(w, h)
    if family == "switch_dr":
        return w * (w <= h)
    return h * w / (w**2 + h)


def _ecdf_on_grid(errors: np.ndarray, grid: np.ndarray) -> np.ndarray:
    return (errors[None, :] <= grid[:, None]).mean(axis=1)


def main() -> None:
    cfg_base = BASE_M2._replace(beta_log=BETA_LOG_HARSH)
    v_true = cached_v_true(cfg_base)
    records: list[RunRecord] = []
    slope_meta = {f: {"index": [], "at_or_above_max_w": 0} for f in FAMILIES}

    for seed in SEEDS:
        d = make_synthetic_bandit_data(cfg_base._replace(seed=seed))
        q_hat = ALPHA_QHAT * d.q_true + (1.0 - ALPHA_QHAT) / 2.0
        ps = d.pscore_logged
        n = len(d.action)
        ips_res = estimate_ips(d.reward, d.action, ps, d.pi_e_dist)
        w = ips_res.weights
        med, mx = float(np.median(w)), float(np.max(w))
        diag = compute_diagnostics(ps, d.action, d.pi_e_dist)
        rng = np.random.default_rng(70_000 + seed)  # 하이퍼 draw 전용 rng (데이터 seed 와 분리)

        def rec(estimator: str, variant: str, knob_value: float, hyper: float,
                value: float, w_used: np.ndarray) -> RunRecord:
            return RunRecord(
                axis_id=AXIS_ID, knob_name=KNOB, knob_value=knob_value, seed=seed,
                estimator=estimator, variant=variant, hyperparam_value=hyper,
                estimate=value, v_true=v_true, ess_ratio_raw=diag.ess_ratio,
                max_weight_raw=diag.max_weight, support_proxy=diag.support_deficiency,
                ess_ratio_used=_ess_used(w_used, n))

        # ── random draws (미튜닝 분포 — IEOE 프로토콜) ────────────────────────
        for family in FAMILIES:
            lo, hi = _hyper_range(family, med, mx)
            draws = np.exp(rng.uniform(np.log(lo), np.log(hi), size=N_DRAWS))
            for h in draws:
                res = _estimate_family(family, d, ps, q_hat, float(h))
                records.append(rec(family, "random", float(h), float(h),
                                   res.value, res.weights))

        # ── fixed 기준선 (hyperparam 없는 단순 estimator) ─────────────────────
        snips_res = estimate_snips(d.reward, d.action, ps, d.pi_e_dist)
        dr_res = estimate_dr(d.reward, d.action, ps, d.pi_e_dist, q_hat)
        for name, res in (("ips", ips_res), ("snips", snips_res), ("dr", dr_res)):
            records.append(rec(name, "fixed", float("nan"), float("nan"),
                               res.value, res.weights))

        # ── SLOPE 선택 (고정 기하 ladder) ────────────────────────────────────
        for family in FAMILIES:
            lo, hi = _hyper_range(family, med, mx)
            ladder = np.geomspace(lo, hi, LADDER_SIZE)
            sr = slope_select(d.reward, d.action, ps, d.pi_e_dist, q_hat,
                              family, ladder)
            w_used = _transform_weights(family, w, sr.hyperparam)
            records.append(rec(family, "slope", sr.hyperparam, sr.hyperparam,
                               sr.value, w_used))
            slope_meta[family]["index"].append(sr.index)
            no_op_thr = mx**2 if family == "dros" else mx
            slope_meta[family]["at_or_above_max_w"] += int(sr.hyperparam >= no_op_thr)

    csv_path = write_axis_csv(AXIS_ID, SLUG, records)

    # ── figure: family 별 3-panel error-CDF (figure↔CSV 1:1 — CSV 재독) ───────
    df = pd.read_csv(csv_path)
    df["rel_err"] = (df["estimate"] - df["v_true"]).abs() / df["v_true"]
    pos = df.loc[df["rel_err"] > 0, "rel_err"]
    # 좌측 절단: 하위 1% 분위 — 극소 오차 소수가 로그축을 수 decade 늘리는 것 방지 (CDF 시작 y≈0.01)
    grid = np.geomspace(pos.quantile(0.01) * 0.8, df["rel_err"].max() * 1.1, 400)
    S = len(list(SEEDS))

    def errs(estimator: str, variant: str) -> pd.DataFrame:
        m = (df["estimator"] == estimator) & (df["variant"] == variant)
        return df.loc[m, ["seed", "rel_err"]]

    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 4.5), sharey=True, sharex=True)
    knob_tag = {"clipped_ips": r"$\lambda$", "switch_dr": r"$\tau$",
                "dros": r"$\lambda$ ($w^2$ scale)"}
    title_tag = {"clipped_ips": r"$\lambda$", "switch_dr": r"$\tau$",
                 "dros": r"$\lambda$, $w^2$ scale"}
    for ax, family in zip(axes, FAMILIES):
        color = ESTIMATOR_COLORS[family]
        # random: seed 별 60-point CDF 의 ensemble mean ± 2SE (S=30)
        er = errs(family, "random")
        per_seed = np.stack([_ecdf_on_grid(g["rel_err"].to_numpy(), grid)
                             for _, g in er.groupby("seed")])
        f_rand = per_seed.mean(axis=0)
        se_rand = per_seed.std(axis=0, ddof=1) / np.sqrt(S)
        ax.plot(grid, f_rand, color=color, lw=2.2, zorder=4,
                label=f"{ESTIMATOR_LABELS[family]} random {knob_tag[family]}")
        ax.fill_between(grid, f_rand - 2 * se_rand, f_rand + 2 * se_rand,
                        color=color, alpha=0.15, lw=0)
        # slope: 30-point CDF — 지시함수 seed-ensemble ±2SE = √(F(1−F)/S)
        e_sl = errs(family, "slope")["rel_err"].to_numpy()
        f_sl = _ecdf_on_grid(e_sl, grid)
        se_sl = np.sqrt(f_sl * (1.0 - f_sl) / S)
        ax.plot(grid, f_sl, color=color, lw=2.0, ls="--", zorder=5,
                drawstyle="steps-post", label="SLOPE-selected")
        ax.fill_between(grid, f_sl - 2 * se_sl, f_sl + 2 * se_sl, step="post",
                        color=color, alpha=0.12, lw=0)
        # snips 강조 기준선 + ips·dr context
        e_sn = errs("snips", "fixed")["rel_err"].to_numpy()
        f_sn = _ecdf_on_grid(e_sn, grid)
        se_sn = np.sqrt(f_sn * (1.0 - f_sn) / S)
        ax.plot(grid, f_sn, color=ESTIMATOR_COLORS["snips"], lw=1.6, zorder=3,
                drawstyle="steps-post", label="SNIPS (fixed)")
        ax.fill_between(grid, f_sn - 2 * se_sn, f_sn + 2 * se_sn, step="post",
                        color=ESTIMATOR_COLORS["snips"], alpha=0.12, lw=0)
        for base, dy, ls in (("ips", -10, ":"), ("dr", 4, "-")):
            e_b = errs(base, "fixed")["rel_err"].to_numpy()
            ax.plot(grid, _ecdf_on_grid(e_b, grid), color=CONTEXT_GRAY, lw=1.2,
                    zorder=2, ls=ls, drawstyle="steps-post",
                    label=f"{ESTIMATOR_LABELS[base]} (fixed)")
            ax.annotate(ESTIMATOR_LABELS[base], xy=(e_b.max(), 1.0),
                        xytext=(4, dy), textcoords="offset points",
                        va="center", fontsize=7.5, color="#6f6e66")
        ax.set_xscale("log")
        ax.set_xlim(grid[0], grid[-1] * 2.2)  # 우측 여유 — 끝단 직접 라벨 공간
        ax.set_ylim(-0.02, 1.06)
        ax.set_title(f"{ESTIMATOR_LABELS[family]} family ({title_tag[family]})")
        ax.set_xlabel(r"Relative absolute error $|\hat V - V|/V$ (log scale)")
        ax.legend(loc="upper left", fontsize=7.5)  # CDF 좌상단은 항상 빈 공간
    axes[0].set_ylabel(r"Empirical CDF  $P(\mathrm{error} \leq x)$")
    ess_mean = df["ess_ratio_raw"].mean()
    mxw_mean = df["max_weight_raw"].mean()
    fig.suptitle(
        "Axis 07 — Hyperparameter sensitivity (IEOE error-CDF): untuned draws vs SLOPE\n"
        rf"$\beta_{{log}}=8$ (mean max $w\approx${mxw_mean:.0f}, ESS ratio$\approx$"
        rf"{ess_mean:.2f}) · $\hat q=0.9q+0.05$ · resolution differs: random = "
        f"{N_DRAWS}×{S}=1800 pooled draws, fixed/SLOPE = {S} pts (1/seed) · "
        "bands: seed-ensemble ±2SE", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ───────────────
    # IEOE 주장은 CDF(불안정성) 주장이므로 median(p50)과 tail(p90)을 함께 판정한다.
    def q_(est: str, var: str, q: float) -> float:
        return float(errs(est, var)["rel_err"].quantile(q))

    sn50, sn90 = q_("snips", "fixed", 0.5), q_("snips", "fixed", 0.9)
    print(f"[07] v_true={v_true:.6f}  → {csv_path.name}, {fig_path.name}")
    print(f"[07]   snips fixed: p50={sn50:.4f} p90={sn90:.4f}")
    worse50, worse90, recover = {}, {}, {}
    for family in FAMILIES:
        r50, r90 = q_(family, "random", 0.5), q_(family, "random", 0.9)
        s50, s90 = q_(family, "slope", 0.5), q_(family, "slope", 0.9)
        worse50[family], worse90[family] = r50 > sn50, r90 > sn90
        gap = r50 - sn50
        recover[family] = (r50 - s50) / gap if gap > 0 else float("nan")
        print(f"[07]   {family}: random p50={r50:.4f} p90={r90:.4f} | "
              f"slope p50={s50:.4f} p90={s90:.4f} → p50 gap-recovery={recover[family]:.2f}")
    print(f"[07] PATTERN random_worse_than_snips_p50_all={all(worse50.values())} "
          f"(per-family {worse50})")
    print(f"[07] PATTERN random_worse_than_snips_p90_all={all(worse90.values())} "
          f"(per-family {worse90})")
    print(f"[07] PATTERN slope_recovers_majority_all="
          f"{all(r >= 0.5 for r in recover.values() if r == r)}"
          f" (nan=premise failed: {recover})")
    for family in FAMILIES:
        idx = np.array(slope_meta[family]["index"])
        frac_noop = slope_meta[family]["at_or_above_max_w"] / S
        print(f"[07] OBSERVE slope_{family}: mean ladder index={idx.mean():.2f}/"
              f"{LADDER_SIZE - 1}, share top-2 rung={np.mean(idx >= LADDER_SIZE - 2):.2f}, "
              f"share hyperparam ≥ max w (no-op zone)={frac_noop:.2f}")


if __name__ == "__main__":
    main()
