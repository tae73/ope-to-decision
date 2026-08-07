"""축 16 — 다중 지표 비즈니스 게이트: guardrail 게이트(비교형 vs 절대형) + 광고주 재분배.

설계(플랜 M6 §3): 후보 정책은 **CTR-relevance 만 보고 price 를 무시**한다. 시나리오 2종 × S=40:
- s1_relevance : 구조 그대로 — 후보 = softmax(β_eval·s), β_eval=6 (incumbent = 로깅 β_log=2).
- s2_price_tilt: 후보 스코어를 s − c·price_rank 로 틀어 **저가 액션 쏠림**(CTR↑·REV↓ 유도).
  price_rank = 액션 가격 오름차순 rank/(K−1) ∈ [0,1]. c=0.5 는 **시연값** — 사전 탐색(결과 조정
  아님)에서 c∈{0.2..1.0} 전 구간 ΔCTR>0 유지·ΔREV<0 이었고, c=0.5 가 REV 참값 하락(−0.0039,
  incumbent 대비 −48%)과 HHI 상승(0.288→0.374)을 함께 주는 중간값이라 채택.

게이트(둘 다 지표 벡터에 동일 진리 기준 — T 를 incumbent 참값에 정렬해 진리 판정이 형식 간 동일):
- 비교형: deploy iff Δ̂CTR>0 ∧ Δ̂REV≥−g ∧ HHI(π_e)≤h. Δ̂_m = V̂_m^IPS(π_e) − r̄_m(같은 로그
  on-policy 평균 — 축 10 "_incumbent_mean_r" 관례·Δ-OPE 상쇄 원리의 벡터 확장).
- 절대형: deploy iff V̂_CTR>T_ctr ∧ V̂_REV≥T_rev ∧ HHI≤h, T_ctr=V_ctr(π_0)·T_rev=V_rev(π_0)−g.
- g=0.05·V_rev(π_0), h=0.35, c=0.5 전부 **시연값(무교정 — "제안" 프레임, 축 08 급 교정 없음)**.
- HHI(π_e)≤h arm 은 **결정적**(π_e 기지 — 정확 계산, OPE 아님): 추정 오류 0. s2 는 이 arm 이
  REV arm 과 독립적으로 배포를 막는다(h=0.35 < HHI(s2)=0.374 — 시연값 선정 이유).

지표 간 오차 상관(핵심 전시): CTR·CVR·REV 의 Δ̂ 가 **같은 w_i 를 공유** → 극단 weight seed 가
3개 Δ̂ 를 동시 오염 → 결합 게이트 오류는 seed 군집으로 발생(주변 오류율 곱이 아님). per-metric
+ 결합 false-go/stop 을 seed 단위 기록, 결합 오류율을 독립 가정 곱과 대비.

발견(정직 기록 — 전 수치 committed gates/advertiser CSV 에서 재도출 가능):
① Δ̂ 수준 동시 오염 성립: comparative-IPS Pearson corr — S1 ctr-cvr 0.401·ctr-rev 0.329·
  cvr-rev 0.537 / S2 0.345·0.220·0.738. 세 Δ̂ 편차의 **부호 동시 일치율 0.525(S1)·0.425(S2)
  vs 독립 기준 0.25** (~2.1·1.7배) — 같은 w_i 에 더해 공유 클릭·전환 이벤트가 원인(cvr-rev
  가 최고 상관인 이유: REV=conv·price 로 전환 이벤트까지 공유). 단 이 regime 의 weight tail
  은 온순(max_w≈2.9–4.5·ESS 0.45–0.70)해서 "극단 weight 단독 지배" 그림은 아니다.
② 비교형 상쇄의 벡터 확장 — 성립하되 **조건부**: S1·IPS 기준 Δ̂CTR sd 0.0012 vs 절대형
  V̂CTR sd 0.0028, Δ̂REV 0.00058 vs 0.00171 (절반 이하). 게이트 오류로는 경계 계쟁인 S1 REV
  arm 에서만 발현: 절대형 false-stop 0.225(IPS)·0.250(SNIPS) vs 비교형 0.000. **반례(S2 REV)**:
  comp sd 0.00098 > abs sd 0.00061 — 저가 쏠림 후보의 매출은 incumbent 매출(고가 액션 price
  tail 지배)과 오차 공유가 약해, baseline 을 빼는 것이 분산을 오히려 더한다. 상쇄는 두 정책의
  오차 공분산이 충분히 클 때만 이득(축 10 원리의 경계 조건 — 정책 괴리가 지표 구성까지 바꾸면
  약화).
③ **기대 불발(정직 보고)**: "결합 게이트 오류의 군집(독립곱 대비 초과)"은 게이트 수준에서
  관측 불가 — CTR arm 참값 마진 +0.0089 가 Δ̂CTR sd 의 ~7배(비교형)·~3배(절대형)여서 40 seed
  에서 CTR arm 오류 0, 결합 오류가 REV 단일 arm 에 국한(관측 결합 오류 = 독립 예측과 자명
  일치). probe-GO 파라미터가 "ΔCTR 이 n=10k 에서 판별 가능"을 보장하도록 선정된 대가다 —
  군집은 복수 arm 이 동시에 경계 계쟁일 때만 게이트 수준으로 올라온다(①의 Δ̂ 동시 오염이
  그 성분; 전시는 figure 패널 B·stdout PATTERN).
④ 결정적 HHI arm: 오류 0(전 seed·양 형식) — S2 는 REV arm 판정과 무관하게 HHI arm
  (0.374 > h=0.35)이 배포를 막는다. 단 h=0.40 이면 이 방어선은 사라진다(시연값 의존 명시).
⑤ 광고주 재분배(정확 계산): 최고가 액션 보유 adv 3 노출 0.301→0.252(S1)→0.100(S2),
  저가 보유 adv 0 은 0.308→0.321→0.507, HHI 0.288→0.301→0.374. subgroup 매출 IPS 는
  not-estimable 0%(최소 가중 클릭 이벤트 22.1 > 임계 10) — 희소 미달이 없는 regime 임을
  정직 기록(임계가 걸리는 희소 regime 은 본 축 스코프 밖).

한계 명시: (i) `hyperparams_from_weights` 는 reward 척도를 무시한다 — REV 의 유효 tail 은
w_i·price 인데 clipping τ·λ 는 w 분위수로만 잡혀 price tail 을 못 잡는다(본 축 게이트는
IPS/SNIPS 라 미사용이지만, clipped 계열을 REV 에 쓸 때의 함정으로 명시). (ii) γ=0 전제 —
confounding 이 있으면 상쇄 논리 자체가 깨진다(축 09 소관). (iii) 리텐션 등 세션 간 지표는
이 층에서 식별 불가(RL OPE 소관 — business.py·PLAYBOOK 경계 선언).

산출: results/tables/16_business_gate.csv (_common.COLUMNS — 스키마 예외: variant=지표명,
incumbent 행은 estimator="_incumbent_mean_r"(축 10 관례);
_incumbent_mean_r 행의 진단 3컬럼(ess_ratio_raw 등)은 같은 로그·후보 π_e 조합의 진단을
편의 기입(on-policy 진단 아님 — w≡1)) +
16_business_gate_gates.csv (seed 단위 게이트 판정 — 자체 스키마) +
16_business_gate_advertiser.csv (정확 노출·HHI·subgroup OPE 요약 events_min/mean/max —
figure 패널 D whisker 재도출 가능 — 자체 스키마) +
results/figures/16_business_gate.png (멀티패널 1장).

비고: `_structure`·`_rates` private 헬퍼를 재사용한다 — tilted 정책의 GT/노출을
`funnel_ground_truth` 와 **같은 MC rng 관례**(1_000_003+struct_seed·200k)로 계산해 x-표집
노이즈가 Δ참값에서 상쇄되게 하기 위함(조립은 experiments 소관 — CLAUDE.md §3).
"""

import numpy as np
import pandas as pd

from _common import RunRecord, write_axis_csv
from _style import apply_style, save_figure
from ope.business import (
    FunnelConfig, METRICS, MIN_SUBGROUP_EVENTS, _rates, _structure,
    funnel_ground_truth, hhi, make_funnel_bandit_data, subgroup_revenue_ips,
)
from ope.diagnostics import compute_diagnostics
from ope.estimators import estimate_ips, estimate_snips
from ope.policies import softmax_policy

AXIS_ID, SLUG = "16", "business_gate"
CFG = FunnelConfig(n=10_000, n_actions=10, dim_context=5, beta_log=2.0, beta_eval=6.0)
SEEDS = range(1000, 1040)  # S=40
SCENARIOS = ("s1_relevance", "s2_price_tilt")
PRICE_TILT = 0.5   # c — 시연값 (docstring 선정 근거)
G_REL = 0.05       # g = G_REL·V_rev(π_0) — 허용 매출 하락, 시연값
H_HHI = 0.35       # h — HHI 상한, 시연값
GATE_ESTIMATORS = ("ips", "snips")  # figure 는 ips, snips 는 CSV·stdout (docstring)

# figure 로컬 색 (estimator entity 색과 무관한 축-국소 entity — 게이트 형식·정책)
FORM_COLORS = {"comparative": "#2a78d6", "absolute": "#eb6834"}
POLICY_COLORS = {"incumbent_b2": "#6f6e66", "s1_relevance": "#008300",
                 "s2_price_tilt": "#4a3aa7"}


def _mc_block():
    """funnel_ground_truth 와 동일한 rng 관례의 200k MC 컨텍스트 블록 (모든 정책 공유)."""
    th_c, b_c, th_v, b_v, price, adv = _structure(CFG)
    rng = np.random.default_rng(1_000_003 + CFG.struct_seed)
    x = rng.normal(size=(200_000, CFG.dim_context))
    s, p_c, p_v = _rates(x, th_c, b_c, th_v, b_v, CFG)
    return s, p_c, p_v, price, adv


def _gt_shares(pi, p_c, p_v, price, adv):
    """정책 분포 pi (n_mc, K) → 지표 3종 GT + 액션/광고주 노출 점유율(정확 계산 — OPE 아님)."""
    gt = {"ctr": float((pi * p_c).sum(1).mean()),
          "cvr": float((pi * p_c * p_v).sum(1).mean()),
          "rev": float((pi * p_c * p_v * price).sum(1).mean())}
    pi_bar = pi.mean(0)
    shares = np.zeros(CFG.n_advertisers)
    np.add.at(shares, adv, pi_bar)
    return gt, pi_bar, shares


def _candidate_scores(s, price_rank, scenario):
    if scenario == "s1_relevance":
        return s
    return s - PRICE_TILT * price_rank


def main() -> None:
    # ── 참값·노출 층 (결정적 — seed 루프 밖) ────────────────────────────────
    s_mc, p_c_mc, p_v_mc, price, adv = _mc_block()
    k = CFG.n_actions
    price_rank = price.argsort().argsort() / (k - 1)  # 0=최저가 .. 1=최고가

    pol_ids = ("incumbent_b2",) + SCENARIOS
    gt, pi_bar, shares, hhi_of = {}, {}, {}, {}
    gt["incumbent_b2"], pi_bar["incumbent_b2"], shares["incumbent_b2"] = _gt_shares(
        softmax_policy(s_mc, CFG.beta_log), p_c_mc, p_v_mc, price, adv)
    for sc in SCENARIOS:
        pi = softmax_policy(_candidate_scores(s_mc, price_rank, sc), CFG.beta_eval)
        gt[sc], pi_bar[sc], shares[sc] = _gt_shares(pi, p_c_mc, p_v_mc, price, adv)
    for p in pol_ids:
        hhi_of[p] = hhi(shares[p])
    # 모듈 GT 와 교차검증 (같은 rng 관례 — 정확 일치해야 함)
    assert np.allclose(list(gt["incumbent_b2"].values()),
                       list(funnel_ground_truth(CFG, beta=CFG.beta_log).values()))
    assert np.allclose(list(gt["s1_relevance"].values()),
                       list(funnel_ground_truth(CFG).values()))

    v0 = gt["incumbent_b2"]
    g_abs = G_REL * v0["rev"]
    t_ctr, t_rev = v0["ctr"], v0["rev"] - g_abs  # 절대형 T — incumbent 참값 정렬(진리 동일화)
    true_arm = {sc: {"ctr": gt[sc]["ctr"] - v0["ctr"] > 0.0,
                     "rev": gt[sc]["rev"] - v0["rev"] >= -g_abs,
                     "hhi": hhi_of[sc] <= H_HHI} for sc in SCENARIOS}
    true_deploy = {sc: {"ope": true_arm[sc]["ctr"] and true_arm[sc]["rev"],
                        "full": true_arm[sc]["ctr"] and true_arm[sc]["rev"]
                                and true_arm[sc]["hhi"]} for sc in SCENARIOS}

    # ── seed 루프: 같은 로그 위 지표 벡터 OPE + 게이트 판정 + subgroup ──────
    records, gate_rows, sub_rows = [], [], []
    for sc in SCENARIOS:
        for seed in SEEDS:
            d = make_funnel_bandit_data(CFG._replace(seed=seed))
            th_c, b_c, th_v, b_v, _, _ = _structure(CFG)
            s_log, _, _ = _rates(d.context, th_c, b_c, th_v, b_v, CFG)
            pi_e = softmax_policy(_candidate_scores(s_log, price_rank, sc), CFG.beta_eval)
            diag = compute_diagnostics(d.pscore, d.action, pi_e)

            v_hat, r_bar = {}, {}
            for est_name, fn in (("ips", estimate_ips), ("snips", estimate_snips)):
                for m in METRICS:
                    res = fn(d.rewards[m], d.action, d.pscore, pi_e)
                    v_hat[(est_name, m)] = res.value
                    w = res.weights
                    ess_used = float(w.sum() ** 2 / (w**2).sum() / len(w))
                    records.append(RunRecord(
                        axis_id=AXIS_ID, knob_name="scenario", knob_value=sc, seed=seed,
                        estimator=est_name, variant=m, hyperparam_value=np.nan,
                        estimate=res.value, v_true=gt[sc][m],
                        ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
                        support_proxy=diag.support_deficiency, ess_ratio_used=ess_used))
            for m in METRICS:
                r_bar[m] = float(d.rewards[m].mean())
                records.append(RunRecord(
                    axis_id=AXIS_ID, knob_name="scenario", knob_value=sc, seed=seed,
                    estimator="_incumbent_mean_r", variant=m, hyperparam_value=np.nan,
                    estimate=r_bar[m], v_true=v0[m],
                    ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
                    support_proxy=diag.support_deficiency, ess_ratio_used=np.nan))

            for est_name in GATE_ESTIMATORS:
                for form in ("comparative", "absolute"):
                    if form == "comparative":
                        stat = {m: v_hat[(est_name, m)] - r_bar[m] for m in METRICS}
                        pass_ctr = stat["ctr"] > 0.0
                        pass_rev = stat["rev"] >= -g_abs
                    else:
                        stat = {m: v_hat[(est_name, m)] for m in METRICS}
                        pass_ctr = stat["ctr"] > t_ctr
                        pass_rev = stat["rev"] >= t_rev
                    pass_hhi = hhi_of[sc] <= H_HHI  # 결정적 — seed 무관, 오류 0
                    dep_ope = pass_ctr and pass_rev
                    dep_full = dep_ope and pass_hhi
                    ta = true_arm[sc]
                    gate_rows.append({
                        "scenario": sc, "estimator": est_name, "gate_form": form,
                        "seed": seed, "stat_ctr": stat["ctr"], "stat_cvr": stat["cvr"],
                        "stat_rev": stat["rev"],
                        "pass_ctr": pass_ctr, "pass_rev": pass_rev, "pass_hhi": pass_hhi,
                        "deploy_ope": dep_ope, "deploy_full": dep_full,
                        "true_pass_ctr": ta["ctr"], "true_pass_rev": ta["rev"],
                        "true_pass_hhi": ta["hhi"],
                        "true_deploy_ope": true_deploy[sc]["ope"],
                        "true_deploy_full": true_deploy[sc]["full"],
                        "err_ctr": pass_ctr != ta["ctr"], "err_rev": pass_rev != ta["rev"],
                        "err_hhi": pass_hhi != ta["hhi"],
                        "err_ope": dep_ope != true_deploy[sc]["ope"],
                        "err_full": dep_full != true_deploy[sc]["full"],
                        "false_go_full": dep_full and not true_deploy[sc]["full"],
                        "false_stop_full": (not dep_full) and true_deploy[sc]["full"],
                        "ess_ratio_raw": diag.ess_ratio, "max_weight_raw": diag.max_weight})

            for r in subgroup_revenue_ips(d._replace(pi_e_dist=pi_e), CFG.n_advertisers):
                sub_rows.append({"scenario": sc, "seed": seed, "advertiser": r.advertiser,
                                 "estimable": r.estimable, "estimate": r.estimate,
                                 "weighted_events": r.weighted_events})

    csv_path = write_axis_csv(AXIS_ID, SLUG, records)
    gates = pd.DataFrame(gate_rows)
    gates_path = csv_path.parent / f"{AXIS_ID}_{SLUG}_gates.csv"
    gates.to_csv(gates_path, index=False)

    # ── 광고주 CSV: 액션/광고주/정책 레벨 (정확 계산 + subgroup OPE 요약) ────
    sub = pd.DataFrame(sub_rows)
    adv_rows = []
    for p in pol_ids:
        for a in range(k):
            adv_rows.append({"scenario": p, "level": "action", "id": a,
                             "advertiser": int(adv[a]), "price": price[a],
                             "share": pi_bar[p][a], "hhi": np.nan,
                             "ne_rate": np.nan, "events_min": np.nan,
                             "events_mean": np.nan, "events_max": np.nan,
                             "subgroup_est_mean": np.nan})
        for gadv in range(CFG.n_advertisers):
            row = {"scenario": p, "level": "advertiser", "id": gadv, "advertiser": gadv,
                   "price": np.nan, "share": shares[p][gadv], "hhi": hhi_of[p],
                   "ne_rate": np.nan, "events_min": np.nan, "events_mean": np.nan,
                   "events_max": np.nan, "subgroup_est_mean": np.nan}
            if p in SCENARIOS:  # subgroup OPE 는 후보 π_e 에만 정의
                gsub = sub[(sub["scenario"] == p) & (sub["advertiser"] == gadv)]
                row.update(ne_rate=float((~gsub["estimable"]).mean()),
                           events_min=float(gsub["weighted_events"].min()),
                           events_mean=float(gsub["weighted_events"].mean()),
                           events_max=float(gsub["weighted_events"].max()),
                           subgroup_est_mean=float(gsub.loc[gsub["estimable"],
                                                            "estimate"].mean()))
            adv_rows.append(row)
    adv_df = pd.DataFrame(adv_rows)
    adv_path = csv_path.parent / f"{AXIS_ID}_{SLUG}_advertiser.csv"
    adv_df.to_csv(adv_path, index=False)

    # ── 집계: 오류율(비교 vs 절대, per arm + 결합) + 독립곱 대비 + Δ̂ 상관 ──
    def agg_errors(df):
        rows = []
        for (sc, est_name, form), grp in df.groupby(["scenario", "estimator", "gate_form"]):
            p_ctr, p_rev = grp["pass_ctr"].mean(), grp["pass_rev"].mean()
            dep = p_ctr * p_rev  # 독립 가정 결합 배포율
            indep_err_ope = (dep if not true_deploy[sc]["ope"] else 1.0 - dep)
            rows.append({"scenario": sc, "estimator": est_name, "gate_form": form,
                         "err_ctr": grp["err_ctr"].mean(), "err_rev": grp["err_rev"].mean(),
                         "err_hhi": grp["err_hhi"].mean(), "err_ope": grp["err_ope"].mean(),
                         "err_full": grp["err_full"].mean(),
                         "indep_err_ope": indep_err_ope,
                         "joint_arm_err": (grp["err_ctr"] & grp["err_rev"]).mean(),
                         "marg_arm_err_prod": grp["err_ctr"].mean() * grp["err_rev"].mean()})
        return pd.DataFrame(rows)

    errs = agg_errors(gates)
    comp_ips = gates[(gates["estimator"] == "ips") & (gates["gate_form"] == "comparative")]
    corr = {sc: comp_ips[comp_ips["scenario"] == sc][["stat_ctr", "stat_cvr", "stat_rev"]]
            .corr().to_numpy() for sc in SCENARIOS}
    # 세 Δ̂ 편차의 부호 동시 일치율(동시 오염 통계) — 독립이면 2·(1/2)³=0.25
    conc, sd_comp, sd_abs = {}, {}, {}
    for sc in SCENARIOS:
        gsc = comp_ips[comp_ips["scenario"] == sc]
        dev = np.column_stack([gsc[f"stat_{m}"] - (gt[sc][m] - v0[m]) for m in METRICS])
        conc[sc] = float((np.abs(np.sign(dev).sum(1)) == 3).mean())
        sd_comp[sc] = {m: float(gsc[f"stat_{m}"].std(ddof=1)) for m in METRICS}
        asc = gates[(gates["estimator"] == "ips") & (gates["gate_form"] == "absolute")
                    & (gates["scenario"] == sc)]
        sd_abs[sc] = {m: float(asc[f"stat_{m}"].std(ddof=1)) for m in METRICS}

    # ── figure: 멀티패널 1장 ────────────────────────────────────────────────
    apply_style()
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.8))

    # A — 게이트 오류율 (ips): arm 별 + 결합, 비교형 vs 절대형, 시나리오 병치
    ax = axes[0, 0]
    arm_keys = ["err_ctr", "err_rev", "err_hhi", "err_ope", "err_full"]
    arm_labels = ["CTR arm", "REV arm", "HHI arm\n(deterministic)", "combined\nOPE arms",
                  "combined\nfull gate"]
    e_ips = errs[errs["estimator"] == "ips"].set_index(["scenario", "gate_form"])
    width, offs = 0.19, [-1.5, -0.5, 0.5, 1.5]
    xs = np.arange(len(arm_keys))
    for j, (sc, form) in enumerate([(s, f) for s in SCENARIOS
                                    for f in ("comparative", "absolute")]):
        vals = [e_ips.loc[(sc, form), a] for a in arm_keys]
        ax.bar(xs + offs[j] * width, vals, width * 0.92, color=FORM_COLORS[form],
               alpha=0.45 if sc == "s1_relevance" else 1.0,
               label=f"{'S1' if sc == 's1_relevance' else 'S2'} {form}")
        ie = e_ips.loc[(sc, form), "indep_err_ope"]
        ax.hlines(ie, xs[3] + (offs[j] - 0.46) * width, xs[3] + (offs[j] + 0.46) * width,
                  color="#1a1a18", linewidth=1.4, linestyle=(0, (2, 1.2)), zorder=4)
    ax.hlines([], [], [], color="#1a1a18", linestyle=(0, (2, 1.2)),
              label="independence prediction (OPE arms)")
    ax.set_xticks(xs, arm_labels, fontsize=8)
    ax.set_ylabel("gate error rate vs true verdict (IPS)")
    ax.set_title("A — Gate errors: only the boundary-contested arm (S1 REV) errs — "
                 "absolute 0.225 vs comparative 0.000;\nCTR margin ≈7σ ⇒ no CTR errors, so "
                 "combined error stays single-arm (observed = independence prediction here)",
                 fontsize=9.0)
    ax.legend(fontsize=7.2, loc="upper left")

    # B — 지표 간 오차 산점 (comparative-IPS Δ̂): 같은 w_i 공유 → 동시 오염
    ax = axes[0, 1]
    for sc in SCENARIOS:
        gsc = comp_ips[comp_ips["scenario"] == sc]
        sz = 12 + 60 * (gsc["max_weight_raw"] - gsc["max_weight_raw"].min()) / (
            gsc["max_weight_raw"].max() - gsc["max_weight_raw"].min())
        ax.scatter(gsc["stat_ctr"], gsc["stat_rev"], s=sz, color=POLICY_COLORS[sc],
                   alpha=0.65, lw=0, label=f"{sc} (r={corr[sc][0, 2]:.2f})")
        ax.scatter(gt[sc]["ctr"] - v0["ctr"], gt[sc]["rev"] - v0["rev"], marker="*",
                   s=190, color=POLICY_COLORS[sc], edgecolor="#1a1a18", zorder=5)
    ax.axvline(0.0, color="#6f6e66", linewidth=1.0, linestyle="--")
    ax.axhline(-g_abs, color="#6f6e66", linewidth=1.0, linestyle="--")
    ax.annotate(f"gate lines: ΔCTR=0, ΔREV=−g\n★ true Δ · marker size ∝ max weight\n"
                f"corr(Δ̂CTR,Δ̂CVR)={corr['s1_relevance'][0, 1]:.2f} (S1), "
                f"{corr['s2_price_tilt'][0, 1]:.2f} (S2)\n"
                f"all-3 deviations same sign: {conc['s1_relevance']:.2f} (S1), "
                f"{conc['s2_price_tilt']:.2f} (S2) vs 0.25 indep",
                xy=(0.02, 0.02), xycoords="axes fraction", fontsize=7.2,
                color="#6f6e66", va="bottom")
    ax.set_xlabel(r"$\hat\Delta$CTR (same-log comparative, IPS)")
    ax.set_ylabel(r"$\hat\Delta$REV")
    ax.set_title("B — One log, one weight vector, three co-moving estimates:\n"
                 "per-seed deviations agree in sign ~2x more often than independence",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper left")

    # C — 광고주 노출 재분배 (정확 계산) + HHI
    ax = axes[1, 0]
    xs_a = np.arange(CFG.n_advertisers)
    for j, p in enumerate(pol_ids):
        ax.bar(xs_a + (j - 1) * 0.26, shares[p], 0.24, color=POLICY_COLORS[p],
               label=f"{p} (HHI {hhi_of[p]:.3f})")
    ax.set_xticks(xs_a, [f"adv {gadv}" for gadv in xs_a])
    ax.set_ylabel("exposure share (exact — $\\pi$ known, not OPE)")
    ax.set_title(f"C — Advertiser exposure redistribution: price-tilt concentrates "
                 f"exposure\n(HHI {hhi_of['incumbent_b2']:.3f} → "
                 f"{hhi_of['s2_price_tilt']:.3f} > h={H_HHI} demo cap; error-free arm)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5)

    # D — subgroup 매출 IPS 추정 가능성: 가중 클릭 이벤트 vs 임계
    ax = axes[1, 1]
    for j, sc in enumerate(SCENARIOS):
        for gadv in range(CFG.n_advertisers):
            gsub = sub[(sub["scenario"] == sc) & (sub["advertiser"] == gadv)]
            x0 = gadv + (j - 0.5) * 0.3
            ax.vlines(x0, gsub["weighted_events"].min(), gsub["weighted_events"].max(),
                      color=POLICY_COLORS[sc], linewidth=2.0, alpha=0.7)
            ax.scatter(x0, gsub["weighted_events"].mean(), s=26, color=POLICY_COLORS[sc],
                       zorder=4, label=sc if gadv == 0 else None)
    ax.axhline(MIN_SUBGROUP_EVENTS, color="#1a1a18", linewidth=1.2, linestyle=":",
               label=f"MIN_SUBGROUP_EVENTS={MIN_SUBGROUP_EVENTS}")
    ne_all = float((~sub["estimable"]).mean())
    ax.annotate(f"not-estimable rate = {ne_all:.1%} across all "
                f"(scenario, seed, advertiser)\nmin weighted events = "
                f"{sub['weighted_events'].min():.1f}", xy=(0.02, 0.05),
                xycoords="axes fraction", fontsize=7.5, color="#6f6e66")
    ax.set_yscale("log")
    ax.set_xticks(xs_a, [f"adv {gadv}" for gadv in xs_a])
    ax.set_ylabel("weighted click events (log)")
    ax.set_title("D — Subgroup revenue IPS estimability: min–mean–max over 40 seeds\n"
                 "(threshold pre-registered; sparse regimes would return not-estimable)",
                 fontsize=9.5)
    ax.legend(fontsize=7.5, loc="center right")

    fig.suptitle(
        "Axis 16 — Multi-metric business gate: metric estimates co-move on a shared log; "
        "gate safety comes from comparison design + deterministic arms\n"
        f"candidate ignores price (β {CFG.beta_log:g}→{CFG.beta_eval:g}; S2 tilts to cheap "
        f"actions: ΔCTR {gt['s2_price_tilt']['ctr'] - v0['ctr']:+.4f}, ΔREV "
        f"{gt['s2_price_tilt']['rev'] - v0['rev']:+.4f}) · n={CFG.n:,} · S={len(list(SEEDS))} "
        f"seeds · g=5%·V_rev(π₀), h={H_HHI}, tilt c={PRICE_TILT} are DEMO values (uncalibrated "
        "— proposal frame)\ncaveats: hyperparams_from_weights ignores reward scale (clipping "
        "cannot tame the w·price tail); γ=0 premise (axis 09); retention out of scope (RL OPE)",
        fontsize=9.2)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ─────────────
    print(f"[16] GT incumbent={ {m: round(v0[m], 6) for m in METRICS} } "
          f"S1={ {m: round(gt['s1_relevance'][m], 6) for m in METRICS} } "
          f"S2={ {m: round(gt['s2_price_tilt'][m], 6) for m in METRICS} }")
    print(f"[16] price={np.round(price, 4).tolist()} advertiser_of={adv.tolist()} "
          f"HHI={ {p: round(hhi_of[p], 4) for p in pol_ids} } (LEDGER 등재 대상)")
    print(f"[16] true arms: { {sc: true_arm[sc] for sc in SCENARIOS} } "
          f"true deploy: {true_deploy}")
    e = errs.set_index(["scenario", "estimator", "gate_form"])
    for est_name in GATE_ESTIMATORS:
        for sc in SCENARIOS:
            c_, a_ = e.loc[(sc, est_name, "comparative")], e.loc[(sc, est_name, "absolute")]
            print(f"[16] PATTERN {est_name}/{sc}: comp err(ctr,rev,ope,full)="
                  f"({c_['err_ctr']:.3f},{c_['err_rev']:.3f},{c_['err_ope']:.3f},"
                  f"{c_['err_full']:.3f}) vs abs=({a_['err_ctr']:.3f},{a_['err_rev']:.3f},"
                  f"{a_['err_ope']:.3f},{a_['err_full']:.3f}) | ope-err obs vs indep: "
                  f"comp {c_['err_ope']:.3f}/{c_['indep_err_ope']:.3f}, "
                  f"abs {a_['err_ope']:.3f}/{a_['indep_err_ope']:.3f} | joint arm-err "
                  f"{c_['joint_arm_err']:.3f} vs marg-prod {c_['marg_arm_err_prod']:.3f}")
    print(f"[16] PATTERN Δ̂ corr (comp-IPS): S1 ctr-cvr={corr['s1_relevance'][0, 1]:.3f} "
          f"ctr-rev={corr['s1_relevance'][0, 2]:.3f} cvr-rev={corr['s1_relevance'][1, 2]:.3f} "
          f"| S2 ctr-cvr={corr['s2_price_tilt'][0, 1]:.3f} "
          f"ctr-rev={corr['s2_price_tilt'][0, 2]:.3f} "
          f"cvr-rev={corr['s2_price_tilt'][1, 2]:.3f}")
    for sc in SCENARIOS:
        print(f"[16] PATTERN {sc}: all-3-dev-same-sign={conc[sc]:.3f} (indep 0.25) | "
              f"sd comp vs abs: ctr {sd_comp[sc]['ctr']:.5f}/{sd_abs[sc]['ctr']:.5f}, "
              f"rev {sd_comp[sc]['rev']:.5f}/{sd_abs[sc]['rev']:.5f}")
    print(f"[16] PATTERN hhi_arm_err=0 (결정적): {bool((gates['err_hhi'] == 0).all())} | "
          f"subgroup not-estimable rate={ne_all:.3f} "
          f"(min events={sub['weighted_events'].min():.1f})")
    print(f"[16] → {csv_path.name}, {gates_path.name}, {adv_path.name}, {fig_path.name}")


if __name__ == "__main__":
    main()
