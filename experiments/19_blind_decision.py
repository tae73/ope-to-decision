"""축 19 — end-to-end blind decision: 로그만으로 GO/NO-GO/AB, 프로토콜의 결정 가치 (M8 본편).

축 10("결정 안전성은 비교 설계의 속성")의 GT-미상 판: 후보 정책까지 **로그에서 구축**한다 —
`split_log`(fit 30% / eval 70%, winner's curse 방어) → fit-half 에서 per-action Ridge q̂ 적합 →
후보 π_cand = softmax(β_cand·q̂) (β_cand ∈ {−2, 2, 8} — **나쁜 후보(−2) 포함**이 설계의 핵심:
false-go 를 측정하려면 진짜로 나쁜 후보가 사다리에 있어야 한다) → eval-half 에서 frontstage
프로토콜 → 결정 → reveal 채점(regret·false-go/false-stop).

비교 대상 naive baseline = **IPS 점추정 > mean(r) 이면 GO**(게이트·CI·battery 없음 — 실무의
기본값). 오염 regime(noised s=1.0)에서 naive 는 E[w] 인플레이션(e^{s²/2}≈1.65)으로 나쁜
후보에도 확신에 찬 GO 를 내는 반면, 프로토콜은 battery 발화 → AB 회귀로 전환한다 —
**결정 가치 = 회피된 false-go, 비용 = 유예(deferral)**. 둘 다 그대로 보고한다.

frontstage 는 후보 분포·로그만 사용(oracle 비접촉 — blindness 테스트 계약 승계). 진짜 후보
가치 V(π_cand)·V(π₀) 는 reveal 층에서만 oracle MC 로 계산한다.

산출: results/tables/19_blind_decision_decision.csv · _reveal.csv · _summary.csv
      ↔ results/figures/19_blind_decision.png
"""

import time

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from _common import BASE_M2, SEEDS_DEFAULT, TAB_DIR
from _practitioner import (
    DECISION_ESTIMATOR, PractitionerLog, reveal, run_protocol, split_log,
    write_decision_csv,
)
from _style import apply_style, save_figure
from ope.dgp import _make_structure, _q_true, make_synthetic_bandit_data
from ope.fitters import CrossFitConfig, fit_q_hat_crossfit
from ope.policies import softmax_policy
from ope.validity import ValidityConfig

AXIS_ID, SLUG = "19", "blind_decision"
SEEDS = SEEDS_DEFAULT
REGIMES = {  # (β_log, pscore 오염 s — 0 이면 무오염)
    "healthy": (1.0, 0.0),
    "low_overlap": (8.0, 0.0),
    "noised": (1.0, 1.0),
}
BETA_CAND = (-2.0, 2.0, 8.0)   # 나쁜 후보(−2) 포함 — false-go 측정의 전제
FIT_FRAC = 0.3
PS_CLIP = (1e-6, 1.0)
NOISE_RNG_OFFSET = 500_000     # 축 05·17 관례
N_MC_TRUTH = 200_000           # reveal 층 oracle MC (house 표준)


def _fit_candidate_models(x_fit, a_fit, r_fit, k: int) -> list:
    """fit-half 에서 per-action Ridge — 후보 구축용(로그 유래·oracle 비접촉)."""
    models = []
    for a in range(k):
        rows = a_fit == a
        if rows.sum() >= 2:
            m = Ridge(alpha=1.0).fit(x_fit[rows], r_fit[rows])
            models.append(("model", m))
        else:
            fallback = float(r_fit[rows].mean()) if rows.any() else float(r_fit.mean())
            models.append(("const", fallback))
    return models


def _predict_q(models, x) -> np.ndarray:
    q = np.empty((len(x), len(models)))
    for a, (kind, m) in enumerate(models):
        q[:, a] = m.predict(x) if kind == "model" else m
    return q


def main() -> None:
    t0 = time.perf_counter()
    n, k = BASE_M2.n, BASE_M2.n_actions
    all_rows, truth_rows = [], []
    for regime, (beta_log, s) in REGIMES.items():
        cfg_r = BASE_M2._replace(beta_log=beta_log)
        # V(π₀) 참값 — reveal 층 (regime 당 1회 oracle MC)
        theta, bvec, dvec = _make_structure(cfg_r)
        mc_rng = np.random.default_rng(2_000_003 + cfg_r.struct_seed)
        x_mc = mc_rng.normal(size=(N_MC_TRUTH, cfg_r.dim_context))
        q_mc = _q_true(x_mc, theta, bvec)
        v0_true = float((softmax_policy(q_mc, beta_log) * q_mc).sum(axis=1).mean())
        for seed in SEEDS:
            c = cfg_r._replace(seed=seed)
            d = make_synthetic_bandit_data(c)
            idx_fit, idx_eval = split_log(n, FIT_FRAC, seed)
            ps_full = d.pscore_logged
            if s > 0:  # 시스템측 기록 오염 (전체 로그에 적용 후 분할)
                z = np.random.default_rng(NOISE_RNG_OFFSET + seed).normal(size=n)
                ps_full = np.clip(d.pscore_logged * np.exp(s * z), *PS_CLIP)
            models = _fit_candidate_models(d.context[idx_fit], d.action[idx_fit],
                                           d.reward[idx_fit], k)
            q_cand_eval = _predict_q(models, d.context[idx_eval])
            q_cand_mc = _predict_q(models, x_mc)  # reveal 층 전용
            q_hat_eval = fit_q_hat_crossfit(d.context[idx_eval], d.action[idx_eval],
                                            d.reward[idx_eval], k,
                                            CrossFitConfig(seed=seed))
            log = PractitionerLog(context=d.context[idx_eval], action=d.action[idx_eval],
                                  reward=d.reward[idx_eval], pscore=ps_full[idx_eval])
            for beta_c in BETA_CAND:
                pi_cand = softmax_policy(q_cand_eval, beta_c)
                run_id = f"{regime}-b{beta_c:g}-{seed}"
                all_rows += run_protocol(
                    log, pi_cand, axis_id=AXIS_ID, scenario=f"{regime}_b{beta_c:g}",
                    run_id=run_id, seed=seed, q_hat=q_hat_eval,
                    cfg=ValidityConfig(seed=seed))
                # ── reveal 층: 후보 참값 (oracle MC — frontstage 비접촉) ────────
                vc = float((softmax_policy(q_cand_mc, beta_c) * q_mc).sum(axis=1).mean())
                truth_rows.append({"run_id": run_id, "v_true": vc, "v0_true": v0_true,
                                   "regime": regime, "beta_cand": beta_c, "seed": seed})
    dec_path = write_decision_csv(AXIS_ID, SLUG, all_rows)
    truth = pd.DataFrame(truth_rows)
    rev_path = reveal(dec_path, truth[["run_id", "v_true"]], "exact_synthetic")

    # ── 결정 채점: 프로토콜 vs naive (IPS 점추정 무게이트) ────────────────────────
    dec = pd.read_csv(dec_path)
    snips = dec[dec["estimator"] == DECISION_ESTIMATOR][
        ["run_id", "scenario", "seed", "decision", "protocol_verdict",
         "incumbent_mean_r", "fragile"]]
    ips = dec[dec["estimator"] == "ips"][["run_id", "estimate"]].rename(
        columns={"estimate": "est_ips"})
    sc = snips.merge(ips, on="run_id").merge(truth, on="run_id")
    sc["truly_better"] = sc["v_true"] > sc["v0_true"]
    sc["naive_go"] = sc["est_ips"] > sc["incumbent_mean_r"]
    sc["proto_false_go"] = (sc["decision"] == "go") & ~sc["truly_better"]
    sc["proto_false_stop"] = (sc["decision"] == "no_go") & sc["truly_better"]
    sc["naive_false_go"] = sc["naive_go"] & ~sc["truly_better"]
    sc["naive_false_stop"] = ~sc["naive_go"] & sc["truly_better"]
    sc["regret_if_go"] = np.where(sc["decision"] == "go",
                                  np.maximum(0.0, sc["v0_true"] - sc["v_true"]), np.nan)
    sc["naive_regret_if_go"] = np.where(sc["naive_go"],
                                        np.maximum(0.0, sc["v0_true"] - sc["v_true"]),
                                        np.nan)
    summary = (sc.groupby(["regime", "beta_cand"])
               .agg(runs=("run_id", "size"),
                    truly_better_share=("truly_better", "mean"),
                    proto_go=("decision", lambda x: (x == "go").mean()),
                    proto_no_go=("decision", lambda x: (x == "no_go").mean()),
                    proto_ab=("decision", lambda x: (x == "ab_test").mean()),
                    proto_false_go=("proto_false_go", "mean"),
                    proto_false_stop=("proto_false_stop", "mean"),
                    naive_false_go=("naive_false_go", "mean"),
                    naive_false_stop=("naive_false_stop", "mean"),
                    proto_regret_go=("regret_if_go", "mean"),
                    naive_regret_go=("naive_regret_if_go", "mean"),
                    fragile_share=("fragile", "mean"))
               .reset_index())
    sum_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_summary.csv"
    summary.to_csv(sum_path, index=False)

    # ── figure: A false-go(프로토콜 vs naive) / B 프로토콜 결정 구성 ──────────────
    apply_style()
    import matplotlib.pyplot as plt
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    order = [(r, b) for r in REGIMES for b in BETA_CAND]
    labels = [f"{r}\nβ={b:g}" for r, b in order]
    xs = np.arange(len(order))
    nfg = [float(summary[(summary.regime == r) & (summary.beta_cand == b)]
                 ["naive_false_go"].iloc[0]) for r, b in order]
    pfg = [float(summary[(summary.regime == r) & (summary.beta_cand == b)]
                 ["proto_false_go"].iloc[0]) for r, b in order]
    ax.bar(xs - 0.19, nfg, width=0.36, color="#e34948",
           label="naive (IPS point > mean r, no gate)")
    ax.bar(xs + 0.19, pfg, width=0.36, color="#44433e", label="protocol (this repo)")
    ax.set_xticks(xs, labels, fontsize=7.5)
    ax.set_ylabel("false-go rate  (deploys a truly worse policy)")
    ax.set_ylim(0, 1.05)
    ax.set_title("A. Decision value — avoided false-go\n(backstage-scored; "
                 "candidates built from logs only)", fontsize=10)
    ax.legend(fontsize=8)

    bottom = np.zeros(len(order))
    for part, color, lab in (("proto_go", "#1baf7a", "go"),
                             ("proto_no_go", "#eb6834", "no-go"),
                             ("proto_ab", "#b3b2a9", "ab_test (defer)")):
        vals = [float(summary[(summary.regime == r) & (summary.beta_cand == b)]
                      [part].iloc[0]) for r, b in order]
        bx.bar(xs, vals, bottom=bottom, width=0.6, color=color, label=lab)
        bottom += np.array(vals)
    bx.set_xticks(xs, labels, fontsize=7.5)
    bx.set_ylabel("protocol decision share")
    bx.set_title("B. The cost side — deferral is explicit\n(distrust/ab_fallback ⇒ "
                 "ab_test by pre-registered rule)", fontsize=10)
    bx.legend(fontsize=8)
    fig.suptitle("Axis 19 — End-to-end decision from logs alone: protocol vs naive "
                 "point-estimate\n(pre-registered rules §3.5-2; candidate ladder includes "
                 "a truly-bad candidate β=−2 by design)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ── PATTERN ──────────────────────────────────────────────────────────────────
    def _cell(r, b, col):
        return float(summary[(summary.regime == r) & (summary.beta_cand == b)][col].iloc[0])

    patterns = {
        "noised×β−2: naive false-go ≥ 0.5 (E[w] 인플레이션의 확신 GO)":
            _cell("noised", -2.0, "naive_false_go") >= 0.5,
        "noised×β−2: protocol false-go = 0 (battery→AB 회귀)":
            _cell("noised", -2.0, "proto_false_go") == 0.0,
        "healthy: protocol 이 결정을 내린다 (전면 AB 아님 — go+no_go > 0.5 어느 β 든)":
            max(_cell("healthy", b, "proto_go") + _cell("healthy", b, "proto_no_go")
                for b in BETA_CAND) > 0.5,
    }
    print(f"[19] → {dec_path.name}, {rev_path.name}, {sum_path.name}, {fig_path.name} "
          f"({time.perf_counter() - t0:.0f}s)")
    for kk, ok in patterns.items():
        print(f"[19] PATTERN {'PASS' if ok else 'FAIL'}: {kk}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
