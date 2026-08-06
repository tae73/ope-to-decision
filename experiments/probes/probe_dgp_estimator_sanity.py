"""PROBE M0-A — 합성 DGP + 코어 estimator 최소구현 sanity (GO/NO-GO)

WHAT GENERALIZES:
    GO 면 M1 본구현(src/ope)이 딛는 세 전제가 선다.
    (1) 이 DGP 설계(softmax 로깅 β·평가 β_e·Bernoulli 보상)에서 참 정책가치 V(π_e)를
        대규모 MC 로 안정적으로 얻는다 — 모든 축 실험의 ground truth 파이프라인.
    (2) IPS 불편성 / SNIPS 분산 절감 / DM 모델-bias / DR 이중강건이 교과서 예측대로
        재현된다 — bias-variance 아크 서사의 최소 성립 조건.
    (3) ESS 진단 공식이 검산된다 — decision gate 의 입력.

THE RESULT (boxed — 수치 정본은 results/tables/probe_dgp_sanity.json):
    ┌────────────────────────────────────────────────────────────────┐
    │ S seed 반복에서: |bias(IPS)| ≤ 3·SE, sd(SNIPS) < sd(IPS),      │
    │ |bias(DM_wrong-q)| ≫ SE, |bias(DR_wrong-q)| ≤ 3·SE,            │
    │ sd(DR_oracle-q) ≤ sd(IPS), ESS(균일가중)=n, ESS/n ∈ (0,1]      │
    └────────────────────────────────────────────────────────────────┘

HONEST reduces_check:
    - 통과는 '이 DGP·이 파라미터(β_log=2, β_eval=6, K=10)'에서의 성립이지 전 regime 보증이
      아니다 — 그 지도는 축 실험(01–10)이 그린다.
    - q_hat 오지정은 인위적 축소변환(0.5q+0.25)이며 실제 학습기 오지정(축 06)과 다르다.
    - 3·SE 판정은 S=40 seed 몬테카를로 오차 기준의 확률적 판정이다.

VERDICT: 실행 후 stdout 및 JSON "verdict" 필드 (GO / NO-GO).
"""

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "tables" / "probe_dgp_sanity.json"

D, K = 5, 10
N, S = 20_000, 40
BETA_LOG, BETA_EVAL = 2.0, 6.0
N_MC = 500_000
STRUCT_SEED = 7


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def softmax(logits, beta):
    z = beta * logits
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def q_true(x, theta, b):
    return sigmoid(x @ theta.T + b)


def main() -> None:
    rng0 = np.random.default_rng(STRUCT_SEED)
    theta = rng0.normal(0.0, 1.0, size=(K, D))
    b = rng0.normal(0.0, 0.5, size=K)

    # ground truth: V(π_e) = E_x[Σ_a π_e(a|x) q(x,a)] (대규모 MC)
    x_mc = rng0.normal(size=(N_MC, D))
    q_mc = q_true(x_mc, theta, b)
    v_true = float((softmax(q_mc, BETA_EVAL) * q_mc).sum(axis=1).mean())

    est = {k: [] for k in ["ips", "snips", "dm_wrong", "dr_wrong", "dr_oracle"]}
    ess_ratios = []
    for s in range(S):
        rng = np.random.default_rng(1000 + s)
        x = rng.normal(size=(N, D))
        q = q_true(x, theta, b)
        pi0 = softmax(q, BETA_LOG)
        pi_e = softmax(q, BETA_EVAL)
        a = (pi0.cumsum(axis=1) > rng.random((N, 1))).argmax(axis=1)
        idx = np.arange(N)
        r = rng.binomial(1, q[idx, a]).astype(float)
        w = pi_e[idx, a] / pi0[idx, a]

        q_wrong = 0.5 * q + 0.25  # 인위적 오지정: 0.5 로 축소 수축

        est["ips"].append(float((w * r).mean()))
        est["snips"].append(float((w * r).sum() / w.sum()))
        dm_wrong = float((pi_e * q_wrong).sum(axis=1).mean())
        est["dm_wrong"].append(dm_wrong)
        est["dr_wrong"].append(dm_wrong + float((w * (r - q_wrong[idx, a])).mean()))
        dm_oracle = float((pi_e * q).sum(axis=1).mean())
        est["dr_oracle"].append(dm_oracle + float((w * (r - q[idx, a])).mean()))
        ess_ratios.append(float(w.sum() ** 2 / (w**2).sum() / N))

    stats = {
        k: {
            "mean": float(np.mean(v)),
            "sd": float(np.std(v, ddof=1)),
            "bias": float(np.mean(v) - v_true),
            "se": float(np.std(v, ddof=1) / np.sqrt(S)),
        }
        for k, v in est.items()
    }
    ones = np.ones(100)
    ess_uniform = float(ones.sum() ** 2 / (ones**2).sum())

    checks = {
        "ips_unbiased_within_3se": bool(abs(stats["ips"]["bias"]) <= 3 * stats["ips"]["se"]),
        "snips_lower_sd_than_ips": bool(stats["snips"]["sd"] < stats["ips"]["sd"]),
        "dm_wrong_clearly_biased": bool(abs(stats["dm_wrong"]["bias"]) > 5 * max(stats["dm_wrong"]["se"], 1e-12)),
        "dr_survives_wrong_q_within_3se": bool(abs(stats["dr_wrong"]["bias"]) <= 3 * stats["dr_wrong"]["se"]),
        "dr_oracle_sd_leq_ips_sd": bool(stats["dr_oracle"]["sd"] <= stats["ips"]["sd"]),
        "ess_uniform_equals_n": bool(abs(ess_uniform - 100.0) < 1e-9),
        "ess_ratio_in_unit_interval": bool(all(0.0 < e <= 1.0 + 1e-12 for e in ess_ratios)),
    }
    verdict = "GO" if all(checks.values()) else "NO-GO"

    payload = {
        "probe": "M0-A dgp_estimator_sanity",
        "config": {"d": D, "k": K, "n": N, "n_seeds": S, "beta_log": BETA_LOG,
                   "beta_eval": BETA_EVAL, "n_mc": N_MC, "struct_seed": STRUCT_SEED},
        "v_true": v_true,
        "estimator_stats": stats,
        "ess_ratio_mean": float(np.mean(ess_ratios)),
        "checks": checks,
        "verdict": verdict,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2))

    print("┌" + "─" * 62 + "┐")
    print(f"│ V_true = {v_true:.6f}")
    for k, st in stats.items():
        print(f"│ {k:10s} mean={st['mean']:.6f} bias={st['bias']:+.6f} sd={st['sd']:.6f} se={st['se']:.6f}")
    print(f"│ ESS/n mean = {np.mean(ess_ratios):.4f}   ESS(uniform,n=100) = {ess_uniform:.1f}")
    for name, ok in checks.items():
        print(f"│ {'PASS' if ok else 'FAIL'}  {name}")
    print(f"│ VERDICT: {verdict}")
    print("└" + "─" * 62 + "┘")


if __name__ == "__main__":
    main()
