"""M8 GT-미상 practitioner 트랙 공용 하네스 — frontstage/reveal 스키마·프로토콜 (PLAN §3.5).

frontstage(본편)와 reveal(백스테이지)의 분리를 **스키마·코드 경로 수준에서 강제**한다:
- `DECISION_COLUMNS` 에는 oracle 컬럼(`v_true`·`q_true`·`pscore_true`·`gt_value`)이 존재하지
  않는다 — tests/test_practitioner_contract.py 가 문자열 수준에서 ban.
- `reveal()` 은 **committed frontstage CSV 를 파일로 읽어서만** 채점한다(in-memory 객체 전달
  금지 — 경로 분리 강제). 축 20(OBD)은 reveal 파일 자체가 없다.
- 결정 규칙(사전등록 §3.5-2): 결정 estimator = SNIPS · incumbent anchor = mean(r) ·
  trust 에서 GO ⇔ SNIPS CI 하한 > anchor / NO-GO ⇔ 상한 < anchor / 그 외 AB_TEST;
  distrust·ab_fallback 은 무조건 AB_TEST. 동시 성립 시 ab_fallback 라벨 우선.
  fragile ⇐ Λ*_flip < 1.5 [제안 — 라벨만, GO 강등 아님].

불확실성 규약(CLAUDE.md §2 M8 예외): frontstage 는 합성 로그여도 단일-로그 joint bootstrap —
실무자에게 로그는 하나다. seed-ensemble 은 reveal 집계 전용.
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ope.diagnostics import PROVISIONAL_THRESHOLDS, compute_diagnostics, decision_gate
from ope.estimators import (
    estimate_clipped_ips, estimate_dm, estimate_dr, estimate_dros,
    estimate_ips, estimate_snips, estimate_switch_dr,
)
from ope.validity import (
    ValidityConfig, bootstrap_joint, lambda_star_vs_anchor, run_validity_checks,
)

from _common import TAB_DIR, hyperparams_from_weights

TRACK = "practitioner"
DECISION_ESTIMATOR = "snips"   # §3.5-2 사전등록 — 사후 변경 금지
LARGE_ERR = 0.10               # reveal 대오차 기준 (축 08 계승)
FRAGILE_LAMBDA = 1.5           # [제안 — 라벨만]
LAM_MAX = 8.0                  # 축 14 grid 상한 계승

# frontstage 스키마 — oracle 컬럼 부재가 계약이다 (v_true 류 금지)
DECISION_COLUMNS = [
    "track", "axis_id", "scenario", "run_id", "seed", "estimator",
    "estimate", "ci_lo", "ci_hi", "hyperparam_value", "incumbent_mean_r",
    "ess_ratio_raw", "max_weight_raw", "support_proxy", "gate_v1",
    "mean_w", "mean_w_state", "harmonic_worst_t", "harmonic_state",
    "placebo_value", "placebo_state", "disagreement", "disagreement_state",
    "dr_correction", "nc_covariate",
    "lam_star_flip", "lam_star_censored", "lam_direction",
    "protocol_verdict", "decision", "fragile",
]

# reveal 스키마 — 백스테이지 전용 (여기만 truth 가 존재)
REVEAL_COLUMNS = [
    "axis_id", "scenario", "run_id", "seed", "estimator", "truth_kind",
    "v_true", "v_true_lo", "v_true_hi", "estimate", "ci_lo", "ci_hi",
    "rel_err", "large_err", "ci_covers_truth",
    "protocol_verdict", "decision",
]

TRUTH_KINDS = ("exact_synthetic", "exact_c2b", "approx_band_obd", "none")


class PractitionerLog(NamedTuple):
    """실무자 시점 로그 뷰 — **로그 층 필드만**(oracle 필드가 타입 수준에서 없다).

    pi_log_dist 는 "내가 배포한 정책은 스코어할 수 있다" 자산(있을 때만 — OBD BTS 는 None).
    """
    context: np.ndarray          # (n, d) — 없으면 shape (n, 0)
    action: np.ndarray           # (n,)
    reward: np.ndarray           # (n,)
    pscore: np.ndarray           # (n,) 기록 propensity — 진위는 보증되지 않는다(축 05·09·18)
    pi_log_dist: np.ndarray = None   # (n, K) 기록된 로깅 분포 — 선택
    timestamps: np.ndarray = None    # (n,) — 선택 (time-split 보고용)


def split_log(n: int, frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """후보구축(fit)/평가(eval) 분리 인덱스 — winner's curse 방어(§3.5 M8 설계 결정).

    반환: (idx_fit, idx_eval) — 셔플 후 앞 ⌊frac·n⌋ 이 fit.
    """
    if not 0.0 < frac < 1.0:
        raise ValueError(f"frac must be in (0,1), got {frac}")
    perm = np.random.default_rng(seed).permutation(n)
    cut = int(np.floor(frac * n))
    return np.sort(perm[:cut]), np.sort(perm[cut:])


def protocol_verdict(gate_v1_decision: str, report) -> str:
    """§3.5-2 결합 규칙 — 동시 성립 시 ab_fallback 우선."""
    if gate_v1_decision == "ab_fallback" or report.harmonic.state == "fail":
        return "ab_fallback"
    if (gate_v1_decision == "distrust" or report.mean_w.state == "fail"
            or report.placebo.state == "fail" or report.disagreement.state == "fail"):
        return "distrust"
    return "trust"


def decide(verdict: str, ci_lo: float, ci_hi: float, incumbent: float) -> str:
    """§3.5-2 결정 규칙 — 결정 estimator(SNIPS)의 CI vs incumbent(mean r)."""
    if verdict != "trust":
        return "ab_test"
    if ci_lo > incumbent:
        return "go"
    if ci_hi < incumbent:
        return "no_go"
    return "ab_test"


def run_protocol(log: PractitionerLog, pi_e_dist: np.ndarray, *, axis_id: str,
                 scenario: str, run_id: str, seed: int, q_hat: np.ndarray = None,
                 cfg: ValidityConfig = None) -> list[dict]:
    """frontstage 프로토콜 1회 실행 → estimator 별 DECISION_COLUMNS dict 행 목록.

    입력은 PractitionerLog + 후보 분포(+선택 q̂ — 로그 유래여야 한다: fitters.py)뿐이다.
    v_true 는 이 함수의 어떤 경로에도 등장하지 않는다(blindness 테스트가 실행 수준에서 고정).
    """
    cfg = cfg if cfg is not None else ValidityConfig(seed=seed)
    w = estimate_ips(log.reward, log.action, log.pscore, pi_e_dist).weights
    hypers = hyperparams_from_weights(w)
    diag = compute_diagnostics(log.pscore, log.action, pi_e_dist)
    gate_v1 = decision_gate(diag, PROVISIONAL_THRESHOLDS).decision

    boot = bootstrap_joint(log.reward, log.action, log.pscore, pi_e_dist,
                           q_hat=q_hat, hypers=hypers, cfg=cfg)
    report = run_validity_checks(log.reward, log.action, log.pscore, pi_e_dist,
                                 q_hat=q_hat, context=log.context, cfg=cfg, boot=boot)

    incumbent = float(log.reward.mean())
    lam = lambda_star_vs_anchor(log.reward, log.action, log.pscore, pi_e_dist,
                                anchor=incumbent, lam_max=LAM_MAX)
    verdict = protocol_verdict(gate_v1, report)

    points = {
        "ips": (estimate_ips(log.reward, log.action, log.pscore, pi_e_dist).value, np.nan),
        "snips": (estimate_snips(log.reward, log.action, log.pscore, pi_e_dist).value, np.nan),
        "clipped_ips": (estimate_clipped_ips(log.reward, log.action, log.pscore,
                                             pi_e_dist, hypers["lam_clip"]).value,
                        hypers["lam_clip"]),
    }
    if q_hat is not None:
        points["dm"] = (estimate_dm(pi_e_dist, q_hat).value, np.nan)
        points["dr"] = (estimate_dr(log.reward, log.action, log.pscore,
                                    pi_e_dist, q_hat).value, np.nan)
        points["switch_dr"] = (estimate_switch_dr(log.reward, log.action, log.pscore,
                                                  pi_e_dist, q_hat, hypers["tau"]).value,
                               hypers["tau"])
        points["dros"] = (estimate_dros(log.reward, log.action, log.pscore,
                                        pi_e_dist, q_hat, hypers["lam_dros"]).value,
                          hypers["lam_dros"])

    snips_lo, snips_hi = np.quantile(boot.estimates["snips"],
                                     [cfg.alpha / 2, 1 - cfg.alpha / 2])
    decision = decide(verdict, float(snips_lo), float(snips_hi), incumbent)
    fragile = bool((not lam.censored) and lam.lam_star < FRAGILE_LAMBDA)

    rows = []
    for name, (value, hp) in points.items():
        lo, hi = np.quantile(boot.estimates[name], [cfg.alpha / 2, 1 - cfg.alpha / 2])
        rows.append({
            "track": TRACK, "axis_id": axis_id, "scenario": scenario,
            "run_id": run_id, "seed": seed, "estimator": name,
            "estimate": float(value), "ci_lo": float(lo), "ci_hi": float(hi),
            "hyperparam_value": float(hp), "incumbent_mean_r": incumbent,
            "ess_ratio_raw": diag.ess_ratio, "max_weight_raw": diag.max_weight,
            "support_proxy": diag.support_deficiency, "gate_v1": gate_v1,
            "mean_w": report.mean_w.value, "mean_w_state": report.mean_w.state,
            "harmonic_worst_t": report.harmonic.value,
            "harmonic_state": report.harmonic.state,
            "placebo_value": report.placebo.value, "placebo_state": report.placebo.state,
            "disagreement": report.disagreement.value,
            "disagreement_state": report.disagreement.state,
            "dr_correction": report.dr_correction, "nc_covariate": report.nc_covariate,
            "lam_star_flip": lam.lam_star, "lam_star_censored": lam.censored,
            "lam_direction": lam.direction,
            "protocol_verdict": verdict, "decision": decision, "fragile": fragile,
        })
    return rows


def write_decision_csv(axis_id: str, slug: str, rows: list[dict]) -> Path:
    """frontstage CSV 커밋 경로 — 컬럼 순서는 DECISION_COLUMNS 그대로(스키마 드리프트 방지)."""
    out = TAB_DIR / f"{axis_id}_{slug}_decision.csv"
    pd.DataFrame(rows)[DECISION_COLUMNS].to_csv(out, index=False)
    return out


def reveal(decision_csv: Path, truth: pd.DataFrame, truth_kind: str) -> Path:
    """백스테이지 채점 — **frontstage CSV 를 파일로 읽어서만** 수행(경로 분리 강제).

    truth: `run_id` + `v_true`(+선택 `v_true_lo`/`v_true_hi` — 근사 band) DataFrame.
    rel_err = |estimate − v_true| / |v_true| · large_err = rel_err > LARGE_ERR ·
    ci_covers_truth = [ci_lo, ci_hi] 가 v_true(또는 band 와 겹침)를 포함.
    truth_kind == "none" 은 reveal 자체가 성립하지 않는다(축 20) — 호출 금지.
    """
    if truth_kind not in TRUTH_KINDS or truth_kind == "none":
        raise ValueError(f"truth_kind must be one of {TRUTH_KINDS[:-1]} for reveal, "
                         f"got {truth_kind!r}")
    decision_csv = Path(decision_csv)
    front = pd.read_csv(decision_csv)
    if any(c in front.columns for c in ("v_true", "q_true", "pscore_true", "gt_value")):
        raise ValueError("frontstage CSV 에 oracle 컬럼이 존재 — 계약 위반")
    t = truth.copy()
    if "v_true_lo" not in t.columns:
        t["v_true_lo"] = t["v_true"]
    if "v_true_hi" not in t.columns:
        t["v_true_hi"] = t["v_true"]
    m = front.merge(t[["run_id", "v_true", "v_true_lo", "v_true_hi"]],
                    on="run_id", how="inner", validate="many_to_one")
    m["truth_kind"] = truth_kind
    m["rel_err"] = (m["estimate"] - m["v_true"]).abs() / m["v_true"].abs()
    m["large_err"] = m["rel_err"] > LARGE_ERR
    m["ci_covers_truth"] = (m["ci_lo"] <= m["v_true_hi"]) & (m["ci_hi"] >= m["v_true_lo"])
    out = decision_csv.with_name(decision_csv.name.replace("_decision.csv", "_reveal.csv"))
    m[REVEAL_COLUMNS].to_csv(out, index=False)
    return out
