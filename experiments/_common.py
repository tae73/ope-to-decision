"""M2 축 실험 공용 러너 — 스키마 상수·RunRecord·hyperparam 정책·v_true memoize.

스키마 드리프트 방지: 모든 축 CSV 는 COLUMNS 순서 그대로 쓴다(검증 테스트가 배리어에서 대조).
estimator 이름은 src/ope/estimators.py 의 `_POINT_ESTIMATORS` 키가 정본.
불확실성 병기: 합성 MC 축은 S-seed ensemble band(±2·SE over seeds) — CLAUDE.md §2 규약.
"""

import csv
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd

from ope.dgp import DGPConfig, SyntheticBanditData, true_policy_value
from ope.diagnostics import compute_diagnostics
from ope.estimators import (
    estimate_clipped_ips, estimate_dm, estimate_dr, estimate_dros,
    estimate_ips, estimate_snips, estimate_switch_dr,
)

ROOT = Path(__file__).resolve().parents[1]
TAB_DIR = ROOT / "results" / "tables"

COLUMNS = ["axis_id", "knob_name", "knob_value", "seed", "estimator", "variant",
           "hyperparam_value", "estimate", "v_true", "ess_ratio_raw", "max_weight_raw",
           "support_proxy", "ess_ratio_used"]

# M2 base config (PLAN §2 — 축별로 필드 하나만 스윕)
BASE_M2 = DGPConfig(n=10_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                    support_deficiency=0.0, reward_noise=0.5, confounding_strength=0.0,
                    seed=0, struct_seed=7)
SEEDS_DEFAULT = range(1000, 1040)  # S=40 (축별 재정의 가능 — 01 은 S=100)


class RunRecord(NamedTuple):
    axis_id: str
    knob_name: str
    knob_value: object       # float 또는 str(모드형 노브)
    seed: int
    estimator: str
    variant: str             # ""(기본) / "q_oracle" / "ps_noised" 등 축내 변형 표기
    hyperparam_value: float  # 해당 estimator 의 λ/τ (없으면 NaN)
    estimate: float
    v_true: float
    ess_ratio_raw: float
    max_weight_raw: float
    support_proxy: float
    ess_ratio_used: float    # estimator 가 실제 쓴 (변형 후) weight 의 ESS ratio (DM=NaN)


_V_TRUE_CACHE: dict[tuple, float] = {}


def cached_v_true(config: DGPConfig) -> float:
    """v_true 는 (struct_seed, beta_eval, K, d) 에만 의존 — memoize (실측 66ms→구조당 1회)."""
    key = (config.struct_seed, config.beta_eval, config.n_actions, config.dim_context)
    if key not in _V_TRUE_CACHE:
        _V_TRUE_CACHE[key] = true_policy_value(config)
    return _V_TRUE_CACHE[key]


def hyperparams_from_weights(w_clean: np.ndarray) -> dict[str, float]:
    """데이터 적응형 hyperparam 정책(문서화·CSV 기록): τ=p95(w), λ_clip=p90(w),
    λ_dros=p90(w)² — DRos 의 λ 는 w² 스케일(Plan 검증 교정). **오염 전(clean) weight 로 계산**
    — 축 05 의 오염된 pscore 로 τ 를 잡으면 축 08 pooling 이 오염된다."""
    return {"tau": float(np.quantile(w_clean, 0.95)),
            "lam_clip": float(np.quantile(w_clean, 0.90)),
            "lam_dros": float(np.quantile(w_clean, 0.90)) ** 2}


def evaluate_run(axis_id: str, knob_name: str, knob_value, seed: int,
                 data: SyntheticBanditData, q_hat: np.ndarray, v_true: float,
                 pscore=None, variant: str = "",
                 hypers: dict[str, float] | None = None) -> list[RunRecord]:
    """한 (knob, seed) 로그에 estimator 7종 평가 → RunRecord 목록.

    pscore 를 주면 estimator 는 그것을 쓴다(축 05 오염 모드) — 진단(raw)은 항상 같은 pscore 기준.
    hypers 미지정 시 clean(전달된) pscore 의 raw weight 로 계산.
    """
    ps = data.pscore_logged if pscore is None else pscore
    n = len(data.action)
    ips_res = estimate_ips(data.reward, data.action, ps, data.pi_e_dist)
    w_raw = ips_res.weights
    if hypers is None:
        hypers = hyperparams_from_weights(w_raw)
    diag = compute_diagnostics(ps, data.action, data.pi_e_dist)

    args = (data.reward, data.action, ps, data.pi_e_dist)
    results = {
        "dm": (estimate_dm(data.pi_e_dist, q_hat), np.nan),
        "ips": (ips_res, np.nan),
        "snips": (estimate_snips(*args), np.nan),
        "clipped_ips": (estimate_clipped_ips(*args, lam=hypers["lam_clip"]), hypers["lam_clip"]),
        "dr": (estimate_dr(*args, q_hat), np.nan),
        "switch_dr": (estimate_switch_dr(*args, q_hat, tau=hypers["tau"]), hypers["tau"]),
        "dros": (estimate_dros(*args, q_hat, lam=hypers["lam_dros"]), hypers["lam_dros"]),
    }
    records = []
    for name, (res, hyper) in results.items():
        if res.weights.size:
            s = float(res.weights.sum() ** 2)
            q = float((res.weights**2).sum())
            ess_used = (s / q / n) if q > 0 else 0.0
        else:
            ess_used = np.nan  # DM sentinel
        records.append(RunRecord(
            axis_id=axis_id, knob_name=knob_name, knob_value=knob_value, seed=seed,
            estimator=name, variant=variant, hyperparam_value=hyper,
            estimate=res.value, v_true=v_true,
            ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
            support_proxy=diag.support_deficiency, ess_ratio_used=ess_used))
    return records


def write_axis_csv(axis_id: str, slug: str, records: list[RunRecord]) -> Path:
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    path = TAB_DIR / f"{axis_id}_{slug}.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(COLUMNS)
        writer.writerows([list(r) for r in records])
    return path


def summarize(csv_path: Path) -> pd.DataFrame:
    """long CSV → (knob_value, variant, estimator) 별 bias²·var·MSE + seed-ensemble SE."""
    df = pd.read_csv(csv_path)
    def agg(g: pd.DataFrame) -> pd.Series:
        err = g["estimate"] - g["v_true"]
        sq = err**2
        s = len(g)
        return pd.Series({
            "n_seeds": s,
            "mean_estimate": g["estimate"].mean(),
            "v_true": g["v_true"].iloc[0],
            "bias": err.mean(),
            "bias2": err.mean() ** 2,
            "var": g["estimate"].var(ddof=1) if s > 1 else np.nan,
            "mse": sq.mean(),
            "mse_se": sq.std(ddof=1) / np.sqrt(s) if s > 1 else np.nan,
            "est_se": g["estimate"].std(ddof=1) / np.sqrt(s) if s > 1 else np.nan,
            "ess_ratio_raw_mean": g["ess_ratio_raw"].mean(),
            "max_weight_raw_mean": g["max_weight_raw"].mean(),
        })
    return (df.groupby(["knob_value", "variant", "estimator"], dropna=False)
              .apply(agg, include_groups=False).reset_index())
