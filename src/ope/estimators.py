"""OPE estimator 코어 7종 + bootstrap CI — 전부 순수 numpy 함수. [M1 구현 예정 — 스텁]

공통 표기 (shape 규약):
    reward        : (n,)   관측 보상 r_i
    action        : (n,)   로깅 정책이 고른 행동 인덱스 a_i ∈ {0..K-1}
    pscore        : (n,)   로깅 propensity π_0(a_i|x_i) — "기록값" (진짜와 다를 수 있음: 축 09)
    pi_e_dist     : (n, K) 평가 정책의 행동 분포 π_e(·|x_i)
    q_hat         : (n, K) reward 모델 예측 q̂(x_i, ·)

수치 검산 기준: experiments/probes/probe_dgp_estimator_sanity.py (M0) →
obp 교차검증 (probe_obp_crossval.py) → M1 property test 로 3중 검증.
"""

from typing import NamedTuple

import numpy as np


class EstimateResult(NamedTuple):
    """단일 estimator 의 점추정 결과."""

    value: float          # V̂(π_e)
    weights: np.ndarray   # 사용된 (변형 후) importance weight — diagnostics 입력


def estimate_dm(pi_e_dist: np.ndarray, q_hat: np.ndarray) -> EstimateResult:
    """Direct Method: V̂ = mean_i Σ_a π_e(a|x_i) q̂(x_i,a). 저분산·모델 bias 극단."""
    raise NotImplementedError("M1")


def estimate_ips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                 pi_e_dist: np.ndarray) -> EstimateResult:
    """Inverse Propensity Scoring: w_i = π_e(a_i|x_i)/π_0(a_i|x_i), V̂ = mean(w·r). unbiased·고분산."""
    raise NotImplementedError("M1")


def estimate_snips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                   pi_e_dist: np.ndarray) -> EstimateResult:
    """Self-Normalized IPS: V̂ = Σ(w·r)/Σw."""
    raise NotImplementedError("M1")


def estimate_clipped_ips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                         pi_e_dist: np.ndarray, lam: float) -> EstimateResult:
    """Clipped IPS: w ← min(w, λ). λ = bias-variance 다이얼 (dag-registry diagnostics.clipping 대응)."""
    raise NotImplementedError("M1")


def estimate_dr(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                pi_e_dist: np.ndarray, q_hat: np.ndarray) -> EstimateResult:
    """Doubly Robust: DM baseline + IPS 보정항. 두 모델 중 하나만 맞아도 consistent."""
    raise NotImplementedError("M1")


def estimate_switch_dr(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                       pi_e_dist: np.ndarray, q_hat: np.ndarray, tau: float) -> EstimateResult:
    """Switch-DR: w_i > τ 구간만 DM 으로 전환 (Wang-Agarwal-Dudík 2017)."""
    raise NotImplementedError("M1")


def estimate_dros(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                  pi_e_dist: np.ndarray, q_hat: np.ndarray, lam: float) -> EstimateResult:
    """DR with Optimistic Shrinkage: w ← λw/(w²+λ) (Su+ 2020). 'clipping 의 원리화'."""
    raise NotImplementedError("M1")


def bootstrap_ci(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                 pi_e_dist: np.ndarray, q_hat: np.ndarray | None,
                 estimator: str, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    """percentile bootstrap CI — 모든 figure 는 점추정이 아니라 구간을 병기한다(OBD 근사참값 규약)."""
    raise NotImplementedError("M1")
