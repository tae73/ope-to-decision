"""합성 DGP — multi-action contextual bandit, 참 정책가치(ground truth) 보유. [M1 구현 예정 — 스텁]

축 01–04·09(+스트레치 13)의 노브가 DGPConfig 필드 하나에 대응한다.
축 05–08·10 은 DGP 가 아니라 estimator/metric 층의 노브다(PLAN.md §2 매핑표 참조).
축 09(confounding)의 핵심 장치: 미관측 U 가 로깅 선택과 보상에 동시 개입하되
*기록되는* pscore 에는 U 가 빠진다 → pscore_logged ≠ pscore_true.
"""

from typing import NamedTuple

import numpy as np


class DGPConfig(NamedTuple):
    n: int                      # 로그 표본 수 (축 01)
    n_actions: int              # K (스트레치 축 13)
    dim_context: int
    beta_log: float             # 로깅 softmax inverse-temperature — overlap 다이얼 (축 02)
    beta_eval: float            # 평가 정책 온도 — 타깃-로깅 괴리 (축 03)
    support_deficiency: float   # π_0(a|x)=0 강제 비율 (축 04)
    reward_noise: float         # 보상 noise σ (축 06 보조)
    confounding_strength: float # U 개입 강도; 0 이면 unconfounded (축 09)
    seed: int


class SyntheticBanditData(NamedTuple):
    context: np.ndarray        # (n, d)
    action: np.ndarray         # (n,)
    reward: np.ndarray         # (n,)
    pscore_logged: np.ndarray  # (n,)  기록된 propensity — estimator 가 보는 값
    pscore_true: np.ndarray    # (n,)  진짜 propensity — 축 09 진단용 (estimator 에게 비공개)
    pi_e_dist: np.ndarray      # (n, K)
    q_true: np.ndarray         # (n, K) 참 기대보상 — GT/oracle 전용
    v_true: float              # V(π_e) ground truth (해석적/대규모 MC)


def make_synthetic_bandit_data(config: DGPConfig) -> SyntheticBanditData:
    """DGPConfig → 로그 + ground truth. 고정 seed 재현."""
    raise NotImplementedError("M1")


def true_policy_value(config: DGPConfig, n_mc: int = 200_000) -> float:
    """V(π_e) = E_x[Σ_a π_e(a|x) q(x,a)] — 대규모 MC (또는 가능 시 해석적)."""
    raise NotImplementedError("M1")
