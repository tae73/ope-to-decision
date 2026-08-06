"""로깅/평가 정책 생성기 — 순수함수."""

import numpy as np


def softmax_policy(q: np.ndarray, beta: float) -> np.ndarray:
    """π(a|x) ∝ exp(β·q(x,a)). β→0 uniform, β→∞ 준결정적(overlap 붕괴), β<0 worse-than-uniform.

    Parameters
    ----------
    q : (n, K) 액션별 스코어(보통 참 기대보상).
    beta : softmax inverse-temperature.

    Returns
    -------
    (n, K) 행별 확률분포 (수치 안정: max-shift 후 exp).
    """
    if q.ndim != 2:
        raise ValueError(f"q must be (n, K), got shape {q.shape}")
    z = beta * q
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def epsilon_greedy_policy(q: np.ndarray, eps: float) -> np.ndarray:
    """argmax q 에 1-ε, 나머지 K-1 개 액션에 각 ε/(K-1).

    ε=0 이면 결정적 정책(로깅 정책으로 쓰면 IPS 식별 불능 — 축 02 의 극단 케이스).
    """
    if q.ndim != 2:
        raise ValueError(f"q must be (n, K), got shape {q.shape}")
    if not 0.0 <= eps <= 1.0:
        raise ValueError(f"eps must be in [0, 1], got {eps}")
    n, k = q.shape
    if k < 2:
        raise ValueError("epsilon-greedy needs K >= 2")
    pi = np.full((n, k), eps / (k - 1))
    pi[np.arange(n), q.argmax(axis=1)] = 1.0 - eps
    return pi
