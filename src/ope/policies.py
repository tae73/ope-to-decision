"""로깅/평가 정책 생성기 — 순수함수. [M1 구현 예정 — 스텁]"""

import numpy as np


def softmax_policy(q: np.ndarray, beta: float) -> np.ndarray:
    """π(a|x) ∝ exp(β·q(x,a)). β→0 uniform, β→∞ 준결정적(overlap 붕괴), β<0 worse-than-uniform."""
    raise NotImplementedError("M1")


def epsilon_greedy_policy(q: np.ndarray, eps: float) -> np.ndarray:
    """argmax q 에 1-ε, 나머지에 ε/(K-1)."""
    raise NotImplementedError("M1")
