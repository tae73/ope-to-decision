"""실데이터 2트랙 로더. [M3 구현 예정 — 스텁]

트랙 1 — classification-to-bandit 변환(표준 프로토콜, Dudík/Agarwal 계열):
    multi-class 라벨 = action, 정답 여부 = reward. optdigits·satimage·pendigits·letter 등.
트랙 2 — Open Bandit Dataset small(ZOZO): 실측 propensity + 로깅정책 2종(random/BTS)
    → random 쪽 on-policy 평균을 *근사* ground truth 로 사용. 근사참값에는 항상 bootstrap CI 병기.
데이터 파일은 커밋하지 않는다(data/README.md).
"""

from typing import NamedTuple

import numpy as np


class RealBanditData(NamedTuple):
    context: np.ndarray
    action: np.ndarray
    reward: np.ndarray
    pscore: np.ndarray
    n_actions: int
    source: str


def classification_to_bandit(dataset_name: str, logging_beta: float, seed: int) -> RealBanditData:
    """OpenML 데이터셋 → logged bandit feedback 변환 (70/30 split, 학습된 stochastic 로깅 정책)."""
    raise NotImplementedError("M3")


def load_obd_small(path: str, campaign: str, policy: str) -> RealBanditData:
    """data/obd/ 에 배치된 OBD small 로드."""
    raise NotImplementedError("M3")
