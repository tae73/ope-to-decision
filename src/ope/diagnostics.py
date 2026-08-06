"""로그에서 계산 가능한 진단 + decision gate. [M1 구현 예정 — 스텁]

dag-registry `docs/dag-design.md` §6.5 의 `diagnostics` 필드(ESS·max importance weight·clipping)를
실행 코드로 실증하는 모듈. decision_gate 의 규칙은 문헌 표준이 아니라 **본 레포의 제안**
(folklore 관행의 체계화 시도)이며, 축 08 에서 예보력을 실증하고 축 09 에서 blind spot 을 전시한다.
"""

from typing import Literal, NamedTuple

import numpy as np


class DiagnosticsReport(NamedTuple):
    ess: float             # (Σw)²/Σw²
    ess_ratio: float       # ESS/n
    max_weight: float
    weight_tail_p99: float
    support_deficiency: float  # π_e 질량 중 π_0 지지 밖 비율 추정


class GateVerdict(NamedTuple):
    decision: Literal["trust", "distrust", "ab_fallback"]
    reasons: tuple[str, ...]


def compute_diagnostics(pscore: np.ndarray, action: np.ndarray, pi_e_dist: np.ndarray,
                        weights: np.ndarray | None = None) -> DiagnosticsReport:
    """로그만으로(참값 없이) 계산 가능한 진단 일체.

    weights 를 주면 estimator 가 실제 사용한 (clipping 등 변형 후) weight 를 진단하고,
    없으면 raw weight w = π_e(a|x)/π_0(a|x) 를 자체 재계산한다 (CLAUDE.md §2 규약).
    """
    raise NotImplementedError("M1")


def decision_gate(report: DiagnosticsReport, thresholds: dict) -> GateVerdict:
    """진단 → 3-way 판정. [제안 — 표준 아님] 임계값 근거는 축 08 실험 결과로만 정당화한다."""
    raise NotImplementedError("M1")
