"""로그에서 계산 가능한 진단 + decision gate.

dag-registry `docs/dag-design.md` §6.5 의 `diagnostics` 필드(ESS·max importance weight·clipping)를
실행 코드로 실증하는 모듈. decision_gate 의 규칙·임계값은 문헌 표준이 아니라 **본 레포의 제안**
(folklore 관행의 체계화 시도)이며, 축 08 에서 예보력을 실증하고 축 09 에서 blind spot 을 전시한다.
"""

from typing import Literal, NamedTuple

import numpy as np

class GateThresholds(NamedTuple):
    """decision_gate 임계값 — NamedTuple 인 이유(§2 규약 + 안전성): raw dict 는 오타 키를
    침묵 무시해 잘못된 gate 판정을 낳는다(M1 적대 리뷰 확정 결함). 생성자가 누락 키를 즉사시킨다."""

    ess_ratio_hard: float          # 미만 → ab_fallback (추정 자체를 신뢰 불가)
    ess_ratio_soft: float          # 미만 → distrust
    max_weight_cap: float          # 초과 → distrust
    support_deficiency_max: float  # proxy 초과 → distrust


# [제안 — 표준 아님] 축 08 실험으로 정당화·교정 예정인 잠정 임계값.
# ess_ratio 하한 계열은 folklore(Owen 관행 등, POSITIONING.md §6 참조)에서 출발한 잠정치다.
PROVISIONAL_THRESHOLDS = GateThresholds(
    ess_ratio_hard=0.01,
    ess_ratio_soft=0.10,
    max_weight_cap=100.0,
    support_deficiency_max=0.02,
)


class DiagnosticsReport(NamedTuple):
    ess: float                 # (Σw)²/Σw²
    ess_ratio: float           # ESS/n
    max_weight: float
    weight_tail_p99: float
    support_deficiency: float  # [제안] proxy: π_e 질량 중 "로그에 0회 등장한 액션" 비율 (아래 주의)


class GateVerdict(NamedTuple):
    decision: Literal["trust", "distrust", "ab_fallback"]
    reasons: tuple[str, ...]


def compute_diagnostics(pscore: np.ndarray, action: np.ndarray, pi_e_dist: np.ndarray,
                        weights: np.ndarray | None = None) -> DiagnosticsReport:
    """로그만으로(참값 없이) 계산 가능한 진단 일체.

    weights 를 주면 estimator 가 실제 사용한 (clipping 등 변형 후) weight 를 진단하고,
    None 또는 빈 배열이면 raw weight w = π_e(a|x)/π_0(a|x) 를 자체 재계산한다 (CLAUDE.md §2 규약;
    DM 의 빈 weights 를 그대로 넘겨도 안전하도록 빈 배열은 None 과 동일 취급).

    support_deficiency 는 **로그만으로 계산 가능한 proxy** — 전역 π̄_e 질량 중 로그에 한 번도
    등장하지 않은 액션의 비율. 컨텍스트-국소적 지지 결핍은 이 proxy 가 놓칠 수 있다(축 04 에서
    민감도를 실증; [제안] 태그 유지).
    """
    n = len(action)
    if weights is None or len(weights) == 0:
        if np.any(pscore <= 0.0):
            raise ValueError("pscore must be strictly positive for logged actions")
        w = pi_e_dist[np.arange(n), action] / pscore
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError(f"weights length {len(w)} != n {n}")

    w_sq_sum = float((w**2).sum())
    ess = float(w.sum() ** 2 / w_sq_sum) if w_sq_sum > 0.0 else 0.0

    pi_bar = pi_e_dist.mean(axis=0)
    observed = np.zeros(pi_e_dist.shape[1], dtype=bool)
    observed[np.unique(action)] = True
    support_proxy = float(pi_bar[~observed].sum())

    return DiagnosticsReport(
        ess=ess,
        ess_ratio=ess / n,
        max_weight=float(w.max()) if n > 0 else 0.0,
        weight_tail_p99=float(np.quantile(w, 0.99)) if n > 0 else 0.0,
        support_deficiency=support_proxy,
    )


def decision_gate(report: DiagnosticsReport, thresholds: GateThresholds) -> GateVerdict:
    """진단 → 3-way 판정. [제안 — 표준 아님] 임계값 근거는 축 08 실험 결과로만 정당화한다.

    규칙(잠정): ① ess_ratio < hard → "ab_fallback" (무엇을 믿을지 이전에 추정 자체가 불성립).
    ② support proxy·soft ESS·max-weight 위반 → "distrust" (이 로그·이 정책 조합의 추정 기각).
    ③ 그 외 → "trust". 축 09 가 보이듯 이 게이트는 confounding 에 원리적으로 blind — 통과가
    무결의 증명이 아님을 사용처마다 병기할 것.

    thresholds 는 GateThresholds 필수(Hydra 배선 시 OmegaConf → GateThresholds 변환 — CLAUDE.md §2).
    """
    if not isinstance(thresholds, GateThresholds):
        raise TypeError(f"thresholds must be GateThresholds, got {type(thresholds).__name__}")

    reasons: list[str] = []
    if report.ess_ratio < thresholds.ess_ratio_hard:
        reasons.append(f"ess_ratio {report.ess_ratio:.4f} < hard floor {thresholds.ess_ratio_hard}")
        return GateVerdict(decision="ab_fallback", reasons=tuple(reasons))

    if report.support_deficiency > thresholds.support_deficiency_max:
        reasons.append(f"support proxy {report.support_deficiency:.4f} > "
                       f"{thresholds.support_deficiency_max}")
    if report.ess_ratio < thresholds.ess_ratio_soft:
        reasons.append(f"ess_ratio {report.ess_ratio:.4f} < soft floor {thresholds.ess_ratio_soft}")
    if report.max_weight > thresholds.max_weight_cap:
        reasons.append(f"max_weight {report.max_weight:.2f} > cap {thresholds.max_weight_cap}")

    if reasons:
        return GateVerdict(decision="distrust", reasons=tuple(reasons))
    return GateVerdict(decision="trust", reasons=())
