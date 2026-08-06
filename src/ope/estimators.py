"""OPE estimator 코어 7종 + bootstrap CI — 전부 순수 numpy 함수.

공통 표기 (shape 규약):
    reward        : (n,)   관측 보상 r_i
    action        : (n,)   로깅 정책이 고른 행동 인덱스 a_i ∈ {0..K-1}
    pscore        : (n,)   로깅 propensity π_0(a_i|x_i) — "기록값" (진짜와 다를 수 있음: 축 09)
    pi_e_dist     : (n, K) 평가 정책의 행동 분포 π_e(·|x_i)
    q_hat         : (n, K) reward 모델 예측 q̂(x_i, ·)

수치 검산 3중: probe M0-A(sanity) → probe M0-B + experiments/m1_crossval(obp/sb-obp 산술 일치)
→ tests/ property test. 공식 출처: DM/IPS/DR Dudík-Langford-Li 2011(arXiv:1103.4601),
SNIPS Swaminathan-Joachims 2015, Switch-DR Wang-Agarwal-Dudík 2017(arXiv:1612.01205),
DRos Su+ 2020(arXiv:1907.09623).
"""

from typing import NamedTuple

import numpy as np


class EstimateResult(NamedTuple):
    """단일 estimator 의 점추정 결과.

    weights 는 estimator 가 실제 사용한 (변형 후) importance weight — diagnostics 의 입력.
    DM 은 weight 를 쓰지 않으므로 빈 배열(길이 0)을 반환한다(진단에 흘리지 말 것).
    """

    value: float
    weights: np.ndarray


def _raw_weights(action: np.ndarray, pscore: np.ndarray, pi_e_dist: np.ndarray) -> np.ndarray:
    if np.any(pscore <= 0.0):
        raise ValueError("pscore must be strictly positive for logged actions")
    return pi_e_dist[np.arange(len(action)), action] / pscore


def estimate_dm(pi_e_dist: np.ndarray, q_hat: np.ndarray) -> EstimateResult:
    """Direct Method: V̂ = mean_i Σ_a π_e(a|x_i) q̂(x_i,a). 저분산·모델 bias 극단."""
    value = float((pi_e_dist * q_hat).sum(axis=1).mean())
    return EstimateResult(value=value, weights=np.empty(0))


def estimate_ips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                 pi_e_dist: np.ndarray) -> EstimateResult:
    """Inverse Propensity Scoring: w_i = π_e(a_i|x_i)/π_0(a_i|x_i), V̂ = mean(w·r). unbiased·고분산."""
    w = _raw_weights(action, pscore, pi_e_dist)
    return EstimateResult(value=float((w * reward).mean()), weights=w)


def estimate_snips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                   pi_e_dist: np.ndarray) -> EstimateResult:
    """Self-Normalized IPS: V̂ = Σ(w·r)/Σw. weights 는 raw w (정규화는 값 계산에서만)."""
    w = _raw_weights(action, pscore, pi_e_dist)
    w_sum = w.sum()
    if w_sum <= 0.0:
        raise ValueError("sum of importance weights is zero — pi_e has no mass on logged actions")
    return EstimateResult(value=float((w * reward).sum() / w_sum), weights=w)


def estimate_clipped_ips(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                         pi_e_dist: np.ndarray, lam: float) -> EstimateResult:
    """Clipped IPS: w ← min(w, λ). λ = bias-variance 다이얼 (dag-registry diagnostics.clipping 대응)."""
    if lam <= 0.0:
        raise ValueError(f"lam must be > 0, got {lam}")
    w = np.minimum(_raw_weights(action, pscore, pi_e_dist), lam)
    return EstimateResult(value=float((w * reward).mean()), weights=w)


def estimate_dr(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                pi_e_dist: np.ndarray, q_hat: np.ndarray) -> EstimateResult:
    """Doubly Robust: DM baseline + IPS 보정항. 두 모델 중 하나만 맞아도 consistent."""
    w = _raw_weights(action, pscore, pi_e_dist)
    dm = (pi_e_dist * q_hat).sum(axis=1).mean()
    resid = reward - q_hat[np.arange(len(action)), action]
    return EstimateResult(value=float(dm + (w * resid).mean()), weights=w)


def estimate_switch_dr(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                       pi_e_dist: np.ndarray, q_hat: np.ndarray, tau: float) -> EstimateResult:
    """Switch-DR: w_i > τ 구간은 보정항을 끄고 DM 에 맡긴다 (Wang-Agarwal-Dudík 2017).

    V̂ = DM + mean(w·1[w≤τ]·(r−q̂)). weights = w·1[w≤τ].
    """
    if tau <= 0.0:
        raise ValueError(f"tau must be > 0, got {tau}")
    w = _raw_weights(action, pscore, pi_e_dist)
    w_switch = w * (w <= tau)
    dm = (pi_e_dist * q_hat).sum(axis=1).mean()
    resid = reward - q_hat[np.arange(len(action)), action]
    return EstimateResult(value=float(dm + (w_switch * resid).mean()), weights=w_switch)


def estimate_dros(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                  pi_e_dist: np.ndarray, q_hat: np.ndarray, lam: float) -> EstimateResult:
    """DR with Optimistic Shrinkage: w_s = λw/(w²+λ) (Su+ 2020). 'clipping 의 원리화'.

    λ→0⁺ 에서 DM, λ→∞ 에서 DR 로 수렴. λ>0 이면 w=0 에서도 0/0 없음.
    """
    if lam <= 0.0:
        raise ValueError(f"lam must be > 0, got {lam}")
    w = _raw_weights(action, pscore, pi_e_dist)
    w_shrink = lam * w / (w**2 + lam)
    dm = (pi_e_dist * q_hat).sum(axis=1).mean()
    resid = reward - q_hat[np.arange(len(action)), action]
    return EstimateResult(value=float(dm + (w_shrink * resid).mean()), weights=w_shrink)


class SlopeResult(NamedTuple):
    """SLOPE hyperparameter 선택 결과 (Su-Srinath-Krishnamurthy, ICML 2020)."""

    index: int              # 선택된 ladder 인덱스 ĵ
    hyperparam: float       # ladder[ĵ]
    value: float            # θ̂_ĵ
    values: np.ndarray      # (M,) ladder 별 추정치
    ses: np.ndarray         # (M,) ladder 별 SE (per-sample 기여항 std/√n)


def _lepski_select(values: np.ndarray, ses: np.ndarray, multiplier: float = 2.0) -> int:
    """Lepski 선택 규칙: ĵ = max{j : I_j 가 **모든** i<j 의 I_i 와 교차}, I=θ̂±c·SE.

    인접쌍만 검사하면 안 된다(전 선행자 pairwise — SLOPE 상수 보장의 원천). j=0 은 자명하게 만족.
    **호출자 책임(방향)**: 입력은 SLOPE 논문 정렬 — index 0 = 저편향·광폭 CI(미규제), index 증가 =
    고편향·협폭 CI(강규제) — 이어야 한다. 반대로 넣으면 협폭 rung 이 광폭 rung 을 거부해 최악
    rung 으로 붕괴한다(M2 축 07 이 실증한 초기 구현 버그 — 회귀 테스트로 고정).
    """
    j_hat = 0
    for j in range(1, len(values)):
        ok = all(abs(values[j] - values[i]) <= multiplier * (ses[j] + ses[i])
                 for i in range(j))
        if ok:
            j_hat = j
    return j_hat


def slope_select(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                 pi_e_dist: np.ndarray, q_hat: np.ndarray | None,
                 estimator: str, ladder: np.ndarray,
                 multiplier: float = 2.0) -> SlopeResult:
    """SLOPE 로 clipping/switch/shrinkage hyperparameter 를 데이터 기반 선택.

    estimator ∈ {clipped_ips, switch_dr, dros}. 입력 **ladder 는 오름차순**(관례 유지)이되,
    내부에서는 SLOPE/Lepski 의 논문 정렬 — **큰 λ/τ(미규제·저편향·광폭 CI)부터 작은 값
    (강규제·고편향·협폭 CI) 순** — 으로 걷는다. 방향을 반대로 걸으면 고편향 rung 의 협폭 CI 가
    나머지를 거부해 최악 rung 으로 붕괴한다(M2 축 07 실증 — 초기 구현 버그, 수정·회귀 고정).
    선택 결과는 "모든 덜-규제된 rung 과 정합한 한도 내에서 가장 강한 규제" = 분산 최소화.
    SE 는 per-sample 기여항의 std/√n (asymptotic), CI 는 ±multiplier·SE (paper 의 inflation=2).
    한계(문서화): heavy-tail weight 구간에서 per-sample SE 는 과소평가될 수 있어 선택이
    공격적으로 치우칠 수 있다 — 축 07 에서 실증. empirical SE 의 단조 역전은 교정하지 않고
    그대로 둔다(기록만). ladder 에 DM 극한(λ→0)은 포함하지 않는다 — 회복 하한은 ladder 최소값이다.
    """
    if estimator not in ("clipped_ips", "switch_dr", "dros"):
        raise ValueError(f"slope_select supports clipped/switch/dros, got {estimator!r}")
    ladder = np.asarray(ladder, dtype=float)
    if len(ladder) == 0 or np.any(np.diff(ladder) <= 0):
        raise ValueError("ladder must be non-empty and strictly ascending")
    if np.any(ladder <= 0):
        raise ValueError("ladder values must be > 0")
    if estimator in ("switch_dr", "dros") and q_hat is None:
        raise ValueError(f"{estimator} requires q_hat")

    n = len(reward)
    w = _raw_weights(action, pscore, pi_e_dist)
    if q_hat is not None:
        dm_terms = (pi_e_dist * q_hat).sum(axis=1)
        resid = reward - q_hat[np.arange(n), action]

    values = np.empty(len(ladder))
    ses = np.empty(len(ladder))
    for i, h in enumerate(ladder):
        if estimator == "clipped_ips":
            terms = np.minimum(w, h) * reward
        elif estimator == "switch_dr":
            terms = dm_terms + w * (w <= h) * resid
        else:  # dros
            terms = dm_terms + (h * w / (w**2 + h)) * resid
        values[i] = terms.mean()
        ses[i] = terms.std(ddof=1) / np.sqrt(n)

    # 논문 정렬(광폭 CI = 큰 hyperparam 먼저)로 뒤집어 Lepski — 결과 인덱스를 원 ladder 로 환산
    j_desc = _lepski_select(values[::-1], ses[::-1], multiplier)
    j_hat = len(ladder) - 1 - j_desc
    return SlopeResult(index=j_hat, hyperparam=float(ladder[j_hat]),
                       value=float(values[j_hat]), values=values, ses=ses)


_POINT_ESTIMATORS = {
    "dm": lambda r, a, ps, pe, q, h: estimate_dm(pe, q),
    "ips": lambda r, a, ps, pe, q, h: estimate_ips(r, a, ps, pe),
    "snips": lambda r, a, ps, pe, q, h: estimate_snips(r, a, ps, pe),
    "clipped_ips": lambda r, a, ps, pe, q, h: estimate_clipped_ips(r, a, ps, pe, h),
    "dr": lambda r, a, ps, pe, q, h: estimate_dr(r, a, ps, pe, q),
    "switch_dr": lambda r, a, ps, pe, q, h: estimate_switch_dr(r, a, ps, pe, q, h),
    "dros": lambda r, a, ps, pe, q, h: estimate_dros(r, a, ps, pe, q, h),
}


def bootstrap_ci(reward: np.ndarray, action: np.ndarray, pscore: np.ndarray,
                 pi_e_dist: np.ndarray, q_hat: np.ndarray | None,
                 estimator: str, n_boot: int, alpha: float, seed: int,
                 hyperparam: float | None = None) -> tuple[float, float]:
    """percentile bootstrap CI — 모든 figure 는 점추정이 아니라 구간을 병기한다(OBD 근사참값 규약).

    estimator ∈ {dm, ips, snips, clipped_ips, dr, switch_dr, dros}.
    clipped_ips/switch_dr/dros 는 hyperparam(λ/τ) 필수. 행 단위 재표집(고정 seed).
    alpha 는 **유의수준**이다: 0.05 → 95% CI ([α/2, 1−α/2] quantile, obp 규약 동일).
    신뢰수준(0.95)으로 오인해 넘기면 조용히 5% 구간이 나오므로 범위를 강제한다.
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1) — significance level (0.05 → 95% CI), got {alpha}")
    if estimator not in _POINT_ESTIMATORS:
        raise ValueError(f"unknown estimator {estimator!r}")
    if estimator in ("clipped_ips", "switch_dr", "dros") and hyperparam is None:
        raise ValueError(f"{estimator} requires hyperparam")
    if estimator in ("dm", "dr", "switch_dr", "dros") and q_hat is None:
        raise ValueError(f"{estimator} requires q_hat")
    fn = _POINT_ESTIMATORS[estimator]
    rng = np.random.default_rng(seed)
    n = len(reward)
    values = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        q_b = q_hat[idx] if q_hat is not None else None
        values[i] = fn(reward[idx], action[idx], pscore[idx], pi_e_dist[idx], q_b, hyperparam).value
    lo, hi = np.quantile(values, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)
