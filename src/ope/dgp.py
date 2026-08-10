"""합성 DGP — multi-action contextual bandit, 참 정책가치(ground truth) 보유.

축 01–04·09(+스트레치 13)의 노브가 DGPConfig 필드 하나에 대응한다.
축 05–08·10 은 DGP 가 아니라 estimator/metric 층의 노브다(PLAN.md §2 매핑표 참조).

설계 결정 (M1, Plan 검증 반영 — 근거는 PLAN.md §2):
- **reward 는 연속형**: r = q(x,a) + γ·κ·U + σ·ε (κ=CONF_REWARD_SCALE). E[U]=0·U⊥(π_e 의 행동선택)
  이므로 V(π_e)=E_x[Σ_a π_e q] 가 confounding 강도 γ 와 무관하게 **정확히** 성립한다(정리).
  binary CTR 현실성은 실데이터 축 11–12 의 몫. (probe M0-A 는 Bernoulli 보상이었다 — self-contained
  probe 로서 유효하며, 본구현이 연속형으로 확정한 것은 노브 직교성 때문임을 기록해 둔다.)
- **confounding(축 09)**: 미관측 U~N(0,1)가 로깅 선택(logits += γ·U·d_a)과 보상(+γ·κ·U)에 동시 개입.
  `pscore_logged` = 의도된(배포 설정) 정책 softmax(β_log·q)의 확률 — 실무에서 기록되는 값.
  `pscore_true` = **U-조건부** 실제 정책 π_0(a|x,U)의 확률 — oracle-IPS 가 참값을 복원하는 정의
  (U-주변화 propensity 는 불편성을 복원하지 못한다 — 정의 혼동 금지). γ=0 ⇒ 두 값 동일.
- **support deficiency(축 04)**: 컨텍스트별 **하위-q ⌊δK⌋개 액션**의 로깅 지지를 구조적으로 제거
  (per-row 랜덤 mask 는 진단 proxy 를 원리적으로 blind 으로 만든다 — Plan 검증 지적).
  mask 는 배포 설정이므로 기록 pscore 에도 반영, π_e 는 불변. δ 스윕은 1/K 단위로 양자화된다.
"""

from typing import NamedTuple

import numpy as np

from ope.policies import softmax_policy

CONF_REWARD_SCALE = 0.5  # κ: U 가 보상에 개입하는 고정 스케일 (γ 하나로 개입 강도를 통제하기 위한 상수)


class DGPConfig(NamedTuple):
    n: int                      # 로그 표본 수 (축 01)
    n_actions: int              # K (스트레치 축 13)
    dim_context: int
    beta_log: float             # 로깅 softmax inverse-temperature — overlap 다이얼 (축 02)
    beta_eval: float            # 평가 정책 온도 — 타깃-로깅 괴리 (축 03)
    support_deficiency: float   # 컨텍스트별 로깅 지지 제거 비율 δ (축 04; ⌊δK⌋개 하위-q 액션)
    reward_noise: float         # 보상 noise σ (축 06 보조)
    confounding_strength: float # γ: U 개입 강도; 0 이면 unconfounded (축 09)
    seed: int                   # 로그 표본 rng (환경 구조와 분리)
    struct_seed: int = 7        # 환경 구조 rng — "같은 환경, 다른 로그" MC 반복용


class SyntheticBanditData(NamedTuple):
    context: np.ndarray        # (n, d)
    action: np.ndarray         # (n,)
    reward: np.ndarray         # (n,)  연속형 (설계 결정 참조)
    pscore_logged: np.ndarray  # (n,)  기록된 propensity — estimator 가 보는 값 (의도된 정책 + mask)
    pscore_true: np.ndarray    # (n,)  진짜(U-조건부) propensity — 축 09 oracle 전용 (estimator 비공개)
    pi_e_dist: np.ndarray      # (n, K)
    q_true: np.ndarray         # (n, K) 참 기대보상 E[r|x,a,U=E[U]] = q(x,a) — GT/oracle 전용
    v_true: float              # V(π_e) ground truth (struct_seed 파생 대규모 MC)
    pi_log_dist: np.ndarray = None  # (n, K) 기록된(의도·배포 설정) 로깅 정책 전체 분포 — **로그 층**
    #   (M8 추가) 실무자의 "내가 배포한 정책은 스코어할 수 있다" 자산. pscore_logged 의 행렬판이며
    #   oracle 이 아니다: pscore_logged = pi_log_dist[i, a_i] 항등(테스트 고정). rng 무영향(기계산 값 반환).


def _stable_sigmoid(z: np.ndarray) -> np.ndarray:
    return np.exp(-np.logaddexp(0.0, -z))


def _make_structure(config: DGPConfig) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """환경 구조 (θ, b, d_a). rng **draw 순서 고정: θ → b → d_a** — 순서 변경은 같은
    struct_seed 의 환경을 조용히 바꾸므로 금지(변경 필요 시 struct_seed 의미 재정의로 문서화)."""
    rng = np.random.default_rng(config.struct_seed)
    theta = rng.normal(0.0, 1.0, size=(config.n_actions, config.dim_context))
    b = rng.normal(0.0, 0.5, size=config.n_actions)
    d = rng.choice(np.array([-1.0, 1.0]), size=config.n_actions)  # confounding 방향 벡터
    return theta, b, d


def _q_true(x: np.ndarray, theta: np.ndarray, b: np.ndarray) -> np.ndarray:
    return _stable_sigmoid(x @ theta.T + b)


def _support_mask(q: np.ndarray, deficiency: float) -> np.ndarray:
    """컨텍스트별 하위-q ⌊δK⌋개 액션의 로깅 지지 제거. True=지지."""
    n, k = q.shape
    if not 0.0 <= deficiency < 1.0:
        raise ValueError(f"support_deficiency must be in [0, 1), got {deficiency}")
    m = int(np.floor(deficiency * k))
    if m == 0:
        return np.ones((n, k), dtype=bool)
    if m >= k:
        raise ValueError("support_deficiency removes all actions")
    order = np.argsort(q, axis=1)  # 오름차순 — 앞 m 개가 하위-q
    mask = np.ones((n, k), dtype=bool)
    np.put_along_axis(mask, order[:, :m], False, axis=1)
    return mask


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, logits, -np.inf)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)  # exp(-inf) = 0
    return e / e.sum(axis=1, keepdims=True)


def make_synthetic_bandit_data(config: DGPConfig) -> SyntheticBanditData:
    """DGPConfig → 로그 + ground truth. 고정 seed 재현.

    로그 rng draw 순서 고정: x → U → action-uniform → reward-noise (순서 변경 = 재현성 파괴).
    """
    theta, b, d = _make_structure(config)
    rng = np.random.default_rng(config.seed)

    x = rng.normal(size=(config.n, config.dim_context))
    q = _q_true(x, theta, b)
    mask = _support_mask(q, config.support_deficiency)
    pi_e = softmax_policy(q, config.beta_eval)

    logits_intended = config.beta_log * q
    pi_log_recorded = _masked_softmax(logits_intended, mask)

    u = rng.normal(size=config.n)
    logits_actual = logits_intended + config.confounding_strength * u[:, None] * d[None, :]
    pi_log_actual = _masked_softmax(logits_actual, mask)

    action = (pi_log_actual.cumsum(axis=1) > rng.random((config.n, 1))).argmax(axis=1)
    idx = np.arange(config.n)

    noise = rng.normal(size=config.n)
    reward = (q[idx, action]
              + CONF_REWARD_SCALE * config.confounding_strength * u
              + config.reward_noise * noise)

    return SyntheticBanditData(
        context=x,
        action=action,
        reward=reward,
        pscore_logged=pi_log_recorded[idx, action],
        pscore_true=pi_log_actual[idx, action],
        pi_e_dist=pi_e,
        q_true=q,
        v_true=true_policy_value(config),
        pi_log_dist=pi_log_recorded,
    )


def true_policy_value(config: DGPConfig, n_mc: int = 200_000) -> float:
    """V(π_e) = E_x[Σ_a π_e(a|x) q(x,a)] — struct_seed 파생 대규모 MC (로그 seed 와 독립).

    연속형 reward 설계에서 U·ε 항은 평균 0 이므로 이 값이 confounding·noise 와 무관한 참값이다(정리).
    """
    theta, b, _ = _make_structure(config)
    rng = np.random.default_rng(1_000_003 + config.struct_seed)
    x = rng.normal(size=(n_mc, config.dim_context))
    q = _q_true(x, theta, b)
    pi_e = softmax_policy(q, config.beta_eval)
    return float((pi_e * q).sum(axis=1).mean())


U_TRUNC = 8.0  # marginal_logging_dist 의 Gaussian 절단 반경 — ∫_{|u|>8} φ ≈ 1.2e-15 (행 정규화가 흡수)


def marginal_logging_dist(config: DGPConfig, context: np.ndarray,
                          n_nodes: int = 800) -> np.ndarray:
    """U-주변화 실제 로깅 분포 P(a|x) = E_U[masked_softmax(β_log·q(x) + γ·U·d, mask)] — (n, K).

    **M8 축 18 전용 사후 순수함수** (probe M8-B 에서 원형 검증 — GO). `make_synthetic_bandit_data`
    의 rng 를 일절 소비하지 않아 기존 축 01–16 산출이 불변이다(tests/test_dgp.py checksum 고정).
    구적은 Gauss–Legendre × φ(u)(절단 ±U_TRUNC) — rng 자체가 불필요한 결정적 계산.
    numpy `hermegauss` 는 degree ≥ ~400 에서 overflow 하므로 쓰지 않는다(probe M8-B 실측).

    이 분포를 기록 pscore 로 쓰면 "기록 = 참 marginal" 인 **관측 동등성 세계**가 된다 —
    그 세계에서 어떤 로그 통계도 confounding 을 구별할 수 없다(PLAN §3.5 원칙).
    """
    theta, b, d = _make_structure(config)
    q = _q_true(context, theta, b)
    mask = _support_mask(q, config.support_deficiency)
    logits = config.beta_log * q
    nodes, wts = np.polynomial.legendre.leggauss(n_nodes)
    u_nodes = nodes * U_TRUNC
    wts = wts * U_TRUNC * np.exp(-0.5 * u_nodes ** 2) / np.sqrt(2.0 * np.pi)
    acc = np.zeros_like(q)
    for u, wt in zip(u_nodes, wts):
        acc += wt * _masked_softmax(
            logits + config.confounding_strength * u * d[None, :], mask)
    return acc / acc.sum(axis=1, keepdims=True)
