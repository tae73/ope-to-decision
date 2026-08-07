"""비즈니스 임팩트 층 (M6) — funnel DGP·지표 벡터·광고주 재분배. 코어 모듈 불변.

설계(probe M6 GO — probe_funnel_dgp.json):
- **정책은 relevance 스코어 s∈(0,1) 위에서 작동**(softmax(β·s) — 코어와 동일 span), 지표는
  캘리브레이션: p_c = base_ctr + amp_ctr·s(x,a), p_v = base_cvr + amp_cvr·s'(x,a),
  revenue = click·conv·price_a. **기저율과 정책 분리력의 구조적 분리**(기저율-압축 상충 해소).
- 지표 3종: CTR=click · CVR(**세션 기준** = click·conv — ratio 함정 회피, 업계 click-조건부
  CVR 과 다름) · REV. 각 GT: 조건부 평균 해석적 × E_x 는 200k MC(true_policy_value 관례).
- price 는 액션별 고정(구조 rng lognormal — heavy tail 은 K=10 draw 조건부 실현임을 문서화,
  price 벡터는 LEDGER 등재 대상).
- **γ(confounding) 노브 없음 — 영구 금지**: Bernoulli 비선형에서 정확-GT 정리가 깨진다.
  confounding 은 축 09(코어 dgp) 소관 경계.
- 광고주 층: 액션→광고주 매핑(구조 rng). **노출 점유율·HHI 는 OPE 가 아니라 정확 계산**
  (π 가 기지 — 200k MC). subgroup 매출 OPE 는 최소 이벤트 임계(사전 등록) 미달 시
  "not estimable" 정직 반환.
- 리텐션: 이 모듈에 없음 — 세션 간·장기 지표는 single-step bandit OPE 로 식별 불가(RL OPE 소관,
  PLAYBOOK 경계 선언). 세션 내 proxy 도 넣지 않는다(자기기만 위험 — 사용자 확정).
"""

from typing import NamedTuple

import numpy as np

from ope.dgp import _stable_sigmoid
from ope.policies import softmax_policy

MIN_SUBGROUP_EVENTS = 10  # 사전 등록: 광고주 subgroup 매출 OPE 의 최소 (가중) 클릭 이벤트 수
METRICS = ("ctr", "cvr", "rev")


class FunnelConfig(NamedTuple):
    n: int
    n_actions: int
    dim_context: int
    beta_log: float
    beta_eval: float
    base_ctr: float = 0.01
    amp_ctr: float = 0.06
    base_cvr: float = 0.05
    amp_cvr: float = 0.25
    n_advertisers: int = 4
    seed: int = 0
    struct_seed: int = 7


class FunnelBanditData(NamedTuple):
    context: np.ndarray
    action: np.ndarray
    pscore: np.ndarray          # 정확 로깅 propensity
    pi_e_dist: np.ndarray
    rewards: dict               # {"ctr": (n,), "cvr": (n,), "rev": (n,)} — 같은 로그·지표 벡터
    gt: dict                    # {"ctr": float, ...} 지표별 참값 (200k MC)
    price: np.ndarray           # (K,) 액션별 고정 가격 — LEDGER 등재 대상
    advertiser_of: np.ndarray   # (K,) 액션→광고주
    p_c: np.ndarray             # (n, K) — DR 용 oracle 확률 (오지정은 호출측 규약)
    p_v: np.ndarray             # (n, K)


def _structure(cfg: FunnelConfig):
    """구조 rng draw 순서 고정: θ_c → b_c → θ_v → b_v → price → advertiser (순서 변경 금지)."""
    rs = np.random.default_rng(cfg.struct_seed)
    th_c = rs.normal(0.0, 1.0, (cfg.n_actions, cfg.dim_context))
    b_c = rs.normal(0.0, 0.5, cfg.n_actions)
    th_v = rs.normal(0.0, 1.0, (cfg.n_actions, cfg.dim_context))
    b_v = rs.normal(0.0, 0.5, cfg.n_actions)
    price = np.exp(rs.normal(0.0, 1.0, cfg.n_actions))  # lognormal — K개 draw 조건부 heavy tail
    advertiser_of = rs.integers(0, cfg.n_advertisers, cfg.n_actions)
    return th_c, b_c, th_v, b_v, price, advertiser_of


def _rates(x, th_c, b_c, th_v, b_v, cfg):
    s = _stable_sigmoid(x @ th_c.T + b_c)
    p_c = cfg.base_ctr + cfg.amp_ctr * s
    p_v = cfg.base_cvr + cfg.amp_cvr * _stable_sigmoid(x @ th_v.T + b_v)
    return s, p_c, p_v


def funnel_ground_truth(cfg: FunnelConfig, beta: float | None = None,
                        n_mc: int = 200_000) -> dict:
    """지표별 V_m(π) — 조건부 평균 해석적(p_c·p_c p_v·p_c p_v price) × E_x MC."""
    th_c, b_c, th_v, b_v, price, _ = _structure(cfg)
    rng = np.random.default_rng(1_000_003 + cfg.struct_seed)
    x = rng.normal(size=(n_mc, cfg.dim_context))
    s, p_c, p_v = _rates(x, th_c, b_c, th_v, b_v, cfg)
    pi = softmax_policy(s, cfg.beta_eval if beta is None else beta)
    return {"ctr": float((pi * p_c).sum(1).mean()),
            "cvr": float((pi * p_c * p_v).sum(1).mean()),
            "rev": float((pi * p_c * p_v * price).sum(1).mean())}


def make_funnel_bandit_data(cfg: FunnelConfig) -> FunnelBanditData:
    """로그 rng draw 순서 고정: x → action-uniform → click → conv."""
    th_c, b_c, th_v, b_v, price, adv = _structure(cfg)
    rng = np.random.default_rng(cfg.seed)
    x = rng.normal(size=(cfg.n, cfg.dim_context))
    s, p_c, p_v = _rates(x, th_c, b_c, th_v, b_v, cfg)
    pi0 = softmax_policy(s, cfg.beta_log)
    pi_e = softmax_policy(s, cfg.beta_eval)
    a = (pi0.cumsum(1) > rng.random((cfg.n, 1))).argmax(1)
    i = np.arange(cfg.n)
    click = rng.binomial(1, p_c[i, a]).astype(float)
    conv = click * rng.binomial(1, p_v[i, a])
    rev = conv * price[a]
    return FunnelBanditData(context=x, action=a, pscore=pi0[i, a], pi_e_dist=pi_e,
                            rewards={"ctr": click, "cvr": conv.astype(float), "rev": rev},
                            gt=funnel_ground_truth(cfg), price=price, advertiser_of=adv,
                            p_c=p_c, p_v=p_v)


def exposure_shares(cfg: FunnelConfig, beta: float, n_mc: int = 200_000) -> np.ndarray:
    """광고주별 기대 노출 점유율 — π 기지이므로 **정확 계산(OPE 아님)**. (n_advertisers,)"""
    th_c, b_c, _, _, _, adv = _structure(cfg)
    rng = np.random.default_rng(1_000_003 + cfg.struct_seed)
    x = rng.normal(size=(n_mc, cfg.dim_context))
    s = _stable_sigmoid(x @ th_c.T + b_c)
    pi_bar = softmax_policy(s, beta).mean(0)  # (K,)
    shares = np.zeros(cfg.n_advertisers)
    np.add.at(shares, adv, pi_bar)
    return shares


def hhi(shares: np.ndarray) -> float:
    """Herfindahl–Hirschman index — 결정적 arm(오류율 0)."""
    return float((shares**2).sum())


class SubgroupResult(NamedTuple):
    advertiser: int
    estimable: bool
    estimate: float      # not estimable 이면 NaN
    weighted_events: float


def subgroup_revenue_ips(d: FunnelBanditData, n_advertisers: int) -> list[SubgroupResult]:
    """광고주 그룹별 매출 IPS — 최소 이벤트 임계(MIN_SUBGROUP_EVENTS) 미달 시 정직 반환."""
    i = np.arange(len(d.action))
    w = d.pi_e_dist[i, d.action] / d.pscore
    out = []
    for g in range(n_advertisers):
        mask = d.advertiser_of[d.action] == g
        events = float((w * mask * d.rewards["ctr"]).sum())  # 가중 클릭 이벤트
        if events < MIN_SUBGROUP_EVENTS:
            out.append(SubgroupResult(g, False, float("nan"), events))
        else:
            out.append(SubgroupResult(g, True, float((w * mask * d.rewards["rev"]).mean()),
                                      events))
    return out
