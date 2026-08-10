"""GT-free validity battery — **필요조건 검사(falsifier)** [제안 — 표준 아님].

PLAN §3.5-1 사전등록 정의의 실행 코드(원형: probe M8-A — GO). gate v1(`diagnostics.py`)과
**독립·병렬**이며 v1 은 불가침이다. 모든 통계는 로그 층(reward·action·pscore·pi_e_dist —
선택적으로 context·q̂)만 입력받는다 — 참값·oracle 층 인자는 시그니처에 존재하지 않으며,
계약 테스트가 **소스 텍스트 수준**에서 oracle 식별자를 ban 한다.

정직성 경계(항상 함께 전시 — CLAUDE.md §5 co-exhibit):
- **통과는 무결의 증명이 아니다.** 기록 pscore 가 marginally calibrated 인 confounding 은
  관측 동등성에 의해 **어떤 로그 통계로도 원리적으로 구별 불가**(축 18 이 구성적으로 전시,
  probe M8-B 실증) — 그 몫은 Λ-감도 구간(`lambda_star_vs_anchor`·축 14 도구)이 담당한다.
- **agreement 는 validity 증거가 아니다** — 모든 estimator 가 일관되게 틀릴 때
  disagreement 는 가장 작다.

gate arm 4종(임계는 ValidityConfig 기본값 = §3.5-1 사전등록값 — 축 17 에서 평가만, 무교정):
mean_w(E[w]=1 HT 항등) · harmonic(per-action T(a*)=mean(1[a=a*]/pscore), E[T]=1 — naive A/A 의
비-vacuous 대체) · placebo(분석가 생성 독립 noise 보상, 참값 0) · disagreement(weighting 계열
스프레드, λ_clip=p90(w) 고정·스케일 바닥 시 inconclusive).
보고 전용: dr_correction=|mean(w·(r−q̂))| · nc_covariate=max_j|mean(w·z_j)−mean(z_j)|.
"""

from typing import NamedTuple

import numpy as np

from ope.estimators import (
    estimate_clipped_ips,
    estimate_ips,
    estimate_snips,
    msm_snips_bounds,
)


class ValidityConfig(NamedTuple):
    n_boot: int = 500              # joint bootstrap 반복 (사전등록 B)
    alpha: float = 0.05
    seed: int = 0                  # placebo ε·bootstrap 인덱스의 SeedSequence 루트
    mean_w_tol: float = 0.10       # [제안] |E[w]−1| 실질 허용 (대표본 자명 기각 방지)
    harmonic_tol: float = 0.25     # [제안] |T(a*)−1| 실질 허용
    disagreement_tol: float = 0.50 # [제안] weighting 계열 정규화 스프레드 허용
    min_action_count: int = 30     # harmonic 대상 action 최소 로그 출현 수
    disagreement_floor: float = 0.01  # |SNIPS| < floor·max(|mean r|,eps) ⇒ inconclusive
    placebo_scale: float = 1.0     # ε ~ N(0, scale·std(r))


class ArmResult(NamedTuple):
    value: float                   # 통계값 (inconclusive 이면 nan 일 수 있음)
    ci_lo: float                   # CI 없는 arm(disagreement)은 nan
    ci_hi: float
    state: str                     # "pass" | "fail" | "inconclusive"


class ValidityReport(NamedTuple):
    mean_w: ArmResult
    harmonic: ArmResult            # value = |T−1| 최대 대상 action 의 T
    placebo: ArmResult
    disagreement: ArmResult
    harmonic_by_action: tuple      # ((action, T, ci_lo, ci_hi, n_a, fail), ...) — 대상 action 만
    dr_correction: float           # 보고 전용 — q̂ 미제공 시 nan
    nc_covariate: float            # 보고 전용 — context 미제공 시 nan
    checks_failed: tuple           # fail 상태 arm 이름 튜플


class JointBootstrap(NamedTuple):
    estimates: dict                # estimator 이름 → (B,) replicate 값
    mean_w: np.ndarray             # (B,)
    placebo: np.ndarray            # (B,)
    harmonic: np.ndarray           # (B, K)
    placebo_reward: np.ndarray     # (n,) — 점추정·재현에 필요한 ε (분석가 생성물, oracle 아님)


class LambdaStar(NamedTuple):
    lam_star: float
    censored: bool                 # lam_max 에서도 결론 불역전 ⇒ True (lam_star = lam_max)
    direction: str                 # "above" | "below" | "at_anchor" (SNIPS vs anchor)


def _rngs(cfg: ValidityConfig) -> tuple[np.random.Generator, np.random.Generator]:
    """(placebo rng, bootstrap rng) — SeedSequence spawn 으로 스트림 분리(draw 순서 문서화)."""
    kids = np.random.SeedSequence(cfg.seed).spawn(2)
    return np.random.default_rng(kids[0]), np.random.default_rng(kids[1])


def make_placebo_reward(reward: np.ndarray, cfg: ValidityConfig) -> np.ndarray:
    """분석가가 자체 생성하는 음성 대조 보상 ε ~ N(0, placebo_scale·std(r)) — 구성상 참값 0."""
    rng, _ = _rngs(cfg)
    return rng.normal(0.0, cfg.placebo_scale * float(np.std(reward)), size=len(reward))


def bootstrap_joint(reward, action, pscore, pi_e_dist, q_hat=None,
                    hypers=None, cfg: ValidityConfig = ValidityConfig(),
                    placebo_reward=None) -> JointBootstrap:
    """joint bootstrap — **같은 재표집 인덱스**에서 estimator 와 battery 통계를 동시 계산(paired).

    replicate 는 사전 계산한 per-row 기여 벡터의 재표집 평균이라 per-estimator 재계산 대비
    ~14× 저렴(PLAN §3.5-1). hyperparam(τ·λ)은 전체 로그에서 1회 계산 후 재표집 간 고정
    (축 12 규약 계승 — CI 는 고정-hyperparam 조건부). q̂ 미제공 시 weighting 3종만.
    """
    w = estimate_ips(reward, action, pscore, pi_e_dist).weights  # 검증 포함 raw weight
    n, k = pi_e_dist.shape
    if hypers is None:
        hypers = {"tau": float(np.quantile(w, 0.95)),
                  "lam_clip": float(np.quantile(w, 0.90)),
                  "lam_dros": float(np.quantile(w, 0.90)) ** 2}
    if placebo_reward is None:
        placebo_reward = make_placebo_reward(reward, cfg)
    _, rng = _rngs(cfg)

    # per-row 기여 벡터 (replicate = 재표집 평균 — 정확 항등은 test 의 brute-force 대조로 고정)
    rows = {"ips": w * reward,
            "clipped_ips": np.minimum(w, hypers["lam_clip"]) * reward}
    if q_hat is not None:
        idx = np.arange(n)
        q_a = q_hat[idx, action]
        qbar = (pi_e_dist * q_hat).sum(axis=1)
        resid = reward - q_a
        w_sw = w * (w <= hypers["tau"])
        w_sh = hypers["lam_dros"] * w / (w ** 2 + hypers["lam_dros"])
        rows.update({"dm": qbar, "dr": qbar + w * resid,
                     "switch_dr": qbar + w_sw * resid, "dros": qbar + w_sh * resid})
    wr = w * reward
    weps = w * placebo_reward
    inv_ps = 1.0 / pscore

    b_est = {name: np.empty(cfg.n_boot) for name in rows}
    b_est["snips"] = np.empty(cfg.n_boot)
    b_mw = np.empty(cfg.n_boot)
    b_pl = np.empty(cfg.n_boot)
    b_hm = np.empty((cfg.n_boot, k))
    for b in range(cfg.n_boot):
        ii = rng.integers(0, n, n)
        for name, vec in rows.items():
            b_est[name][b] = vec[ii].mean()
        wsum = w[ii].sum()
        b_est["snips"][b] = wr[ii].sum() / wsum
        b_mw[b] = wsum / n
        b_pl[b] = weps[ii].mean()
        b_hm[b] = np.bincount(action[ii], weights=inv_ps[ii], minlength=k) / n
    return JointBootstrap(estimates=b_est, mean_w=b_mw, placebo=b_pl,
                          harmonic=b_hm, placebo_reward=placebo_reward)


def _ci(samples: np.ndarray, alpha: float) -> tuple[float, float]:
    lo, hi = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def run_validity_checks(reward, action, pscore, pi_e_dist, q_hat=None, context=None,
                        cfg: ValidityConfig = ValidityConfig(),
                        boot: JointBootstrap = None) -> ValidityReport:
    """battery 실행 — PLAN §3.5-1 fail 규칙 그대로. `boot` 재사용 시 이중 계산 회피.

    fail 규칙(사전등록): mean_w ⇔ CI 1 제외 ∧ |mean_w−1| > tol / harmonic ⇔ 대상 action 중
    하나라도 CI 1 제외 ∧ |T−1| > tol / placebo ⇔ CI 0 제외 / disagreement ⇔ > tol
    (단 |SNIPS| < floor·max(|mean r|, eps) ⇒ inconclusive).
    """
    if not isinstance(cfg, ValidityConfig):
        raise TypeError("cfg must be ValidityConfig (raw dict 는 오타 키를 침묵 통과시킨다)")
    if boot is None:
        boot = bootstrap_joint(reward, action, pscore, pi_e_dist,
                               q_hat=q_hat, cfg=cfg)
    w = estimate_ips(reward, action, pscore, pi_e_dist).weights
    n, k = pi_e_dist.shape

    mw = float(w.mean())
    lo, hi = _ci(boot.mean_w, cfg.alpha)
    mw_fail = (lo > 1.0 or hi < 1.0) and abs(mw - 1.0) > cfg.mean_w_tol
    mean_w_arm = ArmResult(mw, lo, hi, "fail" if mw_fail else "pass")

    n_a = np.bincount(action, minlength=k)
    t_point = np.bincount(action, weights=1.0 / pscore, minlength=k) / n
    by_action, worst = [], None
    harm_fail = False
    for a in range(k):
        if n_a[a] < cfg.min_action_count:
            continue
        alo, ahi = _ci(boot.harmonic[:, a], cfg.alpha)
        fail = (alo > 1.0 or ahi < 1.0) and abs(t_point[a] - 1.0) > cfg.harmonic_tol
        by_action.append((a, float(t_point[a]), alo, ahi, int(n_a[a]), bool(fail)))
        harm_fail = harm_fail or fail
        if worst is None or abs(t_point[a] - 1.0) > abs(worst[1] - 1.0):
            worst = by_action[-1]
    if worst is None:  # 대상 action 없음 (극소 로그) — 판정 불능
        harmonic_arm = ArmResult(float("nan"), float("nan"), float("nan"), "inconclusive")
    else:
        harmonic_arm = ArmResult(worst[1], worst[2], worst[3],
                                 "fail" if harm_fail else "pass")

    pl = float((w * boot.placebo_reward).mean())
    plo, phi = _ci(boot.placebo, cfg.alpha)
    pl_fail = plo > 0.0 or phi < 0.0
    placebo_arm = ArmResult(pl, plo, phi, "fail" if pl_fail else "pass")

    ips = estimate_ips(reward, action, pscore, pi_e_dist).value
    snips = estimate_snips(reward, action, pscore, pi_e_dist).value
    lam_clip = float(np.quantile(w, 0.90))
    clip = estimate_clipped_ips(reward, action, pscore, pi_e_dist, lam_clip).value
    floor = cfg.disagreement_floor * max(abs(float(reward.mean())), 1e-12)
    if abs(snips) < floor:
        disagreement_arm = ArmResult(float("nan"), float("nan"), float("nan"), "inconclusive")
    else:
        vals = (ips, snips, clip)
        disag = float((max(vals) - min(vals)) / max(abs(snips), 1e-12))
        disagreement_arm = ArmResult(disag, float("nan"), float("nan"),
                                     "fail" if disag > cfg.disagreement_tol else "pass")

    idx = np.arange(n)
    dr_corr = float("nan") if q_hat is None else float(
        abs((w * (reward - q_hat[idx, action])).mean()))
    if context is None:
        nc = float("nan")
    else:
        z = (context - context.mean(axis=0)) / np.maximum(context.std(axis=0), 1e-12)
        nc = float(np.max(np.abs((w[:, None] * z).mean(axis=0) - z.mean(axis=0))))

    failed = tuple(name for name, arm in
                   (("mean_w", mean_w_arm), ("harmonic", harmonic_arm),
                    ("placebo", placebo_arm), ("disagreement", disagreement_arm))
                   if arm.state == "fail")
    return ValidityReport(mean_w=mean_w_arm, harmonic=harmonic_arm, placebo=placebo_arm,
                          disagreement=disagreement_arm, harmonic_by_action=tuple(by_action),
                          dr_correction=dr_corr, nc_covariate=nc, checks_failed=failed)


def refit_gap(pscore: np.ndarray, pscore_refit: np.ndarray) -> float:
    """보고 전용(§3.5-1): crossfit π̂₀ 재적합값 vs 기록 pscore 의 평균 절대 상대 괴리.

    기록이 같은 절차의 refit 산물이면 준-null 이 되는 함정(§3.5-1) — 게이트 비발화·보고만.
    """
    pscore = np.asarray(pscore, dtype=float)
    if np.any(pscore <= 0):
        raise ValueError("pscore must be positive")
    return float(np.mean(np.abs(np.asarray(pscore_refit, dtype=float) - pscore) / pscore))


def time_split_gap(reward, action, pscore, pi_e_dist, order) -> float:
    """보고 전용(§3.5-1): 시간순 전/후반 SNIPS 추정 gap — nonstationarity/drift 전용 신호.

    order 는 시간 오름차순 정렬 인덱스(타임스탬프 보유 로그 한정). 게이트 비발화·보고만.
    """
    order = np.asarray(order)
    half = len(order) // 2
    a, b = order[:half], order[half:]
    va = estimate_snips(reward[a], action[a], pscore[a], pi_e_dist[a]).value
    vb = estimate_snips(reward[b], action[b], pscore[b], pi_e_dist[b]).value
    return float(vb - va)


def lambda_star_vs_anchor(reward, action, pscore, pi_e_dist, anchor: float,
                          lam_max: float = 8.0, iters: int = 60) -> LambdaStar:
    """결론(SNIPS vs anchor 의 부호)이 뒤집힐 수 있는 최소 Λ — **Λ\\*_flip** (GLOSSARY §8).

    축 14 `_lam_star`(두 후보 밴드 겹침)의 결정-층 일반화: 후보의 MSM 밴드가 anchor
    (통상 incumbent = mean(r))를 포함하기 시작하는 최소 Λ 를 log-bisection 으로 찾는다.
    Λ 는 데이터에서 식별되지 않는 가정 스케일이다 — Λ\\* 는 robustness 인증서가 아니라
    "결론이 얼마나 작은 감도 가정에서 무너지는가"의 보고서다(축 14 프레임 계승).
    """
    snips = estimate_snips(reward, action, pscore, pi_e_dist).value
    if snips == anchor:
        return LambdaStar(1.0, False, "at_anchor")
    direction = "above" if snips > anchor else "below"

    def crossed(lam: float) -> bool:
        lo, hi = msm_snips_bounds(reward, action, pscore, pi_e_dist, lam)
        return lo <= anchor if direction == "above" else hi >= anchor

    if not crossed(lam_max):
        return LambdaStar(float(lam_max), True, direction)
    lo_l, hi_l = np.log(1.0), np.log(lam_max)
    for _ in range(iters):
        mid = 0.5 * (lo_l + hi_l)
        if crossed(float(np.exp(mid))):
            hi_l = mid
        else:
            lo_l = mid
    return LambdaStar(float(np.exp(hi_l)), False, direction)
