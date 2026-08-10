"""로그에서 적합하는 모델 층 — GT-미상 practitioner 트랙(M8, PLAN §3.5)용.

실전 로그에는 참 propensity 도 참 기대보상도 없다 — q̂·π̂0 를 로그 (x, a, r) 자체에서
적합해야 한다. 여기서는 전부 **cross-fitted**(out-of-fold) 로 적합해 in-sample
자동보정(축 05 estimated 모드의 준-null 함정)을 회피한다.
**`ope.dgp` import 금지** — oracle 누수 차단(계약 테스트 tests/test_fitters.py 로 고정).
"""

from typing import NamedTuple

import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge

from ope.policies import softmax_policy

PS_CLIP_MIN = 1e-6  # p̂ 하한 (0-나눗셈 방지 — 축 05 PS_CLIP 관례의 하한만 계승)


class CrossFitConfig(NamedTuple):
    n_folds: int = 2
    seed: int = 0
    ridge_alpha: float = 1.0
    logistic_c: float = 1.0
    max_iter: int = 300


def _fold_indices(n: int, n_folds: int, seed: int) -> list[np.ndarray]:
    """seed 고정 셔플로 n 행을 n_folds 개 fold 인덱스 배열로 균등 분할(축 12 crossfit 관례)."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    fold = np.random.default_rng(seed).permutation(n) % n_folds
    return [np.flatnonzero(fold == f) for f in range(n_folds)]


def _validate_lengths(context: np.ndarray, *arrays: np.ndarray) -> int:
    """context 와 나머지 (n,) 배열들의 길이 일치를 검증하고 n 을 반환."""
    n = len(context)
    bad = [len(a) for a in arrays if len(a) != n]
    if bad:
        raise ValueError(f"length mismatch: context has {n} rows, got arrays of length {bad}")
    return n


def fit_q_hat_crossfit(context: np.ndarray, action: np.ndarray, reward: np.ndarray,
                       n_actions: int, cfg: CrossFitConfig) -> np.ndarray:
    """out-of-fold q̂ (n, K) — fold 별 per-action Ridge cross-fit.

    fold 분할은 cfg.seed 고정 셔플(`_fold_indices`). 각 test fold 에 대해 나머지(train)
    fold 의 action-a 행들로 Ridge(x → r) 를 적합해 test fold 전 행의 q̂[:, a] 를 예측한다.
    train 쪽 action-a 표본이 2개 미만이면 fallback: action-a 평균 reward(그것도 없으면
    train 전체 평균 reward)를 상수 예측으로 쓴다(축 12 crossfit fallback 관례).

    Parameters
    ----------
    context : (n, d) 컨텍스트.
    action : (n,) 로깅 행동 인덱스 a_i ∈ {0..K-1}.
    reward : (n,) 관측 보상.
    n_actions : 행동 수 K.
    cfg : CrossFitConfig — n_folds ≥ 2 필수.

    Returns
    -------
    (n, K) out-of-fold q̂ — 각 행의 예측은 자기 fold 의 reward 를 보지 않는다.
    """
    context = np.asarray(context, dtype=float)
    action = np.asarray(action)
    reward = np.asarray(reward, dtype=float)
    n = _validate_lengths(context, action, reward)
    folds = _fold_indices(n, cfg.n_folds, cfg.seed)
    q_hat = np.empty((n, n_actions), dtype=float)
    for f, te in enumerate(folds):
        tr = np.concatenate([folds[g] for g in range(cfg.n_folds) if g != f])
        r_tr_mean = float(reward[tr].mean())
        for a in range(n_actions):
            rows = tr[action[tr] == a]
            if len(rows) >= 2:
                model = Ridge(alpha=cfg.ridge_alpha).fit(context[rows], reward[rows])
                q_hat[te, a] = model.predict(context[te])
            elif len(rows) == 1:
                q_hat[te, a] = float(reward[rows[0]])
            else:
                q_hat[te, a] = r_tr_mean
    return q_hat


def fit_pscore_crossfit(context: np.ndarray, action: np.ndarray, n_actions: int,
                        cfg: CrossFitConfig) -> tuple[np.ndarray, np.ndarray]:
    """out-of-fold (p̂_logged (n,), π̂0_dist (n, K)) — multinomial LogisticRegression(x → a).

    train fold 에 없는 action 클래스는 확률 0 열로 남긴다: predict_proba 의 열을
    clf.classes_ 로 전체 K 열에 매핑한다(축 05 의 searchsorted-classes_ 관례를 crossfit 로
    확장 — 여기서는 (n, K) 분포 전체를 채우므로 결측 클래스 열이 자동으로 0 이 된다).
    π̂0 행합은 1 (0 열 포함해도 predict_proba 가 보장). p̂ = π̂0[i, a_i] 를 PS_CLIP_MIN
    으로 하한 클립한다(0-나눗셈 방지).

    Parameters
    ----------
    context : (n, d) 컨텍스트.
    action : (n,) 로깅 행동 인덱스 a_i ∈ {0..K-1}.
    n_actions : 행동 수 K.
    cfg : CrossFitConfig — n_folds ≥ 2 필수.

    Returns
    -------
    (p̂_logged (n,), π̂0_dist (n, K)) — 둘 다 out-of-fold 예측.
    """
    context = np.asarray(context, dtype=float)
    action = np.asarray(action)
    n = _validate_lengths(context, action)
    folds = _fold_indices(n, cfg.n_folds, cfg.seed)
    pi0_dist = np.zeros((n, n_actions), dtype=float)
    for f, te in enumerate(folds):
        tr = np.concatenate([folds[g] for g in range(cfg.n_folds) if g != f])
        clf = LogisticRegression(C=cfg.logistic_c, max_iter=cfg.max_iter)
        clf.fit(context[tr], action[tr])
        cols = clf.classes_.astype(int)  # action 은 {0..K-1} 정수 인덱스 — classes_ = 열 인덱스
        pi0_dist[te[:, None], cols[None, :]] = clf.predict_proba(context[te])
    p_hat = np.clip(pi0_dist[np.arange(n), action], PS_CLIP_MIN, None)
    return p_hat, pi0_dist


def make_log_derived_candidate(context: np.ndarray, action: np.ndarray, reward: np.ndarray,
                               n_actions: int, beta: float, cfg: CrossFitConfig) -> np.ndarray:
    """(n, K) 로그-유래 후보 정책 분포 = softmax_policy(fit_q_hat_crossfit(...), beta).

    oracle 스코어(DGP 내부 참 기대보상) 누수 없이 로그만으로 후보 정책을 구축한다
    (absence H 해소). beta=0 ⇒ uniform, beta 가 클수록 q̂-greedy 에 근접한다.
    """
    q_hat = fit_q_hat_crossfit(context, action, reward, n_actions, cfg)
    return softmax_policy(q_hat, beta)
