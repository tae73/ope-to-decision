"""실데이터 2트랙 로더 (M3 구현).

트랙 1 — classification-to-bandit(c2b) 변환(표준 프로토콜, Dudík et al. 계열):
    multi-class 라벨 = action, 정답 여부 = reward(결정적 1[a=y] — DR 잔차가 순수 모델오차라는
    특성을 문서화해 둔다: 노이즈 채널이 없어 good q̂ 이 압승하는 구조).
    정책 쌍은 **softmax(β·s)** (ε-greedy 쌍은 max w≈1.25 로 overlap 스트레스가 전무 — M3 Plan
    검증 지적). propensity 는 정확히 계산·기록된다(추정 아님 — 합성 트랙과 정합).
    `gt_value` 는 전체 라벨로 **정확**: V(π_e)=mean_B[π_e(y|x)] (gt_ci=None).
트랙 2 — Open Bandit Dataset small(ZOZO): 얇은 로더만 제공(파싱 함정 처리) — 프로토콜 조립은
    experiments/12_obd_small_gate.py 소관. 근사 GT 는 bootstrap CI 병기(PLAN §3.4).
데이터 파일은 커밋하지 않는다(data/README.md — OBD 는 로컬 다운로드 전용·재배포 금지;
repo 코드는 Apache-2.0, 데이터는 ZOZO 별도 조건).
"""

from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from ope.policies import softmax_policy

# OpenML 버전 고정 (version 드리프트 방지 — M3 Plan 검증 지적). name → version.
C2B_DATASETS = {"optdigits": 1, "satimage": 1, "pendigits": 1, "letter": 1}
C2B_SPLIT_SEED = 0  # 구조 seed(정책·split 고정) — 로깅 재표집은 seed 인자만 바꾼다 (struct_seed 관례)


class RealBanditData(NamedTuple):
    context: np.ndarray        # (n, d) — 밴딧化된 평가 절반(B)
    action: np.ndarray         # (n,)
    reward: np.ndarray         # (n,)  c2b: 1[a=y] 결정적
    pscore: np.ndarray         # (n,)  정확 로깅 propensity (추정 아님)
    n_actions: int
    source: str
    pi_e_dist: np.ndarray      # (n, K) 평가 정책 분포 (정확)
    q_scores: np.ndarray       # (n, K) good q̂ = A-적합 LR proba
    q_scores_degraded: np.ndarray  # (n, K) 비-affine 오지정 q̂ = 피처 절반 절단 LR proba
    gt_value: float            # c2b: 전 라벨 정확 참값 / OBD 조립측: 근사 참값
    gt_ci: tuple | None        # c2b: None(정확) / OBD: bootstrap CI — 의미 분리(정직성 규약)


def _c2b_support_mask(s: np.ndarray, deficiency: float) -> np.ndarray:
    """per-row 하위-s ⌊δK⌋ 컬럼의 로깅 지지 제거 mask (True=지지) — M9 축 21 주입 장치.

    dgp._support_mask 와 동일 의미론(구조적 — 랜덤 mask 는 proxy 를 원리적 blind 으로 만듦,
    PLAN §3.6-2)을 LR-score 행렬 s 위에 적용한다.
    """
    n, k = s.shape
    if not 0.0 <= deficiency < 1.0:
        raise ValueError(f"support_deficiency must be in [0, 1), got {deficiency}")
    m = int(np.floor(deficiency * k))
    if m == 0:
        return np.ones((n, k), dtype=bool)
    if m >= k:
        raise ValueError("support_deficiency removes all actions")
    order = np.argsort(s, axis=1)
    mask = np.ones((n, k), dtype=bool)
    np.put_along_axis(mask, order[:, :m], False, axis=1)
    return mask


def classification_to_bandit(dataset_name: str, seed: int,
                             beta_log: float = 2.0, beta_eval: float = 6.0,
                             data_home: str = "data/openml",
                             support_deficiency: float = 0.0) -> RealBanditData:
    """OpenML 데이터셋 → logged bandit feedback (정확 propensity·정확 참값).

    구조(split·정책)는 C2B_SPLIT_SEED 로 고정하고 `seed` 는 로깅 행동 표집에만 쓴다 —
    합성 트랙의 struct_seed/seed 분리 관례와 동일("같은 환경, 다른 로그").
    LabelEncoder 강제: satimage 라벨 {1..7}\\{6} 비연속·letter 문자 라벨 함정(Plan 검증).

    `support_deficiency`(M9 축 21 — PLAN §3.6-2): per-row 하위-s ⌊δK⌋ 액션의 로깅 지지를
    구조적으로 제거하고 π₀ 를 renormalize 한 뒤 **같은 rng 호출 순서로** 행동을 표집한다 —
    기록 pscore 는 masked 세계의 **정확** propensity(기록 자체는 무결·진짜 식별 실패).
    π_e·gt_value 는 불변. **δ=0 이면 기존 산출과 bit-identical**(tests 회귀 고정).
    """
    from sklearn.datasets import fetch_openml

    if dataset_name not in C2B_DATASETS:
        raise ValueError(f"unknown c2b dataset {dataset_name!r} — {sorted(C2B_DATASETS)}")
    bunch = fetch_openml(name=dataset_name, version=C2B_DATASETS[dataset_name],
                         data_home=data_home, as_frame=False, parser="auto")
    x = np.asarray(bunch.data, dtype=float)
    y = LabelEncoder().fit_transform(bunch.target)
    k = int(y.max()) + 1

    x_a, x_b, y_a, y_b = train_test_split(x, y, test_size=0.5, stratify=y,
                                          random_state=C2B_SPLIT_SEED)
    scaler = StandardScaler().fit(x_a)
    x_a, x_b = scaler.transform(x_a), scaler.transform(x_b)

    lr = LogisticRegression(max_iter=1000).fit(x_a, y_a)
    s = lr.predict_proba(x_b)  # (n, K) — good q̂ 이자 정책 스코어
    d_half = x.shape[1] // 2   # 비-affine 오지정: 피처 앞 절반만 본 LR (capacity 제한)
    lr_bad = LogisticRegression(max_iter=1000).fit(x_a[:, :d_half], y_a)
    s_bad = lr_bad.predict_proba(x_b[:, :d_half])

    pi0 = softmax_policy(s, beta_log)
    pi_e = softmax_policy(s, beta_eval)
    mask = _c2b_support_mask(s, support_deficiency)
    if not mask.all():  # δ=0 이면 이 분기 자체를 건너뛰어 bit-항등 보장 (renormalize 부동소수 회피)
        pi0 = np.where(mask, pi0, 0.0)
        pi0 = pi0 / pi0.sum(axis=1, keepdims=True)
    rng = np.random.default_rng(seed)
    n = len(y_b)
    action = (pi0.cumsum(axis=1) > rng.random((n, 1))).argmax(axis=1)
    idx = np.arange(n)
    reward = (action == y_b).astype(float)
    gt_value = float(pi_e[idx, y_b].mean())  # 전 라벨 정확 참값

    return RealBanditData(context=x_b, action=action, reward=reward,
                          pscore=pi0[idx, action], n_actions=k, source=dataset_name,
                          pi_e_dist=pi_e, q_scores=s, q_scores_degraded=s_bad,
                          gt_value=gt_value, gt_ci=None)


def load_obd_small(policy: str, campaign: str = "all",
                   data_dir: str = "data/obd") -> pd.DataFrame:
    """data/obd/{policy}/{campaign}/{campaign}.csv 로드 (얇은 로더 — 프로토콜 조립은 축 12).

    파싱 함정 처리: 첫 열은 무명 인덱스(`index_col=0`), propensity 컬럼명은 CSV 헤더의
    `propensity_score` 가 정본(OBD README 산문의 action_prob 표기와 상충 — CSV 가 맞다).
    데이터 미배치 시 data/README.md 의 다운로드 안내를 에러로 알려준다.
    """
    if policy not in ("random", "bts"):
        raise ValueError(f"policy must be 'random' or 'bts', got {policy!r}")
    path = Path(data_dir) / policy / campaign / f"{campaign}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} 없음 — OBD small 을 로컬에 배치하라 (data/README.md 참조; "
            "github.com/st-tech/zr-obp 의 obd/ 디렉토리, 재배포 금지·로컬 전용)")
    df = pd.read_csv(path, index_col=0)
    required = {"item_id", "position", "click", "propensity_score"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"OBD CSV 에 기대 컬럼 누락: {sorted(missing)}")
    return df
