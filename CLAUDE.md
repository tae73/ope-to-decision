# ope-to-decision — CLAUDE.md (에이전트 작업 규약)

## 1. 정체성 · 비범위

- **한 줄 정체성:** multi-action logged bandit **OPE 벤치마크 + 배포 게이트(decision gate) 플레이북** —
  "A/B 테스트 전에, 이미 쌓인 추천 로그만으로 새 정책의 가치를 추정하고, 더 중요하게는 그 추정을
  **언제 믿으면 안 되는지**까지 판정한다."
- **비범위 (README 에도 선언·상호 링크):** OPL/policy learning·CATE 는 범위 밖
  (→ `../kr_segmentation_causal_targeting_dunnhumby`, `../causal-inference`), slate OPE(PI/IIPS/RIPS)·
  RL OPE(FQE/DICE)도 범위 밖. **proximal 등 confounding 교정 본류는 연구 트랙(decision-frontier) 소관** —
  본 레포는 축 09 의 "진단이 못 보는 것" 대조표(+조건부 스트레치 축 14 Λ-sweep, M8 축 18 의
  calibrated-confounding **경계 전시**)에서 **의도적으로 멈춘다** — GT-미상 트랙(축 17–20)도
  confounding 교정을 주장하지 않는다(출구는 Λ-감도 구간 이월뿐).
- 프로젝트 전체 설계 정본: 비공개 로컬 계획 문서(레포 외부) — 레포 내 정본은 `PLAN.md`. 마일스톤·게이트·
  진행 상태는 `PLAN.md` — 작업 완료 시마다 PLAN.md 상태를 동기화한다.

## 2. 코드 패턴 규약 (하우스 스타일 — dunnhumby·mta-simulation 계승)

- **Config NamedTuple → 순수함수 → Result NamedTuple.** 이것이 유일한 추상화다.
  **클래스 계층·OOP 패턴 금지**, 상태 없는(stateless) 함수 + 명시적 seed 만 사용.

  ```python
  config = DGPConfig(n=10_000, n_actions=10, ..., seed=12345)      # 입력: Config NamedTuple
  data = make_synthetic_bandit_data(config)                        # 순수함수
  est = estimate_ips(data.reward, data.action,
                     data.pscore_logged, data.pi_e_dist)           # 반환: EstimateResult(value, weights)
  ```

- **shape 규약 — `src/ope/estimators.py` docstring 표기가 정본:**
  - `reward` : `(n,)` 관측 보상 r_i
  - `action` : `(n,)` 로깅 정책이 고른 행동 인덱스 a_i ∈ {0..K-1}
  - `pscore` : `(n,)` 로깅 propensity π_0(a_i|x_i) — "기록값" (진짜와 다를 수 있음: 축 09)
  - `pi_e_dist` : `(n, K)` 평가 정책의 행동 분포 π_e(·|x_i)
  - `q_hat` : `(n, K)` reward 모델 예측 q̂(x_i, ·)
- estimator 는 전부 `EstimateResult(value, weights)` 반환 — `weights`(변형 후 importance weight)는
  diagnostics 의 입력으로 재사용된다. **불확실성 병기 규약(M2 정밀화)**: 합성 MC 축(01–10·15–16)은
  **S-seed ensemble band**(mean±2·SE over seeds)로 병기하고, `bootstrap_ci` 는 **실데이터 축 11–12
  (단일 로그·seed 반복 불가) 전용**이다 — 전면 bootstrap 은 런타임 ×1000 으로 비실용(M2 실측).
  **M8 예외(practitioner 트랙 축 17–20)**: frontstage 는 합성 로그여도 단일-로그 **joint bootstrap**
  (B=500 — 같은 재표집 인덱스에서 7종+battery 동시 계산·paired; 실무자에게 로그는 하나)을 쓴다.
  seed-ensemble 은 reveal(백스테이지) 집계 전용 — 정본: PLAN §3.5.
- Hydra 는 **Compose API 로만** 로드(`@hydra.main` 금지), OmegaConf → NamedTuple 변환 후
  src 모듈에는 NamedTuple 만 전달 (mta-simulation 관례). config 루트는 `configs/config.yaml`.
- 네이밍: 함수 verb_noun(`estimate_*`·`compute_*`·`make_*`), 상수 UPPER_SNAKE, private `_prefix`.
  NumPy 스타일 docstring. 문서·주석은 한국어 산문 + 영어 수식·전문용어(estimator 이름·약어는 영어 유지).

## 3. 모듈 맵 (`src/ope/` — 의존은 한 방향, 상호 import 금지)

| 모듈 | 역할 | 내부 의존 |
|---|---|---|
| `policies.py` | softmax(β)·ε-greedy 정책 분포 생성 순수함수 | 없음 (numpy 만) |
| `dgp.py` | 합성 DGP: `DGPConfig` → `SyntheticBanditData`(+ `v_true` ground truth). 축 09 장치 = `pscore_logged ≠ pscore_true` | `policies` |
| `estimators.py` | 코어 7종 DM·IPS·SNIPS·Clipped-IPS·DR·Switch-DR·DRos + `bootstrap_ci` — 배열만 받는 leaf | 없음 (numpy 만) |
| `diagnostics.py` | ESS·max-weight·support 진단(`DiagnosticsReport`) + `decision_gate` 3-way 판정(`GateVerdict`) | 없음 (numpy 만) |
| `datasets.py` | 실데이터 2트랙 로더: classification-to-bandit(OpenML)·OBD small(ZOZO) | 없음 (numpy/sklearn) |
| `business.py` | 비즈니스 층(M6): funnel DGP·지표 벡터(CTR/CVR/REV)·노출/HHI 정확 계산·subgroup 매출 IPS — **γ(confounding) 노브 영구 금지** | `policies`·`dgp`(`_stable_sigmoid`) |

- 조립(데이터 → estimator → 진단 → figure)은 **`experiments/` 스크립트에서만** 한다.
  src 모듈은 결과 파일을 직접 쓰지 않는다(입출력 없음·순수 계산).
- 스텁의 `raise NotImplementedError("M1")`/`("M3")` 표기는 구현 마일스톤 약속 — 구현 시 해당
  마일스톤 게이트(probe·property test) 먼저 확인.

## 4. 실험 규율

- `experiments/NN_slug.py` — **실험 축 ID 불변** (결과 파일·docs 수치와 강결합, mta-simulation 패턴 계승):
  - 코어: 01 표본 n / 02 로깅 β(overlap) / 03 타깃-로깅 괴리 / 04 deficient support /
    05 propensity 오지정 / 06 reward model 오지정 / 07 hyperparameter 민감도(IEOE) /
    08 진단 예보력+결정규칙 / 09 confounding 주입+대조표 / 10 의사결정 metric /
    11 c2b 멀티데이터셋 / 12 OBD small 게이트
  - 스트레치(조건부): 13 액션 수+MIPS / 14 Λ-sweep — 각각 1일 probe GO 시에만 착수
  - 비즈니스 층 15–16(M6, probe 선행): 15 funnel 신뢰도 사다리 / 16 다중 지표 비즈니스 게이트
  - GT-미상 practitioner 트랙(M8 본편 — **사전등록 PLAN §3.5**·probe M8-A/M8-B 게이트):
    17 validity battery / 18 calibrated-confounding 경계 / 19 end-to-end blind decision /
    20 OBD decision card. frontstage 산출 `NN_*_decision.csv` 에 `v_true`·oracle 컬럼 금지
    (계약 테스트), reveal 채점은 `NN_*_reveal.csv` 분리(축 20 은 reveal 없음). tier 분류
    (본편/백스테이지)의 정본은 PLAN §2·experiments/README(`무대` 열은 M8 Stage 5 문서 역전에서
    추가 예정 — 그 전까지는 PLAN §4.9 Stage 5 항목이 배정 기록).
- `experiments/probes/` 는 **self-contained**(src 미의존, 단독 실행 가능) — research-design Stage 3
  포맷(WHAT GENERALIZES / THE RESULT boxed / HONEST reduces_check / VERDICT) + JSON 산출 유지.
- 결과 경로 규약: figure → `results/figures/NN_*.{png,svg}`, 수치 → `results/tables/NN_*.{json,csv}`
  (probe 는 `results/tables/probe_*.json`). `data/` 는 커밋 금지(.gitignore·`data/README.md` 참조).
- 축 추가·스윕 범위 변경 시 `experiments/README.md` 의 축↔ID 매핑과 `PLAN.md` 를 함께 갱신.
- git: 커밋은 사용자가 요청할 때만, force-push 금지, 데이터·env 디렉토리 커밋 금지.

## 5. 정직성 가드레일 (위반 = 검증 단계에서 반려)

- **모든 문서 수치는 `docs/LEDGER.md`(정본 ledger) 경유** — committed 결과 파일에서만 옮겨 적는다.
  수치 자작·반올림 왜곡·기억 인용 금지. README 등 문서에 ledger 에 없는 수치가 나타나면 반려.
- **실험 전 수치 금지:** M0 현재 실험 결과가 없다. 허용되는 수치는 설계 기본값(`configs/`)·
  마일스톤 주수(`PLAN.md`)·출처 URL 병기된 외부 인용뿐이다.
- decision gate 규칙(ESS·max-weight 임계값 → trust/distrust/ab_fallback)은
  **"본 레포의 제안(folklore 관행의 체계화 시도) — 문헌 표준 아님"** 프레임을 고정한다.
  임계값 근거는 축 08 실험으로만 정당화하고, 축 09 의 confounding blind spot 을 반드시 함께 전시.
- OBD small 의 근사참값(random-policy on-policy 평균)은 표본오차를 가진다 — 관련 figure·표에
  **bootstrap CI 병기**, 점 비교 단정 금지.
- 미확인 사실은 **[불확실] 태그** 유지, probe 로 해소하면 정본(JSON) 경로와 함께 해소 기록을 남긴다 —
  선례: obp PyPI 릴리스 버전(0.5.5 vs 0.5.7 상충 기록)은 probe M0-B 의 PyPI 재확인으로 **해소**
  (0.5.7/2023-04-14 — `results/tables/probe_obp_pypi_check.json`, LEDGER m0b-pypi 행).
- **validity battery(M8 축 17–20) 주장 규율**: battery 는 **"필요조건 검사(falsifier)"** 프레임 고정 —
  "GT-free 검증/보증" 류 표현 금지. battery 관련 주장 문단마다 **calibrated confounding 원리적 무검출
  (관측 동등성 — 축 18) co-exhibit 의무**(축 09 co-exhibit 규칙의 확장). family 분리 없는 pooled
  발화율 단독 인용 반려(PLAN §3.5-3). frontstage 산출물에 oracle 컬럼(`v_true`·`q_true`·`pscore_true`·
  `gt_value`)이 나타나면 반려.

## 6. 환경(env) 규약

- **기본 env = uv** (`uv sync`, Python ≥3.11, `pyproject.toml` 정본). 테스트는 `uv run pytest`
  (M0 은 smoke 수준 — M1 에서 property test 로 대체·확장).
- **`.venv-obp/` = Python 3.9 pinned, obp 교차검증 전용.** obp 는 `python>=3.7.1,<3.10`·
  `scikit-learn==1.0.2` 고정(출처: https://raw.githubusercontent.com/st-tech/zr-obp/master/pyproject.toml)
  이므로 기본 env 와 **절대 혼용 금지**. 이 env 에서 실행되는 파일(`probe_obp_crossval.py`)은
  Python 3.9 호환 문법만 사용한다.
- **`.venv-sbobp/` = sb-obp(최신 fork) 폴백 트랙 전용** — obp 설치 실패 시에만 사용,
  probe 산출 JSON 은 `_sbobp` suffix 로 분리 기록.
- 세 env 디렉토리 모두 .gitignore 대상 — 커밋 금지.

## 7. 참조 자산 (신규 작성 금지 — 참조·링크만)

- OPE 진단 스펙 정본: `../dag-registry/docs/dag-design.md` **§6.5** (ESS·max importance weight·
  clipping) — `diagnostics.py` 는 이 문서 스펙의 실행 코드 실증(문서→구현 보완 관계).
- 축별 실험·config 패턴: `../mta-simulation/experiments/` · `../mta-simulation/configs/dgp/default.yaml`.
- 순수함수 시그니처 관례: `../kr_segmentation_causal_targeting_dunnhumby/src/policy.py`.
- 선점확인·스윕 근거(출처 URL 포함): `docs/POSITIONING.md` 로 이관 정리 — 외부 주장 인용 시
  반드시 URL 병기.
