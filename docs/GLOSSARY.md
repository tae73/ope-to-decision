# GLOSSARY — KO/EN 용어 단일기준

> **지위:** 이 표는 `ope-to-decision` 레포의 모든 산출물(README·docs·figure 라벨·코드 docstring·발표자료)에서
> 사용하는 KO/EN 용어의 **단일 정본**이다. 새 용어는 이 표에 먼저 등재한 뒤 사용한다.
> 수치는 이 문서에 등장하지 않는다 — 모든 수치는 [`docs/LEDGER.md`](LEDGER.md) 경유.

## 표기 규칙

1. **Estimator 이름·약어는 영어 그대로.** DM, IPS, SNIPS, Clipped-IPS, DR, Switch-DR, DRos, MIPS, MRDR 등은
   음차·번역 없이 영어 대문자 표기를 유지하고 한국어 조사만 붙인다 (예: "IPS는", "DRos의").
2. **일반 개념어는 첫 등장 시 병기, 이후 정본형.** 본문 첫 등장에서 `KO 표기(정본)` 열의 괄호 병기 형태를 쓰고,
   이후에는 병기 없이 정본형만 쓴다 (예: 첫 등장 "미관측 교란(unobserved confounding)" → 이후 "미관측 교란"
   또는 "confounding").
3. **영어 유지 용어.** overlap, deficient support, clipping, propensity, decision gate, regime map,
   breakdown Λ\*, negative control, bootstrap CI 등 실무·문헌 관례상 영어가 표준인 용어는 번역하지 않고
   영어를 정본으로 한다 ("중첩", "절단" 등 임의 번역 금지).
4. **수학 기호 고정.** π₀ = logging policy, π_e = evaluation policy, w(x,a) = π_e(a∣x)/π₀(a∣x) =
   importance weight, V(π) = policy value, q(x,a) = 기대 reward, q̂ = reward 모델 추정치.
   문서·코드·figure에서 동일 기호를 쓴다.
5. **범위 밖 용어는 이 표에 없다.** slate OPE(PI/IIPS/RIPS)·RL OPE(FQE/DICE)·OPL·CATE 관련 용어는 본 레포
   범위 밖이므로 의도적으로 미등재 — 필요 시 README의 범위 선언과 인접 레포(dunnhumby·causal-inference)
   링크를 따른다. proximal 계열 본류 용어도 연구 레포 소관으로 미등재.

## 1. 문제 설정

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| off-policy evaluation (OPE) | 오프폴리시 평가(OPE) → 이후 "OPE" | 로깅 정책이 수집한 로그만으로, 배포한 적 없는 평가 정책의 가치 V(π_e)를 추정하는 문제. |
| off-policy / on-policy | 오프폴리시 / 온폴리시 | 데이터를 수집한 정책과 평가 대상 정책이 다르면 off-policy, 같으면 on-policy. |
| logging (behavior) policy | 로깅 정책(logging policy, π₀) — behavior policy 동의어, 본 레포는 "로깅 정책"으로 통일 | 로그 데이터를 실제로 생성한 정책; 각 context x에서 action의 확률분포 π₀(a∣x). |
| evaluation (target) policy | 평가 정책(evaluation policy, π_e) — target policy 동의어, 본 레포는 "평가 정책"으로 통일 | 가치를 알고 싶은 새 정책 π_e; 아직 배포되지 않아 로그가 없다. |
| propensity score (logged vs true) | propensity(성향점수) — "기록 propensity" vs "진짜 propensity" 구분 필수 | 로깅 정책이 해당 action을 고를 확률 π₀(a∣x); 시스템에 **기록된 값**과 데이터 생성에 실제 작용한 **진짜 값**이 다를 수 있고, 그 괴리가 confounding·오지정 축(05·09)의 주제다. |
| importance weight | 중요도 가중치(importance weight, w) → 이후 "가중치 w" | w(x,a) = π_e(a∣x)/π₀(a∣x); IPS 계열 재가중의 핵심 비율로, 분포가 무거워질수록 분산이 폭발한다. |

## 2. 진단 (diagnostics)

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| effective sample size (ESS) | ESS(유효 표본 크기) | (Σw)²/Σw²; 가중 표본이 실질적으로 몇 개의 균등가중 표본에 해당하는지를 나타내는 진단 지표(축 08의 예보력 검증 대상). |
| max weight | max weight (영어 유지; 뜻풀이 "최대 importance weight") | 진단 3요소(ESS·max-weight·support) 중 하나; diagnostics 의 max_weight·weight_tail_p99 지표에 대응한다. |
| overlap | overlap (영어 유지; 뜻풀이 "공통 지지") | π_e가 확률을 두는 (x,a) 영역에 π₀도 양의 확률을 두는 정도; OPE 식별의 전제 조건. |
| deficient support | deficient support (영어 유지; 뜻풀이 "지지 결핍") | π_e(a∣x)>0인데 π₀(a∣x)=0인 영역의 존재; IPS 계열은 식별 자체가 불능이 된다 (Sachdeva–Su–Joachims KDD'20, https://arxiv.org/abs/2006.09438). 축 04. |

## 3. Estimator

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| DM (Direct Method) | DM (영어 유지) | reward 모델 q̂(x,a)로 정책가치를 직접 예측; 저분산이지만 모델 bias를 안는 극단 (Dudík–Langford–Li 2011, https://arxiv.org/abs/1103.4601). |
| IPS (Inverse Propensity Scoring) | IPS (영어 유지; IPW 동의어) | 가중치 w로 로그 reward를 재가중; unbiased이지만 고분산인 반대 극단 (같은 참조틀, https://arxiv.org/abs/1103.4601). |
| SNIPS (Self-Normalized IPS) | SNIPS (영어 유지) | 가중치 합으로 정규화한 IPS; 약간의 bias를 대가로 분산을 크게 줄인다 (Swaminathan & Joachims 2015, https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html). |
| clipping / Clipped-IPS | clipping (영어 유지) | 가중치를 상한 λ에서 자르는 분산 제어 장치; λ가 bias–variance 다이얼이며 축 07 민감도 분석의 대상. |
| DR (Doubly Robust) | DR (영어 유지) | DM baseline에 IPS 보정항을 더한 estimator; propensity·reward 모델 중 하나만 맞아도 consistent (Dudík–Langford–Li 2011, https://arxiv.org/abs/1103.4601). |
| Switch-DR | Switch-DR (영어 유지) | 가중치가 임계값 τ를 넘는 표본만 DM으로 전환하는 DR 변형; τ가 bias–variance 다이얼 (Wang–Agarwal–Dudík 2017, https://arxiv.org/abs/1612.01205). |
| DRos (DR with optimistic shrinkage) | DRos (영어 유지) | 가중치 shrinkage로 MSE를 직접 최적화하는 DR 변형 — "clipping의 원리화" (Su et al. 2020, https://arxiv.org/abs/1907.09623). |
| MIPS (Marginalized IPS) | MIPS (영어 유지) — 스트레치(축 13) | action embedding 공간의 marginal 성향비를 쓰는 대규모 액션용 estimator; 액션 수 폭발 시 IPS 붕괴의 처방 (Saito & Joachims ICML 2022, https://arxiv.org/abs/2202.06317). |
| MRDR (More Robust Doubly Robust) | MRDR (영어 유지) — 스트레치 | estimator 분산 최소화를 목적함수로 q̂를 학습하는 DR 변형; reward 모델의 학습목표 자체가 설계변수임을 보인다 (Farajtabar et al. 2018, https://arxiv.org/abs/1802.03493). |

## 4. 오차·불확실성 평가

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| bias–variance tradeoff | bias–variance 트레이드오프 | DM(bias 극단)과 IPS(variance 극단) 사이에서 estimator 패밀리가 위치하는 축; 본 레포 서사의 뼈대. |
| MSE decomposition | MSE 분해 | MSE = bias² + variance; 참값을 보유한 합성 DGP에서 estimator별 오차를 성분 분해하는 본 레포의 기본 평가 프로토콜. |
| bootstrap CI | bootstrap CI (영어 유지) | 실데이터 축(11–12) figure 전용 — 근사참값 비교 시 필수; 합성 MC 축(01–10·15–16)은 seed-ensemble band(+heavy-tail 시 p90) 병기 (CLAUDE.md §2 규약). |
| regime map | regime map (영어 유지) | 표본 크기 × overlap 등 설계 grid의 각 칸에 최저-MSE estimator를 색으로 표시한 히트맵; hero figure ②이자 decision gate의 증거층. |
| SLOPE | SLOPE (영어 유지) | OPE hyperparameter를 데이터 기반으로 선택하는 방법 — Lepski 원리 적용 (Su et al. ICML 2020, http://proceedings.mlr.press/v119/su20d/su20d.pdf). 축 07·M2. |
| IEOE | IEOE (영어 유지) | estimator robustness 자체를 평가하는 프로토콜 — error CDF 비교 (RecSys'21, https://arxiv.org/abs/2108.13703). 축 07. |

## 5. 의사결정·confounding

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| decision gate | decision gate (영어 유지) | 진단(ESS·max-weight·support) → 기각 → estimator 선택 → "믿는다 / 못 믿는다 / A/B로 보낸다" 판정 절차 — **본 레포의 제안(문헌에 산발적으로 존재하는 folklore의 체계화 시도)이며 확립된 표준이 아니다** (관행의 산발성: https://arxiv.org/pdf/2207.00632). 축 08. |
| unobserved confounding | 미관측 교란(unobserved confounding) → 이후 "confounding" 허용 | 로깅 정책의 action 선택과 reward에 동시에 영향을 주지만 로그에 기록되지 않은 변수 U의 존재; 기록 propensity ≠ 진짜 propensity가 되어 표준 진단이 원리적으로 탐지 불능(blind) (RecSys 2023, https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding). 축 09. |
| sensitivity analysis | 민감도 분석(sensitivity analysis) | 미관측 교란의 세기를 가정 파라미터로 두고, 그 가정 아래 결론이 유지되는 범위를 계산하는 분석 틀. |
| marginal sensitivity model (MSM) Λ | MSM Λ (영어 유지) — 조건부 스트레치(축 14, probe GO 시) | 기록 propensity와 진짜 propensity의 odds ratio가 Λ 이내라는 가정 하에 policy value의 worst-case 구간을 계산하는 모형 (Kallus & Zhou, https://arxiv.org/pdf/1805.08593). |
| breakdown Λ\* | breakdown Λ\* (영어 유지) — 조건부 스트레치(축 14) | 정책 순위(또는 gate 판정)가 뒤집히는 최소 Λ; "이 결론이 뒤집히려면 얼마나 큰 confounding이 필요한가"라는 robustness certificate 언어. |
| negative control | negative control (영어 유지; 뜻풀이 "음성 대조") — 미채택(참고 개념 — 본 레포의 확정 축 01–14 에 대응 실험 없음; 연구 브릿지 조사에서 검토된 선택 훅) | 인과 효과가 없음을 아는 action/outcome에서 nonzero 효과가 잡히면 confounding 알람으로 쓰는 진단 — A/A test와 유사한 sanity check. |

## 6. 데이터·ground truth

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| classification-to-bandit (c2b) | classification-to-bandit → 이후 "c2b" | multi-class 분류 데이터의 label을 action으로, 정답 여부를 reward로 바꿔 깨끗한 참값을 가진 bandit 로그를 만드는 표준 변환 프로토콜 (변환 사용 예: https://arxiv.org/pdf/1802.03493). 축 11. |
| OBD (Open Bandit Dataset) | OBD (영어 유지) | ZOZO가 공개한 multi-action bandit 로그 — 실측 propensity와 로깅 정책 2종(uniform random + Bernoulli TS)의 병행 수집으로 근사참값 프로토콜이 성립하는 공개 데이터 (https://arxiv.org/abs/2008.07146 · https://research.zozo.com/data.html). 축 12(small 버전). |
| ground truth (exact vs approximate) | 참값(ground truth) vs 근사참값 — 구분 필수 | 합성 DGP·c2b에서는 V(π_e)를 해석적/Monte Carlo로 아는 **참값**, OBD에서는 random-policy 로그의 on-policy 평균이라 표본 오차를 갖는 **근사참값** — 근사참값 비교에는 bootstrap CI를 병기하고 점 비교 단정을 금지한다. |

## 7. 비즈니스 지표 (M6 — 축 15·16)

| EN 용어 | KO 표기(정본) | 한 줄 정의 |
|---|---|---|
| CTR (click-through rate) | CTR (영어 유지) | 세션(노출)당 클릭 확률 — funnel 최상층 지표. 본 레포 funnel DGP(`src/ope/business.py`)에서 reward=click 인 정책가치 V_ctr(π). 축 15·16. |
| CVR (conversion rate) | CVR (영어 유지) — **세션 기준** 명시 필수 | 본 레포의 CVR 은 **세션 기준** = 노출당 click·conv 의 기대(E[click·conv])다 — **업계 관행의 click-조건부 CVR(E[conv∣click])과 다르다**. 분모(클릭 수)까지 추정치가 되는 ratio-of-estimates 함정을 회피하기 위한 정의 선택이며, 관련 figure 캡션에 고정 병기한다. 축 15·16. |
| funnel | funnel (영어 유지; 뜻풀이 "전환 깔때기") | impression → click → conversion → revenue 로 이어지는 단계 구조. 단계가 깊을수록 이벤트가 희소해져 같은 로그에서 OPE 판별 한계가 커진다 — "funnel 신뢰도 사다리"(축 15). 세션 간 지표(retention 등)는 funnel 에 없다 — single-step bandit OPE 로 식별 불가(RL OPE 소관, PLAYBOOK §8.4). |
| guardrail metric | guardrail (영어 유지) | 주 지표 개선(예: Δ̂CTR>0)을 전제로 부지표 악화 한계(예: Δ̂REV≥−g)·구조 제약(예: HHI≤h)을 함께 요구하는 다중 지표 게이트의 보호 지표 — 한계값 g·h 는 시연값(무교정, 본 레포 "제안" 지위 계승). 축 16. |
| HHI (Herfindahl–Hirschman index) | HHI (영어 유지) | 점유율 제곱합 Σsᵢ² 로 정의되는 집중도 지표 — 본 레포에선 광고주별 기대 노출 점유율에 적용하며, π 가 기지이므로 **OPE 가 아니라 정확 계산**(결정적 arm — 오류율 0)이다. 축 16 (`src/ope/business.py` `hhi`). |
