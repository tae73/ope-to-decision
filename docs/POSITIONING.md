# POSITIONING — 선점확인·포지셔닝 노트 (research-design Stage 1)

- **작성일:** 2026-08-06 (M0 세션) · **지위:** 11-에이전트 웹 스윕(생태계·estimator·축·혼잡도·bridge) 결과의 정리·이관본
- **검증 상태:** 전부 웹검색 기반 외부 조사 — 본 레포의 실험 결과는 아직 없다. star 수·레포 수·릴리스 날짜 등은 검색 시점(2026-08-06) 스냅샷이며 변동 가능. 미해소 사실은 "불확실" 태그로 남기고 probe로 해소한다.
- **핵심 프레임 주의:** 본 레포의 decision gate 규칙은 **본 레포의 제안(문헌에 산재한 folklore의 체계화 시도)이지 확립된 표준이 아니다**. 이 문서의 "갭" 주장도 선점 근거이지, 규칙 자체의 정당성 근거가 아니다.

---

## 1. 공개 공간 혼잡도 — 전체적으로 저혼잡, '플레이북 계층'이 공백

GitHub `off-policy-evaluation` topic은 공개 레포 **36개**에 그친다(2026-08-06 검색 기준, https://github.com/topics/off-policy-evaluation). 구성은 라이브러리 3–4개 + awesome 리스트 + 논문 재현 코드가 대부분이고, 흔한 형태는 다음과 같다.

| 형태 | 예시 | 비고 |
|---|---|---|
| obp quickstart 재탕 | ZOZOTOWN OBD 노트북 변형 (https://github.com/st-tech/zr-obp/blob/master/examples/quickstart/obd.ipynb) | 라이브러리 호출 수준, 의사결정 프레임 없음 |
| 논문 재현 코드 | cascade-dr 13★ (https://github.com/aiueola/wsdm2022-cascade-dr), kdd2023-aips 11★ (https://github.com/aiueola/kdd2023-aips) | 소규모, 논문 스코프에 종속 |
| 단순 estimator 구현 | banditml `offline-policy-evaluation` 220★ (https://github.com/banditml/offline-policy-evaluation) | estimator 나열에서 정지 |
| Kaggle | OPE 노트북이 검색에서 사실상 안 잡힘 | 약한 존재감 — 부재 증명은 아님 |
| 한국어 | velog·tistory·국내 기업 기술블로그에서 검색상 거의 발견 안 됨 | 불확실 — 가시성만 확인, 커버리지 한계 아래 §3 참조 |

완성도 높은 예시 3개와 각각의 강점:

1. **zr-obp** (st-tech, 704★) — 실측 multi-action logged bandit 데이터(OBD) + estimator 전체 스위트 + "evaluation-of-OPE" 표준 프로토콜 (https://github.com/st-tech/zr-obp). 단 유지보수 상태는 §2 참조.
2. **SCOPE-RL** (hakuhodo-technologies, 142★) — 오프라인 RL+OPE/OPS end-to-end, evaluation-of-OPE 강조, EN/JA 이중 quickstart (https://github.com/hakuhodo-technologies/scope-rl).
3. **TDS 튜토리얼**(Adrien Biarnes) + **Eugene Yan 글** — estimator 분류 교육에 강함, 그러나 코드 레포·의사결정 프레임이 없음 (https://towardsdatascience.com/a-complete-tutorial-on-off-policy-evaluation-for-recommender-systems-e92085018afe/ · https://eugeneyan.com/writing/counterfactual-evaluation/).

요약: 라이브러리(estimator 10여 개 제공)와 튜토리얼(분류 교육) 사이의 **"진단 → 기각 → 선택 → A/B 회귀" 의사결정 플레이북 계층이 공개 공간에서 비어 있다**. 본 레포는 그 계층을 정면 타깃으로 한다.

## 2. 라이브러리 생태계 상태 — 사실상 표준(obp)이 stale

| 라이브러리 | 상태 (2026-08 검색 기준) | 본 레포에서의 취급 |
|---|---|---|
| **obp** (st-tech/zr-obp) | 사실상 중단: 마지막 push 2024-06, Snyk "inactive" (https://github.com/st-tech/zr-obp · https://snyk.io/advisor/python/obp). pyproject가 `python >=3.7.1,<3.10` · `scikit-learn==1.0.2` · `scipy 1.7.3` 고정 (https://raw.githubusercontent.com/st-tech/zr-obp/master/pyproject.toml) → 최신 Python(3.11+)·numpy 2.x 설치 실패 가능성 높음. 취약 의존성 이슈 #206 미해결 | 교차검증 전용 — 별도 py3.9 pinned env. **주 구현은 numpy 자작**(라이브러리 종속 회피) |
| **sb-obp** (sb-ai-lab 포크) | 유지되는 포크 존재 (https://github.com/sb-ai-lab/sb-obp · https://pypi.org/project/sb-obp/), 스윕 시점 push 2025-08. 단 star 1개 — 커뮤니티 대체재로 단정하긴 이름 | 교차검증 2차 축(최신 env) — obp pinned env와 이중 대조 |
| **scope-rl** (hakuhodo) | 마지막 push 2024-03 휴면. trajectory OPE(RL) 중심 (https://github.com/hakuhodo-technologies/scope-rl · https://arxiv.org/abs/2311.18206) | single-step bandit 스코프에 과체중 — 참조만, 의존 안 함 |
| **d3rlpy** | 활발(스윕 시점 push 2025-09, 1.7k★, https://github.com/takuseno/d3rlpy). 단 OPE는 FQE만 (https://d3rlpy.readthedocs.io/en/latest/references/off_policy_evaluation.html) | RL policy selection 용도 — 범위 밖 |
| **Vowpal Wabbit** | repo push 2026-07이나 마지막 안정 릴리스 9.6.0(2022-11) — maintenance mode (https://github.com/VowpalWabbit/vowpal_wabbit/releases) | C++/CLI 블랙박스 — "이해도 증명" 목적에 부적합 |
| **RecSim / RecoGym** | RecSim archived(2022-01, https://github.com/google-research/recsim), RecoGym 마지막 push 2021-07 (https://github.com/criteo-research/reco-gym) | 기반으로 삼지 않음 |

> **해소됨 — obp 릴리스 버전 (probe M0-B가 PyPI 재확인, 2026-08-06).** obp PyPI 최신 = **0.5.7**(2023-04-14 업로드), GitHub Releases 태그는 0.5.5에서 멈춤 — 태그·PyPI 불일치 자체가 유지보수 이완 신호다. sb-obp = **0.5.10**(2025-08-19, `requires_python >=3.8.1,<3.13` — 활발). 정본: `results/tables/probe_obp_pypi_check.json` (M0-B, 2026-08-06).

시사점: "사실상 표준" 라이브러리가 stale이라는 사실 자체가 본 레포의 두 설계를 정당화한다 — ① estimator를 numpy로 직접 구현(종속 리스크 회피 + 이해도 신호), ② obp/sb-obp를 **적대 교차검증 상대**로만 사용(자기 구현을 표준 구현에 대조 — 불일치는 그대로 보고).

### 2.1 데이터셋 생태계 — 3트랙 선택의 근거

본 레포의 데이터 3트랙(합성 DGP 본체 · classification-to-bandit 축 11 · OBD small 축 12)은 아래 후보 조사에서 나온 선택이다.

- **Open Bandit Dataset (ZOZO)** — 26M rows. 로깅 정책 2종(uniform random + Bernoulli TS)의 A/B 병행 수집 + **참 propensity 실측 기록**이라는 점에서, random-policy 로그를 근사 ground truth로 쓰는 표준 OPE 벤치마크 프로토콜의 성립을 조사 범위 내에서 확인한 유일한 공개 multi-action 데이터(부재 증명 아님 — §3의 검색 커버리지 한계 준용)(80 actions, position 3). small 샘플이 repo에 동봉, 전체는 별도 배포 (https://arxiv.org/abs/2008.07146 · https://research.zozo.com/data.html). 단 근사 참값 자체에 표본 오차가 있으므로(스윕 심사 지적) 관련 figure에는 bootstrap CI를 병기하고 점 비교를 단정하지 않는다는 규율이 계획에 이미 반영돼 있다.
- **classification-to-bandit 변환** — multi-class label을 action으로, 정답 여부를 reward로 바꾸는 표준 프로토콜(Dudík/Agarwal et al. 2014 계열; 변환 사용 예: https://arxiv.org/pdf/1802.03493). 표준 세트는 optdigits · satimage · pendigits · letter · vehicle · yeast · ecoli · glass (UCI/OpenML). 깨끗한 ground truth 기반 멀티데이터셋 체계 벤치(축 11)에 적합.
- **KuaiRec** — 거의 100% 밀도의 fully-observed matrix로 semi-synthetic OPE에 최적이며 최근 논문들이 실제 사용 (https://arxiv.org/pdf/2202.10842). 그러나 OBD + c2b와 동시 채택은 과잉 — **범위 밖**.
- **Criteo counterfactual test-bed** — 로깅 propensity 포함이나 35GB gzipped로 포트폴리오 규모에 과대 (https://www.cs.cornell.edu/~adith/Criteo/ · https://ailab.criteo.com/dataset-release-evaluation-counterfactual-algorithms/) — **범위 밖**.
- **MIND / MovieLens** — 로깅 propensity 미기록이라 로깅 정책을 시뮬레이션해야 하는 semi-synthetic 용도 — 본 레포에서는 합성 DGP가 그 역할을 대신하므로 **범위 밖**.

## 3. 차별화 갭 5개

각 갭은 웹 스윕에서 "공개 공간 부재"로 확인된 각도다. **공통 한계: 검색은 US-중심 엔진 결과라 부재 증명이 아니라 가시성 판단이다** — 아래 각 항에 커버리지 한계를 병기한다.

1. **의사결정 플레이북 계층.** 라이브러리는 estimator를 주지만 "언제 무엇을 믿을지"(ESS·max-weight·support 진단 → 판정)를 보여주는 공개 레포가 사실상 없다. 관련 지식은 논문·folklore로만 존재한다: IEOE(robustness 평가, https://arxiv.org/abs/2108.13703), ESS 관행·clipping 스케일 권고(https://arxiv.org/pdf/2207.00632). 2025–26에 산업 논문으로 gate 계층이 막 등장 중이나(Adyen https://arxiv.org/html/2501.10470 · marketplace launch-readiness https://arxiv.org/pdf/2605.12840) 공개 레포 형태는 여전히 부재. *한계: GitHub topic·검색 엔진 커버리지 밖의 사내 레포·비영어 자료는 확인 불가. 그리고 이 갭을 메우는 본 레포의 규칙은 어디까지나 제안이다(표준 아님).*
2. **시뮬(ground truth) + 실데이터 교차검증의 독립 포트폴리오.** obp 내부 프로토콜로는 존재하나(https://arxiv.org/abs/2008.07146), 번호 붙은 축별 실험 스크립트 형태의 독립 레포는 희소. 본 레포는 합성 DGP(참값 보유) + classification-to-bandit(축 11) + OBD small(축 12)의 3트랙. *한계: "희소"는 topic 36개 레포 표본 내 판단.*
3. **Confounding sensitivity.** 튜토리얼 대부분이 logged propensity를 참으로 가정한다; unobserved confounding 하 OPE는 논문으로만 존재하고(https://arxiv.org/pdf/2309.04222 · https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding · https://arxiv.org/abs/2003.05623) 포트폴리오 공간에는 거의 부재. 본 레포는 축 09(confounding 주입 + "진단이 못 보는 것" 대조표)로 이 각도를 커버하되, §5의 경계 선언대로 대조표(+조건부 Λ-sweep)에서 멈춘다. *한계: 논문 커버리지는 스윕의 검색 범위 내.*
4. **비즈니스 의사결정 프레임.** "OPE가 A/B 몇 회를 대체하나 / 어떤 조건에서 잘못된 정책을 골랐을까"를 배포 게이트로 프레이밍한 공개 예시가 사실상 전무 — Spotify WSDM'19가 유일한 근접 사례이나 논문 형태다(https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms). 본 레포의 축 10(잘못 배포할 확률·rank-corr)이 이 언어를 코드·figure로 구현한다. *한계: 동일 — 부재 증명 아님.*
5. **KO/EN 이중 문서.** 한국어 OPE 콘텐츠가 검색상 거의 부재해 KO/EN 이중화 자체가 국내 시장 차별화. *한계: 이 항이 커버리지 한계에 가장 취약하다 — US-중심 엔진에서 한국어 자료는 원래 잘 안 잡히므로 "공백" 판단의 신뢰도가 다섯 갭 중 가장 낮음을 명시해 둔다.*

## 4. 산업 수요 신호 — 간접 신호는 강함, JD 직접 근거는 미확인

**정직 고지: 채용공고(JD) 수준의 직접 수요 근거는 확인하지 못했다.** 스윕에서 잡힌 Spotify "Policy & Safety" 검색 결과는 콘텐츠 정책 직군이지 OPE가 아니다(과대해석 금지). 확인된 것은 기업 연구·기술블로그 수준의 간접 신호이며, 이는 "실무에서 쓰인다"의 근거이지 "채용 시장이 요구한다"의 근거가 아니다.

- **Spotify** — "Offline Evaluation to Make Decisions About Playlist Recommendation Algorithms"(WSDM'19): A/B 비용·리스크 절감이 명시 동기 (https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms)
- **Netflix** — budget-constrained recommendations의 off-policy 학습·평가 (https://netflixtechblog.com/reinforcement-learning-for-budget-constrained-recommendations-6cbc5263a32a)
- **Airbnb** — interleaving + counterfactual evaluation으로 search ranking 실험 가속, 2025 (https://arxiv.org/pdf/2508.00751)
- **Amazon** — two-stage recommender candidate generator의 OPE (https://www.amazon.science/publications/off-policy-evaluation-of-candidate-generators-in-two-stage-recommender-systems)
- 공통 등장 맥락: A/B 테스트 비용·리스크 절감 · 배포 전 알고리즘 선택 · 안전 배포 게이트 — 본 레포의 2막 서사(추정 → 신뢰 판정)와 정확히 겹친다.

## 5. 인접 레포 경계 선언표

포트폴리오 그래프 내 중복을 구조적으로 차단한다. 각 행은 README에도 상호 링크로 반영 예정.

| 인접 레포 | 그쪽의 주인공 | 본 레포와의 경계 |
|---|---|---|
| `kr_segmentation_causal_targeting_dunnhumby` | **binary treatment**(쿠폰 지급 여부)의 CATE 학습 → 타겟팅 정책. OPE는 부속 함수 수준 | 본 레포는 **multi-action**(수십~수천 액션)에서 **estimator 패밀리 자체가 주인공**인 벤치마크. 도메인도 소매 쿠폰 vs 추천 위젯으로 분리. OPL/CATE는 본 레포 범위 밖 → dunnhumby 참조 |
| `mta-simulation` | 어트리뷰션 DGP 시뮬레이션 | **하우스 스타일 계승** 관계(중복 아님): 번호 붙은 축별 실험 스크립트(`experiments/NN_slug.py`)·config 패턴을 본 레포 축 01–14에 그대로 이식 |
| `dag-registry` | 인과 설계 스펙 문서(YAML). `docs/dag-design.md` §6.5에 ESS·max-weight·clipping 진단 스펙 보유 | **스펙 ↔ 실증 cross-link**: dag-registry가 문서로 선언한 진단을 본 레포가 실행 코드·실험(축 08)으로 실증. 보완 관계, 상호 링크 |
| `causal-inference` | CATE 방법 카탈로그 | **접점 없음** — 링크만 |
| 연구 레포 (proximal 본류, decision-frontier) | hidden confounding 하 identification(proximal OTR/DTR) — 연구 정체성의 본류 | 본 레포는 **입구에서 의도적으로 멈춘다**: 축 09의 "진단이 못 보는 것" 대조표(+probe GO 시에만 조건부 Λ-sweep, 축 14)까지. proximal bridge·identification 본론은 연구 레포 소관 — 이중 게재 인상 차단 |

**범위 밖 명시(공통):** slate OPE(PseudoInverse/IIPS/RIPS) · RL OPE(FQE/DICE) · OPL · CATE. 레포 정체성은 "multi-action single-step logged bandit OPE 벤치마크 + 배포 게이트 플레이북"으로 고정한다.

## 6. 관련 문헌 핵심 리스트

스윕의 estimator 계보·스트레스 축 카탈로그에서 본 레포 코어·스트레치에 직접 대응하는 대표 논문만 추린다. (실험 축 ID는 계획 확정본: 01 표본 n / 02 로깅 β / 03 타깃-로깅 괴리 / 04 deficient support / 05 propensity 오지정 / 06 reward model 오지정 / 07 hyperparameter 민감도 / 08 진단 예보력+결정규칙 / 09 confounding 주입+대조표 / 10 의사결정 metric / 11 c2b / 12 OBD small / 13[스트레치] 액션 수+MIPS / 14[스트레치] Λ-sweep.)

### Estimator 계보 (코어 7 + 부속)

| 주제 | 문헌 | 대응 |
|---|---|---|
| DM·DR의 참조틀 | Dudík, Langford, Li (2011) — https://arxiv.org/abs/1103.4601 | DM·DR 구현의 기준 |
| IPS/IPW 계보 | Horvitz–Thompson (1952); Precup et al. (2000) — 스윕에서 원전 URL 미수집(고전 문헌), 필요 시 M4 문서화 단계에서 보강 | IPS 구현 |
| SNIPS | Swaminathan & Joachims (2015, NeurIPS) — https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html | SNIPS; "1줄 정규화로 분산 급감" 서사 |
| Switch-DR | Wang, Agarwal, Dudík (2017) — https://arxiv.org/abs/1612.01205 | Clipped-IPS·Switch-DR의 τ 다이얼 |
| DRos | Su et al. (2020) — https://arxiv.org/abs/1907.09623 | shrinkage = "clipping의 원리화" |
| MRDR | Farajtabar et al. (2018) — https://arxiv.org/abs/1802.03493 | 축 06 동기 + classification-to-bandit 변환 예시(축 11) |
| SLOPE | Su, Srinath, Krishnamurthy (2020, ICML) — http://proceedings.mlr.press/v119/su20d/su20d.pdf | hyperparameter 선택 부속 |
| MIPS | Saito & Joachims (2022, ICML) — https://arxiv.org/abs/2202.06317 | 스트레치 축 13 |

### 스트레스 축·평가 프로토콜

| 주제 | 문헌 | 대응 축 |
|---|---|---|
| OBD/OBP 벤치마크 프로토콜 | Saito et al. (2021, NeurIPS D&B) — https://arxiv.org/abs/2008.07146 · https://research.zozo.com/data.html | 01·02·12 |
| IEOE(robustness 평가) | Saito et al. (2021, RecSys) — https://arxiv.org/abs/2108.13703 | 07 |
| Deficient support | Sachdeva, Su, Joachims (2020, KDD) — https://arxiv.org/abs/2006.09438 | 04 |
| Propensity 오지정 하 DR | DRUnknown (2024) — https://arxiv.org/pdf/2404.01830 | 05 |
| Estimator 선택 | Udagawa et al. (2023, AAAI, PAS-IF) — https://arxiv.org/abs/2211.13904 · 프로토콜 표준화 (2025) — https://arxiv.org/pdf/2502.08021 | 03·08 |
| 의사결정 metric(SharpeRatio@k) | Kiyohara et al. (2024, ICLR 계열) — https://arxiv.org/abs/2311.18207 | 10 |
| Unobserved confounding 하 OPE | Namkoong et al. (2020, NeurIPS) — https://arxiv.org/abs/2003.05623 · RecSys'23 — https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding | 09 |
| MSM Λ-sensitivity | Kallus & Zhou (2018) — https://arxiv.org/pdf/1805.08593 | 스트레치 축 14 (probe GO 조건부) |

### 인접 동향 — 범위 밖 선언의 근거 (참고용)

아래는 스윕에서 확인된 인접 전선이다. 본 레포가 다루지 **않는** 이유를 명시하기 위해 기록한다.

- **Estimator 선택·집계의 자동화** — OPERA(가중집계, https://arxiv.org/abs/2405.17708), 지도학습 기반 선택(https://arxiv.org/abs/2406.18022), cross-validated OPE(https://arxiv.org/abs/2405.15332). 진단→선택을 학습으로 대체하려는 연구 전선이며 표준 결정규칙은 미확립 — 본 레포의 gate 규칙이 "제안" 프레임을 유지해야 하는 이유이자, 축 08(진단 예보력)의 대조 배경.
- **Δ-OPE** — 정책 *쌍*의 가치 차이를 직접 추정해 분산 절감 (Jeunen et al., RecSys 2024, https://arxiv.org/abs/2405.10024). 흥미롭지만 코어 서사(단일 정책 가치 + 신뢰 판정) 밖.
- **대규모 액션에서의 OPE vs OPL** — 추정 정확도보다 최적화 안정성이 중요하다는 2025 통찰 (https://arxiv.org/abs/2509.03456). OPL이 범위 밖인 이유를 설명할 때 인용.
- **Slate/ranking OPE** — PseudoInverse (Swaminathan et al. 2017, https://arxiv.org/abs/1605.04812) · IIPS (Li et al. 2018) · RIPS (McInerney et al., KDD 2020) — 위치별 독립성 가정의 사다리. 범위 밖(README 선언).
- **RL(sequential) OPE** — FQE (Le et al. 2019, https://arxiv.org/abs/1903.08738) · DualDICE (Nachum et al. 2019, https://arxiv.org/abs/1906.04733) · 분포적 OPE(SCOPE-RL 구현, https://arxiv.org/abs/2311.18206). 범위 밖.
- **대규모 액션 estimator 후속** — OffCEM (Saito et al., ICML 2023, https://proceedings.mlr.press/v202/saito23b/saito23b.pdf) · POTEC (Saito, Yao, Joachims, ICLR 2025, https://arxiv.org/abs/2402.06151). 축 13(MIPS) 채택 시 개념 소개 수준으로만 언급.
- **LLM 접점** — DR alignment for LLMs(https://arxiv.org/abs/2506.01183) 등이 있으나 "LLM용 OPE"의 canonical 단일 논문은 아직 없음(스윕 판단 — 불확실 유지). 범위 밖.

---

## 결론 — 선점 판정

공개 OPE 공간은 저혼잡(topic 36개 레포)이고, 라이브러리·튜토리얼·논문코드 사이의 **의사결정 플레이북 계층이 공백**이다. 사실상 표준 라이브러리(obp)는 stale이라 직접 구현 + 적대 교차검증 전략이 리스크 회피와 차별화를 동시에 달성한다. 인접 레포와의 경계는 §5 선언표로 잠갔고, 연구 정체성(hidden confounding)과의 연결은 축 09 대조표 한 장의 저비용 티저로 제한한다 — proximal 본류는 연구 레포 소관이며 본 레포는 입구에서 멈춘다.

해소 경과(M0-B, 2026-08-06): obp/sb-obp 릴리스 버전 **해소**(obp PyPI 0.5.7/2023-04-14 · sb-obp 0.5.10/2025-08-19 — 정본 `results/tables/probe_obp_pypi_check.json`), obp pinned env 설치 성공 여부 **해소**(py3.9 설치 성공·교차검증 GO — 단 `matplotlib<3.7` 핀 필수, `results/tables/probe_obp_crossval.json`). 잔여 미해소: Λ-최적화의 multi-action 수치 안정성(→ 축 14 착수 전 1일 probe), 한국어 콘텐츠 공백의 실재(→ 커버리지 한계로 상시 불확실 유지).

## 문서 관계

- 상류: 승인 계획(`~/.claude/plans/off-policy-evaluation-misty-shannon.md`) — 실험 축 ID·마일스톤·게이트의 확정본. 본 문서는 그 계획의 Stage 1 근거층이다.
- 자매: `docs/CONCEPT.md`(Stage 0 — 동기→메커니즘→검증가능 주장), `PLAN.md`(마일스톤·게이트), `docs/LEDGER.md`(정본 수치 — 현재 빈 틀; 본 문서에는 실험 수치가 없어야 정상).
- 원자료: 11-에이전트 스윕 전문(스크래치패드 `tasks/w7e2t88o5.output`) — 본 문서에 없는 세부(estimator 난이도 평가·bridge 실현성 평가 등)는 원문 참조.
- 갱신 규칙: 외부 사실(star 수·릴리스·유지보수 상태)은 검색 시점 스냅샷이므로, publish 직전(M4) 주요 항목을 1회 재확인하고 변경 시 날짜와 함께 수정한다.
