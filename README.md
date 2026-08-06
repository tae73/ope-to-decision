# ope-decision-gate

> **A/B 테스트 전에, 이미 쌓인 추천 로그만으로 새 정책의 가치를 추정하고 — 그 추정을 언제 믿으면 안 되는지까지 판정하는 multi-action OPE 벤치마크 + 배포 게이트 플레이북.**

🇰🇷 한국어 (정본) · 🇺🇸 English — *EN twin 은 M4 에서 저작 (`README.en.md` — 자리만 예약)*

<!-- badges: TBD — CI·license·python 배지는 publish 시점(M4)에 확정 -->

---

> **⚠️ 프로젝트 상태: M0 — 스캐폴드 단계.** 지금 이 레포에는 설계 문서·src 스텁·de-risk probe 만 있다.
> estimator 본구현은 M1, 축 실험·figure 는 M2–M3, 이 README 의 본문 완성은 M4 에 온다.
> 마일스톤·게이트·폴백은 [PLAN.md](PLAN.md) 참조. **아래의 모든 결과 자리는 의도적으로 TBD 다.**
> M0 de-risk probe 2종 GO(상태 표기 — 수치는 [docs/LEDGER.md](docs/LEDGER.md) 정본).

## ⏱️ TL;DR — 30초

- **문제:** 새 추천 정책을 트래픽에 태우기 전에 가치를 알고 싶다. A/B 슬롯은 한정이고 나쁜 정책은 매출·UX 를 태우는데, 이미 쌓인 로그는 *옛 정책이 고른 행동만* 기록했다 — Spotify 가 WSDM'19 에서 명시한 바로 그 동기다 ([논문](https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms)).
- **접근:** off-policy evaluation (OPE) estimator family — DM · IPS · SNIPS · Clipped-IPS · DR · Switch-DR · DRos — 를 numpy 로 직접 구현해 obp 로 적대 교차검증하고, ground truth 를 아는 합성 DGP 에서 축별로 부러뜨린 뒤(축 01–10), 진단(ESS · max-weight · support)이 그 부러짐을 언제 예보하고 언제 원리적으로 못 보는지를 "믿는다 / 못 믿는다 / A/B 회귀" **배포 게이트 decision rule** 로 체계화한다. 실데이터 게이트(classification-to-bandit · OBD small)까지.
- **핵심 결과:** **TBD (M2–M3)** — 모든 수치는 실험 완료 후 [docs/LEDGER.md](docs/LEDGER.md) 경유로만 기입된다.

## 핵심 결과 — hero 3장 (자리)

| ① decision-gate 플로차트 | ② regime map | ③ 진단이 못 보는 것 |
|---|---|---|
| *(예정 — M3·M4, `assets/`)* 로그 진단 → 기각 → estimator 선택 → 믿는다/못 믿는다/A/B 회귀의 3-way 판정 SVG | *(예정 — M2)* 표본 × overlap grid 에서 최저-MSE estimator 히트맵 — 플로차트의 증거층 | *(예정 — M2, 축 09)* unconfounded vs confounded 에서 ESS·max-weight 는 똑같이 양호한데 bias 만 갈리는 대조표 |

## 읽는 방법 (레이어)

| 시간 | 읽을 것 |
|---|---|
| **30초** | 위 TL;DR + hero 3장 |
| **5분** | [2막 서사](#2막-서사) + [실험 축](#실험-축-id-불변) + [비범위](#비범위-경계-선언) |
| **재현** | [Quick Start](#quick-start) + [experiments/README.md](experiments/README.md) |
| **30분** | [docs/CONCEPT.md](docs/CONCEPT.md) · [docs/POSITIONING.md](docs/POSITIONING.md) · [PLAN.md](PLAN.md) |

## 2막 서사

**1막 — 로그로 정책을 평가한다.** 이커머스 추천 위젯 팀이 새 정책 후보를 만들었다. A/B 전에 어제까지의
로그만으로 새 정책의 기대 보상 $V(\pi_e)$ 를 추정한다 — Netflix·Airbnb·Amazon 도 같은 맥락의 동기를
공개한 바 있다 ([Netflix](https://netflixtechblog.com/reinforcement-learning-for-budget-constrained-recommendations-6cbc5263a32a) ·
[Airbnb](https://arxiv.org/pdf/2508.00751) · [Amazon](https://www.amazon.science/publications/off-policy-evaluation-of-candidate-generators-in-two-stage-recommender-systems)).
estimator 는 여기서 *도구*로 등장한다: DM 의 model bias 극단에서 IPS 의 variance 극단까지,
SNIPS → DR → Switch-DR/DRos 로 이어지는 bias-variance 아크.

**2막 — 그 추정을 믿어도 되나.** 진짜 질문은 점추정이 아니라 *신뢰 판정*이다. ESS·max-weight·support
진단이 위험을 예보하는 regime 과, 미관측 교란(unobserved confounding) 처럼 진단이 **원리적으로 못 보는** regime 을 ground truth 로
갈라 보인 뒤([Amazon Science RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding)),
"믿는다 / 못 믿는다 / A/B 로 보낸다"의 배포 게이트로 체계화한다. 이 decision rule 은 **본 레포의 제안
(문헌에 산발적으로 존재하는 folklore 의 체계화 시도)이지 확립된 표준이 아니다.**

## 실험 축 (ID 불변)

상세 계약(slug·스윕 노브·산출물 규약)은 [experiments/README.md](experiments/README.md). 전 축 **실행 전** 상태.

| ID | 축 | 한 줄 |
|---|---|---|
| 01 | 표본 크기 n | 소표본 DM 우세 ↔ 대표본 IPS/DR 우세의 regime 교차 |
| 02 | 로깅 정책 β (overlap) | 준결정적 로깅 → overlap 축소 → weight 폭발 → IPS/DR 붕괴 |
| 03 | 타깃–로깅 괴리 | 평가 정책이 로깅에서 멀어질수록 estimator 순위 역전 |
| 04 | deficient support | 지지 결핍에서 IPS 계열의 파국적(식별 불능) 실패 |
| 05 | propensity 오지정 | true→estimated→noised — DR 이 한쪽 모델로 생존하는 조건 |
| 06 | reward model 오지정 | DM bias 의 표본 불감성과 DR 의 보정 한계 |
| 07 | hyperparameter 민감도 | IEOE 식 error-CDF — 튜닝 없는 고급 estimator 의 불안정성 |
| 08 | 진단 예보력 + 결정규칙 | ESS·max-weight vs 실오차 산점 → decision rule 종합(제안) |
| 09 | confounding 주입 + 대조표 | 진단은 동일 양호, bias 만 상이 — "진단이 못 보는 것" (hero ③) |
| 10 | 의사결정 metric | MSE 대신 잘못-배포 확률·rank-corr — 선택 안전성의 분기 |
| 11 | c2b 멀티데이터셋 | classification-to-bandit (UCI/OpenML) 깨끗한 GT 체계 벤치 |
| 12 | OBD small 게이트 | ZOZO 실데이터 근사 GT 로 synthetic 결론 재현 + 교차검증 |
| 13 | [스트레치] 액션 수 + MIPS | 액션 폭발에서 IPS/DR 분산 붕괴와 MIPS 의 구원 |
| 14 | [스트레치] Λ-sweep | MSM Λ-sensitivity 구간 + 정책 순위가 뒤집히는 breakdown Λ* |

## 비범위 (경계 선언)

- **OPL · CATE · 정책 학습** — 범위 밖. binary-treatment 정책·CATE 는
  [kr_segmentation_causal_targeting_dunnhumby](../kr_segmentation_causal_targeting_dunnhumby/), CATE 방법
  카탈로그는 [causal-inference](../causal-inference/) 소관 (상호 링크).
- **slate/ranking OPE (PI·IIPS·RIPS) · RL OPE (FQE·DICE)** — 범위 밖. 본 레포의 정체성은
  *multi-action single-step logged bandit* OPE 다.
- **confounding 하 식별의 본류(proximal 등)** — 연구 레포 소관. 본 레포는 축 09 의 "진단이 못 보는 것"
  대조표(+ probe GO 시 조건부 축 14 Λ-sweep)에서 **의도적으로 멈춘다**.
- 진단 스펙 문서 ↔ 실행 구현: [dag-registry](../dag-registry/) 와 보완 관계. 축별 실험 패턴은
  [mta-simulation](../mta-simulation/) 하우스 스타일 계승.

## Quick Start

```bash
cd ope-decision-gate
uv sync                    # 본 env (Python 3.11+)
uv run pytest              # M0: 패키지 임포트 + 스텁 계약 smoke

# PROBE M0-A — DGP + estimator 최소구현 sanity (GO/NO-GO)
uv run python experiments/probes/probe_dgp_estimator_sanity.py

# PROBE M0-B — obp 교차검증 (별도 pinned Python 3.9 env에서 실행; 셋업은 experiments/README.md 참조)
```

## Repository Structure

```
ope-decision-gate/
├── src/ope/               # estimators · dgp · diagnostics · policies · datasets (M0: 스텁)
├── experiments/           # 실험 인덱스(README) + probes/ (축 01–14 스크립트는 M2–M3)
├── configs/               # Hydra: config.yaml + dgp/default.yaml (설계 기본값 — 결과 수치 아님)
├── tests/                 # M0 smoke → M1 에서 property test 로 확장
├── results/figures|tables # 실험 산출물 (NN_* 1:1 규약) — 현재 probe JSON 만, figures/ 는 빈 디렉토리 (M2–M4 생성 예정)
├── docs/                  # CONCEPT · POSITIONING · LEDGER · GLOSSARY
├── data/                  # gitignore — 원본 재배포 금지, 배치법은 data/README.md
├── assets/  notebooks/    # hero SVG(M3–M4) · 노트북(후속) — 현재 빈 디렉토리 (M2–M4 생성 예정)
└── PLAN.md  CLAUDE.md     # 마일스톤·게이트 / 에이전트 규약
```

## 정직성 각주

1. **수치는 LEDGER 경유만.** 이 README 를 포함한 모든 문서의 수치는 committed 결과
   (`results/tables/`) 에서 만든 [docs/LEDGER.md](docs/LEDGER.md) 를 거쳐야 기입된다 — 반올림·자작 금지.
   현재 LEDGER 는 빈 틀이고, 따라서 이 스켈레톤에는 실험 결과 수치가 0개다.
2. **decision rule = 제안.** 축 08 의 배포 게이트는 본 레포의 제안(folklore 체계화 시도)이며 표준이
   아니다. 실패 조건 — 특히 축 09 의 confounding blind spot — 을 함께 전시한다.
3. **불확실 태그.** obp 의 마지막 릴리스 버전은 조사 출처 간 표기가 불일치한다(불확실) — probe M0-B 의
   PyPI 재확인으로 해소 예정이며 그 전까지 단정하지 않는다. OBD small 의 근사 ground truth 는 자체 표본
   오차를 가지므로(불확실) 관련 figure 전부에 bootstrap CI 를 병기하고 점 비교를 단정하지 않는다.
4. **데이터 보호.** `data/` 는 라이선스상 재배포 금지로 gitignore 되어 있다 — [data/README.md](data/README.md).

*License: MIT · 저작: 한국어 정본 우선, EN twin 은 M4 (GLOSSARY 정합).*
