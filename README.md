# ope-to-decision

**From logged bandit feedback to deployment decisions.**

> **A/B 테스트 전에, 이미 쌓인 추천 로그만으로 새 정책의 가치를 추정하고 — 그 추정을 언제 믿으면 안 되는지까지 판정하는 multi-action OPE 벤치마크 + 배포 게이트 플레이북.**

🇰🇷 한국어 (정본) · 🇺🇸 English — *EN twin 은 M4 에서 저작 (`README.en.md` — 자리만 예약)*

<!-- badges: TBD — CI·license·python 배지는 publish 시점(M4)에 확정 -->

---

> **⚠️ 프로젝트 상태: M3 완료 — 본문 저작(M4) 전.** src 본구현·obp/sb-obp 교차검증 GO·코어 축 01–10 +
> 실데이터 축 11–12 실행 완료(figure+CSV 페어)·decision-gate 플레이북([docs/PLAYBOOK.md](docs/PLAYBOOK.md))·
> hero 3장 확정. 이 README 의 결과 서술·EN twin·KO 플로차트는 **M4 에서 저작**된다 — 그 전까지 여기의
> 결과 요약은 의도적으로 자리 표시다. 수치 정본: [docs/LEDGER.md](docs/LEDGER.md)(M0–M3 15행 등재).

## ⏱️ TL;DR — 30초

- **문제:** 새 추천 정책을 트래픽에 태우기 전에 가치를 알고 싶다. A/B 슬롯은 한정이고 나쁜 정책은 매출·UX 를 태우는데, 이미 쌓인 로그는 *옛 정책이 고른 행동만* 기록했다 — Spotify 가 WSDM'19 에서 명시한 바로 그 동기다 ([논문](https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms)).
- **접근:** off-policy evaluation (OPE) estimator family — DM · IPS · SNIPS · Clipped-IPS · DR · Switch-DR · DRos — 를 numpy 로 직접 구현해 obp 로 적대 교차검증하고, ground truth 를 아는 합성 DGP 에서 축별로 부러뜨린 뒤(축 01–10), 진단(ESS · max-weight · support)이 그 부러짐을 언제 예보하고 언제 원리적으로 못 보는지를 "믿는다 / 못 믿는다 / A/B 회귀" **배포 게이트 decision rule** 로 체계화한다. 실데이터 게이트(classification-to-bandit · OBD small)까지.
- **핵심 결과:** 실험은 완료(축 01–12), 수치는 전부 [docs/LEDGER.md](docs/LEDGER.md) 등재 — **본문
  결과 서술은 M4 저작 예정**(LEDGER 경유 원칙 유지).

## 핵심 결과 — hero 3장 (확정 — M3)

| ① decision-gate 플로차트 | ② regime map | ③ 진단이 못 보는 것 |
|---|---|---|
| [flowchart SVG](assets/decision_gate_flowchart_en.svg) — 로그 진단 → 기각 → 선택 → 믿는다/못 믿는다/A/B 회귀 (EN; KO twin 은 M4) | [regime map](results/figures/hero_regime_map.png) — n × β_log 28-cell 최저-MSE 승자 지도 + 게이트 다수결 | [대조 figure](results/figures/09_confounding_blindspot.png) — 진단은 평평·bias 만 성장 (축 09) |

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

상세 계약(slug·스윕 노브·산출물 규약)은 [experiments/README.md](experiments/README.md).
**01–12 실행 완료**(figure+CSV 페어), 13–14 는 스트레치(각 1일 probe 선행 조건부).

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
| 12 | OBD small 게이트 | ZOZO 실로그 uniform-target 단방향(근사 GT bootstrap CI·구간 비교) + decision gate 시연 — 판별력 없음 사전 선언, obp 교차검증·역방향은 스코프 밖 |
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
cd ope-to-decision
uv sync                    # 본 env (Python 3.11+)
uv run pytest              # M0: 패키지 임포트 + 스텁 계약 smoke

# PROBE M0-A — DGP + estimator 최소구현 sanity (GO/NO-GO)
uv run python experiments/probes/probe_dgp_estimator_sanity.py

# PROBE M0-B — obp 교차검증 (별도 pinned Python 3.9 env에서 실행; 셋업은 experiments/README.md 참조)
```

## Repository Structure

```
ope-to-decision/
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
   현재 LEDGER 에는 M0–M3 15행이 등재돼 있고, 이 README 의 결과 서술은 M4 저작 시 그 행들만 인용한다.
2. **decision rule = 제안.** 축 08 의 배포 게이트는 본 레포의 제안(folklore 체계화 시도)이며 표준이
   아니다 — M1 사전등록 임계값을 축 08 에서 **평가만** 했고(무튜닝) 실패 조건 — 특히 축 09 의
   confounding blind spot — 을 함께 전시한다. 상세: [docs/PLAYBOOK.md](docs/PLAYBOOK.md).
3. **불확실 태그의 해소 기록.** obp 릴리스 버전 상충은 probe M0-B 의 PyPI 재확인으로 **해소**되었다
   (LEDGER `m0b-pypi`). OBD small 의 근사 ground truth 는 자체 표본 오차를 가지므로 관련 figure 전부에
   bootstrap CI 를 병기하고 점 비교를 단정하지 않는다(축 12 실행에서 준수 — LEDGER `m3-12-gate-demo`).
4. **데이터 보호.** `data/` 는 라이선스상 재배포 금지로 gitignore 되어 있다 — [data/README.md](data/README.md).

*License: MIT · 저작: 한국어 정본 우선, EN twin 은 M4 (GLOSSARY 정합).*
