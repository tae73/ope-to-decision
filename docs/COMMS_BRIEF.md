# COMMS BRIEF — M4 저작 설계 (portfolio-design Stage 2 산출물)

> **지위:** README(KO 정본·EN twin) 저작의 설계 기준. Stage 1(LEDGER) 게이트 통과 확인 후 작성됨.
> 저작 후에는 기록용 — README 가 이 브리프와 어긋나면 README 가 정본이다(이 문서는 갱신하지 않는다).

## Stage 1 게이트 확인 (저작 전 필수)

README 가 인용할 headline 수치는 전부 LEDGER 에 ENTERED 상태다(2026-08-06 기준 15행):
`m1-crossval` · `m2-07-slope` · `m2-08-forecast` · `m2-09-blindspot` · `m2-04-proxy-blind` ·
`m2-10-comparison` · `m3-11-dr-robust` · `m3-12-gate-demo` · `m3-hero-map` — **부족분 없음 → 통과.**
README 의 축약 표기(예: 4.6%)는 LEDGER 원값(0.045911…) 기준 반올림임을 정직성 각주에 명시한다.

## 내러티브 아크 (한 문장씩)

1. **훅:** 모두가 쌓인 로그로 새 정책을 평가하고 싶어한다(A/B 는 비싸고 위험하다).
2. **전환:** estimator 는 도구일 뿐이다 — 진짜 질문은 "이 추정을 *언제* 믿는가".
3. **본론:** ground truth 를 아는 곳에서 7종 estimator 를 12축으로 부러뜨려 승자 지도와
   진단→판정 게이트를 만들었고, 실데이터에서 재현했다.
4. **정직한 절정:** 게이트는 예보한다(4.6% vs 44.4%) — 그러나 confounding 앞에서는 원리적으로
   눈멀며(ESS 평평·bias 성장), 그 경계선까지가 이 레포의 산출물이다.

## 청중 레이어

| 층 | 대상 | 구성 |
|---|---|---|
| 30초 | 채용자·스캐너 | 수치 배지(LEDGER 기반) + TL;DR 3줄 + hero 3장 |
| 5분 | DS·엔지니어 | 2막 서사 + 구성요소·검증 3중 + 축별 발견 표 + 한계·정직 스코핑 |
| 재현 | 실행자 | Quick Start + env 함정 + experiments/README |
| 30분 | 방법론 독자 | PLAYBOOK → CONCEPT → POSITIONING → LEDGER |

## hero 큐레이션 (M3 확정 3장 — 배치·캡션)

1. **decision-gate 플로차트**(`assets/decision_gate_flowchart_ko.svg` — KO README 에는 KO 판,
   EN twin 에는 `_en.svg`): 레포의 "결론"을 한 장으로. 캡션에 "제안 — 표준 아님" 병기.
2. **regime map**(`results/figures/hero_regime_map.png`): 플로차트의 증거층. 캡션에 최대 발견
   ("게이트 검정력의 소표본 실종")을 담는다 — 자랑이 아니라 경고가 hero 인 것이 이 레포의 차별점.
3. **축 09 대조**(`results/figures/09_confounding_blindspot.png`): 진단이 못 보는 것 — 연구 경계
   선언과 연결.

## 배지 (전부 LEDGER 행 매핑 — shields.io static)

| 배지 | LEDGER 행 |
|---|---|
| `gate forecast: 4.6% trust vs 44.4% fallback` | m2-08-forecast (0.045911…/0.444444…) |
| `DR robustness: 4/4 real datasets` | m3-11-dr-robust |
| `obp crossval: rel diff ≤ 1e-8 (7 est × 2 tracks)` | m1-crossval |
| `axes: 01–12 executed` / `tests: 52 passed` | 공정 사실(결과 수치 아님 — LEDGER 불요) |

## 정직 스코핑 섹션 설계 (한계를 결과의 일부로)

불발·부정 결과를 "Limitations" 구석이 아니라 **"부러지지 않은 것들 — 기대가 틀렸던 곳"** 소제목으로
5분 층 본문에 배치: ① 02 cliff 는 β=8 아닌 8→16 사이 ② 03 DM 역전 불발(유계 weight — 진짜 붕괴는
naive clipping) ③ 04 support proxy 전면 blind ④ 10 초기 null → 비교 설계 속성 발견으로 승격
⑤ 12 판별력 없음 사전 선언. + decision rule 은 사전등록·무튜닝 평가(교정 아님), 합성 DGP 는 단일
구조(struct_seed=7) 조건부, c2b 는 결정적 reward, 연구 경계(proximal 은 연구 레포 소관).

## 비주얼·표기 시스템

- figure 는 기존 산출물 재사용(재생성 금지) — entity 고정 색 팔레트 승계.
- 수치 표기: LEDGER 원값 또는 명시적 반올림(각주에 규칙). "약/대략" 표현으로 새 수치 생성 금지.
- 용어: docs/GLOSSARY.md 단일기준(estimator 이름·약어 영어 유지). EN twin 은 자연 재작성(직역 금지).
