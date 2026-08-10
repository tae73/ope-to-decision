# COMMS_BRIEF v2 — M8 재정위(GT-미상 본편) 저작 기준

> **지위.** M8 문서 역전(PLAN §4.9 Stage 5)의 comms design 기준 — [COMMS_BRIEF.md](COMMS_BRIEF.md)(M4)
> 를 **계승**한다. v1 은 동결 기록으로 무접촉 유지(자체 갱신 금지 조항). 충돌 시 README 정본이 이긴다.
> 수치 규율은 v1 과 동일: 모든 수치는 LEDGER 행 경유·행 id 병기·축약 표기는 원값 기준 반올림.

## 1. 새 내러티브 아크 (4문장)

1. **훅**: 당신은 로그와 후보 정책만 가졌다 — 실무에서 참값 V(π_e) 는 아무도 모른다.
2. **동기(구 그림 3 전환)**: 참값을 아는 무대에서 우리는 이미 보았다 — 진단이 평평한 채 bias 만
   자라는 실패가 실재하고, 실무에선 그 bias 곡선 자체를 볼 수 없다.
3. **본론**: 그래서 로그만으로 계산 가능한 신호(진단·게이트 + validity battery + Λ\*)를 사전등록
   프로토콜로 조립하고, 그 신호들이 실제로 오류를 예보하는지·어디서 눈머는지를 참값 보유
   백스테이지에서 blind-then-reveal 로 채점했다.
4. **정직한 절정**: battery 는 misrecording·support 결핍을 잡지만(축 17), marginally calibrated
   confounding 은 관측 동등성으로 **원리적으로 못 잡는다**(축 18) — 출구는 Λ-감도 구간의 폭과
   abstention 이며, 그 경계까지가 이 레포의 지도다.

## 2. 무대 언어 (GLOSSARY §8 정본)

**본편(frontstage)** = 로그 층만으로 계산·판정 — 축 12·14·17–20. **백스테이지(backstage)** =
참값 보유 채점(합성 v_true·c2b 정확 라벨·OBD 근사 band) — 축 01–11·15–16·hero map + 전 reveal.
battery 는 항상 **"필요조건 검사(falsifier)"** 로만 명명 — "GT-free 검증/보증" 금지, 주장 문단마다
calibrated-confounding blind co-exhibit(CLAUDE.md §5).

## 3. hero 큐레이션 (M8)

| 위치 | figure | 역할 |
|---|---|---|
| 동기(도입) | `results/figures/09_confounding_blindspot.png` | **motivation 재캡션** — "실무에선 이 그림의 bias 패널을 볼 수 없다"; 파일·수치 불변(`m2-09-blindspot`) |
| 그림 1 | `assets/decision_gate_flowchart_ko.svg` | 유지 — "본편: 전부 참값 불필요" 강조 |
| 그림 2 | `results/figures/17_validity_battery.png` | detection matrix — 잡는 것과 못 보는 빈칸(`m8-17-matrix`) |
| 그림 3 | `results/figures/20_obd_decision_card.png` | 실전 카드 — reveal 없는 판정(`m8-20-card`) |
| 3막 인라인 | `hero_regime_map.png`·`18_calibrated_boundary.png`·`19_blind_decision.png` | 백스테이지 증거층 — regime map 의 "소표본 검정력 실종" 경고는 2막에서 교차 인용 의무 |

## 4. 배지 ↔ LEDGER 매핑 (v2)

`real_log_card AB_FALLBACK·fragile`(`m8-20-card`) → `decision_value naive 0.9 → protocol 0.0
false-go`(`m8-19-decision-value`) → `backstage forecast 4.6% vs 44.4%`(`m2-08-forecast`) →
`DR robustness 4/4`(`m3-11-dr-robust`) → `obp crossval`(`m1-crossval`) → `axes 01–12·14–20` →
`tests 97`(공정 메타) → `license`.

## 5. 축표 2-tier 규칙 (4벌 동기)

README(KO/EN) 발견 표는 **본편 표 → 백스테이지 표** 2단 분리(tier 내 ID 오름차순, 리드 문장:
"축 번호는 실행 이력 순서 — 서사 순서가 아니며 재부여하지 않는다"). PLAN §2·experiments/README
는 단일 ID-정렬 표 유지 + `무대` 열 추가. CLAUDE.md §4 는 정본 포인터.

## 6. figure 라벨 정책

기존 백스테이지 figure(01–16·hero)는 재생성·재라벨 금지 — GT 라벨("MSE vs true value" 등)은 그
무대의 정직한 표식이며, 백스테이지 표기는 캡션에서만 얹는다. 신규 본편 figure(17–20)는 GT-free
라벨만("E[w]", "fire rate", "decision share"; reveal 패널은 "backstage" 명기).

## 7. 정직성 프레임 추가분 (각주 ⑤·⑥)

⑤ battery = 필요조건 검사 — 통과 ≠ 무결; 기록 propensity 기반 신호 전부(ESS·E[w]·harmonic)는
calibrated confounding 에 공동 blind(축 09→18 계승, `m8-18-boundary` co-exhibit).
⑥ Λ\*·fragile 은 robustness 인증서가 아니라 취약성 보고서 — Λ 는 식별 불가 가정이며 fragile
임계 1.5 는 [제안 — 라벨만]. novelty 계열 문장은 POSITIONING §7(URL 스윕) 확정 전 README 반입 금지.
