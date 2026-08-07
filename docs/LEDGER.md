# LEDGER — 정본 수치 단일 진실표

> **지위:** 이 표는 `ope-to-decision` 레포에 등장하는 **모든 실험 수치의 유일한 정본**이다.
> README·docs·figure 캡션·발표자료의 어떤 수치도 이 표를 경유하지 않고는 존재할 수 없다.
> 용어는 [`docs/GLOSSARY.md`](GLOSSARY.md), 마일스톤·실험 축은 `PLAN.md`를 따른다.

## 규칙

1. **단일 진실(single source of truth).** 문서에 쓰이는 모든 실험 수치는 이 표에 등재된 행을 인용한다.
   표에 없는 수치는 존재하지 않는 수치다 — 저작 중 필요한 수치가 표에 없으면, 문서를 고칠 것이 아니라
   실험을 돌리고 결과를 commit한 뒤 여기 등재한다.
2. **Committed 산출물만 등재.** `source` 는 레포에 commit된 `results/` 하위 파일(JSON·CSV)이어야 한다.
   노트북 셀 출력·터미널 로그·기억·중간 스크래치 파일은 등재 불가.
3. **반올림·자작 금지.** `수치` 필드는 source 파일의 값을 **verbatim**(그대로)으로 적는다. 문서 본문에서
   유효숫자를 줄여 표기할 필요가 있으면 이 표의 원값을 기준으로 하되, 해당 행의 비고에 반올림 규칙을 명시한다.
   존재하지 않는 수치의 자작·보간·"대략" 추정은 금지.
4. **실패도 등재.** probe NO-GO, 교차검증 불일치, NaN/발산, 설치 실패 같은 부정적 결과도 동일 스키마로
   등재한다 — 실패의 은폐는 이 레포의 정직성 규약 위반이다.
5. **행은 삭제하지 않는다.** 재실험으로 수치가 바뀌면 기존 행의 상태를 `SUPERSEDED`로 바꾸고 새 행을
   추가한다(이력 보존).
6. **실험 ID 불변.** `생성 실험 ID` 는 `PLAN.md`의 확정 축 ID를 그대로 쓴다:
   probe `M0-A`(DGP·estimator sanity) / probe `M0-B`(obp 교차검증) /
   `01` 표본 n / `02` 로깅 β / `03` 타깃-로깅 괴리 / `04` deficient support / `05` propensity 오지정 /
   `06` reward model 오지정 / `07` hyperparameter 민감도(IEOE) / `08` 진단 예보력+결정규칙 /
   `09` confounding 주입+대조표 / `10` 의사결정 metric / `11` c2b 멀티데이터셋 / `12` OBD small 게이트 /
   `13`[스트레치] 액션 수+MIPS / `14`[스트레치] Λ-sweep.
7. **VERDICT 는 수치가 아니다.** VERDICT 문자열(GO/NO-GO/INSTALL-FAIL)은 수치가 아닌 상태 표기로서,
   `PLAN.md` 등 진행 문서가 LEDGER 등재 전에도 인용할 수 있다(수치 인용은 불가).

## 표 스키마

| 필드 | 정의 |
|---|---|
| `id` | 행 식별자, kebab-case: `<실험ID 소문자>-<지표 slug>` (예: `m0a-ips-bias`, `01-dr-mse-n800`). 한번 부여하면 불변. |
| `수치` | source 파일의 값 verbatim. 미등재 시 `(미등재)`. |
| `단위` | 물리/논리 단위. 무차원 추정치는 `—`. |
| `source 파일 경로` | 레포 루트 기준 committed 산출물 경로 (`results/tables/...` 등). |
| `생성 실험 ID` | 위 규칙 6의 확정 ID (`M0-A`/`M0-B`/`01`–`14`). |
| `등재일` | `YYYY-MM-DD`. |
| `상태` | `RESERVED`(경로만 예약, 수치 미등재) / `ENTERED`(수치 등재 완료 — NO-GO 등 실패 결과 포함) / `SUPERSEDED`(재실험으로 대체됨). |

## 수치 표 (현재: M0 4행 · M1 1행 · M2 6행 · M3 4행 등재)

M0 de-risk probe 2개(`M0-A`·`M0-B`)의 산출 JSON 4개가 **초기 커밋에 동반 commit** 되어 아래와 같이
등재되었다(등재일 2026-08-06). `수치` 필드는 각 source JSON 의 값 verbatim 이다(규칙 3 — 반올림 금지).

| id | 수치 | 단위 | source 파일 경로 | 생성 실험 ID | 등재일 | 상태 |
|---|---|---|---|---|---|---|
| `m0a-dgp-sanity` | VERDICT `GO` · `v_true` = 0.8095278816936599 · `checks` 7종 전부 `true` | — | `results/tables/probe_dgp_sanity.json` | `M0-A` | 2026-08-06 | ENTERED |
| `m0b-obp-crossval` | VERDICT `GO` · `obp_version` = 0.5.7 · 6종 estimator 전부 `match: true` | — | `results/tables/probe_obp_crossval.json` | `M0-B` | 2026-08-06 | ENTERED |
| `m0b-sbobp` | VERDICT `GO` · 6종 estimator 전부 `match: true` | — | `results/tables/probe_obp_crossval_sbobp.json` | `M0-B` | 2026-08-06 | ENTERED |
| `m0b-pypi` | obp latest 0.5.7 (2023-04-14) · sb-obp latest 0.5.10 (2025-08-19, requires_python >=3.8.1,<3.13) | — | `results/tables/probe_obp_pypi_check.json` | `M0-B` | 2026-08-06 | ENTERED |
| `m1-crossval` | VERDICT `GO` · 7종 estimator(dm·ips·snips·clipped_ips·dr·switch_dr·dros) × 2트랙(obp 0.5.7 py3.9 · sb-obp py3.12) 전부 `match=True` (rel_tol=1e-8, 점추정만 게이트) | — | `results/tables/m1_obp_crossval.csv` | `M1` | 2026-08-06 | ENTERED |
| `m2-gate` | 코어 축 01–10 figure+CSV 페어 완비 · 적대 verify 11-agent(축별 10+교차 1) 수치 재도출 일치 · 지적 9건 전부 수정 반영 | — | `results/figures/01–10_*.png` ↔ `results/tables/01–10_*.csv` | `M2` | 2026-08-06 | ENTERED |
| `m2-08-forecast` | share_large_err(=P(상대오차>0.10)) verbatim: trust=0.045911191480811735 (n=19908) · distrust=0.1414141414141414 (n=396) · ab_fallback=0.4444444444444444 (n=36) · support arm 발화 0회 | — | `results/tables/08_diagnostics_gate_confusion.csv` | `08` | 2026-08-06 | ENTERED |
| `m2-09-blindspot` | mean ESS/n(logged): 0.822986@γ=0 → 0.822315@γ=2.5 (사실상 평평) · bias(ips): −0.000147@0 → −0.056818@2.5 · oracle(pscore_true) ESS/n 0.8230→0.0182 (같은 공식이 진짜 pscore 를 받으면 감지 — CSV oracle_ps 행) | — | `results/tables/09_confounding_blindspot.csv` | `09` | 2026-08-06 | ENTERED |
| `m2-04-proxy-blind` | support proxy = 0.00000 (전 δ) vs oracle 참 미지지 π_e 질량 0.0227(δ=0.1)→0.1434163(δ=0.4) — 전역 proxy 의 전면 blind 실증 | — | `results/tables/04_deficient_support.csv` · `results/tables/04_deficient_support_oracle.csv` | `04` | 2026-08-06 | ENTERED |
| `m2-10-comparison` | comparative gate(ε=0) false-go verbatim: weighting 계열(ips·snips·clipped) max = 0.0 (상쇄) · DM(혼합 비교) fg max = 0.15 / fs max = 0.375 (bias 부활) · DR-계열 boundary fg max = 0.225 (경계 coin-flip — 부분 상쇄) | — | `results/tables/10_decision_metrics_metrics.csv` | `10` | 2026-08-06 | ENTERED |
| `m2-07-slope` | \|상대오차\| 분위(가혹 config β_log=8, CSV 재도출): clipped random p90 = 0.12541997032257812 → **slope p90 = 0.05030123684860422** (tail 회복) · snips fixed p90 = 0.02908100554505259 · switch_dr slope p50 = 0.005143399488432962 (전 항목 최소) · dros slope p50 = 0.009873919515252794 > random p50 = 0.00594413221900361 (강규제 선택의 median 대가 — 정직 병기) | — | `results/tables/07_hyperparam_ieoe.csv` | `07` | 2026-08-06 | ENTERED |
| `m3-gate` | 실데이터 축 11–12 figure+CSV 페어 · 플레이북(LEDGER-only 인용) · hero 3장 확정 · 적대 verify 4-agent 지적 7건 수정 반영 | — | `results/…/11_*, 12_*, hero_regime_map*, docs/PLAYBOOK.md, assets/decision_gate_flowchart_en.svg` | `M3` | 2026-08-06 | ENTERED |
| `m3-11-dr-robust` | DR 의 q̂-오지정 생존 4/4 재현: bias(dm, q_degraded) = −0.1605(optdigits) · −0.0261(satimage) · −0.2774(pendigits) · −0.3869(letter) vs bias(dr, q_degraded) ≤ \|0.0032\| · 게이트 80/80 trust(건강 진단과 정합) · bootstrap CI 는 구조적 bias 보유 9/28 조합에서 gt 미커버(분산만 포착 — 실데이터 교훈) | — | `results/tables/11_c2b_multidataset.csv` · `_ci.csv` | `11` | 2026-08-06 | ENTERED |
| `m3-12-gate-demo` | uniform-target: 근사 GT = 0.0038(38 clicks/10,000) · bootstrap 95% CI (0.0027, 0.0051) — 상대 반폭 ±32% · 클릭 수 random 38 · **bts 42**(`n_clicks_bts`, summary CSV) · 게이트 **DISTRUST**(ess_ratio 0.0340 < soft 0.10 · max w 277.78 > cap 100) · top1 클릭의 IPS 기여 0.3300 · clipped CI 는 GT band 하방 비겹침(clipping bias 가시화 — 정직 발견) | — | `results/tables/12_obd_small_gate.csv` · `_summary.csv` | `12` | 2026-08-06 | ENTERED |
| `m3-hero-map` | 28-cell 승자 지도: dr 9 · switch_dr 8 · dros 11 cells, DM·IPS 계열 outright 0 · tie 21/28(사전 동률 규칙) · **게이트 검정력의 소표본 실종**: β_log=16 열에서 non-trust 다수결이 n=2000→32000 에서 24→28→30/30 인데 n=500 은 trust 21/30(그 cell IPS MSE = 승자의 70.6×) · 단일 seed 지배 최대 0.7097(8000×16) | — | `results/tables/hero_regime_map.csv` · `_summary.csv` | `M3` | 2026-08-06 | ENTERED |

**M3 행 비고:** 표기 정밀도는 재도출 보고 기준 축약 — 전체 정밀도 원값은 각 source CSV 가 정본(규칙 3 의 반올림 표기 조항).

**등재 행 비고:**

- `m0a-dgp-sanity` — probe `M0-A`(`experiments/probes/probe_dgp_estimator_sanity.py`)의
  VERDICT(GO/NO-GO/INSTALL-FAIL) = `GO`. `v_true` = 0.8095278816936599 (JSON verbatim — 반올림 금지).
  `checks` 7종 전부 `true`(PASS): `ips_unbiased_within_3se` · `snips_lower_sd_than_ips` ·
  `dm_wrong_clearly_biased` · `dr_survives_wrong_q_within_3se` · `dr_oracle_sd_leq_ips_sd` ·
  `ess_uniform_equals_n` · `ess_ratio_in_unit_interval`. 지표별 세부 수치(bias·sd·se 등)가 문서에
  추가로 필요해지면 행을 분할 등재한 뒤 인용한다 (`m0a-<지표 slug>`).
- `m0b-obp-crossval` — probe `M0-B`(`experiments/probes/probe_obp_crossval.py`)의
  VERDICT(GO/NO-GO/INSTALL-FAIL) = `GO`. `obp_version` = 0.5.7 (JSON 의 값). 자기 구현 vs obp 대조:
  6종 estimator(dm·ips·snips·dr·switch_dr·dros) 전부 `match: true`.
- `m0b-sbobp` — sb-obp(폴백 트랙) 교차검증도 VERDICT = `GO` 동일: 6종 estimator 전부 `match: true`.
- `m0b-pypi` — probe `M0-B`의 PyPI 재확인 결과: obp latest 0.5.7(2023-04-14), sb-obp latest
  0.5.10(2025-08-19, requires_python >=3.8.1,<3.13). 이로써 기존 [불확실] 표기(obp 최종 릴리스 버전
  0.5.5 vs 0.5.7 상충)는 **0.5.7 로 해소**되었다.
- `m1-crossval` — M1 게이트(`experiments/m1_crossval/` 3단, M0-B probe 와 달리 **src 본구현** 검증):
  같은 배열(npz float64 공유) 위에서 자기 구현 vs obp/sb-obp 7종 estimator 점추정 전부 rel_diff ≤ 1e-8.
  switch(τ=p95(w))·clip(λ=p90(w)) **분기 발동 상태**에서의 일치이며, CI 는 라이브러리별 bootstrap 구현
  상이로 비교 제외(게이트는 점추정만). 개별 추정값·hyperparam 이 문서에 필요해지면 CSV 에서 행 분할 등재.
  주의: 이 표의 추정치는 산술 검증용 인공 설정(오지정 q̂ 포함)의 값 — 축 실험 결과가 아니다.
