# experiments — 실험 인덱스

> **상태(2026-08-10): 축 01–12·14–20 실행 완료**(코어 01–10 + 실데이터 11–12 + 스트레치 14 +
> 비즈니스 층 15–16 + **M8 GT-미상 practitioner 트랙 17–20** — probe M8-A/M8-B GO 후 착수,
> PLAN §3.5 사전등록 선행) — figure+CSV 페어가 `results/` 에 커밋돼 있다. **스트레치 13 은
> probe M5-13 NO-GO 로 drop**(2026-08-07 — 아래 축 표 13 행). 이 문서는
> 실험 ID · slug · 스윕 노브 · 산출물 규약의 *계약*이며, 결과 수치의 정본은
> [docs/LEDGER.md](../docs/LEDGER.md) 경유로만 인용한다. 그 외 실행 완료: M0 probe 2종 ·
> M6 probe(funnel DGP) · M5 probe 2종(13 NO-GO · 14 GO) · M1 교차검증(m1_crossval) · hero regime map.
>
> **실험 ID 는 불변.** ID 는 `results/figures|tables/NN_*` 와 docs 수치에 강결합되므로 재배열·재사용을
> 금지한다 (mta-simulation 관례 계승). 새 축이 필요하면 뒤 번호를 추가한다.

## 산출물 규약

- 실험 스크립트: `experiments/NN_slug.py` — Hydra 루트 config(`configs/config.yaml`) 공유, CLI
  오버라이드로 축별 스윕. 스윕 노브의 설계 기본값은 `configs/dgp/default.yaml` (결과 수치 아님).
- **figure ↔ table 1:1**: `results/figures/NN_*.{png,svg}` ↔ `results/tables/NN_*.csv` — 같은 `NN`·같은 스템.
  figure 에 실린 모든 수치는 짝 CSV 에서 재계산 가능해야 한다.
- 문서에 인용되는 수치는 committed CSV → `docs/LEDGER.md` 경유로만 (반올림·자작 금지).

## Probes — de-risk (M0 · M5 · M6 · M8)

research-design Stage 3 포맷(WHAT GENERALIZES / THE RESULT / HONEST reduces_check / VERDICT).
NO-GO 여도 세션 실패가 아니다 — 폴백 경로는 [PLAN.md](../PLAN.md) 에 기록된다.

| probe | 파일 | 무엇을 de-risk 하나 | 산출물 (JSON verdict) |
|---|---|---|---|
| M0-A | [`probes/probe_dgp_estimator_sanity.py`](probes/probe_dgp_estimator_sanity.py) | DGP 설계에서 참 정책가치 파이프라인 성립 + IPS 불편성 · SNIPS 분산 절감 · DM model-bias · DR 이중강건 · ESS 검산 → GO/NO-GO | `results/tables/probe_dgp_sanity.json` |
| M0-B | [`probes/probe_obp_crossval.py`](probes/probe_obp_crossval.py) | 자기 numpy 구현 vs obp 수치 대조(적대 교차검증 전략의 실행 가능성) + obp 릴리스 버전 사실 재확인(해소 — `results/tables/probe_obp_pypi_check.json`) → GO / NO-GO / INSTALL-FAIL | `results/tables/probe_obp_crossval.json` |
| M6 | [`probes/probe_funnel_dgp.py`](probes/probe_funnel_dgp.py) | funnel DGP 의 기저율–분리력 상충 해소 — 저기저율 CTR ∧ 정책 간 Δ 판별 가능 ∧ weight 건전(ESS)을 동시 만족하는 파라미터 존재 확인 → GO/NO-GO (축 15·16 착수 게이트) | `results/tables/probe_funnel_dgp.json` |
| M5-13 | [`probes/probe_mips_scale.py`](probes/probe_mips_scale.py) | 축 13(액션 폭발+MIPS) 착수 게이트: K 스케일업(50→2000)에서 IPS 분산 붕괴 + MIPS 구원 서사가 본 DGP family 에서 성립하는가 → GO/NO-GO | `results/tables/probe_mips_scale.json` — **NO-GO** |
| M5-14 | [`probes/probe_lambda_msm.py`](probes/probe_lambda_msm.py) | 축 14(Λ-sweep) 착수 게이트: MSM 정규화 bound 정렬-임계 정확해의 multi-action 수치 안정성(Λ=1 SNIPS 항등 · Λ 단조 · n=30k 안정 · oracle coverage) → GO/NO-GO | `results/tables/probe_lambda_msm.json` — **GO** |
| M8-A | [`probes/probe_validity_battery.py`](probes/probe_validity_battery.py) | 축 17·19·20 착수 게이트: GT-free validity battery(E[w]·harmonic calibration·placebo·disagreement — PLAN §3.5-1 사전등록 정의)의 방향성 발화 + E[w] 의 컨텍스트-국소 support 결핍 회복(축 04 blind 의 GT-미상 대응) + joint bootstrap 런타임 → GO/NO-GO (기준: PLAN §3.5-4) | `results/tables/probe_validity_battery.json` — **GO** |
| M8-B | [`probes/probe_calibrated_confounding.py`](probes/probe_calibrated_confounding.py) | 축 18 착수 게이트: U-주변화(calibrated) 기록 pscore 사후 순수함수의 수치·메모리 안정 + battery null 정합 + IPS bias 잔존 + as-recorded 대조 발화 실측 → GO/NO-GO (기준: PLAN §3.5-4 — DGP 생성기 본체 불변, 결정적 구적 GL×φ) | `results/tables/probe_calibrated_confounding.json` — **GO** |
| M9-A | [`probes/probe_c2b_injection.py`](probes/probe_c2b_injection.py) | 축 21 착수 게이트: c2b 주입 기전(구조 mask 역학·gt_value 불변·재표집) + support/noised 방향(발화 아님 — 발화 요구는 주입 튜닝) + K=26·n=10k 런타임 → GO/NO-GO (기준: PLAN §3.6-5) + 부수 실측: dataset×δ 별 masked π_e 질량 | `results/tables/probe_c2b_injection.json` — **NO-GO(② satimage 방향 한정 — §3.6-5 폴백 ② 분기로 축 21 실행·확정 반증 등재, `m9-probe-a`)** |

**상태:** M0-A VERDICT **GO** · M0-B 두 트랙(obp·sb-obp) 모두 VERDICT **GO** · M6 VERDICT **GO** ·
**M5-13 VERDICT NO-GO(축 13 drop — 정직 기록)** · **M5-14 VERDICT GO(축 14 실행 완료)** ·
**M8-A·M8-B 모두 VERDICT GO(2026-08-10 — 축 17–20 착수 가능; 부수 실측: as-recorded 로그에서도
battery 비발화 0/5 — PLAN §3.5-4 판정 기록)** —
여기는 상태 표기만, 수치 정본은 [docs/LEDGER.md](../docs/LEDGER.md).

M0-B 는 obp 의 의존성 핀 때문에 **별도 pinned Python 3.9 env** 안에서 실행한다(본 env 아님).
설치 실패는 NO-GO 가 아니라 INSTALL-FAIL 로 구분 기록하고, 소스 설치 → sb-obp → 수기 검산·property
test 대체의 폴백 사다리를 탄다.

재현 셋업 (M0-B 에서 실측 검증된 레시피):

```bash
# obp 트랙 (pinned py3.9 — obp 는 python<3.10·sklearn==1.0.2 고정)
uv venv .venv-obp --python 3.9
uv pip install --python .venv-obp/bin/python obp 'matplotlib<3.7'
#  └ matplotlib<3.7 핀 필수: obp 가 끌어오는 구식 seaborn 이 matplotlib 3.9+ 의 register_cmap 제거와 충돌 (M0-B 실측)
.venv-obp/bin/python experiments/probes/probe_obp_crossval.py
# sb-obp 트랙 (requires_python >=3.8.1,<3.13)
uv venv .venv-sbobp --python 3.12
uv pip install --python .venv-sbobp/bin/python sb-obp
.venv-sbobp/bin/python experiments/probes/probe_obp_crossval.py _sbobp
```

API 함정: obp 0.5.x 의 `SwitchDoublyRobust` 임계값 인자는 `tau` 가 아니라 `lambda_` 다 (M0-B 실측).

## m1_crossval — M1 게이트: src 구현 vs obp/sb-obp 교차검증 (실행 완료)

M0-B probe(자체 내장 최소 구현)와 달리 **`src/ope` 본구현**을 검증한다. env 혼용 금지·numpy
버전 격리 때문에 3단 분리: 같은 배열을 npz(float64·allow_pickle=False)로 공유해 순수 산술만 비교.

```bash
uv run python experiments/m1_crossval/run_mine.py                     # 1단: src 로 데이터·자기추정 → npz
.venv-obp/bin/python experiments/m1_crossval/run_obp.py               # 2단: obp(py3.9) 트랙
.venv-sbobp/bin/python experiments/m1_crossval/run_obp.py _sbobp      #      sb-obp(py3.12) 트랙
uv run python experiments/m1_crossval/make_table.py                   # 3단: 비교표 + verdict
```

**상태:** VERDICT **GO** — 7종 estimator(clipped_ips 포함 — obp 의 `IPW(lambda_)` 가 동일 산술)
× 두 트랙 전부 rel_diff ≤ 1e-8, switch(τ=p95)·clip(λ=p90) 분기 발동 상태에서. **점추정만 게이트**
(CI 는 라이브러리별 bootstrap 구현이 상이 — 비교 제외). 수치 정본: `results/tables/m1_obp_crossval.csv`
→ [docs/LEDGER.md](../docs/LEDGER.md).

## 축 01–21 표 (01–12·14–20 실행 완료 · 13 은 probe NO-GO 로 drop · 21 은 M9 사전등록·미실행 — ID 는 확정 불변)

| ID | slug 후보 | 스윕 노브 | 보려는 것 | 근거 (URL) | 무대 |
|---|---|---|---|---|---|
| 01 | `01_sample_size` | `dgp.n` 로그스케일 | 소표본 DM 우세 ↔ 대표본 IPS/DR 우세 regime 교차 | [OBP 벤치마크](https://arxiv.org/abs/2008.07146) | **백스테이지** |
| 02 | `02_logging_beta` | `dgp.beta_log` | 준결정적 로깅 → overlap 축소 → weight 폭발 | [OBP docs](https://zr-obp.readthedocs.io/en/latest/) | **백스테이지** |
| 03 | `03_policy_gap` | `dgp.beta_eval` (타깃–로깅 괴리) | 괴리 증가에서 estimator 순위 역전 | [PAS-IF, AAAI'23](https://ojs.aaai.org/index.php/AAAI/article/view/26195) | **백스테이지** |
| 04 | `04_deficient_support` | `dgp.support_deficiency` | IPS 계열의 파국적(식별 불능) 실패와 진단의 한계 | [Sachdeva–Su–Joachims KDD'20](https://arxiv.org/abs/2006.09438) | **백스테이지** |
| 05 | `05_propensity_misspec` | propensity 모드: true → estimated → noised (스크립트 노브) | DR 이 한쪽 모델로 생존하는 조건, 둘 다 틀릴 때의 실패 | [DRUnknown](https://arxiv.org/pdf/2404.01830) | **백스테이지** |
| 06 | `06_reward_misspec` | $\hat q$ 학습기 용량·오지정 정도 (스크립트 노브) | DM bias 의 표본 불감성, DR 의 보정 한계 | [MRDR](https://arxiv.org/abs/1802.03493) | **백스테이지** |
| 07 | `07_hyperparam_ieoe` | Switch τ · DRos λ · clip 임계값을 분포로 샘플 | IEOE 식 error-CDF — 튜닝 없는 고급 estimator 의 불안정 | [IEOE, RecSys'21](https://arxiv.org/abs/2108.13703) | **백스테이지** |
| 08 | `08_diagnostics_gate` | 축 01–07 의 진단 로그 종합 (ESS·max-weight vs 실오차) | 진단 예보력 산점 + decision rule 종합 — **본 레포의 제안, 표준 아님** | [Eligible Actions](https://arxiv.org/pdf/2207.00632) | **백스테이지** |
| 09 | `09_confounding_blindspot` | `dgp.confounding_strength` | 표준 진단은 동일 양호·bias 만 상이 — "진단이 못 보는 것" 대조표 (hero ③) | [Amazon Science RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) · [Namkoong+ NeurIPS'20](https://arxiv.org/abs/2003.05623) | **백스테이지** |
| 10 | `10_decision_metrics` | regime β_log × candidate β_eval **자체 factorial**(n=2000 자체 로그 — 축 CSV 재사용 아님) | 결정 안전성은 비교 설계의 속성: 절대 게이트=오차 상속 vs 같은-로그 비교=상쇄(경계·혼합비교 예외) | [SharpeRatio@k, ICLR'24](https://arxiv.org/abs/2311.18207) · [Δ-OPE, RecSys'24](https://arxiv.org/abs/2405.10024) | **백스테이지** |
| 11 | `11_c2b_multidataset` | classification-to-bandit 데이터셋 스윕 (optdigits · satimage · pendigits · letter) | 깨끗한 GT 의 멀티데이터셋 체계 벤치 | [c2b 변환 관례 예](https://arxiv.org/abs/1802.03493) | **백스테이지** |
| 12 | `12_obd_small_gate` | OBD small 단일 프로토콜(campaign=all·BTS 로그 고정 — 스윕 아님) | uniform-target 단방향(§3.4 규약: 근사 GT bootstrap CI 병기·구간 비교) + decision gate 시연 — 판별력 없음 사전 선언(클릭 희소·GT CI ±30%대); obp 교차검증·역방향은 스코프 밖 선언 | [OBD 논문](https://arxiv.org/abs/2008.07146) · [ZOZO data](https://research.zozo.com/data.html) | **본편** |
| 13 | `13_action_scale_mips` [스트레치] | `dgp.n_actions` 스케일 + action embedding (미착수) | **probe M5-13 NO-GO 로 drop (2026-08-07)** — 유계 logit softmax DGP 에선 K=2000 에도 max_w≈2.8 로 IPS 무붕괴(`results/tables/probe_mips_scale.json` — 축 03 유계-weight 교훈과 일관). "액션 폭발에서 IPS 붕괴 → MIPS 구원" 서사는 본 DGP family 에서 성립 불가 — **DGP 재설계 없이 착수 금지** (정직 기록: 수치는 LEDGER `m5-probe-13`) | [MIPS, ICML'22](https://arxiv.org/abs/2202.06317) | **―(drop)** |
| 14 | `14_lambda_sweep` [스트레치 — **probe GO 후 실행 완료**] | MSM Λ grid(1→8 기하) × γ∈{0.5,1.5} × 정책 후보 2(β_eval∈{3,5}) → breakdown Λ*(순위 단정 불능점) — probe M5-14 GO 선행(`results/tables/probe_lambda_msm.json`) | 기록 propensity 왜곡의 worst-case 구간(정규화 SNIPS형 bound)과 순위 단정이 무너지는 감도 수준 Λ* — **기존 published 방법의 도구 시연**(본 레포 제안 아님·식별 본류는 범위 밖); 자체 스키마 CSV 2본(long+breakdown — 스크립트 docstring 정본), 수치는 LEDGER `m5-14-lambda` | [Kallus & Zhou](https://arxiv.org/pdf/1805.08593) | **본편** |
| 15 | `15_funnel_reliability` | 지표 벡터(CTR·CVR·REV) × `n` 스윕 — 같은 로그·같은 weight(`FunnelConfig`, `src/ope/business.py`) | funnel 신뢰도 사다리: 깊은 지표일수록 이벤트 희소(+price heavy tail)로 판별 한계 급증 · **진단은 지표 불변**(게이트 trust ≠ 깊은 지표 판별력) · seed-ensemble band+p90 병기·이벤트 수 컬럼(0-이벤트 퇴화 정직 기록) — 리텐션 단은 의도적 부재(RL OPE 소관, PLAYBOOK §8.4) | probe M6 GO (`results/tables/probe_funnel_dgp.json`) · CVR 세션 기준([GLOSSARY §7](../docs/GLOSSARY.md)) | **백스테이지** |
| 16 | `16_business_gate` | 트레이드오프 시나리오(저가 액션 쏠림 등) × guardrail 게이트(Δ̂CTR>0 ∧ Δ̂REV≥−g ∧ HHI≤h) — 비교형 vs 절대형 | 다중 지표 guardrail: 중첩 지표의 같은-weight 공유로 결합 게이트 오류 **군집**(지표별 곱 아님 — seed 단위 기록) · 광고주 노출 재분배·HHI 는 정확 계산(OPE 아님) · subgroup 매출 OPE 는 임계 미달 시 not-estimable 정직 반환 · g·h 는 시연값(무교정) | [Δ-OPE, RecSys'24](https://arxiv.org/abs/2405.10024) 비교형 원리의 벡터 확장 | **백스테이지** |
| 17 | `17_validity_battery` [M8 — 실행 완료] | 오염 family 사전등록 목록(PLAN §3.5-3) × knob × S=40 — frontstage 는 로그 층만(`_practitioner.py` 스키마, `v_true` 컬럼 부재) | GT-free **필요조건 검사** 배터리(E[w]·harmonic·placebo·disagreement)가 대오차를 family 별로 예보/못 보는지 blind-then-reveal 채점 — 축 08 의 GT-미상 일반화 · **pooled 단독 보고 금지** · **calibrated confounding 은 관측 동등성으로 원리적 무검출**(축 18 co-exhibit) · 채점(reveal)은 별도 CSV | PLAN §3.5 (probe M8-A 게이트) · 검사 계보(해소 — POSITIONING §7.1): harmonic 은 [Li et al. WWW'15](https://arxiv.org/abs/1403.1891)(Bottou et al. 사적 교신 귀속)·E[w] 는 [Lefortier et al. 2016](https://arxiv.org/abs/1612.00367)·placebo 는 [Lipsitch et al. 2010](https://pubmed.ncbi.nlm.nih.gov/20335814/) | **본편** |
| 18 | `18_calibrated_boundary` [M8 — 실행 완료] | γ × 기록 방식 2종(as-recorded vs **U-주변화 calibrated** — 사후 순수함수, 생성기 불변) | 관측 동등성 하 battery 전항 **원리적 null**·IPS bias 만 성장 — 잡히는 것(miscalibration)/못 잡는 것(consistent confounding)의 구성적 분리, 출구는 Λ-밴드(축 14 도구) — 그림 3(축 09)의 GT-미상 세대교체 | PLAN §3.5 (probe M8-B 게이트) · [Kallus & Zhou](https://arxiv.org/pdf/1805.08593) 도구 재사용 | **본편** |
| 19 | `19_blind_decision` [M8 — 실행 완료] | `split_log` → crossfit q̂ 후보(β ladder) × 로깅 regime — 후보도 로그 유래(oracle 누수 금지) | 로그만으로 GO/NO-GO/AB → reveal 채점(regret·false-go/false-stop) — **naive(IPS 점추정·무게이트) 대비 프로토콜의 결정 가치**; 축 10 의 GT-미상 판 | PLAN §3.5 (probe M8-A 게이트 공유) · [Δ-OPE](https://arxiv.org/abs/2405.10024) 비교형 계승 | **본편** |
| 20 | `20_obd_decision_card` [M8 — 실행 완료] | OBD small 단일 프로토콜(축 12 계승 — 스윕 아님) | **reveal 없는 완전 실전** 1-page decision card(추정+CI·진단·battery — 적용불가/inconclusive 정직 표기·Λ-부채꼴 vs anchor·verdict) — 검증 주장 없음(시연 프레임)·근사 GT 대조는 축 12 LEDGER 행 참조만 | PLAN §3.5 · [OBD](https://arxiv.org/abs/2008.07146) | **본편** |
| 21 | `21_c2b_injection` [M9 — 사전등록·미실행] | 시나리오 7종(PLAN §3.6-1 고정) × c2b 4 datasets × S=20 — 오염은 전부 로깅측(`gt_value` 불변, `truth_kind="exact_c2b"`) | **replication(외적 타당성)**: 축 17 battery 예보력의 실제 공변량·정책 기하(K∈{6,10,26}) 재현 + q̂-품질 채널(무반응 예측) · **impossible family 는 실데이터 구성 불가 선언**(§3.6-3 — 축 18 co-exhibit) · matrix 28행 pooled 금지 | PLAN §3.6 (probe M9-A 게이트) · c2b 변환 관례는 축 11 과 동일 출처 | **본편** |

## 게이트 메모

- 코어 = 01–12(+ 비즈니스 층 15–16 — M6, probe 선행 GO). 스트레치 13·14 는 각각 **착수 전 1일 probe 선행** 후 GO/NO-GO —
  **판정 완료(2026-08-07): M5-13 NO-GO → 축 13 drop · M5-14 GO → 축 14 실행 완료** ([PLAN.md](../PLAN.md) §4.7).
  14 의 사전 [불확실] 항목(Λ-최적화의 multi-action 수치 안정성)은 probe M5-14 로 해소되었다.
- **M8 GT-미상 트랙(17–20)**: 사전등록(PLAN §3.5 — battery 정의·[제안] 임계값·실패 family·결합 규칙)이
  **실험 수치보다 먼저 커밋**되어 있고, probe M8-A/M8-B GO 시에만 축 착수. frontstage 산출
  `results/tables/NN_*_decision.csv` 에는 `v_true` 류 컬럼이 존재하지 않으며(계약 테스트), reveal 채점은
  `NN_*_reveal.csv` 로 분리 커밋된다(축 20 은 reveal 없음 — 시연 프레임).
- 축 09 에서 멈추는 것은 의도된 경계다(M8 이후 그 경계의 GT-미상 판은 축 18 의 calibrated-confounding
  **경계 전시** — 역시 교정 없이 Λ-밴드 이월에서 멈춘다) — confounding 하 식별의 본류(proximal 등)는
  연구 레포 소관 (README [비범위](../README.md#비범위-경계-선언) 참조).
