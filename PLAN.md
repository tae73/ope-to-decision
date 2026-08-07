# PLAN — 마일스톤 · 게이트 · 진행 추적

> **정본 규칙.** 이 문서에는 실험 결과 수치가 없다(M0 작성 시점 기준 — 현재 상태는 §4 체크리스트·LEDGER 참조). 등장하는 값은 설계 기본값(`configs/`)·
> 마일스톤 주수·출처 URL 병기 외부 인용뿐이며, 결과 수치는 실험 후 `docs/LEDGER.md` 를 단일 경유지로 한다.
> **decision gate 규칙은 본 레포의 제안(산발적 folklore 의 체계화 시도)이지 확립된 표준이 아니다**
> ([Eligible Actions](https://arxiv.org/pdf/2207.00632) 등에서 ESS 관행이 산발적으로만 확인됨).
> **경계 선언.** hidden confounding 의 proximal 본류는 연구 레포(decision-frontier) 소관 — 본 레포는 축 09
> "진단이 못 보는 것" 대조표(+조건부 축 14 Λ-sweep)에서 **의도적으로 멈춘다**. OPL/CATE/slate/RL 은 범위 밖
> (OPL·CATE 는 `kr_segmentation_causal_targeting_dunnhumby`, CATE 카탈로그는 `causal-inference` 와 상호 링크).

## 1. 마일스톤 M0–M5 + 최종 publish

| 마일스톤 | 내용 | 기간(파트타임) | 게이트(§3 상세) |
|---|---|---|---|
| **M0** (이번 세션) | 스캐폴드 + 문서 7종 + de-risk probe 2종 | 1 세션 | probe verdict 판독 → GO/NO-GO |
| **M1** | DGP 본구현 + 코어 estimator 7종(DM·IPS·SNIPS·Clipped-IPS·DR·Switch-DR·DRos) + property test + obp/sb-obp 교차검증 표 | 1.5–2.5주 | 수치 일치 — **불일치는 그대로 보고**(은폐 금지) |
| **M2** | 코어 축 01–10 실험 + figure + 진단(ESS·max-weight·support) 배선 + SLOPE 구현(축 07 hyperparameter 민감도와 함께) | 2–3주 | 축별 figure+CSV 페어 완비 — **publishable 최소선** |
| **M3** | 실데이터 이중 트랙(축 11–12) + decision-gate 플레이북 + hero 3장 확정 | 1.5–2.5주 | OBD 근사 GT 규약 준수(§3.4) |
| **M4** | portfolio-design Stage 1–5: LEDGER 확정 → comms design 브리프 → KO 정본 README → EN twin(GLOSSARY 정합) → SVG | 1–1.5주 | LEDGER 확정 전 README 수치 저작 금지 |
| **M5** (조건부 스트레치) | 축 13(액션 수+MIPS)·축 14(Λ-sweep) — **각각 1일 probe 선행** + PAS-IF 확장 | +1.5–2주 | probe GO 시에만 착수(§3.3) |
| **M6** | 비즈니스 임팩트 층: funnel probe 선행 → `business.py` → 축 15(funnel 신뢰도 사다리)·16(다중 지표 guardrail+광고주 재분배) → PLAYBOOK·README KO/EN 통합 | 1–1.5주 | probe GO 게이트 + verify 지적 0 잔존 |
| **최종** | portfolio-design Stage 6 적대검증(모든 수치 = LEDGER 삼각일치) → Stage 7 publish(GitHub) → 선택: lowellth-publish | 0.5–1주 | 검증 실패 항목 0 |

**달력 정직성.** 연구(proximal OTR/DTR) 병행 파트타임이므로 달력 시간으로 **총 2–3개월**로 잡는 것이 정직하다.
M2 종료 시점에 이미 레포가 성립하도록(부분 완성으로도 공개 가능) 설계했고, M5 는 drop 해도 코어 서사가 완결된다.

## 2. 축 ↔ 실험 ID 매핑 (ID 불변 — `experiments/NN_slug.py`)

DGP 노브는 `src/ope/dgp.py` 의 `DGPConfig` 필드와 1:1 대응한다(기본값: `configs/dgp/default.yaml`).
"—" 는 DGP 필드가 아니라 estimator/metric 층 노브라는 뜻이다.

| ID | 축 | 스윕 노브 | `DGPConfig` 필드 | 보이려는 패턴 (출처) |
|---|---|---|---|---|
| 01 | 표본 크기 | n 로그스케일 스윕 | `n` | 소표본=IPS 분산 지배(DM 우세) ↔ 대표본=DM bias 지배 — regime 교차 ([OBP 벤치마크](https://arxiv.org/abs/2008.07146)) |
| 02 | 로깅 stochasticity | softmax inverse-temperature 스윕 | `beta_log` | 준결정적 로깅 → overlap 축소 → weight 폭발, IPS/DR 붕괴 ([OBP docs](https://zr-obp.readthedocs.io/en/latest/)) |
| 03 | 타깃–로깅 괴리 | 평가정책 온도 스윕(로깅 고정) | `beta_eval` | 괴리↑ → 분산 폭발·estimator 순위 역전 ([PAS-IF, AAAI'23](https://ojs.aaai.org/index.php/AAAI/article/view/26195)) |
| 04 | deficient support | π₀(a\|x)=0 강제 비율 | `support_deficiency` | IPS 계열의 파국적(식별 불능) 실패와 진단의 한계 ([Sachdeva-Su-Joachims, KDD'20](https://arxiv.org/abs/2006.09438)). *M1 설계: per-row 랜덤이 아닌 **구조적 mask**(컨텍스트별 하위-q ⌊δK⌋개 제거 — 랜덤 mask 는 support proxy 를 원리적으로 blind 으로 만듦); δ 스윕은 1/K 양자화 — 세밀 스윕은 K↑ 로* |
| 05 | propensity 오지정 | p̂ 를 true→estimated→noised 로 교체 | — (estimator 입력측; `pscore_logged`/`pscore_true` 배열 활용) | IPS bias 직결; DR 은 q̂ 한쪽만 맞아도 생존, 둘 다 틀리면 실패 ([DRUnknown](https://arxiv.org/pdf/2404.01830)) |
| 06 | reward model 오지정 | q̂ 학습기 용량·오지정 정도 | — (estimator 측; 보조 `reward_noise`) | DM bias 의 표본 불감성, DR 보정의 한계 ([MRDR](https://arxiv.org/pdf/1802.03493)) |
| 07 | hyperparameter 민감도 | clip λ·switch τ·shrinkage 를 분포로 샘플 | — (estimator 하이퍼 층) | 튜닝 없는 고급 estimator 가 단순 estimator 보다 불안정 — IEOE error-CDF ([IEOE](https://arxiv.org/abs/2108.13703)) |
| 08 | 진단 예보력 + 결정규칙 | 축 01–07 grid 산출 재사용 | (전 축 필드 재사용) | ESS·max-weight vs 실오차 산점 — 진단이 예보하는 축과 못 보는 축의 대비; 결정규칙은 **본 레포의 제안**으로 프레임 |
| 09 | confounding 주입 + 대조표 | U 개입 강도(기록 pscore ≠ 진짜) | `confounding_strength` | unconfounded vs confounded 에서 ESS·max-weight 는 동일 양호 범위, bias 만 상이 — 표준 진단이 원리적으로 blind ([RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) · [Namkoong+](https://arxiv.org/abs/2003.05623)). *M1 설계: U 는 로깅 logits(γ·U·d_a)와 reward(+γ·κ·U, κ=0.5)에 동시 개입; `pscore_logged`=의도 정책 기록값·`pscore_true`=**U-조건부** 실제값(oracle-IPS 가 참값 복원하는 정의 — tests/test_statistical.py 의 대조 항등으로 검증)* |
| 10 | 의사결정 metric | MSE 대신 잘못 배포 확률·rank-corr | (전 축 grid 재사용; metric 층) | MSE 동률 estimator 가 정책 *선택* 안전성에선 갈림 ([SharpeRatio@k, ICLR'24](https://arxiv.org/abs/2311.18207)) |
| 11 | 실데이터 c2b 멀티데이터셋 | optdigits·satimage·pendigits·letter | — (`datasets.py`; 표준 변환 프로토콜) | 깨끗한 GT 에서 synthetic 결론의 체계 벤치 재현 ([c2b 변환 사용 예 — MRDR(Farajtabar+ 2018)](https://arxiv.org/pdf/1802.03493) — 프로토콜 계보는 POSITIONING §2.1 참조) |
| 12 | 실데이터 OBD small 게이트 | ZOZO 2 로깅정책·실측 propensity | — (`datasets.py`) | random-policy 근사 GT 대비 재현 — bootstrap CI 병기 필수(§3.4) ([OBD](https://arxiv.org/abs/2008.07146)) |
| 13 | [스트레치] 액션 수 + MIPS | K 스윕(10→수천) | `n_actions` | IPS/DR 분산 폭발과 MIPS 의 구원 ([Saito-Joachims, ICML'22](https://arxiv.org/abs/2202.06317)) |
| 14 | [스트레치] Λ-sweep + breakdown Λ* | MSM Λ 스윕(축 09 DGP 재사용) | `confounding_strength` (+ estimator 층 Λ) | 정책 순위가 뒤집히는 breakdown Λ* 리포트 ([Kallus & Zhou](https://arxiv.org/pdf/1805.08593)) |
| 15 | funnel 신뢰도 사다리 (비즈니스 층) | 지표 벡터(CTR·CVR·REV) × n 스윕 — 같은 로그·같은 weight | — (`src/ope/business.py` `FunnelConfig` — 코어 DGP 아님) | 깊은 지표일수록 이벤트 희소(+price heavy tail)로 판별 한계 급증 · 진단은 지표 불변(게이트 trust ≠ 깊은 지표 판별력) · 리텐션 단 의도적 부재(RL OPE 소관) — probe M6 GO 선행(`results/tables/probe_funnel_dgp.json`) |
| 16 | 다중 지표 비즈니스 게이트 (비즈니스 층) | 트레이드오프 시나리오 × guardrail(Δ̂CTR>0 ∧ Δ̂REV≥−g ∧ HHI≤h) 비교형 vs 절대형 | — (`FunnelConfig`; 광고주 매핑은 구조 rng) | 중첩 지표의 같은-weight 공유 → 결합 게이트 오류 군집(독립 곱 아님) · 노출 재분배·HHI 는 정확 계산(OPE 아님) · subgroup 매출 OPE 희소 시 not-estimable 정직 반환 — 비교형 원칙([Δ-OPE, RecSys'24](https://arxiv.org/abs/2405.10024))의 벡터 확장 |

비고: `dim_context`·`seed`·`struct_seed` 는 스윕 축이 아니라 통제 변수다 — `struct_seed`(M1 추가 필드)는
환경 구조(θ·b·d_a)를 고정하고 `seed` 만 바꿔 "같은 환경, 다른 로그" MC 반복을 만든다(구조 rng draw 순서는
dgp.py 에 고정·문서화 — 순서 변경 = 환경 변경). 축 13·14 는 독립 모듈로 격리해 drop 가능하게 만든다.
**M1 reward 설계 결정**: reward 는 연속형 `r = q + γ·κ·U + σ·ε` — E[U]=E[ε]=0 이므로 v_true=E_x[Σπ_e q] 가
γ·σ 와 무관하게 정확히 성립(테스트로 고정). binary CTR 현실성은 축 11–12 실데이터 트랙의 몫이며,
probe M0-A(Bernoulli)와의 이탈은 dgp.py docstring 에 기록했다.
**M6 설계 결정**: 축 15·16 은 코어 `DGPConfig` 가 아니라 `src/ope/business.py` 의
`FunnelConfig`(funnel DGP: click~Bern(p_c)·conv|click~Bern(p_v)·revenue=click·conv·price_a — 정책은
relevance 스코어 위 softmax, 기저율과 분리력을 구조적으로 분리) 위에서 돈다. CVR 은 **세션 기준**
(click·conv — 업계 click-조건부와 다름, GLOSSARY §7), γ(confounding) 노브는 Bernoulli 비선형에서
정확-GT 정리가 깨져 **영구 금지**(confounding 은 축 09 소관 경계), 리텐션·세션 간 지표는 single-step
bandit OPE 식별 불가로 범위 밖(RL OPE 소관 — PLAYBOOK §8.4 경계 선언, 세션 내 proxy 미채택).

## 3. GO/NO-GO 게이트와 폴백

### 3.1 M0 게이트 — probe 2종
- **probe M0-A** (`experiments/probes/probe_dgp_estimator_sanity.py`): DGP+최소 estimator 의 교과서 성질
  (IPS 불편성·SNIPS 분산 절감·DR 이중강건·ESS 검산) → JSON verdict. NO-GO 면 DGP 설계 자체를 재검토.
- **probe M0-B** (`experiments/probes/probe_obp_crossval.py`): pinned Python 3.9 env(`.venv-obp`)에서 자기 구현
  vs obp 수치 대조 → GO / NO-GO / INSTALL-FAIL. **동시에 obp 최종 릴리스 버전을 PyPI 에서 재확인해 기록**한다 —
  탐색 단계 기록에 0.5.5 vs 0.5.7 불일치가 있어 **[불확실]** 태그 유지, 이 probe 로 해소 예정(단정 금지).
- probe NO-GO 여도 세션 실패가 아니다: 아래 폴백 경로를 기록하고 진행한다.

### 3.2 obp 설치 실패 폴백 체인 (M0-B → M1 교차검증)
① pinned py3.9 env pip 설치(obp 는 `python>=3.7.1,<3.10`·`scikit-learn==1.0.2` 고정 —
[pyproject](https://raw.githubusercontent.com/st-tech/zr-obp/master/pyproject.toml) · [PyPI](https://pypi.org/project/obp/)) →
② 소스 설치 + 의존성 핀 완화 → ③ [sb-obp fork](https://github.com/sb-ai-lab/sb-obp) 최신 env(대체재 적합성 **[불확실]** — 채택 시 근거 기록) →
④ 전부 실패 시 교차검증 범위 축소: **논문 수치 재현 + property test**(불편성·분산 순서·이중강건 스모크)로 대체하고
어느 단계에서 멈췄는지 README 에 그대로 보고한다(정직성 규약).
실측 함정: `.venv-obp`(py3.9) 에는 `matplotlib<3.7` 핀 필수 — obp 가 끌어오는 구식 seaborn 이 matplotlib 3.9+ 의 `register_cmap` 제거와 충돌(M0-B 실측, 2026-08-06).

### 3.3 Λ probe 게이트 (M5 진입 조건)
축 14 착수 전 **1일 probe**: multi-action per-sample weight 최적화(정렬 해/소형 LP)의 수치 안정성은 문헌 조사만으로
미확인이다. probe NO-GO 시 축 14 를 drop 하고 **축 09 는 대조표만으로 완결**한다 — 대조표가 코어 축이므로
hero ③("진단이 못 보는 것")과 decision-gate 서사는 훼손되지 않는다. 축 13 도 동일하게 1일 probe 선행.

### 3.4 OBD small 근사 ground truth 규약 (M3)
random-policy 로그 기반 on-policy 근사 GT 는 자체 표본 오차를 가진다(근사 정밀도 수치는 **[불확실]** — 축 12 에서
실측 후 LEDGER 기록). 따라서 축 12 의 **모든 figure 에 bootstrap CI 병기, 점 비교 단정 금지**.

## 4. M0 체크리스트 (진행 표시)

- [x] 스캐폴드 트리: `src/ope/` 스텁 5종 · `configs/`(Hydra) · `tests/` · `results/{figures,tables}` · `data/`(.gitignore 보호)
- [x] `pyproject.toml`(uv) · `uv.lock` · `LICENSE`(MIT) · `.gitignore`
- [x] probe M0-A 실행 → `results/tables/probe_dgp_sanity.json`, VERDICT **GO** (상태 표기 — 수치 등재는 LEDGER, 초기 커밋 동반)
- [x] probe M0-B 실행 → obp(py3.9)·sb-obp(py3.12) 두 트랙 모두 VERDICT **GO** (상태 표기 — 수치 등재는 LEDGER, 초기 커밋 동반) (`probe_obp_crossval.json`·`probe_obp_crossval_sbobp.json`) + PyPI 사실 확인(`probe_obp_pypi_check.json`)
- [x] 문서 7종 작성·검증·수정: `README.md`(수치 0 스켈레톤) · `CLAUDE.md` · `PLAN.md`(본 문서) · `docs/CONCEPT.md` · `docs/POSITIONING.md`(전 주장 출처 URL) · `docs/LEDGER.md`(빈 틀+규칙) · `docs/GLOSSARY.md`
- [x] `experiments/README.md`(축↔ID 매핑 — 본 문서 §2 와 동일 정본)
- [x] `uv sync` + `pytest` 통과 확인 (M0: 스텁 smoke → M1: property test 로 대체·확장)
- [x] `git init` + 초기 커밋 1회(데이터 미포함 · force-push 금지) — 본 커밋(2026-08-06)

## 4.1 M1 체크리스트 (완료 — 2026-08-06)

- [x] `src/ope/` 본구현 4모듈: policies(안정 softmax·ε-greedy) · dgp(연속형 reward·U-조건부
  `pscore_true`·구조적 support mask·`struct_seed` — §2 설계 결정 참조) · estimators(코어 7종 +
  `bootstrap_ci`) · diagnostics(ESS·max-weight·support proxy + `GateThresholds`·`decision_gate`).
  datasets.py 는 M3 스텁 유지.
- [x] property test **39개 green** (`uv run pytest`): 정확 항등식(switch τ=∞≡DR·clipped λ=∞≡IPS·
  DRos 극한·DR q̂=0≡IPS·SNIPS 스케일 불변) + 통계(IPS 불편·분산 순서·DR 이중강건) + **종단 on-policy
  검산** + **confounding 대조 항등**(oracle 불편 ∧ 기록-pscore 편향) + DGP·진단·게이트 속성.
- [x] obp/sb-obp 교차검증 표 → VERDICT **GO** (`results/tables/m1_obp_crossval.csv`, LEDGER
  `m1-crossval` 행): 7종 × 2트랙 rel_diff ≤ 1e-8, 분기 발동 상태(τ=p95·λ=p90). 게이트 통과.
- [x] 적대 코드리뷰(3렌즈 → 발견별 반증 검증): 발견 19건 중 **확정 2건 수정 완료**
  (① `bootstrap_ci` alpha 무검증 fail-silent → (0,1) 강제 + 의미론 docstring, ② gate 임계값 raw dict
  오타 키 침묵 무시 → `GateThresholds` NamedTuple 강제), 17건은 반증 기각(재현 불가·도달 불가·문서화된
  설계). 수정 후 39 테스트 재green.
- [x] LEDGER `m1-crossval` ENTERED · PLAN §2 설계 결정 반영 · M1 커밋

## 4.2 M2 체크리스트 (완료 — 2026-08-06)

- [x] CI 병기 규약 정밀화(CLAUDE.md §2 — 합성 MC 축=seed-ensemble band, bootstrap 은 실데이터 전용)
- [x] SLOPE 구현(`slope_select`+`_lepski_select`) — **축 07 실험이 ladder 방향 반전 버그를 실증 적발**
  (논문은 광폭 CI 부터 — 반대로 걸으면 고편향 rung 이 veto → clipped 30/30 최소 rung 붕괴) → 수정 +
  회귀 테스트 고정, 수정 후 clipped p90 0.125→0.050 회복. 테스트 48 green.
- [x] 실험 인프라: `experiments/_common.py`(RunRecord·COLUMNS·hyperparam 정책·v_true memoize) +
  `_style.py`(entity 고정 색 — dataviz validator ALL PASS) + 축 01 exemplar 계약.
- [x] **코어 축 01–10 전부 실행 완료** — figure+CSV 페어 10축(+companion CSV: 04 oracle·08 confusion·
  10 metrics/thresholds). fan-out 7 agents + 집계 2축 인라인.
- [x] 정직 보고(기대 불발 포함): 02 β=8 cliff 불발(8→16 사이)·03 DM 역전 불발(weight 유계)·
  04 support proxy 전면 blind(0 vs oracle 0.143)·07 IEOE 불안정은 clipped 전용·10 비교형 게이트
  상쇄(초기 null 의 원인 규명 포함) — 전부 figure·docstring 에 그대로.
- [x] 적대 verify 11-agent(축별 10+교차 1): 전 축 수치 재도출 일치, 지적 9건(문구 한정·stale·색·
  ylim 잘림·companion CSV) 전부 수정 반영.
- [x] LEDGER: `m2-gate`·`m2-08-forecast`·`m2-09-blindspot`·`m2-04-proxy-blind` ENTERED · M2 커밋

## 4.3 M3 체크리스트 (완료 — 2026-08-06)

- [x] `src/ope/datasets.py` 구현: c2b(softmax 정책 쌍 — ε-greedy 는 overlap 무스트레스로 기각·정확
  propensity·정확 참값·비-affine 오지정 q̂) + OBD 얇은 로더(파싱 함정 처리). 테스트 52 green
  (테스트 자체 버그 1건 교정 — 로깅행동 log-likelihood 는 확신도 지배 척도라 Brier 로 교체).
- [x] 축 11(c2b 4 데이터셋): **DR 의 q̂-오지정 생존 4/4 재현** + 게이트 80/80 trust + bootstrap CI 의
  bias 미포착 실증(9/28 미커버) + good-q̂ DM 압승 예상 부분 불발(pendigits·letter — 정직 보고).
- [x] 축 12(OBD small, uniform-target 단방향): §3.4 규약 완수(GT bootstrap CI ±32% 병기·구간 비교) +
  게이트 실로그 **DISTRUST** 판정 + 판별력 없음 사전 선언 실증 + clipped CI 하방 비겹침(정직 발견).
  데이터는 로컬 전용(data/ gitignored·재배포 금지·라이선스 각주).
- [x] **hero 3장 확정**: ① `assets/decision_gate_flowchart_en.svg`(플로차트 — KO twin 은 M4)
  ② `results/figures/hero_regime_map.png`(28-cell 승자 지도 — 최대 발견: 게이트 검정력의 소표본 실종)
  ③ `results/figures/09_confounding_blindspot.png`(진단이 못 보는 것 대조).
- [x] `docs/PLAYBOOK.md`: LEDGER 행만 수치 인용(m2-08·09·04·10·07-slope)·"제안 — 표준 아님" 프레임·
  비교형 게이트 우선 원칙·confounding 면책·support proxy 신뢰 금지.
- [x] 적대 verify 4-agent: hero PASS·지적 7건(수치 범위 과대·bias 기전 명명·README 축 12 불일치·
  provisional 마커·**calibration 과대표현 삼각 불일치**·SVG tint 주석·LEDGER 헤더) 전부 수정 반영.
- [x] LEDGER m3 4행 + `m2-07-slope` ENTERED · M3 커밋

## 4.4 M4 체크리스트 (완료 — 2026-08-07)

- [x] **Stage 1** LEDGER 확정: README 인용 대상 headline 수치 전 행 ENTERED 점검(부족분 0) —
  verify 에서 1건 적발(bts 클릭 42 미등재) → `m3-12-gate-demo` 행 확장으로 해소.
- [x] **Stage 2** comms design 브리프: `docs/COMMS_BRIEF.md` — 아크·청중 레이어·hero 캡션 설계·
  배지↔LEDGER 매핑·"부러지지 않은 것들" 섹션 설계. 사용자 확정: 수치 배지·고밀도 표준형·EN 은 README 만.
- [x] **Stage 3** KO 정본 README 전면 저작(~250줄): 수치 배지 6종 → TL;DR 3줄 → hero 3장 임베드 →
  2막 서사 → 구성요소·검증 3중 표 → 축별 발견 12줄 표(LEDGER id 병기) → **불발 6건을 본문 섹션으로
  승격**("부러지지 않은 것들") → Quick Start → 문서 지도 → Attribution → 정직성 각주.
- [x] **Stage 4** EN twin `README.en.md`: 자연 재작성(직역 금지)·GLOSSARY EN 열 정합·수치 0 드리프트·
  언어 토글 상호 링크.
- [x] **Stage 5** `assets/decision_gate_flowchart_ko.svg`: EN 판과 구조·좌표·팔레트 동일, 텍스트만
  KO(GLOSSARY 표기), rsvg 렌더 검수 통과. KO README=ko.svg / EN=en.svg 임베드.
- [x] 적대 verify 2렌즈(수치=LEDGER·링크·배지 / parity·렌더): 지적 10건 전부 반영 — LEDGER 미경유
  수치(42) 등재, 표기 정밀화(|DR bias|), HTML 블록 내 백틱→`<code>`, GLOSSARY 첫등장 병기, EN SVG
  용어 통일("blind in principle"), img alt, 메타데이터 각주.
- [x] **publish 시점 결정 사항 기록**: 두 README 의 인접 레포 상대 링크 4종(`../dag-registry/` 등)은
  GitHub 단독 레포에서 404 — Stage 7 publish 때 GitHub 절대 URL 로 치환하거나 코드체 레포명으로 강등
  (하우스 관례 확인 후 일괄 적용).
- [x] PLAN §4.4 · applied 인덱스 갱신 · M4 커밋

## 4.5 M6 체크리스트 (완료 — 2026-08-07)

- [x] probe M6(funnel DGP) **GO** — 기저율-분리력 상충을 relevance 스코어 분리 설계로 해소
  (LEDGER `m6-probe-funnel`).
- [x] `src/ope/business.py`: funnel DGP(지표 3종 정확 GT — CVR 은 **세션 기준**)·노출/HHI **정확 계산**
  (OPE 아님)·subgroup 매출 IPS(not-estimable 정직 반환)·**γ 노브 영구 금지**(테스트 고정). tests 59 green.
- [x] 축 15: funnel 신뢰도 사다리 성립 + **진단은 지표 불변**(게이트 trust ≠ 깊은 지표 판별력) —
  리텐션 단은 의도적 부재(RL OPE 소관·세션내 proxy 미채택 — 사용자 확정) (LEDGER `m6-15-ladder`).
- [x] 축 16: 비교형 상쇄의 조건부성(arm 별)·지표 간 오차 군집(Δ̂ 수준 성립·게이트 수준 불발 정직 기록)·
  HHI 결정적 arm·시연값 무교정 프레임 (LEDGER `m6-16-gate`).
- [x] 문서 통합: PLAYBOOK §8 비즈니스 번역·README KO/EN 섹션+축 표+배지·GLOSSARY §7·experiments/
  README·CLAUDE.md 동기 — verify 2렌즈 지적 8건 전부 수정(tests 59·LEDGER 스키마 01–16·bootstrap CI
  정의 stale 등).
- [x] LEDGER m6 4행 ENTERED · M6 커밋

## 4.6 최종 마일스톤 체크리스트 (완료 — 2026-08-07)

- [x] **Stage 6 전면 적대검증**(4렌즈): 수치 삼각일치(LEDGER 표본 6행 source CSV 재도출 일치 —
  반올림 왜곡 0) · 링크/경로 무결 · KO/EN parity·렌더 · 정직성/publish 안전(비밀·데이터 누출 0).
  지적 전부 수정: PLAYBOOK 현행화(실데이터·m6 행 인용 — pre-M3 stale 해소), README 축 15 캡션 갱신,
  **로컬 경로 누출 2곳 제거**, stale M0 문구 3곳 frozen 주석.
- [x] 인접 레포 링크 치환: 공개 2종 → GitHub 절대 URL, 비공개 2종(`causal-inference`·`dag-registry`)
  → 코드체 강등. 인접 상대 링크 잔존 0.
- [x] GitHub publish: `tae73/ope-to-decision` public, main push, 렌더 확인. tests 59 green.

## 5. 리듬 규약

- **한 stage = 한 `/goal`.** M1 부터는 마일스톤(필요 시 마일스톤 내 소단계)을 `/goal` 로 잠그고 게이트 판정 후 해제.
- **ultracode 는 fan-out 단계에서만**: M0 문서 저작 병렬, M2 축 실험 스크립트 병렬, M4 KO/EN twin 병렬.
  꺼져 있으면 자동 구동하지 않고 제안만 한다(비용 규율).
- 재사용 자산 참조(신규 작성 금지): dag-registry `docs/dag-design.md` §6.5(진단 스펙) ·
  mta-simulation `experiments/`(축별 패턴) · dunnhumby `src/policy.py`(순수함수 관례).

## 6. 실험 규율

- **ID 불변**: 실험 ID 01–14 는 재부여·재정렬 금지. 축 추가는 새 번호로만.
- **한 축 = 한 스크립트 = 한 figure**: `experiments/NN_slug.py` 가 해당 축의 유일한 진입점.
- **figure ↔ 데이터 1:1 페어링**: `results/figures/NN_*.{png,svg}` 마다 `results/tables/NN_*.csv` 동반 커밋 — 페어 없는 figure 금지.
- **모든 문서 수치는 LEDGER 경유**: committed 결과만 `docs/LEDGER.md` 에 등재하고, README·리포트는 LEDGER 만 인용
  (반올림·자작·과대표현 금지). LEDGER 미등재 수치가 문서에 나타나면 검증 단계에서 반려.
- **재현성**: 고정 seed · 설정은 `configs/` 경유 · 실행 환경(uv lock) 고정. probe 는 research-design Stage 3 포맷
  (WHAT GENERALIZES / RESULT / VERDICT + JSON)을 유지한다.
