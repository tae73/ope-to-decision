# experiments — 실험 인덱스

> **상태: 축 01–14 는 전부 실행 전(예정)이다.** 이 문서는 실험 ID · slug · 스윕 노브 · 산출물 규약의
> *계약*이며, 결과 수치는 어디에도 없다. 실행이 완료된 것은 아래 M0 probe 2종뿐이다.
>
> **실험 ID 는 불변.** ID 는 `results/figures|tables/NN_*` 와 docs 수치에 강결합되므로 재배열·재사용을
> 금지한다 (mta-simulation 관례 계승). 새 축이 필요하면 뒤 번호를 추가한다.

## 산출물 규약

- 실험 스크립트: `experiments/NN_slug.py` — Hydra 루트 config(`configs/config.yaml`) 공유, CLI
  오버라이드로 축별 스윕. 스윕 노브의 설계 기본값은 `configs/dgp/default.yaml` (결과 수치 아님).
- **figure ↔ table 1:1**: `results/figures/NN_*.{png,svg}` ↔ `results/tables/NN_*.csv` — 같은 `NN`·같은 스템.
  figure 에 실린 모든 수치는 짝 CSV 에서 재계산 가능해야 한다.
- 문서에 인용되는 수치는 committed CSV → `docs/LEDGER.md` 경유로만 (반올림·자작 금지).

## Probes — M0 (유일하게 실행 완료된 것)

research-design Stage 3 포맷(WHAT GENERALIZES / THE RESULT / HONEST reduces_check / VERDICT).
NO-GO 여도 세션 실패가 아니다 — 폴백 경로는 [PLAN.md](../PLAN.md) 에 기록된다.

| probe | 파일 | 무엇을 de-risk 하나 | 산출물 (JSON verdict) |
|---|---|---|---|
| M0-A | [`probes/probe_dgp_estimator_sanity.py`](probes/probe_dgp_estimator_sanity.py) | DGP 설계에서 참 정책가치 파이프라인 성립 + IPS 불편성 · SNIPS 분산 절감 · DM model-bias · DR 이중강건 · ESS 검산 → GO/NO-GO | `results/tables/probe_dgp_sanity.json` |
| M0-B | [`probes/probe_obp_crossval.py`](probes/probe_obp_crossval.py) | 자기 numpy 구현 vs obp 수치 대조(적대 교차검증 전략의 실행 가능성) + obp 릴리스 버전 사실 재확인(해소 — `results/tables/probe_obp_pypi_check.json`) → GO / NO-GO / INSTALL-FAIL | `results/tables/probe_obp_crossval.json` |

**상태:** M0-A VERDICT **GO** · M0-B 두 트랙(obp·sb-obp) 모두 VERDICT **GO** — 여기는 상태 표기만,
수치 정본은 [docs/LEDGER.md](../docs/LEDGER.md).

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

## 축 01–14 예정표 (전부 미실행 — slug 는 후보, ID 는 확정)

| ID | slug 후보 | 스윕 노브 | 보려는 것 | 근거 (URL) |
|---|---|---|---|---|
| 01 | `01_sample_size` | `dgp.n` 로그스케일 | 소표본 DM 우세 ↔ 대표본 IPS/DR 우세 regime 교차 | [OBP 벤치마크](https://arxiv.org/abs/2008.07146) |
| 02 | `02_logging_beta` | `dgp.beta_log` | 준결정적 로깅 → overlap 축소 → weight 폭발 | [OBP docs](https://zr-obp.readthedocs.io/en/latest/) |
| 03 | `03_policy_gap` | `dgp.beta_eval` (타깃–로깅 괴리) | 괴리 증가에서 estimator 순위 역전 | [PAS-IF, AAAI'23](https://ojs.aaai.org/index.php/AAAI/article/view/26195) |
| 04 | `04_deficient_support` | `dgp.support_deficiency` | IPS 계열의 파국적(식별 불능) 실패와 진단의 한계 | [Sachdeva–Su–Joachims KDD'20](https://arxiv.org/abs/2006.09438) |
| 05 | `05_propensity_misspec` | propensity 모드: true → estimated → noised (스크립트 노브) | DR 이 한쪽 모델로 생존하는 조건, 둘 다 틀릴 때의 실패 | [DRUnknown](https://arxiv.org/pdf/2404.01830) |
| 06 | `06_reward_misspec` | $\hat q$ 학습기 용량·오지정 정도 (스크립트 노브) | DM bias 의 표본 불감성, DR 의 보정 한계 | [MRDR](https://arxiv.org/abs/1802.03493) |
| 07 | `07_hyperparam_ieoe` | Switch τ · DRos λ · clip 임계값을 분포로 샘플 | IEOE 식 error-CDF — 튜닝 없는 고급 estimator 의 불안정 | [IEOE, RecSys'21](https://arxiv.org/abs/2108.13703) |
| 08 | `08_diagnostics_gate` | 축 01–07 의 진단 로그 종합 (ESS·max-weight vs 실오차) | 진단 예보력 산점 + decision rule 종합 — **본 레포의 제안, 표준 아님** | [Eligible Actions](https://arxiv.org/pdf/2207.00632) |
| 09 | `09_confounding_blindspot` | `dgp.confounding_strength` | 표준 진단은 동일 양호·bias 만 상이 — "진단이 못 보는 것" 대조표 (hero ③) | [Amazon Science RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) · [Namkoong+ NeurIPS'20](https://arxiv.org/abs/2003.05623) |
| 10 | `10_decision_metrics` | metric 층: 잘못-배포 확률 · rank-corr (축 시나리오 재활용) | MSE 동률 estimator 가 정책 *선택* 안전성에선 갈림 | [SharpeRatio@k, ICLR'24](https://arxiv.org/abs/2311.18207) |
| 11 | `11_c2b_multidataset` | classification-to-bandit 데이터셋 스윕 (optdigits · satimage · pendigits · letter) | 깨끗한 GT 의 멀티데이터셋 체계 벤치 | [c2b 변환 관례 예](https://arxiv.org/abs/1802.03493) |
| 12 | `12_obd_small_gate` | OBD small 캠페인 × 로깅정책 쌍 | random-policy 근사 GT 로 synthetic 결론 재현 + obp 교차검증 (근사 GT 는 불확실 — bootstrap CI 병기) | [OBD 논문](https://arxiv.org/abs/2008.07146) · [ZOZO data](https://research.zozo.com/data.html) |
| 13 | `13_action_scale_mips` [스트레치] | `dgp.n_actions` 스케일 + action embedding | 액션 폭발에서 IPS/DR 분산 붕괴와 MIPS 의 구원 | [MIPS, ICML'22](https://arxiv.org/abs/2202.06317) |
| 14 | `14_lambda_sweep` [스트레치·조건부] | MSM Λ grid → breakdown Λ* | 기록 propensity 가 틀렸을 때 worst-case 구간과 정책 순위 반전점 | [Kallus & Zhou](https://arxiv.org/pdf/1805.08593) |

## 게이트 메모

- 코어 = 01–12. 스트레치 13·14 는 각각 **착수 전 1일 probe 선행** 후 GO/NO-GO — 특히 14 는
  Λ-최적화의 multi-action 수치 안정성이 미검증(불확실)이라 probe GO 시에만 진입 ([PLAN.md](../PLAN.md)).
- 축 09 에서 멈추는 것은 의도된 경계다 — confounding 하 식별의 본류(proximal 등)는 연구 레포 소관
  (README [비범위](../README.md#비범위-경계-선언) 참조).
