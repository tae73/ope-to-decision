# CONCEPT — `ope-to-decision` (research-design Stage 0 · 1-pager)

> **one-liner.** A/B 테스트를 돌리기 전에, 이미 쌓인 추천 로그만으로 새 정책의 가치를 추정하고 —
> 더 중요하게는 **그 추정을 언제 믿으면 안 되는지**까지 판정하는 multi-action OPE 벤치마크 + 배포 게이트 플레이북.

**문서 지위.** Stage 0 컨셉 정본. 본 문서에 등장하는 수치는 설계 기본값(`configs/`)과 마일스톤 주수, 출처 있는
외부 인용뿐이다 — **실험 결과 수치는 0개**(아직 실험이 없다). 모든 결과 수치는 이후 `docs/LEDGER.md` 경유로만 등장한다.

## 1. 동기 — 비즈니스 문제

이커머스 추천 위젯 팀의 상황을 그대로 옮긴다. ML 엔지니어가 새 추천 정책 후보 2–3개를 만들었는데,

- **A/B 슬롯이 희소하다.** 온라인 실험 슬롯은 분기당 한정이라 모든 후보를 트래픽으로 검증할 수 없다. Spotify는
  플레이리스트 추천에서 정확히 이 동기(A/B 비용·리스크 절감)로 offline evaluation을 의사결정 도구로 썼다
  ([Spotify WSDM'19](https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms)).
- **나쁜 정책을 태우는 리스크가 실재한다.** 매출·UX 훼손은 실험이 끝나도 되돌릴 수 없다. Netflix는 budget-constrained
  추천의 off-policy 학습·평가를 같은 맥락에서 다루고([Netflix TechBlog](https://netflixtechblog.com/reinforcement-learning-for-budget-constrained-recommendations-6cbc5263a32a)),
  Airbnb는 counterfactual evaluation으로 search ranking 실험을 가속했다([Airbnb 2025](https://arxiv.org/pdf/2508.00751)).
  Amazon도 two-stage recsys의 candidate generator 평가에 OPE를 쓴다([Amazon Science](https://www.amazon.science/publications/off-policy-evaluation-of-candidate-generators-in-two-stage-recommender-systems)).

질문은 두 겹이다. (1) 어제까지의 로그 — 구정책이 고른 행동과 그 보상만 기록됨 — 로 **새 정책의 가치를 추정할 수 있나?**
(2) 그 추정치를 믿고 배포 게이트를 통과시켜도 되나, 아니면 **A/B로 돌려보내야 하나?** 공개 포트폴리오 공간에서
(1)의 estimator 나열은 존재하지만 (2)의 판정 계층은 사실상 비어 있다([혼잡도 근거는 POSITIONING.md](POSITIONING.md)).

## 2. 메커니즘 — 왜 가능한가, 왜 어려운가

**왜 가능한가.** 로깅 정책 $\pi_0$가 행동을 *확률적으로* 골랐고 그 propensity $\pi_0(a|x)$가 기록돼 있다면,
importance weight $w_i = \pi_e(a_i|x_i)/\pi_0(a_i|x_i)$ 로 관측 보상을 재가중한 평균은 새 정책 가치
$V(\pi_e)=\mathbb{E}[w\,r]$ 의 unbiased 추정이 된다 (Horvitz–Thompson 1952 계보(고전 문헌 — 원전 서지·URL 은
M4 문서화 단계에 보강, [POSITIONING.md §6](POSITIONING.md) 참조)의 IPS; bandit OPE 정식화는
[Dudík, Langford, Li 2011](https://arxiv.org/abs/1103.4601)). 로그가 곧 "무작위화의 흔적"을 담고 있기 때문에
counterfactual 평가가 성립한다.

**왜 어려운가.** 세 겹의 함정이 있다.
1. **bias–variance.** unbiased인 IPS는 weight 폭발로 분산이 커지고, 저분산인 DM(reward 모델 직접 예측)은 모델
   bias를 진다. SNIPS·Clipped-IPS·DR·Switch-DR·DRos 계보 전체가 이 trade-off 위의 다이얼이다
   ([SNIPS](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html) ·
   [Switch-DR](https://arxiv.org/abs/1612.01205) · [DRos](https://arxiv.org/abs/1907.09623)).
2. **overlap / support.** $\pi_e$가 가려는 (x,a)를 $\pi_0$가 밟은 적 없으면(deficient support) 재가중으로 복원할
   표본 자체가 없어 식별이 무너진다 ([Sachdeva–Su–Joachims KDD'20](https://arxiv.org/abs/2006.09438)).
3. **진단의 한계.** ESS·max-weight·support 체크는 분산형 위험을 경보하지만 임계값은 문헌상 산발적 folklore이지
   통합 프로토콜이 아니고([예: eligible actions 논의](https://arxiv.org/abs/2207.00632)), 결정적으로 **기록된
   propensity 자체가 틀린 경우**(미관측 교란(unobserved confounding) — 로깅이 기록 안 된 state에 의존)에는 원리적으로 blind다
   ([Amazon RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) ·
   [Namkoong et al. NeurIPS'20](https://arxiv.org/abs/2003.05623)).

## 3. 서사 — 2막 구조

- **1막 (도구):** estimator 패밀리의 bias–variance 아크. DM → IPS → SNIPS → Clipped-IPS → DR → Switch-DR → DRos —
  전부 numpy 직접 구현이 주인공이고, ground truth를 보유한 합성 DGP에서 bias²·variance·MSE 를 참값 대비 분해한다.
- **2막 (클라이맥스):** "**그 추정을 믿어도 되나?**" 진단(ESS·max-weight·support) → 기각 → estimator 선택 →
  `믿는다 / 못 믿는다 / A/B로 회귀` 3-way 판정의 **decision gate**. 이 게이트 규칙은 **본 레포의 제안 —
  산발적 folklore의 체계화 시도이며 표준이 아니다.** 예보력은 축 08·10에서 실증하고 실패 조건도 같이 전시한다.
- **에필로그 (정직성):** **진단이 못 보는 것.** confounding을 주입해도 ESS·max-weight가 unconfounded 케이스와
  똑같이 "양호"하게 나오는 대조표 — 게이트의 원리적 한계를 스스로 폭로하는 장으로 서사를 닫는다.
- **hero 3장(README 정면):** ① decision-gate 플로차트 SVG ② regime map 히트맵(표본×overlap grid 최저-MSE
  estimator — 플로차트의 증거층) ③ "진단이 못 보는 것" 대조표.

## 4. 검증가능 주장 목록

각 주장은 반증 가능한 형태로 적고, 검증 수단(실험 축 ID)과 현재 상태를 병기한다. 지금은 전부 [TARGET] —
결과가 LEDGER에 커밋되면 상태가 갱신된다(M0 작성 시점 기록 — 검증 완료 여부의 현재 상태는
docs/LEDGER.md·PLAN §4 가 정본). 축 ID는 PLAN.md 확정본과 동일하다.

**A. 알려진 붕괴 패턴의 재현** (문헌의 패턴이 본 DGP·실데이터에서 재현되는가)

| # | 주장 | 근거 문헌 | 검증 축 | 상태 |
|---|---|---|---|---|
| A0 | 자기 numpy 구현이 obp 와 수치 일치한다 (불일치 시 그대로 보고) | [obp](https://github.com/st-tech/zr-obp) | probe M0-B → M1 교차검증 표 | [TARGET] |
| A1 | 소표본에서 IPS/DR 분산 지배로 DM 우세, 표본 증가 시 우세가 역전된다(regime 교차) | [OBP 벤치마크](https://arxiv.org/abs/2008.07146) | 01 표본 n | [TARGET] |
| A2 | 로깅 softmax β 증가(준결정적) → overlap 축소 → weight 폭발로 IPS 계열 MSE 악화 | [OBP docs](https://zr-obp.readthedocs.io/en/latest/) | 02 로깅 β | [TARGET] |
| A3 | 타깃–로깅 괴리 증가에서 분산 폭발과 estimator 순위 역전이 일어난다 | [PAS-IF AAAI'23](https://ojs.aaai.org/index.php/AAAI/article/view/26195) | 03 괴리 | [TARGET] |
| A4 | deficient support 에서 IPS 계열이 파국적 bias(식별 불능)를 보인다 | [KDD'20](https://arxiv.org/abs/2006.09438) | 04 support | [TARGET] |
| A5 | propensity 오지정 시 IPS 는 bias 직결, DR 은 reward 모델이 맞으면 생존·둘 다 틀리면 실패 | [DRUnknown](https://arxiv.org/abs/2404.01830) | 05 propensity 오지정 | [TARGET] |
| A6 | reward model 오지정 시 DM bias 는 표본을 늘려도 사라지지 않는다 | [MRDR](https://arxiv.org/abs/1802.03493) | 06 reward 오지정 | [TARGET] |
| A7 | 튜닝 없는 Switch-DR·DRos 가 error-CDF 상 단순 estimator 보다 불안정하다(IEOE 재현) | [IEOE](https://arxiv.org/abs/2108.13703) | 07 hyperparameter(IEOE) | [TARGET] |
| A8 | 합성 DGP 의 결론이 classification-to-bandit 멀티데이터셋에서 재현된다 | [c2b 변환 사용 예](https://arxiv.org/pdf/1802.03493) (프로토콜 계보는 Dudík et al. 2011/2014 계열 — [POSITIONING.md §2.1](POSITIONING.md) 참조) | 11 c2b | [TARGET] |
| A9 | OBD small 근사 ground truth 게이트에서 재현된다 (모든 비교에 bootstrap CI 병기, 점 비교 단정 금지) | [OBD](https://arxiv.org/abs/2008.07146) | 12 OBD small | [TARGET] |
| A10 | 액션 수 증가에서 IPS/DR 분산이 비실용화되고 MIPS 가 이를 구제한다 | [MIPS ICML'22](https://arxiv.org/abs/2202.06317) | 13 액션 수+MIPS | [TARGET·스트레치] |

**B. 제안 게이트의 예보력 + blind spot** (본 레포 고유의 클라이맥스)

| # | 주장 | 근거 문헌 | 검증 축 | 상태 |
|---|---|---|---|---|
| B1 | ESS·max-weight·support 진단값이 축 01–04 형 스트레스에서 실추정오차를 예보한다(순위 상관) | folklore 관행([예](https://arxiv.org/abs/2207.00632)) | 08 진단 예보력 | [TARGET] |
| B2 | 제안 decision gate 가 무진단 배포 대비 "잘못된 정책을 배포할 확률"을 낮춘다 — **본 레포의 제안, 표준 아님** | (본 레포 제안) | 08+10 | [TARGET] |
| B3 | MSE 가 비슷한 estimator 들이 의사결정 metric(잘못 배포 확률·rank-corr)에서는 크게 갈린다 | [SharpeRatio@k ICLR'24](https://arxiv.org/abs/2311.18207) | 10 의사결정 metric | [TARGET] |
| B4 | confounding 주입 시(기록 propensity≠진짜) 표준 estimator 가 일제히 편향되는데 ESS·max-weight 는 unconfounded 케이스와 구분 불가 — 대조표로 전시 | [RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) · [NeurIPS'20](https://arxiv.org/abs/2003.05623) | 09 confounding 대조표 | [TARGET] |
| B5 | MSM Λ-sweep 으로 결론이 뒤집히는 breakdown Λ\* 를 보고할 수 있다 | [Kallus & Zhou](https://arxiv.org/abs/1805.08593) | 14 Λ-sweep | [TARGET·조건부: 1일 수치안정성 probe GO 시에만] |

**불확실 태그.** ① obp 의 PyPI 릴리스 버전 표기가 조사 내 상충한다(0.5.5 vs 0.5.7) — probe M0-B
(`experiments/probes/probe_obp_crossval.py`)에서 PyPI 재확인으로 해소 예정, 여기서 단정하지 않는다.
② OBD small 근사 ground truth 의 표본 오차 크기는 M3 에서 실측 전까지 미확인 — 그래서 A9 에 CI 병기 규칙을 미리 박아 둔다.
③ Λ-최적화의 multi-action 수치 안정성 미검증 — B5 를 조건부로 격리한다.

## 5. 청중 레이어

- **30초 (채용자):** "모든 회사가 로그는 쌓지만, 그 로그로 *안 해본 정책*을 평가하는 건 통계적 함정투성이다.
  이 레포는 언제 어떤 estimator 가 부러지는지 ground-truth 시뮬레이션으로 보이고, 부러짐을 사전에 감지하는
  배포 게이트를 코드로 준다." — hero 3장(플로차트·regime map·대조표)만 보고 닫아도 완결.
- **5분 (실무 DS/MLE):** §2 메커니즘 + decision gate 사용법(진단 체크리스트 → 3-way 판정) + regime map 읽는 법.
  단서 포함: 게이트는 본 레포의 제안이며, §4-B4 의 blind spot 이 적용 한계다.
- **30분 (방법론 독자):** estimator 계보의 bias²·variance·MSE 분해, IEOE 식 robustness 프로토콜, confounding
  대조표의 식별 논리(왜 진단이 원리적으로 blind 인가), obp 이중 교차검증·LEDGER 정직성 장치까지.

## 6. 범위 · 비범위

**범위(코어).** multi-action *single-step* logged bandit OPE. 코어 estimator 7종(DM·IPS·SNIPS·Clipped-IPS·DR·
Switch-DR·DRos, 전부 numpy) + ESS/max-weight/support 진단 + bootstrap CI + SLOPE(hyperparameter 선택 — 축 07 과
함께 M2 구현). 데이터 3트랙: 합성 DGP(본체) ·
classification-to-bandit(UCI/OpenML) · OBD small(ZOZO, [데이터](https://research.zozo.com/data.html)).
실험 축 01–12 가 코어, 13·14 는 스트레치(각각 선행 probe GO 조건부).

**비범위(명시적 제외).**
- **OPL / policy learning · CATE** — 포트폴리오 내 역할 분담: dunnhumby 레포(binary treatment·CATE 정책) ·
  causal-inference 레포(CATE 카탈로그)와 상호 링크로 경계 선언.
- **slate/ranking OPE(PI·IIPS·RIPS) · sequential RL OPE(FQE·DICE)** — 레포 정체성 유지를 위해 제외.
- **proximal identification 본류** — 연구 레포(decision-frontier) 소관. 본 레포는 confounding
  **대조표(축 09) + 조건부 Λ-sweep(축 14)에서 의도적으로 멈춘다** — 입구만 보여주고 본류로 넘긴다.

**설계 기본값(configs — 결과 수치 아님).** n=10,000 · K(n_actions)=10 · dim_context=5 · β_log=1.0 · β_eval=3.0 ·
reward_noise σ=0.5 · support_deficiency=0.0 · confounding_strength=0.0 · n_seeds=50 · bootstrap 2,000회(α=0.05).

**마일스톤(달력 정직 — 연구 병행 파트타임, 총 2–3개월).** M0 스캐폴드+probe 2개(GO/NO-GO) → M1 1.5–2.5주
(DGP+estimator 7종+obp 교차검증) → M2 2–3주(코어 축 01–10 — publishable 최소선) → M3 1.5–2.5주(11–12+플레이북+hero)
→ M4 1–1.5주(portfolio-design 문서) → M5 조건부 스트레치(13·14). 상세는 PLAN.md.

---

## 7. M8 재정위 부록 (2026-08-10 — 본문 §1–6 은 M0 시점 기록으로 동결)

§3 의 2막 서사("ground truth 를 보유한 합성 DGP 에서 분해")는 M0–M7 의 역사 기록이다. M8 부터
현행 서사는 **3막(GT-미상 본편 / 백스테이지 채점)** 으로 역전되었다 — 실무의 기본 상태(참값을
아무도 모른다)를 본편으로 올리고, 참값 보유 무대는 본편 신호(게이트·validity battery·Λ\*)의
예보력과 맹점을 채점하는 백스테이지가 되었다. 실험 축은 17–20(사전등록 PLAN §3.5)이 더해졌고
§4 의 주장 표는 백스테이지 무대의 주장으로 유효하게 남는다. 현행 서사의 정본은 README 와
[COMMS_BRIEF_v2.md](COMMS_BRIEF_v2.md), 용어는 [GLOSSARY.md](GLOSSARY.md) §8 이다.
