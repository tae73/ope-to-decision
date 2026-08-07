# ope-to-decision

**From logged bandit feedback to deployment decisions.**

> **A multi-action OPE benchmark + deployment-gate playbook: before the A/B test, estimate a new
> policy's value from the recommendation logs you already have — and rule on when that estimate
> cannot be trusted.**

[🇰🇷 한국어 (canonical)](README.md) · **🇺🇸 English**

![gate forecast](https://img.shields.io/badge/gate_forecast-4.6%25_trust_vs_44.4%25_fallback-2a78d6)
![DR robustness](https://img.shields.io/badge/DR_robustness-4%2F4_real_datasets-008300)
![obp crossval](https://img.shields.io/badge/obp_crossval-rel__diff%E2%89%A41e--8_(7_est_%C3%97_2_tracks)-4a3aa7)
![axes](https://img.shields.io/badge/axes-01%E2%80%9312%C2%B714%E2%80%9316_executed_%C2%B7_13_dropped--by--probe-6f6e66)
![tests](https://img.shields.io/badge/tests-63_passed-1baf7a)
![license](https://img.shields.io/badge/license-MIT-b3b2a9)

---

## ⏱️ TL;DR — 30 seconds

- **Problem.** You want to know what a new recommendation policy is worth before it touches
  traffic. A/B slots are scarce and a bad policy burns revenue and UX — yet the logs you have
  only recorded the actions the *old* policy chose. This is precisely the motivation Spotify laid
  out at WSDM'19 ([paper](https://research.atspotify.com/publications/offline-evaluation-to-make-decisions-about-playlistrecommendation-algorithms)).
- **Approach.** We implemented seven OPE estimators (DM · IPS · SNIPS · Clipped-IPS · DR ·
  Switch-DR · DRos) in plain numpy, adversarially cross-validated them against obp/sb-obp, broke
  them systematically along 12 axes in a synthetic DGP with known ground truth, and then
  distilled when log-computable diagnostics (ESS · max-weight) forecast those breakdowns into a
  **"trust / distrust / fall back to A/B" deployment gate**. Reproduced on real data
  (4 UCI datasets, ZOZO production logs).
- **Three headline results.**
  1. **The gate forecasts.** Under pre-registered thresholds, the share of large errors
     (relative error > 10%) is **4.6%** on "trust" verdicts vs **44.4%** on "fall back to A/B"
     (LEDGER `m2-08-forecast`).
  2. **It holds on real data** — DR's robustness to misspecification reproduces on 4/4 datasets
     (DM bias −0.026 to −0.387 vs |DR bias| ≤ 0.0032), and the gate correctly rules the ZOZO production
     logs **DISTRUST** (LEDGER `m3-11-dr-robust` · `m3-12-gate-demo`).
  3. **The limits are mapped too.** Under unobserved confounding the diagnostics are blind in
     principle: ESS sits flat at 0.823→0.822 while IPS bias grows from 0 to −0.057
     (LEDGER `m2-09-blindspot`).
- **Impact.** A reproducible decision playbook for which OPE estimate to trust, and when —
  demonstrated against ground truth both where it works and where it fails. Full rules:
  [docs/PLAYBOOK.md](docs/PLAYBOOK.md).

## Headline results — three hero figures

<p align="center"><img src="assets/decision_gate_flowchart_en.svg" width="860" alt="decision gate flowchart — from diagnostics to the 3-way verdict"><br>
<sub><b>Figure 1 — the decision gate.</b> Log-side diagnostics (ESS · max-weight · support) →
rejection → estimator selection → a 3-way verdict: trust / distrust / fall back to A/B. The
thresholds were pre-registered in M1 and only <b>evaluated</b> on axis 08 (no tuning) — they are
this repo's <b>proposal</b>, not a standard.</sub></p>

<p align="center"><img src="results/figures/hero_regime_map.png" width="860" alt="regime map — lowest-MSE winner across the n×β_log grid"><br>
<sub><b>Figure 2 — regime map (the evidence layer behind the flowchart).</b> Lowest-MSE winner
map + gate majority vote over a 28-cell n × β_log grid. The three DR-family variants trade
dominance; DM and the IPS family take zero outright wins. The biggest finding is a warning:
<b>diagnostic power goes missing in small samples</b> — the gate catches near-deterministic
logging (β=16) only at n≥2000, while at n=500 the majority vote says trust even though that
cell's IPS MSE is 70.6× the winner's (LEDGER <code>m3-hero-map</code>).</sub></p>

<p align="center"><img src="results/figures/09_confounding_blindspot.png" width="860" alt="confounding blind spot — diagnostics flat while bias grows"><br>
<sub><b>Figure 3 — what the diagnostics can't see.</b> As confounding strength γ grows, ESS
computed from logged propensities stays flat (0.823→0.822) while bias keeps growing (0→−0.057).
Hand the <i>true</i> propensities to the same formula and it collapses to 0.018, detecting the
failure — the blindness lives in the input, not the formula (LEDGER
<code>m2-09-blindspot</code>).</sub></p>

## How to read this (layers)

| Time | What to read |
|---|---|
| **30 sec** | The TL;DR above + the three hero figures |
| **5 min** | [The two-act story](#the-two-act-story) → [Findings by axis](#findings-by-axis--one-line-each) → [Translating into business impact](#translating-into-business-impact) → [What didn't break](#what-didnt-break--where-our-expectations-were-wrong) |
| **Deep dive** | [notebooks/](notebooks/README.md) — log EDA → DGP anatomy → estimator walkthrough → diagnostics & gate → results deep-dive (5 notebooks, outputs included) |
| **Reproduce** | [Quick Start](#quick-start) + [experiments/README.md](experiments/README.md) |
| **30 min** | [docs/PLAYBOOK.md](docs/PLAYBOOK.md) → [docs/CONCEPT.md](docs/CONCEPT.md) → [docs/POSITIONING.md](docs/POSITIONING.md) → [docs/LEDGER.md](docs/LEDGER.md) |

## The two-act story

**Act 1 — evaluating a policy from logs.** An e-commerce recommendation team has a new candidate
policy. Before any A/B test, they estimate its expected reward $V(\pi_e)$ from yesterday's logs
alone — a motivation Netflix, Airbnb, and Amazon have all published as well
([Netflix](https://netflixtechblog.com/reinforcement-learning-for-budget-constrained-recommendations-6cbc5263a32a) ·
[Airbnb](https://arxiv.org/pdf/2508.00751) · [Amazon](https://www.amazon.science/publications/off-policy-evaluation-of-candidate-generators-in-two-stage-recommender-systems)).
Estimators enter the story here as *tools*: a bias–variance arc running from DM's model-bias
extreme to IPS's variance extreme, through SNIPS → DR → Switch-DR/DRos (example figure:
[axis 01](results/figures/01_sample_size.png) — the regime crossover from DM winning at small n
to IPS/DR winning at large n).

**Act 2 — should you trust that estimate?** The real question is not the point estimate but the
*trust verdict*. Using ground truth, we separate the regimes where ESS · max-weight · support
diagnostics forecast the risk from the regimes they are **blind to in principle** — such as
unobserved confounding
([Amazon Science RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding)) —
and systematize the result into a "trust / distrust / send it to A/B" deployment gate. This
decision rule is **this repo's proposal (an attempt to systematize folklore scattered across the
literature), not an established standard.**

## What we built and how we verified it

| Component | What it is | Verification |
|---|---|---|
| 7 estimators | DM · IPS · SNIPS · Clipped-IPS · DR · Switch-DR · DRos — pure numpy (`src/ope/estimators.py`) | Triple-checked: 63 property tests + **two-track cross-validation against obp (py3.9) and sb-obp (py3.12) with rel_diff ≤ 1e-8**, branches actually firing (LEDGER `m1-crossval`) + hand-computed identities |
| Diagnostics & gate | ESS · max-weight · support proxy + 3-way `decision_gate` (`src/ope/diagnostics.py`) — an executable demonstration of the OPE spec in `dag-registry` (private repo) | Pre-registered thresholds **evaluated** on axis 08 (no tuning) — forecasting power demonstrated (LEDGER `m2-08-forecast`) |
| SLOPE | Data-driven hyperparameter selection (Su+ ICML'20) | **The axis-07 experiment caught a reversed-ladder bug in our implementation** → fixed and pinned with a regression test — post-fix, clipped tail p90 recovers 0.125→0.050 (LEDGER `m2-07-slope`) |
| Synthetic DGP | Multi-action bandit with known ground truth — knobs for overlap, support, misspecification, confounding (`src/ope/dgp.py`) | On-policy end-to-end check, confounding contrast identity, and other property tests |
| Real data, two tracks | classification-to-bandit (4 UCI datasets, exact ground truth) + OBD small (ZOZO production logs, approximate GT + CI) | §3.4 protocol: bootstrap CIs accompany every approximate-GT figure; no point-comparison claims |
| Playbook | [docs/PLAYBOOK.md](docs/PLAYBOOK.md) — gate rules, comparative-first principle, confounding disclaimer | Every number cites a LEDGER row (zero hand-made) |

## Findings by axis — one line each

Rows with numbers carry their LEDGER id; for the remaining axes, each figure and its paired CSV
are canonical for the quantitative detail.

| Axis | Finding | Evidence |
|---|---|---|
| 01 | The regime crossover, demonstrated: DM wins at small n, IPS/DR at large n — "trust the model when data is scarce, trust the data once it isn't," in a single figure | [figure](results/figures/01_sample_size.png) |
| 02 | ESS is not monotone in logging temperature (it peaks where logging aligns with the evaluation policy, then collapses) — and the MSE cliff sits between β=8 and 16, not at β=8 | [figure](results/figures/02_logging_beta.png) |
| 03 | The policy-gap sweep fails to flip DM (weights stay bounded) — the real casualty is naive clipping | [figure](results/figures/03_policy_gap.png) |
| 04 | π_e mass off the logged support is unidentifiable from logs — even DR retains residual bias, and the global support proxy is **fully blind** (0 signal vs 0.143 true deficiency) | `m2-04-proxy-blind` |
| 05 | Multiplicative pscore contamination makes IPS hundreds of times worse, while DR with a correct q̂ survives flat — "what saves DR is getting one of its two models right" | [figure](results/figures/05_propensity_misspec.png) |
| 06 | DM bias is sample-size-insensitive — growing n from 2.5k to 40k leaves misspecification bias untouched | [figure](results/figures/06_reward_misspec.png) |
| 07 | Untuned-hyperparameter instability is exclusive to the model-free family (clipped) — SLOPE (paper's ladder direction required) recovers the tail (p90 0.125→0.050) | `m2-07-slope` |
| 08 | Forecasting power of the pre-registered gate: P(large error \| trust) = 4.6% vs P(large error \| fall back to A/B) = 44.4% — the support arm never fired (degenerate, honestly recorded) | `m2-08-forecast` |
| 09 | Under confounding the diagnostics stay flat (ESS 0.823→0.822) while only the bias grows (−0.057) — hand the same formula the true propensities and it detects the failure (0.018) | `m2-09-blindspot` |
| 10 | Decision safety is a property of the **comparison design**, not the estimator: same-log comparisons cancel errors (fg 0.0), mixed comparisons revive the bias (DM fg 0.15 / fs 0.375), and absolute thresholds inherit all of it | `m2-10-comparison` |
| 11 | On real data (4 UCI datasets) DR robustness reproduces 4/4 + the percentile bootstrap cannot catch structural bias (9/28 CIs fail to cover the truth) | `m3-11-dr-robust` |
| 12 | The gate correctly rules the ZOZO production logs DISTRUST (ESS/n 0.034 · max w 278) — the lack of discriminative power (38 random / 42 bts clicks · GT CI ±32%) was declared up front | `m3-12-gate-demo` |
| 13 | [stretch] **Probe returned NO-GO — dropped (recorded honestly)**: under this bounded-logit softmax DGP, even K=2000 leaves max_w≈2.8 and IPS uncollapsed — the "action explosion → MIPS to the rescue" story cannot arise in this DGP family (consistent with the bounded-weight lesson of axis 03); no restart without a DGP redesign | `m5-probe-13` |
| 14 | [stretch] The MSM Λ-sweep, executed after its probe returned GO — the breakdown Λ* where policy ranking becomes undecidable has median ≈1.07 (γ=0.5) and ≈1.04 (γ=1.5): in this setup, the stronger the confounding, the smaller the assumed distortion that already breaks the ranking. The MSM bound is an existing published method (Kallus & Zhou 2018), shown as a **tool demonstration** — Λ is an analyst's assumption, not identifiable from data | `m5-14-lambda` |
| 15 | Same log, same weights — but as the metric deepens from CTR to CVR to REV, the detectability limit climbs like a ladder (events get sparse, price adds a heavy tail); and since the diagnostics only look at weights, they are **metric-invariant**: a gate trust verdict is no guarantee of discriminative power on deep metrics | [figure](results/figures/15_funnel_reliability.png) |
| 16 | The multi-metric guardrail gate (Δ̂CTR>0 ∧ Δ̂REV≥−g ∧ HHI≤h) is the vector extension of the comparative-first principle — but overlapping metrics share the same weights, so joint-gate errors arrive in **clusters** (never multiply per-metric error rates); advertiser exposure shares and HHI are computed exactly, no OPE involved | [figure](results/figures/16_business_gate.png) |

## Translating into business impact

<p align="center"><img src="results/figures/15_funnel_reliability.png" width="860" alt="axis 15 — the funnel reliability ladder: on the same log, the detectability limit steepens from CTR to CVR to REV"><br>
<sub><b>Axis 15 — the funnel reliability ladder.</b> Same log, same weights; only the metric deepens
from CTR to CVR to REV. The paired CSV (<code>results/tables/15_funnel_reliability.csv</code>) is
canonical for the quantitative detail — numbers flow through the entered LEDGER rows
<code>m6-15-ladder</code> · <code>m6-16-gate</code> (raw values re-derivable from the paired CSVs).</sub></p>

The same log and the same importance weights evaluate an entire vector of business metrics at once
(CTR · CVR · REV — CVR is per-session, see [GLOSSARY](docs/GLOSSARY.md) §7), but the trust they
deserve is not the same: the deeper the funnel, the sparser the events and the heavier the
price tail, so the detectability limit steepens like a ladder (axis 15). What makes this dangerous
is that the diagnostics only look at weights and are therefore **metric-invariant** — a gate trust
verdict forecasts weight-variance risk, it does not certify discriminative power on deep metrics
like revenue. The multi-metric guardrail gate (axis 16) extends the comparative-first principle to
that vector, but overlapping metrics share the same weights, so joint-gate errors arrive in
clusters rather than as a product of per-metric error rates. The canonical rules and warnings live
in [docs/PLAYBOOK.md](docs/PLAYBOOK.md) §8 ("business translation"). Retention and other
cross-session, long-horizon metrics are unidentifiable from single-step bandit OPE, and we declare
them **honestly out of bounds** (RL OPE territory — even within-session proxies were rejected as
self-deception risks).

## What didn't break — where our expectations were wrong

This repo's honesty protocol treats non-events as results. Where we expected one thing and
report another:

1. **Axis 02** — the "explodes at β≥8" expectation was only half right: at β=8, IPS/DR still
   beat DM. The cliff sits between 8 and 16.
2. **Axis 03** — the expected "DM flip at large policy gap" never happened: the logging logits
   are bounded, so weights simply don't explode. The real finding is that λ=p90 naive clipping
   becomes worse than DM.
3. **Axis 04** — the support proxy turned out worse than the expected "partial signal": it is
   **fully blind**. The playbook states explicitly that the gate's support arm is kept for form
   and not trusted.
4. **Axis 10** — the original design handed every estimator a perfect score (a null). After
   root-causing it (common-error cancellation + rank-preserving misspecification), we redesigned
   around the better question — "a property of the comparison design" — with the design history
   preserved in the script docstring.
5. **Axis 12** — OBD small has no estimator-discriminating power (declared up front). The axis's
   value is not discrimination but demonstration: real-log protocol compliance, and whether the
   gate actually rejects logs like these.
6. **The SLOPE implementation bug** — the axis-07 experiment caught a reversed ladder (opposite
   to the paper). Recorded, with the fix and a regression test, as a case of an experiment
   validating the code.

**Additional scoping.** The synthetic conclusions are conditional on a single environment
structure (struct_seed=7) — regime boundary locations may shift across environments (directional
claims only). c2b rewards are deterministic, so DR residuals carry no noise channel. The decision
rule's thresholds are pre-registered and untuned — porting them to another domain without
recalibration is unfounded. **The identification-and-correction mainline for unobserved
confounding (proximal methods and their kin) is out of scope** — this repo deliberately stops at
exhibiting the blind spot (axis 09).

## Out of scope (boundary declaration)

- **OPL · CATE · policy learning** — out of scope. Binary-treatment policies and CATE live in
  [kr_segmentation_causal_targeting_dunnhumby](https://github.com/tae73/kr_segmentation_causal_targeting_dunnhumby),
  and the CATE method catalog in `causal-inference` (private repo, cross-linked).
- **Slate/ranking OPE (PI · IIPS · RIPS) · RL OPE (FQE · DICE)** — out of scope. This repo's
  identity is *multi-action single-step logged bandit* OPE.
- **The mainline of identification under confounding (proximal methods and their kin)** — the
  research track's territory. This repo **deliberately stops** at the axis-09 "what diagnostics
  can't see" contrast (plus the axis-14 Λ-sweep, executed after its probe returned GO — a tool
  demonstration of an existing published bound).
- Diagnostics spec document ↔ executable implementation: complementary to
  `dag-registry` (private repo). The per-axis experiment pattern inherits the
  [mta-simulation](https://github.com/tae73/mta-simulation) house style.

## Quick Start

```bash
cd ope-to-decision
uv sync --extra dev        # main env (Python 3.11+)
uv run pytest              # 63 property tests (identities, statistical properties, regression pins)

# Reproduce an axis experiment (e.g. axis 01 — regenerates the figure + CSV pair)
uv run python experiments/01_sample_size.py

# obp/sb-obp cross-validation (pinned-env setup, pitfalls included)
#   → see the "m1_crossval" section of experiments/README.md (matplotlib<3.7 pin required, etc.)
```

Real data: the four OpenML datasets download automatically on first run (cached under
`data/openml`); place OBD small locally as described in [data/README.md](data/README.md)
(no redistribution).

## Notebooks — the deep-dive layer (EDA → results)

If the README is the curated conclusion, the notebooks show the process — five volumes,
committed with executed outputs. Notebooks are a derived/exploratory layer: canonical numbers
flow only through [docs/LEDGER.md](docs/LEDGER.md) (conventions: [notebooks/README.md](notebooks/README.md)).

| Volume | One line |
|---|---|
| [00 Log EDA](notebooks/00_log_eda.ipynb) | Look at the log before OPE — reward sparsity, propensity tails, and exposure long-tail in the OBD log + weight geometry across the four c2b datasets |
| [01 DGP anatomy](notebooks/01_dgp_anatomy.ipynb) | Why this benchmark knows the truth — weight geometry per knob (β·δ·γ), the confounding device, and a v_true cross-check |
| [02 Estimator walkthrough](notebooks/02_estimator_walkthrough.ipynb) | All seven estimators, formula → code → number, line by line — how each tames the weights, plus a mini bias-variance arc |
| [03 Diagnostics & gate anatomy](notebooks/03_diagnostics_gate.ipynb) | ESS, max-weight, and support proxy computed step by step + gate verdict traces + a confounding-blind-spot reproduction |
| [04 Results deep-dive](notebooks/04_results_deepdive.ipynb) | Re-reading the committed CSVs — regime-map margins, non-monotone ESS, hyperparameter tails, funnel quantiles, Λ* distributions |

## Repository Structure

```
ope-to-decision/
├── src/ope/               # estimators (7 + SLOPE + bootstrap) · dgp · diagnostics (gate) · policies · datasets · business (funnel metric vector)
├── experiments/           # axes 01–12·14–16 + probes/ + m1_crossval/ + hero_regime_map (index: experiments/README.md)
├── notebooks/             # 5-volume deep-dive layer (00 EDA → 04 results) — derived layer (notebooks/README.md)
├── results/figures|tables # experiment outputs — NN_* figure↔CSV 1:1 pairing (numbers flow through docs/LEDGER.md)
├── docs/                  # PLAYBOOK · CONCEPT · POSITIONING · LEDGER · GLOSSARY · COMMS_BRIEF
├── assets/                # decision-gate flowchart SVGs (ko/en)
├── configs/  tests/       # Hydra design defaults / 63 property tests
├── data/                  # gitignored — no redistribution of source data (data/README.md)
└── PLAN.md  CLAUDE.md     # milestones & gates / agent working agreement
```

## Document map (the 30-minute layer)

| Document | Contents |
|---|---|
| [docs/PLAYBOOK.md](docs/PLAYBOOK.md) | **The deployment-gate playbook** — 3-way rules, the comparative-gate-first principle, the confounding disclaimer (numbers cite LEDGER rows only) |
| [docs/CONCEPT.md](docs/CONCEPT.md) | Concept one-pager — motivation, mechanism, testable claims |
| [docs/POSITIONING.md](docs/POSITIONING.md) | Prior-art check, differentiation gaps, adjacent-repo boundaries (every claim carries a source URL) |
| [docs/LEDGER.md](docs/LEDGER.md) | **The single source of truth for numbers** — every number in this README flows through it |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | The single KO/EN terminology standard |
| [PLAN.md](PLAN.md) | Milestones, gates, axis mapping, fallback chains (including the completed M0–M4 history) |

## Attribution & References

- **Data:** [Open Bandit Dataset](https://research.zozo.com/data.html) (ZOZO Research — separate
  terms of use; this repo commits conversion scripts only) · UCI/OpenML (optdigits · satimage ·
  pendigits · letter).
- **Cross-validation references:** [obp / Open Bandit Pipeline](https://github.com/st-tech/zr-obp) ·
  [sb-obp](https://github.com/sb-ai-lab/sb-obp) — APIs consulted for reference; every
  implementation is our own numpy.
- **Key literature:** DM/IPS/DR [Dudík-Langford-Li 2011](https://arxiv.org/abs/1103.4601) · SNIPS
  [Swaminathan-Joachims 2015](https://papers.nips.cc/paper/2015/hash/39027dfad5138c9ca0c474d71db915c3-Abstract.html) ·
  Switch-DR [Wang+ 2017](https://arxiv.org/abs/1612.01205) · DRos [Su+ 2020](https://arxiv.org/abs/1907.09623) ·
  SLOPE [Su+ ICML'20](http://proceedings.mlr.press/v119/su20d/su20d.pdf) · IEOE
  [Saito+ RecSys'21](https://arxiv.org/abs/2108.13703) · deficient support
  [Sachdeva+ KDD'20](https://arxiv.org/abs/2006.09438) · Δ-OPE [Jeunen+ RecSys'24](https://arxiv.org/abs/2405.10024) ·
  confounded eval [Amazon RecSys'23](https://www.amazon.science/publications/offline-recommender-system-evaluation-under-unobserved-confounding) ·
  OBD [Saito+ 2020](https://arxiv.org/abs/2008.07146). Full lineage: [docs/POSITIONING.md](docs/POSITIONING.md) §6.

## Honesty footnotes

1. **Numbers flow through the LEDGER only.** Every result number in this README cites a row of
   [docs/LEDGER.md](docs/LEDGER.md), built from committed outputs (`results/tables/`), with the
   row id attached. Abbreviated figures in the text and badges (4.6%, 44.4%, −0.057, …) are
   roundings of those rows' raw values — the LEDGER is canonical for raw values and precision.
   Process metadata such as test counts (the tests badge) are repo facts, not experimental
   results, and sit outside the LEDGER's scope.
2. **The decision rule is a proposal.** The gate rules and thresholds were pre-registered as
   folklore in M1 and only **evaluated** on axis 08 (no tuning, no calibration) — not a standard,
   and the failure conditions (the axis-09 blind spot, the degenerate support arm) are exhibited
   alongside.
3. **Approximate-ground-truth protocol.** OBD small's ground truth is approximate, so every
   related figure carries bootstrap CIs and no point comparison is asserted (kept semantically
   distinct from the exact ground truth of the synthetic axes — LEDGER `m3-12-gate-demo`).
4. **Data protection.** `data/` is gitignored — the licenses forbid redistribution
   ([data/README.md](data/README.md)).

*License: MIT (code) — data follow the terms of their respective sources. This English README is
the twin of the [Korean canonical](README.md): a natural rewrite, not a literal translation
(terminology per [docs/GLOSSARY.md](docs/GLOSSARY.md)), with every number citing the same
[docs/LEDGER.md](docs/LEDGER.md) rows.*
