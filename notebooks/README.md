# notebooks/ — 상세 분석 노트북 (EDA → 결과 심층)

README 가 큐레이션된 결론이라면, 노트북은 **과정을 보여주는 층**이다 — 로그를 눈으로 보고,
DGP 를 해부하고, estimator 를 수식→코드→수치로 한 단계씩 밟고, committed 결과를 다시 파헤친다.

## 지위 규약 (canonical-owner)

노트북은 **파생·재현·탐색 층**이다 — 정본이 아니다.

- 이 레포의 **정본 수치는 `docs/LEDGER.md`** 뿐이다. 노트북이 기존 결과를 인용할 때는 LEDGER
  행 id(예: `m2-08-forecast`)를 병기한다.
- 노트북 셀이 **새로 계산한 수치는 재현/탐색 지위**(LEDGER 미등재)다 — 각 권 상단 배너에 명시.
  문서(README·PLAYBOOK)에 승격하려면 실험 스크립트 → committed CSV → LEDGER 등재 경로를 밟아야 한다.
- 주제별 canonical owner: 실험 결과 = `experiments/NN_*.py` + `results/tables/NN_*.csv`,
  방법론 산문 = `docs/PLAYBOOK.md`, 용어 = `docs/GLOSSARY.md`. 노트북은 이들을 **참조**한다.

## 읽는 순서

| 권 | 주제 | 정본 owner (참조 대상) |
|---|---|---|
| [`00_log_eda.ipynb`](00_log_eda.ipynb) | 로그드 밴딧 데이터 EDA — OBD 실로그 + c2b 4종: "OPE 전에 로그부터 보라" | `src/ope/datasets.py`, 축 11–12 |
| [`01_dgp_anatomy.ipynb`](01_dgp_anatomy.ipynb) | 합성 DGP 해부 — 노브(β·δ·γ)별 weight 기하·v_true 검산 | `src/ope/dgp.py`, PLAN §2 |
| [`02_estimator_walkthrough.ipynb`](02_estimator_walkthrough.ipynb) | estimator 7종 수식→코드→수치 + bias-variance 아크 | `src/ope/estimators.py`, 축 01–07 |
| [`03_diagnostics_gate.ipynb`](03_diagnostics_gate.ipynb) | 진단·게이트 해부 — ESS·max-weight 계산 과정 + blind spot | `src/ope/diagnostics.py`, 축 08–09 |
| [`04_results_deepdive.ipynb`](04_results_deepdive.ipynb) | committed CSV 심층 재해석 — regime map·funnel·축 크로스컷 | `results/tables/*.csv`, LEDGER |

## 재현

저작은 py:percent 소스(`_src/*.py`) → jupytext 변환 → 실행 커밋 파이프라인이다:

```bash
uv sync --extra dev
uv run jupytext --to notebook --execute \
  --output notebooks/00_log_eda.ipynb notebooks/_src/00_log_eda.py
```

- 00권의 OBD 섹션은 `data/obd/` 미배치 시 안내 후 자동 스킵된다(`data/README.md` 참조).
- 데이터 보호 규약: OBD **원자료 row 출력 금지** — 노트북에는 집계 통계·figure 만 담는다(재배포 아님).
