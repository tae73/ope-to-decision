# notebooks/ — 상세 분석 노트북 (본편 walkthrough + 백스테이지 심층)

README 가 큐레이션된 결론이라면, 노트북은 **과정을 보여주는 층**이다 — 로그를 눈으로 보고,
DGP 를 해부하고, estimator 를 수식→코드→수치로 한 단계씩 밟고, committed 결과를 다시 파헤친다.
M9 부터 6권은 **무대(본편/백스테이지 — GLOSSARY §8)** 로 나뉘어 읽힌다.

## 지위 규약 (canonical-owner)

노트북은 **파생·재현·탐색 층**이다 — 정본이 아니다.

- 이 레포의 **정본 수치는 `docs/LEDGER.md`** 뿐이다. 노트북이 기존 결과를 인용할 때는 LEDGER
  행 id(예: `m2-08-forecast`·`m8-17-matrix`)를 병기한다.
- 노트북 셀이 **새로 계산한 수치는 재현/탐색 지위**(LEDGER 미등재)다 — 각 권 상단 배너에 명시.
  문서(README·PLAYBOOK)에 승격하려면 실험 스크립트 → committed CSV → LEDGER 등재 경로를 밟아야 한다.
- 주제별 canonical owner: 실험 결과 = `experiments/NN_*.py` + `results/tables/NN_*.csv`,
  방법론 산문 = `docs/PLAYBOOK.md`, 용어 = `docs/GLOSSARY.md`. 노트북은 이들을 **참조**한다.
- 본편 규약(05권): `write_decision_csv`·`reveal()` 은 노트북에서 호출 금지(정본 산출물 보호) —
  frontstage live 계산은 합성 1로그 한정, 축 집계는 committed CSV 읽기 전용.

## 읽는 순서 — 본편(frontstage) 먼저, 백스테이지(backstage)는 근거로

**본편 — 로그만 보는 실무자의 길** (00 → 03 → 05):

| 권 | 주제 | 무대 | 정본 owner (참조 대상) |
|---|---|---|---|
| [`00_log_eda.ipynb`](00_log_eda.ipynb) | 로그드 밴딧 데이터 EDA — OBD 실로그 + c2b 4종: "OPE 전에 로그부터 보라" | 본편 시점 | `src/ope/datasets.py`, 축 11–12·20–21 |
| [`03_diagnostics_gate.ipynb`](03_diagnostics_gate.ipynb) | 진단·gate v1 해부 — ESS·max-weight 계산 과정 + blind spot(battery 합류는 05권) | 본편 | `src/ope/diagnostics.py`, 축 08–09 |
| [`05_gt_unknown_protocol.ipynb`](05_gt_unknown_protocol.ipynb) | GT-미상 프로토콜 walkthrough — battery 4-arm 손 재계산·오염 주입 실연·축 17–21 채점표·reveal 없는 실전 카드 | 본편 | `experiments/_practitioner.py`·`src/ope/validity.py`, 축 17–21·LEDGER M8/M9 행 |

**백스테이지 — 참값 보유 무대의 해부** (01 → 02 → 04):

| 권 | 주제 | 무대 | 정본 owner (참조 대상) |
|---|---|---|---|
| [`01_dgp_anatomy.ipynb`](01_dgp_anatomy.ipynb) | 합성 DGP 해부 — 백스테이지가 참값을 아는 이유(로그 층/oracle 층·노브별 weight 기하·v_true 검산) | 백스테이지 | `src/ope/dgp.py`, PLAN §2 |
| [`02_estimator_walkthrough.ipynb`](02_estimator_walkthrough.ipynb) | estimator 7종 수식→코드→수치 + bias-variance 아크 | 본편 도구·백스테이지 채점 | `src/ope/estimators.py`, 축 01–07 |
| [`04_results_deepdive.ipynb`](04_results_deepdive.ipynb) | committed CSV 심층 재해석 — regime map·funnel·Λ\* 분포·축 크로스컷 | 백스테이지(§E 만 본편 통계) | `results/tables/*.csv`, LEDGER |

## 재현

저작은 py:percent 소스(`_src/*.py`) → jupytext 변환 → 실행 커밋 파이프라인이다:

```bash
uv sync --extra dev
for v in 00_log_eda 01_dgp_anatomy 02_estimator_walkthrough 03_diagnostics_gate \
         04_results_deepdive 05_gt_unknown_protocol; do
  uv run jupytext --to notebook --execute --output "notebooks/${v}.ipynb" "notebooks/_src/${v}.py"
done
```

- 00권의 OBD 섹션은 `data/obd/` 미배치 시 안내 후 자동 스킵된다(`data/README.md` 참조).
  05권은 OBD 원자료가 필요 없다(committed 카드 CSV 만 읽음).
- 데이터 보호 규약: OBD **원자료 row 출력 금지** — 노트북에는 집계 통계·figure 만 담는다(재배포 아님).
- `experiments/_style.py` 가 Agg 백엔드를 강제하므로 각 권은 import 직후 `%matplotlib inline`
  매직으로 복원한다(M7 실측 함정 — PLAN §4.8).
