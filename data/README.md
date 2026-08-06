# data/ — 로컬 전용 (커밋 금지)

이 디렉토리의 내용물은 `.gitignore`로 보호된다(이 README만 예외). 원본 데이터는 재배포하지 않는다.

| 데이터 | 배치 경로 | 출처 |
|---|---|---|
| Open Bandit Dataset (small) | `data/obd/` | https://github.com/st-tech/zr-obp (repo 동봉 샘플) · 전체: https://research.zozo.com/data.html |
| classification-to-bandit 원본 (optdigits 등) | `data/openml/` | OpenML/UCI — `src/ope/datasets.py`가 최초 실행 시 다운로드 |

라이선스: OBD는 ZOZO Research 배포 조건을 따른다. 본 repo는 변환 스크립트만 커밋한다.
