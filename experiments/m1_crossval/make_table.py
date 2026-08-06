"""M1 교차검증 3단 — 비교표 생성 (main env).

게이트: 점추정 산술 일치(rel_diff ≤ 1e-8) — CI 는 라이브러리별 bootstrap 구현이 달라 비교 제외.
산출: results/tables/m1_obp_crossval.csv (+ stdout verdict).
"""

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TAB = ROOT / "results" / "tables"
OUT = TAB / "m1_obp_crossval.csv"
REL_TOL = 1e-8
ESTIMATORS = ["dm", "ips", "snips", "clipped_ips", "dr", "switch_dr", "dros"]


def main() -> None:
    mine = json.loads((TAB / "m1_crossval_mine.json").read_text())["mine"]
    tracks = {}
    for suffix, label in [("", "obp_py39"), ("_sbobp", "sbobp_py312")]:
        p = TAB / f"m1_crossval_obp{suffix}.json"
        tracks[label] = json.loads(p.read_text()) if p.exists() else None

    rows = []
    all_match = True
    for name in ESTIMATORS:
        row = {"estimator": name, "mine": f"{mine[name]:.15g}"}
        for label, blob in tracks.items():
            if blob is None:
                row[label], row[f"rel_diff_{label}"], row[f"match_{label}"] = "MISSING", "", False
                all_match = False
                continue
            v = blob["obp"].get(name)
            if isinstance(v, float):
                rel = abs(mine[name] - v) / max(abs(mine[name]), 1e-12)
                ok = rel <= REL_TOL
                row[label], row[f"rel_diff_{label}"], row[f"match_{label}"] = f"{v:.15g}", f"{rel:.3e}", ok
                all_match = all_match and ok
            else:
                row[label], row[f"rel_diff_{label}"], row[f"match_{label}"] = str(v), "", False
                all_match = False
        rows.append(row)

    verdict = "GO" if all_match else "NO-GO"
    meta_rows = [{"estimator": f"# track {label}: obp_version="
                               f"{blob['env']['obp_version'] if blob else 'MISSING'}, "
                               f"python={blob['env']['python'] if blob else '-'}, "
                               f"numpy={blob['env']['numpy'] if blob else '-'}"}
                 for label, blob in tracks.items()]
    fieldnames = list(rows[0].keys())
    with OUT.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for m in meta_rows:
            f.write(m["estimator"] + "\n")
        writer.writerows(rows)
        f.write(f"# rel_tol={REL_TOL}, gate=point-estimates-only, verdict={verdict}\n")

    for row in rows:
        print(f"{row['estimator']:12s} mine={row['mine']:<22s} "
              f"obp={row.get('obp_py39', '-'):<22s} match={row.get('match_obp_py39')} "
              f"sbobp={row.get('sbobp_py312', '-'):<22s} match={row.get('match_sbobp_py312')}")
    print(f"VERDICT: {verdict} → {OUT.name}")


if __name__ == "__main__":
    main()
