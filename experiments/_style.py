"""M2 figure 공용 스타일 — dataviz 규율의 고정 파라미터.

- estimator→색은 **entity 고정**(전 figure 동일 매핑, 순서 재배정 금지). 팔레트는 dataviz 기본
  categorical(사전 검증 순서의 prefix 7색) — validator 실행 결과 ALL CHECKS PASS(light, 2026-08-06).
  contrast WARN 3색(#1baf7a·#eda100·#e87ba4)은 "직접 라벨 또는 table view 필수" 조건부 —
  본 레포는 강조 계열 직접 라벨 + figure↔CSV 1:1 페어(table view)로 충족한다.
- 7계열 과밀 축은 강조 3–4(entity 색+직접 라벨) + 나머지 context(회색, 끝단 소형 라벨) 패턴.
- 단일 y축(이중축 금지) · 범례 항상 · 그리드는 recessive · figure 라벨은 영어.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "results" / "figures"

ESTIMATOR_COLORS = {
    "dm": "#2a78d6",           # blue
    "ips": "#eb6834",          # orange
    "snips": "#1baf7a",        # aqua
    "clipped_ips": "#eda100",  # yellow
    "dr": "#e87ba4",           # magenta
    "switch_dr": "#008300",    # green
    "dros": "#4a3aa7",         # violet
}
ESTIMATOR_LABELS = {
    "dm": "DM", "ips": "IPS", "snips": "SNIPS", "clipped_ips": "Clipped-IPS",
    "dr": "DR", "switch_dr": "Switch-DR", "dros": "DRos",
}
ESTIMATOR_ORDER = list(ESTIMATOR_COLORS)  # 범례·직접라벨 순서 고정
CONTEXT_GRAY = "#b3b2a9"
SURFACE = "#fcfcfb"
ORACLE_COLOR = "#555550"  # oracle 참조선(estimator 아님 — 점선 전용)


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#e6e5df", "grid.linewidth": 0.6,
        "legend.frameon": False, "legend.fontsize": 8.5,
        "lines.linewidth": 2.0, "lines.markersize": 5,
    })


def series_kwargs(estimator: str, emphasized: bool) -> dict:
    """entity 색 유지 원칙: 강조=entity 색·2px, context=회색·1.2px (식별은 끝단 직접 라벨이 담당)."""
    if emphasized:
        return {"color": ESTIMATOR_COLORS[estimator], "linewidth": 2.0, "zorder": 3,
                "label": ESTIMATOR_LABELS[estimator]}
    return {"color": CONTEXT_GRAY, "linewidth": 1.2, "zorder": 2,
            "label": ESTIMATOR_LABELS[estimator]}


def end_label(ax, x, y, estimator: str, emphasized: bool) -> None:
    """끝단 직접 라벨 — context 계열의 identity 를 색 없이 보장."""
    color = ESTIMATOR_COLORS[estimator] if emphasized else "#6f6e66"
    ax.annotate(ESTIMATOR_LABELS[estimator], xy=(x, y), xytext=(4, 0),
                textcoords="offset points", va="center", fontsize=7.5, color=color)


def save_figure(fig, axis_id: str, slug: str) -> Path:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = FIG_DIR / f"{axis_id}_{slug}.png"
    fig.savefig(path)
    plt.close(fig)
    return path
