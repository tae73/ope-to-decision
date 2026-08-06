"""축 12 — OBD small 게이트: BTS 실로그로 uniform 정책 평가, random 로그 근사 GT 대비 (PLAN §2 축 12·§3.4).

프로토콜(uniform-target).
    로그 = ZOZO Open Bandit Dataset **small**, "all" 캠페인 BTS(Bernoulli Thompson Sampling)
    (n=10,000 행 · K=80 · position 1–3 풀링 — 행=표본). 평가 정책 = **uniform** π_e(a|x)=1/80
    (position 무관 동일) — `pi_e_dist = np.full((n, 80), 1/80)`. pscore 는 CSV 의
    `propensity_score` **실측 기록값**(추정 아님). 근사 GT = random 로그(=uniform 로깅)의
    on-policy 평균 click + percentile bootstrap CI(자체 구현·행 재표집). 추정측도 같은 단일
    로그에서 `bootstrap_ci` 로 CI 병기. **모든 비교는 구간 언어 — 점 비교 단정 금지**(§3.4).
    estimator 는 ips·snips·clipped_ips 코어 3종 + dr 은 **참고(reference)**: q̂ = per-(item_id,
    position) 2-fold cross-fitted 평균 click 인데 클릭이 희소해(전 로그 수십 건) 대부분 셀이
    0 — q̂ 이 사실상 무정보라 DR 보정이 의미를 갖기 어렵다(참고 표기 이유).

판별력 없음 — 사전 선언 (본 축의 존재 이유는 규약 시연이지 estimator 판별이 아니다).
    로컬 실측(정확값은 companion CSV `12_obd_small_gate_summary.csv` 가 정본): random 클릭
    ~38건·bts ~42건(각 n=10,000), 근사 GT 95% CI 상대 반폭 ~±30%, bts min pscore ~4.5e-05 →
    uniform-target max w = (1/80)/min pscore ~수백 — **클릭 한 건이 IPS 추정을 지배할 수 있다**
    (지배도는 summary CSV `top1_click_share_ips` 행). 따라서 이 축은 "어느 estimator 가 더
    정확한가"를 판별할 수 없고, **§3.4 규약(근사 GT 에 CI 병기·구간 비교) 준수 + decision gate
    가 이런 희소·저 overlap 실로그를 어떻게 판정하는가**를 시연하는 축이다.

역방향 스코프 밖 선언.
    역방향(random 로그로 BTS 정책 평가)은 하지 않는다: BTS 의 counterfactual action 분포
    (`action_dist`) 재구성은 obp 패키지가 배포하는 학습된 beta prior·시뮬레이션 자산에
    의존하므로 자기 구현(순수 numpy) 스코프 밖 — uniform-target 단방향만 수행한다.

bootstrap·표본 한계 (정직 문서화).
    ① position 행 상관: 같은 노출(round)의 position 1–3 행은 원리적으로 상관되나 행 단위
    재표집은 이를 독립으로 취급한다. 단 small 표본 실측으로는 timestamp 중복이 극소
    (random 9쌍·bts 1쌍 — summary CSV `dup_timestamp_pairs_*` 행)라 이 근사의 실해는 미미할
    것으로 보되, round id 컬럼이 없어 cluster bootstrap 으로의 교정은 불가.
    ② small 표본 대표성: 계약 가정은 "시간순 선두 절단"이었으나 **로컬 실측은 이를 반증**한다
    — timestamp 단조 증가·캠페인 7일 전 구간 커버(일별 1.1–1.7k 행)로, 선두 절단이 아니라
    전 기간에 걸친 시간순 부분표본이다. 다만 full 버전(수백만 행 규모 —
    https://research.zozo.com/data.html ) 에서 어떤 규칙으로 뽑힌 부분표본인지는 배포측
    문서로 확정하지 못했으므로, 본 축의 수치를 full OBD 로 일반화하지 말 것.
    ③ DR 의 CI 는 q̂ 고정 조건부(재표집마다 refit 하지 않음 — cross-fit 구조는 원 로그
    기준), clipped 의 λ 도 전체 로그에서 한 번 계산해 고정한다.

데이터·라이선스 각주 (재배포 금지).
    zr-obp 리포지토리의 **코드**는 Apache-2.0 이지만 **Open Bandit Dataset 데이터 자체는
    ZOZO 의 별도 이용조건**을 따른다(CC-BY 아님 — https://github.com/st-tech/zr-obp ·
    https://research.zozo.com/data.html ). `data/` 는 .gitignore 보호 **로컬 전용** 경로이며
    본 스크립트는 원 URL(raw.githubusercontent.com/st-tech/zr-obp)에서의 직접 다운로드만
    수행한다(파일이 있으면 스킵). item_context.csv 는 받지 않는다 — q̂ 이 per-(item,
    position) 평균이라 컨텍스트 피처를 쓰지 않는다.

실측 정직 기록 (실행 후 — 정확값은 committed summary CSV 가 정본).
    ips·snips·dr 의 95% CI 는 GT band 와 겹치나, **clipped_ips 의 95% CI 는 GT band 바로
    아래에 통째로 위치**(겹침 실패): M2 공통 hyperparam 정책 λ=p90(w)(~1.77)를 이 로그에
    이식하면 heavy-tail 상위 weight 가 강하게 절단되는데 클릭이 고-w 행에 몰려 있어 하향
    bias 가 구간 수준에서 가시화된다. 이는 estimator 판별 주장이 아니라(판별력 없음 선언
    유지) **데이터 적응형 clipping 정책을 희소 실로그에 그대로 이식할 때의 bias 위험**을
    보여주는 구간 언어의 관찰 기록이다 — 은폐하지 않고 figure 주석·stdout 에 병기.

산출·스키마 (예외 문서화).
    results/tables/12_obd_small_gate.csv — long, `_common.COLUMNS` 준수. 스키마 예외 2:
    `seed`=0 placeholder(실로그 단일 — 합성 축의 MC seed 반복이 없다; bootstrap seed 는
    summary CSV detail 에 기록), `v_true`=**근사** GT 점추정(random on-policy 평균 —
    자체 표본오차 보유, §3.4; 합성 축의 정확 v_true 와 의미가 다르다).
    results/tables/12_obd_small_gate_summary.csv — companion(자유 스키마: name·role·value·
    ci_lo·ci_hi·detail): estimator CI·근사 GT CI·진단·게이트 verdict·판별력 수치.
    results/figures/12_obd_small_gate.png — 추정치+CI vs GT band(±CI) 겹침 시각화.
"""

import csv
import subprocess

import numpy as np

from _common import ROOT, TAB_DIR, RunRecord, hyperparams_from_weights, write_axis_csv
from _style import ESTIMATOR_COLORS, ESTIMATOR_LABELS, ORACLE_COLOR, apply_style, save_figure
from ope.datasets import load_obd_small
from ope.diagnostics import PROVISIONAL_THRESHOLDS, compute_diagnostics, decision_gate
from ope.estimators import (
    bootstrap_ci, estimate_clipped_ips, estimate_dr, estimate_ips, estimate_snips,
)

AXIS_ID, SLUG = "12", "obd_small_gate"
KNOB = "log_policy"          # 스윕 축이 아니라 단일 프로토콜 — knob_value="bts" 고정
N_ACTIONS = 80
N_POSITIONS = 3
N_BOOT_EST = 2000            # estimator 측 bootstrap_ci 반복수
N_BOOT_GT = 10000            # 근사 GT percentile bootstrap 반복수 (mean 만이라 저렴)
ALPHA = 0.05                 # 유의수준 → 95% CI (bootstrap_ci 시그니처 규약)
SEED_BOOT_GT = 12
SEED_BOOT_EST = 13
SEED_QFOLD = 0
DATA_DIR = ROOT / "data" / "obd"
OBD_RAW = "https://raw.githubusercontent.com/st-tech/zr-obp/master/obd"


def ensure_obd_small() -> None:
    """data/obd/{random,bts}/all/all.csv 없으면 curl 다운로드(있으면 스킵) — 로컬 전용·재배포 금지."""
    for pol in ("random", "bts"):
        dst = DATA_DIR / pol / "all" / "all.csv"
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        url = f"{OBD_RAW}/{pol}/all/all.csv"
        print(f"[12] downloading {url} -> {dst} (local-only, no redistribution)")
        subprocess.run(["curl", "-sSfL", url, "-o", str(dst)], check=True)


def bootstrap_mean_ci(x: np.ndarray, n_boot: int, alpha: float, seed: int) -> tuple[float, float]:
    """percentile bootstrap CI of mean(x) — 근사 GT 용 자체 구현(행 재표집·고정 seed)."""
    rng = np.random.default_rng(seed)
    n = len(x)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = x[rng.integers(0, n, size=n)].mean()
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def make_crossfit_q_hat(item: np.ndarray, pos_code: np.ndarray, click: np.ndarray,
                        n_folds: int = 2, seed: int = SEED_QFOLD) -> np.ndarray:
    """q̂(x_i, a) = per-(item, position) cross-fitted 평균 click — (n, K) 반환.

    fold 별로 complement fold 의 (position, item) 셀 평균을 쓰고, 관측 0 셀은 complement
    전역 평균으로 fallback(클릭 희소로 대부분 셀이 이 fallback 에 떨어진다 — 참고 표기 근거).
    """
    rng = np.random.default_rng(seed)
    n = len(item)
    fold = rng.permutation(n) % n_folds
    q_hat = np.empty((n, N_ACTIONS))
    cell = pos_code * N_ACTIONS + item
    n_cells = N_POSITIONS * N_ACTIONS
    for f in range(n_folds):
        tr = fold != f
        sums = np.bincount(cell[tr], weights=click[tr], minlength=n_cells)
        cnts = np.bincount(cell[tr], minlength=n_cells)
        table = np.where(cnts > 0, sums / np.maximum(cnts, 1), click[tr].mean())
        q_hat[fold == f] = table.reshape(N_POSITIONS, N_ACTIONS)[pos_code[fold == f]]
    return q_hat


def main() -> None:
    ensure_obd_small()
    df_rand = load_obd_small("random", data_dir=str(DATA_DIR))
    df_bts = load_obd_small("bts", data_dir=str(DATA_DIR))

    # ---- 근사 GT: random 로그 on-policy 평균 click + bootstrap CI (§3.4) ----
    r_rand = df_rand["click"].to_numpy(dtype=float)
    gt = float(r_rand.mean())
    gt_lo, gt_hi = bootstrap_mean_ci(r_rand, N_BOOT_GT, ALPHA, SEED_BOOT_GT)
    gt_rel_halfwidth = (gt_hi - gt_lo) / 2.0 / gt

    # ---- BTS 로그 → uniform 타깃 평가 ----
    reward = df_bts["click"].to_numpy(dtype=float)
    action = df_bts["item_id"].to_numpy(dtype=int)
    pscore = df_bts["propensity_score"].to_numpy(dtype=float)
    pos_code = df_bts["position"].to_numpy(dtype=int) - 1  # 1..3 → 0..2
    n = len(reward)
    assert action.min() >= 0 and action.max() < N_ACTIONS
    assert set(np.unique(pos_code)) <= set(range(N_POSITIONS))
    pi_e = np.full((n, N_ACTIONS), 1.0 / N_ACTIONS)

    ips_res = estimate_ips(reward, action, pscore, pi_e)
    w_raw = ips_res.weights
    lam = hyperparams_from_weights(w_raw)["lam_clip"]  # p90(w) — M2 공통 hyperparam 정책
    q_hat = make_crossfit_q_hat(action, pos_code, reward)

    results = {
        "ips": (ips_res, np.nan, None),
        "snips": (estimate_snips(reward, action, pscore, pi_e), np.nan, None),
        "clipped_ips": (estimate_clipped_ips(reward, action, pscore, pi_e, lam), lam, lam),
        "dr": (estimate_dr(reward, action, pscore, pi_e, q_hat), np.nan, None),
    }
    cis = {}
    for name, (_res, _hyper, boot_hyper) in results.items():
        q_arg = q_hat if name == "dr" else None
        cis[name] = bootstrap_ci(reward, action, pscore, pi_e, q_arg, name,
                                 N_BOOT_EST, ALPHA, SEED_BOOT_EST, hyperparam=boot_hyper)

    # ---- 진단 + decision gate ----
    diag = compute_diagnostics(pscore, action, pi_e)
    verdict = decision_gate(diag, PROVISIONAL_THRESHOLDS)
    contrib = w_raw * reward
    top1_share = float(contrib.max() / contrib.sum())  # 클릭 한 건의 IPS 지배도

    # ---- long CSV (_common.COLUMNS — seed placeholder·근사 v_true 예외는 모듈 docstring) ----
    records = []
    for name, (res, hyper, _bh) in results.items():
        s = float(res.weights.sum() ** 2)
        q = float((res.weights**2).sum())
        records.append(RunRecord(
            axis_id=AXIS_ID, knob_name=KNOB, knob_value="bts", seed=0,
            estimator=name, variant="reference" if name == "dr" else "",
            hyperparam_value=hyper, estimate=res.value, v_true=gt,
            ess_ratio_raw=diag.ess_ratio, max_weight_raw=diag.max_weight,
            support_proxy=diag.support_deficiency,
            ess_ratio_used=(s / q / n) if q > 0 else 0.0))
    csv_path = write_axis_csv(AXIS_ID, SLUG, records)

    # ---- companion summary CSV (자유 스키마 — docstring 문서화) ----
    boot_note = f"percentile bootstrap B={N_BOOT_EST}, alpha={ALPHA}, seed={SEED_BOOT_EST}"
    rows = [
        ("on_policy_gt", "ground_truth", gt, gt_lo, gt_hi,
         f"random log on-policy mean click (approximate GT, own sampling error — PLAN §3.4); "
         f"percentile bootstrap B={N_BOOT_GT}, alpha={ALPHA}, seed={SEED_BOOT_GT}"),
        ("ips", "estimator", results["ips"][0].value, *cis["ips"],
         f"uniform target 1/80 on bts log; {boot_note}"),
        ("snips", "estimator", results["snips"][0].value, *cis["snips"], boot_note),
        ("clipped_ips", "estimator", results["clipped_ips"][0].value, *cis["clipped_ips"],
         f"lam=p90(w)={lam:.6g} fixed across resamples; {boot_note}"),
        ("dr", "estimator_reference", results["dr"][0].value, *cis["dr"],
         "reference only — q_hat = per-(item,position) 2-fold cross-fitted mean click; "
         f"clicks too sparse for informative q_hat; CI conditional on fixed q_hat; {boot_note}"),
        ("ess_ratio_raw", "diagnostic", diag.ess_ratio, "", "", "uniform target on bts log"),
        ("max_weight_raw", "diagnostic", diag.max_weight, "", "", "= (1/80)/min pscore"),
        ("weight_tail_p99", "diagnostic", diag.weight_tail_p99, "", "", ""),
        ("support_proxy", "diagnostic", diag.support_deficiency, "", "",
         "pi_e mass on actions never logged (all 80 items appear -> 0)"),
        ("min_pscore_bts", "diagnostic", float(pscore.min()), "", "", ""),
        ("gate_verdict", "gate", verdict.decision, "", "", "; ".join(verdict.reasons)),
        ("n_rounds_bts", "data", n, "", "", ""),
        ("n_clicks_bts", "data", int(reward.sum()), "", "", ""),
        ("n_rounds_random", "data", len(r_rand), "", "", ""),
        ("n_clicks_random", "data", int(r_rand.sum()), "", "", ""),
        ("dup_timestamp_pairs_random", "data", int(df_rand["timestamp"].duplicated().sum()),
         "", "", "row-bootstrap independence check (docstring 한계 ①)"),
        ("dup_timestamp_pairs_bts", "data", int(df_bts["timestamp"].duplicated().sum()),
         "", "", "row-bootstrap independence check (docstring 한계 ①)"),
        ("gt_rel_ci_halfwidth", "derived", gt_rel_halfwidth, "", "",
         "(hi-lo)/2/gt — 판별력 없음 사전 선언의 근거 수치"),
        ("top1_click_share_ips", "derived", top1_share, "", "",
         "max(w_i r_i)/sum(w_i r_i) — 클릭 한 건의 IPS 지배도"),
    ]
    summary_path = TAB_DIR / f"{AXIS_ID}_{SLUG}_summary.csv"
    with summary_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "role", "value", "ci_lo", "ci_hi", "detail"])
        writer.writerows(rows)

    # ---- figure: 추정치+CI vs GT band — 겹침/구간 시각 언어 (점 비교 단정 금지) ----
    apply_style()
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    ax.axhspan(gt_lo, gt_hi, color=ORACLE_COLOR, alpha=0.13, zorder=1,
               label="approx. GT band (random log, 95% bootstrap CI)")
    ax.axhline(gt, color=ORACLE_COLOR, linestyle="--", linewidth=1.4, zorder=2)
    order = ["ips", "snips", "clipped_ips", "dr"]
    for i, name in enumerate(order):
        val = results[name][0].value
        lo, hi = cis[name]
        ax.errorbar(i, val, yerr=[[val - lo], [hi - val]], fmt="o", capsize=5,
                    color=ESTIMATOR_COLORS[name], markersize=6, linewidth=2.0, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([ESTIMATOR_LABELS[o] + (" (reference)" if o == "dr" else "")
                        for o in order])
    ax.set_xlim(-0.6, len(order) - 0.4)
    ax.set_ylim(bottom=0)
    ax.set_ylabel("Policy value V(uniform) — per-row click rate")
    ax.set_title("Axis 12 — OBD small gate: uniform target on BTS log vs on-policy GT\n"
                 "intervals only — overlap = compatibility, NOT accuracy evidence "
                 "(pre-declared: no discriminative power)")
    ax.legend(loc="upper left")
    note = (f"gate: {verdict.decision.upper()} (provisional thresholds — this repo's proposal): "
            f"ESS/n={diag.ess_ratio:.3f} (<0.10 soft), "
            f"max w={diag.max_weight:.0f} (>100 cap)\n"
            f"{int(r_rand.sum())}/{int(reward.sum())} clicks (random/bts, n=10k each) · "
            f"GT 95% CI rel. halfwidth ±{gt_rel_halfwidth:.0%} · "
            f"top-1 clicked row = {top1_share:.0%} of IPS estimate\n"
            f"honest note: Clipped-IPS interval sits below the band — clipping bias visible "
            f"(lam=p90(w)={lam:.2f} on a heavy-tail log)\n"
            "this axis demonstrates the §3.4 protocol + decision gate on a sparse real log")
    # 주석 박스는 축 밖(하단)에 — 축 안에 두면 CI 하단 cap 을 가린다(육안 검수로 교정)
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=7.5, color="#55544f",
             bbox={"boxstyle": "round,pad=0.4", "facecolor": "#f4f3ee", "edgecolor": "#d8d7cf"})
    fig_path = save_figure(fig, AXIS_ID, SLUG)

    # ---- 정직 패턴 체크 (stdout — 검증 워크플로가 CSV 로 재도출) ----
    overlaps = {name: (cis[name][1] >= gt_lo and cis[name][0] <= gt_hi) for name in order}
    print(f"[12] gt={gt:.6f} CI=({gt_lo:.6f},{gt_hi:.6f}) rel_halfwidth={gt_rel_halfwidth:.3f}"
          f"  -> {csv_path.name}, {summary_path.name}, {fig_path.name}")
    for name in order:
        lo, hi = cis[name]
        print(f"[12]   {name:<12} est={results[name][0].value:.6f} CI=({lo:.6f},{hi:.6f})"
              f" overlap_gt_band={overlaps[name]}")
    print(f"[12] PATTERN gate_verdict={verdict.decision} reasons={list(verdict.reasons)}")
    print(f"[12] PATTERN ess_ratio={diag.ess_ratio:.4f} max_w={diag.max_weight:.1f} "
          f"support_proxy={diag.support_deficiency:.4f} min_pscore={pscore.min():.2e}")
    print(f"[12] PATTERN all_ci_overlap_gt_band={all(overlaps.values())} "
          f"top1_click_share_ips={top1_share:.3f} (no-discriminative-power declaration)")


if __name__ == "__main__":
    main()
