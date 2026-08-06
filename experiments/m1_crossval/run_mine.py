"""M1 교차검증 1단 — main env: src 구현으로 데이터·자기추정 생성 → npz 공유.

npz 공유가 필요한 이유: obp env(py3.9·numpy 1.x)와 main env(py3.11+·numpy 2.x)는 CLAUDE.md §6 에
따라 혼용 금지이며, 난수 스트림 cross-version 의존을 피하려면 **같은 배열**을 파일로 공유해
순수 산술만 비교해야 한다. float64 고정·allow_pickle=False.
"""

import json
from pathlib import Path

import numpy as np

from ope.dgp import DGPConfig, make_synthetic_bandit_data
from ope.estimators import (
    estimate_clipped_ips, estimate_dm, estimate_dr, estimate_dros,
    estimate_ips, estimate_snips, estimate_switch_dr,
)

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results" / "tables" / "m1_crossval_data.npz"
OUT = ROOT / "results" / "tables" / "m1_crossval_mine.json"

CFG = DGPConfig(n=50_000, n_actions=10, dim_context=5, beta_log=1.0, beta_eval=3.0,
                support_deficiency=0.0, reward_noise=0.5, confounding_strength=0.0,
                seed=20260806, struct_seed=7)
LAM_DROS = 5.0


def main() -> None:
    d = make_synthetic_bandit_data(CFG)
    q_hat = 0.5 * d.q_true + 0.25  # 고의 오지정 reward 모델 — 비교 목적은 산술 일치

    raw_w = estimate_ips(d.reward, d.action, d.pscore_logged, d.pi_e_dist).weights
    tau = float(np.quantile(raw_w, 0.95))       # switch 분기가 실제 발동하도록 p95
    lam_clip = float(np.quantile(raw_w, 0.90))  # clipping 도 실제 발동하도록 p90

    args = (d.reward, d.action, d.pscore_logged, d.pi_e_dist)
    mine = {
        "dm": estimate_dm(d.pi_e_dist, q_hat).value,
        "ips": estimate_ips(*args).value,
        "snips": estimate_snips(*args).value,
        "clipped_ips": estimate_clipped_ips(*args, lam=lam_clip).value,
        "dr": estimate_dr(*args, q_hat).value,
        "switch_dr": estimate_switch_dr(*args, q_hat, tau=tau).value,
        "dros": estimate_dros(*args, q_hat, lam=LAM_DROS).value,
    }

    NPZ.parent.mkdir(parents=True, exist_ok=True)
    np.savez(NPZ,
             reward=d.reward.astype(np.float64),
             action=d.action.astype(np.int64),
             pscore=d.pscore_logged.astype(np.float64),
             pi_e_dist=d.pi_e_dist.astype(np.float64),
             q_hat=q_hat.astype(np.float64),
             tau=np.float64(tau), lam_clip=np.float64(lam_clip), lam_dros=np.float64(LAM_DROS))
    OUT.write_text(json.dumps({
        "config": CFG._asdict(),
        "hyperparams": {"tau": tau, "lam_clip": lam_clip, "lam_dros": LAM_DROS,
                        "share_w_gt_tau": float((raw_w > tau).mean()),
                        "share_w_gt_lam_clip": float((raw_w > lam_clip).mean())},
        "mine": mine,
    }, indent=2))
    for k, v in mine.items():
        print(f"{k:12s} {v:.12f}")
    print(f"tau(p95)={tau:.6f}  lam_clip(p90)={lam_clip:.6f}  → {NPZ.name}, {OUT.name}")


if __name__ == "__main__":
    main()
