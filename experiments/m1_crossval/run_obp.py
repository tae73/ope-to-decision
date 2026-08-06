"""M1 교차검증 2단 — obp env(py3.9) 또는 sb-obp env(py3.12) 안에서 실행.

npz(1단 산출)를 로드해 동일 배열 위에서 obp estimator 를 돌린다. src 미의존(py3.9 호환 문법만).
사용법: <env-python> run_obp.py [suffix]   (sb-obp 트랙은 suffix="_sbobp")
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
NPZ = ROOT / "results" / "tables" / "m1_crossval_data.npz"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
OUT = ROOT / "results" / "tables" / "m1_crossval_obp{}.json".format(SUFFIX)


def main():
    import obp
    from obp.ope import (
        DirectMethod,
        DoublyRobust,
        DoublyRobustWithShrinkage,
        InverseProbabilityWeighting,
        SelfNormalizedInverseProbabilityWeighting,
        SwitchDoublyRobust,
    )

    z = np.load(str(NPZ), allow_pickle=False)
    reward, action, pscore = z["reward"], z["action"], z["pscore"]
    pi_e_dist, q_hat = z["pi_e_dist"], z["q_hat"]
    tau, lam_clip, lam_dros = float(z["tau"]), float(z["lam_clip"]), float(z["lam_dros"])

    n = len(action)
    position = np.zeros(n, dtype=int)
    action_dist = pi_e_dist[:, :, np.newaxis]
    erm = q_hat[:, :, np.newaxis]
    common = dict(reward=reward, action=action, pscore=pscore,
                  action_dist=action_dist, position=position)

    def build(name):
        if name == "dm":
            return DirectMethod(), dict(action_dist=action_dist, position=position,
                                        estimated_rewards_by_reg_model=erm)
        if name == "ips":
            return InverseProbabilityWeighting(), dict(common)
        if name == "snips":
            return SelfNormalizedInverseProbabilityWeighting(), dict(common)
        if name == "clipped_ips":
            return InverseProbabilityWeighting(lambda_=lam_clip), dict(common)
        if name == "dr":
            return DoublyRobust(), dict(common, estimated_rewards_by_reg_model=erm)
        if name == "switch_dr":
            try:
                est = SwitchDoublyRobust(lambda_=tau)  # obp 0.5.x 표기
            except TypeError:
                est = SwitchDoublyRobust(tau=tau)
            return est, dict(common, estimated_rewards_by_reg_model=erm)
        if name == "dros":
            return DoublyRobustWithShrinkage(lambda_=lam_dros), dict(common,
                                                                     estimated_rewards_by_reg_model=erm)
        raise ValueError(name)

    out = {}
    for name in ["dm", "ips", "snips", "clipped_ips", "dr", "switch_dr", "dros"]:
        try:
            estimator, kwargs = build(name)
            out[name] = float(estimator.estimate_policy_value(**kwargs))
        except Exception as exc:  # noqa: BLE001 — API 차이를 그대로 기록
            out[name] = "ERROR: {}: {}".format(type(exc).__name__, exc)

    OUT.write_text(json.dumps({
        "env": {"python": sys.version.split()[0], "numpy": np.__version__,
                "obp_version": obp.__version__, "track": "sb-obp" if SUFFIX else "obp"},
        "obp": out,
    }, indent=2))
    for k, v in out.items():
        print("{:12s} {}".format(k, v))
    print("→ {}".format(OUT.name))


if __name__ == "__main__":
    main()
