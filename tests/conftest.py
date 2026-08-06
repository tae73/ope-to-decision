import pytest

from ope.dgp import DGPConfig, make_synthetic_bandit_data

BASE = DGPConfig(n=5_000, n_actions=5, dim_context=4, beta_log=1.0, beta_eval=3.0,
                 support_deficiency=0.0, reward_noise=0.3, confounding_strength=0.0,
                 seed=101, struct_seed=7)


@pytest.fixture(scope="session")
def data():
    return make_synthetic_bandit_data(BASE)
