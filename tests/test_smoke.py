"""M0 smoke — 패키지 임포트와 스텁 계약(NotImplementedError)만 확인. 본 테스트는 M1에서 property test 로 대체·확장."""

import numpy as np
import pytest

import ope
from ope import datasets, dgp, diagnostics, estimators, policies


def test_import():
    assert ope.__version__


def test_stubs_raise_not_implemented():
    with pytest.raises(NotImplementedError):
        policies.softmax_policy(np.zeros((2, 3)), beta=1.0)
    with pytest.raises(NotImplementedError):
        estimators.estimate_ips(np.zeros(2), np.zeros(2, dtype=int), np.ones(2), np.ones((2, 3)) / 3)
