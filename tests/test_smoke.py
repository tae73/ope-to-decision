"""패키지 임포트 smoke. 실질 검증은 property test(test_estimators·test_dgp·test_statistical 등)가 담당."""

import ope
from ope import datasets, dgp, diagnostics, estimators, policies  # noqa: F401


def test_import():
    assert ope.__version__
