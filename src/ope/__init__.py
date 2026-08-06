"""ope — multi-action logged bandit OPE estimators, diagnostics, and decision gate.

설계 규약(CLAUDE.md 참조): Config NamedTuple → 순수함수 → Result NamedTuple. 클래스 계층/OOP 금지.
모든 수치 산출물은 experiments/NN_*.py 가 results/ 에 기록하고, 문서는 docs/LEDGER.md 정본만 인용한다.
"""

__version__ = "0.1.0"
