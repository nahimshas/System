"""
Guards tools/analysis/backtest.py's LIVE dict against drifting from the shipped
model constants. Every checkpoint evaluation uses LIVE as the baseline — if a
future model change updates edge_finder/config but forgets the LIVE dict, all
health-routine verdicts silently run against a stale model. This test makes
that drift a loud test failure instead.

Importable constants are compared directly; function-local constants
(_INJ_RUNS_PER_PCT, _OFF_W, _PITCH_COEFF) are asserted via source-text match.
"""
import re
from pathlib import Path

from tools.analysis.backtest import LIVE
from src.models.edge_finder import (
    MLB_SPREAD_STD, MLB_CRED_CAP, MLB_RUN_DIFF_CAP, MLB_RUNLINE_SIGMA,
)
from src.config import MLB_HOME_ADVANTAGE, BUDGET_MIN_EDGE

_EDGE_FINDER_SRC = Path("src/models/edge_finder.py").read_text()


def _source_const(name: str) -> float:
    m = re.search(rf"^\s*{name}\s*=\s*([\d.]+)", _EDGE_FINDER_SRC, re.MULTILINE)
    assert m, f"{name} not found in edge_finder.py — update this test if it was renamed"
    return float(m.group(1))


class TestBacktestLiveSync:
    def test_importable_constants_match(self):
        assert LIVE["STD"] == MLB_SPREAD_STD
        assert LIVE["CRED"] == MLB_CRED_CAP
        assert LIVE["RDCAP"] == MLB_RUN_DIFF_CAP
        assert LIVE["RLSIG"] == MLB_RUNLINE_SIGMA
        assert LIVE["HA"] == MLB_HOME_ADVANTAGE
        assert LIVE["MINE"] == BUDGET_MIN_EDGE

    def test_function_local_constants_match(self):
        assert LIVE["INJ"] == _source_const("_INJ_RUNS_PER_PCT")
        assert LIVE["OFF"] == _source_const("_OFF_W")
        assert LIVE["PITCH"] == _source_const("_PITCH_COEFF")
