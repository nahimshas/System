"""Every per-player StatsAPI call must ask for the whole roster.

The API default is playerPool=QUALIFIED (batting-title eligibility, ~3.1 PA per
team game). Measured across all 30 teams on 2026-10-29 that is 130 of 830
batters — 16%. get_bullpen_stats carried the flag from the day it was written;
the three hitting calls did not, which is why MLB props produced zero picks and
why the injury value-multiplier feature was close to inert.

Source-level, so it runs offline and pins the contract rather than the weather.
"""

import ast
import inspect
import unittest

import src.data.mlb_stats as mlb_stats


def _stats_calls():
    """Every `_get("/stats", {...})` literal in the module, as {param: value}."""
    tree = ast.parse(inspect.getsource(mlb_stats))
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Name) and fn.id == "_get"):
            continue
        if not node.args or not isinstance(node.args[0], ast.Constant):
            continue
        if node.args[0].value != "/stats":
            continue
        if len(node.args) < 2 or not isinstance(node.args[1], ast.Dict):
            continue
        params = {}
        for k, v in zip(node.args[1].keys, node.args[1].values):
            if isinstance(k, ast.Constant):
                params[k.value] = v.value if isinstance(v, ast.Constant) else "<expr>"
        out.append(params)
    return out


class TestPlayerPool(unittest.TestCase):
    def test_there_are_per_player_stats_calls_to_check(self):
        # Guards the test itself: if the call shape is refactored, this fails
        # loudly instead of passing vacuously over an empty list.
        self.assertGreaterEqual(len(_stats_calls()), 4)

    def test_every_stats_call_requests_the_full_player_pool(self):
        for params in _stats_calls():
            self.assertEqual(
                params.get("playerPool"), "ALL",
                f"/stats call missing playerPool=ALL — it will silently see only "
                f"qualified players: {params}")

    def test_no_stats_call_keeps_the_old_limit_of_50(self):
        # A full season reaches 37 batters for the deepest roster, so 50 was not
        # truncating yet — but it had no headroom, and a truncated roster fails
        # the same silent way.
        for params in _stats_calls():
            limit = params.get("limit")
            if isinstance(limit, int):
                self.assertGreaterEqual(limit, 100, f"limit too tight: {params}")


class TestHittingCallersExist(unittest.TestCase):
    """The three functions the flag was missing from."""

    def test_functions_are_present(self):
        for name in ("get_batter_props_stats", "get_injury_value_mults",
                     "get_team_batting_leaders", "get_bullpen_stats"):
            self.assertTrue(callable(getattr(mlb_stats, name, None)), name)

    def test_empty_inputs_short_circuit_without_a_request(self):
        self.assertEqual(mlb_stats.get_batter_props_stats(0, []), {})
        self.assertEqual(mlb_stats.get_batter_props_stats(147, []), {})
        self.assertEqual(mlb_stats.get_injury_value_mults(0, []), {})
        self.assertEqual(mlb_stats.get_injury_value_mults(147, []), {})


if __name__ == "__main__":
    unittest.main()
