"""
The decision log must carry BOTH SIDES OF EVERY MARKET WE BET.

It is a pure analysis archive with no backfill: a market only has history from
the day it is wired in. Two failures have already cost us data —

  1. Aug 10-24 2026: the F5 analyzer overwrote the full-game candidates
     (fixed by merging in _stamp_decision).
  2. F5 Spread and F5 Total were never added to the F5 stamp at all, so the
     two most-produced F5 markets logged ZERO candidate rows from the day the
     suite shipped.

This test enumerates the markets each analyzer PRODUCES and asserts the stamp
covers them, so a new market cannot ship half-wired again.
"""
import inspect
import re

import src.models.edge_finder as ef


def _f5_source():
    return inspect.getsource(ef.analyze_mlb_f5_game)


def test_f5_stamp_covers_every_market_the_analyzer_produces():
    src = _f5_source()
    produced = set(re.findall(r'_rec\([^,]+,\s*"(F5 [^"]+)"', src))
    stamped = set(re.findall(r'_f5_cands\.append\(\("(F5 [^"]+)"', src))
    assert produced, "no F5 markets found — did _rec get renamed?"
    missing = produced - stamped
    assert not missing, f"F5 markets produced but never logged: {sorted(missing)}"


def test_f5_spread_and_total_are_both_stamped_two_sided():
    src = _f5_source()
    for market in ("F5 Spread", "F5 Total"):
        appends = re.findall(rf'_f5_cands\.append\(\("{market}",\s*([^,]+),', src)
        assert len(appends) == 2, f"{market} needs both sides stamped, found {appends}"


def test_f5_moneyline_stamps_post_cap_and_raw_separately():
    """model_prob is post-cap and model_prob_raw is pre-cap. Passing the raw
    value in both slots makes the credibility cap's effect unmeasurable."""
    src = _f5_source()
    m = re.search(r'_f5_cands\.append\(\("F5 Moneyline",\s*home,\s*(\w+),\s*(\w+),', src)
    assert m, "F5 Moneyline home stamp not found"
    post, raw = m.groups()
    assert post != raw, f"post-cap and raw are the same variable ({post})"


def test_stamp_decision_merges_across_analyzers():
    """Regression guard for the Aug 10 data loss — full-game then F5 on one
    game dict must not erase each other."""
    game = {}
    ef._stamp_decision(game, 0.05, {}, [("Moneyline", "Home", 0.55, 0.55, 0.52, None)])
    ef._stamp_decision(game, 0.05, {}, [("F5 Total", "over", 0.51, 0.51, 0.50, 3.5)])
    kinds = {c["market_type"] for c in game["_decision"]["candidates"]}
    assert kinds == {"Moneyline", "F5 Total"}


def test_side_conventions_match_the_full_game_analyzer():
    """Spread sides are bare team names, Total sides are over/under. A
    mismatch would fragment analysis across two naming schemes."""
    src = _f5_source()
    assert re.search(r'_f5_cands\.append\(\("F5 Spread",\s*home,', src)
    assert re.search(r'_f5_cands\.append\(\("F5 Total",\s*"over",', src)
    assert re.search(r'_f5_cands\.append\(\("F5 Total",\s*"under",', src)
