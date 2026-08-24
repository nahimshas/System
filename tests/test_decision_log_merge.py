"""
_stamp_decision must MERGE, not replace.

Aug 10-24 2026: MLB full-game candidates vanished from the decision log while
F5 rows kept landing. Cause: analyze_mlb_game and analyze_mlb_f5_game run over
the SAME game dict, and _stamp_decision assigned game["_decision"] outright, so
the F5 pass erased the full-game candidates before main.py recorded them. Two
weeks of MLB full-game analysis substrate was lost (no backfill is possible).
"""
from src.models.edge_finder import _stamp_decision


def _markets(prefix):
    # (market_type, side, model_prob_POST, model_prob_RAW, market_prob, line)
    return [
        (f"{prefix}Moneyline", "Home", 0.55, 0.56, 0.52, None),
        (f"{prefix}Moneyline", "Away", 0.45, 0.44, 0.48, None),
    ]


def test_second_analyzer_does_not_erase_the_first():
    game = {}
    _stamp_decision(game, 0.05, {"home_sp_score": 1.1}, _markets(""))
    _stamp_decision(game, 0.05, {"f5_pitch": 0.44}, _markets("F5 "))
    kinds = {c["market_type"] for c in game["_decision"]["candidates"]}
    assert "Moneyline" in kinds, "full-game candidates were erased by the F5 pass"
    assert "F5 Moneyline" in kinds
    assert len(game["_decision"]["candidates"]) == 4


def test_features_from_both_passes_are_kept():
    game = {}
    _stamp_decision(game, 0.05, {"home_sp_score": 1.1}, _markets(""))
    _stamp_decision(game, 0.05, {"f5_pitch": 0.44}, _markets("F5 "))
    feats = game["_decision"]["features"]
    assert feats.get("home_sp_score") == 1.1 and feats.get("f5_pitch") == 0.44


def test_rerunning_the_same_analyzer_updates_in_place():
    """Re-runs must not duplicate rows — the key is (market_type, side)."""
    game = {}
    _stamp_decision(game, 0.05, {}, _markets(""))
    _stamp_decision(game, 0.05, {}, [
        ("Moneyline", "Home", 0.61, 0.62, 0.52, None),
        ("Moneyline", "Away", 0.39, 0.38, 0.48, None),
    ])
    cands = game["_decision"]["candidates"]
    assert len(cands) == 2, "re-run duplicated candidates"
    home = next(c for c in cands if c["side"] == "Home")
    assert home["model_prob"] == 0.61, "re-run did not update the existing row"


def test_first_stamp_on_a_clean_game_still_works():
    game = {}
    _stamp_decision(game, 0.05, {"x": 1}, _markets(""))
    assert len(game["_decision"]["candidates"]) == 2
