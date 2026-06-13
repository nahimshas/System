"""
Tests for the decision log — the full candidate + feature archive.

Verifies: both-side capture, the `made` threshold flag, idempotent re-runs,
game-lock freezing, and self-contained feature storage. Uses a temp dir so it
never touches real state.
"""

from datetime import date, datetime, timedelta, timezone

import src.state.decision_log as dl


def _candidates():
    return [
        {"market_type": "Moneyline", "side": "Atlanta Braves",
         "model_prob": 0.62, "market_prob": 0.55, "made": True},
        {"market_type": "Moneyline", "side": "Miami Marlins",
         "model_prob": 0.38, "market_prob": 0.45, "made": False},
        {"market_type": "Total", "side": "over",
         "model_prob": 0.51, "market_prob": 0.50, "made": False, "line": 8.5},
        {"market_type": "Total", "side": "under",
         "model_prob": 0.49, "market_prob": 0.50, "made": False, "line": 8.5},
    ]


def _future_ct():
    return (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat().replace("+00:00", "Z")


def _past_ct():
    return (datetime.now(timezone.utc) - timedelta(hours=6)).isoformat().replace("+00:00", "Z")


def test_records_both_sides_and_made_flag(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DECISION_LOG_DIR", tmp_path)
    d = date(2026, 6, 13)
    n = dl.record_candidates(
        d, "MLB", "Miami Marlins @ Atlanta Braves", _future_ct(),
        "Atlanta Braves", "Miami Marlins", _candidates(),
        features={"home_sp_xfip": 3.4, "away_sp_xfip": 4.9},
    )
    assert n == 4
    shard = dl._load_shard(dl._shard_path(d))
    entries = shard["entries"]
    assert len(entries) == 4
    # rejected side is logged
    away = entries["2026-06-13|MLB|Miami Marlins @ Atlanta Braves|Moneyline|Miami Marlins"]
    assert away["made"] is False
    # made flag + edge derived
    home = entries["2026-06-13|MLB|Miami Marlins @ Atlanta Braves|Moneyline|Atlanta Braves"]
    assert home["made"] is True
    assert abs(home["edge"] - 0.07) < 1e-9
    # features are self-contained on every row
    assert away["features"]["home_sp_xfip"] == 3.4
    assert home["line"] is None
    assert entries["2026-06-13|MLB|Miami Marlins @ Atlanta Braves|Total|over"]["line"] == 8.5


def test_idempotent_rerun_updates_not_duplicates(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DECISION_LOG_DIR", tmp_path)
    d = date(2026, 6, 13)
    ct = _future_ct()
    dl.record_candidates(d, "MLB", "G", ct, "H", "A", _candidates())
    dl.record_candidates(d, "MLB", "G", ct, "H", "A", _candidates())
    shard = dl._load_shard(dl._shard_path(d))
    assert len(shard["entries"]) == 4  # not 8


def test_first_pick_price_preserved_across_reruns(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DECISION_LOG_DIR", tmp_path)
    d = date(2026, 6, 13)
    ct = _future_ct()
    dl.record_candidates(d, "MLB", "G", ct, "H", "A", [
        {"market_type": "Moneyline", "side": "H", "model_prob": 0.6, "market_prob": 0.55, "made": True},
    ])
    # line moved on the re-run — first-pick price must stay, last-update tracks it
    dl.record_candidates(d, "MLB", "G", ct, "H", "A", [
        {"market_type": "Moneyline", "side": "H", "model_prob": 0.6, "market_prob": 0.58, "made": True},
    ])
    e = dl._load_shard(dl._shard_path(d))["entries"]["2026-06-13|MLB|G|Moneyline|H"]
    assert abs(e["market_prob_at_first_pick"] - 0.55) < 1e-9
    assert abs(e["market_prob_at_last_update"] - 0.58) < 1e-9


def test_game_lock_freezes_started_games(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "DECISION_LOG_DIR", tmp_path)
    d = date(2026, 6, 13)
    # first record AFTER the game started → row is written game_locked=True
    dl.record_candidates(d, "MLB", "G", _past_ct(), "H", "A", [
        {"market_type": "Moneyline", "side": "H", "model_prob": 0.6, "market_prob": 0.55, "made": True},
    ])
    # a later run must NOT overwrite a frozen (locked) row
    dl.record_candidates(d, "MLB", "G", _past_ct(), "H", "A", [
        {"market_type": "Moneyline", "side": "H", "model_prob": 0.99, "market_prob": 0.10, "made": True},
    ])
    e = dl._load_shard(dl._shard_path(d))["entries"]["2026-06-13|MLB|G|Moneyline|H"]
    assert e["model_prob"] == 0.6      # frozen at first write
    assert e["game_locked"] is True
