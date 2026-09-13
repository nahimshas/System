"""Watchlist settlement must follow the registry, not hand-written lists.

CFB produced picks from Sep 3 and displayed them, but the tile sat at 0-0
forever: nothing settled CFB. Two separate hand-maintained sport lists were
responsible, in the same file --
  1. check_and_settle_watchlist() had a ~35-line block per sport, and CFB was
     never added to the pile.
  2. load_watchlist_performance() iterated a literal tuple of sport names.
Both failed silently, because an unsettled sport looks exactly like a sport
that has not played yet.
"""
import re

import pytest


def _src():
    with open("src/data/outcome_checker.py") as f:
        return f.read()


class TestSettlerIsTableDriven:
    def test_table_exists_and_includes_cfb(self):
        src = _src()
        assert "WATCHLIST_SETTLERS" in src
        block = src.split("WATCHLIST_SETTLERS")[1][:900]
        for sport in ("NHL", "WNBA", "CFB", "MLS", "WC", "LIGAMX"):
            assert f'"{sport}"' in block, f"{sport} missing from settler table"

    def test_cfb_reads_its_own_display_key(self):
        block = _src().split("WATCHLIST_SETTLERS")[1][:900]
        assert '"cfb_display"' in block

    def test_soccer_keeps_its_own_grader(self):
        """A soccer draw LOSES a team bet; it must not use the generic grader."""
        block = _src().split("WATCHLIST_SETTLERS")[1][:900]
        for sport in ("MLS", "WC", "LIGAMX"):
            row = next(l for l in block.splitlines() if f'"{sport}"' in l)
            assert "_determine_mls_outcome" in row, f"{sport} lost its soccer grader"
        for sport in ("NHL", "WNBA", "CFB"):
            row = next(l for l in block.splitlines() if f'"{sport}"' in l)
            assert "_determine_mls_outcome" not in row

    def test_no_per_sport_blocks_remain(self):
        """The old shape: `wnba_picks = [...]`, one per sport."""
        src = _src()
        assert not re.search(r"^\s+wnba_picks\s*=", src, re.M)
        assert not re.search(r"^\s+ligamx_picks\s*=", src, re.M)

    def test_cfb_has_an_espn_path(self):
        assert '"CFB": "football/college-football"' in _src()


class TestPerformanceIsDerived:
    def test_not_a_hardcoded_tuple(self):
        src = _src()
        assert 'for sport in ("NHL", "IPL", "WNBA", "MLS", "WC", "LIGAMX", "F5")' not in src

    def test_uses_slug_not_label(self):
        """label is "Liga MX" but history rows key on "LIGAMX" — using label
        creates a second, permanently-empty tile for the same sport."""
        src = _src()
        assert "e.slug.upper()" in src
        block = src.split("def load_watchlist_performance")[1][:1200]
        assert "e.label" not in block

    def test_counts_a_sport_present_only_in_history(self, tmp_path, monkeypatch):
        """F5 has no registry entry — it must still be counted."""
        from src.data import outcome_checker as oc
        rows = [
            {"date": "2026-09-11", "sport": "F5", "game": "A @ B",
             "pick": "A", "result": "WON"},
            {"date": "2026-09-11", "sport": "CFB", "game": "C @ D",
             "pick": "C", "result": "LOST"},
        ]
        monkeypatch.setattr(oc, "_load_watchlist_history", lambda: rows)
        perf = oc.load_watchlist_performance()
        assert perf["F5"]["won"] == 1
        assert perf["CFB"]["lost"] == 1

    def test_registry_sport_with_no_results_still_gets_a_tile(self, monkeypatch):
        from src.data import outcome_checker as oc
        monkeypatch.setattr(oc, "_load_watchlist_history", lambda: [])
        perf = oc.load_watchlist_performance()
        assert "CFB" in perf
        assert perf["CFB"]["total"] == 0
        assert perf["CFB"]["win_rate_pct"] is None

    def test_no_label_shaped_duplicates(self, monkeypatch):
        from src.data import outcome_checker as oc
        monkeypatch.setattr(oc, "_load_watchlist_history", lambda: [])
        perf = oc.load_watchlist_performance()
        assert "Liga MX" not in perf
        assert "World Cup" not in perf


class TestRealSettledData:
    """The 8 picks recovered on Sep 13, graded against real ESPN finals."""

    def test_cfb_history_present_and_balanced(self):
        from src.data.outcome_checker import _load_watchlist_history
        cfb = [r for r in _load_watchlist_history() if r.get("sport") == "CFB"]
        assert len(cfb) >= 8
        assert {r["result"] for r in cfb} <= {"WON", "LOST"}

    def test_known_gradings(self):
        from src.data.outcome_checker import _load_watchlist_history
        by_pick = {r["pick"]: r["result"]
                   for r in _load_watchlist_history() if r.get("sport") == "CFB"}
        # Missouri 38 @ Kansas 21 — lost outright, and by 17 against +5.5.
        assert by_pick.get("Kansas Jayhawks") == "LOST"
        assert by_pick.get("Kansas Jayhawks +5.5") == "LOST"
        # Iowa State 13 @ Iowa 16 — lost by 3, covers +13.5.
        assert by_pick.get("Iowa State Cyclones +13.5") == "WON"
        # NDSU 38 @ Air Force 32 — won by 6, covers -1.5.
        assert by_pick.get("North Dakota State Bison -1.5") == "WON"


class TestSettlementRetryWindow:
    """A pick that missed its one attempt used to sit PENDING forever."""

    def _patch(self, monkeypatch, states, scores):
        import src.state.manager as mgr
        from src.data import outcome_checker as oc
        fetched = []

        monkeypatch.setattr(mgr, "load_state",
                            lambda d: states.get(d.isoformat()))

        def _wl(sport, day):
            fetched.append((sport, day.isoformat()))
            return scores.get((sport, day.isoformat()), {})
        monkeypatch.setattr(oc, "_fetch_watchlist_final_scores", _wl)

        def _mlb(sport, day, *a, **k):
            fetched.append(("MLB_F5", day.isoformat()))
            return {}
        monkeypatch.setattr(oc, "_fetch_espn_final_scores", _mlb)

        saved = {}
        monkeypatch.setattr(oc, "_load_watchlist_history",
                            lambda: saved.get("rows", []))
        monkeypatch.setattr(oc, "_save_watchlist_history",
                            lambda rows: saved.__setitem__("rows", rows))
        monkeypatch.setattr(oc, "_find_game_score",
                            lambda sc, h, a, **k: sc.get((h, a)))
        return fetched, saved

    def _pick(self):
        return {"sport": "CFB", "pick": "Kansas Jayhawks", "bet_type": "Moneyline",
                "game": "Missouri Tigers @ Kansas Jayhawks",
                "home_team": "Kansas Jayhawks", "away_team": "Missouri Tigers"}

    def test_constant_is_a_real_window(self):
        from src.data.outcome_checker import WATCHLIST_SETTLE_LOOKBACK_DAYS as L
        assert L >= 3

    def test_recovers_a_pick_from_several_days_back(self, monkeypatch):
        """The whole point: day-4 pick, scores only available now."""
        from datetime import date
        from src.data import outcome_checker as oc
        states = {"2026-09-09": {"cfb_display": [self._pick()]}}
        scores = {("CFB", "2026-09-09"):
                  {("Kansas Jayhawks", "Missouri Tigers"):
                   {"home_score": 21, "away_score": 38}}}
        _, saved = self._patch(monkeypatch, states, scores)
        n = oc.check_and_settle_watchlist(date(2026, 9, 13))
        assert n == 1
        assert saved["rows"][0]["result"] == "LOST"
        assert saved["rows"][0]["date"] == "2026-09-09"

    def test_does_not_settle_today(self, monkeypatch):
        """Today's games may still be running."""
        from datetime import date
        from src.data import outcome_checker as oc
        states = {"2026-09-13": {"cfb_display": [self._pick()]}}
        scores = {("CFB", "2026-09-13"):
                  {("Kansas Jayhawks", "Missouri Tigers"):
                   {"home_score": 21, "away_score": 38}}}
        _, saved = self._patch(monkeypatch, states, scores)
        assert oc.check_and_settle_watchlist(date(2026, 9, 13)) == 0

    def test_no_fetch_when_nothing_outstanding(self, monkeypatch):
        """A wider window must cost nothing on an ordinary day."""
        from datetime import date
        from src.data import outcome_checker as oc
        states = {"2026-09-12": {"cfb_display": [self._pick()]}}
        fetched, saved = self._patch(monkeypatch, states, {})
        saved["rows"] = [{"date": "2026-09-12", "sport": "CFB",
                          "pick": "Kansas Jayhawks",
                          "game": "Missouri Tigers @ Kansas Jayhawks",
                          "result": "LOST"}]
        assert oc.check_and_settle_watchlist(date(2026, 9, 13)) == 0
        assert not [f for f in fetched if f[0] == "CFB"]

    def test_is_idempotent_across_runs(self, monkeypatch):
        from datetime import date
        from src.data import outcome_checker as oc
        states = {"2026-09-12": {"cfb_display": [self._pick()]}}
        scores = {("CFB", "2026-09-12"):
                  {("Kansas Jayhawks", "Missouri Tigers"):
                   {"home_score": 21, "away_score": 38}}}
        _, saved = self._patch(monkeypatch, states, scores)
        first = oc.check_and_settle_watchlist(date(2026, 9, 13))
        second = oc.check_and_settle_watchlist(date(2026, 9, 13))
        assert (first, second) == (1, 0)
        assert len(saved["rows"]) == 1

    def test_unresolvable_pick_is_retried_not_poisoned(self, monkeypatch):
        """No score today -> stays outstanding -> settles once ESPN catches up."""
        from datetime import date
        from src.data import outcome_checker as oc
        states = {"2026-09-12": {"cfb_display": [self._pick()]}}
        _, saved = self._patch(monkeypatch, states, {})      # no scores at all
        assert oc.check_and_settle_watchlist(date(2026, 9, 13)) == 0
        assert saved.get("rows", []) == []
        # ESPN publishes; same call now settles it.
        monkeypatch.setattr(
            oc, "_fetch_watchlist_final_scores",
            lambda s, d: {("Kansas Jayhawks", "Missouri Tigers"):
                          {"home_score": 21, "away_score": 38}})
        assert oc.check_and_settle_watchlist(date(2026, 9, 13)) == 1

    def test_f5_is_inside_the_window_too(self):
        """F5 used to sit outside the day loop on `yesterday`."""
        with open("src/data/outcome_checker.py") as f:
            src = f.read()
        body = src.split("def check_and_settle_watchlist")[1].split("\ndef ")[0]
        assert "F5 watchlist settled" in body
        # no stale single-day variables left in the function
        assert "yesterday.isoformat()" not in body
