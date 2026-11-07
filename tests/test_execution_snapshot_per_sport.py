"""The execution snapshot must read EACH SPORT'S Kalshi books (Nov 7 2026).

`snapshot()` built its series list from the bare MLB `SERIES` map — the exact
mistake the comment inside `resolve_pick` warns about ("Other sports MUST pass
their own, or every lookup lands in the MLB books and silently resolves
nothing"). The warning was two functions above the call that ignored it.

⚠️ WHY IT SURVIVED SO LONG: the feature looked healthy for as long as MLB was
the budget sport. All 201 rows ever written are MLB. It did not go quiet when
it broke; it went quiet when the MLB season ended, nine days later — so the
health alert read as a seasonal lull, which is what a stalled log looks like.

Third instance of the shape "a fix applied to N-1 of N call sites": the
`series_map` parameter was threaded through the CLV passes and not through
this one. The last test below is the shape guard, not an instance guard.
"""
import inspect

import pytest

from src.data import kalshi
from src.data.kalshi_clv import SPORT_SERIES


def _pick(sport, bet_type="Moneyline", pick="Some Team", game="A @ B"):
    return {"sport": sport, "bet_type": bet_type, "pick": pick, "game": game}


@pytest.fixture
def spy(monkeypatch):
    """Record which series were fetched and which map reached resolve_pick."""
    seen = {"fetched": [], "maps": []}

    def fake_fetch(series):
        seen["fetched"].append(series)
        return [{"ticker": series + "-X"}]

    def fake_resolve(pick, markets, game_date, require_quote=True, series_map=None):
        seen["maps"].append((pick.get("sport"), series_map))
        return {"ticker": "T", "series": "S", "side": "yes", "bid": 0.4, "ask": 0.42}

    monkeypatch.setattr(kalshi, "enabled", lambda: True)
    monkeypatch.setattr(kalshi, "fetch_open_markets", fake_fetch)
    monkeypatch.setattr(kalshi, "resolve_pick", fake_resolve)
    return seen


class TestPerSportRouting:
    def test_an_nba_pick_reads_the_nba_books(self, spy):
        kalshi.snapshot([_pick("NBA", "Spread")], "2026-11-07")
        assert spy["fetched"] == ["KXNBASPREAD"]

    def test_an_nba_pick_is_never_resolved_against_mlb_series(self, spy):
        """The bug: the MLB map reached every sport."""
        kalshi.snapshot([_pick("NBA", "Spread")], "2026-11-07")
        (sport, smap), = spy["maps"]
        assert smap is SPORT_SERIES["NBA"]
        assert smap is not kalshi.SERIES

    def test_each_sport_gets_its_own_map_in_one_mixed_slate(self, spy):
        picks = [_pick("NBA", "Spread"), _pick("NHL"), _pick("MLB"), _pick("MLS", "Draw")]
        kalshi.snapshot(picks, "2026-11-07")
        got = {sp: m for sp, m in spy["maps"]}
        for sp in ("NBA", "NHL", "MLB", "MLS"):
            assert got[sp] is SPORT_SERIES[sp], sp

    def test_every_needed_series_is_fetched_once(self, spy):
        picks = [_pick("NHL", "Spread"), _pick("NHL", "Spread", pick="Other"),
                 _pick("NBA", "Total")]
        kalshi.snapshot(picks, "2026-11-07")
        assert sorted(spy["fetched"]) == ["KXNBATOTAL", "KXNHLSPREAD"]

    def test_a_sport_with_no_map_is_skipped_not_misrouted(self, spy):
        """IPL has no per-match Kalshi book. Skipping it is correct; sending it
        to the MLB books and logging a 'match' would not be."""
        kalshi.snapshot([_pick("IPL"), _pick("NBA", "Spread")], "2026-11-07")
        assert [sp for sp, _ in spy["maps"]] == ["NBA"]

    def test_a_pick_with_no_sport_still_falls_back_to_mlb(self, spy):
        """Older rows predate the sport field; they were all MLB."""
        p = _pick("MLB"); p.pop("sport")
        kalshi.snapshot([p], "2026-11-07")
        (sport, smap), = spy["maps"]
        assert smap is SPORT_SERIES["MLB"]


class TestPreservedBehaviour:
    def test_mlb_f5_spread_still_reaches_the_f5_winner_book(self, spy):
        kalshi.snapshot([_pick("MLB", "F5 Spread", pick="Boston -0.5")], "2026-11-07")
        assert "KXMLBF5" in spy["fetched"]

    def test_the_f5_special_case_does_not_leak_to_other_sports(self, spy):
        kalshi.snapshot([_pick("NBA", "F5 Spread")], "2026-11-07")
        assert "KXMLBF5" not in spy["fetched"]

    def test_it_never_raises(self, monkeypatch):
        monkeypatch.setattr(kalshi, "enabled", lambda: True)
        monkeypatch.setattr(kalshi, "fetch_open_markets",
                            lambda s: (_ for _ in ()).throw(RuntimeError("boom")))
        assert kalshi.snapshot([_pick("NBA", "Spread")], "2026-11-07") == []

    def test_disabled_by_the_kill_switch(self, monkeypatch):
        monkeypatch.setattr(kalshi, "enabled", lambda: False)
        assert kalshi.snapshot([_pick("NBA")], "2026-11-07") == []


class TestCfbPlaceholderDates:
    def test_cfb_walks_candidate_dates(self, monkeypatch):
        """CFB tickers carry a placeholder date — an exact-date join drops the
        whole Saturday slate. That trap already cost us CFB CLV once."""
        tried = []

        def fake_resolve(pick, markets, game_date, require_quote=True, series_map=None):
            tried.append(game_date)
            return None

        monkeypatch.setattr(kalshi, "enabled", lambda: True)
        monkeypatch.setattr(kalshi, "fetch_open_markets", lambda s: [{"t": 1}])
        monkeypatch.setattr(kalshi, "resolve_pick", fake_resolve)
        kalshi.snapshot([_pick("CFB", "Spread")], "2026-11-07")
        assert len(tried) > 1, tried
        assert "2026-11-06" in tried and "2026-11-08" in tried

    def test_a_non_placeholder_sport_tries_one_date_only(self, spy):
        kalshi.snapshot([_pick("NBA", "Spread")], "2026-11-07")
        assert len(spy["maps"]) == 1


class TestTheShapeNotTheInstance:
    """Guard the class of bug, not this one occurrence."""

    def test_snapshot_never_reads_the_bare_mlb_series_map(self):
        src = inspect.getsource(kalshi.snapshot)
        assert "SPORT_SERIES" in src
        # The bare name `SERIES` (not SPORT_SERIES) is what caused this.
        import re
        assert not re.search(r"(?<![_A-Z])SERIES\[", src), src

    @staticmethod
    def _callers():
        """Every call to KALSHI's resolve_pick, with whether it passes a map.

        A module is in scope if it imports the function, or defines the real
        one — identified by `series_map` in its OWN signature. That structural
        test is what separates it from `results_snapshot.py`'s namesake, which
        grades outcomes from ESPN scores and takes no map at all.
        """
        import ast
        import os
        out = []
        for root, _, files in os.walk("src"):
            for f in sorted(files):
                if not f.endswith(".py"):
                    continue
                path = os.path.join(root, f)
                tree = ast.parse(open(path).read(), path)
                imports_it = any(
                    isinstance(n, ast.ImportFrom)
                    and (n.module or "").startswith("src.data.kalshi")
                    and any(a.name == "resolve_pick" for a in n.names)
                    for n in ast.walk(tree))
                defs = [n for n in ast.walk(tree)
                        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                        and n.name == "resolve_pick"]
                defines_real = any(
                    any(a.arg == "series_map" for a in d.args.args + d.args.kwonlyargs)
                    for d in defs)
                if defs and not defines_real:
                    continue                      # a namesake, not our function
                if not (imports_it or defines_real):
                    continue
                for n in ast.walk(tree):
                    if not isinstance(n, ast.Call):
                        continue
                    if (getattr(n.func, "attr", None) or getattr(n.func, "id", None)) \
                            != "resolve_pick":
                        continue
                    passes = any(k.arg == "series_map" for k in n.keywords)
                    out.append((f"{path}:{n.lineno}", passes))
        return out

    def test_every_resolve_pick_caller_passes_a_series_map(self):
        """The parameter exists precisely because omitting it fails silently.

        Walks the whole repo, so the NEXT caller added without it fails here
        rather than after a season's worth of empty measurement.
        """
        misses = [loc for loc, ok in self._callers() if not ok]
        assert not misses, "resolve_pick called without series_map: " + ", ".join(misses)

    def test_the_guard_can_actually_see_the_callers(self):
        """A scoped guard that scopes itself down to nothing passes forever.

        The first draft skipped any module defining its own `resolve_pick` —
        which skipped kalshi.py, the file the bug was actually in.
        """
        callers = self._callers()
        files = {loc.split(":")[0] for loc, _ in callers}
        # Two real sites today: snapshot() and the shared CLV resolver, which
        # both CLV passes go through since they were merged. Named files below
        # so this cannot pass by finding two of something else.
        assert len(callers) >= 2, f"guard sees only {callers} — it has gone blind"
        assert "src/data/kalshi.py" in files, files
        assert "src/data/kalshi_clv.py" in files, files
