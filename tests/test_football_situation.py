"""Possession / down & distance on football cards (Sep 20 2026).

User: "can a football be added under the logo of who has the ball and in which
yard and that? ... and the down too".

NFL and CFB share one ESPN shape, so one renderer serves both. `situation` was
already captured generically by parseEvent for every sport — only the football
rendering was missing.

⚠️ ESPN DROPS `situation` ENTIRELY once a game ends, and populates it
inconsistently around kickoffs, timeouts and reviews. "Absent" is the normal
case, not an error, and must never leave a stray football under a logo. The
logic below was exercised against synthetic ESPN payloads in node and rendered
in a browser; the field names come from the live play data (down / distance /
yardLine / team.id) plus ESPN's documented scoreboard extras.
"""
import re

import pytest


def _js():
    with open("src/report/templates/_report_js.html") as f:
        return f.read()


def _fn(name):
    """Source of one top-level function, brace-matched."""
    js = _js()
    i = js.index("function " + name + "(")
    d, j = 0, js.index("{", i)
    for k in range(j, len(js)):
        if js[k] == "{":
            d += 1
        elif js[k] == "}":
            d -= 1
            if d == 0:
                return js[i:k + 1]
    raise AssertionError(name)


class TestTeamIdsAreCaptured:
    """possession is a TEAM ID — without ids there is nothing to match it to."""

    def test_parse_event_exposes_ids(self):
        js = _js()
        assert "homeId: homeId, awayId: awayId," in js
        assert 'var homeId    = String((homeComp.team || {}).id || "");' in js


class TestDefensiveByDefault:
    def test_only_renders_for_in_progress_games(self):
        assert 'ev.stateId !== "2"' in _fn("_footballSituation")

    def test_returns_null_without_situation(self):
        body = _fn("_footballSituation")
        assert "if (!st) return null;" in body

    def test_returns_null_when_nothing_usable(self):
        """No side, no down, no field position -> render nothing at all."""
        assert "if (!side && !dd && !fieldPos) return null;" in _fn("_footballSituation")

    def test_possession_mark_is_side_scoped(self):
        body = _fn("_possessionMark")
        assert "sit.side !== which" in body
        assert "🏈" in body

    def test_situation_line_empty_without_bits(self):
        assert "if (!bits.length) return \"\";" in _fn("_footballSituationLine")


class TestFallbacks:
    def test_prefers_espn_preformatted_text(self):
        body = _fn("_footballSituation")
        assert "st.downDistanceText" in body
        assert "st.shortDownDistanceText" in body
        assert "st.possessionText" in body

    def test_composes_from_raw_numbers_when_text_missing(self):
        body = _fn("_footballSituation")
        assert "st.down" in body and "st.distance" in body and "st.yardLine" in body

    def test_goal_to_go_when_distance_is_zero(self):
        assert '& Goal' in _fn("_footballSituation")

    def test_possession_falls_back_to_abbreviation(self):
        """`possession` id missing -> read the abbr off possessionText."""
        body = _fn("_footballSituation")
        assert "indexOf(String(ev.homeAbbr).toUpperCase()) === 0" in body

    def test_ordinals_cover_all_four_downs(self):
        body = _fn("_ordinalDown")
        for o in ("1st", "2nd", "3rd", "4th"):
            assert o in body


class TestWiring:
    def test_only_football_sports_compute_it(self):
        js = _js()
        assert 'sport === "NFL" || sport === "CFB"' in js

    def test_marks_are_passed_as_the_under_logo_extras(self):
        js = _js()
        assert '_possessionMark(_fSit, "away")' in js
        assert '_possessionMark(_fSit, "home")' in js

    def test_line_is_in_the_centre_column(self):
        assert "_footballSituationLine(_fSit)" in _js()

    def test_scoreboard_supports_per_team_extras(self):
        """The hook this rides on — extras render under each team's record."""
        assert "function _renderScoreboard(ev, centerHtml, awayExtra, homeExtra)" in _js()

    def test_red_zone_is_highlighted(self):
        body = _fn("_footballSituationLine")
        assert "redZone" in body and "#ef4444" in body
