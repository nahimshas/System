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


# ── Field position (Nov 2 2026) ────────────────────────────────────────────
# User, reading the live NFL card: "says the ball is on chicago's 65, what does
# that mean?" — it meant nothing. A field only counts to 50 from each sideline,
# so no yard line above 50 exists. ESPN's `yardLine` is a 0-100 ABSOLUTE
# measured from the home team's goal line and we printed it raw.
#
# These tests RUN the shipped function in node rather than grepping it, because
# the broken version read perfectly well as source: `abbr + " " + st.yardLine`
# is exactly what you would write if you believed yardLine were a yard marker.
# Only evaluating it against ESPN's own labelled pairs shows the difference.
import json
import shutil
import subprocess

_NODE = shutil.which("node")


def _run_situation(cases):
    """Evaluate the SHIPPED _footballSituation over `cases` in node."""
    harness = (
        _fn("_ordinalDown") + "\n" + _fn("_footballSituation") + "\n"
        "var out = JSON.parse(process.argv[1]).map(function (ev) {\n"
        "  return _footballSituation(ev);\n"
        "});\n"
        "console.log(JSON.stringify(out));\n"
    )
    res = subprocess.run([_NODE, "-e", harness, json.dumps(cases)],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def _ev(yard_line, home="SEA", away="CHI", possession="3", **kw):
    ev = {"stateId": "2", "homeAbbr": home, "awayAbbr": away,
          "homeId": "26", "awayId": "3",
          "situation": {"yardLine": yard_line, "possession": possession}}
    ev["situation"].update(kw)
    return ev


@pytest.mark.skipif(_NODE is None, reason="node not installed")
class TestFieldPosition:
    # ESPN's own start/end labels for CHI @ SEA, read live on Nov 2 2026.
    # Home is SEA, so yardLine counts up from Seattle's goal line.
    ESPN_LABELLED = [(32, "SEA 32"), (80, "CHI 20"), (0, "SEA 0"),
                     (96, "CHI 4"), (91, "CHI 9"), (46, "SEA 46"),
                     (75, "CHI 25"), (72, "CHI 28"), (65, "CHI 35")]

    def test_matches_espns_own_labels(self):
        cases = [_ev(y) for y, _ in self.ESPN_LABELLED]
        got = _run_situation(cases)
        for (y, want), g in zip(self.ESPN_LABELLED, got):
            assert g["fieldPos"] == want, f"yardLine {y} -> {g['fieldPos']}"

    def test_the_reported_bug_specifically(self):
        """yardLine 65, the value live on the card the user was looking at."""
        assert _run_situation([_ev(65)])[0]["fieldPos"] == "CHI 35"

    def test_no_yard_line_above_fifty_can_ever_be_printed(self):
        """The invariant the bug violated, over the whole domain."""
        got = _run_situation([_ev(y) for y in range(0, 101)])
        for g in got:
            n = int(g["fieldPos"].split()[1])
            assert 0 <= n <= 50, g["fieldPos"]

    def test_the_label_names_the_half_not_the_possessor(self):
        """A Seattle drive ending on Chicago's 9 is 'CHI 9'.

        The old code chose the abbreviation from `possession`, which is right
        only while a team is in its own half.
        """
        got = _run_situation([_ev(91, possession="26")])   # SEA has the ball
        assert got[0]["side"] == "home"
        assert got[0]["fieldPos"] == "CHI 9"

    def test_possession_text_still_wins_when_espn_sends_it(self):
        got = _run_situation([_ev(65, possessionText="SEA 35")])
        assert got[0]["fieldPos"] == "SEA 35"

    def test_field_position_survives_unknown_possession(self):
        """ESPN drops `possession` around timeouts — which is exactly when the
        fallback runs. Requiring a side meant printing nothing."""
        ev = _ev(65, possession=None)
        got = _run_situation([ev])
        assert got[0]["fieldPos"] == "CHI 35"

    def test_junk_yard_lines_are_dropped_not_printed(self):
        for bad in (-1, 101, "nope", None):
            got = _run_situation([_ev(bad, down=2, distance=7)])
            assert got[0]["fieldPos"] == "", bad
