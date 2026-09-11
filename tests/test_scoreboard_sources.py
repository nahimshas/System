"""Scoreboard polling must follow the ESPN map (Sep 11 2026).

Adding CFB to the ESPN endpoint map did NOT make CFB poll: SCOREBOARD_SOURCES
listed every sport by name, so the key existed with no consumer. The tab
rendered with no logos, no live scores and no settlement -- silently, because a
sport that is never polled looks exactly like a sport with no games.
"""
import re


def _js():
    with open("src/report/templates/_report_js.html") as f:
        return f.read()


class TestSourcesAreDerived:
    def test_sources_are_built_from_the_map(self):
        assert re.search(r"SCOREBOARD_SOURCES\s*=\s*Object\.keys\(ESPN\)", _js())

    def test_no_hand_listed_sports_remain(self):
        """The whole point: no `{ url: ESPN.XXX, sport: "XXX" }` rows."""
        block = _js().split("SCOREBOARD_SOURCES")[1][:400]
        assert not re.search(r"url:\s*ESPN\.[A-Z]+", block)

    def test_every_map_entry_is_polled(self):
        js = _js()
        keys = set(re.findall(r"^\s{4}([A-Z]+):\s*\"https://site\.api\.espn\.com",
                              js, re.M))
        assert {"MLB", "NFL", "CFB", "NBA", "NHL"} <= keys, keys
        # Derived construction means every key is polled by definition; this
        # guards the construction from being replaced by a hand list again.
        assert "Object.keys(ESPN)" in js


class TestCfbEndpoint:
    def test_cfb_present_with_fbs_and_page_size(self):
        m = re.search(r"CFB:\s*\"([^\"]+)\"", _js())
        assert m
        url = m.group(1)
        assert "football/college-football/scoreboard" in url
        assert "groups=80" in url
        assert int(re.search(r"limit=(\d+)", url).group(1)) >= 100
