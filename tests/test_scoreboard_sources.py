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
        """Derived from ESPN, however it is spelled — a map value may now be a
        single URL or an array of them (soccer needs one fetch per UTC date)."""
        js = _js()
        # NB: splitting on the name is unreliable — the push line contains it
        # too — so assert against the construction region as a whole.
        i = js.index("var SCOREBOARD_SOURCES")
        block = js[i:i + 600]
        assert "Object.keys(ESPN)" in block
        assert "SCOREBOARD_SOURCES.push" in block or ".map(" in block

    def test_no_hand_listed_sports_remain(self):
        """The whole point: no `{ url: ESPN.XXX, sport: "XXX" }` rows."""
        assert not re.search(r"url:\s*ESPN\.[A-Z]+", _js())

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
