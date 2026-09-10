"""
Spreads must be quoted at lines Robinhood/Kalshi actually offer.

Kalshi event contracts are BINARY — there is no push — so the spread ladders
list only half points (verified Sep 2026: KXNFLSPREAD offers 1.5, 2.5, 3.5 ...
and no whole numbers). A book line of -3 is therefore not a bet the user can
place, and the Sep 9 NFL card shipped exactly that: "Seattle Seahawks -3.0".

This is NOT a labelling fix. -2.5 and -3.5 are materially different bets and
the gap between them is the push mass at exactly 3 — the most common NFL
margin. Relabelling -3 as -2.5 would hand us those pushes for free and
overstate the edge.
"""
import pytest
from scipy.stats import norm

from src.models.edge_finder import half_point_line


class TestHalfPointSnap:
    def test_whole_line_moves_against_the_side_we_favour(self):
        """Conservative direction: when the true bettable price is unknown, the
        only safe move is the one that makes our own bet harder."""
        assert half_point_line(-3.0, 0.60) == -3.5    # we like home -> harder
        assert half_point_line(-3.0, 0.40) == -2.5    # we like away -> harder for away
        assert half_point_line(3.0, 0.60) == 2.5
        assert half_point_line(3.0, 0.40) == 3.5

    def test_half_lines_are_left_alone(self):
        for line in (-2.5, -1.5, 0.5, 7.5, -13.5):
            assert half_point_line(line, 0.6) == line
            assert half_point_line(line, 0.4) == line

    def test_pick_em_is_snapped_too(self):
        """A 0 line pushes on any tie and is not listed on Kalshi."""
        assert half_point_line(0.0, 0.60) == -0.5
        assert half_point_line(0.0, 0.40) == 0.5

    def test_none_passes_through(self):
        assert half_point_line(None, 0.5) is None

    def test_snapping_reduces_our_cover_probability(self):
        """The whole point: moving to a bettable line COSTS us probability, and
        that cost must be paid rather than hidden."""
        std = 13.5
        margin = float(norm.ppf(0.62)) * std
        at_whole = float(norm.cdf(margin + -3.0, 0, std))
        at_half = float(norm.cdf(margin + half_point_line(-3.0, at_whole), 0, std))
        assert at_half < at_whole
        assert at_whole - at_half > 0.005, "snap should cost real probability"


class TestNoWholeLinesReachTheCard:
    def test_nfl_analyzer_snaps(self):
        src = open("src/models/edge_finder.py").read()
        i = src.index("def analyze_nfl_game")
        j = src.index("def ", i + 10)
        assert "half_point_line(" in src[i:j], "NFL spread never snapped"

    def test_nba_analyzer_snaps(self):
        src = open("src/models/edge_finder.py").read()
        i = src.index("def analyze_nba_game")
        j = src.index("def ", i + 10)
        assert "half_point_line(" in src[i:j], "NBA spread never snapped"

    def test_cfb_analyzer_snaps(self):
        src = open("src/models/edge_finder.py").read()
        i = src.index("def analyze_cfb_game")
        assert "half_point_line(" in src[i:], "CFB spread never snapped"

    def test_market_prob_is_repriced_not_just_relabelled(self):
        """If only the line moved and the market prob stayed, the edge would be
        silently inflated by the push mass."""
        src = open("src/models/edge_finder.py").read()
        i = src.index("def analyze_nfl_game")
        j = src.index("def ", i + 10)
        block = src[i:j]
        assert "market_home_cover = float(" in block and "_mkt_margin" in block, \
            "market probability must be shifted to the snapped line too"
