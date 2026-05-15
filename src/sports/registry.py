"""
Central sport registry.

REGISTRY maps each sport's short slug (the key used in --league flags and
SPORT_ACTIVE_MONTHS) to a SportEntry — a lightweight record combining the
odds-api key, display label, and SportCapabilities.

Display routing quick-reference
--------------------------------
Sport   enters_budget  in_main_display_pool  track_in_main_history  uses_pending
NBA     True           True                  True                   False
MLB     True           True                  True                   False
NFL     True           True                  True                   False
NHL     False          True                  False                  False   ← display-only in main card
IPL     False          False                 False                  True    ← own tile + pending file
WNBA    False          False                 False                  False   ← own tile
MLS     False          False                 False                  False   ← own tile

Usage::

    from src.sports.registry import REGISTRY, get_sport, active_sports

    caps = REGISTRY["wnba"].caps
    if caps.enters_budget:
        ...

    for slug, entry in active_sports(today_month=5).items():
        print(entry.label, entry.caps.is_watchlist)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.sports.base import SportCapabilities


# ---------------------------------------------------------------------------
# Entry type
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SportEntry:
    """
    Registry record for one sport.

    Attributes:
        slug:   Short identifier used in CLI flags and state keys
                (e.g. "nba", "ipl").
        key:    Odds API sport key (e.g. "basketball_nba").
        label:  Human-readable display name (e.g. "NBA").
        caps:   SportCapabilities — routing flags used by main.py.
    """

    slug: str
    key: str
    label: str
    caps: SportCapabilities


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

REGISTRY: dict[str, SportEntry] = {

    # ── Budget sports (enter daily pool + parlays + main history) ─────── #

    "nba": SportEntry(
        slug="nba",
        key="basketball_nba",
        label="NBA",
        caps=SportCapabilities(
            enters_budget=True,
            enters_parlays=True,
            track_in_main_history=True,
            uses_pending_file=False,
            in_main_display_pool=True,
            active_months=frozenset({10, 11, 12, 1, 2, 3, 4, 5, 6}),
            hours_lookahead=24,
        ),
    ),

    "mlb": SportEntry(
        slug="mlb",
        key="baseball_mlb",
        label="MLB",
        caps=SportCapabilities(
            enters_budget=True,
            enters_parlays=True,
            track_in_main_history=True,
            uses_pending_file=False,
            in_main_display_pool=True,
            active_months=frozenset({3, 4, 5, 6, 7, 8, 9, 10}),
            hours_lookahead=24,
        ),
    ),

    "nfl": SportEntry(
        slug="nfl",
        key="americanfootball_nfl",
        label="NFL",
        caps=SportCapabilities(
            enters_budget=True,
            enters_parlays=True,
            track_in_main_history=True,
            uses_pending_file=False,
            in_main_display_pool=True,
            active_months=frozenset({9, 10, 11, 12, 1, 2}),
            hours_lookahead=24,
        ),
    ),

    # ── NHL — display-only in main card, watchlist history, no budget ─── #
    # NHL picks appear in the per-league card section alongside budget sports
    # (in_main_display_pool=True) but are never allocated budget dollars,
    # never enter parlays, and settle into watchlist_history.json not history.json.

    "nhl": SportEntry(
        slug="nhl",
        key="icehockey_nhl",
        label="NHL",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=False,
            in_main_display_pool=True,
            active_months=frozenset({10, 11, 12, 1, 2, 3, 4, 5, 6}),
            hours_lookahead=24,
        ),
    ),

    # ── Watchlist sports (own display tile, watchlist history) ────────── #

    "ipl": SportEntry(
        slug="ipl",
        key="cricket_ipl",
        label="IPL",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=True,   # games span the 9am run window
            in_main_display_pool=False,
            active_months=frozenset({3, 4, 5, 6}),
            hours_lookahead=36,       # capture tomorrow's match tonight
        ),
    ),

    "wnba": SportEntry(
        slug="wnba",
        key="basketball_wnba",
        label="WNBA",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=False,
            in_main_display_pool=False,
            active_months=frozenset({5, 6, 7, 8, 9, 10}),
            hours_lookahead=24,
        ),
    ),

    "mls": SportEntry(
        slug="mls",
        key="soccer_usa_mls",
        label="MLS",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=False,
            in_main_display_pool=False,
            active_months=frozenset({2, 3, 4, 5, 6, 7, 8, 9, 10, 11}),
            hours_lookahead=24,
        ),
    ),
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_sport(slug: str) -> Optional[SportEntry]:
    """Return the SportEntry for *slug*, or None if not registered."""
    return REGISTRY.get(slug)


def active_sports(today_month: int) -> dict[str, SportEntry]:
    """
    Return the subset of REGISTRY whose active_months includes *today_month*.

    Sports with an empty active_months set are treated as always active.

    Args:
        today_month: Calendar month number (1–12).

    Returns:
        Ordered dict (insertion order = registration order) of slug → SportEntry.
    """
    return {
        slug: entry
        for slug, entry in REGISTRY.items()
        if not entry.caps.active_months or today_month in entry.caps.active_months
    }


def budget_sports() -> dict[str, SportEntry]:
    """Return only sports that enter the daily budget pool."""
    return {slug: e for slug, e in REGISTRY.items() if e.caps.enters_budget}


def watchlist_sports() -> dict[str, SportEntry]:
    """Return only watchlist/monitor sports (no budget allocation)."""
    return {slug: e for slug, e in REGISTRY.items() if e.caps.is_watchlist}


def main_display_sports() -> dict[str, SportEntry]:
    """Return sports whose picks merge into the shared main display pool (NBA/MLB/NFL/NHL)."""
    return {slug: e for slug, e in REGISTRY.items() if e.caps.in_main_display_pool}


def own_display_sports() -> dict[str, SportEntry]:
    """Return sports that have their own separate display tile (IPL/WNBA/MLS)."""
    return {slug: e for slug, e in REGISTRY.items() if not e.caps.in_main_display_pool}
