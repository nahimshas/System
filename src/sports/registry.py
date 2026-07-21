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
NHL     True           True                  True                   False   ← graduated to budget (Jun 2026); also still watchlist-tracked
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
            has_props=True,
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
            has_props=True,
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
            has_props=False,   # flip to True + implement fetch_props() when NFL props are ready
            in_main_display_pool=True,
            active_months=frozenset({9, 10, 11, 12, 1, 2}),
            hours_lookahead=24,
        ),
    ),

    # ── NHL — GRADUATED to budget sport (June 2026) ──────────────────── #
    # NHL was watchlist-only; promoted to the budget pool as a learning exercise.
    # The 3 flags below (enters_budget / enters_parlays / track_in_main_history)
    # route its placed bets into the allocation table, parlays, and history.json
    # (→ the by-sport P&L tile in Model Performance).
    #
    # "Track both in parallel": check_and_settle_watchlist() still reads NHL from
    # state["singles_display"] (ALL model picks) and logs them to
    # watchlist_history.json (→ the Watchlist Tracking W-L tile), while the budget
    # settler check_and_settle() logs the PLACED bets from state["singles"] to
    # history.json. Both settlers run unconditionally in main.py and read
    # different lists, so the model-accuracy record and the money record both keep
    # accumulating with no settlement-code changes.
    #
    # NOTE: graduated mid-playoffs as a learning exercise — real validation waits
    # for next NHL season's volume.

    "nhl": SportEntry(
        slug="nhl",
        key="icehockey_nhl",
        label="NHL",
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

    # FIFA World Cup 2026 — watchlist-only own tile. June 11 – July 19, 2026.
    # Elo-driven model (national teams, no club xG). Neutral venues → home edge
    # only for host nations. Settles date-based via check_and_settle_watchlist().
    "wc": SportEntry(
        slug="wc",
        key="soccer_fifa_world_cup",
        label="World Cup",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=False,
            in_main_display_pool=False,
            active_months=frozenset({6, 7}),
            hours_lookahead=24,
        ),
    ),

    # Liga MX (Mexican Primera División) — watchlist-only own tile. Added Jul 2026
    # (Robinhood: Win/Tie/Don't-Win). Elo-driven (no club xG for Mexico), like WC
    # but with real home advantage. Settles date-based via check_and_settle_watchlist().
    "ligamx": SportEntry(
        slug="ligamx",
        key="soccer_mexico_ligamx",
        label="Liga MX",
        caps=SportCapabilities(
            enters_budget=False,
            enters_parlays=False,
            track_in_main_history=False,
            uses_pending_file=False,
            in_main_display_pool=False,
            active_months=frozenset({1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12}),
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
