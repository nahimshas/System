"""
Sport protocol and capability declaration.

SportCapabilities declares *what* a sport module can do so that main.py
can route its output (budget pool, parlays, history file) without needing
a chain of ``if sport == "nba": …`` conditionals.

Sport is a structural Protocol — any class that supplies the right
attributes and methods satisfies it without inheriting from anything.
Phase 2 sport modules (src/sports/nba.py, etc.) will implement this.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SportCapabilities:
    """
    Immutable capability declaration for one sport.

    Attributes:
        enters_budget:
            True  → picks are eligible for the daily budget singles pool
                    and Kelly sizing (NBA, MLB, NFL, NHL).
            False → watchlist / monitor only (IPL, WNBA, MLS).

        enters_parlays:
            True  → qualifying picks are fed into the parlay builder.
            False → no parlay construction for this sport.
            Note: currently only budget sports enter parlays; kept separate
            in case a watchlist sport is ever promoted selectively.

        track_in_main_history:
            True  → settled results land in state/history.json (the main
                    P&L ledger).
            False → results go to state/watchlist_history.json.

        uses_pending_file:
            True  → picks are staged in state/watchlist_pending.json and
                    settled asynchronously (IPL pattern — games span the
                    run window).
            False → picks are settled by date on the next morning run
                    (NHL/WNBA pattern) or by the budget settler.

        active_months:
            Set of calendar month numbers (1–12) when this sport is in
            season.  Empty set means always active.

        hours_lookahead:
            How many hours ahead to query the Odds API for upcoming games.
            Default 24 (today's games).  IPL uses 36 because games that
            start at ~7 am PST finish at ~11 am — past the 9 am run window
            — so tomorrow's match needs to be captured the night before.
    """

    enters_budget: bool = False
    enters_parlays: bool = False
    track_in_main_history: bool = False
    uses_pending_file: bool = False
    active_months: frozenset[int] = field(default_factory=frozenset)
    hours_lookahead: int = 24

    @property
    def is_watchlist(self) -> bool:
        """Convenience: True when the sport never touches the budget pool."""
        return not self.enters_budget


# ---------------------------------------------------------------------------
# Sport protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class Sport(Protocol):
    """
    Structural interface every sport module must satisfy.

    Phase 1 defines the contract.  Phase 2 provides concrete implementations
    in src/sports/<name>.py.  main.py will iterate REGISTRY rather than
    maintaining per-sport if/else blocks.

    Attributes:
        key   — odds-api sport key (e.g. "basketball_nba")
        label — human-readable name shown in logs and the report
        caps  — SportCapabilities instance declaring routing behaviour
    """

    key: str
    label: str
    caps: SportCapabilities

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch_games(self, today: str) -> list[dict[str, Any]]:
        """
        Return a list of today's game dicts from the Odds API (and any
        supplemental enrichment, e.g. player props).

        Args:
            today: ISO date string (YYYY-MM-DD) in Pacific time.

        Returns:
            List of game dicts — same shape as get_game_odds() output,
            optionally enriched with extra keys (player_props, etc.).
            Empty list if the sport is out of season or the API fails.
        """
        ...

    def fetch_context(self, today: str, games: list[dict[str, Any]]) -> dict[str, Any]:
        """
        Fetch supplemental statistical context needed by the analyzer
        (season stats, recent form, injuries, weather, …).

        Args:
            today: ISO date string.
            games: The list returned by fetch_games() — lets the context
                   fetcher pre-filter by team names / matchups.

        Returns:
            Arbitrary dict consumed by analyze_games().  Shape is
            sport-specific; callers treat it as opaque.
        """
        ...

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def analyze_games(
        self,
        games: list[dict[str, Any]],
        context: dict[str, Any],
        *,
        min_edge: float = 0.0,
    ) -> list[Any]:
        """
        Run the sport's edge-finding model over every game.

        Args:
            games:    Output of fetch_games().
            context:  Output of fetch_context().
            min_edge: Minimum fractional edge to include in the return
                      list.  Pass 0.0 for the display list (all picks);
                      pass MIN_EDGE for the budget/qualify list.

        Returns:
            List of Bet-like objects (namedtuple or dataclass) with at
            least: .edge, .home_team, .away_team, .pick, .bet_type.
        """
        ...

    # ------------------------------------------------------------------
    # Settlement
    # ------------------------------------------------------------------

    def settle(self, today: str) -> int:
        """
        Check and settle yesterday's picks for this sport.

        Returns the number of picks that were newly settled.
        Implementations should be idempotent (safe to call multiple times).
        """
        ...
