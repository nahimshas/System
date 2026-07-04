"""
Global, non-sport-specific configuration.

Covers: API credentials, odds API settings, sport keys/labels,
betting parameters, cross-sport signal thresholds, and output paths.
"""

import os
from typing import List

# --- API Keys (loaded from environment / GitHub Secrets) ---
ODDS_API_KEY: str = os.environ.get("ODDS_API_KEY", "")
EMAIL_PASSWORD: str = os.environ.get("EMAIL_PASSWORD", "")
EMAIL_FROM: str = os.environ.get("EMAIL_FROM", "nahimshas@hotmail.com")
EMAIL_TO: List[str] = os.environ.get("EMAIL_TO", "nahimshas@hotmail.com").split(",")

# --- The Odds API ---
ODDS_API_BASE = "https://api.the-odds-api.com/v4"
PREFERRED_BOOK = "draftkings"
FALLBACK_BOOKS = ["fanduel", "betmgm", "bovada"]

# --- Sports keys ---
NBA_SPORT = "basketball_nba"
MLB_SPORT = "baseball_mlb"
NFL_SPORT = "americanfootball_nfl"
NHL_SPORT = "icehockey_nhl"
IPL_SPORT = "cricket_ipl"
WNBA_SPORT = "basketball_wnba"
MLS_SPORT = "soccer_usa_mls"
SPORT_LABELS = {
    NBA_SPORT: "NBA",
    MLB_SPORT: "MLB",
    NFL_SPORT: "NFL",
    NHL_SPORT: "NHL",
    IPL_SPORT: "IPL",
    WNBA_SPORT: "WNBA",
    MLS_SPORT: "MLS",
}

# --- Active-season calendar (month numbers) ---
# Used by main.py to skip analysis for leagues not currently in-season.
# Months are 1-indexed. Overlapping months (e.g. NBA/NHL both in Oct-Apr) are fine.
SPORT_ACTIVE_MONTHS = {
    "nba": list(range(10, 13)) + list(range(1, 7)),   # Oct – Jun (regular + playoffs)
    "mlb": list(range(3, 12)),                          # Mar – Nov
    "nfl": list(range(9, 13)) + [1, 2],                # Sep – Feb
    "nhl": list(range(10, 13)) + list(range(1, 7)),    # Oct – Jun
    "ipl": [3, 4, 5],                                  # Mar – May
    "wnba": [5, 6, 7, 8, 9],                           # May – Sep
    "mls":  list(range(2, 12)),                        # Feb – Nov
}

# --- Betting parameters ---
DAILY_BUDGET: float = float(os.environ.get("DAILY_BUDGET", "100"))
ROBINHOOD_COMMISSION: float = 0.02   # $0.02 per contract bought
KELLY_FRACTION: float = 0.50          # 1/2 Kelly
MIN_EDGE: float = 0.03                # minimum 3% edge to recommend (analyzers + display pools)
BUDGET_MIN_EDGE: float = 0.05         # Jul 4 2026 optimization: budget (real-money) entry requires ≥5% effective edge — sub-5% edges showed no win-rate relationship (noise); display/watchlist/logs keep MIN_EDGE
MAX_SINGLE_BETS: int = 5              # max single-game bets per day
MAX_PARLAYS: int = 2                  # max 2-leg parlay recommendations
MAX_PROPS_PER_SPORT: int = 6          # max props per sport (NBA/MLB); mirrors analyzer cap
MIN_PARLAY_LEG_EDGE: float = 0.025   # each parlay leg must have ≥ 2.5% edge

# ── Display / today's-card gating (DISPLAY-ONLY — never affects logging) ──────
# What the PWA shows and the today's card bets is filtered by these; the shadow
# log and decision log keep recording EVERY candidate for analysis regardless.
#   • PWA_HIDDEN_MARKETS — markets the model is structurally bad at (negative CLV
#     everywhere). Gated from display + budget, still fully logged.
#   • PWA_MODEL_PROB_FLOOR — hide picks the model itself rates below this to win.
#     The deep-longshot tail is where the model is most overconfident and CLV is
#     least reliable, and realized ROI there is negative. 0.38 keeps the proven
#     40–50% bucket while cutting the <40% longshots. Still logged.
PWA_HIDDEN_MARKETS: frozenset = frozenset({"Total", "Draw"})
PWA_MODEL_PROB_FLOOR: float = 0.38

# --- Schedule load (7-day fatigue) ---
# Applied when a team has played many games in the last 7 days
SCHEDULE_LOAD_THRESHOLDS = {5: 0.01, 6: 0.02, 7: 0.03}  # games_in_7d → penalty

# --- Line movement signal ---
LINE_MOVE_THRESHOLD = 0.03           # 3% probability shift triggers sharp-money signal

# --- Output ---
REPORT_DIR = "docs"
REPORT_FILE = "index.html"

# --- Performance history ---
HISTORY_FILE = "state/history.json"
