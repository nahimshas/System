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
SPORT_LABELS = {
    NBA_SPORT: "NBA",
    MLB_SPORT: "MLB",
    NFL_SPORT: "NFL",
    NHL_SPORT: "NHL",
    IPL_SPORT: "IPL",
    WNBA_SPORT: "WNBA",
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
}

# --- Betting parameters ---
DAILY_BUDGET: float = float(os.environ.get("DAILY_BUDGET", "100"))
ROBINHOOD_COMMISSION: float = 0.02   # $0.02 per contract bought
KELLY_FRACTION: float = 0.25          # 1/4 Kelly to reduce variance
MIN_EDGE: float = 0.03                # minimum 3% edge to recommend
MAX_SINGLE_BETS: int = 5              # max single-game bets per day
MAX_PARLAYS: int = 2                  # max 2-leg parlay recommendations
MAX_PROPS_PER_SPORT: int = 6          # max props per sport (NBA/MLB); mirrors analyzer cap
MIN_PARLAY_LEG_EDGE: float = 0.025   # each parlay leg must have ≥ 2.5% edge

# --- Adjustment weights for probability model ---
# These represent the RESIDUAL edge beyond what the market already prices in.
# B2B, rest, and injuries are public — the market adjusts for them quickly.
# We only claim a small fraction of the full effect as an additional edge.
NBA_HOME_ADVANTAGE = 0.025            # 2.5% (market partially prices home court)
NBA_BACK_TO_BACK_PENALTY = 0.020     # 2.0% residual (was 4.0%; market already adjusts ~2%)
NBA_REST_BONUS_PER_DAY = 0.008       # 0.8% per day (was 1.5%; max 3 days = 2.4% total)
NBA_RECENT_FORM_WEIGHT = 0.30        # weight toward last-14d vs season avg (regular season)

MLB_HOME_ADVANTAGE = 0.015           # 1.5% (was 2.0%)
MLB_BACK_TO_BACK_PENALTY = 0.010     # 1.0% residual (was 1.5%)
MLB_REST_BONUS_PER_DAY = 0.005       # 0.5% per day (was 0.8%)

# --- Playoff context adjustments ---
# NBA playoffs (mid-April → mid-June): slower pace, tighter defense, lower scoring
NBA_PLAYOFF_SCORING_FACTOR = 0.94    # ~6% scoring reduction vs regular season
NBA_PLAYOFF_PACE_FACTOR    = 0.96    # ~4% pace reduction
NBA_PLAYOFF_RECENT_WEIGHT  = 0.55    # flip toward recent form — playoff games >> reg season
NBA_TOTAL_STD              = 15.0    # model prediction uncertainty for game totals (~15 pts)
NBA_PLAYOFF_TOTAL_STD      = 13.0    # slightly tighter in playoffs (better film, fewer blowouts)

# MLB playoffs (October – early November): aces pitch more, starters pulled earlier
MLB_PLAYOFF_SCORING_FACTOR = 0.91    # ~9% run reduction (4.5 → ~4.1 R/G)
MLB_PLAYOFF_STARTER_IP     = 4.8     # shorter leash vs 5.5 inn regular season
MLB_PLAYOFF_RECENT_WEIGHT  = 0.55    # same flip toward recent form

# NHL playoffs (mid-April → mid-June): peak goaltending, tighter defence, slower scoring
NHL_PLAYOFF_SCORING_FACTOR = 0.92    # ~8% goal reduction vs regular season (6.2 → ~5.7 G/G)

# --- Schedule load (7-day fatigue) ---
# Applied when a team has played many games in the last 7 days
SCHEDULE_LOAD_THRESHOLDS = {5: 0.01, 6: 0.02, 7: 0.03}  # games_in_7d → penalty

# --- WNBA — watchlist only ---
WNBA_HOME_ADVANTAGE = 0.020
WNBA_BACK_TO_BACK_PENALTY = 0.040
WNBA_RECENT_WEIGHT = 0.45
WNBA_SPREAD_STD = 8.5
WNBA_REPLACEMENT_RATE = 0.55
WNBA_MAX_LINEUP_PENALTY = 0.30
WNBA_STATUS_WEIGHTS: dict = {"Out": 1.0, "Doubtful": 0.75, "Questionable": 0.40, "Day-To-Day": 0.20, "Probable": 0.05}

# --- IPL (Indian Premier League cricket) — watchlist only ---
# T20 home advantage is substantial; ~5% raw but market prices most of it.
# Recent form dominates in a short tournament (10 teams, ~14 matches each).
IPL_HOME_ADV      = 0.025   # ~2.5% residual home advantage beyond market pricing
IPL_RECENT_WEIGHT = 0.65    # T20 form is highly volatile — weight recent heavily

# --- Line movement signal ---
LINE_MOVE_THRESHOLD = 0.03           # 3% probability shift triggers sharp-money signal

# --- Output ---
REPORT_DIR = "docs"
REPORT_FILE = "index.html"

# --- Performance history ---
HISTORY_FILE = "state/history.json"
