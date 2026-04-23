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
SPORT_LABELS = {NBA_SPORT: "NBA", MLB_SPORT: "MLB"}

# --- Betting parameters ---
DAILY_BUDGET: float = float(os.environ.get("DAILY_BUDGET", "100"))
ROBINHOOD_COMMISSION: float = 0.02   # $0.02 per contract bought
KELLY_FRACTION: float = 0.25          # 1/4 Kelly to reduce variance
MIN_EDGE: float = 0.03                # minimum 3% edge to recommend
MAX_SINGLE_BETS: int = 5              # max single-game bets per day
MAX_PARLAYS: int = 2                  # max 2-leg parlay recommendations
MIN_PARLAY_LEG_EDGE: float = 0.025   # each parlay leg must have ≥ 2.5% edge

# --- Adjustment weights for probability model ---
NBA_HOME_ADVANTAGE = 0.030            # 3% home win probability boost
NBA_BACK_TO_BACK_PENALTY = 0.040     # 4% penalty for team on B2B
NBA_REST_BONUS_PER_DAY = 0.015       # 1.5% per extra rest day (max 3 days)
NBA_RECENT_FORM_WEIGHT = 0.30        # weight toward last-10 vs season avg

MLB_HOME_ADVANTAGE = 0.020
MLB_BACK_TO_BACK_PENALTY = 0.015
MLB_REST_BONUS_PER_DAY = 0.008

# --- Output ---
REPORT_DIR = "docs"
REPORT_FILE = "index.html"
