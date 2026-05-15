"""
WNBA model constants — watchlist only. Active May–September.
"""

WNBA_HOME_ADVANTAGE = 0.020
WNBA_BACK_TO_BACK_PENALTY = 0.040
WNBA_RECENT_WEIGHT = 0.45
WNBA_SPREAD_STD = 8.5
WNBA_REPLACEMENT_RATE = 0.55
WNBA_MAX_LINEUP_PENALTY = 0.30
WNBA_STATUS_WEIGHTS: dict = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.40,
    "Day-To-Day": 0.20,
    "Probable": 0.05,
}
