"""
WNBA model constants — watchlist only. Active May–September.
"""

WNBA_HOME_ADVANTAGE = 0.020
WNBA_BACK_TO_BACK_PENALTY = 0.040
WNBA_RECENT_WEIGHT = 0.45
WNBA_SPREAD_STD = 13.0   # was 8.5 (far too tight → favorite overconfidence).
                         # Empirical WNBA game-margin SD is ~15.3 (2025 season, 93-game
                         # sample); 13.0 sits ~15% below raw, mirroring NBA's 12-vs-14
                         # choice (the model's net-rating prediction explains part of the
                         # raw variance, so σ should sit a bit under the raw margin SD).
                         # Pairs with the real points-for-minus-against season net rating.
WNBA_REPLACEMENT_RATE = 0.55
WNBA_MAX_LINEUP_PENALTY = 0.30
WNBA_STATUS_WEIGHTS: dict = {
    "Out": 1.0,
    "Doubtful": 0.75,
    "Questionable": 0.40,
    "Day-To-Day": 0.20,
    "Probable": 0.05,
}
