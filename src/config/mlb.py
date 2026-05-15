"""
MLB-specific model constants.
"""

MLB_HOME_ADVANTAGE = 0.015           # 1.5% (was 2.0%)
MLB_BACK_TO_BACK_PENALTY = 0.010     # 1.0% residual (was 1.5%)
MLB_REST_BONUS_PER_DAY = 0.005       # 0.5% per day (was 0.8%)

# --- Playoff context adjustments ---
# MLB playoffs (October – early November): aces pitch more, starters pulled earlier
MLB_PLAYOFF_SCORING_FACTOR = 0.91    # ~9% run reduction (4.5 → ~4.1 R/G)
MLB_PLAYOFF_STARTER_IP     = 4.8     # shorter leash vs 5.5 inn regular season
MLB_PLAYOFF_RECENT_WEIGHT  = 0.55    # flip toward recent form
