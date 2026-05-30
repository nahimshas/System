"""
NBA-specific model constants.

These represent the RESIDUAL edge beyond what the market already prices in.
B2B, rest, and injuries are public — the market adjusts for them quickly.
We only claim a small fraction of the full effect as an additional edge.
"""

NBA_HOME_ADVANTAGE = 0.025            # 2.5% (market partially prices home court)
NBA_BACK_TO_BACK_PENALTY = 0.020     # 2.0% residual (was 4.0%; market already adjusts ~2%)
NBA_REST_BONUS_PER_DAY = 0.008       # 0.8% per day (was 1.5%; max 3 days = 2.4% total)
NBA_RECENT_FORM_WEIGHT = 0.30        # weight toward last-14d vs season avg (regular season)

# --- Playoff context adjustments ---
# NBA playoffs (mid-April → mid-June): slower pace, tighter defense, lower scoring
NBA_PLAYOFF_SCORING_FACTOR = 0.94    # ~6% scoring reduction vs regular season
NBA_PLAYOFF_PACE_FACTOR    = 0.96    # ~4% pace reduction
NBA_PLAYOFF_RECENT_WEIGHT  = 0.55    # flip toward recent form — playoff games >> reg season
NBA_TOTAL_STD              = 16.5    # model prediction uncertainty for game totals (was 15.0);
                                     # widened because totals realised ~51% vs ~73% predicted —
                                     # the distribution was too tight, manufacturing false edges.
NBA_PLAYOFF_TOTAL_STD      = 16.5    # was 13.0 (tighter than regular season — backwards: that
                                     # raised totals confidence in exactly the worst-performing
                                     # period). Now equal to the regular-season value.
