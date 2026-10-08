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

# ── Cold-start warm start (Oct 2026) ─────────────────────────────────────────
# NBA had NO prior-season blending, unlike NFL. Measured on the live preseason
# sample: net ratings from one game spanned -34..+34 with sd 13.7, against a
# real NBA spread of roughly -12..+11, sd 6.1 — 3.4x too wide. A simulated
# matchup produced a +15.0% moneyline edge with the credibility cap FIRING,
# which is the same "cap-pinned phantom edge" signature as the CFB first slate.
#
# NBA_PRIOR_REGRESSION is MEASURED, not guessed: the season-over-season
# regression slope of team net rating across four pairs (2022->23 .. 2025->26)
# was 0.445 / 0.779 / 0.633 / 0.543, mean 0.600, with a notably stable
# correlation of r = 0.52-0.59. NBA regresses slightly MORE than NFL (0.67),
# which is the opposite of what roster stability would suggest — hence measuring.
NBA_PRIOR_REGRESSION    = 0.60   # keep 60% of last season's net rating
NBA_LEAGUE_AVG_PPG      = 113.0  # anchor for regressing ppg/oppg toward the mean

# Ramp derived from when the current season out-informs the prior. With a
# game-margin sd of 12.0, the standard error of a team's mean net rating is
# 12/sqrt(n); the prior's residual sd is 6.0*sqrt(1-r^2) ~ 5.0. They cross at
# about 8 games, so a linear ramp to full-current by 15 leaves the blend
# prior-weighted exactly while the prior is still the better estimate.
NBA_WARMSTART_RAMP_GAMES = 15
