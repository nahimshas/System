"""
MLS (Major League Soccer) model constants — watchlist only. Active February–November.

Uses a Poisson goal-scoring model with xG data from the ASA API.
"""

MLS_LEAGUE_HOME_XG    = 1.68    # league baseline home xG/game (was 1.35 — too low vs
                                # actual ~1.68). FALLBACK only; live league averages are
                                # computed from ASA game data each run.
MLS_LEAGUE_AWAY_XG    = 1.29    # league baseline away xG/game (was 1.15 — too low vs
                                # actual ~1.29). Fallback only.
MLS_RECENT_WEIGHT     = 0.40    # 40% recent form, 60% season
MLS_MIN_HOME_GAMES    = 5       # min home games to use team-specific home advantage
MLS_HOME_ADV_DEFAULT  = 0.02    # league-wide home win prob boost (2%)
MLS_STRONG_VENUE_ADV  = 0.03    # fortress venue boost (3%)
MLS_MAX_INJURY_PENALTY = 0.15   # max 15% lambda reduction from injuries per team
MLS_DC_RHO            = -0.10    # Dixon-Coles low-score correlation. Independent Poisson
                                 # under-counts draws (~23.5% vs actual ~25.5%); this
                                 # adjusts the 0-0/1-0/0-1/1-1 cells. Calibrated to 2024
                                 # MLS: rho=-0.10 → model draw 25.6% vs actual 25.5%.

# Teams with notably strong home atmospheres
MLS_STRONG_VENUES = {
    "Seattle Sounders", "Portland Timbers", "Atlanta United",
    "FC Cincinnati", "Columbus Crew", "LAFC", "LA Galaxy",
    "Sporting Kansas City", "Toronto FC",
}
