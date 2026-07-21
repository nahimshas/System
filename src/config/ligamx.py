"""
Liga MX (Mexican Primera División) model constants — watchlist only.

Robinhood added Liga MX in July 2026 with three bet types: Win / Tie /
Don't-Win. All three derive from a single 3-way match probability
(home-win / draw / away-win), so the model only needs to produce those.

Like the World Cup — and UNLIKE MLS — there is no free club-level xG feed for
Liga MX (ASA is US-soccer only). So the model is Elo-driven:

    Elo ratings, bootstrapped by replaying ~1 year of ESPN Liga MX results
    from a neutral baseline, then self-updated from new results
      → Elo supremacy → Poisson goal expectations (λ_home, λ_away)
      → Dixon-Coles scoreline grid → 3-way (Win / Tie / Don't-Win) probabilities.

Elo is purpose-built for thin-data / cold-start settings (a season restarts
every ~6 months with the Apertura/Clausura split), which is exactly why it is
the right backbone here rather than a goals-Poisson model that needs several
games before its strength estimates stabilise.
"""

# Odds API sport key for Liga MX.
LIGAMX_SPORT = "soccer_mexico_ligamx"

# ── Elo → goals mapping ──────────────────────────────────────────────────────
LIGAMX_ELO_DEFAULT   = 1500.0   # neutral baseline; every team starts here, ratings
                                # differentiate as results are replayed in bootstrap
LIGAMX_BASE_TOTAL    = 2.65     # Liga MX league-average goals/game (higher-scoring
                                # than MLS/WC; refined from data after launch)
LIGAMX_ELO_PER_GOAL  = 150.0    # Elo points of supremacy ≈ one goal of expected margin
LIGAMX_MAX_SUPREMACY = 3.0      # cap on |expected goal margin| from Elo diff
LIGAMX_DC_RHO        = -0.10    # Dixon-Coles low-score correlation (same as MLS/WC)

# ── Home advantage ───────────────────────────────────────────────────────────
# Unlike the World Cup (neutral venues), Liga MX plays at real home stadiums, so
# the home side always gets an edge. Expressed in Elo points. Mexican home
# advantage is historically strong (altitude, travel, crowd).
LIGAMX_HOME_ELO_BONUS = 70.0

# ── Dynamic Elo update (self-learning) ───────────────────────────────────────
LIGAMX_ELO_K          = 24.0    # K-factor for club-league matches (lower than WC's
                                # 40 — club games are less high-variance than a WC)
LIGAMX_ELO_GOAL_MULT  = True    # scale K by goal-difference (eloratings.net style)

# ── Bootstrap ────────────────────────────────────────────────────────────────
LIGAMX_BOOTSTRAP_DAYS = 365     # how far back to replay results on first-ever run
                                # (≈ one full calendar year = Apertura + Clausura)

# ── Edge-finding safety ──────────────────────────────────────────────────────
LIGAMX_CRED_CAP        = 0.10   # max drift of model prob from market (watchlist noise)
LIGAMX_LOWCONF_CAP_FACTOR = 0.5  # tighten the cap when a side's Elo is a fallback guess
