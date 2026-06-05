"""
FIFA World Cup 2026 model constants — watchlist only.

Tournament runs June 11 – July 19, 2026 across the USA, Canada and Mexico.

Unlike MLS (which has rich club-level xG from the ASA API), international
national teams have almost no usable shared-competition form during the group
stage. So the World Cup model is driven by an **Elo strength rating** instead:

    seed Elo (src/data/wc_elo_seed.json)
      → self-updated from results as the tournament progresses (state/wc_elo.json)
      → Elo supremacy → Poisson goal expectations (λ_home, λ_away)
      → Dixon-Coles scoreline grid → 3-way / total / spread probabilities.

Elo is purpose-built for thin-data settings, so it is the right tool here.
"""

# Odds API sport key for the FIFA World Cup.
WC_SPORT = "soccer_fifa_world_cup"

# ── Elo → goals mapping ──────────────────────────────────────────────────────
WC_ELO_DEFAULT       = 1620.0   # fallback rating for any team not in the seed table
WC_BASE_TOTAL        = 2.55     # league-average goals/game for a neutral, even WC match
WC_ELO_PER_GOAL      = 165.0    # Elo points of supremacy ≈ one goal of expected margin
WC_MAX_SUPREMACY     = 3.0      # cap on |expected goal margin| from Elo diff
WC_DC_RHO            = -0.10    # Dixon-Coles low-score correlation (same as MLS — soccer)

# ── Home / host advantage ────────────────────────────────────────────────────
# World Cup games are at neutral venues, so the odds feed's home/away designation
# is usually arbitrary. We grant a real home edge ONLY to the three host nations
# (when listed as the home side), expressed in Elo points.
WC_HOST_NATIONS      = {"United States", "USA", "Mexico", "Canada"}
WC_HOST_ELO_BONUS    = 60.0     # Elo points added to a host nation playing at home
WC_NEUTRAL_ELO_BONUS = 0.0      # no home edge for ordinary neutral-venue matches

# ── Dynamic Elo update (self-learning through the tournament) ─────────────────
WC_ELO_K             = 40.0     # K-factor for World Cup matches (high-importance)
WC_ELO_GOAL_MULT     = True     # scale K by goal-difference (eloratings.net style)

# ── Edge-finding safety ──────────────────────────────────────────────────────
WC_CRED_CAP          = 0.10     # max model-vs-market divergence before the credibility
                                # cap pulls the model back (auto-relaxed by calibration)
