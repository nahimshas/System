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

# ── Rest / fatigue differential ──────────────────────────────────────────────
# 2026 spans a huge geography with uneven rest between matchdays. The better-
# rested side gets a small Elo nudge proportional to the rest-day differential.
WC_REST_ELO_PER_DAY  = 10.0     # Elo per extra day of rest vs the opponent
WC_MAX_REST_ELO      = 40.0     # cap on the rest adjustment (≈ ¼ goal)

# ── Dead-rubber / match-stakes damping ───────────────────────────────────────
# 2026 format = 12 groups of 4, top 2 + 8 best third-placed teams advance, so
# almost nobody is mathematically safe/out after 2 games. We therefore damp ONLY
# the clearly-safe case: a team on ≥6 points heading into its 3rd group game may
# rotate its squad. Applied only during the group stage.
WC_DEAD_RUBBER_MIN_PTS  = 6     # points after 2 games that imply likely qualification
WC_DEAD_RUBBER_ELO_DAMP = 60.0  # Elo removed from a likely-qualified (rotating) side
WC_GROUP_STAGE_END      = "2026-06-27"  # last group-stage date (ISO); dead-rubber only before this

# ── Low-confidence shrinkage ─────────────────────────────────────────────────
# When a side's Elo is a guess (unseeded minnow → WC_ELO_DEFAULT), pull the model
# harder toward the market by tightening the credibility cap, so we don't make
# overconfident picks on teams we barely know.
WC_LOWCONF_CAP_FACTOR = 0.5     # multiplies the credibility cap when a side is unseeded

# ── Venue: altitude / climate ────────────────────────────────────────────────
# Static per-venue effects on total goals (joined from wc_venue_config.json).
# High altitude → ball travels further + faster fatigue → slightly more goals.
# Hot/humid open-air → cagier, tired legs late → slightly fewer goals.
# Climate-controlled domes → neutral.
WC_ALT_HIGH_M        = 2000.0   # metres above which the altitude bump applies
WC_ALT_TOTAL_MULT    = 1.05     # total-goals multiplier at high altitude
WC_HOT_TOTAL_MULT    = 0.95     # total-goals multiplier for hot/humid open venues
