"""
College football (FBS) model constants — WATCHLIST ONLY.

Why Elo → margin, and not a stats model like MLB's:

  CFB has ~134 FBS teams playing only ~12 games each, with enormous talent
  dispersion (spreads reach -45; an MLB game is close to a coin flip). Per-game
  stat aggregation needs a sample no CFB team ever accumulates. A power rating
  that converts directly to an expected POINT MARGIN is the right shape, and
  Elo is purpose-built for the thin-data / cold-start case — which CFB is in
  permanently, not just in September.

  margin = (elo_home - elo_away) / ELO_TO_POINTS + HOME_ADV_POINTS
  P(home covers L) = Phi((margin + L) / MARGIN_STD)
  P(home wins)     = Phi(margin / MARGIN_STD)

TOTALS ARE DELIBERATELY NOT MODELLED. They need a separate pace x efficiency
model, and they are the worst-priced market on Kalshi (median open interest 0,
18-cent spreads, measured Aug 29 2026). Adding them would manufacture edges we
could never act on.
"""

# ── Elo ────────────────────────────────────────────────────────────────────
CFB_ELO_DEFAULT: float = 1500.0
# K is high relative to soccer (~20) because a CFB team plays ~12 games a year:
# ratings must move meaningfully on each result or they never converge.
CFB_ELO_K: float = 42.0
# ~2.6 points of home field, the long-run FBS average. Applied inside the Elo
# UPDATE so home wins are discounted appropriately.
CFB_HOME_ELO_BONUS: float = 65.0

# ── Rating → points ────────────────────────────────────────────────────────
# 25 Elo points ≈ 1 point of spread is the standard football conversion.
CFB_ELO_TO_POINTS: float = 25.0
# Home-field advantage in POINTS, used when predicting a margin. Separate from
# the Elo bonus above, which only shapes rating updates.
CFB_HOME_ADV_POINTS: float = 2.6
# Std-dev of actual margin around the predicted margin. CFB is noisier than the
# NFL (~13.5) because talent gaps are wider and garbage time is longer.
CFB_MARGIN_STD: float = 16.5

# ── Cold start ─────────────────────────────────────────────────────────────
# CFB rosters turn over far harder than the NFL's, so last season's rating is a
# weaker prior. Regress toward the mean at each season boundary: a team keeps
# this share of its distance from average.
CFB_PRIOR_REGRESSION: float = 0.60
# Games into the new season before the prior stops being blended out.
# Must exceed CFB_MIN_RATED_GAMES, or the ramp completes exactly when betting
# becomes allowed and never damps a single live pick.
CFB_WARMSTART_RAMP_GAMES: int = 6

# ── Safety ─────────────────────────────────────────────────────────────────
# Credibility cap: how far the model may disagree with the market. Tighter than
# MLB's 0.10 because early-season CFB ratings are the least trustworthy inputs
# anywhere in the system.
CFB_CRED_CAP: float = 0.08
# Ignore FCS/non-FBS opponents — their ratings are meaningless and the games are
# unpriced blowouts.
CFB_MIN_ELO_GAMES: int = 3


# ── Sanity gate (added Sep 3 2026 after the model shipped garbage) ─────────
# The market's spread IS its estimate of the margin. If our projected margin
# disagrees with it by more than this, we are not finding an edge — we are
# failing to understand the game, and the credibility cap will silently clamp
# the result to exactly the cap on every pick.
#
# WHAT HAPPENED: week-1 ratings spanned only 372 Elo (max expressible margin
# 14.9 points) while real CFB lines reach 45. Every pick came out at EXACTLY
# +8.0% edge — the cap value — on +26.5 to +42.5 underdogs the model believed
# were near-even. A cap that fires on 100% of picks is not a safety net, it is
# a symptom.
CFB_MAX_MARGIN_DISAGREE: float = 10.0

# Minimum rated games (this season or prior) before a team's rating is trusted
# enough to bet against a market line at all.
CFB_MIN_RATED_GAMES: int = 4
