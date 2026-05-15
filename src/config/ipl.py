"""
IPL (Indian Premier League cricket) model constants — watchlist only.

T20 home advantage is substantial; ~5% raw but market prices most of it.
Recent form dominates in a short tournament (10 teams, ~14 matches each).
"""

IPL_HOME_ADV      = 0.025   # ~2.5% residual home advantage beyond market pricing
IPL_RECENT_WEIGHT = 0.65    # T20 form is highly volatile — weight recent heavily
