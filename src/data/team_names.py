"""Canonical team-name normalisation shared by the Kalshi resolver and the
outcome settlers.

WHY THIS EXISTS
The odds feed is not consistent about city names: it emits "NY Jets" and
"NY Giants" while both our Kalshi team map and ESPN say "New York Jets" /
"New York Giants". That single difference broke two unrelated things at once,
in ways that looked nothing alike:

  * Kalshi CLV — `_team_token("NY Jets")` returned None, so the game was never
    found and the pick got no CLV. Worse, it produced NO warning either,
    because an unmapped team empties the market pool before any ladder is read,
    which is the path that reports "not buyable".
  * Settlement — `_determine_outcome("NY Jets", ..., away="New York Jets")`
    returned UNKNOWN, so those picks stayed permanently unsettled.

Both were silent. Keeping the expansion in ONE place is the point: fixing it in
the resolver alone would have left the settler broken, and the two failures
give no hint they share a cause.
"""
from __future__ import annotations

# Feed shorthand -> the full city name our maps and ESPN both use.
# Leading-token only, so a club legitimately containing one of these tokens
# later in its name is untouched.
CITY_ABBREV = {
    "NY": "New York",
    "LA": "Los Angeles",
    "SF": "San Francisco",
    "KC": "Kansas City",
    "TB": "Tampa Bay",
    "NE": "New England",
    "GB": "Green Bay",
    "NO": "New Orleans",
    "SD": "San Diego",
}


def expand_city(name: str) -> str:
    """"NY Jets" -> "New York Jets". Returns `name` unchanged when nothing applies."""
    # Always return a STRING. _determine_outcome calls .lower() on the result,
    # so leaking a None through here would turn a missing team name into an
    # AttributeError mid-settlement.
    text = str(name or "")
    parts = text.split()
    if len(parts) >= 2 and parts[0].upper() in CITY_ABBREV:
        return " ".join([CITY_ABBREV[parts[0].upper()]] + parts[1:])
    return text
