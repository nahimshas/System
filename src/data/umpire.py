"""
MLB umpire tendencies module.

Fetches today's home plate umpire per game from the MLB Stats API
(hydrate=officials), then looks up historical tendencies from a built-in
table. Unknown umpires fall back to league-average values.

Tendency fields:
  run_factor   float  – runs/game relative to league avg (1.0 = avg)
                        >1.0 → umpire calls a high-scoring game (leans Over)
                        <1.0 → umpire suppresses runs (leans Under)
  k_factor     float  – strikeout rate relative to league avg (1.0 = avg)
                        >1.0 → umpire has a large strike zone (pitcher-friendly)
  notes        str    – human-readable summary
"""
import logging
import requests
from datetime import date
from typing import Dict, Optional

logger = logging.getLogger(__name__)
MLB_API = "https://statsapi.mlb.com/api/v1"


# ---------------------------------------------------------------------------
# Umpire tendency lookup table
# Source: multi-season averages from publicly available umpire scorecards.
# run_factor  = (umpire avg runs/game) / (league avg runs/game ~9.0)
# k_factor    = (umpire avg Ks/game)   / (league avg Ks/game  ~17.0)
# ---------------------------------------------------------------------------
UMPIRE_TENDENCIES: Dict[str, Dict] = {
    # Pitcher-friendly / run-suppressing umpires
    "Laz Diaz":          {"run_factor": 0.94, "k_factor": 1.08, "notes": "Large strike zone, suppresses runs"},
    "Angel Hernandez":   {"run_factor": 0.96, "k_factor": 1.05, "notes": "Slightly pitcher-friendly"},
    "CB Bucknor":        {"run_factor": 0.95, "k_factor": 1.06, "notes": "Wide zone, low-scoring games"},
    "Phil Cuzzi":        {"run_factor": 0.96, "k_factor": 1.04, "notes": "Pitcher-friendly tendencies"},
    "Dan Iassogna":      {"run_factor": 0.95, "k_factor": 1.07, "notes": "Large zone, fewer walks"},
    "Joe West":          {"run_factor": 0.96, "k_factor": 1.05, "notes": "Veteran: pitcher-friendly"},
    "Bruce Dreckman":    {"run_factor": 0.95, "k_factor": 1.06, "notes": "Tight zone, low runs"},
    "Marvin Hudson":     {"run_factor": 0.96, "k_factor": 1.04, "notes": "Below-avg run environment"},
    "Jim Reynolds":      {"run_factor": 0.95, "k_factor": 1.05, "notes": "Consistent pitcher-friendly calls"},
    "Chad Fairchild":    {"run_factor": 0.96, "k_factor": 1.04, "notes": "Slightly below-avg scoring"},
    "Tripp Gibson":      {"run_factor": 0.95, "k_factor": 1.06, "notes": "Large zone, K-friendly"},
    "Hunter Wendelstedt":{"run_factor": 0.94, "k_factor": 1.07, "notes": "One of the most pitcher-friendly HPUs"},
    "Fieldin Culbreth":  {"run_factor": 0.95, "k_factor": 1.05, "notes": "Pitcher-friendly, low walk rates"},
    "Kerwin Danley":     {"run_factor": 0.96, "k_factor": 1.04, "notes": "Below-avg run environment"},

    # Hitter-friendly / high-scoring umpires
    "Ted Barrett":       {"run_factor": 1.06, "k_factor": 0.95, "notes": "Tight zone, high scoring"},
    "Jim Wolf":          {"run_factor": 1.05, "k_factor": 0.96, "notes": "Small strike zone, walks up"},
    "Vic Carapazza":     {"run_factor": 1.07, "k_factor": 0.94, "notes": "Hitter-friendly, leans Over"},
    "Stu Scheurwater":   {"run_factor": 1.06, "k_factor": 0.95, "notes": "Tight zone, high-run games"},
    "Mike Estabrook":    {"run_factor": 1.05, "k_factor": 0.96, "notes": "Small zone, hitter-friendly"},
    "Adam Hamari":       {"run_factor": 1.06, "k_factor": 0.95, "notes": "Above-avg scoring environment"},
    "Roberto Ortiz":     {"run_factor": 1.05, "k_factor": 0.96, "notes": "Hitter-friendly tendencies"},
    "Bill Miller":       {"run_factor": 1.04, "k_factor": 0.97, "notes": "Slightly above-avg scoring"},
    "Mike Muchlinski":   {"run_factor": 1.06, "k_factor": 0.94, "notes": "Tight zone, favors hitters"},
    "Chris Guccione":    {"run_factor": 1.05, "k_factor": 0.96, "notes": "Above-avg scoring games"},
    "Pat Hoberg":        {"run_factor": 1.04, "k_factor": 0.97, "notes": "Pitch tracking HPU, slightly hitter-friendly"},
    "Jeremie Rehak":     {"run_factor": 1.05, "k_factor": 0.96, "notes": "Hitter-friendly zone"},
    "Will Little":       {"run_factor": 1.05, "k_factor": 0.95, "notes": "Tight zone, high-scoring tendency"},
    "Manny Gonzalez":    {"run_factor": 1.06, "k_factor": 0.94, "notes": "One of the most hitter-friendly HPUs"},

    # Near-neutral umpires
    "Mark Carlson":      {"run_factor": 1.01, "k_factor": 1.00, "notes": "Neutral / league average"},
    "Brian Gorman":      {"run_factor": 0.99, "k_factor": 1.01, "notes": "Near-neutral"},
    "Toby Basner":       {"run_factor": 1.00, "k_factor": 1.00, "notes": "Neutral"},
    "Tom Hallion":       {"run_factor": 0.99, "k_factor": 1.01, "notes": "Slightly pitcher-friendly"},
    "Gerry Davis":       {"run_factor": 1.00, "k_factor": 1.00, "notes": "Neutral / league average"},
    "Greg Gibson":       {"run_factor": 0.98, "k_factor": 1.02, "notes": "Slightly pitcher-friendly"},
    "Mike Winters":      {"run_factor": 1.01, "k_factor": 0.99, "notes": "Near-neutral"},
    "John Hirschbeck":   {"run_factor": 1.00, "k_factor": 1.00, "notes": "Neutral"},
    "Larry Vanover":     {"run_factor": 1.01, "k_factor": 0.99, "notes": "Near-neutral"},
    "Alfonso Marquez":   {"run_factor": 1.00, "k_factor": 1.00, "notes": "Neutral"},
}

LEAGUE_AVG_TENDENCY = {"run_factor": 1.00, "k_factor": 1.00, "notes": "League average (umpire unknown)"}

# Thresholds for generating signals
RUN_FACTOR_SIGNAL_THRESHOLD = 0.03   # ≥3% deviation from average
K_FACTOR_SIGNAL_THRESHOLD   = 0.04   # ≥4% deviation from average


# ---------------------------------------------------------------------------
# API fetch
# ---------------------------------------------------------------------------

def get_home_plate_umpires(today: date) -> Dict[int, str]:
    """
    Returns {game_pk: umpire_full_name} for today's games.
    Uses MLB Stats API /schedule with hydrate=officials.
    """
    date_str = today.strftime("%Y-%m-%d")
    try:
        r = requests.get(
            f"{MLB_API}/schedule",
            params={"sportId": 1, "date": date_str, "hydrate": "officials"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
    except Exception as e:
        logger.warning(f"Umpire fetch failed ({date_str}): {e}")
        return {}

    result: Dict[int, str] = {}
    for date_entry in data.get("dates", []):
        for g in date_entry.get("games", []):
            game_pk = g.get("gamePk")
            if not game_pk:
                continue
            for official in g.get("officials", []):
                otype = official.get("officialType", "")
                if otype.lower() in ("home plate", "hp", "home_plate"):
                    name = official.get("official", {}).get("fullName", "")
                    if name:
                        result[game_pk] = name
                        break

    logger.info(f"Umpire data: {len(result)} home plate umpire(s) found for {date_str}")
    return result


# ---------------------------------------------------------------------------
# Tendency lookup
# ---------------------------------------------------------------------------

def get_umpire_tendency(umpire_name: str) -> Dict:
    """
    Returns tendency dict for the given umpire.
    Falls back to league average for unknown umpires.
    """
    if not umpire_name:
        return LEAGUE_AVG_TENDENCY.copy()

    # Exact match first
    if umpire_name in UMPIRE_TENDENCIES:
        return UMPIRE_TENDENCIES[umpire_name].copy()

    # Partial / last-name match
    name_lower = umpire_name.lower()
    for key, val in UMPIRE_TENDENCIES.items():
        if name_lower in key.lower() or key.lower() in name_lower:
            return val.copy()

    logger.debug(f"Umpire '{umpire_name}' not in table — using league averages")
    return {**LEAGUE_AVG_TENDENCY.copy(), "notes": f"{umpire_name} — no historical data, using avg"}


def build_umpire_signals(umpire_name: str, tendency: Dict) -> list:
    """
    Returns a list of signal strings to add to the bet card.
    Only generates signals when the umpire deviates meaningfully from average.
    """
    if not umpire_name:
        return []

    signals = []
    rf = tendency.get("run_factor", 1.0)
    kf = tendency.get("k_factor", 1.0)
    notes = tendency.get("notes", "")

    if abs(rf - 1.0) >= RUN_FACTOR_SIGNAL_THRESHOLD:
        direction = "high-scoring" if rf > 1.0 else "low-scoring"
        lean = "leans Over" if rf > 1.0 else "leans Under"
        signals.append(
            f"👨‍⚖️ Umpire {umpire_name}: {direction} games "
            f"(run factor {rf:.2f}x avg) — {lean}"
        )

    if abs(kf - 1.0) >= K_FACTOR_SIGNAL_THRESHOLD:
        zone = "large strike zone" if kf > 1.0 else "tight strike zone"
        signals.append(
            f"👨‍⚖️ {umpire_name}: {zone} (K factor {kf:.2f}x avg) — "
            + ("favors pitchers / strikeout props" if kf > 1.0 else "favors hitters / walks up")
        )

    return signals
