"""
MLB weather module — uses wttr.in (free, no API key required).

For each MLB game, looks up the venue city and fetches current/forecast
conditions. Returns temperature, wind (speed + direction), and precipitation
probability so the edge finder can adjust expected run totals.

Wind direction conventions:
  "out"  – blowing toward the outfield (increases HR / run scoring)
  "in"   – blowing in from the outfield (suppresses HR)
  "cross"– cross wind (minimal effect on scoring)

Temperature effect:
  < 50°F  → ball doesn't carry as well, ~0.4 fewer expected runs
  50–65°F → slight suppression (~0.2 runs)
  > 85°F  → ball carries well, slight boost (~0.1 runs)
"""
import logging
import requests
from typing import Dict, Optional

logger = logging.getLogger(__name__)

WTTR_URL = "https://wttr.in/{city}?format=j1"

# ---------------------------------------------------------------------------
# Venue → city mapping  (covers all 30 MLB stadiums)
# ---------------------------------------------------------------------------
VENUE_CITY: Dict[str, str] = {
    # AL East
    "Fenway Park":                    "Boston",
    "Yankee Stadium":                 "New York",
    "Camden Yards":                   "Baltimore",
    "Tropicana Field":                "St. Petersburg",
    "Rogers Centre":                  "Toronto",
    # AL Central
    "Guaranteed Rate Field":          "Chicago",
    "Progressive Field":              "Cleveland",
    "Comerica Park":                  "Detroit",
    "Kauffman Stadium":               "Kansas City",
    "Target Field":                   "Minneapolis",
    # AL West
    "Minute Maid Park":               "Houston",
    "Angel Stadium":                  "Anaheim",
    "Oakland Coliseum":               "Oakland",
    "T-Mobile Park":                  "Seattle",
    "Globe Life Field":               "Arlington",
    # NL East
    "Truist Park":                    "Atlanta",
    "Marlins Park":                   "Miami",
    "LoanDepot Park":                 "Miami",
    "Citi Field":                     "New York",
    "Citizens Bank Park":             "Philadelphia",
    "Nationals Park":                 "Washington",
    # NL Central
    "Wrigley Field":                  "Chicago",
    "Great American Ball Park":       "Cincinnati",
    "American Family Field":          "Milwaukee",
    "Busch Stadium":                  "St. Louis",
    "PNC Park":                       "Pittsburgh",
    # NL West
    "Chase Field":                    "Phoenix",
    "Coors Field":                    "Denver",
    "Dodger Stadium":                 "Los Angeles",
    "Petco Park":                     "San Diego",
    "Oracle Park":                    "San Francisco",
}

# Outfield orientation per stadium (compass bearing to center field).
# Wind blowing FROM that direction → blowing "in" (suppresses HR).
# Wind blowing TOWARD that direction → blowing "out" (boosts HR).
# Bearing = degrees FROM North, clockwise.
VENUE_CF_BEARING: Dict[str, int] = {
    "Fenway Park":              90,   # CF roughly east
    "Yankee Stadium":           90,
    "Wrigley Field":            75,   # famous Lake Michigan wind
    "Coors Field":              45,
    "Oracle Park":              270,  # McCovey Cove in left, CF west
    "Petco Park":               315,
    "Dodger Stadium":           90,
    "Minute Maid Park":         0,    # retractable roof — wind less relevant
    "Globe Life Field":         0,    # retractable roof
    "Chase Field":              0,    # retractable roof
    "Truist Park":              90,
    "Citi Field":               90,
    "Citizens Bank Park":       90,
    "Great American Ball Park": 180,
    "Busch Stadium":            270,
    "PNC Park":                 315,
    "Nationals Park":           180,
    "American Family Field":    0,    # retractable roof
    "Target Field":             270,
    "Kauffman Stadium":         180,
    "Comerica Park":            270,
    "Progressive Field":        90,
    "Camden Yards":             90,
    "Rogers Centre":            0,    # indoor retractable
    "Tropicana Field":          0,    # domed
    "T-Mobile Park":            0,    # retractable roof
    "Oakland Coliseum":         270,
    "Angel Stadium":            180,
}

# Stadiums with retractable roofs / domes — weather has minimal effect
INDOOR_VENUES = {
    "Tropicana Field", "Rogers Centre",
    "Minute Maid Park",   # retractable — check if open, but default to neutral
    "Globe Life Field",
    "Chase Field",
    "American Family Field",
    "T-Mobile Park",
}

# Thresholds for generating signals
WIND_SIGNAL_MPH     = 10    # wind speed at or above which we generate a signal
WIND_STRONG_MPH     = 15    # strong wind — larger run adjustment
COLD_TEMP_F         = 50    # below this → significant run suppression
COOL_TEMP_F         = 62    # below this → mild suppression
HOT_TEMP_F          = 85    # above this → mild boost
PRECIP_SIGNAL_PCT   = 40    # precipitation probability above which we warn


# ---------------------------------------------------------------------------
# Wind direction helpers
# ---------------------------------------------------------------------------

def _wind_to_bearing(wind_dir_str: str) -> Optional[int]:
    """Convert wttr.in winddir16Point (e.g. 'NW', 'SSE') to degrees."""
    compass = {
        "N": 0, "NNE": 22, "NE": 45, "ENE": 67,
        "E": 90, "ESE": 112, "SE": 135, "SSE": 157,
        "S": 180, "SSW": 202, "SW": 225, "WSW": 247,
        "W": 270, "WNW": 292, "NW": 315, "NNW": 337,
    }
    return compass.get(wind_dir_str.upper())


def _wind_effect(wind_bearing: int, cf_bearing: int) -> str:
    """
    Returns 'out', 'in', or 'cross' based on angular difference between
    wind direction (where wind is blowing TO) and center field bearing.
    """
    # wind_bearing = direction wind is blowing FROM; convert to blowing TO
    blowing_to = (wind_bearing + 180) % 360
    diff = abs(blowing_to - cf_bearing) % 360
    if diff > 180:
        diff = 360 - diff
    if diff <= 45:
        return "out"
    if diff >= 135:
        return "in"
    return "cross"


# ---------------------------------------------------------------------------
# wttr.in fetch
# ---------------------------------------------------------------------------

def _fetch_weather(city: str) -> Optional[Dict]:
    """Raw wttr.in JSON fetch. Returns None on failure."""
    try:
        r = requests.get(
            WTTR_URL.format(city=city.replace(" ", "+")),
            timeout=10,
            headers={"Accept": "application/json"},
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning(f"Weather fetch failed for {city}: {e}")
        return None


def _parse_wttr(data: Dict, city: str, game_hour_local: int = 19) -> Dict:
    """
    Extract relevant fields from wttr.in JSON response.
    game_hour_local: local hour (0-23) the game starts — used to pick the
    closest hourly forecast block instead of always defaulting to noon.
    wttr.in returns blocks at 0, 300, 600, 900, 1200, 1500, 1800, 2100.
    """
    try:
        current = data.get("current_condition", [{}])[0]
        temp_f    = int(current.get("temp_F", 70))
        wind_mph  = int(current.get("windspeedMiles", 0))
        wind_dir  = current.get("winddir16Point", "N")
        today_wx  = data.get("weather", [{}])[0]
        hourly    = today_wx.get("hourly", [{}])
        # Pick the hourly block whose time_val is closest to game_hour_local
        # wttr blocks have "time" = "0","300","600",...,"2100"
        def _block_hour(blk):
            try:
                return int(blk.get("time", "0")) // 100
            except (ValueError, TypeError):
                return 0
        game_block = min(hourly, key=lambda b: abs(_block_hour(b) - game_hour_local)) if hourly else {}
        precip_pct = int(game_block.get("chanceofrain", 0))
        precip_pct = max(precip_pct, int(game_block.get("chanceofthunder", 0)))
        desc = current.get("weatherDesc", [{}])
        conditions = desc[0].get("value", "") if desc else ""
        return {
            "city":        city,
            "temp_f":      temp_f,
            "wind_mph":    wind_mph,
            "wind_dir":    wind_dir,
            "precip_pct":  precip_pct,
            "conditions":  conditions,
        }
    except Exception as e:
        logger.warning(f"Weather parse error for {city}: {e}")
        return {"city": city, "temp_f": 70, "wind_mph": 0, "wind_dir": "N",
                "precip_pct": 0, "conditions": ""}


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

def get_game_weather(venue: str, commence_time_utc: str = "") -> Dict:
    """
    Returns weather dict for the given venue, or neutral defaults on failure.
    Keys: temp_f, wind_mph, wind_dir, wind_effect, precip_pct, conditions, indoor

    commence_time_utc: ISO-8601 UTC string (e.g. "2026-04-27T23:10:00Z").
    When provided, the hourly forecast block nearest to local game time is
    used instead of always defaulting to noon.
    """
    neutral = {
        "temp_f": 70, "wind_mph": 0, "wind_dir": "N", "wind_effect": "cross",
        "precip_pct": 0, "conditions": "", "indoor": False, "venue": venue,
    }

    # Indoor/dome venues — weather irrelevant
    for indoor_name in INDOOR_VENUES:
        if indoor_name.lower() in venue.lower() or venue.lower() in indoor_name.lower():
            neutral["indoor"] = True
            return neutral

    # Find city
    city = None
    for vname, vcity in VENUE_CITY.items():
        if vname.lower() in venue.lower() or venue.lower() in vname.lower():
            city = vcity
            break
    if not city:
        # Last-ditch: try the raw venue string (works for "<City> Stadium" patterns)
        city = venue.split()[0] if venue else None
    if not city:
        logger.debug(f"No city mapping for venue '{venue}' — using neutral weather")
        return neutral

    raw = _fetch_weather(city)
    if not raw:
        return neutral

    # Derive local game hour from UTC commence time (rough: UTC - venue offset)
    game_hour_local = 19  # default evening game if no time available
    if commence_time_utc:
        try:
            from datetime import datetime, timezone
            import re
            # Parse ISO-8601 (handles trailing Z or +00:00)
            ct = commence_time_utc.rstrip("Z").split("+")[0]
            utc_dt = datetime.strptime(ct, "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc)
            utc_hour = utc_dt.hour
            # Approximate UTC→local offset by city (good enough for forecast slot selection)
            TZ_OFFSET = {
                "Chicago": -5, "New York": -4, "Boston": -4, "Philadelphia": -4,
                "Baltimore": -4, "Washington": -4, "Cleveland": -4, "Detroit": -4,
                "Toronto": -4, "Atlanta": -4, "Miami": -4, "Pittsburgh": -4,
                "St. Louis": -5, "Cincinnati": -4, "Milwaukee": -5, "Minneapolis": -5,
                "Kansas City": -5, "Houston": -5, "Arlington": -5, "Denver": -6,
                "Phoenix": -7, "Los Angeles": -7, "San Diego": -7,
                "San Francisco": -7, "Seattle": -7, "Oakland": -7,
            }
            offset = TZ_OFFSET.get(city, -5)
            game_hour_local = (utc_hour + offset) % 24
        except Exception:
            pass

    wx = _parse_wttr(raw, city, game_hour_local)

    # Determine wind effect relative to this ballpark's CF orientation
    wind_bearing = _wind_to_bearing(wx["wind_dir"])
    cf_bearing   = None
    for vname, bearing in VENUE_CF_BEARING.items():
        if vname.lower() in venue.lower() or venue.lower() in vname.lower():
            cf_bearing = bearing
            break

    if wind_bearing is not None and cf_bearing is not None and cf_bearing != 0:
        effect = _wind_effect(wind_bearing, cf_bearing)
    else:
        effect = "cross"   # retractable-roof venues or unknown orientation

    return {
        "venue":       venue,
        "city":        city,
        "temp_f":      wx["temp_f"],
        "wind_mph":    wx["wind_mph"],
        "wind_dir":    wx["wind_dir"],
        "wind_effect": effect,
        "precip_pct":  wx["precip_pct"],
        "conditions":  wx["conditions"],
        "indoor":      False,
    }


# ---------------------------------------------------------------------------
# Run adjustments + signals
# ---------------------------------------------------------------------------

def weather_run_adjustment(wx: Dict) -> float:
    """
    Returns expected run total adjustment (positive = more runs, negative = fewer).
    Applied to the combined expected total for both teams.
    """
    if wx.get("indoor"):
        return 0.0

    adj = 0.0
    wind_mph   = wx.get("wind_mph", 0)
    wind_effect = wx.get("wind_effect", "cross")
    temp_f     = wx.get("temp_f", 70)

    # Wind effect
    if wind_mph >= WIND_STRONG_MPH:
        if wind_effect == "out":
            adj += 0.8
        elif wind_effect == "in":
            adj -= 0.6
        # cross: no significant effect
    elif wind_mph >= WIND_SIGNAL_MPH:
        if wind_effect == "out":
            adj += 0.4
        elif wind_effect == "in":
            adj -= 0.3

    # Temperature effect
    if temp_f < COLD_TEMP_F:
        adj -= 0.5
    elif temp_f < COOL_TEMP_F:
        adj -= 0.2
    elif temp_f > HOT_TEMP_F:
        adj += 0.15

    return round(adj, 2)


def build_weather_signals(wx: Dict) -> list:
    """
    Returns signal strings for the bet card.
    Only generates signals when conditions meaningfully affect scoring.
    """
    if wx.get("indoor"):
        return []

    signals = []
    wind_mph    = wx.get("wind_mph", 0)
    wind_effect = wx.get("wind_effect", "cross")
    wind_dir    = wx.get("wind_dir", "")
    temp_f      = wx.get("temp_f", 70)
    precip_pct  = wx.get("precip_pct", 0)
    conditions  = wx.get("conditions", "")
    city        = wx.get("city", wx.get("venue", ""))

    if wind_mph >= WIND_SIGNAL_MPH:
        strength = "strong" if wind_mph >= WIND_STRONG_MPH else "moderate"
        if wind_effect == "out":
            signals.append(
                f"🌬️ Wind {wind_mph} mph blowing OUT ({wind_dir}) at {city} — "
                f"ball carries well, {strength} Over lean (+{0.8 if wind_mph >= WIND_STRONG_MPH else 0.4:.1f} runs)"
            )
        elif wind_effect == "in":
            signals.append(
                f"🌬️ Wind {wind_mph} mph blowing IN ({wind_dir}) at {city} — "
                f"suppresses HR, {strength} Under lean ({-0.6 if wind_mph >= WIND_STRONG_MPH else -0.3:.1f} runs)"
            )
        else:
            signals.append(
                f"🌬️ Cross wind {wind_mph} mph ({wind_dir}) at {city} — minimal scoring impact"
            )

    if temp_f < COLD_TEMP_F:
        signals.append(f"🌡️ Cold ({temp_f}°F) at {city} — ball doesn't carry, Under lean (−0.5 runs)")
    elif temp_f < COOL_TEMP_F:
        signals.append(f"🌡️ Cool ({temp_f}°F) at {city} — slight run suppression (−0.2 runs)")
    elif temp_f > HOT_TEMP_F:
        signals.append(f"🌡️ Hot ({temp_f}°F) at {city} — ball carries well, slight Over lean (+0.15 runs)")

    if precip_pct >= PRECIP_SIGNAL_PCT:
        signals.append(
            f"🌧️ {precip_pct}% precipitation chance at {city} "
            f"({'rain likely' if precip_pct >= 70 else 'rain possible'}) — "
            "monitor for delays/postponement"
        )

    return signals
