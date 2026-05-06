"""
Player prop analysis powered by Odds API market lines.
Each prop generates a model projection, compares to the actual book line,
and calculates edge% exactly like the main picks model.

Distributions:
  Normal  — Points, Rebounds, Assists, Threes, Strikeouts, Hits, Total Bases, HRR
  Poisson — Steals, Blocks, Home Runs (discrete low-count events)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from scipy.stats import norm as _norm, poisson as _poisson

from src.models.edge_finder import _is_nba_playoff, _is_mlb_playoff
from src.config import MLB_PLAYOFF_STARTER_IP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class PropPick:
    sport:         str
    player:        str
    team:          str
    opponent:      str
    prop_type:     str
    model_line:    float   # model's projected stat
    market_line:   float   # Odds API line (e.g. 22.5)
    market_prob:   float   # no-vig consensus P(over)
    model_prob:    float   # model P(over market_line)
    edge:          float   # model_prob - market_prob
    odds_display:  str     # with-vig implied % from preferred book
    book:          str     # e.g. "draftkings"
    confidence:    str     # "HIGH" or "MEDIUM"
    note:          str
    model_margin:  float       = 0.0
    signals:       List[str]   = field(default_factory=list)
    research:      List[str]   = field(default_factory=list)
    commence_time: str         = ""


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

_POISSON_PROPS = {"Steals Over", "Blocks Over", "Home Runs Over"}

_SIGMA_COEF = {
    "Points Over":      0.38,
    "Rebounds Over":    0.45,
    "Assists Over":     0.45,
    "Threes Over":      0.70,
    "Strikeouts Over":  0.35,
    "Hits Over":        0.55,
    "Total Bases Over": 0.55,
    "HRR Over":         0.45,
}
_SIGMA_MIN = {
    "Points Over":      4.0,
    "Rebounds Over":    2.0,
    "Assists Over":     1.5,
    "Threes Over":      0.8,
    "Strikeouts Over":  1.5,
    "Hits Over":        0.5,
    "Total Bases Over": 0.8,
    "HRR Over":         1.0,
}

HIGH_PROP_EDGE = 0.08
MIN_PROP_EDGE  = 0.04


def _cover_prob(prop_type: str, model_line: float, market_line: float) -> float:
    """P(actual stat > market_line) given model projects model_line."""
    if prop_type in _POISSON_PROPS:
        k   = int(market_line)       # Over 0.5 → k=0, Over 1.5 → k=1
        lam = max(0.01, model_line)
        return float(1 - _poisson.cdf(k, lam))
    coef  = _SIGMA_COEF.get(prop_type, 0.40)
    sigma = max(_SIGMA_MIN.get(prop_type, 2.0), model_line * coef)
    return float(1 - _norm.cdf(market_line, model_line, sigma))


def _confidence(edge: float, model_line: float, market_line: float) -> str:
    if edge >= HIGH_PROP_EDGE and model_line > market_line:
        return "HIGH"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# NBA stat projections
# ---------------------------------------------------------------------------

_NBA_LEAGUE_AVG = 112.0
_NBA_B2B_FACTOR = 0.92


def _project_nba_stat(
    prop_type: str,
    pstats:    Dict,
    team_stats: Dict,
    opp_stats:  Dict,
    is_b2b:    bool,
    playoff:   bool,
) -> Optional[float]:
    if prop_type == "Points Over":
        base = pstats.get("pts", 0.0)
        min_base = 12.0 if playoff else 5.0
        if base < min_base:
            return None
        opp_def = opp_stats.get("def_rtg", _NBA_LEAGUE_AVG)
        adj = base * (opp_def / _NBA_LEAGUE_AVG)
        if playoff:
            if base >= 23:
                adj *= 1.05    # stars: elevated usage in playoffs
            elif base >= 15:
                adj *= 0.96    # starters: slight reduction (tighter defense)
            else:
                adj *= 0.88    # role players: significant reduction
        if is_b2b:
            adj *= _NBA_B2B_FACTOR
        return round(adj, 1)

    if prop_type == "Rebounds Over":
        base = pstats.get("reb", 0.0)
        min_base = 5.0 if playoff else 2.0
        if base < min_base:
            return None
        if playoff:
            if base >= 10:
                adj = base * 1.03  # elite rebounders: slightly more in playoff battle
            elif base >= 7:
                adj = base * 0.97  # solid rebounders: minor reduction
            else:
                adj = base * 0.90  # role players: reduced minutes/opportunities
        else:
            adj = base
        return round(adj * (0.95 if is_b2b else 1.0), 1)

    if prop_type == "Assists Over":
        base = pstats.get("ast", 0.0)
        min_base = 4.0 if playoff else 1.0
        if base < min_base:
            return None
        pace = (team_stats.get("pace", 100) + opp_stats.get("pace", 100)) / 2
        adj = base * (pace / 100.0)
        if playoff:
            if base >= 7:
                adj *= 1.03   # elite playmakers: usage stays high
            elif base >= 5:
                adj *= 0.96   # solid playmakers: slightly tighter playoff defense
            else:
                adj *= 0.90   # marginal passers: fewer opportunities
        return round(adj * (0.95 if is_b2b else 1.0), 1)

    if prop_type == "Steals Over":
        base = pstats.get("stl", 0.0)
        min_base = 0.5 if playoff else 0.3
        if base < min_base:
            return None
        return round(base, 2)

    if prop_type == "Blocks Over":
        base = pstats.get("blk", 0.0)
        min_base = 0.5 if playoff else 0.3
        if base < min_base:
            return None
        return round(base, 2)

    if prop_type == "Threes Over":
        base = pstats.get("three_pm", 0.0)
        min_base = 1.5 if playoff else 0.5
        if base < min_base:
            return None
        if playoff:
            if base >= 3.0:
                adj = base * 1.02   # elite shooters: looks for them in playoffs
            elif base >= 2.0:
                adj = base * 0.96   # solid shooters: tighter contest
            else:
                adj = base * 0.90   # role shooters: reduced opportunities
        else:
            adj = base
        return round(adj * (0.95 if is_b2b else 1.0), 1)

    return None


# ---------------------------------------------------------------------------
# MLB batter stat projections
# ---------------------------------------------------------------------------

def _project_mlb_batter_stat(
    prop_type:   str,
    bstats:      Dict,
    pitcher_fip: float,
    park_factor: float,
) -> Optional[float]:
    league_avg_fip  = 4.20
    pitcher_quality = league_avg_fip / max(pitcher_fip, 2.0)

    if prop_type == "Hits Over":
        base = bstats.get("hits_pg", 0.0)
        return round(base * pitcher_quality * park_factor, 2) if base >= 0.3 else None

    if prop_type == "Total Bases Over":
        base = bstats.get("tb_pg", 0.0)
        return round(base * pitcher_quality * park_factor, 2) if base >= 0.5 else None

    if prop_type == "Home Runs Over":
        base = bstats.get("hr_pg", 0.0)
        return round(base * pitcher_quality * park_factor, 3) if base >= 0.02 else None

    if prop_type == "HRR Over":
        base = bstats.get("hrr_pg", 0.0)
        return round(base * pitcher_quality * park_factor, 2) if base >= 0.5 else None

    return None


# ---------------------------------------------------------------------------
# NBA Props
# ---------------------------------------------------------------------------

def nba_player_props(games: List[Dict], nba_ctx: Dict, min_edge: float = None) -> List[PropPick]:
    from src.data.nba_stats import normalize as nba_normalize, get_nba_player_props_stats
    _min = min_edge if min_edge is not None else MIN_PROP_EDGE

    season_stats = nba_ctx.get("season_stats", {})
    rest_days    = nba_ctx.get("rest_days", {})
    playoff      = _is_nba_playoff()

    all_player_names = list(dict.fromkeys(
        name for game in games for name in game.get("player_props", {}).keys()
    ))
    if not all_player_names:
        logger.info("No Odds API player prop lines — skipping NBA props")
        return []

    team_names_today = [nba_normalize(t) for g in games
                        for t in [g["home_team"], g["away_team"]]]
    player_data = get_nba_player_props_stats(all_player_names, team_names_today, nba_ctx=nba_ctx)

    picks: List[PropPick] = []

    for game in games:
        home = nba_normalize(game["home_team"])
        away = nba_normalize(game["away_team"])
        game_commence = game.get("commence_time", "")
        player_props  = game.get("player_props", {})

        for player_name, prop_markets in player_props.items():
            entry = player_data.get(player_name)
            if not entry:
                continue

            pstats    = entry["stats"]
            espn_team = entry.get("team", "")
            team = home if espn_team == home else (away if espn_team == away else home)
            opp  = away if team == home else home

            team_stats = season_stats.get(team, {})
            opp_stats  = season_stats.get(opp, {})
            is_b2b     = rest_days.get(team, 1) == 0

            base_research = [
                f"{team}: Net RTG {team_stats.get('net_rtg', 0):+.1f} | PPG {team_stats.get('off_rtg', 0):.1f} | OPPG {team_stats.get('def_rtg', 0):.1f}",
                f"{opp}: Net RTG {opp_stats.get('net_rtg', 0):+.1f} | PPG {opp_stats.get('off_rtg', 0):.1f} | OPPG {opp_stats.get('def_rtg', 0):.1f}",
                f"{player_name}: {pstats.get('pts',0):.1f} PPG / {pstats.get('reb',0):.1f} RPG / {pstats.get('ast',0):.1f} APG / "
                f"{pstats.get('stl',0):.2f} SPG / {pstats.get('blk',0):.2f} BPG / {pstats.get('three_pm',0):.1f} 3PM "
                f"({pstats.get('games',0)} games)",
            ]

            for prop_label, market_info in prop_markets.items():
                market_line  = market_info["line"]
                market_prob  = market_info["market_prob"]
                odds_display = market_info["odds_display"]
                over_price   = market_info["over_price"]
                book         = market_info["book"]

                model_line = _project_nba_stat(prop_label, pstats, team_stats, opp_stats, is_b2b, playoff)
                if model_line is None:
                    continue

                model_prob = _cover_prob(prop_label, model_line, market_line)
                edge       = model_prob - market_prob
                if edge < _min:
                    continue

                conf = _confidence(edge, model_line, market_line)

                signals = [
                    f"Market: Over {market_line} at {odds_display} ({book})",
                    f"Model projects {model_line} → edge {edge*100:+.1f}%",
                ]
                if prop_label == "Points Over":
                    signals.append(
                        f"{opp} allows {opp_stats.get('def_rtg',0):.1f} PPG "
                        f"({'above' if opp_stats.get('def_rtg',112)>112 else 'below'} avg)"
                    )
                if is_b2b:
                    signals.append(f"⚠ {team} on back-to-back — reduction applied")
                if playoff and prop_label == "Points Over":
                    pts = pstats.get("pts", 0)
                    if pts >= 23:
                        signals.append("🏆 Playoffs: star usage boost applied (+5%)")
                    elif pts >= 15:
                        signals.append("🏆 Playoffs: starter adjustment applied (-4%)")
                    else:
                        signals.append("🏆 Playoffs: role player reduction applied (-12%)")

                picks.append(PropPick(
                    sport="NBA", player=player_name, team=team, opponent=opp,
                    prop_type=prop_label,
                    model_line=model_line, market_line=market_line,
                    market_prob=round(market_prob, 4), model_prob=round(model_prob, 4),
                    edge=round(edge, 4), odds_display=odds_display, book=book,
                    confidence=conf,
                    model_margin=round(model_line - market_line, 2),
                    note=f"{book.title()}: Over {market_line} at {odds_display}. Model projects {model_line}.",
                    signals=signals,
                    research=base_research[:],
                    commence_time=game_commence,
                ))

    seen: set = set()
    deduped: List[PropPick] = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    deduped.sort(key=lambda p: (0 if p.confidence == "HIGH" else 1, -p.edge))
    # When called for display (min_edge explicitly set), don't cap here —
    # the generator applies its own MAX_PROPS_PER_SPORT cap.
    return deduped if min_edge is not None else deduped[:6]


# ---------------------------------------------------------------------------
# MLB Props
# ---------------------------------------------------------------------------

def mlb_player_props(games: List[Dict], pitcher_stats_map: Dict, min_edge: float = None) -> List[PropPick]:
    from src.data.mlb_stats import get_park_factor, get_batter_props_stats, get_team_batting_stats
    _min = min_edge if min_edge is not None else MIN_PROP_EDGE
    playoff = _is_mlb_playoff()
    picks: List[PropPick] = []

    for game in games:
        home  = game.get("home_team", "")
        away  = game.get("away_team", "")
        venue = game.get("venue", "")
        game_commence = game.get("commence_time", "")
        player_props  = game.get("player_props", {})
        if not player_props:
            continue

        pf           = get_park_factor(venue)
        home_team_id = game.get("home_team_id")
        away_team_id = game.get("away_team_id")
        home_pitcher = game.get("home_pitcher_name", "TBD")
        away_pitcher = game.get("away_pitcher_name", "TBD")
        home_p_stats = pitcher_stats_map.get(home_pitcher, {})
        away_p_stats = pitcher_stats_map.get(away_pitcher, {})
        ump_k_factor = game.get("umpire_k_factor", 1.0)
        ump_name     = game.get("umpire_name", "")

        # Pre-fetch batter stats for both teams
        batter_names = [
            name for name, markets in player_props.items()
            if any(pt in markets for pt in ["Hits Over","Total Bases Over","Home Runs Over","HRR Over"])
        ]
        batter_stats: Dict[str, Dict] = {}
        # Track which team each batter is on
        batter_team:  Dict[str, str]  = {}
        if batter_names:
            if home_team_id:
                home_batters = get_batter_props_stats(home_team_id, batter_names)
                for bn in home_batters:
                    batter_stats[bn] = home_batters[bn]
                    batter_team[bn]  = home
            if away_team_id:
                away_batters = get_batter_props_stats(away_team_id, batter_names)
                for bn in away_batters:
                    if bn not in batter_stats:   # home takes precedence if duplicate
                        batter_stats[bn] = away_batters[bn]
                        batter_team[bn]  = away

        for player_name, prop_markets in player_props.items():
            for prop_label, market_info in prop_markets.items():
                market_line  = market_info["line"]
                market_prob  = market_info["market_prob"]
                odds_display = market_info["odds_display"]
                over_price   = market_info["over_price"]
                book         = market_info["book"]

                # ── Pitcher strikeouts ──────────────────────────────────────
                if prop_label == "Strikeouts Over":
                    stats = pitcher_stats_map.get(player_name, {})
                    if not stats or stats.get("innings_pitched", 0) < 20:
                        continue

                    k9    = stats.get("k_per_9", 7.0)
                    fip   = stats.get("fip", 4.20)
                    xfip  = stats.get("xfip")
                    babip = stats.get("babip")
                    ip    = stats.get("innings_pitched", 0)
                    era   = stats.get("era", 4.50)
                    bb9   = stats.get("bb_per_9", 3.0)
                    hr9   = stats.get("hr_per_9", 1.2)
                    expected_innings = MLB_PLAYOFF_STARTER_IP if playoff else 5.5

                    if player_name == home_pitcher:
                        team, opp, opp_team_id = home, away, away_team_id
                    else:
                        team, opp, opp_team_id = away, home, home_team_id

                    LEAGUE_K_PCT = 0.228
                    opp_k_pct   = LEAGUE_K_PCT
                    if opp_team_id:
                        try:
                            opp_bat   = get_team_batting_stats(opp_team_id)
                            opp_k_pct = opp_bat.get("k_pct", LEAGUE_K_PCT)
                        except Exception:
                            pass

                    era_trap = (babip is not None and isinstance(babip, float)
                                and babip < 0.260 and ip < 60 and era < fip - 0.40)

                    k9_adj     = round(k9 * ump_k_factor * (opp_k_pct / LEAGUE_K_PCT), 1)
                    model_line = round(k9_adj / 9 * expected_innings, 1)

                    model_prob = _cover_prob(prop_label, model_line, market_line)
                    edge       = model_prob - market_prob
                    if edge < _min:
                        continue

                    conf     = _confidence(edge, model_line, market_line)
                    xfip_str = f" / xFIP {xfip:.2f}" if xfip else ""
                    babip_str= f" | BABIP {babip:.3f}" if babip else ""

                    signals = [
                        f"Market: Over {market_line} Ks at {odds_display} ({book})",
                        f"K/9 {k9:.1f} (adj {k9_adj:.1f}) → {model_line} Ks in {expected_innings} inn | Edge {edge*100:+.1f}%",
                        f"FIP {fip:.2f}{xfip_str}{babip_str}",
                        f"Opp K%: {opp_k_pct:.1%} vs league avg {LEAGUE_K_PCT:.1%}",
                    ]
                    if ump_name and abs(ump_k_factor - 1.0) >= 0.03:
                        signals.append(f"👨‍⚖️ Umpire {ump_name}: K factor {ump_k_factor:.2f}x")
                    if era_trap:
                        signals.append(f"⚠ ERA trap: BABIP {babip:.3f} — ERA likely luck-driven")

                    picks.append(PropPick(
                        sport="MLB", player=player_name, team=team, opponent=opp,
                        prop_type=prop_label,
                        model_line=model_line, market_line=market_line,
                        market_prob=round(market_prob, 4), model_prob=round(model_prob, 4),
                        edge=round(edge, 4), odds_display=odds_display, book=book,
                        confidence=conf, model_margin=round(k9_adj - 7.0, 1),
                        note=f"{book.title()}: Over {market_line} Ks at {odds_display}. Model: {model_line} Ks in {expected_innings} inn.",
                        signals=signals,
                        research=[
                            f"{player_name}: ERA {era:.2f} | FIP {fip:.2f}{xfip_str} | K/9 {k9:.1f} | BB/9 {bb9:.1f} | HR/9 {hr9:.2f}{babip_str}",
                            f"Season IP: {ip:.0f} | Projected start: ~{expected_innings} inn",
                            f"Venue: {venue} (park factor {pf:.2f})" if venue else "",
                        ],
                        commence_time=game_commence,
                    ))
                    continue

                # ── Batter props ────────────────────────────────────────────
                bstats = batter_stats.get(player_name)
                if not bstats:
                    continue

                team = batter_team.get(player_name, home)
                opp  = away if team == home else home
                facing_p_stats = away_p_stats if team == home else home_p_stats
                facing_name    = away_pitcher if team == home else home_pitcher
                pitcher_fip    = facing_p_stats.get("fip", 4.20)

                model_line = _project_mlb_batter_stat(prop_label, bstats, pitcher_fip, pf)
                if model_line is None:
                    continue

                model_prob = _cover_prob(prop_label, model_line, market_line)
                edge       = model_prob - market_prob
                if edge < _min:
                    continue

                conf = _confidence(edge, model_line, market_line)

                picks.append(PropPick(
                    sport="MLB", player=player_name, team=team, opponent=opp,
                    prop_type=prop_label,
                    model_line=round(model_line, 2), market_line=market_line,
                    market_prob=round(market_prob, 4), model_prob=round(model_prob, 4),
                    edge=round(edge, 4), odds_display=odds_display, book=book,
                    confidence=conf, model_margin=round(model_line - market_line, 2),
                    note=f"{book.title()}: Over {market_line} at {odds_display}. Model: {model_line:.2f}.",
                    signals=[
                        f"Market: Over {market_line} at {odds_display} ({book})",
                        f"Model projects {model_line:.2f} → edge {edge*100:+.1f}%",
                        f"vs {facing_name} (FIP {pitcher_fip:.2f}) at {venue} (park factor {pf:.2f})",
                    ],
                    research=[
                        f"{player_name}: AVG {bstats['avg']:.3f} | OBP {bstats['obp']:.3f} | "
                        f"H/G {bstats['hits_pg']:.2f} | TB/G {bstats['tb_pg']:.2f} | "
                        f"HR/G {bstats['hr_pg']:.3f} | HRR/G {bstats['hrr_pg']:.2f} ({bstats['pa']} PA)",
                        f"Facing {facing_name}: FIP {pitcher_fip:.2f}",
                        f"Park factor {pf:.2f} at {venue}" if venue else "",
                    ],
                    commence_time=game_commence,
                ))

    seen: set = set()
    deduped: List[PropPick] = []
    for p in picks:
        key = (p.player, p.prop_type)
        if key not in seen:
            seen.add(key)
            deduped.append(p)
    deduped.sort(key=lambda p: (0 if p.confidence == "HIGH" else 1, -p.edge))
    return deduped if min_edge is not None else deduped[:6]
