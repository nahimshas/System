"""
Core probability model. Takes market odds + stats + injuries, returns edge analysis.

Approach:
  1. Use market implied probability (no-vig) as base
  2. Apply statistical adjustments for factors the market may have mispriced
  3. Compare adjusted probability to market — flag positive edges
"""
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional
from scipy.stats import norm

from src.config import (
    NBA_HOME_ADVANTAGE, NBA_BACK_TO_BACK_PENALTY, NBA_REST_BONUS_PER_DAY,
    NBA_RECENT_FORM_WEIGHT, NBA_TOTAL_STD, MLB_HOME_ADVANTAGE, MIN_EDGE,
    NBA_PLAYOFF_SCORING_FACTOR, NBA_PLAYOFF_PACE_FACTOR,
    NBA_PLAYOFF_RECENT_WEIGHT, NBA_PLAYOFF_TOTAL_STD,
    MLB_PLAYOFF_SCORING_FACTOR, MLB_PLAYOFF_STARTER_IP,
    MLB_PLAYOFF_RECENT_WEIGHT,
    SCHEDULE_LOAD_THRESHOLDS,
)
from src.data.injuries import injury_adjustment
from src.data.nba_stats import normalize as nba_normalize
from src.models.kelly import robinhood_kelly, has_positive_ev, BetSizing

logger = logging.getLogger(__name__)

# Standard deviation of game point/run differentials used in norm.cdf win-prob model.
# NBA: real-world NBA point differential std ≈ 14 pts. Using 12 was too tight,
#      causing the model to be overconfident on lopsided matchups.
# MLB: real-world MLB run differential std ≈ 3.0 runs. 1.8 was far too tight,
#      inflating all MLB edges significantly.
NBA_SPREAD_STD = 14.0   # was 12.0
MLB_SPREAD_STD = 3.0    # was 1.8


def _is_nba_playoff(dt: Optional[datetime] = None) -> bool:
    """NBA playoffs: mid-April through mid-June."""
    d = (dt or datetime.now(timezone.utc)).date()
    return (d.month == 4 and d.day >= 14) or d.month in (5, 6)


def _is_mlb_playoff(dt: Optional[datetime] = None) -> bool:
    """MLB playoffs: October through early November."""
    d = (dt or datetime.now(timezone.utc)).date()
    return d.month == 10 or (d.month == 11 and d.day <= 5)


def _schedule_load_penalty(games_last_7: int) -> float:
    """Returns probability penalty for heavy schedule load."""
    for threshold in sorted(SCHEDULE_LOAD_THRESHOLDS.keys(), reverse=True):
        if games_last_7 >= threshold:
            return SCHEDULE_LOAD_THRESHOLDS[threshold]
    return 0.0


def _utc_to_pdt(utc_str: str) -> str:
    """Converts UTC ISO string to PDT time string for display."""
    try:
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        offset = -7 if 3 <= dt.month <= 10 else -8
        label = "PDT" if offset == -7 else "PST"
        local = dt + timedelta(hours=offset)
        return local.strftime(f"%-I:%M %p {label}")
    except Exception:
        return ""


@dataclass
class BetRecommendation:
    sport: str
    game: str
    bet_type: str
    pick: str
    market_prob: float
    model_prob: float
    edge: float
    contract_price: float
    sizing: BetSizing
    confidence: str
    signals: List[str] = field(default_factory=list)
    research: List[str] = field(default_factory=list)   # deeper stat context
    home_team: str = ""
    away_team: str = ""
    game_time: str = ""                                  # e.g. "7:10 PM PDT"
    commence_time: str = ""                              # raw UTC ISO — for game-started detection
    locked: bool = False                                 # True once game has started


def _confidence_label(edge: float, signal_count: int, stats_available: bool) -> str:
    """HIGH requires strong edge + multiple signals + stats available."""
    if edge >= 0.07 and signal_count >= 3 and stats_available:
        return "HIGH"
    return "MEDIUM"


# ---------------------------------------------------------------------------
# NBA Edge Finder
# ---------------------------------------------------------------------------

def _nba_team_strength(team: str, ctx: Dict, playoff: bool = False) -> float:
    season = ctx["season_stats"].get(team, {})
    recent = ctx["recent_form"].get(team, {})
    season_net = season.get("net_rtg", 0.0)
    recent_net = recent.get("recent_net_rtg", season_net)
    w = NBA_PLAYOFF_RECENT_WEIGHT if playoff else NBA_RECENT_FORM_WEIGHT
    return (1 - w) * season_net + w * recent_net


def _nba_margin_to_prob(expected_margin: float) -> float:
    return float(norm.cdf(expected_margin, 0, NBA_SPREAD_STD))


def analyze_nba_game(game: Dict, nba_ctx: Dict, nba_injuries: Dict) -> List[BetRecommendation]:
    home = nba_normalize(game["home_team"])
    away = nba_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []

    playoff = _is_nba_playoff()
    stats_available = bool(nba_ctx["season_stats"].get(home) or nba_ctx["season_stats"].get(away))

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_stats = nba_ctx["season_stats"].get(home, {})
        away_stats = nba_ctx["season_stats"].get(away, {})
        home_recent = nba_ctx["recent_form"].get(home, {})
        away_recent = nba_ctx["recent_form"].get(away, {})

        home_strength = _nba_team_strength(home, nba_ctx, playoff)
        away_strength = _nba_team_strength(away, nba_ctx, playoff)
        base_margin = home_strength - away_strength

        signals = []
        research = []
        adj = 0.0

        # --- Stats unavailability warning ---
        if not stats_available:
            signals.append("⚠ NBA stats unavailable — using market baseline only")

        # --- Home court ---
        adj += NBA_HOME_ADVANTAGE
        signals.append(f"Home court advantage: {home} (+{NBA_HOME_ADVANTAGE*100:.0f}%)")

        # --- Team strength research ---
        if home_stats:
            research.append(
                f"{home}: OffRtg {home_stats.get('off_rtg', '?'):.1f} | "
                f"DefRtg {home_stats.get('def_rtg', '?'):.1f} | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.1f}"
            )
        if away_stats:
            research.append(
                f"{away}: OffRtg {away_stats.get('off_rtg', '?'):.1f} | "
                f"DefRtg {away_stats.get('def_rtg', '?'):.1f} | "
                f"NetRtg {away_stats.get('net_rtg', '?'):.1f}"
            )

        # --- Recent form ---
        if home_recent:
            research.append(
                f"{home} last 14 days: NetRtg {home_recent.get('recent_net_rtg', '?'):.1f} | "
                f"Win% {home_recent.get('recent_w_pct', 0)*100:.0f}%"
            )
        if away_recent:
            research.append(
                f"{away} last 14 days: NetRtg {away_recent.get('recent_net_rtg', '?'):.1f} | "
                f"Win% {away_recent.get('recent_w_pct', 0)*100:.0f}%"
            )

        # --- Net rating edge signal ---
        if abs(home_strength - away_strength) > 3:
            stronger = home if home_strength > away_strength else away
            signals.append(
                f"Rating edge: {stronger} (blended NetRtg diff {home_strength - away_strength:+.1f})"
            )

        # --- Rest / B2B ---
        home_rest = nba_ctx["rest_days"].get(home, 1)
        away_rest = nba_ctx["rest_days"].get(away, 1)

        if home_rest == 0:
            adj -= NBA_BACK_TO_BACK_PENALTY
            signals.append(f"{home} on back-to-back (-{NBA_BACK_TO_BACK_PENALTY*100:.0f}%)")
        if away_rest == 0:
            adj += NBA_BACK_TO_BACK_PENALTY
            signals.append(f"{away} on back-to-back — favors {home} (+{NBA_BACK_TO_BACK_PENALTY*100:.0f}%)")

        rest_diff = min(home_rest - away_rest, 3)
        if abs(rest_diff) >= 1:
            rest_adj = rest_diff * NBA_REST_BONUS_PER_DAY
            adj += rest_adj
            direction = home if rest_diff > 0 else away
            signals.append(f"Rest edge: {direction} has {abs(rest_diff)} more rest day(s) (+{abs(rest_adj)*100:.1f}%)")

        research.append(f"Rest days — {home}: {home_rest} | {away}: {away_rest}")

        # --- Schedule load (7-day fatigue) ---
        home_load = nba_ctx.get("schedule_load", {}).get(home, 0)
        away_load = nba_ctx.get("schedule_load", {}).get(away, 0)
        home_load_pen = _schedule_load_penalty(home_load)
        away_load_pen = _schedule_load_penalty(away_load)
        if home_load_pen > 0:
            adj -= home_load_pen
            signals.append(f"⚠ {home} schedule load: {home_load} games in 7 days (-{home_load_pen*100:.0f}%)")
        if away_load_pen > 0:
            adj += away_load_pen
            signals.append(f"⚠ {away} schedule load: {away_load} games in 7 days — favors {home} (+{away_load_pen*100:.0f}%)")

        # --- Playoff context ---
        if playoff:
            signals.append("🏆 Playoffs: defensive intensity higher, recent form weighted 55%")
            research.append("Playoff adjustment: scoring/pace factors applied to totals model")

        # --- Injuries ---
        home_inj = injury_adjustment(home, nba_injuries, "nba")
        away_inj = injury_adjustment(away, nba_injuries, "nba")
        home_inj_list = nba_injuries.get(home, [])
        away_inj_list = nba_injuries.get(away, [])

        if home_inj_list:
            for p in home_inj_list:
                research.append(f"⚕ {home} — {p['player']} ({p['position']}): {p['status'].upper()}")
        if away_inj_list:
            for p in away_inj_list:
                research.append(f"⚕ {away} — {p['player']} ({p['position']}): {p['status'].upper()}")

        if home_inj > 0.005:
            adj -= home_inj
            signals.append(f"{home} injury impact (-{home_inj*100:.1f}%)")
        if away_inj > 0.005:
            adj += away_inj
            signals.append(f"{away} injuries benefit {home} (+{away_inj*100:.1f}%)")

        if not home_inj_list and not away_inj_list:
            research.append("No significant injuries reported for either team")

        adjusted_home_prob = min(0.90, max(0.10, _nba_margin_to_prob(base_margin) + adj))
        adjusted_away_prob = 1 - adjusted_home_prob

        home_edge = adjusted_home_prob - market_home_prob
        away_edge = adjusted_away_prob - market_away_prob

        if home_edge >= MIN_EDGE and has_positive_ev(adjusted_home_prob, market_home_prob):
            sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=adjusted_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing,
                    confidence=_confidence_label(home_edge, len(signals), stats_available),
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= MIN_EDGE and has_positive_ev(adjusted_away_prob, market_away_prob):
            sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=adjusted_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing,
                    confidence=_confidence_label(away_edge, len(signals), stats_available),
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Spread ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", 0.0)   # e.g. -6.5 means home favoured by 6.5
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            # Derive effective point margin from win probability (inverse of _nba_margin_to_prob).
            # This keeps the spread model consistent with every adjustment already applied
            # (home court, rest, injuries, schedule load) without recomputing them separately.
            effective_margin = float(norm.ppf(adjusted_home_prob)) * NBA_SPREAD_STD

            # P(home covers) = P(actual margin > −home_spread_line)
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, NBA_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover

            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= MIN_EDGE and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing,
                        confidence=_confidence_label(home_sp_edge, len(signals), stats_available),
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_sp_edge >= MIN_EDGE and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing,
                        confidence=_confidence_label(away_sp_edge, len(signals), stats_available),
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    # --- Total ---
    total = game.get("total")
    if total and nba_ctx["season_stats"]:
        home_stats = nba_ctx["season_stats"].get(home, {})
        away_stats = nba_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            avg_pace = (home_stats.get("pace", 100) + away_stats.get("pace", 100)) / 2
            if playoff:
                avg_pace *= NBA_PLAYOFF_PACE_FACTOR
            expected_home_pts = (home_stats.get("off_rtg", 110) + away_stats.get("def_rtg", 110)) / 2 * avg_pace / 100
            expected_away_pts = (away_stats.get("off_rtg", 110) + home_stats.get("def_rtg", 110)) / 2 * avg_pace / 100
            expected_total = expected_home_pts + expected_away_pts
            if playoff:
                expected_total *= NBA_PLAYOFF_SCORING_FACTOR

            market_line = total["line"]
            total_std   = NBA_PLAYOFF_TOTAL_STD if playoff else NBA_TOTAL_STD
            model_over_prob = float(1 - norm.cdf(market_line, expected_total, total_std))
            market_over_prob = total["over_prob"]
            market_under_prob = total["under_prob"]

            # Apply B2B scoring penalty directly to expected total
            # (4% penalty per team ≈ ~4 pts off expected scoring for that team)
            b2b_teams = []
            if home_rest == 0:
                expected_total -= 4.0
                b2b_teams.append(home)
            if away_rest == 0:
                expected_total -= 4.0
                b2b_teams.append(away)

            # Recalculate with B2B-adjusted expected total
            model_over_prob = float(1 - norm.cdf(market_line, expected_total, total_std))

            base_total_signals = [
                f"Model expected total: {expected_total:.1f} vs market line {market_line}",
                f"Combined pace: {avg_pace:.1f} possessions/game",
            ]
            total_research = [
                f"{home} OffRtg: {home_stats.get('off_rtg', '?'):.1f} vs {away} DefRtg: {away_stats.get('def_rtg', '?'):.1f}",
                f"{away} OffRtg: {away_stats.get('off_rtg', '?'):.1f} vs {home} DefRtg: {home_stats.get('def_rtg', '?'):.1f}",
            ]
            if b2b_teams:
                total_research.append(
                    f"B2B: {', '.join(b2b_teams)} — pace/scoring reduction applied to model (-4 pts per team)"
                )

            over_edge  = model_over_prob - market_over_prob
            under_edge = (1 - model_over_prob) - market_under_prob

            # Direction-specific signals so B2B note is never contradictory
            over_signals  = base_total_signals[:]
            under_signals = base_total_signals[:]
            if b2b_teams:
                b2b_str = ", ".join(b2b_teams)
                under_signals.append(f"⚠ {b2b_str} on B2B — pace reduction supports Under")
                over_signals.append( f"⚠ {b2b_str} on B2B — model projects Over despite pace reduction")

            if over_edge >= MIN_EDGE and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=f"Over {market_line}",
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(over_signals), stats_available),
                        signals=over_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))
            if under_edge >= MIN_EDGE and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=f"Under {market_line}",
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(under_signals), stats_available),
                        signals=under_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    return recs


# ---------------------------------------------------------------------------
# MLB Edge Finder
# ---------------------------------------------------------------------------

def _pitcher_quality_score(stats: Dict) -> float:
    fip = stats.get("fip", 4.20)
    return (4.20 - fip) / 1.50


def analyze_mlb_game(game: Dict, home_pitcher_stats: Dict, away_pitcher_stats: Dict,
                     home_batting: Dict, away_batting: Dict,
                     home_bullpen: Dict, away_bullpen: Dict,
                     mlb_injuries: Dict,
                     home_schedule_load: int = 0,
                     away_schedule_load: int = 0,
                     umpire_tendency: Optional[Dict] = None,
                     weather: Optional[Dict] = None) -> List[BetRecommendation]:
    from src.data.mlb_stats import get_park_factor
    from src.data.umpire import build_umpire_signals
    from src.data.weather import build_weather_signals, weather_run_adjustment
    home = game["home_team"]
    away = game["away_team"]
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    venue = game.get("venue", "")
    park_factor = get_park_factor(venue)
    playoff = _is_mlb_playoff()
    recs = []
    umpire_tendency = umpire_tendency or {}
    weather = weather or {}

    signals = []
    research = []
    home_pitcher_name = game.get("home_pitcher_name", "TBD")
    away_pitcher_name = game.get("away_pitcher_name", "TBD")

    stats_available = bool(home_pitcher_stats or away_pitcher_stats)

    # --- Pitcher research ---
    if home_pitcher_stats:
        fip = home_pitcher_stats.get('fip', '?')
        era = home_pitcher_stats.get('era', '?')
        k9 = home_pitcher_stats.get('k_per_9', '?')
        bb9 = home_pitcher_stats.get('bb_per_9', '?')
        ip = home_pitcher_stats.get('innings_pitched', '?')
        research.append(f"🔵 {home_pitcher_name} (home): ERA {era} | FIP {fip} | K/9 {k9} | BB/9 {bb9} | IP {ip:.0f}" if isinstance(ip, float) else f"🔵 {home_pitcher_name} (home): ERA {era} | FIP {fip} | K/9 {k9}")
        signals.append(f"{home_pitcher_name} FIP: {fip} | K/9: {k9}")
    else:
        research.append(f"🔵 {home_pitcher_name} (home): stats unavailable")

    if away_pitcher_stats:
        fip = away_pitcher_stats.get('fip', '?')
        era = away_pitcher_stats.get('era', '?')
        k9 = away_pitcher_stats.get('k_per_9', '?')
        bb9 = away_pitcher_stats.get('bb_per_9', '?')
        ip = away_pitcher_stats.get('innings_pitched', '?')
        research.append(f"🔴 {away_pitcher_name} (away): ERA {era} | FIP {fip} | K/9 {k9} | BB/9 {bb9} | IP {ip:.0f}" if isinstance(ip, float) else f"🔴 {away_pitcher_name} (away): ERA {era} | FIP {fip} | K/9 {k9}")
        signals.append(f"{away_pitcher_name} FIP: {fip} | K/9: {k9}")
    else:
        research.append(f"🔴 {away_pitcher_name} (away): stats unavailable")

    # --- Batting research ---
    if home_batting:
        research.append(
            f"{home} offense: OPS {home_batting.get('ops', '?'):.3f} | "
            f"AVG {home_batting.get('avg', '?'):.3f} | "
            f"R/G {home_batting.get('runs_per_game', '?'):.2f}"
        )
    if away_batting:
        research.append(
            f"{away} offense: OPS {away_batting.get('ops', '?'):.3f} | "
            f"AVG {away_batting.get('avg', '?'):.3f} | "
            f"R/G {away_batting.get('runs_per_game', '?'):.2f}"
        )

    # --- Bullpen research ---
    home_bp_era = home_bullpen.get("bullpen_era", 4.20)
    away_bp_era = away_bullpen.get("bullpen_era", 4.20)
    research.append(f"Bullpen ERA — {home}: {home_bp_era:.2f} | {away}: {away_bp_era:.2f}")

    # --- Park factor ---
    if park_factor != 1.0:
        direction = "hitter-friendly" if park_factor > 1.02 else "pitcher-friendly"
        signals.append(f"Park factor {park_factor:.2f} ({direction}: {venue})")
        research.append(f"Venue: {venue} — park factor {park_factor:.2f} ({direction})")
    else:
        research.append(f"Venue: {venue} — neutral park")

    # --- Injuries ---
    home_inj = injury_adjustment(home, mlb_injuries, "mlb")
    away_inj = injury_adjustment(away, mlb_injuries, "mlb")
    home_inj_list = mlb_injuries.get(home, [])
    away_inj_list = mlb_injuries.get(away, [])

    for p in home_inj_list:
        research.append(f"⚕ {home} — {p['player']} ({p['position']}): {p['status'].upper()}")
    for p in away_inj_list:
        research.append(f"⚕ {away} — {p['player']} ({p['position']}): {p['status'].upper()}")

    if not home_inj_list and not away_inj_list:
        research.append("No significant injuries reported")

    if home_inj > 0.01:
        signals.append(f"{home} injury impact (-{home_inj*100:.1f}%)")
    if away_inj > 0.01:
        signals.append(f"{away} injury impact (-{away_inj*100:.1f}%)")

    # --- Umpire tendencies ---
    ump_run_factor = umpire_tendency.get("run_factor", 1.0)
    ump_k_factor   = umpire_tendency.get("k_factor", 1.0)
    ump_name       = game.get("umpire_name", "")
    ump_signals    = build_umpire_signals(ump_name, umpire_tendency)
    if ump_signals:
        signals.extend(ump_signals)
        research.append(
            f"👨‍⚖️ HP Umpire: {ump_name} — run factor {ump_run_factor:.2f}x | "
            f"K factor {ump_k_factor:.2f}x | {umpire_tendency.get('notes', '')}"
        )
    elif ump_name:
        research.append(f"👨‍⚖️ HP Umpire: {ump_name} (near-neutral tendencies)")

    # --- Weather ---
    wx_adj     = weather_run_adjustment(weather)
    wx_signals = build_weather_signals(weather)
    if wx_signals:
        signals.extend(wx_signals)
    if not weather.get("indoor") and weather.get("city"):
        wx_city = weather.get("city", "")
        research.append(
            f"🌤 Weather ({wx_city}): {weather.get('temp_f', 70)}°F | "
            f"Wind {weather.get('wind_mph', 0)} mph {weather.get('wind_dir', '')} "
            f"({'blowing ' + weather.get('wind_effect', 'cross') if weather.get('wind_effect') != 'cross' else 'cross wind'}) | "
            f"Precip {weather.get('precip_pct', 0)}%"
        )

    # --- Expected runs ---
    home_ops = home_batting.get("ops", 0.720)
    away_ops = away_batting.get("ops", 0.720)
    league_avg_runs = 4.5

    home_sp_score = _pitcher_quality_score(home_pitcher_stats) if home_pitcher_stats else 0
    away_sp_score = _pitcher_quality_score(away_pitcher_stats) if away_pitcher_stats else 0
    home_offense_adj = (home_ops - 0.720) / 0.080
    away_offense_adj = (away_ops - 0.720) / 0.080
    home_bullpen_adj = (4.20 - home_bp_era) / 2.0
    away_bullpen_adj = (4.20 - away_bp_era) / 2.0

    expected_home_runs = max(1.5, league_avg_runs
                             - away_sp_score * 0.8
                             + home_offense_adj * 0.6
                             + away_bullpen_adj * 0.3
                             + MLB_HOME_ADVANTAGE * 5)
    expected_away_runs = max(1.5, league_avg_runs
                             - home_sp_score * 0.8
                             + away_offense_adj * 0.6
                             + home_bullpen_adj * 0.3)

    expected_home_runs *= park_factor
    expected_away_runs *= park_factor

    # Apply umpire run factor (scales both teams equally)
    if abs(ump_run_factor - 1.0) >= 0.02:
        expected_home_runs *= ump_run_factor
        expected_away_runs *= ump_run_factor

    # Apply weather run adjustment (split evenly between teams)
    if wx_adj != 0.0:
        expected_home_runs += wx_adj / 2
        expected_away_runs += wx_adj / 2
        expected_home_runs = max(1.5, expected_home_runs)
        expected_away_runs = max(1.5, expected_away_runs)

    run_diff = expected_home_runs - expected_away_runs
    model_home_prob = float(norm.cdf(run_diff, 0, MLB_SPREAD_STD))
    model_home_prob = min(0.85, max(0.15, model_home_prob - home_inj + away_inj))
    model_away_prob = 1 - model_home_prob

    # --- Schedule load ---
    home_load_pen = _schedule_load_penalty(home_schedule_load)
    away_load_pen = _schedule_load_penalty(away_schedule_load)
    if home_load_pen > 0:
        expected_home_runs *= (1 - home_load_pen)
        signals.append(f"⚠ {home} schedule load: {home_schedule_load} games in 7 days")
    if away_load_pen > 0:
        expected_away_runs *= (1 - away_load_pen)
        signals.append(f"⚠ {away} schedule load: {away_schedule_load} games in 7 days")

    # --- Playoff context ---
    if playoff:
        expected_home_runs *= MLB_PLAYOFF_SCORING_FACTOR
        expected_away_runs *= MLB_PLAYOFF_SCORING_FACTOR
        signals.append("🏆 Playoffs: ace starters, shorter leash, lower scoring applied")
        research.append(f"Playoff adjustment: runs scaled by {MLB_PLAYOFF_SCORING_FACTOR} | starter IP capped at {MLB_PLAYOFF_STARTER_IP}")

    signals.append(f"Model projected score: {home} {expected_home_runs:.1f} — {away} {expected_away_runs:.1f}")
    research.append(f"Model expected runs: {home} {expected_home_runs:.2f} | {away} {expected_away_runs:.2f}")

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_edge = model_home_prob - market_home_prob
        away_edge = model_away_prob - market_away_prob

        if home_edge >= MIN_EDGE and has_positive_ev(model_home_prob, market_home_prob):
            sizing = robinhood_kelly(model_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=model_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing,
                    confidence=_confidence_label(home_edge, len(signals), stats_available),
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= MIN_EDGE and has_positive_ev(model_away_prob, market_away_prob):
            sizing = robinhood_kelly(model_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=model_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing,
                    confidence=_confidence_label(away_edge, len(signals), stats_available),
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Run Line (Spread) ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", 0.0)   # almost always ±1.5 in MLB
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            # Derive effective run margin from win probability, consistent with all adjustments above.
            effective_margin = float(norm.ppf(model_home_prob)) * MLB_SPREAD_STD

            # P(home covers run line) = P(actual margin > −home_spread_line)
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, MLB_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover

            away_spread_line = -home_spread_line

            home_rl_edge = model_home_cover - market_home_cover
            away_rl_edge = model_away_cover - market_away_cover

            if home_rl_edge >= MIN_EDGE and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_rl_edge, contract_price=market_home_cover,
                        sizing=sizing,
                        confidence=_confidence_label(home_rl_edge, len(signals), stats_available),
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_rl_edge >= MIN_EDGE and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_rl_edge, contract_price=market_away_cover,
                        sizing=sizing,
                        confidence=_confidence_label(away_rl_edge, len(signals), stats_available),
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    # --- Total ---
    total = game.get("total")
    if total:
        expected_total = expected_home_runs + expected_away_runs
        market_line = total["line"]
        model_over_prob = float(1 - norm.cdf(market_line, expected_total, MLB_SPREAD_STD * 1.5))
        market_over_prob = total["over_prob"]
        market_under_prob = total["under_prob"]

        total_signals = signals[:] + [f"Model expected total: {expected_total:.1f} vs line {market_line}"]
        total_research = research[:]

        over_edge = model_over_prob - market_over_prob
        under_edge = (1 - model_over_prob) - market_under_prob

        if over_edge >= MIN_EDGE and has_positive_ev(model_over_prob, market_over_prob):
            sizing = robinhood_kelly(model_over_prob, market_over_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=f"Over {market_line}",
                    market_prob=market_over_prob, model_prob=model_over_prob,
                    edge=over_edge, contract_price=market_over_prob,
                    sizing=sizing,
                    confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))
        if under_edge >= MIN_EDGE and has_positive_ev(1 - model_over_prob, market_under_prob):
            sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=f"Under {market_line}",
                    market_prob=market_under_prob, model_prob=1 - model_over_prob,
                    edge=under_edge, contract_price=market_under_prob,
                    sizing=sizing,
                    confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

    return recs
