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
NBA_SPREAD_STD = 12.0   # model uncertainty for point margin; 14.0 was too wide (suppressed all edges)
MLB_SPREAD_STD = 3.0    # was 1.8

# Injury credibility gate ─────────────────────────────────────────────────────
# When a team's injury_adjustment ≥ INJURY_GATE, our season net-rating baseline
# is stale (the injured player's contributions are baked in but unavailable
# tonight).  We cap the model probability so it can't disagree with the market
# by more than INJURY_CRED_MARGIN on the injured team's side.  This collapses
# phantom edges caused by missing star players while still allowing small
# genuine edges (e.g. a big underdog covering a spread).
# Any bet on an injury-capped team is automatically locked to MEDIUM confidence.
_INJURY_GATE   = 0.030   # ≈ one key starter fully out
_INJURY_MARGIN = 0.10    # model allowed at most market_prob × (1 + 10%) for injured team


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


def analyze_nba_game(game: Dict, nba_ctx: Dict, nba_injuries: Dict, min_edge: float = None) -> List[BetRecommendation]:
    home = nba_normalize(game["home_team"])
    away = nba_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    playoff = _is_nba_playoff()
    stats_available = bool(nba_ctx["season_stats"].get(home) or nba_ctx["season_stats"].get(away))

    # Pre-initialise so Total block can reference these even if ml block is skipped
    home_rest = nba_ctx["rest_days"].get(home, 1)
    away_rest = nba_ctx["rest_days"].get(away, 1)

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

        # --- Model projected score (shown on all bet cards for this game) ---
        # B2B adjustment applied here so ML/Spread/Total all show the same number.
        if home_stats and away_stats:
            _avg_pace = (home_stats.get("pace", 100) + away_stats.get("pace", 100)) / 2
            if playoff:
                _avg_pace *= NBA_PLAYOFF_PACE_FACTOR
            _exp_home = (home_stats.get("off_rtg", 110) + away_stats.get("def_rtg", 110)) / 2 * _avg_pace / 100
            _exp_away = (away_stats.get("off_rtg", 110) + home_stats.get("def_rtg", 110)) / 2 * _avg_pace / 100
            if playoff:
                _exp_home *= NBA_PLAYOFF_SCORING_FACTOR
                _exp_away *= NBA_PLAYOFF_SCORING_FACTOR
            # Apply the same -4 pt B2B penalty used in the Total block
            if home_rest == 0:
                _exp_home -= 4.0
            if away_rest == 0:
                _exp_away -= 4.0
            signals.append(f"Model projected score: {home} {_exp_home:.0f} — {away} {_exp_away:.0f}")

        adjusted_home_prob = min(0.90, max(0.10, _nba_margin_to_prob(base_margin) + adj))
        adjusted_away_prob = 1 - adjusted_home_prob

        # ── Injury credibility cap ────────────────────────────────────────────
        # Season net ratings include injured players' contributions, so the model
        # overstates a depleted team's probability.  When injury_adj ≥ _INJURY_GATE
        # (≈ one key starter out), cap that team's probability to within
        # _INJURY_MARGIN of the market — the market has already repriced.
        # The cap flows into the spread calculation automatically (spread prob is
        # derived from adjusted_home_prob via inverse-CDF below).
        home_injury_capped = False
        away_injury_capped = False

        if home_inj >= _INJURY_GATE:
            max_home_prob = min(market_home_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_home_prob > max_home_prob:
                adjusted_home_prob = max_home_prob
                adjusted_away_prob = 1 - adjusted_home_prob
                home_injury_capped = True
                research.append(
                    f"⚠ {home} injury credibility cap applied — season stats include "
                    f"injured players; model probability anchored near market "
                    f"({max_home_prob*100:.0f}% vs raw {(adjusted_home_prob)*100:.0f}%)"
                )

        if away_inj >= _INJURY_GATE:
            max_away_prob = min(market_away_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_away_prob > max_away_prob:
                adjusted_away_prob = max_away_prob
                adjusted_home_prob = 1 - adjusted_away_prob
                away_injury_capped = True
                research.append(
                    f"⚠ {away} injury credibility cap applied — season stats include "
                    f"injured players; model probability anchored near market "
                    f"({max_away_prob*100:.0f}% vs raw {(adjusted_away_prob)*100:.0f}%)"
                )

        home_edge = adjusted_home_prob - market_home_prob
        away_edge = adjusted_away_prob - market_away_prob

        if home_edge >= _min and has_positive_ev(adjusted_home_prob, market_home_prob):
            sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(home_edge, len(signals), stats_available)
                if home_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=adjusted_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= _min and has_positive_ev(adjusted_away_prob, market_away_prob):
            sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(away_edge, len(signals), stats_available)
                if away_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NBA", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=adjusted_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Spread ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", 0.0)   # e.g. -6.5 means home favoured by 6.5

            # Market spread probability comes directly from the Odds API spread market
            # (consensus no-vig across all books), same as the moneyline market probability.
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            # Derive effective point margin from the (already injury-capped) win probability.
            # This keeps the spread model consistent with every adjustment already applied.
            effective_margin = float(norm.ppf(adjusted_home_prob)) * NBA_SPREAD_STD

            # P(home covers) = P(actual margin > −home_spread_line)
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, NBA_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover

            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
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
            if playoff:
                expected_home_pts *= NBA_PLAYOFF_SCORING_FACTOR
                expected_away_pts *= NBA_PLAYOFF_SCORING_FACTOR

            # Apply B2B penalty to individual team scores so per-team projection
            # and expected_total are always consistent with each other.
            b2b_teams = []
            if home_rest == 0:
                expected_home_pts -= 4.0
                b2b_teams.append(home)
            if away_rest == 0:
                expected_away_pts -= 4.0
                b2b_teams.append(away)

            expected_total = expected_home_pts + expected_away_pts   # naturally B2B-adjusted

            market_line = total["line"]
            total_std   = NBA_PLAYOFF_TOTAL_STD if playoff else NBA_TOTAL_STD
            model_over_prob = float(1 - norm.cdf(market_line, expected_total, total_std))
            market_over_prob = total["over_prob"]
            market_under_prob = total["under_prob"]

            base_total_signals = [
                f"Model projected score: {home} {expected_home_pts:.0f} — {away} {expected_away_pts:.0f}",
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

            # Use .5 lines to eliminate push risk: "Over 10" → "Over 9.5", "Under 8" → "Under 8.5"
            _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
            _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

            if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(over_signals), stats_available),
                        signals=over_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=_under_label,
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
    """
    Returns a quality score for a starting pitcher.

    Improvements:
    - Prefers xFIP over FIP (normalises HR/FB rate, more stable early-season)
    - Applies a small-sample blend toward league average when IP < 50:
      a pitcher with 20 IP gets 40 % weight on actual stats, 60 % on the mean,
      preventing an outlier ERA/FIP over a tiny sample from dominating the model.
    """
    LEAGUE_AVG = 4.20
    ip   = stats.get("innings_pitched", 0)
    xfip = stats.get("xfip")          # None when airOuts data unavailable
    fip  = stats.get("fip", LEAGUE_AVG)

    # xFIP normalises HR luck via fly-ball rate — more reliable early-season
    base = xfip if xfip is not None else fip

    # Below 20 IP the numbers are very noisy — blend toward league mean.
    # 20 IP ≈ 4-5 starts: enough to apply real signal.
    # (Old threshold was 50 IP which was too aggressive and collapsed most edges.)
    if ip < 20:
        weight = max(0.0, ip / 20.0)          # 0 IP → 0 %, 20 IP → 100 %
        base   = base * weight + LEAGUE_AVG * (1.0 - weight)

    return (LEAGUE_AVG - base) / 1.50


def _era_trap_severity(stats: Dict) -> float:
    """
    Continuous ERA-trap severity score.

    Combines three factors:
      - ERA gap:     how much xFIP exceeds ERA  (core signal)
      - IP weight:   smaller sample → higher uncertainty → amplifies the gap
      - BABIP mult:  BABIP well below league average (.300) confirms luck

    Rough thresholds:
      < 0.15  → negligible (ignore)
      0.15–0.40 → MILD   (display in research only)
      0.40–0.80 → MODERATE (cap confidence on trap team; flag opponent edge)
      > 0.80  → SEVERE   (strong opponent edge signal; lower edge threshold)

    Elite-pitcher guard:
      When xFIP itself is below 3.20 (league-average FIP constant), the pitcher
      has genuinely elite underlying stuff — low BABIP partly reflects contact
      suppression skill, not pure luck.  In this case severity is capped at
      MODERATE (0.79) regardless of the BABIP multiplier, preventing false
      SEVERE tags on pitchers like Ohtani whose contact metrics are legitimately
      elite (low exit velocity, low hard-hit rate) but whose BABIP still sits
      below league average.  SEVERE is reserved for pitchers who are average-
      or-worse by xFIP yet show an anomalously low ERA/BABIP (e.g. Cease 2026).
    """
    era   = stats.get("era")
    xfip  = stats.get("xfip") or stats.get("fip", 4.20)
    babip = stats.get("babip")
    ip    = stats.get("innings_pitched", 0)

    if not isinstance(era, float) or not isinstance(ip, float) or ip < 10:
        return 0.0

    xfip_val  = float(xfip)
    era_gap   = max(0.0, xfip_val - era)              # how much xFIP exceeds ERA
    ip_weight = max(0.0, 1.0 - ip / 80.0)             # decays to 0 at 80+ IP

    # BABIP penalty: pitcher league avg ≈ .300; each 0.050 below avg adds 1.0×
    BABIP_LEAGUE_AVG = 0.300
    babip_mult = 1.0
    if isinstance(babip, float) and babip > 0:
        babip_mult = 1.0 + max(0.0, (BABIP_LEAGUE_AVG - babip) / 0.050)

    severity = era_gap * ip_weight * babip_mult

    # Elite-pitcher guard: xFIP < 3.20 means genuinely dominant true talent.
    # Low BABIP for elite pitchers is partly skill — cap at MODERATE.
    if xfip_val < 3.20:
        severity = min(severity, 0.79)

    return round(severity, 3)


def _mlb_conf(edge: float, signal_count: int, stats_available: bool,
              own_trap_sev: float = 0.0, opp_trap_sev: float = 0.0,
              injury_capped: bool = False) -> str:
    """
    MLB-specific confidence label that accounts for ERA trap severity.

    own_trap_sev  — severity of the ERA trap on THIS team's starting pitcher.
                    High → cap at MEDIUM (market is not as wrong as raw edge suggests).
    opp_trap_sev  — severity of the ERA trap on the OPPONENT's starter.
                    High → lower the edge threshold required for HIGH confidence
                    (market is overvaluing the opponent; our edge is more reliable).
    """
    if injury_capped:
        return "MEDIUM"
    # Our pitcher's trap: the market edge may partly be noise from sample-size luck
    if own_trap_sev >= 0.40:
        return "MEDIUM"
    # Opponent's trap lowers the bar for HIGH: the market systematically misprices
    # teams facing a pitcher with an inflated ERA
    if opp_trap_sev >= 0.80:
        min_edge = 0.05   # severe opponent trap
    elif opp_trap_sev >= 0.40:
        min_edge = 0.06   # moderate opponent trap
    else:
        min_edge = 0.07   # no trap — standard
    if edge >= min_edge and signal_count >= 3 and stats_available:
        return "HIGH"
    return "MEDIUM"


def analyze_mlb_game(game: Dict, home_pitcher_stats: Dict, away_pitcher_stats: Dict,
                     home_batting: Dict, away_batting: Dict,
                     home_bullpen: Dict, away_bullpen: Dict,
                     mlb_injuries: Dict,
                     home_schedule_load: int = 0,
                     away_schedule_load: int = 0,
                     umpire_tendency: Optional[Dict] = None,
                     weather: Optional[Dict] = None,
                     min_edge: float = None) -> List[BetRecommendation]:
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
    _min = min_edge if min_edge is not None else MIN_EDGE
    umpire_tendency = umpire_tendency or {}
    weather = weather or {}

    signals = []
    research = []
    home_pitcher_name = game.get("home_pitcher_name", "TBD")
    away_pitcher_name = game.get("away_pitcher_name", "TBD")

    stats_available = bool(home_pitcher_stats or away_pitcher_stats)

    # --- Pitcher research ---
    def _pitcher_lines(name: str, stats: Dict, colour: str) -> None:
        """Append research + signal lines for one starter; flag ERA traps."""
        era   = stats.get("era", "?")
        fip   = stats.get("fip", "?")
        xfip  = stats.get("xfip")
        babip = stats.get("babip")
        k9    = stats.get("k_per_9", "?")
        bb9   = stats.get("bb_per_9", "?")
        ip    = stats.get("innings_pitched", "?")

        fip_str  = f"FIP {fip}"
        xfip_str = f" / xFIP {xfip:.2f}" if xfip is not None else ""
        babip_str = f" | BABIP {babip:.3f}" if babip is not None else ""
        ip_str   = f"{ip:.0f}" if isinstance(ip, float) else str(ip)

        research.append(
            f"{colour} {name}: ERA {era} | {fip_str}{xfip_str} | "
            f"K/9 {k9} | BB/9 {bb9} | IP {ip_str}{babip_str}"
        )
        signals.append(f"{name} {fip_str}{xfip_str} | K/9: {k9}")

        # ERA-trap flag: BABIP < .260 AND ERA well below FIP AND small sample
        try:
            if (
                babip is not None
                and isinstance(era, float)
                and isinstance(fip, float)
                and isinstance(ip, float)
                and babip < 0.260
                and ip < 60
                and era < fip - 0.40
            ):
                signals.append(
                    f"⚠ {name}: BABIP {babip:.3f} is unusually low — "
                    f"ERA {era:.2f} likely overstates quality (true talent closer to FIP {fip:.2f})"
                )
        except Exception:
            pass

    if home_pitcher_stats:
        _pitcher_lines(home_pitcher_name, home_pitcher_stats, "🔵")
    else:
        research.append(f"🔵 {home_pitcher_name} (home): stats unavailable")

    if away_pitcher_stats:
        _pitcher_lines(away_pitcher_name, away_pitcher_stats, "🔴")
    else:
        research.append(f"🔴 {away_pitcher_name} (away): stats unavailable")

    # --- ERA trap severity (continuous score) ---
    # Computed here so the same values feed BOTH signal generation and
    # the per-bet confidence logic below.
    home_trap_sev = _era_trap_severity(home_pitcher_stats) if home_pitcher_stats else 0.0
    away_trap_sev = _era_trap_severity(away_pitcher_stats) if away_pitcher_stats else 0.0

    _TRAP_MILD = 0.15
    _TRAP_MOD  = 0.40
    _TRAP_SEV  = 0.80

    def _trap_label(sev: float) -> str:
        if sev >= _TRAP_SEV:  return "SEVERE"
        if sev >= _TRAP_MOD:  return "MODERATE"
        return "MILD"

    for pitcher_name, pitcher_stats, team_name, sev in [
        (home_pitcher_name, home_pitcher_stats, home, home_trap_sev),
        (away_pitcher_name, away_pitcher_stats, away, away_trap_sev),
    ]:
        if sev < _TRAP_MILD or not pitcher_stats:
            continue
        era_v   = pitcher_stats.get("era", "?")
        xfip_v  = pitcher_stats.get("xfip") or pitcher_stats.get("fip", "?")
        ip_v    = pitcher_stats.get("innings_pitched", 0)
        babip_v = pitcher_stats.get("babip")
        babip_s = f" · BABIP {babip_v:.3f}" if isinstance(babip_v, float) else ""
        xfip_s  = f"{xfip_v:.2f}" if isinstance(xfip_v, float) else str(xfip_v)
        era_s   = f"{era_v:.2f}" if isinstance(era_v, float) else str(era_v)
        ip_s    = f"{ip_v:.0f}" if isinstance(ip_v, float) else str(ip_v)
        signals.append(
            f"⚠ ERA trap [{_trap_label(sev)}] — {pitcher_name}: "
            f"ERA {era_s} vs xFIP {xfip_s} over {ip_s} IP{babip_s} "
            f"(severity {sev:.2f}) — {team_name} ML may be overpriced by market"
        )

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

        # ── Injury credibility cap (MLB) ──────────────────────────────────────
        # Same logic as NBA: when a team's injury adjustment ≥ _INJURY_GATE,
        # cap its model probability near market so stale season stats don't
        # produce phantom edges caused by missing key players.
        home_injury_capped = False
        away_injury_capped = False

        if home_inj >= _INJURY_GATE:
            max_home_prob = min(market_home_prob * (1 + _INJURY_MARGIN), 0.85)
            if model_home_prob > max_home_prob:
                model_home_prob = max_home_prob
                model_away_prob = 1 - model_home_prob
                home_injury_capped = True
                research.append(
                    f"⚠ {home} injury credibility cap applied — model probability "
                    f"anchored near market ({max_home_prob*100:.0f}%)"
                )

        if away_inj >= _INJURY_GATE:
            max_away_prob = min(market_away_prob * (1 + _INJURY_MARGIN), 0.85)
            if model_away_prob > max_away_prob:
                model_away_prob = max_away_prob
                model_home_prob = 1 - model_away_prob
                away_injury_capped = True
                research.append(
                    f"⚠ {away} injury credibility cap applied — model probability "
                    f"anchored near market ({max_away_prob*100:.0f}%)"
                )

        home_edge = model_home_prob - market_home_prob
        away_edge = model_away_prob - market_away_prob

        if home_edge >= _min and has_positive_ev(model_home_prob, market_home_prob):
            sizing = robinhood_kelly(model_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                # own_trap = home pitcher trap hurts home ML confidence
                # opp_trap = away pitcher trap boosts home ML edge reliability
                conf = _mlb_conf(home_edge, len(signals), stats_available,
                                 own_trap_sev=home_trap_sev, opp_trap_sev=away_trap_sev,
                                 injury_capped=home_injury_capped)
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=model_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= _min and has_positive_ev(model_away_prob, market_away_prob):
            sizing = robinhood_kelly(model_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                # own_trap = away pitcher trap hurts away ML confidence
                # opp_trap = home pitcher trap boosts away ML edge reliability
                conf = _mlb_conf(away_edge, len(signals), stats_available,
                                 own_trap_sev=away_trap_sev, opp_trap_sev=home_trap_sev,
                                 injury_capped=away_injury_capped)
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=model_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Run Line (Spread) ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", 0.0)   # almost always ±1.5 in MLB

            # Market run-line probability comes directly from the Odds API spread market
            # (consensus no-vig across all books), same as the moneyline market probability.
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            # Derive effective run margin from the (already injury-capped) win probability.
            effective_margin = float(norm.ppf(model_home_prob)) * MLB_SPREAD_STD

            # P(home covers run line) = P(actual margin > −home_spread_line)
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, MLB_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover

            away_spread_line = -home_spread_line

            home_rl_edge = model_home_cover - market_home_cover
            away_rl_edge = model_away_cover - market_away_cover

            if home_rl_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _mlb_conf(home_rl_edge, len(signals), stats_available,
                                     own_trap_sev=home_trap_sev, opp_trap_sev=away_trap_sev,
                                     injury_capped=home_injury_capped)
                    recs.append(BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_rl_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_rl_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _mlb_conf(away_rl_edge, len(signals), stats_available,
                                     own_trap_sev=away_trap_sev, opp_trap_sev=home_trap_sev,
                                     injury_capped=away_injury_capped)
                    recs.append(BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_rl_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
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

        # Use .5 lines to eliminate push risk: "Over 10" → "Over 9.5", "Under 8" → "Under 8.5"
        _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
        _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

        if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
            sizing = robinhood_kelly(model_over_prob, market_over_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=_over_label,
                    market_prob=market_over_prob, model_prob=model_over_prob,
                    edge=over_edge, contract_price=market_over_prob,
                    sizing=sizing,
                    confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))
        if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
            sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
            if sizing.num_contracts > 0:
                recs.append(BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=_under_label,
                    market_prob=market_under_prob, model_prob=1 - model_over_prob,
                    edge=under_edge, contract_price=market_under_prob,
                    sizing=sizing,
                    confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

    return recs


# ---------------------------------------------------------------------------
# NFL Edge Finder
# ---------------------------------------------------------------------------

# NFL point differential std ≈ 14 pts (covers range).  Home field ~3 pts.
NFL_SPREAD_STD   = 14.0
NFL_TOTAL_STD    = 10.0
NFL_HOME_ADV     = 0.025   # ~2.5% residual beyond what market prices
NFL_BYE_BONUS    = 0.025   # coming off a bye week (14 days rest)
NFL_RECENT_WEIGHT = 0.35   # weight toward last-14d form vs season avg


def _is_nfl_playoff(dt=None) -> bool:
    """NFL playoffs: mid-January through early February."""
    d = (dt or datetime.now(timezone.utc)).date()
    return (d.month == 1 and d.day >= 13) or (d.month == 2 and d.day <= 15)


def _nfl_team_strength(team: str, ctx: dict, playoff: bool = False) -> float:
    season = ctx["season_stats"].get(team, {})
    recent = ctx["recent_form"].get(team, {})
    season_net = season.get("net_rtg", 0.0)
    recent_net = recent.get("recent_net_rtg", season_net)
    w = 0.50 if playoff else NFL_RECENT_WEIGHT
    return (1 - w) * season_net + w * recent_net


def _nfl_margin_to_prob(expected_margin: float) -> float:
    return float(norm.cdf(expected_margin, 0, NFL_SPREAD_STD))


def analyze_nfl_game(game: Dict, nfl_ctx: Dict, nfl_injuries: Dict, min_edge: float = None) -> List[BetRecommendation]:
    from src.data.nfl_stats import normalize as nfl_normalize
    home = nfl_normalize(game["home_team"])
    away = nfl_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    playoff = _is_nfl_playoff()
    stats_available = bool(nfl_ctx["season_stats"].get(home) or nfl_ctx["season_stats"].get(away))

    home_rest = nfl_ctx["rest_days"].get(home, 7)
    away_rest = nfl_ctx["rest_days"].get(away, 7)

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_stats  = nfl_ctx["season_stats"].get(home, {})
        away_stats  = nfl_ctx["season_stats"].get(away, {})
        home_recent = nfl_ctx["recent_form"].get(home, {})
        away_recent = nfl_ctx["recent_form"].get(away, {})

        home_strength = _nfl_team_strength(home, nfl_ctx, playoff)
        away_strength = _nfl_team_strength(away, nfl_ctx, playoff)
        base_margin = home_strength - away_strength

        signals = []
        research = []
        adj = 0.0

        if not stats_available:
            signals.append("⚠ NFL stats unavailable — using market baseline only")

        # Home field advantage
        adj += NFL_HOME_ADV
        signals.append(f"Home field advantage: {home} (+{NFL_HOME_ADV*100:.0f}%)")

        # Team strength research
        if home_stats:
            research.append(
                f"{home}: {home_stats.get('ppg', '?'):.1f} PPG | "
                f"{home_stats.get('oppg', '?'):.1f} OPP PPG | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.1f}"
            )
        if away_stats:
            research.append(
                f"{away}: {away_stats.get('ppg', '?'):.1f} PPG | "
                f"{away_stats.get('oppg', '?'):.1f} OPP PPG | "
                f"NetRtg {away_stats.get('net_rtg', '?'):.1f}"
            )

        # Recent form
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

        # Net rating edge
        if abs(home_strength - away_strength) > 4:
            stronger = home if home_strength > away_strength else away
            signals.append(
                f"Rating edge: {stronger} (blended PPG-diff {home_strength - away_strength:+.1f})"
            )

        # Bye week bonus (14+ days rest)
        if home_rest >= 14:
            adj += NFL_BYE_BONUS
            signals.append(f"Bye week: {home} coming off bye (+{NFL_BYE_BONUS*100:.0f}%)")
        if away_rest >= 14:
            adj -= NFL_BYE_BONUS
            signals.append(f"Bye week: {away} coming off bye — favors {away} (+{NFL_BYE_BONUS*100:.0f}%)")

        research.append(f"Rest days — {home}: {home_rest} | {away}: {away_rest}")

        # Playoff context
        if playoff:
            signals.append("🏆 Playoffs: recent form weighted 50%")

        # Injuries
        home_inj = injury_adjustment(home, nfl_injuries, "nfl")
        away_inj = injury_adjustment(away, nfl_injuries, "nfl")
        home_inj_list = nfl_injuries.get(home, [])
        away_inj_list = nfl_injuries.get(away, [])

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

        adjusted_home_prob = min(0.90, max(0.10, _nfl_margin_to_prob(base_margin) + adj))
        adjusted_away_prob = 1 - adjusted_home_prob

        # Injury credibility cap
        home_injury_capped = False
        away_injury_capped = False

        if home_inj >= _INJURY_GATE:
            max_home_prob = min(market_home_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_home_prob > max_home_prob:
                adjusted_home_prob = max_home_prob
                adjusted_away_prob = 1 - adjusted_home_prob
                home_injury_capped = True

        if away_inj >= _INJURY_GATE:
            max_away_prob = min(market_away_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_away_prob > max_away_prob:
                adjusted_away_prob = max_away_prob
                adjusted_home_prob = 1 - adjusted_away_prob
                away_injury_capped = True

        home_edge = adjusted_home_prob - market_home_prob
        away_edge = adjusted_away_prob - market_away_prob

        if home_edge >= _min and has_positive_ev(adjusted_home_prob, market_home_prob):
            sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(home_edge, len(signals), stats_available)
                if home_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NFL", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=adjusted_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= _min and has_positive_ev(adjusted_away_prob, market_away_prob):
            sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(away_edge, len(signals), stats_available)
                if away_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NFL", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=adjusted_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Spread ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", 0.0)
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            effective_margin = float(norm.ppf(adjusted_home_prob)) * NFL_SPREAD_STD
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, NFL_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover
            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NFL", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NFL", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    # --- Total ---
    total = game.get("total")
    if total and nfl_ctx["season_stats"]:
        home_stats = nfl_ctx["season_stats"].get(home, {})
        away_stats = nfl_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            exp_home = (home_stats.get("ppg", 21.0) + away_stats.get("oppg", 21.0)) / 2
            exp_away = (away_stats.get("ppg", 21.0) + home_stats.get("oppg", 21.0)) / 2
            expected_total = exp_home + exp_away

            market_line = total["line"]
            model_over_prob  = float(1 - norm.cdf(market_line, expected_total, NFL_TOTAL_STD))
            market_over_prob  = total["over_prob"]
            market_under_prob = total["under_prob"]

            total_signals = [
                f"Model projected score: {home} {exp_home:.0f} — {away} {exp_away:.0f}",
                f"Model expected total: {expected_total:.1f} vs market line {market_line}",
            ]
            total_research = [
                f"{home} PPG: {home_stats.get('ppg','?'):.1f} vs {away} OPP PPG: {away_stats.get('oppg','?'):.1f}",
                f"{away} PPG: {away_stats.get('ppg','?'):.1f} vs {home} OPP PPG: {home_stats.get('oppg','?'):.1f}",
            ]

            _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
            _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

            over_edge  = model_over_prob - market_over_prob
            under_edge = (1 - model_over_prob) - market_under_prob

            if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NFL", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NFL", game=label, bet_type="Total", pick=_under_label,
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    return recs


# ---------------------------------------------------------------------------
# NHL Edge Finder
# ---------------------------------------------------------------------------

# NHL goal differential std ≈ 1.7 goals.  Puck line = always ±1.5.
NHL_SPREAD_STD   = 1.7
NHL_TOTAL_STD    = 1.3
NHL_HOME_ADV     = 0.020   # ~2% residual beyond market
NHL_B2B_PENALTY  = 0.025   # back-to-back (1 day rest) penalty
NHL_RECENT_WEIGHT = 0.40   # NHL form is more volatile, weight recent more


def _is_nhl_playoff(dt=None) -> bool:
    """NHL playoffs: mid-April through mid-June."""
    d = (dt or datetime.now(timezone.utc)).date()
    return (d.month == 4 and d.day >= 15) or d.month in (5, 6)


def _nhl_team_strength(team: str, ctx: dict, playoff: bool = False) -> float:
    season = ctx["season_stats"].get(team, {})
    recent = ctx["recent_form"].get(team, {})
    season_net = season.get("net_rtg", 0.0)
    recent_net = recent.get("recent_net_rtg", season_net)
    w = 0.55 if playoff else NHL_RECENT_WEIGHT
    return (1 - w) * season_net + w * recent_net


def _nhl_margin_to_prob(expected_margin: float) -> float:
    return float(norm.cdf(expected_margin, 0, NHL_SPREAD_STD))


def analyze_nhl_game(game: Dict, nhl_ctx: Dict, nhl_injuries: Dict, min_edge: float = None) -> List[BetRecommendation]:
    from src.data.nhl_stats import normalize as nhl_normalize
    home = nhl_normalize(game["home_team"])
    away = nhl_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    playoff = _is_nhl_playoff()
    stats_available = bool(nhl_ctx["season_stats"].get(home) or nhl_ctx["season_stats"].get(away))

    home_rest = nhl_ctx["rest_days"].get(home, 2)
    away_rest = nhl_ctx["rest_days"].get(away, 2)

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        home_stats  = nhl_ctx["season_stats"].get(home, {})
        away_stats  = nhl_ctx["season_stats"].get(away, {})
        home_recent = nhl_ctx["recent_form"].get(home, {})
        away_recent = nhl_ctx["recent_form"].get(away, {})

        home_strength = _nhl_team_strength(home, nhl_ctx, playoff)
        away_strength = _nhl_team_strength(away, nhl_ctx, playoff)
        base_margin = home_strength - away_strength

        signals = []
        research = []
        adj = 0.0

        if not stats_available:
            signals.append("⚠ NHL stats unavailable — using market baseline only")

        # Home ice advantage
        adj += NHL_HOME_ADV
        signals.append(f"Home ice advantage: {home} (+{NHL_HOME_ADV*100:.0f}%)")

        # Team stats
        if home_stats:
            research.append(
                f"{home}: {home_stats.get('gpg', '?'):.2f} GPG | "
                f"{home_stats.get('gapg', '?'):.2f} GAPG | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.2f}"
            )
        if away_stats:
            research.append(
                f"{away}: {away_stats.get('gpg', '?'):.2f} GPG | "
                f"{away_stats.get('gapg', '?'):.2f} GAPG | "
                f"NetRtg {away_stats.get('net_rtg', '?'):.2f}"
            )

        # Recent form
        if home_recent:
            research.append(
                f"{home} last 14 days: NetRtg {home_recent.get('recent_net_rtg', '?'):.2f} | "
                f"Win% {home_recent.get('recent_w_pct', 0)*100:.0f}%"
            )
        if away_recent:
            research.append(
                f"{away} last 14 days: NetRtg {away_recent.get('recent_net_rtg', '?'):.2f} | "
                f"Win% {away_recent.get('recent_w_pct', 0)*100:.0f}%"
            )

        # Net rating signal
        if abs(home_strength - away_strength) > 0.3:
            stronger = home if home_strength > away_strength else away
            signals.append(
                f"Rating edge: {stronger} (blended goal-diff {home_strength - away_strength:+.2f})"
            )

        # Back-to-back (NHL plays many B2Bs — 1 day rest is common)
        if home_rest == 1:
            adj -= NHL_B2B_PENALTY
            signals.append(f"{home} on back-to-back (-{NHL_B2B_PENALTY*100:.0f}%)")
        if away_rest == 1:
            adj += NHL_B2B_PENALTY
            signals.append(f"{away} on back-to-back — favors {home} (+{NHL_B2B_PENALTY*100:.0f}%)")

        research.append(f"Rest days — {home}: {home_rest} | {away}: {away_rest}")

        # Playoff context
        if playoff:
            signals.append("🏆 Playoffs: recent form weighted 55%, tighter defence expected")

        # Injuries
        home_inj = injury_adjustment(home, nhl_injuries, "nhl")
        away_inj = injury_adjustment(away, nhl_injuries, "nhl")
        home_inj_list = nhl_injuries.get(home, [])
        away_inj_list = nhl_injuries.get(away, [])

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

        adjusted_home_prob = min(0.90, max(0.10, _nhl_margin_to_prob(base_margin) + adj))
        adjusted_away_prob = 1 - adjusted_home_prob

        # Injury credibility cap
        home_injury_capped = False
        away_injury_capped = False

        if home_inj >= _INJURY_GATE:
            max_home_prob = min(market_home_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_home_prob > max_home_prob:
                adjusted_home_prob = max_home_prob
                adjusted_away_prob = 1 - adjusted_home_prob
                home_injury_capped = True

        if away_inj >= _INJURY_GATE:
            max_away_prob = min(market_away_prob * (1 + _INJURY_MARGIN), 0.90)
            if adjusted_away_prob > max_away_prob:
                adjusted_away_prob = max_away_prob
                adjusted_home_prob = 1 - adjusted_away_prob
                away_injury_capped = True

        home_edge = adjusted_home_prob - market_home_prob
        away_edge = adjusted_away_prob - market_away_prob

        if home_edge >= _min and has_positive_ev(adjusted_home_prob, market_home_prob):
            sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(home_edge, len(signals), stats_available)
                if home_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NHL", game=label, bet_type="Moneyline", pick=home,
                    market_prob=market_home_prob, model_prob=adjusted_home_prob,
                    edge=home_edge, contract_price=market_home_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        if away_edge >= _min and has_positive_ev(adjusted_away_prob, market_away_prob):
            sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
            if sizing.num_contracts > 0:
                conf = _confidence_label(away_edge, len(signals), stats_available)
                if away_injury_capped:
                    conf = "MEDIUM"
                recs.append(BetRecommendation(
                    sport="NHL", game=label, bet_type="Moneyline", pick=away,
                    market_prob=market_away_prob, model_prob=adjusted_away_prob,
                    edge=away_edge, contract_price=market_away_prob,
                    sizing=sizing, confidence=conf,
                    signals=signals[:], research=research[:],
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                ))

        # --- Puck line (NHL spread = always ±1.5) ---
        sp = game.get("spread")
        if sp:
            home_spread_line = sp.get("home_spread", -1.5)
            market_home_cover = sp.get("home_prob", 0.50)
            market_away_cover = sp.get("away_prob", 0.50)

            effective_margin = float(norm.ppf(adjusted_home_prob)) * NHL_SPREAD_STD
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, NHL_SPREAD_STD))
            model_away_cover = 1.0 - model_home_cover
            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NHL", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    recs.append(BetRecommendation(
                        sport="NHL", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))

    # --- Total (goals) ---
    total = game.get("total")
    if total and nhl_ctx["season_stats"]:
        home_stats = nhl_ctx["season_stats"].get(home, {})
        away_stats = nhl_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            exp_home = (home_stats.get("gpg", 3.0) + away_stats.get("gapg", 3.0)) / 2
            exp_away = (away_stats.get("gpg", 3.0) + home_stats.get("gapg", 3.0)) / 2
            if home_rest == 1:
                exp_home -= 0.20
            if away_rest == 1:
                exp_away -= 0.20
            expected_total = exp_home + exp_away

            market_line = total["line"]
            model_over_prob  = float(1 - norm.cdf(market_line, expected_total, NHL_TOTAL_STD))
            market_over_prob  = total["over_prob"]
            market_under_prob = total["under_prob"]

            total_signals = [
                f"Model projected score: {home} {exp_home:.1f} — {away} {exp_away:.1f}",
                f"Model expected total: {expected_total:.1f} vs market line {market_line}",
            ]
            total_research = [
                f"{home} GPG: {home_stats.get('gpg','?'):.2f} vs {away} GAPG: {away_stats.get('gapg','?'):.2f}",
                f"{away} GPG: {away_stats.get('gpg','?'):.2f} vs {home} GAPG: {home_stats.get('gapg','?'):.2f}",
            ]

            _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
            _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

            over_edge  = model_over_prob - market_over_prob
            under_edge = (1 - model_over_prob) - market_under_prob

            if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NHL", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    recs.append(BetRecommendation(
                        sport="NHL", game=label, bet_type="Total", pick=_under_label,
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    ))


# ---------------------------------------------------------------------------
# IPL — Indian Premier League cricket (moneyline / match-winner only)
# ---------------------------------------------------------------------------

# IPL home advantage: T20 home-pitch knowledge + crowd effect.
# Raw historical home win rate ≈ 55–57%; residual beyond market ≈ 2.5%.
# This is the base rate — scaled per-venue by home_adv_modifier in venue config.
IPL_HOME_ADV             = 0.025
# League-average home win rate used to calibrate live venue history.
IPL_LEAGUE_HOME_WIN_PCT  = 0.55
# Form weight: T20 is highly volatile — tilt heavily toward recent form.
IPL_RECENT_WEIGHT        = 0.65
# Playoff form weight: knockout pressure — lean even more on recent form.
IPL_PLAYOFF_RECENT_WEIGHT = 0.75
# Min matches at a venue before trusting live home_win_pct over config modifier.
IPL_VENUE_MIN_MATCHES    = 5
# H2H adjustment cap: ±4% at perfect dominance, scales linearly with H2H edge.
IPL_H2H_MAX_ADJ          = 0.04
# Rest penalty: applied when a team has had < 2 days since their last match.
IPL_SHORT_REST_PENALTY   = 0.015


def _is_ipl_playoff(dt: Optional[datetime] = None) -> bool:
    """IPL group stage ends ~May 18; knockouts run until ~May 26."""
    d = (dt or datetime.now(timezone.utc)).date()
    return d.month == 5 and d.day >= 18


def _ipl_team_strength(team: str, ctx: dict, playoff: bool = False) -> float:
    """
    Blended win rate for an IPL team.
    Playoff mode tilts more heavily toward recent form (last 7 days).
    Falls back to 0.5 (coin flip) when no data is available.
    """
    form = ctx["season_form"].get(team, {})
    if not form:
        return 0.5
    season_wpct = form.get("win_pct", 0.5)
    recent_wpct = form.get("recent_win_pct", season_wpct)
    recent_n    = form.get("recent_total", 0)
    weight = IPL_PLAYOFF_RECENT_WEIGHT if playoff else IPL_RECENT_WEIGHT
    w = weight if recent_n >= 2 else 0.0
    return (1 - w) * season_wpct + w * recent_wpct


def _ipl_venue_home_adv(venue_name: str, ipl_ctx: dict) -> float:
    """
    Returns venue-specific home advantage adjustment.

    Priority:
      1. Live season history at this venue (if ≥ IPL_VENUE_MIN_MATCHES games played).
         Uses the gap between observed home_win_pct and league average, scaled by 0.5
         to account for market pricing, added to the base IPL_HOME_ADV.
      2. Config modifier from ipl_venue_config.json (scales IPL_HOME_ADV).
      3. Flat IPL_HOME_ADV if no venue data at all.
    """
    if not venue_name:
        return IPL_HOME_ADV

    vstats = ipl_ctx.get("venue_stats", {}).get(venue_name, {})
    vcfg   = ipl_ctx.get("venue_config", {}).get(venue_name, {})

    n = vstats.get("total_matches", 0)
    if n >= IPL_VENUE_MIN_MATCHES:
        hist_home_pct = vstats.get("home_win_pct", IPL_LEAGUE_HOME_WIN_PCT)
        # Residual over league average, halved (market prices some of it already)
        return IPL_HOME_ADV + (hist_home_pct - IPL_LEAGUE_HOME_WIN_PCT) * 0.5

    modifier = vcfg.get("home_adv_modifier", 1.0)
    return IPL_HOME_ADV * modifier


def analyze_ipl_game(game: Dict, ipl_ctx: Dict, min_edge: float = None) -> List[BetRecommendation]:
    """
    Analyse a single IPL match for moneyline (match-winner) value.

    IPL on Robinhood is h2h only — no spreads or run-line bets.
    Returns at most one BetRecommendation per game (the side with edge).
    """
    from src.data.ipl_stats import normalize as ipl_normalize
    home = ipl_normalize(game["home_team"])
    away = ipl_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    ml = game.get("moneyline")
    if not ml:
        return recs

    market_home_prob = ml["home_prob"]
    market_away_prob = ml["away_prob"]

    playoff = _is_ipl_playoff()

    home_form = ipl_ctx["season_form"].get(home, {})
    away_form = ipl_ctx["season_form"].get(away, {})
    stats_available = bool(home_form or away_form)

    home_strength = _ipl_team_strength(home, ipl_ctx, playoff=playoff)
    away_strength = _ipl_team_strength(away, ipl_ctx, playoff=playoff)

    # Venue + match context
    venue_name  = ipl_ctx.get("match_venues", {}).get((home, away), "")
    vcfg        = ipl_ctx.get("venue_config", {}).get(venue_name, {})
    vstats      = ipl_ctx.get("venue_stats",  {}).get(venue_name, {})
    h2h         = ipl_ctx.get("h2h",          {}).get((home, away), {})
    match_flags = ipl_ctx.get("match_flags",  {}).get((home, away), {})
    is_night    = match_flags.get("is_night", True)  # default True (most IPL games are evening)

    signals  = []
    research = []

    if playoff:
        signals.append("IPL playoff — recent form weighted more heavily")
    if not stats_available:
        signals.append("⚠ IPL form data unavailable — using market baseline only")

    # Normalise team strengths → base win probability
    total = home_strength + away_strength
    base_home_prob = home_strength / total if total > 0 else 0.5

    adj = 0.0

    # --- Home venue advantage (venue-specific) ---
    home_adv = _ipl_venue_home_adv(venue_name, ipl_ctx)
    adj += home_adv
    venue_label = f" at {venue_name}" if venue_name else ""
    signals.append(f"Home venue advantage: {home}{venue_label} (+{home_adv*100:.1f}%)")

    # --- Short rest penalty ---
    home_rest = ipl_ctx["rest_days"].get(home, 3)
    away_rest = ipl_ctx["rest_days"].get(away, 3)
    if home_rest < 2:
        adj -= IPL_SHORT_REST_PENALTY
        signals.append(f"Short rest: {home} ({home_rest}d since last match, -{IPL_SHORT_REST_PENALTY*100:.1f}%)")
    if away_rest < 2:
        adj += IPL_SHORT_REST_PENALTY
        signals.append(f"Short rest: {away} ({away_rest}d since last match, +{IPL_SHORT_REST_PENALTY*100:.1f}% for {home})")
    research.append(f"Rest days — {home}: {home_rest} | {away}: {away_rest}")

    # --- Team form stats in research panel ---
    if home_form:
        avg_m = home_form.get("avg_margin", 0)
        research.append(
            f"{home}: {home_form.get('wins', '?')}W-{home_form.get('losses', '?')}L "
            f"({home_form.get('total', 0)} matches) | "
            f"Win% {home_form.get('win_pct', 0)*100:.0f}% | "
            f"Recent Win% {home_form.get('recent_win_pct', 0)*100:.0f}% "
            f"(last {home_form.get('recent_total', 0)} games) | "
            f"Avg run margin {avg_m:+.0f}"
        )
    if away_form:
        avg_m = away_form.get("avg_margin", 0)
        research.append(
            f"{away}: {away_form.get('wins', '?')}W-{away_form.get('losses', '?')}L "
            f"({away_form.get('total', 0)} matches) | "
            f"Win% {away_form.get('win_pct', 0)*100:.0f}% | "
            f"Recent Win% {away_form.get('recent_win_pct', 0)*100:.0f}% "
            f"(last {away_form.get('recent_total', 0)} games) | "
            f"Avg run margin {avg_m:+.0f}"
        )

    # --- Form differential signal ---
    strength_gap = home_strength - away_strength
    if abs(strength_gap) >= 0.10:
        stronger = home if strength_gap > 0 else away
        signals.append(f"Form edge: {stronger} (blended win-rate gap {abs(strength_gap)*100:.0f}%)")

    # --- Recent form divergence (hot vs cold team) ---
    home_recent = home_form.get("recent_win_pct", home_strength)
    away_recent = away_form.get("recent_win_pct", away_strength)
    if home_form.get("recent_total", 0) >= 2 and away_form.get("recent_total", 0) >= 2:
        recent_gap = home_recent - away_recent
        if abs(recent_gap) >= 0.30:
            hot  = home if recent_gap > 0 else away
            cold = away if recent_gap > 0 else home
            signals.append(f"Hot/cold streak: {hot} hot ({home_recent*100:.0f}% last 7d) vs {cold} cold ({away_recent*100:.0f}%)")

    # --- Head-to-head (current season) ---
    h2h_total = h2h.get("total", 0)
    if h2h_total >= 2:
        h2h_home_pct = h2h.get("home_h2h_pct", 0.5)
        h2h_adj = (h2h_home_pct - 0.5) * (IPL_H2H_MAX_ADJ * 2)  # ±4% at 100%/0%
        h2h_adj = max(-IPL_H2H_MAX_ADJ, min(IPL_H2H_MAX_ADJ, h2h_adj))
        if abs(h2h_adj) >= 0.01:
            adj += h2h_adj
            favoured = home if h2h_adj > 0 else away
            h2h_wins = h2h.get("home_wins") if h2h_adj > 0 else h2h.get("away_wins")
            signals.append(
                f"H2H edge: {favoured} leads {h2h_wins}-{h2h_total - h2h_wins} "
                f"this season ({h2h_adj*100:+.1f}%)"
            )

    # --- Venue research notes ---
    if venue_name:
        dew_risk = vcfg.get("dew_risk")
        boundary = vcfg.get("boundary_rating")
        n_at_venue        = vstats.get("total_matches", 0)
        avg_first_innings = vstats.get("avg_first_innings")
        chasing_win_pct   = vstats.get("chasing_win_pct")
        venue_hw          = vstats.get("home_win_pct")

        venue_notes = [f"Venue: {venue_name} ({'night' if is_night else 'day'} match)"]
        if dew_risk is not None:
            # Only flag dew for night matches — dew is irrelevant for day games
            if is_night:
                dew_label = "HIGH" if dew_risk >= 0.55 else "moderate" if dew_risk >= 0.35 else "low"
                venue_notes.append(f"dew risk {dew_label} ({dew_risk:.0%}) — favours chasing team")
            else:
                venue_notes.append("day match — dew not a factor")
        if boundary is not None:
            venue_notes.append(f"boundary rating {boundary:.2f}x")
        if avg_first_innings and n_at_venue > 0:
            venue_notes.append(f"avg 1st innings {avg_first_innings:.0f} runs ({n_at_venue} matches this season)")
        if chasing_win_pct is not None and n_at_venue >= 3:
            venue_notes.append(f"chasing win% {chasing_win_pct:.0%} this season")
        if venue_hw is not None and n_at_venue >= IPL_VENUE_MIN_MATCHES:
            venue_notes.append(f"home win% at venue {venue_hw:.0%}")
        research.append(" | ".join(venue_notes))

    adjusted_home_prob = min(0.90, max(0.10, base_home_prob + adj))
    adjusted_away_prob = 1.0 - adjusted_home_prob

    home_edge = adjusted_home_prob - market_home_prob
    away_edge = adjusted_away_prob - market_away_prob

    if home_edge >= _min and has_positive_ev(adjusted_home_prob, market_home_prob):
        sizing = robinhood_kelly(adjusted_home_prob, market_home_prob)
        if sizing.num_contracts > 0:
            recs.append(BetRecommendation(
                sport="IPL", game=label, bet_type="Moneyline", pick=home,
                market_prob=market_home_prob, model_prob=adjusted_home_prob,
                edge=home_edge, contract_price=market_home_prob,
                sizing=sizing,
                confidence=_confidence_label(home_edge, len(signals), stats_available),
                signals=signals[:], research=research[:],
                home_team=home, away_team=away, game_time=game_time,
                commence_time=commence_time,
            ))

    if away_edge >= _min and has_positive_ev(adjusted_away_prob, market_away_prob):
        sizing = robinhood_kelly(adjusted_away_prob, market_away_prob)
        if sizing.num_contracts > 0:
            recs.append(BetRecommendation(
                sport="IPL", game=label, bet_type="Moneyline", pick=away,
                market_prob=market_away_prob, model_prob=adjusted_away_prob,
                edge=away_edge, contract_price=market_away_prob,
                sizing=sizing,
                confidence=_confidence_label(away_edge, len(signals), stats_available),
                signals=signals[:], research=research[:],
                home_team=home, away_team=away, game_time=game_time,
                commence_time=commence_time,
            ))

    return recs
