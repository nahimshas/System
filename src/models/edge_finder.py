"""
Core probability model. Takes market odds + stats + injuries, returns edge analysis.

Approach:
  1. Use market implied probability (no-vig) as base
  2. Apply statistical adjustments for factors the market may have mispriced
  3. Compare adjusted probability to market — flag positive edges
"""
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple
from scipy.stats import norm

from src.config import (
    NBA_HOME_ADVANTAGE, NBA_BACK_TO_BACK_PENALTY, NBA_REST_BONUS_PER_DAY,
    NBA_RECENT_FORM_WEIGHT, NBA_TOTAL_STD, MLB_HOME_ADVANTAGE, MIN_EDGE,
    NBA_PLAYOFF_SCORING_FACTOR, NBA_PLAYOFF_PACE_FACTOR,
    NBA_PLAYOFF_RECENT_WEIGHT, NBA_PLAYOFF_TOTAL_STD,
    MLB_PLAYOFF_SCORING_FACTOR, MLB_PLAYOFF_STARTER_IP,
    MLB_PLAYOFF_RECENT_WEIGHT, MLB_RECENT_FORM_WEIGHT,
    NHL_PLAYOFF_SCORING_FACTOR,
    SCHEDULE_LOAD_THRESHOLDS,
)
from src.data.injuries import injury_adjustment
from src.data.nba_stats import normalize as nba_normalize
from src.models.kelly import robinhood_kelly, has_positive_ev, BetSizing

logger = logging.getLogger(__name__)

# Standard deviation of game point/run differentials used in norm.cdf win-prob model.
# NBA: real-world NBA point differential std ≈ 14 pts. Using 12 was too tight,
#      causing the model to be overconfident on lopsided matchups.
# MLB: std of 1.8 keeps the 60–70% probability range well-calibrated (confirmed by backtest).
#      The tanh cap below handles overconfidence at the high end without touching mid-range picks.
NBA_SPREAD_STD  = 12.0  # model uncertainty for point margin; 14.0 was too wide (suppressed all edges)
MLB_SPREAD_STD  = 2.2   # Jul 4 2026 optimization: 1.8 → 2.2 — flatter run_diff→prob mapping; 1.8 produced systematically overconfident mid-range probabilities (validated via decision-log reconstruction sweep)

# Probability→margin conversion for the NBA model. The residual adjustments
# (home court, B2B, rest, schedule load, injuries) are tuned as probability
# deltas near 50%. To add them in MARGIN space (so the single CDF compresses
# them correctly in the tails instead of overshooting) we divide by the central
# CDF slope = pdf(0). At σ=12 this is ≈30.1 pts per 1.00 prob, i.e. a 2.5% home
# edge ≈ 0.75 pts. By construction the conversion leaves mid-range (~50%) picks
# unchanged and only deflates the overconfident tails.
_NBA_PROB_TO_MARGIN = 1.0 / float(norm.pdf(0, 0, NBA_SPREAD_STD))
# Soft ceiling on raw run differential before norm.cdf conversion.
# When multiple factors stack (great SP + good bullpen + strong offense + injuries),
# the linear sum overshoots realistic expectation. tanh(x / CAP) * CAP compresses
# large differentials toward the cap using diminishing returns while leaving small
# differentials (< ~1.0 run) nearly untouched. This prevents norm.cdf from
# producing 85%+ probabilities that the backtest shows are only winning ~46%.
# At CAP=1.8: a raw diff of 2.5 runs → capped to 1.71 runs → ~83% max probability.
MLB_RUN_DIFF_CAP = 1.8

# Standard deviation for actual final-margin variance in an MLB game, used ONLY
# for the run line (+/-1.5) cover probability conversion. This is a SEPARATE
# concept from MLB_SPREAD_STD (1.8), which calibrates pitcher-quality →
# moneyline win probability. The actual SD of historical MLB final-margins is
# ~3.0–3.5 runs, so 1.5 runs represents ~0.4–0.5σ (not 0.83σ as the prior
# formula assumed). Using 1.8 here amplified ML→RL jumps to +24–32%; using 3.5
# brings them to +12–17%, matching the empirical sportsbook spread between ML
# and RL lines.
MLB_RUNLINE_SIGMA = 3.5

# Standard deviation for the GAME TOTAL (sum of both teams' runs) cover model.
# Previously this used MLB_SPREAD_STD * 1.5 = 2.7, which is tighter than even the
# run-line margin sigma (3.5) — backwards, since a sum has at least as much
# variance as a difference. The realised SD of MLB game totals is ~4 runs; 2.7
# made small projection-vs-line gaps look like large edges and manufactured
# false Over edges (backtest: model "Over 66%" picks realised ~39%). 3.8 sits
# just above the run-line sigma and better-calibrates the totals probability.
# Raised 3.8 → 4.3 (Jun 2026): the true SD of MLB game totals is ~4.3 runs;
# the tighter 3.8 manufactured phantom edges (model claimed +14% edge on totals
# it bet, realised ~50% win). Widening to the empirical SD deflates overconfident
# totals probabilities so fewer junk totals clear MIN_EDGE.
MLB_TOTAL_STD = 4.3

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

# General credibility caps ─────────────────────────────────────────────────────
# Standardised across sports as a safety net during the cold-start period
# before calibration data accumulates. Bounds how far model probability can
# drift from market on any pick (not just injury-triggered ones).
#
# Once the calibration engine has enough settled picks, it will auto-relax
# (or tighten) these caps based on counterfactual evidence comparing raw vs
# capped realised hit rates — see calibration system docs.
#
# Tighter for MLS due to small-sample noise in early-season xG data.
NBA_CRED_CAP  = 0.15
MLB_CRED_CAP  = 0.10   # Jul 4 2026 optimization: 0.15 → 0.10 — tighter market anchor (MLB model-market disagreements beyond 10pp were noise, not edge)
NFL_CRED_CAP  = 0.15
NHL_CRED_CAP  = 0.15
WNBA_CRED_CAP = 0.15
IPL_CRED_CAP  = 0.15
MLS_CRED_CAP  = 0.10


def _cred_cap(sport: str, default: float, cap_type: str = "credibility") -> float:
    """
    Return the effective credibility cap for `sport`/`cap_type`, dynamically
    adjusted by the cap auto-relaxation system. Falls back to the hardcoded
    default if cap_state isn't available — so the analyzer continues to work
    identically on a fresh install or if state files are missing.
    """
    try:
        from src.state.cap_state import get_current_cap
        return get_current_cap(sport, cap_type)
    except Exception:
        return default


def _stamp_recs_calibration(
    recs: List["BetRecommendation"],
    *,
    home_team: str = "",
    away_team: str = "",
    home_raw: Optional[float] = None,
    away_raw: Optional[float] = None,
    cred_fired: bool = False,
    hardcap_fired: bool = False,
    home_injury_fired: bool = False,
    away_injury_fired: bool = False,
    stats_avail: bool = True,
    game: Optional[Dict] = None,
) -> None:
    """
    Stamp per-game calibration metadata onto every rec just before return.

    Centralises the analyzer integration boilerplate so the shadow log can
    later run cap counterfactual analysis (raw vs capped realised rates).

    For ML/Spread picks, model_prob_raw is set to the home or away raw
    probability based on pick text. For Total and prop picks the raw value
    is stamped by bet-specific cap logic before this function is called;
    this function skips those recs (sentinel: model_prob_raw is not None).
    """
    # Per-game season type (stamped by src/data/game_types.py in main.py) —
    # carried onto every rec so the shadow log can segment CLV/calibration by
    # game type (regular vs play-in vs postseason) and answer empirically
    # whether finer model distinctions are warranted.
    _gt   = (game or {}).get("season_game_type", "") or ""
    _note = (game or {}).get("season_game_note", "") or ""

    for rec in recs:
        # Game-level flags — always stamp
        rec.hardcap_fired   = hardcap_fired
        rec.stats_available = stats_avail
        rec.game_type       = _gt
        if _note and not any(_note in r for r in rec.research):
            rec.research.append(f"Game type: {_note}")
        if _gt == "play_in" and not any("Play-In" in r for r in rec.research):
            rec.research.append("Game type: Play-In — playoff treatment applied")

        # Per-pick calibration — skip if already stamped by bet-specific cap logic
        if rec.model_prob_raw is None:
            bt   = rec.bet_type or ""
            pick = (rec.pick or "").strip()
            rec.credibility_cap_fired = cred_fired
            if bt in ("Moneyline", "Spread"):
                if home_team and pick.startswith(home_team):
                    rec.model_prob_raw   = home_raw
                    rec.injury_cap_fired = home_injury_fired
                elif away_team and pick.startswith(away_team):
                    rec.model_prob_raw   = away_raw
                    rec.injury_cap_fired = away_injury_fired
        # else: bet-specific cap already populated model_prob_raw / credibility_cap_fired


def _apply_credibility_cap(
    model_prob: float, market_prob: float, max_drift: float
) -> Tuple[float, bool]:
    """
    Mode 0 — Hard clip. Bound model probability to within ±max_drift of market.

    Sharp boundary at ±max_drift. The original / default cap form.

    Returns (capped_prob, fired) — `fired` is True if the value was actually
    clipped (used by the shadow log for cap counterfactual analysis).
    """
    hi = market_prob + max_drift
    lo = market_prob - max_drift
    if model_prob > hi:
        return hi, True
    if model_prob < lo:
        return lo, True
    return model_prob, False


def _apply_tanh_saturation_cap(
    model_prob: float, market_prob: float, max_drift: float
) -> Tuple[float, bool]:
    """
    Mode 1 — tanh saturation. Smooth pull toward market with diminishing returns.

    Unlike Mode 0's sharp clip, this never fully truncates the model's belief
    — it just bends extreme predictions toward market asymptotically. At small
    |raw - market| there's essentially no change; as |raw - market| grows the
    pull approaches max_drift but never exceeds it.

    Mathematically:
      delta  = raw - market
      smooth = max_drift × tanh(delta / max_drift)
      pulled = market + smooth

    The "fired" flag indicates whether the saturation meaningfully changed
    the value (>0.5pp difference) for counterfactual tracking parity with
    Mode 0.
    """
    delta  = model_prob - market_prob
    if max_drift <= 0:
        return model_prob, False
    smooth = max_drift * math.tanh(delta / max_drift)
    pulled = market_prob + smooth
    # Clamp into [0, 1] for safety (tanh keeps it bounded but be paranoid)
    pulled = max(0.0, min(1.0, pulled))
    fired  = abs(pulled - model_prob) > 0.005
    return pulled, fired


def _apply_credibility_cap_dispatched(
    model_prob: float, market_prob: float, max_drift: float, sport: str,
    cap_type: str = "credibility",
) -> Tuple[float, bool]:
    """
    Mode-aware credibility cap dispatcher.

    Looks up the current cap MODE for `sport`/`cap_type` from cap_state
    (0 = hard clip, 1 = tanh saturation, future: 2 = logistic blend) and
    applies the matching function. Falls back to Mode 0 if cap_state isn't
    available, so behaviour is identical to the original hard-clip
    implementation on fresh installs.
    """
    try:
        from src.state.cap_state import get_current_cap_mode
        mode = get_current_cap_mode(sport, cap_type)
    except Exception:
        mode = 0

    if mode == 1:
        return _apply_tanh_saturation_cap(model_prob, market_prob, max_drift)
    # Mode 0 (default) and any unknown mode → hard clip
    return _apply_credibility_cap(model_prob, market_prob, max_drift)

# Totals model helpers
_NBA_INJ_TO_PTS    = 75.0  # converts win-prob injury_adj → expected pts reduction in totals
_TOTAL_MARKET_ANCHOR = 0.40  # 40% weight toward market line — was 0.20; raised to neutralise
                             # the systematic upward (Over) projection bias measured in MLB totals
                             # (model "Over 66%" realised ~39%). The market total line is essentially
                             # unbiased, so anchoring harder removes the bias at its source while
                             # leaving genuine large model/market disagreements as (smaller) edges.


def _game_playoff(game: Optional[Dict], calendar_fallback) -> bool:
    """
    Per-game playoff flag. Prefers the ESPN-derived `season_game_type` stamped
    by src/data/game_types.py (handles mixed days — mid-April NBA has regular,
    play-in, and playoff games in the same week — and yearly schedule shifts).
    Unstamped games fall back to the legacy calendar window, so behavior on
    any data failure is exactly the pre-deployment behavior.
    """
    gt = (game or {}).get("season_game_type")
    if gt:
        return gt in ("postseason", "play_in", "superbowl")
    return calendar_fallback()


def _stamp_decision(game, min_edge, features, markets, recs=None):
    """
    Assemble the FULL candidate set (both sides of every market) + the structured
    model-input `features`, and stamp it onto game["_decision"] for the
    decision_log archive. Purely diagnostic — never affects pick selection.

    `markets` is an iterable of (market_type, side, model_prob, model_prob_raw,
    market_prob, line) tuples — model_prob is POST-credibility-cap, model_prob_raw
    is PRE-cap (pass None if unavailable). Sides with a missing post-cap prob or
    market prob are skipped. When `recs` is passed, each MADE candidate is also
    tagged with its confidence label, and its raw prob falls back to the rec if
    not supplied in the tuple. Fully exception-safe.
    """
    try:
        conf, rawp = {}, {}
        if recs:
            try:
                from src.state.shadow_log import _market_type as _mt, _pick_side as _ps
                for r in recs:
                    _k = (_mt(r), _ps(r))
                    conf[_k] = getattr(r, "confidence", None)
                    rawp[_k] = getattr(r, "model_prob_raw", None)
            except Exception:
                conf, rawp = {}, {}
        cands = []
        for mtype, side, mp, mp_raw, mk, line in markets:
            if mp is None or mk is None:
                continue
            mp_f, mk_f = float(mp), float(mk)
            edge = mp_f - mk_f
            # Raw (pre-credibility-cap) prob: prefer the value passed in the tuple
            # (available for BOTH made and rejected sides); fall back to the rec's
            # stamped raw for made picks. model_prob / edge are POST-cap.
            _raw = mp_raw if mp_raw is not None else rawp.get((mtype, side))
            _raw_f = float(_raw) if _raw is not None else None
            cands.append({
                "market_type": mtype,
                "side": side,
                "model_prob": round(mp_f, 4),                 # post-cap (acted on)
                "model_prob_raw": round(_raw_f, 4) if _raw_f is not None else None,
                "market_prob": round(mk_f, 4),
                "edge": round(edge, 4),                       # post-cap edge
                "raw_edge": round(_raw_f - mk_f, 4) if _raw_f is not None else None,
                "made": edge >= min_edge,
                "line": (round(float(line), 1) if line is not None else None),
                "confidence": conf.get((mtype, side)),
            })
        game["_decision"] = {"features": features or {}, "candidates": cands}
    except Exception:
        pass


def _is_nba_playoff(dt: Optional[datetime] = None) -> bool:
    """NBA playoffs: mid-April through mid-June. (Calendar FALLBACK — primary
    signal is the per-game season type; see _game_playoff.)"""
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

    # ── Calibration metadata (used by shadow log → calibration engine) ──
    # These fields are optional and back-compat — defaults preserve old
    # behaviour. Sport analyzers populate them when they apply caps so the
    # calibration system can later run counterfactual analysis (raw vs
    # capped realised rates).
    model_prob_raw: Optional[float] = None       # pre-any-cap model probability
    model_prob_calibrated: Optional[float] = None  # market + edge × calibration ratio — what Kelly sizes on (set in main.py)
    credibility_cap_fired: bool = False          # general ±cred-cap fired
    injury_cap_fired:      bool = False          # injury cap fired
    hardcap_fired:         bool = False          # [lo, hi] hard prob bound fired
    stats_available:       bool = True           # sport stats were available for this game
    game_pk:               str  = ""            # MLB game primary key — for live pitcher/batter lookup
    dog_better_starter:    bool = False          # MLB run-line dog backed by the stronger starter (validated pattern — confidence-promoted; tracked in shadow/decision logs)
    game_type:             str  = ""            # per-game season type (regular/play_in/postseason/superbowl)


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


def _nba_blend_per100(season_rtg: float, recent_per_game, pace, w: float) -> float:
    """
    Blend a season per-100-possessions rating with recent per-GAME scoring.

    off_rtg/def_rtg are points per 100 possessions; recent_ppg/recent_oppg are
    points per game. Blending them directly (the old code) mixed units — it only
    looked fine because NBA pace ≈ 100. We convert the recent per-game value to
    per-100 (× 100 / pace) before blending so the result is a clean per-100
    rating, which the caller then re-paces to the matchup via × avg_pace / 100.
    Falls back to the season rating when no recent value is available.
    """
    if recent_per_game is None:
        return season_rtg
    p = pace if (pace and pace > 0) else 100.0
    recent_per100 = recent_per_game * 100.0 / p
    return (1 - w) * season_rtg + w * recent_per100


def analyze_nba_game(game: Dict, nba_ctx: Dict, nba_injuries: Dict, min_edge: float = None) -> List[BetRecommendation]:
    home = nba_normalize(game["home_team"])
    away = nba_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    playoff = _game_playoff(game, _is_nba_playoff)
    stats_available = bool(nba_ctx["season_stats"].get(home) or nba_ctx["season_stats"].get(away))

    # Pre-initialise so Total block can reference these even if ml block is skipped
    home_rest = nba_ctx["rest_days"].get(home, 1)
    away_rest = nba_ctx["rest_days"].get(away, 1)
    home_inj = injury_adjustment(home, nba_injuries, "nba")
    away_inj = injury_adjustment(away, nba_injuries, "nba")

    # ── Pre-compute projected score shared by all bet types ───────────────────
    # Uses the blended formula (season + recent form + B2B + injury pts) so
    # ML, Spread, and Total cards always show the same number for a given game.
    _proj_score_signal: Optional[str] = None
    _home_stats_pre = nba_ctx["season_stats"].get(home, {})
    _away_stats_pre = nba_ctx["season_stats"].get(away, {})
    if _home_stats_pre and _away_stats_pre:
        _hrf_pre = nba_ctx["recent_form"].get(home, {})
        _arf_pre = nba_ctx["recent_form"].get(away, {})
        _w_pre = NBA_PLAYOFF_RECENT_WEIGHT if playoff else NBA_RECENT_FORM_WEIGHT
        _home_pace_pre = _home_stats_pre.get("pace", 100)
        _away_pace_pre = _away_stats_pre.get("pace", 100)
        _home_off_pre = _nba_blend_per100(_home_stats_pre.get("off_rtg", 110), _hrf_pre.get("recent_ppg"),  _home_pace_pre, _w_pre)
        _home_def_pre = _nba_blend_per100(_home_stats_pre.get("def_rtg", 110), _hrf_pre.get("recent_oppg"), _home_pace_pre, _w_pre)
        _away_off_pre = _nba_blend_per100(_away_stats_pre.get("off_rtg", 110), _arf_pre.get("recent_ppg"),  _away_pace_pre, _w_pre)
        _away_def_pre = _nba_blend_per100(_away_stats_pre.get("def_rtg", 110), _arf_pre.get("recent_oppg"), _away_pace_pre, _w_pre)
        _avg_pace_pre = (_home_stats_pre.get("pace", 100) + _away_stats_pre.get("pace", 100)) / 2
        if playoff:
            _avg_pace_pre *= NBA_PLAYOFF_PACE_FACTOR
        _proj_home = (_home_off_pre + _away_def_pre) / 2 * _avg_pace_pre / 100
        _proj_away = (_away_off_pre + _home_def_pre) / 2 * _avg_pace_pre / 100
        if home_rest == 0:
            _proj_home -= 4.0
        if away_rest == 0:
            _proj_away -= 4.0
        _inj_home_pts_pre = min(home_inj * _NBA_INJ_TO_PTS, 6.0)
        _inj_away_pts_pre = min(away_inj * _NBA_INJ_TO_PTS, 6.0)
        if _inj_home_pts_pre > 0.5:
            _proj_home -= _inj_home_pts_pre
        if _inj_away_pts_pre > 0.5:
            _proj_away -= _inj_away_pts_pre
        _proj_score_signal = f"Model projected score: {home} {_proj_home:.0f} — {away} {_proj_away:.0f}"

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
            _h_w, _h_l = home_stats.get('wins', 0), home_stats.get('losses', 0)
            _h_rec = f"{_h_w}W-{_h_l}L | " if _h_w + _h_l > 0 else ""
            research.append(
                f"{home}: {_h_rec}OffRtg {home_stats.get('off_rtg', '?'):.1f} | "
                f"DefRtg {home_stats.get('def_rtg', '?'):.1f} | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.1f}"
            )
        if away_stats:
            _a_w, _a_l = away_stats.get('wins', 0), away_stats.get('losses', 0)
            _a_rec = f"{_a_w}W-{_a_l}L | " if _a_w + _a_l > 0 else ""
            research.append(
                f"{away}: {_a_rec}OffRtg {away_stats.get('off_rtg', '?'):.1f} | "
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

        # Rest-difference bonus: only applied when neither team is on a B2B.
        # If either team has 0 rest, the B2B penalty above already captures that
        # team's fatigue disadvantage — adding a rest_diff on top double-counts it.
        # Cap symmetrically at ±3 days (beyond 3 days, additional rest adds nothing).
        if home_rest > 0 and away_rest > 0:
            rest_diff = max(min(home_rest - away_rest, 3), -3)
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
        # home_inj / away_inj pre-computed above (shared with totals block)
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

        # --- Model projected score (pre-computed above; same formula as Total block) ---
        if _proj_score_signal:
            signals.append(_proj_score_signal)

        # Calibration capture: raw prob (pre-any-cap) + cap firing trackers.
        # Stamped onto each rec at the end so the shadow log can later run
        # cap counterfactual analysis.
        #
        # Add the accumulated adjustments in MARGIN space, then convert with a
        # single CDF. Previously `adj` was added in probability space on top of
        # the CDF output, which overshoots in the tails (a flat +X% means far
        # more in margin terms at 80% than at 50%) — the mechanical source of the
        # high-edge overconfidence. Converting via the central slope leaves
        # mid-range picks unchanged and compresses only the tails. The spread
        # block below derives its margin from this prob via inverse-CDF, so it
        # stays consistent automatically.
        _nba_raw_home = _nba_margin_to_prob(base_margin + adj * _NBA_PROB_TO_MARGIN)
        _nba_hardcap_fired = (_nba_raw_home > 0.90) or (_nba_raw_home < 0.10)
        adjusted_home_prob = min(0.90, max(0.10, _nba_raw_home))
        adjusted_away_prob = 1 - adjusted_home_prob

        # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
        adjusted_home_prob, _nba_cred_fired = _apply_credibility_cap_dispatched(
            adjusted_home_prob, market_home_prob, _cred_cap("nba", NBA_CRED_CAP, "credibility_moneyline"), "nba", "credibility_moneyline"
        )
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

            # Apply spread credibility cap
            _sp_raw_home = model_home_cover
            _sp_raw_away = model_away_cover
            model_home_cover, _sp_cred_home = _apply_credibility_cap_dispatched(
                model_home_cover, market_home_cover, _cred_cap("nba", NBA_CRED_CAP, "credibility_spread"), "nba", "credibility_spread"
            )
            model_away_cover, _sp_cred_away = _apply_credibility_cap_dispatched(
                model_away_cover, market_away_cover, _cred_cap("nba", NBA_CRED_CAP, "credibility_spread"), "nba", "credibility_spread"
            )

            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    _sp_rec = BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _sp_rec.model_prob_raw = _sp_raw_home
                    _sp_rec.credibility_cap_fired = _sp_cred_home
                    recs.append(_sp_rec)

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    _sp_rec = BetRecommendation(
                        sport="NBA", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _sp_rec.model_prob_raw = _sp_raw_away
                    _sp_rec.credibility_cap_fired = _sp_cred_away
                    recs.append(_sp_rec)

    # --- Total ---
    total = game.get("total")
    if total and nba_ctx["season_stats"]:
        home_stats = nba_ctx["season_stats"].get(home, {})
        away_stats = nba_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            # Blend season per-100 ratings with recent per-GAME form, unit-corrected.
            # recent_form carries recent_ppg / recent_oppg (per game) from the
            # 14-day window; _nba_blend_per100 converts them to per-100 before
            # blending so the result is a clean per-100 rating (same weight as the
            # ML/Spread strength model).
            home_recent_form = nba_ctx["recent_form"].get(home, {})
            away_recent_form = nba_ctx["recent_form"].get(away, {})
            w = NBA_PLAYOFF_RECENT_WEIGHT if playoff else NBA_RECENT_FORM_WEIGHT
            home_pace = home_stats.get("pace", 100)
            away_pace = away_stats.get("pace", 100)
            home_off = _nba_blend_per100(home_stats.get("off_rtg", 110), home_recent_form.get("recent_ppg"),  home_pace, w)
            home_def = _nba_blend_per100(home_stats.get("def_rtg", 110), home_recent_form.get("recent_oppg"), home_pace, w)
            away_off = _nba_blend_per100(away_stats.get("off_rtg", 110), away_recent_form.get("recent_ppg"),  away_pace, w)
            away_def = _nba_blend_per100(away_stats.get("def_rtg", 110), away_recent_form.get("recent_oppg"), away_pace, w)

            avg_pace = (home_stats.get("pace", 100) + away_stats.get("pace", 100)) / 2
            if playoff:
                avg_pace *= NBA_PLAYOFF_PACE_FACTOR
            expected_home_pts = (home_off + away_def) / 2 * avg_pace / 100
            expected_away_pts = (away_off + home_def) / 2 * avg_pace / 100
            # NBA_PLAYOFF_SCORING_FACTOR is intentionally NOT applied here.
            # The pace factor already captures fewer possessions; applying the scoring
            # factor on top creates a ~10% combined reduction that systematically
            # under-projects totals vs market lines. The market line already reflects
            # playoff defensive intensity.

            # Apply B2B penalty to individual team scores so per-team projection
            # and expected_total are always consistent with each other.
            b2b_teams = []
            if home_rest == 0:
                expected_home_pts -= 4.0
                b2b_teams.append(home)
            if away_rest == 0:
                expected_away_pts -= 4.0
                b2b_teams.append(away)

            # Injury impact on scoring — missing players reduce expected output.
            # Scale win-prob adjustment to estimated pts loss; residual after market repricing.
            inj_home_pts = min(home_inj * _NBA_INJ_TO_PTS, 6.0)
            inj_away_pts = min(away_inj * _NBA_INJ_TO_PTS, 6.0)
            if inj_home_pts > 0.5:
                expected_home_pts -= inj_home_pts
            if inj_away_pts > 0.5:
                expected_away_pts -= inj_away_pts

            expected_total = expected_home_pts + expected_away_pts

            market_line = total["line"]
            total_std   = NBA_PLAYOFF_TOTAL_STD if playoff else NBA_TOTAL_STD

            # Partial market anchor — blends model projection with market line to
            # dampen any remaining systematic bias without eliminating genuine edges.
            blended_total = (1 - _TOTAL_MARKET_ANCHOR) * expected_total + _TOTAL_MARKET_ANCHOR * market_line
            # Stamp onto game dict so props_analyzer can use the same projection
            game["model_total"]  = round(blended_total, 1)
            game["market_total"] = market_line
            game["home_inj_pts"] = round(inj_home_pts, 1)
            game["away_inj_pts"] = round(inj_away_pts, 1)
            model_over_prob = float(1 - norm.cdf(market_line, blended_total, total_std))
            market_over_prob = total["over_prob"]
            market_under_prob = total["under_prob"]

            _total_raw_over = model_over_prob
            model_over_prob, _total_cred_fired = _apply_credibility_cap_dispatched(
                model_over_prob, market_over_prob, _cred_cap("nba", NBA_CRED_CAP, "credibility_total"), "nba", "credibility_total"
            )

            base_total_signals = [
                _proj_score_signal or f"Model projected score: {home} {expected_home_pts:.0f} — {away} {expected_away_pts:.0f}",
                f"Model expected total: {blended_total:.1f} vs market line {market_line}",
                f"Combined pace: {avg_pace:.1f} possessions/game",
            ]
            total_research = [
                f"{home} off (blended): {home_off:.1f} vs {away} def (blended): {away_def:.1f}",
                f"{away} off (blended): {away_off:.1f} vs {home} def (blended): {home_def:.1f}",
                f"Recent form weight: {w*100:.0f}% | Season weight: {(1-w)*100:.0f}%",
            ]
            if b2b_teams:
                total_research.append(
                    f"B2B: {', '.join(b2b_teams)} — pace/scoring reduction applied to model (-4 pts per team)"
                )
            if inj_home_pts > 0.5:
                total_research.append(f"⚕ {home} injury drag: −{inj_home_pts:.1f} pts from totals projection")
            if inj_away_pts > 0.5:
                total_research.append(f"⚕ {away} injury drag: −{inj_away_pts:.1f} pts from totals projection")

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
                    _over_rec = BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(over_signals), stats_available),
                        signals=over_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _over_rec.model_prob_raw = _total_raw_over
                    _over_rec.credibility_cap_fired = _total_cred_fired
                    recs.append(_over_rec)
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    _under_rec = BetRecommendation(
                        sport="NBA", game=label, bet_type="Total", pick=_under_label,
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(under_signals), stats_available),
                        signals=under_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _under_rec.model_prob_raw = 1.0 - _total_raw_over
                    _under_rec.credibility_cap_fired = _total_cred_fired
                    recs.append(_under_rec)

    # Stamp per-game calibration metadata onto every rec so the shadow log
    # can later run cap counterfactual analysis. Vars may not all be defined
    # if the ml block was skipped — locals().get() handles that safely.
    _lv = locals()
    _home_raw = _lv.get("_nba_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home, away_team=away,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_nba_cred_fired", False),
        hardcap_fired=_lv.get("_nba_hardcap_fired", False),
        home_injury_fired=_lv.get("home_injury_capped", False),
        away_injury_fired=_lv.get("away_injury_capped", False),
        stats_avail=stats_available,
    )
    _nba_rh = _lv.get("_nba_raw_home"); _nba_ro = _lv.get("_total_raw_over")
    _stamp_decision(game, _min, {
        "home_strength": _lv.get("home_strength"), "away_strength": _lv.get("away_strength"),
        "base_margin": _lv.get("base_margin"), "effective_margin": _lv.get("effective_margin"),
        "win_prob_adj": _lv.get("adj"), "rest_diff": _lv.get("rest_diff"),
        "expected_total": _lv.get("blended_total"), "total_std": _lv.get("total_std"),
        "home_rest": _lv.get("home_rest"), "away_rest": _lv.get("away_rest"),
        "home_inj": _lv.get("home_inj"), "away_inj": _lv.get("away_inj"),
        "playoff": _lv.get("playoff"), "stats_available": stats_available,
    }, [
        ("Moneyline", home, _lv.get("adjusted_home_prob"), _nba_rh, _lv.get("market_home_prob"), None),
        ("Moneyline", away, _lv.get("adjusted_away_prob"),
         (1.0 - _nba_rh) if _nba_rh is not None else None, _lv.get("market_away_prob"), None),
        ("Spread", home, _lv.get("model_home_cover"), _lv.get("_sp_raw_home"),
         _lv.get("market_home_cover"), _lv.get("home_spread_line")),
        ("Spread", away, _lv.get("model_away_cover"), _lv.get("_sp_raw_away"),
         _lv.get("market_away_cover"),
         (-_lv["home_spread_line"]) if _lv.get("home_spread_line") is not None else None),
        ("Total", "over", _lv.get("model_over_prob"), _nba_ro, _lv.get("market_over_prob"), _lv.get("market_line")),
        ("Total", "under",
         (1.0 - _lv["model_over_prob"]) if _lv.get("model_over_prob") is not None else None,
         (1.0 - _nba_ro) if _nba_ro is not None else None,
         _lv.get("market_under_prob"), _lv.get("market_line")),
    ], recs)
    return recs


# ---------------------------------------------------------------------------
# MLB Edge Finder
# ---------------------------------------------------------------------------

def _pitcher_quality_score(stats: Dict) -> float:
    """
    Pitcher quality score feeding directly into run projections.

    Formula: 100 - 10(xFIP) - 5(BB/9) + 1.5(K/9) + 2*(ERA-xFIP)*min(1,IP/50)
    Normalized so a league-average pitcher scores 0 and the output range matches
    the prior (LEAGUE_AVG - xFIP) / 1.50 scale — no retuning of run-projection
    coefficients required.

    Inputs:
    - xFIP: already regressed toward league average by _blend_xfip (called before
      this function) — small-sample pitchers are pulled toward 4.20 automatically.
    - BB/9: walk rate penalises command-poor pitchers independently of xFIP.
    - K/9: strikeout rate rewards swing-and-miss stuff.
    - ERA trap (IP-scaled): when ERA diverges from xFIP over a meaningful sample,
      adjust the score. ERA > xFIP means the pitcher has been unlucky (boost score);
      ERA < xFIP means ERA looks better than true talent (reduce score).
      Scaled by min(1, IP/50) so tiny samples can't generate large trap bonuses.
    """
    LEAGUE_AVG_XFIP  = 4.20
    LEAGUE_AVG_BB9   = 3.0
    LEAGUE_AVG_K9    = 8.5
    FORMULA_BASELINE = 55.75   # raw score of a league-average pitcher
    FORMULA_SCALE    = 20.0    # normalises to same range as prior formula

    ip   = stats.get("innings_pitched", 0) or 0
    xfip = stats.get("xfip") or stats.get("fip", LEAGUE_AVG_XFIP)
    era  = stats.get("era")
    bb9  = stats.get("bb_per_9") or LEAGUE_AVG_BB9
    k9   = stats.get("k_per_9")  or LEAGUE_AVG_K9

    ip_conf = min(1.0, ip / 50.0)

    # Regress BB/9 and K/9 toward league average at small samples — same logic
    # as _blend_xfip does for xFIP. K% stabilises ~20 IP, BB% ~50 IP; using
    # ip_conf for both is conservative but consistent.
    eff_bb9 = ip_conf * float(bb9) + (1.0 - ip_conf) * LEAGUE_AVG_BB9
    eff_k9  = ip_conf * float(k9)  + (1.0 - ip_conf) * LEAGUE_AVG_K9

    era_trap = 0.0
    if isinstance(era, float) and ip >= 10:
        era_trap = 2.0 * (float(era) - float(xfip)) * ip_conf

    raw = (100.0
           - 10.0 * float(xfip)
           - 5.0  * eff_bb9
           + 1.5  * eff_k9
           + era_trap)

    quality = (raw - FORMULA_BASELINE) / FORMULA_SCALE

    # Short-leash penalty: a starter who is consistently pulled before facing the
    # lineup twice is performing below what xFIP captures. Stat-based regression
    # gives them league-average quality; persistent early exits are an additional
    # signal that their stuff isn't working in game contexts.
    # Penalty = (avg_ip - 4.0) / 4.0 — ranges from 0 at 4 IP to -0.25 at 3 IP.
    # Requires ≥ 3 starts so the pattern is established, not a one-off.
    gs   = int(stats.get("games_started", 0) or 0)
    avip = stats.get("avg_ip_per_start")
    if gs >= 3 and avip is not None and float(avip) < 4.0:
        quality += (float(avip) - 4.0) / 4.0

    return quality


def _era_trap_severity(stats: Dict) -> float:
    """
    Continuous ERA-trap severity score.

    Combines four factors:
      - ERA gap:    how much xFIP exceeds ERA (core signal)
      - ip_conf:    confidence ramp — grows from 0 at 10 IP to 1.0 at 30+ IP.
                    Larger samples mean we trust the gap more, not less.
                    (Replaces the old ip_weight that counter-intuitively amplified
                    small samples, treating uncertainty as danger.)
      - BABIP mult: BABIP well below league average (.300) confirms luck
      - K/9 guard:  high strikeout pitchers legitimately outperform xFIP because
                    strikeouts are never balls in play — fewer BIP means lower BABIP
                    independent of luck. Each K/9 above 9.0 reduces severity by 25%,
                    floored at 50% so extreme K rates can't zero out a real gap.

    Rough thresholds (after all factors applied):
      < 0.15  → negligible (ignore)
      0.15–0.40 → MILD     (display in research only)
      0.40–0.80 → MODERATE (cap confidence on trap team; flag opponent edge)
      > 0.80  → SEVERE    (strong opponent edge signal; lower edge threshold)

    Elite-pitcher guard:
      When xFIP itself is below 3.20, the pitcher has genuinely dominant
      underlying stuff — severity capped at MODERATE (0.79) to prevent false
      SEVERE tags on pitchers like Ohtani.
    """
    era   = stats.get("era")
    fip   = stats.get("fip")
    xfip  = stats.get("xfip") or fip or 4.20
    babip = stats.get("babip")
    ip    = stats.get("innings_pitched", 0)
    k9    = stats.get("k_per_9", 8.5)

    if not isinstance(era, float) or not isinstance(ip, float) or ip < 10:
        return 0.0

    # Prerequisite: pitcher must be outperforming their FIP (ERA < FIP).
    # Without this gate, pitchers like Wheeler (ERA > FIP but xFIP > ERA due to
    # HR/FB luck) trigger a false trap — the market already prices in the FIP gap.
    if isinstance(fip, float) and era >= fip:
        return 0.0

    xfip_val = float(xfip)
    era_gap  = max(0.0, xfip_val - era)

    # IP confidence ramp: 0.0 at 10 IP → 1.0 at 30+ IP.
    # More innings = more confidence the ERA/xFIP gap is structural, not noise.
    ip_conf = min(1.0, max(0.0, (ip - 10) / 20.0))

    # BABIP multiplier: each 0.050 below league avg (.300) adds 1.0×
    BABIP_LEAGUE_AVG = 0.300
    babip_mult = 1.0
    if isinstance(babip, float) and babip > 0:
        babip_mult = 1.0 + max(0.0, (BABIP_LEAGUE_AVG - babip) / 0.050)

    severity = era_gap * ip_conf * babip_mult

    # Elite-pitcher guard: xFIP < 3.20 → genuinely dominant, cap at MODERATE.
    if xfip_val < 3.20:
        severity = min(severity, 0.79)

    # K/9 guard: high-K pitchers legitimately suppress ERA below xFIP via fewer BIP.
    # Each K/9 above 9.0 reduces severity 25%, floored at 50%.
    if isinstance(k9, (int, float)) and k9 > 9.0:
        k9_factor = max(0.5, 1.0 - (k9 - 9.0) * 0.25)
        severity  = severity * k9_factor

    return round(severity, 3)


def _tbd_pitcher_cap(own_sp_score: float, own_win_pct: float,
                     own_sp_tbd: bool, opp_sp_tbd: bool) -> tuple:
    """
    Returns (should_cap: bool, reason: str) for TBD-pitcher confidence adjustment.

    Two anchors can sustain HIGH confidence when the opposing pitcher is unknown:
      1. Strong own pitcher  — quality score ≥ 0.40 (roughly xFIP ≤ 3.60)
      2. Strong own team     — season win rate ≥ .540

    Rationale: a known ace gives a concrete run-suppression edge regardless of
    who the opponent sends. A winning team indicates overall depth (offence +
    bullpen) that can weather an unknown starter. Either anchor alone is not
    enough — if the team is weak, even a good pitcher might not compensate for
    the uncertainty; if the pitcher is average, a .560 team still faces a
    meaningful unknown. Both together justify HIGH; anything less → MEDIUM.

    Own pitcher TBD → always cap. We can't anchor the edge without knowing our
    own main defensive weapon.
    """
    if own_sp_tbd:
        return True, "own starter TBD"
    if opp_sp_tbd:
        strong_pitcher = own_sp_score >= 0.40   # xFIP roughly ≤ 3.60
        strong_team    = own_win_pct  >= 0.540
        if strong_pitcher and strong_team:
            return False, ""   # both anchors present — HIGH still valid
        parts = []
        if not strong_pitcher:
            parts.append("own pitcher not strong enough to anchor edge")
        if not strong_team:
            parts.append(f"team win rate ({own_win_pct:.0%}) below .540 threshold")
        return True, "opposing starter TBD — " + "; ".join(parts)
    return False, ""


def _mlb_conf(edge: float, signal_count: int, stats_available: bool,
              own_trap_sev: float = 0.0, opp_trap_sev: float = 0.0,
              injury_capped: bool = False,
              tbd_capped: bool = False) -> str:
    """
    MLB-specific confidence label that accounts for ERA trap severity and
    TBD-pitcher uncertainty.

    own_trap_sev  — severity of the ERA trap on THIS team's starting pitcher.
                    High → cap at MEDIUM (market is not as wrong as raw edge suggests).
    opp_trap_sev  — severity of the ERA trap on the OPPONENT's starter.
                    High → lower the edge threshold required for HIGH confidence.
    tbd_capped    — True when TBD-pitcher uncertainty is too high to warrant HIGH.
                    See _tbd_pitcher_cap() for the two-anchor logic.
    """
    if injury_capped:
        return "MEDIUM"
    if tbd_capped:
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


def _hand_label(code: Optional[str]) -> str:
    """Map a pitcher hand code to a readable label for card display."""
    return {"L": "LHP", "R": "RHP"}.get(code or "", "opposing SP")


def _platoon_ops(overall_ops: float, split: Optional[Dict]) -> float:
    """
    Blend a lineup's overall OPS toward its split vs the opposing starter's hand,
    weighted by how much the split sample supports it. Full split weight at
    ~300 PA vs that hand; thin early-season samples lean on overall OPS. Falls
    back to overall when no split is available.
    """
    if not split or not split.get("ops"):
        return overall_ops
    pa = float(split.get("pa", 0) or 0)
    w  = min(1.0, pa / 300.0)
    return (1 - w) * overall_ops + w * float(split["ops"])


def analyze_mlb_game(game: Dict, home_pitcher_stats: Dict, away_pitcher_stats: Dict,
                     home_batting: Dict, away_batting: Dict,
                     home_bullpen: Dict, away_bullpen: Dict,
                     mlb_injuries: Dict,
                     home_schedule_load: int = 0,
                     away_schedule_load: int = 0,
                     umpire_tendency: Optional[Dict] = None,
                     weather: Optional[Dict] = None,
                     home_season_stats: Optional[Dict] = None,
                     away_season_stats: Optional[Dict] = None,
                     home_off_split: Optional[Dict] = None,
                     away_off_split: Optional[Dict] = None,
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
    playoff = _game_playoff(game, _is_mlb_playoff)
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
    def _pitcher_lines(name: str, stats: Dict, colour: str, team: str) -> None:
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
        # Throwing hand (R/L), same value the platoon split is computed against —
        # lets you corroborate the lineup's "vs RHP/LHP" split uses the right hand.
        hand_str = {"R": " · RHP", "L": " · LHP"}.get(stats.get("hand", ""), "")

        research.append(
            f"{colour} {name} ({team}){hand_str}: ERA {era} | {fip_str}{xfip_str} | "
            f"K/9 {k9} | BB/9 {bb9} | IP {ip_str}{babip_str}"
        )
        signals.append(f"{name} {fip_str}{xfip_str} | K/9: {k9}")

        # Short-leash warning: flag starters who consistently can't get through 3 innings
        _gs   = int(stats.get("games_started", 0) or 0)
        _avip = stats.get("avg_ip_per_start")
        if _gs >= 3 and _avip is not None and float(_avip) < 4.0:
            signals.append(
                f"⚠ Short leash: {name} averaging {_avip:.1f} IP/start over {_gs} starts "
                f"— bullpen carries {9 - float(_avip):.1f} innings"
            )
            research.append(
                f"⚠ {name} short-leash risk: {_avip:.1f} IP/start avg ({_gs} GS) — "
                f"effective opener, opponent's 2nd/3rd look already at bullpen"
            )

    if home_pitcher_stats:
        _pitcher_lines(home_pitcher_name, home_pitcher_stats, "🔵", home)
    else:
        research.append(f"🔵 {home_pitcher_name} ({home}): stats unavailable")

    if away_pitcher_stats:
        _pitcher_lines(away_pitcher_name, away_pitcher_stats, "🔴", away)
    else:
        research.append(f"🔴 {away_pitcher_name} ({away}): stats unavailable")

    # --- xFIP blend + pitcher quality scores ---
    # Must come before ERA trap severity (which reads xfip) and TBD cap (which
    # reads sp_score). Definitions AND their closed-over locals all live here.
    _LEAGUE_AVG_XFIP = 4.20
    _RECENT_WEIGHT   = 0.35   # weight given to last-4-starts xFIP vs season xFIP

    def _blend_xfip(stats: Dict) -> float:
        """Blends season xFIP with recent-form xFIP; regresses small samples toward league avg."""
        season_xfip = stats.get("xfip") or _LEAGUE_AVG_XFIP
        season_ip   = stats.get("innings_pitched", 0) or 0
        recent_xfip = stats.get("recent_xfip")
        recent_ip   = stats.get("recent_ip", 0) or 0
        if recent_xfip is not None and recent_ip >= 5:
            blended = season_xfip * (1 - _RECENT_WEIGHT) + recent_xfip * _RECENT_WEIGHT
        else:
            blended = float(season_xfip)
        if season_ip < 30:
            regression = max(0.0, (30 - season_ip) / 30)
            blended = blended * (1 - regression) + _LEAGUE_AVG_XFIP * regression
        return round(blended, 3)

    def _effective_avg_ip(stats: Dict) -> float:
        """Returns best available avg IP per start — recent form preferred over season.

        Fallback is 4.0 (not 5.0) when no data exists: a pitcher whose stats are
        absent from the API is more likely a rookie / spot starter than an established
        6-inning workhorse, so 4.0 puts more weight on the bullpen by default.

        Recent form requires ≥ 2 starts to override the season average. A single
        outing in the game log is too noisy — a pitcher whose game log only shows
        one qualifying start could have a 7-IP avg from an outlier start, masking
        a pattern of early exits (e.g. a rookie averaging 3.3 IP/start season-wide
        but the recent window only sees his one long outing).
        """
        recent        = stats.get("recent_avg_ip_per_start")
        recent_starts = int(stats.get("recent_starts", 0) or 0)
        season        = stats.get("avg_ip_per_start")
        valid_recent  = recent if (recent is not None and recent_starts >= 2) else None
        return float(valid_recent or season or 4.0)

    if home_pitcher_stats:
        home_blended_xfip = _blend_xfip(home_pitcher_stats)
        home_pitcher_stats = {**home_pitcher_stats, "xfip": home_blended_xfip}
    if away_pitcher_stats:
        away_blended_xfip = _blend_xfip(away_pitcher_stats)
        away_pitcher_stats = {**away_pitcher_stats, "xfip": away_blended_xfip}

    home_sp_score = _pitcher_quality_score(home_pitcher_stats) if home_pitcher_stats else 0
    away_sp_score = _pitcher_quality_score(away_pitcher_stats) if away_pitcher_stats else 0

    # --- ERA trap severity (continuous score) ---
    # Computed here so the same values feed BOTH signal generation and
    # the per-bet confidence logic below.
    home_trap_sev = _era_trap_severity(home_pitcher_stats) if home_pitcher_stats else 0.0
    away_trap_sev = _era_trap_severity(away_pitcher_stats) if away_pitcher_stats else 0.0

    # --- TBD pitcher confidence caps ---
    # Computed once here; passed to every _mlb_conf call below.
    # home_sp_tbd / away_sp_tbd: True when we have no stats for that starter.
    home_sp_tbd = not bool(home_pitcher_stats)
    away_sp_tbd = not bool(away_pitcher_stats)

    def _win_pct(season_stats: Optional[Dict]) -> float:
        if not season_stats:
            return 0.50  # unknown → assume average
        w = season_stats.get("wins", 0)
        l = season_stats.get("losses", 0)
        return w / max(1, w + l)

    home_win_pct = _win_pct(home_season_stats)
    away_win_pct = _win_pct(away_season_stats)

    # For a HOME pick: own = home side, opp = away side
    home_tbd_cap, home_tbd_reason = _tbd_pitcher_cap(
        own_sp_score=home_sp_score, own_win_pct=home_win_pct,
        own_sp_tbd=home_sp_tbd,     opp_sp_tbd=away_sp_tbd,
    )
    # For an AWAY pick: own = away side, opp = home side
    away_tbd_cap, away_tbd_reason = _tbd_pitcher_cap(
        own_sp_score=away_sp_score, own_win_pct=away_win_pct,
        own_sp_tbd=away_sp_tbd,     opp_sp_tbd=home_sp_tbd,
    )

    # Emit a research note when a TBD cap fires so it's visible on the card
    if home_tbd_cap and home_tbd_reason:
        research.append(f"⚠ {home} confidence capped at MEDIUM: {home_tbd_reason}")
    if away_tbd_cap and away_tbd_reason:
        research.append(f"⚠ {away} confidence capped at MEDIUM: {away_tbd_reason}")

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

    # --- Batting research (W-L record prepended if standings available) ---
    _h_ss = home_season_stats or {}
    _a_ss = away_season_stats or {}
    _h_w, _h_l = _h_ss.get("wins", 0), _h_ss.get("losses", 0)
    _a_w, _a_l = _a_ss.get("wins", 0), _a_ss.get("losses", 0)
    _h_rec = f"{_h_w}W-{_h_l}L | " if _h_w + _h_l > 0 else ""
    _a_rec = f"{_a_w}W-{_a_l}L | " if _a_w + _a_l > 0 else ""
    if home_batting:
        research.append(
            f"{home} offense: {_h_rec}OPS {home_batting.get('ops', '?'):.3f} | "
            f"AVG {home_batting.get('avg', '?'):.3f} | "
            f"R/G {home_batting.get('runs_per_game', '?'):.2f}"
        )
    elif _h_w + _h_l > 0:
        research.append(f"{home}: {_h_rec.rstrip(' | ')}")
    if away_batting:
        research.append(
            f"{away} offense: {_a_rec}OPS {away_batting.get('ops', '?'):.3f} | "
            f"AVG {away_batting.get('avg', '?'):.3f} | "
            f"R/G {away_batting.get('runs_per_game', '?'):.2f}"
        )
    elif _a_w + _a_l > 0:
        research.append(f"{away}: {_a_rec.rstrip(' | ')}")

    # --- Bullpen research ---
    home_bp_era  = home_bullpen.get("bullpen_era",  4.20)
    away_bp_era  = away_bullpen.get("bullpen_era",  4.20)
    home_bp_whip = home_bullpen.get("bullpen_whip", 1.30)
    away_bp_whip = away_bullpen.get("bullpen_whip", 1.30)
    research.append(
        f"Bullpen ERA — {home}: {home_bp_era:.2f} (WHIP {home_bp_whip:.2f}) | "
        f"{away}: {away_bp_era:.2f} (WHIP {away_bp_whip:.2f})"
    )

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
    home_ops = home_batting.get("ops", 0.735)
    away_ops = away_batting.get("ops", 0.735)

    # Platoon split adjustment REMOVED (Jun 2026). Added May 30 (Tier 3 8a), it
    # blended each lineup's OPS toward its raw vs-hand split at full weight by
    # 300 PA — far too aggressive for a stat that needs ~1000+ PA to stabilise
    # and should be regressed ~50%. The unregressed split manufactured edge on
    # the side the market had already priced: platoon-fired MLB moneylines won
    # 45.7% (vs 58.1% for picks without it) over 51 settled picks. Season OPS is
    # left untouched. _platoon_ops() is retained (unused) in case a properly
    # regressed version is reintroduced later and validated against CLV first.

    # Coors Field correction — Colorado's season OPS and pitcher xFIP/ERA are
    # heavily inflated by home games at altitude. When the Rockies play AWAY,
    # their true offensive output is ~8% lower than season OPS suggests.
    # When opponents pitch AT Coors, their stats are similarly inflated.
    # The park_factor (1.30) already adjusts run totals for the venue, but
    # the Rockies' season batting stats need correcting before that step.
    # Road/season OPS ratio from ESPN splits 2022-2025: 0.872, 0.910, 0.908, 0.870 → avg 0.89
    _COORS_OPS_DEFLATOR = 0.87    # Rockies road OPS ≈ 87% of season OPS (empirical 4-yr avg ~0.89, conservative)
    # Visiting teams hit ~10% better at altitude than their neutral-park season OPS suggests.
    # Derived from Coors OPS park factor (~1.15); conservative estimate pending decision-log validation.
    _COORS_VISITOR_INFLATOR = 1.10
    _is_coors = "Coors" in venue
    if "Colorado Rockies" == away and not _is_coors:
        # Rockies batting away from Coors — deflate their inflated season OPS
        away_ops = away_ops * _COORS_OPS_DEFLATOR
    if "Colorado Rockies" == home and not _is_coors:
        # Shouldn't happen (Rockies always home at Coors) but guard anyway
        home_ops = home_ops * _COORS_OPS_DEFLATOR
    if _is_coors and away != "Colorado Rockies":
        # Visiting team batting at Coors — inflate their neutral-park season OPS
        away_ops = away_ops * _COORS_VISITOR_INFLATOR

    # Lowered 4.5 → 4.35 (Jun 2026): 4.5 R/G per team (9.0 baseline) sat ~0.3 runs
    # above current MLB scoring (~4.35 R/G / ~8.7 total), giving every total a
    # constant upward (Over) thumb. MLB Over picks realised 42.9% vs the 65.7%
    # the model claimed — the asymmetry vs Unders (only 7.7pp overconfident) is
    # the signature of this baseline bias. 4.35 removes the lean at its source.
    league_avg_runs = 4.35

    # Bullpen quality score uses same scale as SP: (4.20 - ERA) / 1.50
    # Positive = better than average (suppresses opponent runs), negative = worse.
    home_bullpen_score = (4.20 - home_bp_era) / 1.50
    away_bullpen_score = (4.20 - away_bp_era) / 1.50

    home_offense_adj = (home_ops - 0.735) / 0.080   # baseline updated to current MLB avg OPS
    away_offense_adj = (away_ops - 0.735) / 0.080

    # Innings-based SP / bullpen split.
    # Each pitcher's expected innings determine how much of the pitching coefficient
    # they own. A SP going 6+ IP makes the bullpen nearly irrelevant; an opener
    # scenario (3 IP) makes the bullpen the dominant factor.
    # avg_ip_per_start is fetched from season stats; cap to [2, 7] to avoid extremes.
    _TOTAL_INN   = 9.0
    _PITCH_COEFF = 0.80   # total pitching influence on expected runs (unchanged)

    home_sp_ip = min(7.0, max(2.0, _effective_avg_ip(home_pitcher_stats))) \
                 if home_pitcher_stats else 5.0
    away_sp_ip = min(7.0, max(2.0, _effective_avg_ip(away_pitcher_stats))) \
                 if away_pitcher_stats else 5.0

    home_sp_coeff = _PITCH_COEFF * (home_sp_ip / _TOTAL_INN)
    home_bp_coeff = _PITCH_COEFF * ((_TOTAL_INN - home_sp_ip) / _TOTAL_INN)
    away_sp_coeff = _PITCH_COEFF * (away_sp_ip / _TOTAL_INN)
    away_bp_coeff = _PITCH_COEFF * ((_TOTAL_INN - away_sp_ip) / _TOTAL_INN)

    # A better pitcher (positive score) REDUCES the opponent's expected runs → subtract.
    # A worse pitcher (negative score) INCREASES the opponent's expected runs → subtract negative = add.
    # Offense weight 0.6 → 0.4 (Jul 4 2026 optimization): season OPS deviations
    # were over-projecting run differences the market had already discounted.
    _OFF_W = 0.4
    expected_home_runs = max(1.5, league_avg_runs
                             - away_sp_score     * away_sp_coeff   # away SP quality
                             - away_bullpen_score * away_bp_coeff  # away bullpen quality
                             + home_offense_adj  * _OFF_W
                             + MLB_HOME_ADVANTAGE * 5)
    expected_away_runs = max(1.5, league_avg_runs
                             - home_sp_score     * home_sp_coeff   # home SP quality
                             - home_bullpen_score * home_bp_coeff  # home bullpen quality
                             + away_offense_adj  * _OFF_W)

    # --- Strikeout matchup adjustment ---
    # High-K% offense facing a strikeout pitcher is a run-suppression signal the
    # market sometimes underweights. Both conditions must be extreme to adjust.
    _K_PCT_THRESHOLD = 0.245   # ~17th percentile worst batting K% (league avg ~0.228)
    _K9_THRESHOLD    = 9.5     # above-average strikeout rate for starters
    _K_MATCHUP_COEFF = 0.20    # max run reduction per team (~0.2 runs at full mismatch)

    home_k_pct  = home_batting.get("k_pct", 0.228)
    away_k_pct  = away_batting.get("k_pct", 0.228)
    away_sp_k9  = away_pitcher_stats.get("k_per_9", 8.0) if away_pitcher_stats else 8.0
    home_sp_k9  = home_pitcher_stats.get("k_per_9", 8.0) if home_pitcher_stats else 8.0

    home_k_penalty = max(0.0, home_k_pct - _K_PCT_THRESHOLD) * max(0.0, away_sp_k9 - _K9_THRESHOLD) * _K_MATCHUP_COEFF
    away_k_penalty = max(0.0, away_k_pct - _K_PCT_THRESHOLD) * max(0.0, home_sp_k9 - _K9_THRESHOLD) * _K_MATCHUP_COEFF

    if home_k_penalty > 0.01:
        expected_home_runs = max(1.5, expected_home_runs - home_k_penalty)
        signals.append(
            f"⚠ K matchup: {home} K% {home_k_pct:.1%} vs {away_pitcher_name} K/9 {away_sp_k9:.1f} "
            f"— run suppression ({home_k_penalty:.2f} runs)"
        )
    if away_k_penalty > 0.01:
        expected_away_runs = max(1.5, expected_away_runs - away_k_penalty)
        signals.append(
            f"⚠ K matchup: {away} K% {away_k_pct:.1%} vs {home_pitcher_name} K/9 {home_sp_k9:.1f} "
            f"— run suppression ({away_k_penalty:.2f} runs)"
        )

    # --- Recent offensive form nudge ---
    # Blends each team's last-10-game runs-scored rate with their season R/G.
    # Additive delta so the existing formula structure is preserved: a team
    # scoring 30% more recently than their season avg nudges expected runs up
    # by (recent_rpg - season_rpg) * MLB_RECENT_FORM_WEIGHT.
    # Only emits a signal when the adjustment reaches ±0.10 runs.
    home_season_rpg = home_batting.get("runs_per_game", league_avg_runs)
    away_season_rpg = away_batting.get("runs_per_game", league_avg_runs)
    home_recent_rpg = home_batting.get("recent_rpg")
    away_recent_rpg = away_batting.get("recent_rpg")

    if home_recent_rpg is not None and home_season_rpg > 0:
        home_form_delta = (home_recent_rpg - home_season_rpg) * MLB_RECENT_FORM_WEIGHT
        expected_home_runs = max(1.5, expected_home_runs + home_form_delta)
        if abs(home_form_delta) >= 0.10:
            hw = home_batting.get("recent_wins", "?")
            hg = home_batting.get("recent_games", 10)
            direction = "↑" if home_form_delta > 0 else "↓"
            signals.append(
                f"Recent form: {home} {hw}-{hg - (hw if isinstance(hw, int) else 0)} last {hg} "
                f"| {home_recent_rpg:.1f} R/G vs {home_season_rpg:.1f} season "
                f"({direction}{abs(home_form_delta):.2f} run adj)"
            )

    if away_recent_rpg is not None and away_season_rpg > 0:
        away_form_delta = (away_recent_rpg - away_season_rpg) * MLB_RECENT_FORM_WEIGHT
        expected_away_runs = max(1.5, expected_away_runs + away_form_delta)
        if abs(away_form_delta) >= 0.10:
            aw = away_batting.get("recent_wins", "?")
            ag = away_batting.get("recent_games", 10)
            direction = "↑" if away_form_delta > 0 else "↓"
            signals.append(
                f"Recent form: {away} {aw}-{ag - (aw if isinstance(aw, int) else 0)} last {ag} "
                f"| {away_recent_rpg:.1f} R/G vs {away_season_rpg:.1f} season "
                f"({direction}{abs(away_form_delta):.2f} run adj)"
            )

    # Park factor is intentionally NOT applied to individual run totals here.
    # Multiplying both teams equally inflates run_diff by park_factor, producing
    # a spurious win-probability boost at hitter-friendly parks (e.g. Coors +~3pp).
    # Park factor is applied to expected_total in the Totals section below,
    # where it correctly widens Over/Under projections without touching ML/Spread.

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

    # --- Schedule load ---
    # Applied BEFORE run_diff so ML/Spread model sees the fatigue penalty,
    # not just the totals. Previously these modified expected_runs after
    # model_home_prob was already computed, making them totals-only signals.
    home_load_pen = _schedule_load_penalty(home_schedule_load)
    away_load_pen = _schedule_load_penalty(away_schedule_load)
    if home_load_pen > 0:
        expected_home_runs *= (1 - home_load_pen)
        signals.append(f"⚠ {home} schedule load: {home_schedule_load} games in 7 days")
    if away_load_pen > 0:
        expected_away_runs *= (1 - away_load_pen)
        signals.append(f"⚠ {away} schedule load: {away_schedule_load} games in 7 days")

    # --- Playoff context ---
    # Also applied BEFORE run_diff for the same reason — playoff run reduction
    # should flow into win probability, not just totals.
    if playoff:
        expected_home_runs *= MLB_PLAYOFF_SCORING_FACTOR
        expected_away_runs *= MLB_PLAYOFF_SCORING_FACTOR
        signals.append("🏆 Playoffs: ace starters, shorter leash, lower scoring applied")
        research.append(f"Playoff adjustment: runs scaled by {MLB_PLAYOFF_SCORING_FACTOR} | starter IP capped at {MLB_PLAYOFF_STARTER_IP}")

    # --- Injury run adjustment: DISABLED (Jul 4 2026 optimization) ---
    # _INJ_RUNS_PER_PCT 0.08 → 0.0. Injury-driven picks won 38% over 126 samples
    # with FLAT CLV — the market already prices injuries into the line, so
    # subtracting runs for them double-counted the information and manufactured
    # phantom edges on the healthy side. The injury CREDIBILITY CAP below is
    # intentionally kept: it only anchors the model toward market (safety),
    # never creates edge. Injury data still flows into signals/logs unchanged.
    _INJ_RUNS_PER_PCT = 0.0
    if home_inj > 0.01:
        home_inj_runs = home_inj * _INJ_RUNS_PER_PCT * 100   # inj is a fraction, e.g. 0.025
        expected_home_runs = max(1.5, expected_home_runs - home_inj_runs)
    if away_inj > 0.01:
        away_inj_runs = away_inj * _INJ_RUNS_PER_PCT * 100
        expected_away_runs = max(1.5, expected_away_runs - away_inj_runs)

    signals.append(f"Model projected score: {home} {expected_home_runs:.1f} — {away} {expected_away_runs:.1f}")
    research.append(f"Model expected runs: {home} {expected_home_runs:.2f} | {away} {expected_away_runs:.2f}")

    run_diff = expected_home_runs - expected_away_runs
    # Apply diminishing-returns cap: small edges pass through unchanged,
    # large stacked differentials are compressed toward MLB_RUN_DIFF_CAP.
    run_diff_capped = MLB_RUN_DIFF_CAP * math.tanh(run_diff / MLB_RUN_DIFF_CAP)
    if abs(run_diff_capped) < abs(run_diff) - 0.05:
        research.append(
            f"tanh cap active: raw run diff {run_diff:+.2f} → capped {run_diff_capped:+.2f}"
        )
    # Calibration capture: raw prob (post-tanh, pre-hardcap) + cap trackers.
    # Stamped onto each rec at the end (see _stamp_recs_calibration call).
    _mlb_raw_home = float(norm.cdf(run_diff_capped, 0, MLB_SPREAD_STD))
    _mlb_hardcap_fired = (_mlb_raw_home > 0.85) or (_mlb_raw_home < 0.15)
    model_home_prob = min(0.85, max(0.15, _mlb_raw_home))
    model_away_prob = 1 - model_home_prob

    ml = game.get("moneyline")
    if ml:
        market_home_prob = ml["home_prob"]
        market_away_prob = ml["away_prob"]

        # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
        model_home_prob, _mlb_cred_fired = _apply_credibility_cap_dispatched(
            model_home_prob, market_home_prob, _cred_cap("mlb", MLB_CRED_CAP, "credibility_moneyline"), "mlb", "credibility_moneyline"
        )
        model_away_prob = 1 - model_home_prob

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
                                 injury_capped=home_injury_capped,
                                 tbd_capped=home_tbd_cap)
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
                                 injury_capped=away_injury_capped,
                                 tbd_capped=away_tbd_cap)
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
            # NOTE: ML→effective_margin uses MLB_SPREAD_STD (pitcher-quality calibration),
            # but the run line cover probability uses MLB_RUNLINE_SIGMA (actual margin
            # variance ~3.5 runs). Using the same 1.8σ for both was amplifying ML→RL
            # jumps to ~+25–32% when sportsbook empirical jumps are ~+12–18%.
            effective_margin = float(norm.ppf(model_home_prob)) * MLB_SPREAD_STD

            # P(home covers run line) = P(actual margin > −home_spread_line)
            model_home_cover = float(norm.cdf(effective_margin + home_spread_line, 0, MLB_RUNLINE_SIGMA))
            model_away_cover = 1.0 - model_home_cover

            # Apply spread credibility cap
            _mlb_sp_raw_home = model_home_cover
            _mlb_sp_raw_away = model_away_cover
            model_home_cover, _mlb_sp_cred_home = _apply_credibility_cap_dispatched(
                model_home_cover, market_home_cover, _cred_cap("mlb", MLB_CRED_CAP, "credibility_spread"), "mlb", "credibility_spread"
            )
            model_away_cover, _mlb_sp_cred_away = _apply_credibility_cap_dispatched(
                model_away_cover, market_away_cover, _cred_cap("mlb", MLB_CRED_CAP, "credibility_spread"), "mlb", "credibility_spread"
            )

            away_spread_line = -home_spread_line

            home_rl_edge = model_home_cover - market_home_cover
            away_rl_edge = model_away_cover - market_away_cover

            if home_rl_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _mlb_conf(home_rl_edge, len(signals), stats_available,
                                     own_trap_sev=home_trap_sev, opp_trap_sev=away_trap_sev,
                                     injury_capped=home_injury_capped,
                                     tbd_capped=home_tbd_cap)
                    _mlb_sp_rec = BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_rl_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _mlb_sp_rec.model_prob_raw = _mlb_sp_raw_home
                    _mlb_sp_rec.credibility_cap_fired = _mlb_sp_cred_home
                    recs.append(_mlb_sp_rec)

            if away_rl_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _mlb_conf(away_rl_edge, len(signals), stats_available,
                                     own_trap_sev=away_trap_sev, opp_trap_sev=home_trap_sev,
                                     injury_capped=away_injury_capped,
                                     tbd_capped=away_tbd_cap)
                    _mlb_sp_rec = BetRecommendation(
                        sport="MLB", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_rl_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _mlb_sp_rec.model_prob_raw = _mlb_sp_raw_away
                    _mlb_sp_rec.credibility_cap_fired = _mlb_sp_cred_away
                    recs.append(_mlb_sp_rec)

    # --- Total ---
    total = game.get("total")
    if total:
        # Apply park factor here (totals only) — both teams score more at hitter-friendly
        # parks, so the combined total is correctly inflated. Park factor is NOT applied
        # to run_diff (see ML/Spread section above) to avoid inflating win probability.
        expected_total = (expected_home_runs + expected_away_runs) * park_factor
        market_line = total["line"]
        blended_total = (1 - _TOTAL_MARKET_ANCHOR) * expected_total + _TOTAL_MARKET_ANCHOR * market_line
        # Stamp onto game dict so props_analyzer can use the same projection
        game["model_total"]  = round(blended_total, 1)
        game["market_total"] = market_line
        model_over_prob = float(1 - norm.cdf(market_line, blended_total, MLB_TOTAL_STD))
        market_over_prob = total["over_prob"]
        market_under_prob = total["under_prob"]

        _mlb_total_raw_over = model_over_prob
        model_over_prob, _mlb_total_cred_fired = _apply_credibility_cap_dispatched(
            model_over_prob, market_over_prob, _cred_cap("mlb", MLB_CRED_CAP, "credibility_total"), "mlb", "credibility_total"
        )

        total_signals = signals[:] + [f"Model expected total: {blended_total:.1f} vs line {market_line}"]
        total_research = research[:]

        over_edge = model_over_prob - market_over_prob
        under_edge = (1 - model_over_prob) - market_under_prob

        # Use .5 lines to eliminate push risk: "Over 10" → "Over 9.5", "Under 8" → "Under 8.5"
        _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
        _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

        if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
            sizing = robinhood_kelly(model_over_prob, market_over_prob)
            if sizing.num_contracts > 0:
                _over_rec = BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=_over_label,
                    market_prob=market_over_prob, model_prob=model_over_prob,
                    edge=over_edge, contract_price=market_over_prob,
                    sizing=sizing,
                    confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                )
                _over_rec.model_prob_raw = _mlb_total_raw_over
                _over_rec.credibility_cap_fired = _mlb_total_cred_fired
                recs.append(_over_rec)
        if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
            sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
            if sizing.num_contracts > 0:
                _under_rec = BetRecommendation(
                    sport="MLB", game=label, bet_type="Total", pick=_under_label,
                    market_prob=market_under_prob, model_prob=1 - model_over_prob,
                    edge=under_edge, contract_price=market_under_prob,
                    sizing=sizing,
                    confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                    signals=total_signals, research=total_research,
                    home_team=home, away_team=away, game_time=game_time,
                    commence_time=commence_time,
                )
                _under_rec.model_prob_raw = 1.0 - _mlb_total_raw_over
                _under_rec.credibility_cap_fired = _mlb_total_cred_fired
                recs.append(_under_rec)

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _home_raw = _lv.get("_mlb_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home, away_team=away,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_mlb_cred_fired", False),
        hardcap_fired=_lv.get("_mlb_hardcap_fired", False),
        home_injury_fired=_lv.get("home_injury_capped", False),
        away_injury_fired=_lv.get("away_injury_capped", False),
        stats_avail=stats_available,
    )
    # Stamp game_pk so the template can do live pitcher/batter lookups.
    _gpk = str(game.get("game_pk") or "")
    for _r in recs:
        _r.game_pk = _gpk

    # ── Decision-log capture (diagnostic; never affects selection) ───────────
    # Log BOTH sides of every market the model evaluated + the structured inputs
    # behind the projection, so CLV/outcome can later be measured for the picks
    # the model REJECTED too. Reads from locals() so block-local vars (spread /
    # total only exist when those markets are offered) resolve safely to None.
    try:
        _hp = home_pitcher_stats or {}
        _ap = away_pitcher_stats or {}
        _feat = {
            "expected_home_runs": _lv.get("expected_home_runs"),
            "expected_away_runs": _lv.get("expected_away_runs"),
            "run_diff": _lv.get("run_diff"),
            "run_diff_capped": _lv.get("run_diff_capped"),
            "park_factor": _lv.get("park_factor"),
            "home_ops": _lv.get("home_ops"),
            "away_ops": _lv.get("away_ops"),
            "league_avg_runs": _lv.get("league_avg_runs"),
            "home_inj": _lv.get("home_inj"),
            "away_inj": _lv.get("away_inj"),
            "home_bp_era": _lv.get("home_bp_era"),
            "away_bp_era": _lv.get("away_bp_era"),
            "home_sp_xfip": _hp.get("xfip"),
            "away_sp_xfip": _ap.get("xfip"),
            "home_sp_fip": _hp.get("fip"),
            "away_sp_fip": _ap.get("fip"),
            "home_sp_era": _hp.get("era"),
            "away_sp_era": _ap.get("era"),
            "home_sp_bb9": _hp.get("bb_per_9"),
            "away_sp_bb9": _ap.get("bb_per_9"),
            "home_sp_k9": _hp.get("k_per_9"),
            "away_sp_k9": _ap.get("k_per_9"),
            # Pitcher quality + weighting (directly drives how much the starter
            # moves the projection — the pitcher-weight question).
            "home_sp_score": _lv.get("home_sp_score"),
            "away_sp_score": _lv.get("away_sp_score"),
            "home_sp_ip": _lv.get("home_sp_ip"),
            "away_sp_ip": _lv.get("away_sp_ip"),
            "home_sp_coeff": _lv.get("home_sp_coeff"),
            "away_sp_coeff": _lv.get("away_sp_coeff"),
            "home_trap_sev": _lv.get("home_trap_sev"),
            "away_trap_sev": _lv.get("away_trap_sev"),
            "home_bullpen_score": _lv.get("home_bullpen_score"),
            "away_bullpen_score": _lv.get("away_bullpen_score"),
            "home_bp_whip": _lv.get("home_bp_whip"),
            "away_bp_whip": _lv.get("away_bp_whip"),
            "home_offense_adj": _lv.get("home_offense_adj"),
            "away_offense_adj": _lv.get("away_offense_adj"),
            # Team strength + every run adjustment in the projection
            "home_win_pct": _lv.get("home_win_pct"),
            "away_win_pct": _lv.get("away_win_pct"),
            "home_season_rpg": _lv.get("home_season_rpg"),
            "away_season_rpg": _lv.get("away_season_rpg"),
            "home_recent_rpg": _lv.get("home_recent_rpg"),
            "away_recent_rpg": _lv.get("away_recent_rpg"),
            "home_form_delta": _lv.get("home_form_delta"),
            "away_form_delta": _lv.get("away_form_delta"),
            "home_k_penalty": _lv.get("home_k_penalty"),
            "away_k_penalty": _lv.get("away_k_penalty"),
            "home_schedule_load": _lv.get("home_schedule_load"),
            "away_schedule_load": _lv.get("away_schedule_load"),
            "home_load_pen": _lv.get("home_load_pen"),
            "away_load_pen": _lv.get("away_load_pen"),
            "ump_run_factor": _lv.get("ump_run_factor"),
            "weather_run_adj": _lv.get("wx_adj"),
            "playoff": _lv.get("playoff"),
            "stats_available": stats_available,
            "is_coors": _lv.get("_is_coors"),
            "coors_ops_deflator_applied": _lv.get("_COORS_OPS_DEFLATOR") if (_lv.get("_is_coors") is False and ("Colorado Rockies" in [home, away])) else None,
            "coors_visitor_inflator_applied": _lv.get("_COORS_VISITOR_INFLATOR") if _lv.get("_is_coors") else None,
        }
        _mlb_rh = _lv.get("_mlb_raw_home")
        _mlb_ro = _lv.get("_mlb_total_raw_over")
        _markets = [
            ("Moneyline", home, _lv.get("model_home_prob"), _mlb_rh, _lv.get("market_home_prob"), None),
            ("Moneyline", away, _lv.get("model_away_prob"),
             (1.0 - _mlb_rh) if _mlb_rh is not None else None, _lv.get("market_away_prob"), None),
            ("Spread", home, _lv.get("model_home_cover"), _lv.get("_mlb_sp_raw_home"),
             _lv.get("market_home_cover"), _lv.get("home_spread_line")),
            ("Spread", away, _lv.get("model_away_cover"), _lv.get("_mlb_sp_raw_away"),
             _lv.get("market_away_cover"),
             (-_lv["home_spread_line"]) if _lv.get("home_spread_line") is not None else None),
            ("Total", "over", _lv.get("model_over_prob"), _mlb_ro, _lv.get("market_over_prob"), _lv.get("market_line")),
            ("Total", "under",
             (1.0 - _lv["model_over_prob"]) if _lv.get("model_over_prob") is not None else None,
             (1.0 - _mlb_ro) if _mlb_ro is not None else None,
             _lv.get("market_under_prob"), _lv.get("market_line")),
        ]
        _stamp_decision(game, _min, _feat, _markets, recs)
    except Exception:
        pass

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

    playoff = _game_playoff(game, _is_nfl_playoff)
    stats_available = bool(nfl_ctx["season_stats"].get(home) or nfl_ctx["season_stats"].get(away))

    home_rest = nfl_ctx["rest_days"].get(home, 7)
    away_rest = nfl_ctx["rest_days"].get(away, 7)

    # ── Pre-compute projected score shared by all bet types ───────────────────
    # Mirrors the Total block formula (season ppg/oppg average) so ML, Spread,
    # and Total cards always show the same number for a given game.
    _nfl_proj_signal: Optional[str] = None
    _nfl_hs_pre = nfl_ctx["season_stats"].get(home, {})
    _nfl_as_pre = nfl_ctx["season_stats"].get(away, {})
    if _nfl_hs_pre and _nfl_as_pre:
        _nfl_ph = (_nfl_hs_pre.get("ppg", 21.0) + _nfl_as_pre.get("oppg", 21.0)) / 2
        _nfl_pa = (_nfl_as_pre.get("ppg", 21.0) + _nfl_hs_pre.get("oppg", 21.0)) / 2
        _nfl_proj_signal = f"Model projected score: {home} {_nfl_ph:.0f} — {away} {_nfl_pa:.0f}"

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

        # Home field advantage — zeroed for the Super Bowl (neutral site; the
        # designated "home" team has no real venue edge there).
        if game.get("season_game_type") == "superbowl":
            signals.append("Neutral site (Super Bowl) — no home field advantage applied")
        else:
            adj += NFL_HOME_ADV
            signals.append(f"Home field advantage: {home} (+{NFL_HOME_ADV*100:.0f}%)")

        # Team strength research
        if home_stats:
            _h_w, _h_l = home_stats.get('wins', 0), home_stats.get('losses', 0)
            _h_rec = f"{_h_w}W-{_h_l}L | " if _h_w + _h_l > 0 else ""
            research.append(
                f"{home}: {_h_rec}{home_stats.get('ppg', '?'):.1f} PPG | "
                f"{home_stats.get('oppg', '?'):.1f} OPP PPG | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.1f}"
            )
        if away_stats:
            _a_w, _a_l = away_stats.get('wins', 0), away_stats.get('losses', 0)
            _a_rec = f"{_a_w}W-{_a_l}L | " if _a_w + _a_l > 0 else ""
            research.append(
                f"{away}: {_a_rec}{away_stats.get('ppg', '?'):.1f} PPG | "
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

        # Projected score (pre-computed above; same formula as Total block)
        if _nfl_proj_signal:
            signals.append(_nfl_proj_signal)

        # Calibration capture: raw prob + cap firing trackers.
        _nfl_raw_home = _nfl_margin_to_prob(base_margin) + adj
        _nfl_hardcap_fired = (_nfl_raw_home > 0.90) or (_nfl_raw_home < 0.10)
        adjusted_home_prob = min(0.90, max(0.10, _nfl_raw_home))
        adjusted_away_prob = 1 - adjusted_home_prob

        # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
        adjusted_home_prob, _nfl_cred_fired = _apply_credibility_cap_dispatched(
            adjusted_home_prob, market_home_prob, _cred_cap("nfl", NFL_CRED_CAP, "credibility_moneyline"), "nfl", "credibility_moneyline"
        )
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

            # Apply spread credibility cap
            _nfl_sp_raw_home = model_home_cover
            _nfl_sp_raw_away = model_away_cover
            model_home_cover, _nfl_sp_cred_home = _apply_credibility_cap_dispatched(
                model_home_cover, market_home_cover, _cred_cap("nfl", NFL_CRED_CAP, "credibility_spread"), "nfl", "credibility_spread"
            )
            model_away_cover, _nfl_sp_cred_away = _apply_credibility_cap_dispatched(
                model_away_cover, market_away_cover, _cred_cap("nfl", NFL_CRED_CAP, "credibility_spread"), "nfl", "credibility_spread"
            )

            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    _nfl_sp_rec = BetRecommendation(
                        sport="NFL", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _nfl_sp_rec.model_prob_raw = _nfl_sp_raw_home
                    _nfl_sp_rec.credibility_cap_fired = _nfl_sp_cred_home
                    recs.append(_nfl_sp_rec)

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    _nfl_sp_rec = BetRecommendation(
                        sport="NFL", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _nfl_sp_rec.model_prob_raw = _nfl_sp_raw_away
                    _nfl_sp_rec.credibility_cap_fired = _nfl_sp_cred_away
                    recs.append(_nfl_sp_rec)

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

            _nfl_total_raw_over = model_over_prob
            model_over_prob, _nfl_total_cred_fired = _apply_credibility_cap_dispatched(
                model_over_prob, market_over_prob, _cred_cap("nfl", NFL_CRED_CAP, "credibility_total"), "nfl", "credibility_total"
            )

            total_signals = [
                _nfl_proj_signal or f"Model projected score: {home} {exp_home:.0f} — {away} {exp_away:.0f}",
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
                    _over_rec = BetRecommendation(
                        sport="NFL", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _over_rec.model_prob_raw = _nfl_total_raw_over
                    _over_rec.credibility_cap_fired = _nfl_total_cred_fired
                    recs.append(_over_rec)
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    _under_rec = BetRecommendation(
                        sport="NFL", game=label, bet_type="Total", pick=_under_label,
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _under_rec.model_prob_raw = 1.0 - _nfl_total_raw_over
                    _under_rec.credibility_cap_fired = _nfl_total_cred_fired
                    recs.append(_under_rec)

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _home_raw = _lv.get("_nfl_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home, away_team=away,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_nfl_cred_fired", False),
        hardcap_fired=_lv.get("_nfl_hardcap_fired", False),
        home_injury_fired=_lv.get("home_injury_capped", False),
        away_injury_fired=_lv.get("away_injury_capped", False),
        stats_avail=stats_available,
    )
    _nfl_rh = _lv.get("_nfl_raw_home"); _nfl_ro = _lv.get("_nfl_total_raw_over")
    _stamp_decision(game, _min, {
        "home_strength": _lv.get("home_strength"), "away_strength": _lv.get("away_strength"),
        "base_margin": _lv.get("base_margin"), "effective_margin": _lv.get("effective_margin"),
        "win_prob_adj": _lv.get("adj"), "rest_diff": _lv.get("rest_diff"),
        "expected_total": _lv.get("blended_total"), "total_std": _lv.get("total_std"),
        "home_rest": _lv.get("home_rest"), "away_rest": _lv.get("away_rest"),
        "home_inj": _lv.get("home_inj"), "away_inj": _lv.get("away_inj"),
        "playoff": _lv.get("playoff"), "stats_available": stats_available,
    }, [
        ("Moneyline", home, _lv.get("adjusted_home_prob"), _nfl_rh, _lv.get("market_home_prob"), None),
        ("Moneyline", away, _lv.get("adjusted_away_prob"),
         (1.0 - _nfl_rh) if _nfl_rh is not None else None, _lv.get("market_away_prob"), None),
        ("Spread", home, _lv.get("model_home_cover"), _lv.get("_nfl_sp_raw_home"),
         _lv.get("market_home_cover"), _lv.get("home_spread_line")),
        ("Spread", away, _lv.get("model_away_cover"), _lv.get("_nfl_sp_raw_away"),
         _lv.get("market_away_cover"),
         (-_lv["home_spread_line"]) if _lv.get("home_spread_line") is not None else None),
        ("Total", "over", _lv.get("model_over_prob"), _nfl_ro, _lv.get("market_over_prob"), _lv.get("market_line")),
        ("Total", "under",
         (1.0 - _lv["model_over_prob"]) if _lv.get("model_over_prob") is not None else None,
         (1.0 - _nfl_ro) if _nfl_ro is not None else None,
         _lv.get("market_under_prob"), _lv.get("market_line")),
    ], recs)
    return recs


# ---------------------------------------------------------------------------
# NHL Edge Finder
# ---------------------------------------------------------------------------

# Empirically measured from 289 NHL games (2024-25): final-margin SD ≈ 2.53,
# total-goals SD ≈ 2.19. The prior values (1.7 / 1.5) were far too tight and
# drove severe win-prob overconfidence (model ~70% vs realised ~55%). NHL is a
# high-variance sport (OT/SO coin-flips, hot goalies), so win probability should
# be flat in team strength. Set ~5% below the measured SDs (small allowance for
# the portion of variance the net-rating prediction explains).
NHL_SPREAD_STD   = 2.4   # was 1.7; measured margin SD ≈ 2.53. Also used for puck-line cover.
NHL_TOTAL_STD    = 2.1   # was 1.5; measured total SD ≈ 2.19
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


# Probability→margin conversion for adding the NHL residual adjustments (home
# ice, B2B, injuries) in MARGIN space, then converting with a single CDF — so a
# flat ±X% no longer overshoots in the tails. Dividing by the central CDF slope
# (pdf(0)) keeps mid-range picks unchanged and compresses only the tails.
_NHL_PROB_TO_MARGIN = 1.0 / float(norm.pdf(0, 0, NHL_SPREAD_STD))


def analyze_nhl_game(game: Dict, nhl_ctx: Dict, nhl_injuries: Dict, min_edge: float = None) -> List[BetRecommendation]:
    from src.data.nhl_stats import normalize as nhl_normalize
    home = nhl_normalize(game["home_team"])
    away = nhl_normalize(game["away_team"])
    label = f"{away} @ {home}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    playoff = _game_playoff(game, _is_nhl_playoff)
    stats_available = bool(nhl_ctx["season_stats"].get(home) or nhl_ctx["season_stats"].get(away))

    home_rest = nhl_ctx["rest_days"].get(home, 2)
    away_rest = nhl_ctx["rest_days"].get(away, 2)

    # ── Pre-compute projected score shared by all bet types ───────────────────
    # Uses the blended formula (season + recent form + B2B + playoff factor) so
    # ML, Spread, and Total cards always show the same number for a given game.
    _nhl_proj_signal: Optional[str] = None
    _nhl_hs_pre = nhl_ctx["season_stats"].get(home, {})
    _nhl_as_pre = nhl_ctx["season_stats"].get(away, {})
    if _nhl_hs_pre and _nhl_as_pre:
        _nhl_hrf = nhl_ctx["recent_form"].get(home, {})
        _nhl_arf = nhl_ctx["recent_form"].get(away, {})
        _nhl_w = 0.55 if playoff else NHL_RECENT_WEIGHT
        _nhl_hgpg  = ((1 - _nhl_w) * _nhl_hs_pre.get("gpg",  3.0)
                      + _nhl_w * _nhl_hrf.get("recent_gpg",  _nhl_hs_pre.get("gpg",  3.0)))
        _nhl_hgapg = ((1 - _nhl_w) * _nhl_hs_pre.get("gapg", 3.0)
                      + _nhl_w * _nhl_hrf.get("recent_gapg", _nhl_hs_pre.get("gapg", 3.0)))
        _nhl_agpg  = ((1 - _nhl_w) * _nhl_as_pre.get("gpg",  3.0)
                      + _nhl_w * _nhl_arf.get("recent_gpg",  _nhl_as_pre.get("gpg",  3.0)))
        _nhl_agapg = ((1 - _nhl_w) * _nhl_as_pre.get("gapg", 3.0)
                      + _nhl_w * _nhl_arf.get("recent_gapg", _nhl_as_pre.get("gapg", 3.0)))
        _nhl_ph = (_nhl_hgpg  + _nhl_agapg) / 2
        _nhl_pa = (_nhl_agpg  + _nhl_hgapg) / 2
        if home_rest == 1:
            _nhl_ph -= 0.20
        if away_rest == 1:
            _nhl_pa -= 0.20
        if playoff:
            _nhl_ph *= NHL_PLAYOFF_SCORING_FACTOR
            _nhl_pa *= NHL_PLAYOFF_SCORING_FACTOR
        _nhl_proj_signal = f"Model projected score: {home} {_nhl_ph:.1f} — {away} {_nhl_pa:.1f}"

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
            _h_w  = home_stats.get('wins', 0)
            _h_l  = home_stats.get('losses', 0)
            _h_otl = home_stats.get('ot_losses', 0)
            _h_rec = f"{_h_w}W-{_h_l}L-{_h_otl}OTL | " if _h_w + _h_l > 0 else ""
            research.append(
                f"{home}: {_h_rec}{home_stats.get('gpg', '?'):.2f} GPG | "
                f"{home_stats.get('gapg', '?'):.2f} GAPG | "
                f"NetRtg {home_stats.get('net_rtg', '?'):.2f}"
            )
        if away_stats:
            _a_w  = away_stats.get('wins', 0)
            _a_l  = away_stats.get('losses', 0)
            _a_otl = away_stats.get('ot_losses', 0)
            _a_rec = f"{_a_w}W-{_a_l}L-{_a_otl}OTL | " if _a_w + _a_l > 0 else ""
            research.append(
                f"{away}: {_a_rec}{away_stats.get('gpg', '?'):.2f} GPG | "
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
        rest_diff = home_rest - away_rest
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

        # Projected score (pre-computed above; same formula as Total block)
        if _nhl_proj_signal:
            signals.append(_nhl_proj_signal)

        # Calibration capture: raw prob + cap firing trackers.
        # Add the accumulated adjustments (home ice / B2B / injuries) in MARGIN
        # space, then convert with a single CDF — previously `+ adj` added flat
        # probability points on top of the CDF, which overshoots in the tails.
        # The puck-line block below derives its margin from this prob via
        # inverse-CDF, so it stays consistent automatically.
        _nhl_raw_home = _nhl_margin_to_prob(base_margin + adj * _NHL_PROB_TO_MARGIN)
        _nhl_hardcap_fired = (_nhl_raw_home > 0.90) or (_nhl_raw_home < 0.10)
        adjusted_home_prob = min(0.90, max(0.10, _nhl_raw_home))
        adjusted_away_prob = 1 - adjusted_home_prob

        # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
        adjusted_home_prob, _nhl_cred_fired = _apply_credibility_cap_dispatched(
            adjusted_home_prob, market_home_prob, _cred_cap("nhl", NHL_CRED_CAP, "credibility_moneyline"), "nhl", "credibility_moneyline"
        )
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

            # Apply spread credibility cap
            _nhl_sp_raw_home = model_home_cover
            _nhl_sp_raw_away = model_away_cover
            model_home_cover, _nhl_sp_cred_home = _apply_credibility_cap_dispatched(
                model_home_cover, market_home_cover, _cred_cap("nhl", NHL_CRED_CAP, "credibility_spread"), "nhl", "credibility_spread"
            )
            model_away_cover, _nhl_sp_cred_away = _apply_credibility_cap_dispatched(
                model_away_cover, market_away_cover, _cred_cap("nhl", NHL_CRED_CAP, "credibility_spread"), "nhl", "credibility_spread"
            )

            away_spread_line = -home_spread_line

            home_sp_edge = model_home_cover - market_home_cover
            away_sp_edge = model_away_cover - market_away_cover

            if home_sp_edge >= _min and has_positive_ev(model_home_cover, market_home_cover):
                sizing = robinhood_kelly(model_home_cover, market_home_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(home_sp_edge, len(signals), stats_available)
                    if home_injury_capped:
                        conf = "MEDIUM"
                    _nhl_sp_rec = BetRecommendation(
                        sport="NHL", game=label, bet_type="Spread",
                        pick=f"{home} {home_spread_line:+.1f}",
                        market_prob=market_home_cover, model_prob=model_home_cover,
                        edge=home_sp_edge, contract_price=market_home_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _nhl_sp_rec.model_prob_raw = _nhl_sp_raw_home
                    _nhl_sp_rec.credibility_cap_fired = _nhl_sp_cred_home
                    recs.append(_nhl_sp_rec)

            if away_sp_edge >= _min and has_positive_ev(model_away_cover, market_away_cover):
                sizing = robinhood_kelly(model_away_cover, market_away_cover)
                if sizing.num_contracts > 0:
                    conf = _confidence_label(away_sp_edge, len(signals), stats_available)
                    if away_injury_capped:
                        conf = "MEDIUM"
                    _nhl_sp_rec = BetRecommendation(
                        sport="NHL", game=label, bet_type="Spread",
                        pick=f"{away} {away_spread_line:+.1f}",
                        market_prob=market_away_cover, model_prob=model_away_cover,
                        edge=away_sp_edge, contract_price=market_away_cover,
                        sizing=sizing, confidence=conf,
                        signals=signals[:], research=research[:],
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _nhl_sp_rec.model_prob_raw = _nhl_sp_raw_away
                    _nhl_sp_rec.credibility_cap_fired = _nhl_sp_cred_away
                    recs.append(_nhl_sp_rec)

    # --- Total (goals) ---
    total = game.get("total")
    if total and nhl_ctx["season_stats"]:
        home_stats = nhl_ctx["season_stats"].get(home, {})
        away_stats = nhl_ctx["season_stats"].get(away, {})
        if home_stats and away_stats:
            # Blend season GPG/GAPG with recent form — same weight used by ML/Spread
            home_recent_form = nhl_ctx["recent_form"].get(home, {})
            away_recent_form = nhl_ctx["recent_form"].get(away, {})
            w = 0.55 if playoff else NHL_RECENT_WEIGHT

            home_gpg  = ((1 - w) * home_stats.get("gpg",  3.0)
                         + w * home_recent_form.get("recent_gpg",  home_stats.get("gpg",  3.0)))
            home_gapg = ((1 - w) * home_stats.get("gapg", 3.0)
                         + w * home_recent_form.get("recent_gapg", home_stats.get("gapg", 3.0)))
            away_gpg  = ((1 - w) * away_stats.get("gpg",  3.0)
                         + w * away_recent_form.get("recent_gpg",  away_stats.get("gpg",  3.0)))
            away_gapg = ((1 - w) * away_stats.get("gapg", 3.0)
                         + w * away_recent_form.get("recent_gapg", away_stats.get("gapg", 3.0)))

            exp_home = (home_gpg + away_gapg) / 2
            exp_away = (away_gpg + home_gapg) / 2

            if home_rest == 1:
                exp_home -= 0.20
            if away_rest == 1:
                exp_away -= 0.20

            # Playoff scoring factor — NHL playoffs average ~8% fewer goals
            if playoff:
                exp_home *= NHL_PLAYOFF_SCORING_FACTOR
                exp_away *= NHL_PLAYOFF_SCORING_FACTOR

            expected_total = exp_home + exp_away

            market_line = total["line"]
            total_std = NHL_TOTAL_STD
            blended_total = (1 - _TOTAL_MARKET_ANCHOR) * expected_total + _TOTAL_MARKET_ANCHOR * market_line
            model_over_prob  = float(1 - norm.cdf(market_line, blended_total, total_std))
            market_over_prob  = total["over_prob"]
            market_under_prob = total["under_prob"]

            _nhl_total_raw_over = model_over_prob
            model_over_prob, _nhl_total_cred_fired = _apply_credibility_cap_dispatched(
                model_over_prob, market_over_prob, _cred_cap("nhl", NHL_CRED_CAP, "credibility_total"), "nhl", "credibility_total"
            )

            total_signals = [
                _nhl_proj_signal or f"Model projected score: {home} {exp_home:.1f} — {away} {exp_away:.1f}",
                f"Model expected total: {blended_total:.1f} vs market line {market_line}",
            ]
            total_research = [
                f"{home} GPG (blended): {home_gpg:.2f} vs {away} GAPG (blended): {away_gapg:.2f}",
                f"{away} GPG (blended): {away_gpg:.2f} vs {home} GAPG (blended): {home_gapg:.2f}",
                f"Recent form weight: {w*100:.0f}% | Season weight: {(1-w)*100:.0f}%",
            ]

            _over_label  = f"Over {market_line - 0.5}"  if market_line % 1 == 0 else f"Over {market_line}"
            _under_label = f"Under {market_line + 0.5}" if market_line % 1 == 0 else f"Under {market_line}"

            over_edge  = model_over_prob - market_over_prob
            under_edge = (1 - model_over_prob) - market_under_prob

            if over_edge >= _min and has_positive_ev(model_over_prob, market_over_prob):
                sizing = robinhood_kelly(model_over_prob, market_over_prob)
                if sizing.num_contracts > 0:
                    _over_rec = BetRecommendation(
                        sport="NHL", game=label, bet_type="Total", pick=_over_label,
                        market_prob=market_over_prob, model_prob=model_over_prob,
                        edge=over_edge, contract_price=market_over_prob,
                        sizing=sizing,
                        confidence=_confidence_label(over_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _over_rec.model_prob_raw = _nhl_total_raw_over
                    _over_rec.credibility_cap_fired = _nhl_total_cred_fired
                    recs.append(_over_rec)
            if under_edge >= _min and has_positive_ev(1 - model_over_prob, market_under_prob):
                sizing = robinhood_kelly(1 - model_over_prob, market_under_prob)
                if sizing.num_contracts > 0:
                    _under_rec = BetRecommendation(
                        sport="NHL", game=label, bet_type="Total", pick=_under_label,
                        market_prob=market_under_prob, model_prob=1 - model_over_prob,
                        edge=under_edge, contract_price=market_under_prob,
                        sizing=sizing,
                        confidence=_confidence_label(under_edge, len(total_signals), stats_available),
                        signals=total_signals, research=total_research,
                        home_team=home, away_team=away, game_time=game_time,
                        commence_time=commence_time,
                    )
                    _under_rec.model_prob_raw = 1.0 - _nhl_total_raw_over
                    _under_rec.credibility_cap_fired = _nhl_total_cred_fired
                    recs.append(_under_rec)

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _home_raw = _lv.get("_nhl_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home, away_team=away,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_nhl_cred_fired", False),
        hardcap_fired=_lv.get("_nhl_hardcap_fired", False),
        home_injury_fired=_lv.get("home_injury_capped", False),
        away_injury_fired=_lv.get("away_injury_capped", False),
        stats_avail=stats_available,
    )
    _nhl_rh = _lv.get("_nhl_raw_home"); _nhl_ro = _lv.get("_nhl_total_raw_over")
    _stamp_decision(game, _min, {
        "home_strength": _lv.get("home_strength"), "away_strength": _lv.get("away_strength"),
        "base_margin": _lv.get("base_margin"), "effective_margin": _lv.get("effective_margin"),
        "win_prob_adj": _lv.get("adj"), "rest_diff": _lv.get("rest_diff"),
        "expected_total": _lv.get("blended_total"), "total_std": _lv.get("total_std"),
        "home_rest": _lv.get("home_rest"), "away_rest": _lv.get("away_rest"),
        "home_inj": _lv.get("home_inj"), "away_inj": _lv.get("away_inj"),
        "playoff": _lv.get("playoff"), "stats_available": stats_available,
    }, [
        ("Moneyline", home, _lv.get("adjusted_home_prob"), _nhl_rh, _lv.get("market_home_prob"), None),
        ("Moneyline", away, _lv.get("adjusted_away_prob"),
         (1.0 - _nhl_rh) if _nhl_rh is not None else None, _lv.get("market_away_prob"), None),
        ("Spread", home, _lv.get("model_home_cover"), _lv.get("_nhl_sp_raw_home"),
         _lv.get("market_home_cover"), _lv.get("home_spread_line")),
        ("Spread", away, _lv.get("model_away_cover"), _lv.get("_nhl_sp_raw_away"),
         _lv.get("market_away_cover"),
         (-_lv["home_spread_line"]) if _lv.get("home_spread_line") is not None else None),
        ("Total", "over", _lv.get("model_over_prob"), _nhl_ro, _lv.get("market_over_prob"), _lv.get("market_line")),
        ("Total", "under",
         (1.0 - _lv["model_over_prob"]) if _lv.get("model_over_prob") is not None else None,
         (1.0 - _nhl_ro) if _nhl_ro is not None else None,
         _lv.get("market_under_prob"), _lv.get("market_line")),
    ], recs)
    return recs


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
# Player unavailabilities: penalty per confirmed absent key player, capped per team.
IPL_ABSENT_PLAYER_IMPACT = 0.04
IPL_ABSENT_MAX_IMPACT    = 0.08


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

    # --- Known player unavailabilities ---
    unavailabilities = ipl_ctx.get("unavailabilities", {})
    home_absent = unavailabilities.get(home, [])
    away_absent = unavailabilities.get(away, [])

    if home_absent:
        penalty = min(len(home_absent) * IPL_ABSENT_PLAYER_IMPACT, IPL_ABSENT_MAX_IMPACT)
        adj -= penalty
        names = ", ".join(home_absent)
        signals.append(f"🚫 {home} missing: {names} (-{penalty*100:.0f}%)")
        research.append(f"Unavailable ({home}): {names}")

    if away_absent:
        penalty = min(len(away_absent) * IPL_ABSENT_PLAYER_IMPACT, IPL_ABSENT_MAX_IMPACT)
        adj += penalty
        names = ", ".join(away_absent)
        signals.append(f"🚫 {away} missing: {names} (+{penalty*100:.0f}% for {home})")
        research.append(f"Unavailable ({away}): {names}")

    # --- Projected score (runs) ---
    _ipl_avg_fi = 165.0
    _fi = vstats.get("avg_first_innings") or _ipl_avg_fi
    _home_margin = home_form.get("avg_margin", 0.0)
    _away_margin = away_form.get("avg_margin", 0.0)
    _proj_home_runs = _fi + _home_margin * 0.15
    _proj_away_runs = _fi + _away_margin * 0.15
    signals.append(f"Model projected score: {home} {_proj_home_runs:.0f} — {away} {_proj_away_runs:.0f}")

    # Calibration capture: raw prob + cap firing trackers.
    _ipl_raw_home = base_home_prob + adj
    _ipl_hardcap_fired = (_ipl_raw_home > 0.90) or (_ipl_raw_home < 0.10)
    adjusted_home_prob = min(0.90, max(0.10, _ipl_raw_home))
    adjusted_away_prob = 1.0 - adjusted_home_prob

    # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
    adjusted_home_prob, _ipl_cred_fired = _apply_credibility_cap_dispatched(
        adjusted_home_prob, market_home_prob, _cred_cap("ipl", IPL_CRED_CAP, "credibility_moneyline"), "ipl", "credibility_moneyline"
    )
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

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _home_raw = _lv.get("_ipl_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home, away_team=away,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_ipl_cred_fired", False),
        hardcap_fired=_lv.get("_ipl_hardcap_fired", False),
        stats_avail=stats_available,
    )
    _stamp_decision(game, _min, {
        "base_home_prob": _lv.get("base_home_prob"), "win_prob_adj": _lv.get("adj"),
        "home_strength": _lv.get("home_strength"), "away_strength": _lv.get("away_strength"),
        "home_rest": _lv.get("home_rest"), "away_rest": _lv.get("away_rest"),
        "recent_gap": _lv.get("recent_gap"),
        "stats_available": stats_available,
    }, [
        ("Moneyline", home, _lv.get("adjusted_home_prob"), _lv.get("_ipl_raw_home"), _lv.get("market_home_prob"), None),
        ("Moneyline", away,
         (1.0 - _lv["adjusted_home_prob"]) if _lv.get("adjusted_home_prob") is not None else None,
         (1.0 - _lv["_ipl_raw_home"]) if _lv.get("_ipl_raw_home") is not None else None,
         _lv.get("market_away_prob"), None),
    ], recs)
    return recs


# ---------------------------------------------------------------------------
# WNBA — moneyline analysis
# ---------------------------------------------------------------------------

def analyze_wnba_game(
    game: Dict,
    wnba_ctx: Dict,
    wnba_injuries: Dict,
    min_edge: float = None,
) -> List[BetRecommendation]:
    """
    Moneyline-only edge finder for WNBA (watchlist, no budget allocation).
    Uses team offensive/defensive ratings, recent form, B2B, home advantage,
    and a points-share based lineup penalty for injured players.
    """
    from src.data.wnba_stats import normalize as wnba_normalize, _name_match
    from src.config import (
        WNBA_HOME_ADVANTAGE, WNBA_BACK_TO_BACK_PENALTY, WNBA_RECENT_WEIGHT,
        WNBA_SOS_WEIGHT,
        WNBA_SPREAD_STD, WNBA_REPLACEMENT_RATE, WNBA_MAX_LINEUP_PENALTY,
        WNBA_STATUS_WEIGHTS,
    )
    from scipy.stats import norm as _norm
    _zero_sizing = BetSizing(
        dollar_allocation=0, num_contracts=0, contract_price=0,
        total_cost=0, profit_if_win=0, loss_if_lose=0,
        expected_value=0, kelly_fraction=0,
    )

    home_raw = game["home_team"]
    away_raw = game["away_team"]
    label    = f"{away_raw} @ {home_raw}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    recs = []
    _min = min_edge if min_edge is not None else MIN_EDGE

    ml = game.get("moneyline")
    if not ml:
        return recs

    market_home_prob = ml["home_prob"]
    market_away_prob = ml["away_prob"]

    # Resolve team names against ctx keys (flexible matching)
    def _resolve(raw: str, ctx_dict: Dict) -> str:
        for key in ctx_dict:
            if _name_match(raw, key):
                return key
        return raw

    home = _resolve(home_raw, wnba_ctx["season_stats"])
    away = _resolve(away_raw, wnba_ctx["season_stats"])

    home_stats  = wnba_ctx["season_stats"].get(home, {})
    away_stats  = wnba_ctx["season_stats"].get(away, {})
    home_recent = wnba_ctx["recent_form"].get(home, {})
    away_recent = wnba_ctx["recent_form"].get(away, {})
    home_rest   = wnba_ctx["rest_days"].get(home, 3)
    away_rest   = wnba_ctx["rest_days"].get(away, 3)

    stats_available = bool(home_stats or away_stats)

    signals  = []
    research = []

    if not stats_available:
        signals.append("⚠ WNBA stats unavailable — using market baseline only")

    # ── Base strength from season stats ──────────────────────────────────────
    home_ppg = home_stats.get("ppg", 82.0)
    away_ppg = away_stats.get("ppg", 82.0)
    home_net  = home_stats.get("net_rtg", 0.0)
    away_net  = away_stats.get("net_rtg", 0.0)

    # Blended net rating: season + recent form
    home_recent_net = home_recent.get("recent_net_rtg", home_net)
    away_recent_net = away_recent.get("recent_net_rtg", away_net)
    home_blended = (1 - WNBA_RECENT_WEIGHT) * home_net + WNBA_RECENT_WEIGHT * home_recent_net
    away_blended = (1 - WNBA_RECENT_WEIGHT) * away_net + WNBA_RECENT_WEIGHT * away_recent_net

    # ── Strength of schedule ──────────────────────────────────────────────────
    # Nudge each team's rating by the average net rating of opponents faced
    # (partial weight). Beating a tough slate → rating revised up; feasting on
    # weak opponents → revised down. When both teams' SOS is similar it cancels
    # in net_diff, so this only moves the line when schedules genuinely differ.
    sos_map  = wnba_ctx.get("sos", {})
    home_sos = sos_map.get(home, 0.0)
    away_sos = sos_map.get(away, 0.0)
    home_blended += WNBA_SOS_WEIGHT * home_sos
    away_blended += WNBA_SOS_WEIGHT * away_sos
    if abs(home_sos) >= 0.5 or abs(away_sos) >= 0.5:
        research.append(
            f"Strength of schedule — {home_raw}: opp avg net {home_sos:+.1f} | "
            f"{away_raw}: opp avg net {away_sos:+.1f} (×{WNBA_SOS_WEIGHT:g} applied)"
        )

    # Point margin from blended net ratings. The probability adjustments below
    # (home court, B2B, injuries) are accumulated in `adj` as probability points,
    # then converted to MARGIN points and added to net_diff before a single
    # Normal CDF maps the total to a win probability (see _wnba_raw_home). This
    # replaces the old `base_prob + adj` (adding flat percentage points on top of
    # the CDF), which overshot in the tails. Conversion via the central CDF slope
    # leaves mid-range picks unchanged and compresses only the tails.
    net_diff = home_blended - away_blended
    _wnba_prob_to_margin = 1.0 / float(_norm.pdf(0, 0, WNBA_SPREAD_STD))

    adj = 0.0

    # ── Home advantage ────────────────────────────────────────────────────────
    adj += WNBA_HOME_ADVANTAGE
    signals.append(f"Home court: {home_raw} (+{WNBA_HOME_ADVANTAGE*100:.0f}%)")

    # ── Stats research ────────────────────────────────────────────────────────
    if home_stats:
        _h_w, _h_l = home_stats.get('wins', 0), home_stats.get('losses', 0)
        _h_rec = f"{_h_w}W-{_h_l}L | " if _h_w + _h_l > 0 else ""
        research.append(
            f"{home_raw}: {_h_rec}{home_ppg} PPG | FG {home_stats.get('fg_pct', 0):.1f}% | "
            f"AST/TO {home_stats.get('ast_to', 0):.2f} | NetRtg {home_net:+.1f}"
        )
    if away_stats:
        _a_w, _a_l = away_stats.get('wins', 0), away_stats.get('losses', 0)
        _a_rec = f"{_a_w}W-{_a_l}L | " if _a_w + _a_l > 0 else ""
        research.append(
            f"{away_raw}: {_a_rec}{away_ppg} PPG | FG {away_stats.get('fg_pct', 0):.1f}% | "
            f"AST/TO {away_stats.get('ast_to', 0):.2f} | NetRtg {away_net:+.1f}"
        )

    # ── Recent form ───────────────────────────────────────────────────────────
    if home_recent and home_recent.get("games", 0) >= 3:
        research.append(
            f"{home_raw} last {home_recent['games']} games: "
            f"{home_recent['recent_ppg']} PPG | {home_recent['recent_w_pct']*100:.0f}% W"
        )
        if home_recent["recent_net_rtg"] > 5:
            signals.append(f"{home_raw} in strong form (last {home_recent['games']} games: +{home_recent['recent_net_rtg']:.1f} net)")
        elif home_recent["recent_net_rtg"] < -5:
            signals.append(f"{home_raw} struggling recently (last {home_recent['games']} games: {home_recent['recent_net_rtg']:.1f} net)")

    if away_recent and away_recent.get("games", 0) >= 3:
        research.append(
            f"{away_raw} last {away_recent['games']} games: "
            f"{away_recent['recent_ppg']} PPG | {away_recent['recent_w_pct']*100:.0f}% W"
        )
        if away_recent["recent_net_rtg"] > 5:
            signals.append(f"{away_raw} in strong form (last {away_recent['games']} games: +{away_recent['recent_net_rtg']:.1f} net)")
        elif away_recent["recent_net_rtg"] < -5:
            signals.append(f"{away_raw} struggling recently (last {away_recent['games']} games: {away_recent['recent_net_rtg']:.1f} net)")

    # ── Back-to-back ──────────────────────────────────────────────────────────
    research.append(f"Rest — {home_raw}: {home_rest} day(s) | {away_raw}: {away_rest} day(s)")
    if home_rest == 1:
        adj -= WNBA_BACK_TO_BACK_PENALTY
        signals.append(f"{home_raw} on back-to-back (-{WNBA_BACK_TO_BACK_PENALTY*100:.0f}%)")
    if away_rest == 1:
        adj += WNBA_BACK_TO_BACK_PENALTY
        signals.append(f"{away_raw} on back-to-back — favors {home_raw} (+{WNBA_BACK_TO_BACK_PENALTY*100:.0f}%)")

    # ── Projected score (quality/pace-aware, additive offense-vs-defense) ──────
    # Standard ratings combination: a team's expected output = its offense + the
    # opponent's points-allowed − league average. This correctly spreads the
    # projection for tempo/quality mismatches (strong offense vs leaky defense
    # projects high; vs elite defense projects low) instead of regressing to the
    # mean like a simple average. Uses the real opp_ppg now available (Tier 1).
    _wnba_avg = 82.0
    _away_def_allowed = away_recent.get("recent_opp_ppg", away_stats.get("opp_ppg", _wnba_avg))
    _home_def_allowed = home_recent.get("recent_opp_ppg", home_stats.get("opp_ppg", _wnba_avg))
    _proj_home_pts = max(50.0, min(120.0, home_ppg + _away_def_allowed - _wnba_avg))
    _proj_away_pts = max(50.0, min(120.0, away_ppg + _home_def_allowed - _wnba_avg))
    signals.append(f"Model projected score: {home_raw} {_proj_home_pts:.0f} — {away_raw} {_proj_away_pts:.0f}")

    # ── Lineup penalty (points-share based) ───────────────────────────────────
    def _lineup_penalty(team_display: str, team_raw: str) -> float:
        """Compute compound lineup penalty from injured players' points share."""
        inj_list = []
        for key, val in wnba_injuries.items():
            if _name_match(team_display, key) or _name_match(team_raw, key):
                inj_list = val
                break
        if not inj_list:
            return 0.0

        # Compound: penalty = 1 - product(1 - single_player_penalty)
        compound = 1.0
        for p in inj_list:
            status_wt = WNBA_STATUS_WEIGHTS.get(p.get("status", "Out"), 1.0)
            weight    = p.get("player_weight", 0.0)
            single    = weight * status_wt * (1.0 - WNBA_REPLACEMENT_RATE)
            single    = min(single, 0.15)   # per-player cap
            compound  *= (1.0 - single)

        total = min(1.0 - compound, WNBA_MAX_LINEUP_PENALTY)
        return round(total, 3)

    home_lineup_pen = _lineup_penalty(home, home_raw)
    away_lineup_pen = _lineup_penalty(away, away_raw)

    # Injury research lines
    for key, inj_list in wnba_injuries.items():
        if _name_match(home, key) or _name_match(home_raw, key):
            for p in inj_list:
                ppg_str = f" ({p['ppg']} PPG, {p['points_share']*100:.0f}% pts share)" if p.get("ppg", 0) > 0 else ""
                research.append(f"⚕ {home_raw} — {p['player']}{ppg_str}: {p['status'].upper()}")
        if _name_match(away, key) or _name_match(away_raw, key):
            for p in inj_list:
                ppg_str = f" ({p['ppg']} PPG, {p['points_share']*100:.0f}% pts share)" if p.get("ppg", 0) > 0 else ""
                research.append(f"⚕ {away_raw} — {p['player']}{ppg_str}: {p['status'].upper()}")

    if home_lineup_pen > 0.02:
        adj -= home_lineup_pen
        signals.append(f"{home_raw} lineup impact (-{home_lineup_pen*100:.1f}%)")
    if away_lineup_pen > 0.02:
        adj += away_lineup_pen
        signals.append(f"{away_raw} injuries benefit {home_raw} (+{away_lineup_pen*100:.1f}%)")

    if not home_lineup_pen and not away_lineup_pen:
        research.append("No significant injuries reported for either team")

    # ── Final probability ─────────────────────────────────────────────────────
    # Calibration capture: raw prob + cap firing trackers.
    # Adjustments added in margin space, then one CDF (see net_diff comment).
    _wnba_raw_home = float(_norm.cdf((net_diff + adj * _wnba_prob_to_margin) / WNBA_SPREAD_STD))
    _wnba_hardcap_fired = (_wnba_raw_home > 0.90) or (_wnba_raw_home < 0.10)
    adj_home_prob = min(0.90, max(0.10, _wnba_raw_home))
    adj_away_prob = 1.0 - adj_home_prob

    # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
    # Standardised across sports; tighter than the prior ±0.20 to limit cold-start
    # overconfidence on small-sample WNBA stats.
    adj_home_prob, _wnba_cred_fired = _apply_credibility_cap_dispatched(
        adj_home_prob, market_home_prob, _cred_cap("wnba", WNBA_CRED_CAP, "credibility_moneyline"), "wnba", "credibility_moneyline"
    )
    adj_away_prob = 1.0 - adj_home_prob

    # ── Build recommendations ─────────────────────────────────────────────────
    for team_raw, model_prob, market_prob in [
        (home_raw, adj_home_prob, market_home_prob),
        (away_raw, adj_away_prob, market_away_prob),
    ]:
        edge = model_prob - market_prob
        if edge < _min:
            continue

        conf = _confidence_label(edge, len(signals), stats_available)

        recs.append(BetRecommendation(
            sport="WNBA",
            game=label,
            home_team=home_raw,
            away_team=away_raw,
            pick=team_raw,
            bet_type="Moneyline",
            model_prob=model_prob,
            market_prob=market_prob,
            edge=edge,
            contract_price=market_prob,
            sizing=_zero_sizing,
            confidence=conf,
            signals=signals[:],
            research=research[:],
            game_time=game_time,
            commence_time=commence_time,
        ))

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _home_raw = _lv.get("_wnba_raw_home")
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home_raw, away_team=away_raw,
        home_raw=_home_raw,
        away_raw=(1.0 - _home_raw) if _home_raw is not None else None,
        cred_fired=_lv.get("_wnba_cred_fired", False),
        hardcap_fired=_lv.get("_wnba_hardcap_fired", False),
        stats_avail=stats_available,
    )
    _stamp_decision(game, _min, {
        "home_net": _lv.get("home_net"), "away_net": _lv.get("away_net"),
        "net_rtg_source": home_stats.get("net_rtg_source", "schedule"),
        "home_recent_net": _lv.get("home_recent_net"), "away_recent_net": _lv.get("away_recent_net"),
        "net_diff": _lv.get("net_diff"), "win_prob_adj": _lv.get("adj"),
        "home_sos": _lv.get("home_sos"), "away_sos": _lv.get("away_sos"),
        "home_lineup_pen": _lv.get("home_lineup_pen"), "away_lineup_pen": _lv.get("away_lineup_pen"),
        "home_rest": _lv.get("home_rest"), "away_rest": _lv.get("away_rest"),
        "stats_available": stats_available,
    }, [
        ("Moneyline", home_raw, _lv.get("adj_home_prob"), _lv.get("_wnba_raw_home"), _lv.get("market_home_prob"), None),
        ("Moneyline", away_raw, _lv.get("adj_away_prob"),
         (1.0 - _lv["_wnba_raw_home"]) if _lv.get("_wnba_raw_home") is not None else None,
         _lv.get("market_away_prob"), None),
    ], recs)
    return recs


# ---------------------------------------------------------------------------
# MLS — Poisson goal model
# ---------------------------------------------------------------------------

def _poisson_prob(lam: float, k: int) -> float:
    from math import exp, factorial
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return exp(-lam) * (lam ** k) / factorial(k)


def _dc_tau(i: int, j: int, lam_home: float, lam_away: float, rho: float) -> float:
    """
    Dixon-Coles low-score correction factor. Independent Poisson misprices the
    four lowest-scoring cells (real soccer goals are mildly negatively correlated
    at low scores, producing more draws than independence implies). rho < 0
    raises 0-0/1-1 and lowers 1-0/0-1.
    """
    if i == 0 and j == 0:
        return 1.0 - lam_home * lam_away * rho
    if i == 0 and j == 1:
        return 1.0 + lam_home * rho
    if i == 1 and j == 0:
        return 1.0 + lam_away * rho
    if i == 1 and j == 1:
        return 1.0 - rho
    return 1.0


def _mls_prob_matrix(lam_home: float, lam_away: float, max_goals: int = 10,
                     rho: float = 0.0) -> dict:
    """
    Returns {(i, j): probability} for all goal combinations 0..max_goals.

    With rho != 0, applies the Dixon-Coles low-score correction and renormalizes
    so the matrix still sums to 1. rho defaults to 0 (pure independent Poisson)
    for backward compatibility / callers that don't pass it.
    """
    matrix = {}
    for i in range(max_goals + 1):
        ph = _poisson_prob(lam_home, i)
        for j in range(max_goals + 1):
            matrix[(i, j)] = ph * _poisson_prob(lam_away, j)

    if rho:
        for (i, j) in matrix:
            if i <= 1 and j <= 1:
                matrix[(i, j)] = max(0.0, matrix[(i, j)] * _dc_tau(i, j, lam_home, lam_away, rho))
        total = sum(matrix.values())
        if total > 0:
            for k in matrix:
                matrix[k] /= total
    return matrix


def analyze_mls_game(
    game: Dict,
    mls_ctx: Dict,
    mls_injuries: Dict,
    min_edge: float = None,
) -> List[BetRecommendation]:
    """
    Poisson-based edge finder for MLS (watchlist, no budget allocation).
    Generates Moneyline, Draw, Total (Over/Under), and Spread picks.
    """
    from src.data.mls_stats import normalize as mls_normalize
    from src.config import (
        MLS_LEAGUE_HOME_XG, MLS_LEAGUE_AWAY_XG, MLS_RECENT_WEIGHT,
        MLS_STRONG_VENUES, MLS_STRONG_VENUE_ADV, MLS_MAX_INJURY_PENALTY,
        MLS_DC_RHO,
    )

    _zero_sizing = BetSizing(
        dollar_allocation=0, num_contracts=0, contract_price=0,
        total_cost=0, profit_if_win=0, loss_if_lose=0,
        expected_value=0, kelly_fraction=0,
    )

    home_raw = game["home_team"]
    away_raw = game["away_team"]
    label    = f"{away_raw} @ {home_raw}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    _min = min_edge if min_edge is not None else MIN_EDGE
    recs = []

    ml = game.get("moneyline")
    if not ml or "draw_prob" not in ml:
        return recs  # soccer moneyline must have 3-way probs

    market_home_prob = ml["home_prob"]
    market_draw_prob = ml["draw_prob"]
    market_away_prob = ml["away_prob"]

    # Resolve names
    season_stats = mls_ctx.get("season_stats", {})
    recent_form  = mls_ctx.get("recent_form", {})
    rest_days    = mls_ctx.get("rest_days", {})

    home = mls_normalize(home_raw, season_stats)
    away = mls_normalize(away_raw, season_stats)

    home_stats  = season_stats.get(home, {})
    away_stats  = season_stats.get(away, {})
    home_recent = recent_form.get(home, {})
    away_recent = recent_form.get(away, {})

    stats_available = bool(home_stats or away_stats)

    signals  = []
    research = []

    # Base xGF/xGA (season)
    home_xgf = home_stats.get("xgf_per_game", MLS_LEAGUE_HOME_XG)
    home_xga = home_stats.get("xga_per_game", MLS_LEAGUE_AWAY_XG)
    away_xgf = away_stats.get("xgf_per_game", MLS_LEAGUE_AWAY_XG)
    away_xga = away_stats.get("xga_per_game", MLS_LEAGUE_HOME_XG)

    # Recent form blend
    home_rxgf = home_recent.get("recent_xgf", home_xgf)
    home_rxga = home_recent.get("recent_xga", home_xga)
    away_rxgf = away_recent.get("recent_xgf", away_xgf)
    away_rxga = away_recent.get("recent_xga", away_xga)

    blended_home_xgf = (1 - MLS_RECENT_WEIGHT) * home_xgf + MLS_RECENT_WEIGHT * home_rxgf
    blended_home_xga = (1 - MLS_RECENT_WEIGHT) * home_xga + MLS_RECENT_WEIGHT * home_rxga
    blended_away_xgf = (1 - MLS_RECENT_WEIGHT) * away_xgf + MLS_RECENT_WEIGHT * away_rxgf
    blended_away_xga = (1 - MLS_RECENT_WEIGHT) * away_xga + MLS_RECENT_WEIGHT * away_rxga

    # League averages — live from ASA (current season), config fallback. The
    # model normalizes each team's attack/defense by the OVERALL league mean,
    # then scales the home side to the league HOME scoring level and the away
    # side to the AWAY level. That fixes the prior bug (defense normalized by the
    # attack baseline, inflating total λ ~17%: avg matchup 3.54 vs actual ~3.03)
    # and makes league home advantage intrinsic to the scaling.
    _la = mls_ctx.get("league_avg") or {}
    LG_HOME    = _la.get("home", MLS_LEAGUE_HOME_XG)
    LG_AWAY    = _la.get("away", MLS_LEAGUE_AWAY_XG)
    LG_OVERALL = _la.get("overall") or ((LG_HOME + LG_AWAY) / 2)

    attack_home   = blended_home_xgf / LG_OVERALL
    defense_away  = blended_away_xga / LG_OVERALL
    attack_away   = blended_away_xgf / LG_OVERALL
    defense_home  = blended_home_xga / LG_OVERALL

    lam_home = max(0.3, min(4.0, LG_HOME * attack_home * defense_away))
    lam_away = max(0.3, min(4.0, LG_AWAY * attack_away * defense_home))

    # Home advantage is now intrinsic to the home/away scaling above (the home
    # side scales toward the league HOME level, the away side toward the lower
    # AWAY level). Apply only a small extra bump for known fortress venues;
    # team-specific home-field modeling is a future refinement (it previously
    # double-counted the league home edge).
    if home_raw in MLS_STRONG_VENUES:
        lam_home = min(4.0, lam_home * (1 + MLS_STRONG_VENUE_ADV))
        signals.append(f"Home fortress venue: {home_raw} (+{MLS_STRONG_VENUE_ADV*100:.0f}%)")

    # Research: team stats
    n_home = home_recent.get("games", 0)
    n_away = away_recent.get("games", 0)
    if home_stats:
        _h_w, _h_l = home_stats.get('wins', 0), home_stats.get('losses', 0)
        _h_rec = f"{_h_w}W-{_h_l}L | " if _h_w + _h_l > 0 else ""
        research.append(
            f"{home_raw}: {_h_rec}{home_xgf:.2f} xGF/g | {home_xga:.2f} xGA/g | "
            f"{home_xgf - home_xga:+.2f} xGD/g"
        )
    if away_stats:
        _a_w, _a_l = away_stats.get('wins', 0), away_stats.get('losses', 0)
        _a_rec = f"{_a_w}W-{_a_l}L | " if _a_w + _a_l > 0 else ""
        research.append(
            f"{away_raw}: {_a_rec}{away_xgf:.2f} xGF/g | {away_xga:.2f} xGA/g | "
            f"{away_xgf - away_xga:+.2f} xGD/g"
        )
    if n_home >= 3:
        research.append(
            f"Recent ({n_home}g) — {home_raw}: {home_rxgf:.2f} xGF | {home_rxga:.2f} xGA"
        )
        if home_rxgf - home_rxga > 0.4:
            signals.append(f"{home_raw} recent form: {home_rxgf - home_rxga:+.2f} xGD last {n_home} games")
        elif home_rxgf - home_rxga < -0.4:
            signals.append(f"{home_raw} struggling recently: {home_rxgf - home_rxga:+.2f} xGD last {n_home} games")
    if n_away >= 3:
        research.append(
            f"Recent ({n_away}g) — {away_raw}: {away_rxgf:.2f} xGF | {away_rxga:.2f} xGA"
        )
        if away_rxgf - away_rxga > 0.4:
            signals.append(f"{away_raw} recent form: {away_rxgf - away_rxga:+.2f} xGD last {n_away} games")
        elif away_rxgf - away_rxga < -0.4:
            signals.append(f"{away_raw} struggling recently: {away_rxgf - away_rxga:+.2f} xGD last {n_away} games")

    # xG edge signal
    home_xgd = blended_home_xgf - blended_home_xga
    away_xgd = blended_away_xgf - blended_away_xga
    xgd_diff = home_xgd - away_xgd
    if abs(xgd_diff) >= 0.25:
        stronger = home_raw if xgd_diff > 0 else away_raw
        signals.append(f"xG edge: {stronger} ({xgd_diff:+.2f} xGD advantage)")

    # Injury penalties
    def _injury_penalty(team_display: str) -> float:
        inj_list = []
        for key, val in mls_injuries.items():
            if key.lower() == team_display.lower() or team_display.lower() in key.lower() or key.lower() in team_display.lower():
                inj_list = val
                break
        if not inj_list:
            return 0.0
        compound = 1.0
        for p in inj_list:
            if p.get("status", "").lower() in ("out", "doubtful"):
                gc = p.get("goals_contrib", 0.3)
                single = min(0.08, gc * 0.15)
                compound *= (1.0 - single)
        return min(MLS_MAX_INJURY_PENALTY, 1.0 - compound)

    home_inj_pen = _injury_penalty(home_raw)
    away_inj_pen = _injury_penalty(away_raw)
    if home_inj_pen > 0.02:
        lam_home *= (1.0 - home_inj_pen)
        signals.append(f"⚕ {home_raw} injury impact (-{home_inj_pen*100:.0f}%)")
    if away_inj_pen > 0.02:
        lam_away *= (1.0 - away_inj_pen)
        signals.append(f"⚕ {away_raw} injury impact (-{away_inj_pen*100:.0f}%)")

    # Log injury research
    for key, inj_list in mls_injuries.items():
        if key.lower() in home_raw.lower() or home_raw.lower() in key.lower():
            for p in inj_list:
                research.append(f"⚕ {home_raw} — {p['player']}: {p['status']}")
        if key.lower() in away_raw.lower() or away_raw.lower() in key.lower():
            for p in inj_list:
                research.append(f"⚕ {away_raw} — {p['player']}: {p['status']}")

    # Clip lambdas after injury adjustments
    lam_home = max(0.3, min(4.0, lam_home))
    lam_away = max(0.3, min(4.0, lam_away))

    # Probability matrix
    matrix = _mls_prob_matrix(lam_home, lam_away, rho=MLS_DC_RHO)
    p_home_win = sum(v for (i, j), v in matrix.items() if i > j)
    p_draw     = sum(v for (i, j), v in matrix.items() if i == j)
    p_away_win = sum(v for (i, j), v in matrix.items() if i < j)

    total_p = p_home_win + p_draw + p_away_win
    if total_p > 0:
        p_home_win /= total_p
        p_draw     /= total_p
        p_away_win /= total_p

    research.append(
        f"xG projection: {home_raw} {lam_home:.2f} – {away_raw} {lam_away:.2f}"
    )
    research.append(
        f"Win probs: H {p_home_win:.1%} | D {p_draw:.1%} | A {p_away_win:.1%}"
    )
    signals.append(f"xG projection: {lam_home:.2f} – {lam_away:.2f} goals")

    # Calibration capture: raw 3-way probabilities (pre-cred-cap).
    # No hard cap is applied in MLS (probs come from Poisson matrix and sum to 1),
    # so hardcap_fired = False.
    _mls_raw_home = p_home_win
    _mls_raw_away = p_away_win

    # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
    # Tighter than other sports (MLS_CRED_CAP = 0.10) due to small-sample noise
    # in early-season xG data; will be auto-relaxed by the calibration engine as
    # the season progresses and more game data accumulates.
    p_home_win, _mls_cred_home_fired = _apply_credibility_cap_dispatched(
        p_home_win, market_home_prob, _cred_cap("mls", MLS_CRED_CAP, "credibility_moneyline"), "mls", "credibility_moneyline"
    )
    p_draw, _mls_cred_draw_fired = _apply_credibility_cap_dispatched(
        p_draw, market_draw_prob, _cred_cap("mls", MLS_CRED_CAP, "credibility_draw"), "mls", "credibility_draw"
    )
    p_away_win = 1.0 - p_home_win - p_draw
    p_away_win, _mls_cred_away_fired = _apply_credibility_cap_dispatched(
        p_away_win, market_away_prob, _cred_cap("mls", MLS_CRED_CAP, "credibility_moneyline"), "mls", "credibility_moneyline"
    )
    _mls_cred_fired = _mls_cred_home_fired or _mls_cred_draw_fired or _mls_cred_away_fired

    # Build BetRecommendations
    def _make_rec(pick_str, bet_type, model_prob, market_prob):
        edge = model_prob - market_prob
        if edge < _min:
            return None
        conf = _confidence_label(edge, len(signals), stats_available)
        return BetRecommendation(
            sport="MLS", game=label,
            home_team=home_raw, away_team=away_raw,
            pick=pick_str, bet_type=bet_type,
            model_prob=model_prob, market_prob=market_prob, edge=edge,
            contract_price=market_prob,
            sizing=_zero_sizing, confidence=conf,
            signals=signals[:], research=research[:],
            game_time=game_time, commence_time=commence_time,
        )

    # Moneyline
    for pick_str, model_p, market_p in [
        (home_raw, p_home_win, market_home_prob),
        (away_raw, p_away_win, market_away_prob),
    ]:
        r = _make_rec(pick_str, "Moneyline", model_p, market_p)
        if r:
            recs.append(r)

    # Draw
    r = _make_rec("Draw", "Draw", p_draw, market_draw_prob)
    if r:
        recs.append(r)

    # Totals
    tot = game.get("total")
    if tot:
        line = tot.get("point") or tot.get("line", 2.5)
        market_over_prob  = tot.get("over_prob", 0.5)
        market_under_prob = tot.get("under_prob", 0.5)
        model_over  = sum(v for (i, j), v in matrix.items() if i + j > line)
        model_under = sum(v for (i, j), v in matrix.items() if i + j < line)
        _mls_total_raw_over  = model_over
        _mls_total_raw_under = model_under
        model_over,  _mls_total_cred_over  = _apply_credibility_cap_dispatched(
            model_over, market_over_prob, _cred_cap("mls", MLS_CRED_CAP, "credibility_total"), "mls", "credibility_total"
        )
        model_under, _mls_total_cred_under = _apply_credibility_cap_dispatched(
            model_under, market_under_prob, _cred_cap("mls", MLS_CRED_CAP, "credibility_total"), "mls", "credibility_total"
        )
        for pick_str, model_p, market_p, raw_p, cap_fired in [
            (f"Over {line}",  model_over,  market_over_prob,  _mls_total_raw_over,  _mls_total_cred_over),
            (f"Under {line}", model_under, market_under_prob, _mls_total_raw_under, _mls_total_cred_under),
        ]:
            r = _make_rec(pick_str, "Total", model_p, market_p)
            if r:
                r.model_prob_raw = raw_p
                r.credibility_cap_fired = cap_fired
                recs.append(r)

    # Spread (Asian handicap)
    sp = game.get("spread")
    if sp:
        home_point     = sp.get("home_spread") or sp.get("home_point", 0)
        market_home_sp = sp.get("home_prob", 0.5)
        market_away_sp = sp.get("away_prob", 0.5)
        # Adjust integer spread lines to .5 to eliminate push risk.
        # Robinhood only offers .5 lines; integer lines (e.g. +2) can push
        # (e.g. team wins by exactly 2). Shift toward 0 by 0.5 to match.
        if home_point % 1 == 0 and home_point != 0:
            home_point = (home_point - 0.5) if home_point > 0 else (home_point + 0.5)
        model_home_sp = sum(
            v for (i, j), v in matrix.items() if (i - j) > -home_point
        )
        model_away_sp = 1.0 - model_home_sp
        _mls_sp_raw_home = model_home_sp
        _mls_sp_raw_away = model_away_sp
        model_home_sp, _mls_sp_cred_home = _apply_credibility_cap_dispatched(
            model_home_sp, market_home_sp, _cred_cap("mls", MLS_CRED_CAP, "credibility_spread"), "mls", "credibility_spread"
        )
        model_away_sp, _mls_sp_cred_away = _apply_credibility_cap_dispatched(
            model_away_sp, market_away_sp, _cred_cap("mls", MLS_CRED_CAP, "credibility_spread"), "mls", "credibility_spread"
        )
        away_point = -home_point
        for pick_str, model_p, market_p, raw_p, cap_fired in [
            (f"{home_raw} {home_point:+.1f}", model_home_sp, market_home_sp, _mls_sp_raw_home, _mls_sp_cred_home),
            (f"{away_raw} {away_point:+.1f}", model_away_sp, market_away_sp, _mls_sp_raw_away, _mls_sp_cred_away),
        ]:
            r = _make_rec(pick_str, "Spread", model_p, market_p)
            if r:
                r.model_prob_raw = raw_p
                r.credibility_cap_fired = cap_fired
                recs.append(r)

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home_raw, away_team=away_raw,
        home_raw=_lv.get("_mls_raw_home"),
        away_raw=_lv.get("_mls_raw_away"),
        cred_fired=_lv.get("_mls_cred_fired", False),
        hardcap_fired=False,  # MLS uses Poisson — no hard prob cap currently
        stats_avail=stats_available,
    )
    _stamp_decision(game, _min, {
        "lam_home": _lv.get("lam_home"), "lam_away": _lv.get("lam_away"),
        "home_xgd": _lv.get("home_xgd"), "away_xgd": _lv.get("away_xgd"), "xgd_diff": _lv.get("xgd_diff"),
        "attack_home": _lv.get("attack_home"), "attack_away": _lv.get("attack_away"),
        "defense_home": _lv.get("defense_home"), "defense_away": _lv.get("defense_away"),
        "home_inj_pen": _lv.get("home_inj_pen"), "away_inj_pen": _lv.get("away_inj_pen"),
        "stats_available": stats_available,
    }, [
        ("Moneyline", home_raw, _lv.get("p_home_win"), _lv.get("_mls_raw_home"), _lv.get("market_home_prob"), None),
        ("Moneyline", away_raw, _lv.get("p_away_win"), _lv.get("_mls_raw_away"), _lv.get("market_away_prob"), None),
        ("Draw", "draw", _lv.get("p_draw"), None, _lv.get("market_draw_prob"), None),
        ("Total", "over", _lv.get("model_over"), _lv.get("_mls_total_raw_over"), _lv.get("market_over_prob"),
         _lv.get("line") if _lv.get("line") is not None else _lv.get("market_line")),
        ("Total", "under",
         _lv.get("model_under") if _lv.get("model_under") is not None
         else ((1.0 - _lv["model_over"]) if _lv.get("model_over") is not None else None),
         _lv.get("_mls_total_raw_under"),
         _lv.get("market_under_prob"),
         _lv.get("line") if _lv.get("line") is not None else _lv.get("market_line")),
        ("Spread", home_raw, _lv.get("model_home_sp"), _lv.get("_mls_sp_raw_home"), _lv.get("market_home_sp"),
         _lv.get("home_disp_line") if _lv.get("home_disp_line") is not None else _lv.get("home_point")),
        ("Spread", away_raw, _lv.get("model_away_sp"), _lv.get("_mls_sp_raw_away"), _lv.get("market_away_sp"), _lv.get("away_disp_line")),
    ], recs)
    return recs


def analyze_wc_game(
    game: Dict,
    wc_ctx: Dict,
    wc_injuries: Dict,
    min_edge: float = None,
) -> List[BetRecommendation]:
    """
    Elo-driven edge finder for the FIFA World Cup (watchlist, no budget).

    International teams have little usable group-stage form, so strength comes
    from Elo ratings (seed + self-updated from results) rather than club xG.
    Elo supremacy → Poisson goal expectations → Dixon-Coles scoreline grid →
    Moneyline / Draw / Total / Spread probabilities, compared to the market.
    """
    from src.data.wc_stats import normalize as wc_normalize, _canon as _wc_canon
    from src.config import (
        WC_ELO_DEFAULT, WC_BASE_TOTAL, WC_ELO_PER_GOAL, WC_MAX_SUPREMACY,
        WC_DC_RHO, WC_HOST_NATIONS, WC_HOST_ELO_BONUS, WC_CRED_CAP,
        WC_REST_ELO_PER_DAY, WC_MAX_REST_ELO, WC_DEAD_RUBBER_MIN_PTS,
        WC_DEAD_RUBBER_ELO_DAMP, WC_GROUP_STAGE_END, WC_LOWCONF_CAP_FACTOR,
        WC_ALT_HIGH_M, WC_ALT_ACCLIM_ELO, WC_ALTITUDE_NATIONS, WC_HOT_TOTAL_MULT,
    )

    _zero_sizing = BetSizing(
        dollar_allocation=0, num_contracts=0, contract_price=0,
        total_cost=0, profit_if_win=0, loss_if_lose=0,
        expected_value=0, kelly_fraction=0,
    )

    home_raw = game["home_team"]
    away_raw = game["away_team"]
    label    = f"{away_raw} @ {home_raw}"
    commence_time = game.get("commence_time", "")
    game_time = _utc_to_pdt(commence_time)
    _min = min_edge if min_edge is not None else MIN_EDGE
    recs = []

    ml = game.get("moneyline")
    if not ml or "draw_prob" not in ml:
        return recs  # soccer moneyline must have 3-way probs

    market_home_prob = ml["home_prob"]
    market_draw_prob = ml["draw_prob"]
    market_away_prob = ml["away_prob"]

    elo_map = wc_ctx.get("elo", {}) or {}
    home_key = wc_normalize(home_raw, elo_map)
    away_key = wc_normalize(away_raw, elo_map)
    r_home = elo_map.get(home_key, WC_ELO_DEFAULT)
    r_away = elo_map.get(away_key, WC_ELO_DEFAULT)
    stats_available = bool(elo_map) and (home_key in elo_map or away_key in elo_map)

    signals  = []
    research = []

    # Host-nation advantage: World Cup venues are neutral, so a real home edge is
    # granted only when the designated home side is one of the three hosts.
    host_bonus = 0.0
    if any(h.lower() in home_raw.lower() or home_raw.lower() in h.lower() for h in WC_HOST_NATIONS):
        host_bonus = WC_HOST_ELO_BONUS
        signals.append(f"Host nation advantage: {home_raw} (+{WC_HOST_ELO_BONUS:.0f} Elo)")

    # Match date (for rest + group-stage gating), parsed from commence_time.
    try:
        from datetime import date as _date_cls
        _match_date = _date_cls.fromisoformat(commence_time[:10]) if commence_time else None
    except Exception:
        _match_date = None

    # ── Rest / fatigue differential ──────────────────────────────────────────
    # The better-rested side gets a small Elo nudge (uneven rest in a
    # geographically huge tournament). No signal for first matches (no prior game).
    rest_elo = 0.0
    last_played = wc_ctx.get("last_played", {}) or {}
    if _match_date and last_played:
        hp = last_played.get(home_key)
        ap = last_played.get(away_key)
        if hp and ap:
            rest_diff = (_match_date - hp).days - (_match_date - ap).days
            rest_elo = max(-WC_MAX_REST_ELO, min(WC_MAX_REST_ELO, rest_diff * WC_REST_ELO_PER_DAY))
            if abs(rest_diff) >= 1:
                better = home_raw if rest_diff > 0 else away_raw
                signals.append(f"Rest edge: {better} (+{abs(rest_diff)}d rest → {abs(rest_elo):.0f} Elo)")

    # ── Dead-rubber damping (group stage only) ───────────────────────────────
    # 2026's best-third-placed rule makes elimination ambiguous after 2 games, so
    # we damp ONLY the clearly-safe side: ≥6 pts heading into its 3rd group game
    # (likely qualified → may rotate). Asymmetric case only (one safe, one fighting).
    dr_home = dr_away = 0.0
    standings = wc_ctx.get("standings", {}) or {}
    _in_group_stage = False
    try:
        _in_group_stage = bool(_match_date) and _match_date <= _date_cls.fromisoformat(WC_GROUP_STAGE_END)
    except Exception:
        _in_group_stage = False
    if _in_group_stage and standings:
        sh = standings.get(home_key)
        sa = standings.get(away_key)
        home_q = bool(sh and sh.get("gp", 0) >= 2 and sh.get("pts", 0) >= WC_DEAD_RUBBER_MIN_PTS)
        away_q = bool(sa and sa.get("gp", 0) >= 2 and sa.get("pts", 0) >= WC_DEAD_RUBBER_MIN_PTS)
        if home_q and not away_q:
            dr_home = WC_DEAD_RUBBER_ELO_DAMP
            signals.append(f"Dead rubber: {home_raw} likely qualified — rotation risk (-{dr_home:.0f} Elo)")
        elif away_q and not home_q:
            dr_away = WC_DEAD_RUBBER_ELO_DAMP
            signals.append(f"Dead rubber: {away_raw} likely qualified — rotation risk (-{dr_away:.0f} Elo)")

    # ── Venue: altitude acclimatisation (strength edge) + heat (totals) ──────
    # Altitude is NOT a totals effect (Azteca data shows no goals relation; ball-
    # physics and fatigue cancel). The real effect is an advantage to the
    # ACCLIMATISED side vs a lowland opponent (McSharry, BMJ 2007). Heat modestly
    # suppresses totals.
    base_total = WC_BASE_TOTAL
    alt_home = alt_away = 0.0
    fixtures = wc_ctx.get("fixtures", {}) or {}
    venue_info = fixtures.get((_wc_canon(home_raw), _wc_canon(away_raw)))
    if venue_info:
        if venue_info.get("altitude_m", 0) >= WC_ALT_HIGH_M:
            def _is_native(team):
                return any(n.lower() in team.lower() or team.lower() in n.lower() for n in WC_ALTITUDE_NATIONS)
            home_native, away_native = _is_native(home_raw), _is_native(away_raw)
            if home_native and not away_native:
                alt_home = WC_ALT_ACCLIM_ELO
                signals.append(f"Altitude edge: {home_raw} acclimatised at {venue_info.get('name')} ({venue_info.get('altitude_m')}m) vs lowland {away_raw} (+{alt_home:.0f} Elo)")
            elif away_native and not home_native:
                alt_away = WC_ALT_ACCLIM_ELO
                signals.append(f"Altitude edge: {away_raw} acclimatised at {venue_info.get('name')} ({venue_info.get('altitude_m')}m) vs lowland {home_raw} (+{alt_away:.0f} Elo)")
        elif venue_info.get("climate") == "hot":
            base_total *= WC_HOT_TOTAL_MULT
            signals.append(f"Hot venue ({venue_info.get('name')}): lower totals")

    # ── Effective Elo difference (host + rest + dead-rubber + altitude) ───────
    elo_diff = (r_home + host_bonus - dr_home + alt_home) - (r_away - dr_away + alt_away) + rest_elo
    # Expected goal supremacy from Elo difference, clamped.
    supremacy = max(-WC_MAX_SUPREMACY, min(WC_MAX_SUPREMACY, elo_diff / WC_ELO_PER_GOAL))

    lam_home = max(0.2, (base_total + supremacy) / 2.0)
    lam_away = max(0.2, (base_total - supremacy) / 2.0)

    # ── Low-confidence shrinkage ─────────────────────────────────────────────
    # If a side's Elo is a guess (unseeded → fell back to default), tighten the
    # credibility cap so the model is pulled harder toward the market.
    home_known = home_key in elo_map
    away_known = away_key in elo_map
    lowconf_factor = WC_LOWCONF_CAP_FACTOR if not (home_known and away_known) else 1.0
    wc_cap = WC_CRED_CAP * lowconf_factor
    if lowconf_factor < 1.0:
        research.append("Low-confidence Elo (unseeded side) — model pulled toward market")

    research.append(f"Elo: {home_raw} {r_home:.0f} vs {away_raw} {r_away:.0f} ({elo_diff:+.0f})")
    if abs(elo_diff) >= 60:
        stronger = home_raw if elo_diff > 0 else away_raw
        signals.append(f"Elo edge: {stronger} ({abs(elo_diff):.0f} pts → {abs(supremacy):.2f} goal supremacy)")

    # Probability matrix (Dixon-Coles low-score correction)
    matrix = _mls_prob_matrix(lam_home, lam_away, rho=WC_DC_RHO)
    p_home_win = sum(v for (i, j), v in matrix.items() if i > j)
    p_draw     = sum(v for (i, j), v in matrix.items() if i == j)
    p_away_win = sum(v for (i, j), v in matrix.items() if i < j)

    total_p = p_home_win + p_draw + p_away_win
    if total_p > 0:
        p_home_win /= total_p
        p_draw     /= total_p
        p_away_win /= total_p

    research.append(f"Goal projection: {home_raw} {lam_home:.2f} – {away_raw} {lam_away:.2f}")
    research.append(f"Win probs: H {p_home_win:.1%} | D {p_draw:.1%} | A {p_away_win:.1%}")
    # "Model projected score:" prefix is load-bearing — the PWA card template
    # renders any context line with this prefix as the teal headline (same as
    # every other sport) instead of falling back to a bare "Details" toggle.
    signals.append(f"Model projected score: {home_raw} {lam_home:.1f} — {away_raw} {lam_away:.1f}")

    # Calibration capture: raw 3-way probabilities (pre-cred-cap).
    _wc_raw_home = p_home_win
    _wc_raw_away = p_away_win

    # ── General credibility cap (cold-start safety; auto-relaxed by calibration) ─
    p_home_win, _wc_cred_home_fired = _apply_credibility_cap_dispatched(
        p_home_win, market_home_prob, _cred_cap("wc", wc_cap, "credibility_moneyline"), "wc", "credibility_moneyline"
    )
    p_draw, _wc_cred_draw_fired = _apply_credibility_cap_dispatched(
        p_draw, market_draw_prob, _cred_cap("wc", wc_cap, "credibility_draw"), "wc", "credibility_draw"
    )
    p_away_win = 1.0 - p_home_win - p_draw
    p_away_win, _wc_cred_away_fired = _apply_credibility_cap_dispatched(
        p_away_win, market_away_prob, _cred_cap("wc", wc_cap, "credibility_moneyline"), "wc", "credibility_moneyline"
    )
    _wc_cred_fired = _wc_cred_home_fired or _wc_cred_draw_fired or _wc_cred_away_fired

    # Build BetRecommendations
    def _make_rec(pick_str, bet_type, model_prob, market_prob):
        edge = model_prob - market_prob
        if edge < _min:
            return None
        conf = _confidence_label(edge, len(signals), stats_available)
        return BetRecommendation(
            sport="WC", game=label,
            home_team=home_raw, away_team=away_raw,
            pick=pick_str, bet_type=bet_type,
            model_prob=model_prob, market_prob=market_prob, edge=edge,
            contract_price=market_prob,
            sizing=_zero_sizing, confidence=conf,
            signals=signals[:], research=research[:],
            game_time=game_time, commence_time=commence_time,
        )

    # Moneyline
    for pick_str, model_p, market_p in [
        (home_raw, p_home_win, market_home_prob),
        (away_raw, p_away_win, market_away_prob),
    ]:
        r = _make_rec(pick_str, "Moneyline", model_p, market_p)
        if r:
            recs.append(r)

    # Draw
    r = _make_rec("Draw", "Draw", p_draw, market_draw_prob)
    if r:
        recs.append(r)

    # Totals
    # Robinhood only offers half-integer lines, and goals are integers, so each
    # side is expressed at the exact half-line its probability corresponds to.
    # Quarter and whole book lines map exactly: P(sum > 3.0) = P(sum ≥ 4) = the
    # "Over 3.5" probability; P(sum > 2.25) = P(sum ≥ 3) = "Over 2.5". Market
    # prices stay from the book's line (same convention as the NBA ".5 label
    # shift"); for whole lines the book price includes push protection, which
    # overstates the market prob and understates our edge — conservative.
    # If the Odds API does not return a totals market for this game, we still
    # compute Over/Under 2.5 from the matrix and surface it in research so the
    # user always sees the model's goals projection.
    tot = game.get("total")
    _model_over_25  = sum(v for (i, j), v in matrix.items() if i + j >= 3)
    _model_under_25 = sum(v for (i, j), v in matrix.items() if i + j <= 2)
    if tot:
        line = tot.get("point") or tot.get("line", 2.5)
        market_over_prob  = tot.get("over_prob", 0.5)
        market_under_prob = tot.get("under_prob", 0.5)
        k_over  = math.floor(line) + 1          # min total goals for Over to win
        k_under = math.ceil(line) - 1           # max total goals for Under to win
        over_disp_line  = k_over - 0.5
        under_disp_line = k_under + 0.5
        model_over  = sum(v for (i, j), v in matrix.items() if i + j >= k_over)
        model_under = sum(v for (i, j), v in matrix.items() if i + j <= k_under)
        _wc_total_raw_over  = model_over
        _wc_total_raw_under = model_under
        model_over,  _wc_total_cred_over  = _apply_credibility_cap_dispatched(
            model_over, market_over_prob, _cred_cap("wc", wc_cap, "credibility_total"), "wc", "credibility_total"
        )
        model_under, _wc_total_cred_under = _apply_credibility_cap_dispatched(
            model_under, market_under_prob, _cred_cap("wc", wc_cap, "credibility_total"), "wc", "credibility_total"
        )
        for pick_str, model_p, market_p, raw_p, cap_fired in [
            (f"Over {over_disp_line}",  model_over,  market_over_prob,  _wc_total_raw_over,  _wc_total_cred_over),
            (f"Under {under_disp_line}", model_under, market_under_prob, _wc_total_raw_under, _wc_total_cred_under),
        ]:
            r = _make_rec(pick_str, "Total", model_p, market_p)
            if r:
                r.model_prob_raw = raw_p
                r.credibility_cap_fired = cap_fired
                recs.append(r)
        if abs(line - 2.5) > 0.01:
            # Book line differs from 2.5 — also note the 2.5 model probability
            research.append(f"Goals O/U 2.5 (model): Over {_model_over_25:.1%} | Under {_model_under_25:.1%}")
    else:
        # No market totals line available — show model projection only
        research.append(f"Goals O/U 2.5 (model, no market line): Over {_model_over_25:.1%} | Under {_model_under_25:.1%}")

    # Spread (Asian handicap)
    # +0.5 ("don't lose" = win or draw) is a distinct bet from ML and is valid
    # on Robinhood. -0.5 ("must win") is identical to ML — skip it. All lines
    # ≥ ±1.5 are meaningful and always emitted. The ±1.5 model probability from
    # the matrix is always added to research regardless of whether a pick fires.
    sp = game.get("spread")
    # Always compute model ±1.5 probabilities for research.
    _model_home_15 = sum(v for (i, j), v in matrix.items() if (i - j) >= 2)   # home wins by 2+
    _model_away_15 = sum(v for (i, j), v in matrix.items() if (i - j) <= -2)  # away wins by 2+
    research.append(f"±1.5 spread (model): {home_raw} {_model_home_15:.1%} | {away_raw} {_model_away_15:.1%}")
    if sp:
        home_point     = sp.get("home_spread") or sp.get("home_point", 0)
        market_home_sp = sp.get("home_prob", 0.5)
        market_away_sp = sp.get("away_prob", 0.5)
        tau    = -float(home_point)             # home margin required by the book line
        k_home = math.floor(tau) + 1            # min home margin for home cover
        k_away = math.ceil(tau) - 1             # max home margin for away cover
        home_disp_line = -(k_home - 0.5)
        away_disp_line = k_away + 0.5
        model_home_sp = sum(v for (i, j), v in matrix.items() if (i - j) >= k_home)
        model_away_sp = sum(v for (i, j), v in matrix.items() if (i - j) <= k_away)
        _wc_sp_raw_home = model_home_sp
        _wc_sp_raw_away = model_away_sp
        model_home_sp, _wc_sp_cred_home = _apply_credibility_cap_dispatched(
            model_home_sp, market_home_sp, _cred_cap("wc", wc_cap, "credibility_spread"), "wc", "credibility_spread"
        )
        model_away_sp, _wc_sp_cred_away = _apply_credibility_cap_dispatched(
            model_away_sp, market_away_sp, _cred_cap("wc", wc_cap, "credibility_spread"), "wc", "credibility_spread"
        )
        for pick_str, model_p, market_p, disp_line, raw_p, cap_fired in [
            (f"{home_raw} {home_disp_line:+.1f}", model_home_sp, market_home_sp, home_disp_line, _wc_sp_raw_home, _wc_sp_cred_home),
            (f"{away_raw} {away_disp_line:+.1f}", model_away_sp, market_away_sp, away_disp_line, _wc_sp_raw_away, _wc_sp_cred_away),
        ]:
            if disp_line == -0.5:
                continue  # "must win" — identical to ML, skip
            r = _make_rec(pick_str, "Spread", model_p, market_p)
            if r:
                r.model_prob_raw = raw_p
                r.credibility_cap_fired = cap_fired
                recs.append(r)

    # Stamp per-game calibration metadata onto every rec.
    _lv = locals()
    _stamp_recs_calibration(
        recs,
        game=game,
        home_team=home_raw, away_team=away_raw,
        home_raw=_lv.get("_wc_raw_home"),
        away_raw=_lv.get("_wc_raw_away"),
        cred_fired=_lv.get("_wc_cred_fired", False),
        hardcap_fired=False,  # WC uses Poisson — no hard prob cap
        stats_avail=stats_available,
    )
    _stamp_decision(game, _min, {
        "lam_home": _lv.get("lam_home"), "lam_away": _lv.get("lam_away"),
        "elo_diff": _lv.get("elo_diff"), "supremacy": _lv.get("supremacy"),
        "r_home": _lv.get("r_home"), "r_away": _lv.get("r_away"),
        "host_bonus": _lv.get("host_bonus"), "rest_elo": _lv.get("rest_elo"),
        "alt_home": _lv.get("alt_home"), "alt_away": _lv.get("alt_away"),
        "dead_rubber_home": _lv.get("dr_home"), "dead_rubber_away": _lv.get("dr_away"),
        "base_total": _lv.get("base_total"), "stats_available": stats_available,
    }, [
        ("Moneyline", home_raw, _lv.get("p_home_win"), _lv.get("_wc_raw_home"), _lv.get("market_home_prob"), None),
        ("Moneyline", away_raw, _lv.get("p_away_win"), _lv.get("_wc_raw_away"), _lv.get("market_away_prob"), None),
        ("Draw", "draw", _lv.get("p_draw"), None, _lv.get("market_draw_prob"), None),
        ("Total", "over", _lv.get("model_over"), _lv.get("_wc_total_raw_over"), _lv.get("market_over_prob"),
         _lv.get("line") if _lv.get("line") is not None else _lv.get("market_line")),
        ("Total", "under",
         _lv.get("model_under") if _lv.get("model_under") is not None
         else ((1.0 - _lv["model_over"]) if _lv.get("model_over") is not None else None),
         _lv.get("_wc_total_raw_under"),
         _lv.get("market_under_prob"),
         _lv.get("line") if _lv.get("line") is not None else _lv.get("market_line")),
        ("Spread", home_raw, _lv.get("model_home_sp"), _lv.get("_wc_sp_raw_home"), _lv.get("market_home_sp"),
         _lv.get("home_disp_line") if _lv.get("home_disp_line") is not None else _lv.get("home_point")),
        ("Spread", away_raw, _lv.get("model_away_sp"), _lv.get("_wc_sp_raw_away"), _lv.get("market_away_sp"), _lv.get("away_disp_line")),
    ], recs)
    return recs
