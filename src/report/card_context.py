"""
Card context builder.

Takes raw signals + research from edge_finder and produces:
  • narrative    — 2-3 sentence plain-English explanation of why the pick was made
  • context      — single merged, deduplicated list replacing the two separate sections

The confidence label is computed upstream (edge_finder → _confidence_label) before
this runs, so these transformations never affect pick selection or sizing.
"""
import re
from typing import Dict, List, Tuple, Optional


# ── Drop patterns ─────────────────────────────────────────────────────────────
#
# Signals that are superseded by fuller research lines:
#   - Pitcher stat signal: "Name FIP X / xFIP X | K/9: X"
#     → research has ERA + FIP + xFIP + K/9 + BB/9 + IP
#   - Park factor signal: "Park factor 0.97 (pitcher-friendly: Venue)"
#     → research has "Venue: Name — park factor X (type)"
#   - Wind/temp signals: "🌬️ Cross wind..." / "🌡️ Cool..."
#     → research has full weather line (temp + wind + precip)
#   - Umpire signal lines (two lines per ump in signals)
#     → research has single complete umpire line
#
# NBA/NHL:
#   - "Rating edge: Team (blended NetRtg diff +X.X)"
#     → research has OffRtg / DefRtg / NetRtg per team (more complete)

_SIGNAL_DROP = [
    # MLB: pitcher stat lines contain "FIP X.X / xFIP" regardless of name format
    re.compile(r"\bFIP [\d.]+ / xFIP"),
    # MLB: park factor signal
    re.compile(r"^Park factor [\d.]"),
    # MLB: cross-wind signal (🌬️) — research weather line is fuller
    re.compile(r"^🌬️"),
    # MLB: temperature signal (🌡️) — research weather line is fuller
    re.compile(r"^🌡️"),
    # MLB: umpire signal lines — research has the complete single-line summary
    re.compile(r"^👨‍⚖️ Umpire "),
    re.compile(r"^👨‍⚖️ \w[\w\s.\-]*: (?:high|low|tight|near|one of)"),
    # NBA/NHL: rating edge (research has fuller OffRtg/DefRtg/NetRtg breakdown)
    re.compile(r"^Rating edge:"),
]

# Research lines to suppress (internal model details or cleaner version in signals):
#   - "Model expected runs: X | Y" → signal "Model projected score" is cleaner
#   - "tanh cap active: ..."        → internal model implementation detail
#   - "Recent form weight: ..."     → internal weighting detail
#   - "Playoff adjustment: ..."     → internal note

_RESEARCH_DROP = [
    re.compile(r"^Model expected runs:"),
    re.compile(r"^tanh cap active:"),
    re.compile(r"^Recent form weight:"),
    re.compile(r"^Playoff adjustment:"),
]


def _filter_signals(signals: List[str]) -> List[str]:
    return [s for s in signals if not any(p.search(s) for p in _SIGNAL_DROP)]


def _filter_research(research: List[str]) -> List[str]:
    return [r for r in research if not any(p.search(r) for p in _RESEARCH_DROP)]


def merge_context(signals: List[str], research: List[str]) -> List[str]:
    """Merge signals + research into a single deduplicated display list."""
    return _filter_signals(signals) + _filter_research(research)


# ── Context sort ──────────────────────────────────────────────────────────────
#
# Priority lists: first matching pattern wins → lower index = shown higher up.
# Items that match no pattern are appended at the end, preserving their
# relative insertion order (signals before research within that tail group).

_CONTEXT_PRIORITY: Dict[str, List[re.Pattern]] = {
    # ── Sports-page order ──────────────────────────────────────────────────────
    # 1. Projected score / expected total  (headline number)
    # 2. Team season records / stat lines  (standings context)
    # 3. Game-specific context             (pitchers, weather, venue)
    # 4. Schedule / rest                   (situational)
    # 5. Injuries                          (roster health)
    # 6. Edge signals                      (why we differ from market)
    "MLB": [
        re.compile(r"^Model projected score"),              # 0  projected score
        re.compile(r"^Model expected total"),               # 1  expected total
        re.compile(r"(?i)\boffense:|\bbatting:"),           # 2  team batting / records
        re.compile(r"^Platoon:"),                           # 3  platoon split vs opp hand
        re.compile(r"(?i)\bBullpen"),                       # 4  team bullpen stats
        re.compile(r"^(?:🔵|🔴)"),                         # 4  pitcher matchup
        re.compile(r"^Venue:"),                             # 5  venue / park factor
        re.compile(r"^(?:🌤|Weather)"),                    # 6  weather (emoji prefix)
        re.compile(r"^👨‍⚖️"),                         # 7  umpire
        re.compile(r"(?i)schedule|back-to-back|\brest\b"), # 8  schedule / rest
        re.compile(r"(?i)injur|⚕"),                        # 9  injuries (% impact + roster)
        re.compile(r"⚠ ERA trap"),                         # 10 ERA trap warnings
        re.compile(r"⚠ K matchup"),                        # 11 K matchup warning
    ],
    "NBA": [
        re.compile(r"^Model projected score"),              # projected score
        re.compile(r"^Model expected total"),               # expected total
        re.compile(r"OffRtg|DefRtg"),                       # season records + ratings
        re.compile(r"recent|last \d+ days"),                # recent form ("last 14 days: …")
        re.compile(r"back-to-back"),                        # B2B fatigue
        re.compile(r"[Rr]est"),                             # rest days
        re.compile(r"schedule"),                            # schedule load
        re.compile(r"⚕"),                                   # injured player details
        re.compile(r"injury impact"),                       # injury impact signal
        re.compile(r"[Hh]ome court"),                       # home advantage
    ],
    "NHL": [
        re.compile(r"^Model projected score"),              # projected score
        re.compile(r"GPG|GAPG"),                            # season records + ratings
        re.compile(r"recent|last \d+ days"),                # recent form ("last 14 days: …")
        re.compile(r"back-to-back"),                        # B2B fatigue
        re.compile(r"[Rr]est"),                             # rest days
        re.compile(r"injur|⚕"),                            # injuries
        re.compile(r"[Hh]ome ice"),                         # home advantage
    ],
    "NFL": [
        re.compile(r"^Model projected score"),              # projected score
        re.compile(r"PPG|OPP PPG|NetRtg"),                 # season records + ratings
        re.compile(r"recent"),                              # recent form
        re.compile(r"[Bb]ye week"),                         # bye week rest
        re.compile(r"[Rr]est"),                             # rest days
        re.compile(r"injur"),                               # injuries
        re.compile(r"[Hh]ome field"),                       # home advantage
    ],
    "WNBA": [
        re.compile(r"^Model projected score"),              # projected score
        re.compile(r"NetRtg|FG "),                          # season records + ratings (recent line has neither)
        re.compile(r"^Strength of schedule"),               # SOS adjustment
        re.compile(r"recent|last \d+ games"),               # recent form ("last N games: …")
        re.compile(r"back-to-back"),                        # B2B fatigue
        re.compile(r"[Rr]est"),                             # rest days
        re.compile(r"⚕"),                                   # injured player details
        re.compile(r"lineup impact|injuries benefit"),      # injury impact signals
        re.compile(r"[Hh]ome court"),                       # home advantage
    ],
    "IPL": [
        re.compile(r"^Model projected score"),              # projected score
        re.compile(r"\d+W-\d+L"),                          # team season records
        re.compile(r"^Venue:"),                             # venue / pitch info
        re.compile(r"[Pp]itch|[Dd]ew"),                    # pitch / dew conditions
        re.compile(r"[Rr]est|days since"),                  # rest days
        re.compile(r"Form edge"),                           # form edge signal
        re.compile(r"[Hh]ome venue"),                       # home venue advantage
        re.compile(r"H2H"),                                 # head-to-head
    ],
    "MLS": [
        re.compile(r"xG projection"),                       # xG projection (headline)
        re.compile(r"xGF"),                                 # team season stats
        re.compile(r"xG edge"),                             # xG edge
        re.compile(r"recent form"),                         # recent form
        re.compile(r"[Rr]est"),                             # rest days
        re.compile(r"[Hh]ome.*[Vv]enue|[Ff]ortress"),     # home venue
        re.compile(r"injur|⚕|🚫"),                          # injuries
    ],
}


def _sort_context(sport: str, items: List[str]) -> List[str]:
    """
    Sort context items by sport-specific importance order.
    Items matching an earlier pattern in _CONTEXT_PRIORITY appear first.
    Items that match no pattern are appended last (preserving insertion order).
    """
    patterns = _CONTEXT_PRIORITY.get(sport, [])
    if not patterns:
        return items

    n = len(patterns)

    def _priority(item: str) -> int:
        for i, p in enumerate(patterns):
            if p.search(item):
                return i
        return n  # unmatched → tail

    # stable sort: items at the same priority keep their original order
    return sorted(items, key=_priority)


# ── Narrative helpers ─────────────────────────────────────────────────────────

def _search_first(pattern: str, items: List[str]) -> Optional[re.Match]:
    """Return the first regex match found across a list of strings."""
    for item in items:
        m = re.search(pattern, item)
        if m:
            return m
    return None


def _edge_word(edge: float) -> str:
    if edge >= 0.12:
        return "strong"
    if edge >= 0.07:
        return "solid"
    if edge >= 0.05:
        return "modest"
    return "slight"


# ── MLB narrative ─────────────────────────────────────────────────────────────

def _mlb_narrative(pick: str, bet_type: str, signals: List[str], research: List[str],
                   edge: float) -> str:
    # ── Parse signals ─────────────────────────────────────────────────────────
    era_traps = []
    for s in signals:
        m = re.search(
            r"ERA trap \[(\w+)\] — (.+?): ERA ([\d.]+) vs xFIP ([\d.]+) over (\d+) IP"
            r".*?— (.+?) ML may be overpriced",
            s,
        )
        if m:
            era_traps.append(m)  # groups: severity, pitcher, era, xfip, ip, trap_team

    injuries = []
    for s in signals:
        m = re.search(r"^(.+?) injury impact \(([-\d.]+)%\)", s)
        if m:
            injuries.append((m.group(1), float(m.group(2))))

    # Platoon splits: "Platoon: TEAM vs RHP — .806 OPS (695 PA), season .735 → model .772"
    platoons = []
    for s in signals:
        m = re.search(
            r"Platoon: (.+?) vs (\w+) — ([\d.]+) OPS \((\d+) PA\), "
            r"season ([\d.]+) → model ([\d.]+)",
            s,
        )
        if m:
            platoons.append(m)  # groups: team, hand, split_ops, pa, season, blended

    score_m = _search_first(
        r"Model projected score: (.+?) ([\d.]+) — (.+?) ([\d.]+)", signals
    )
    total_m = _search_first(
        r"Model expected total: ([\d.]+) vs (?:market )?line ([\d.]+)", signals
    )

    # ── Parse research ────────────────────────────────────────────────────────
    home_p = _search_first(
        r"🔵 (.+?): ERA [\d.]+ \| FIP [\d.]+ / xFIP ([\d.]+) \| K/9 ([\d.]+)", research
    )
    away_p = _search_first(
        r"🔴 (.+?): ERA [\d.]+ \| FIP [\d.]+ / xFIP ([\d.]+) \| K/9 ([\d.]+)", research
    )

    # ── Build narrative ───────────────────────────────────────────────────────
    parts = []
    ew = _edge_word(edge)

    severe_traps = [t for t in era_traps if t.group(1) in ("SEVERE", "MODERATE")]
    big_injuries = [(team, pct) for team, pct in injuries if abs(pct) >= 2.5]

    # The team we're actually backing (strip any spread suffix like " +1.5").
    picked_team = re.sub(r"\s[+-]?\d+(?:\.\d+)?$", "", pick).strip()

    def _same_team(a: str, b: str) -> bool:
        a, b = (a or "").lower(), (b or "").lower()
        return bool(a) and bool(b) and (a in b or b in a)

    # A driver must support the pick's DIRECTION, or the narrative contradicts itself
    # ("edge against <the very team we picked>"). Two guards:
    #  • An ERA trap supports the pick only when it's on the OPPONENT's pitcher
    #    (their team is overpriced → we fade them). A trap on our own pitcher
    #    argues against the pick, so it can't be the stated driver.
    supporting_traps = [t for t in severe_traps if not _same_team(picked_team, t.group(6))]
    #  • A pitching xFIP mismatch supports the pick only when the BETTER pitcher is
    #    on the team we're backing; otherwise it favours the opponent.
    mismatch_better = None
    if home_p and away_p and abs(float(home_p.group(2)) - float(away_p.group(2))) >= 0.6:
        _hx, _ax = float(home_p.group(2)), float(away_p.group(2))
        _better = home_p.group(1) if _hx < _ax else away_p.group(1)
        _worse  = away_p.group(1) if _hx < _ax else home_p.group(1)
        if _same_team(picked_team, _better):
            mismatch_better = (_better, min(_hx, _ax), _worse, max(_hx, _ax))

    if bet_type == "Total":
        if total_m:
            proj = float(total_m.group(1))
            line = float(total_m.group(2))
            diff = proj - line
            direction = "over" if diff > 0 else "under"
            parts.append(
                f"The model projects {proj:.1f} combined runs vs the market line of {line}, "
                f"a {abs(diff):.1f}-run lean toward the {direction}."
            )
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            parts.append(f"Projected: {t1} {r1} — {t2} {r2}.")
        if severe_traps:
            t = severe_traps[0]
            parts.append(
                f"Note: {t.group(2)}'s {t.group(3)} ERA vs {t.group(4)} xFIP "
                f"over {t.group(5)} IP is an ERA trap signal that may affect run distribution."
            )

    elif supporting_traps:
        t = supporting_traps[0]
        sev = "significant" if t.group(1) == "SEVERE" else "moderate"
        parts.append(
            f"The model finds a {ew} edge against {t.group(6)}, driven by a {sev} "
            f"ERA trap on {t.group(2)} — {t.group(3)} ERA vs {t.group(4)} xFIP "
            f"in {t.group(5)} innings suggests regression risk the market hasn't fully priced."
        )
        if big_injuries:
            inj = " and ".join(f"{team} is depleted ({abs(pct):.1f}% impact)" for team, pct in big_injuries)
            parts.append(f"{inj}.")
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            leader = t1 if float(r1) > float(r2) else t2
            diff = abs(float(r1) - float(r2))
            parts.append(f"Model projects {leader} by {diff:.1f} runs ({t1} {r1} — {t2} {r2}).")

    elif mismatch_better:
        better, b_xfip, worse, w_xfip = mismatch_better
        parts.append(
            f"A {ew} pitching mismatch drives this pick — {better} ({b_xfip:.2f} xFIP) "
            f"holds a meaningful advantage over {worse} ({w_xfip:.2f} xFIP)."
        )
        if big_injuries:
            inj = " and ".join(f"{team} ({abs(pct):.1f}%)" for team, pct in big_injuries)
            parts.append(f"Injury drag further tilts the matchup: {inj}.")
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            leader = t1 if float(r1) > float(r2) else t2
            diff = abs(float(r1) - float(r2))
            parts.append(f"Model projects {leader} by {diff:.1f} runs ({t1} {r1} — {t2} {r2}).")

    elif big_injuries:
        inj = " and ".join(
            f"{team} ({abs(pct):.1f}% win probability impact)" for team, pct in big_injuries
        )
        parts.append(
            f"The primary driver is injury context — {inj}, "
            f"which the model believes is only partially reflected in current market pricing."
        )
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            leader = t1 if float(r1) > float(r2) else t2
            diff = abs(float(r1) - float(r2))
            parts.append(f"Model projects {leader} by {diff:.1f} runs ({t1} {r1} — {t2} {r2}).")

    else:
        parts.append(
            f"The model builds a {ew} edge from a combination of pitching quality, "
            f"roster health, and situational factors."
        )
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            leader = t1 if float(r1) > float(r2) else t2
            diff = abs(float(r1) - float(r2))
            parts.append(f"Model projects {leader} by {diff:.1f} runs ({t1} {r1} — {t2} {r2}).")

    # Favorable platoon for the team we're backing (ML/Spread only — totals
    # already fold both lineups' splits into the projected total above).
    if bet_type != "Total":
        for m in platoons:
            pteam, hand = m.group(1), m.group(2)
            split_ops, season, blended = float(m.group(3)), float(m.group(5)), float(m.group(6))
            if (pteam in pick or pick.startswith(pteam)) and (blended - season) >= 0.012:
                parts.append(
                    f"{pteam} also profiles well against {hand} this year "
                    f"({split_ops:.3f} OPS vs {season:.3f} overall), which the model "
                    f"folds into its run projection."
                )
                break

    return " ".join(parts)


# ── NBA narrative ─────────────────────────────────────────────────────────────

def _nba_narrative(pick: str, bet_type: str, signals: List[str], research: List[str],
                   edge: float) -> str:
    b2b = [re.search(r"^(.+?) on back-to-back", s) for s in signals]
    b2b = [m.group(1) for m in b2b if m]

    injuries = []
    for s in signals:
        m = re.search(r"^(.+?) injury impact \(([-\d.]+)%\)", s)
        if m:
            injuries.append((m.group(1), float(m.group(2))))

    score_m = _search_first(r"Model projected score: (.+?) (\d+) — (.+?) (\d+)", signals)
    total_m = _search_first(r"Model expected total: ([\d.]+) vs market line ([\d.]+)", signals)

    ew = _edge_word(edge)
    parts = []

    if bet_type == "Total":
        if total_m:
            proj = float(total_m.group(1))
            line = float(total_m.group(2))
            diff = proj - line
            direction = "over" if diff > 0 else "under"
            parts.append(
                f"The model projects {proj:.1f} combined points vs the market line of {line:.1f}, "
                f"a {abs(diff):.1f}-point lean toward the {direction}."
            )
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            parts.append(f"Projected: {t1} {r1} — {t2} {r2}.")
        if b2b:
            parts.append(f"{b2b[0]} is on back-to-back, which may suppress their scoring output.")
    else:
        if b2b:
            parts.append(
                f"Key situational edge: {b2b[0]} is on zero rest (back-to-back), "
                f"giving {pick} a meaningful fatigue advantage tonight."
            )
        big_injuries = [(t, p) for t, p in injuries if abs(p) >= 2.0]
        if big_injuries:
            inj = " and ".join(f"{t} ({abs(p):.1f}%)" for t, p in big_injuries)
            parts.append(f"Injury drag on {inj} further shifts the model's probability.")
        if not b2b and not big_injuries:
            parts.append(
                f"The model finds a {ew} edge based on net rating differential "
                f"and situational adjustments."
            )
        if score_m:
            t1, r1, t2, r2 = score_m.groups()
            pts_diff = abs(int(r1) - int(r2))
            leader = t1 if int(r1) > int(r2) else t2
            parts.append(
                f"Projected {leader} by {pts_diff} points ({t1} {r1} — {t2} {r2})."
            )

    return " ".join(parts)


# ── NHL narrative ─────────────────────────────────────────────────────────────

def _nhl_narrative(pick: str, bet_type: str, signals: List[str], research: List[str],
                   edge: float) -> str:
    b2b = [re.search(r"^(.+?) on back-to-back", s) for s in signals]
    b2b = [m.group(1) for m in b2b if m]

    injuries = []
    for s in signals:
        m = re.search(r"^(.+?) injury(?:es benefit .+? \()?([-\d.]+)%\)", s)
        if m:
            injuries.append((m.group(1), float(m.group(2))))

    ew = _edge_word(edge)
    parts = []

    if b2b:
        parts.append(
            f"{b2b[0]} is on back-to-back (zero rest), creating a {ew} "
            f"situational edge for {pick} tonight."
        )
    big_injuries = [(t, p) for t, p in injuries if abs(p) >= 1.5]
    if big_injuries:
        inj = " and ".join(f"{t} ({abs(p):.1f}%)" for t, p in big_injuries)
        parts.append(f"Injury impact: {inj}.")
    if not parts:
        parts.append(
            f"The model finds a {ew} edge driven by season net rating "
            f"and recent form differential."
        )

    return " ".join(parts)


# ── Props narrative ───────────────────────────────────────────────────────────

def _prop_narrative(prop_type: str, player: str, team: str, opponent: str,
                    signals: List[str], research: List[str],
                    model_line: float, market_line: float, edge: float) -> str:
    ew = _edge_word(edge)
    margin = model_line - market_line

    # Pitcher context from signals
    pitcher_m = _search_first(
        r"vs (.+?) \(FIP ([\d.]+)(?:/xFIP ([\d.]+))? \| WHIP ([\d.]+)\)", signals
    )
    k_m = _search_first(r"K% matchup: batter ([\d.]+)%", signals)

    parts = []

    if prop_type in ("Hits Over", "HRR Over", "Total Bases Over"):
        parts.append(
            f"The model projects {player} at {model_line:.2f} {prop_type.replace(' Over', '')} "
            f"vs the line of {market_line} — a {ew} {margin:+.2f} edge."
        )
        if pitcher_m:
            p_name = pitcher_m.group(1)
            p_whip = pitcher_m.group(4)
            xfip = pitcher_m.group(3) or pitcher_m.group(2)
            parts.append(
                f"Facing {p_name} ({xfip} xFIP, {p_whip} WHIP) — "
                f"{'a hittable matchup' if float(xfip) >= 4.0 else 'respectable contact opportunity'}."
            )
    elif prop_type == "Strikeouts Over":
        parts.append(
            f"The model projects {player} at {model_line:.1f} strikeouts "
            f"vs the line of {market_line} — a {ew} edge of {margin:+.2f}."
        )
    elif prop_type in ("Points Over", "Rebounds Over", "Assists Over",
                       "Blocks Over", "Steals Over"):
        stat = prop_type.replace(" Over", "")
        parts.append(
            f"Model projects {player} at {model_line:.1f} {stat.lower()} "
            f"vs the market line of {market_line} ({margin:+.1f} edge)."
        )
    else:
        parts.append(
            f"Model projects {player} at {model_line:.2f} vs line of {market_line} "
            f"— {ew} {margin:+.2f} edge."
        )

    return " ".join(parts)


# ── WNBA narrative ────────────────────────────────────────────────────────────

def _wnba_narrative(pick: str, signals: List[str], research: List[str],
                    edge: float) -> str:
    ew = _edge_word(edge)
    # "{team} injuries benefit {other_team} (+X%)" — {team} is the injured side
    inj_m = _search_first(
        r"(.+?) injuries benefit (.+?) \(\+(\d+(?:\.\d+)?)%\)", signals
    )
    # "{team} lineup impact (-X%)" — that team is short-handed (hurts them, helps opponent)
    lineup_m = _search_first(
        r"(.+?) lineup impact \(-(\d+(?:\.\d+)?)%\)", signals
    )
    # "{team} on back-to-back" — that team has fatigue (hurts them)
    b2b_m = _search_first(r"(.+?) on back-to-back", signals)

    parts = []

    # Identify who has back-to-back fatigue (if any)
    b2b_team      = b2b_m.group(1).strip() if b2b_m else None
    pick_has_b2b  = b2b_team and (pick in b2b_team or b2b_team in pick)
    opp_has_b2b   = b2b_team and not pick_has_b2b

    # Identify opponent lineup impact — this is a POSITIVE signal for the pick
    opp_lineup_pct = None
    opp_lineup_team = None
    if lineup_m:
        lt = lineup_m.group(1).strip()
        if not (pick in lt or lt in pick):          # lineup hit is on the opponent
            opp_lineup_team = lt
            opp_lineup_pct  = float(lineup_m.group(2))

    if inj_m:
        injured_team = inj_m.group(1).strip()
        beneficiary  = inj_m.group(2).strip()
        inj_pct      = float(inj_m.group(3))

        pick_is_beneficiary = (pick in beneficiary or beneficiary in pick)
        pick_is_injured     = (pick in injured_team or injured_team in pick)

        if pick_is_beneficiary:
            # The injured team is the opponent — good for the pick
            if opp_lineup_pct is not None:
                # Both injury signals say the same opponent is hurt
                own_pct_str = (
                    f" despite {pick}'s own lineup losses ({opp_lineup_pct:.0f}% pts-share)"
                    if opp_lineup_team and (pick in opp_lineup_team or opp_lineup_team in pick)
                    else ""
                )
                parts.append(
                    f"Key absences for {injured_team} ({inj_pct:.0f}% of their scoring output) "
                    f"shift win probability toward {pick}."
                )
            else:
                parts.append(
                    f"Key absences for {injured_team} ({inj_pct:.0f}% of their scoring output) "
                    f"shift win probability toward {pick}."
                )

        elif pick_is_injured:
            # The PICK team has a minor injury — but check if the opponent has a bigger problem
            if opp_lineup_pct is not None:
                # Opponent's lineup impact is the dominant reason for the pick
                b2b_clause = ""
                if pick_has_b2b:
                    b2b_clause = ", and despite a back-to-back,"
                elif opp_has_b2b:
                    b2b_clause = f", aided by {b2b_team}'s back-to-back fatigue,"
                parts.append(
                    f"{opp_lineup_team}'s heavy injury load ({opp_lineup_pct:.0f}% pts-share missing)"
                    f"{b2b_clause} gives {pick} a {ew} edge despite their own minor absences "
                    f"({inj_pct:.0f}%)."
                )
            elif pick_has_b2b:
                # Pick is on back-to-back AND injured — model still likes them
                parts.append(
                    f"Despite {pick}'s minor lineup losses ({inj_pct:.0f}% pts-share) and "
                    f"back-to-back fatigue, the underlying net rating advantage still favors {pick}."
                )
            elif opp_has_b2b:
                parts.append(
                    f"Although {pick} is missing some contributors ({inj_pct:.0f}% pts-share), "
                    f"{b2b_team}'s back-to-back fatigue gives {pick} the edge."
                )
            else:
                parts.append(
                    f"The model continues to favor {pick} despite their lineup losses "
                    f"({inj_pct:.0f}% pts-share), as their net rating edge holds a {ew} advantage."
                )

    # No inj_m but opponent has lineup impact
    if not parts and opp_lineup_pct is not None:
        b2b_clause = f" aided by {b2b_team}'s back-to-back fatigue and" if opp_has_b2b else ""
        parts.append(
            f"Key absences for {opp_lineup_team} ({opp_lineup_pct:.0f}% pts-share missing)"
            f"{b2b_clause} shift win probability toward {pick}."
        )

    # No injury signals but opponent is on back-to-back
    if not parts and opp_has_b2b:
        parts.append(
            f"{b2b_team}'s back-to-back fatigue reduces their effective strength, "
            f"creating a {ew} {edge*100:.1f}% edge for {pick}."
        )

    if not parts:
        parts.append(
            f"The model rates {pick} at a {ew} {edge*100:.1f}% edge based on "
            f"blended season and recent net ratings."
        )

    # Strength-of-schedule note when the pick has faced a meaningfully tougher
    # slate than its opponent (the model credits that in its rating).
    sos_m = _search_first(
        r"Strength of schedule — (.+?): opp avg net ([+-][\d.]+) \| (.+?): opp avg net ([+-][\d.]+)",
        research,
    )
    if sos_m:
        t1, s1, t2, s2 = sos_m.group(1), float(sos_m.group(2)), sos_m.group(3), float(sos_m.group(4))
        pick_sos = s1 if (pick in t1 or t1 in pick) else (s2 if (pick in t2 or t2 in pick) else None)
        opp_sos  = s2 if (pick in t1 or t1 in pick) else (s1 if (pick in t2 or t2 in pick) else None)
        if pick_sos is not None and opp_sos is not None and (pick_sos - opp_sos) >= 1.0:
            parts.append(
                f"{pick} has also faced a tougher schedule (opponents avg "
                f"{pick_sos:+.1f} net vs {opp_sos:+.1f}), which the model credits."
            )

    return " ".join(parts)


# ── IPL narrative ─────────────────────────────────────────────────────────────

def _ipl_narrative(pick: str, signals: List[str], research: List[str],
                   edge: float) -> str:
    ew = _edge_word(edge)
    form_m  = _search_first(
        r"Form edge: (.+?) \(blended win-rate gap (\d+)%\)", signals
    )
    venue_m = _search_first(
        r"Home venue advantage: (.+?) \(\+", signals
    )

    parts = []

    if form_m:
        form_team = form_m.group(1).strip()
        gap       = int(form_m.group(2))
        pick_has_form = (pick in form_team or form_team in pick)

        if pick_has_form:
            if venue_m:
                venue_team = venue_m.group(1).strip()
                pick_is_home = (pick in venue_team or venue_team in pick)
                if pick_is_home:
                    parts.append(
                        f"{pick} combines home venue advantage with a {gap}% form edge "
                        f"for a {ew} {edge*100:.1f}% model advantage."
                    )
                else:
                    parts.append(
                        f"{pick}'s {gap}% blended win-rate edge overcomes "
                        f"{venue_team}'s home venue advantage, giving the model a {ew} edge."
                    )
            else:
                parts.append(
                    f"{pick}'s {gap}% blended win-rate form edge drives "
                    f"a {ew} {edge*100:.1f}% advantage over current market odds."
                )
        else:
            # Pick doesn't have form edge but is still favored (venue or other)
            if venue_m:
                venue_team = venue_m.group(1).strip()
                if pick in venue_team or venue_team in pick:
                    parts.append(
                        f"{pick} benefits from home venue advantage — "
                        f"the market underestimates their win probability by {edge*100:.1f}%."
                    )

    elif venue_m:
        venue_team = venue_m.group(1).strip()
        if pick in venue_team or venue_team in pick:
            parts.append(
                f"{pick} holds home venue advantage, which the model rates at "
                f"a {ew} {edge*100:.1f}% edge over market odds."
            )

    if not parts:
        parts.append(
            f"The model rates {pick} at a {ew} {edge*100:.1f}% edge based on "
            f"form, venue, and head-to-head context."
        )

    return " ".join(parts)


# ── MLS narrative ─────────────────────────────────────────────────────────────

def _mls_narrative(pick: str, bet_type: str, signals: List[str], research: List[str],
                   edge: float) -> str:
    ew = _edge_word(edge)
    proj_m   = _search_first(r"xG projection: ([\d.]+) – ([\d.]+)", signals)
    xgd_m    = _search_first(r"xG edge: (.+?) \(([+-][\d.]+) xGD", signals)
    form_m   = _search_first(r"(.+?) recent form: ([+-][\d.]+) xGD last (\d+)", signals)
    inj_m    = _search_first(r"⚕ (.+?) injury impact \(-(\d+)%\)", signals)
    venue_m  = _search_first(r"Home (?:fortress )?venue: (.+?) \(", signals)

    bt = bet_type.lower()
    parts = []

    if bt == "draw":
        if proj_m:
            lh, la = proj_m.group(1), proj_m.group(2)
            parts.append(
                f"The model projects a closely contested match ({lh} – {la} xG) "
                f"with a {_edge_word(edge)} edge on the draw — above the market's implied probability."
            )
        else:
            parts.append(
                f"The model gives a {ew} edge on the draw, rating both teams as evenly matched."
            )

    elif bt == "total":
        direction = "over" if pick.lower().startswith("over") else "under"
        if proj_m:
            lh, la = proj_m.group(1), proj_m.group(2)
            total_proj = float(lh) + float(la)
            line_m = _search_first(r"([\d.]+)$", pick)
            line = line_m.group(1) if line_m else "?"
            parts.append(
                f"Model projects {total_proj:.2f} total goals ({lh} + {la}), "
                f"{'above' if direction == 'over' else 'below'} the line of {line} — a {ew} edge."
            )
        else:
            parts.append(
                f"The model finds a {ew} {edge*100:.1f}% edge on the {pick} total."
            )

    elif bt == "spread":
        line_m = _search_first(r"([+-][\d.]+)\s*$", pick)
        line = line_m.group(1) if line_m else ""
        if proj_m:
            lh, la = proj_m.group(1), proj_m.group(2)
            parts.append(
                f"With a {lh} – {la} xG projection, "
                f"the model gives a {ew} edge on {pick}."
            )
        else:
            parts.append(
                f"The model gives a {ew} edge on {pick} covering the {line} spread."
            )

    else:  # Moneyline
        if xgd_m:
            team = xgd_m.group(1)
            diff = xgd_m.group(2)
            parts.append(
                f"{team} holds a {diff} xGD advantage, giving the model a {ew} edge on {pick} ML."
            )
        elif form_m:
            team = form_m.group(1)
            xgd  = form_m.group(2)
            n    = form_m.group(3)
            parts.append(
                f"{team}'s recent form ({xgd} xGD over last {n} games) "
                f"drives a {ew} {edge*100:.1f}% edge on {pick}."
            )
        elif inj_m:
            inj_team = inj_m.group(1)
            pct = inj_m.group(2)
            if inj_team.lower() not in pick.lower():
                parts.append(
                    f"Injuries to {inj_team} ({pct}% lineup impact) shift the edge toward {pick}."
                )
            else:
                parts.append(
                    f"Despite injuries affecting {pick} ({pct}%), the model still sees a {ew} edge on their ML."
                )
        elif venue_m:
            parts.append(
                f"Home venue advantage for {venue_m.group(1)} supports the {ew} {edge*100:.1f}% edge on {pick}."
            )
        else:
            parts.append(
                f"The model rates {pick} at a {ew} {edge*100:.1f}% edge based on xG-adjusted team strength."
            )

    return " ".join(parts)


# ── Main entry points ─────────────────────────────────────────────────────────

def build_card_context(
    sport: str,
    pick: str,
    bet_type: str,
    signals: List[str],
    research: List[str],
    model_prob: float,
    market_prob: float,
    edge: float,
) -> Tuple[str, List[str]]:
    """
    Returns (narrative, context_items) for display on a singles/display card.

    narrative     — plain-English explanation of the pick
    context_items — merged deduplicated stat list (replaces signals + research sections)

    The raw signals and research are preserved in the bet dict for state management.
    Confidence is computed upstream and is NOT affected by this function.
    """
    context = merge_context(signals, research)
    context = _sort_context(sport, context)

    if sport == "MLB":
        narrative = _mlb_narrative(pick, bet_type, signals, research, edge)
    elif sport == "NBA":
        narrative = _nba_narrative(pick, bet_type, signals, research, edge)
    elif sport == "NHL":
        narrative = _nhl_narrative(pick, bet_type, signals, research, edge)
    elif sport == "WNBA":
        narrative = _wnba_narrative(pick, signals, research, edge)
    elif sport == "IPL":
        narrative = _ipl_narrative(pick, signals, research, edge)
    elif sport == "MLS":
        narrative = _mls_narrative(pick, bet_type, signals, research, edge)
    else:
        narrative = ""

    return narrative, context


def build_prop_context(
    sport: str,
    prop_type: str,
    player: str,
    team: str,
    opponent: str,
    signals: List[str],
    research: List[str],
    model_line: float,
    market_line: float,
    edge: float,
) -> Tuple[str, List[str]]:
    """Returns (narrative, context_items) for a prop pick card."""
    context = merge_context(signals, research)
    narrative = _prop_narrative(
        prop_type, player, team, opponent, signals, research,
        model_line, market_line, edge,
    )
    return narrative, context
