"""
Card context builder.

Takes raw signals + research from edge_finder and produces:
  • narrative    — 2-3 sentence plain-English explanation of why the pick was made
  • context      — single merged, deduplicated list replacing the two separate sections

The confidence label is computed upstream (edge_finder → _confidence_label) before
this runs, so these transformations never affect pick selection or sizing.
"""
import re
from typing import List, Tuple, Optional


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

    elif severe_traps:
        t = severe_traps[0]
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

    elif home_p and away_p and abs(float(home_p.group(2)) - float(away_p.group(2))) >= 0.6:
        h_xfip, a_xfip = float(home_p.group(2)), float(away_p.group(2))
        better = home_p.group(1) if h_xfip < a_xfip else away_p.group(1)
        b_xfip = min(h_xfip, a_xfip)
        worse = away_p.group(1) if h_xfip < a_xfip else home_p.group(1)
        w_xfip = max(h_xfip, a_xfip)
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
    # Pattern: "{team} injuries benefit {other_team} (+X%)"
    inj_m = _search_first(
        r"(.+?) injuries benefit (.+?) \(\+(\d+(?:\.\d+)?)%\)", signals
    )
    # Pattern: "{team} lineup impact (-X%)" — that team is short-handed
    own_inj = _search_first(
        r"(.+?) lineup impact \(-(\d+(?:\.\d+)?)%\)", signals
    )
    # B2B: "{team} on back-to-back (-X%)"
    b2b_m = _search_first(r"(.+?) on back-to-back", signals)

    parts = []

    if inj_m:
        injured_team = inj_m.group(1).strip()
        beneficiary  = inj_m.group(2).strip()
        inj_pct      = float(inj_m.group(3))

        pick_is_beneficiary = (pick in beneficiary or beneficiary in pick)
        pick_is_injured     = (pick in injured_team or injured_team in pick)

        if pick_is_beneficiary:
            # Pick's opponent is injured
            if own_inj and (pick in own_inj.group(1) or own_inj.group(1) in pick):
                own_pct = float(own_inj.group(2))
                parts.append(
                    f"Despite {pick}'s own lineup losses ({own_pct:.0f}% pts-share), "
                    f"{injured_team}'s heavier absences ({inj_pct:.0f}%) give the model a net edge."
                )
            else:
                parts.append(
                    f"Key absences for {injured_team} ({inj_pct:.0f}% of their scoring output) "
                    f"shift win probability toward {pick}."
                )
        elif pick_is_injured:
            # Pick's own team is injured but model still favors them
            if b2b_m:
                b2b_team = b2b_m.group(1).strip()
                parts.append(
                    f"Although {pick} is missing key contributors ({inj_pct:.0f}% pts-share), "
                    f"{b2b_team}'s back-to-back fatigue and the underlying net rating advantage "
                    f"still favor {pick}."
                )
            else:
                parts.append(
                    f"The model continues to favor {pick} despite their lineup losses "
                    f"({inj_pct:.0f}% pts-share), as their net rating edge holds a {ew} advantage."
                )

    if not parts and b2b_m:
        b2b_team = b2b_m.group(1).strip()
        if b2b_team not in pick and pick not in b2b_team:
            parts.append(
                f"{b2b_team}'s back-to-back fatigue reduces their effective strength, "
                f"creating a {ew} {edge*100:.1f}% edge for {pick}."
            )

    if not parts:
        parts.append(
            f"The model rates {pick} at a {ew} {edge*100:.1f}% edge based on "
            f"blended season and recent net ratings."
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
