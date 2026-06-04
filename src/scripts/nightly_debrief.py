#!/usr/bin/env python3
"""
Nightly debrief — 10 PM PST
Fetches ESPN results for today's picks, uses Claude + web search to analyze
what happened vs. what the model predicted, then renders a styled HTML report
saved to docs/debrief_latest.html.
"""

import datetime
import json
import os
import re
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("America/Los_Angeles")
except ImportError:
    _TZ = None

import anthropic

# ── Constants ─────────────────────────────────────────────────────────────────

MODEL = "claude-opus-4-8"

ESPN_SPORT_PATH = {
    "MLB":  "baseball/mlb",
    "NBA":  "basketball/nba",
    "NHL":  "hockey/nhl",
    "WNBA": "basketball/wnba",
    "MLS":  "soccer/usa.1",
    "NFL":  "football/nfl",
    "IPL":  "cricket/8048",
}

SPORT_EMOJI = {
    "MLB": "⚾", "NBA": "🏀", "NHL": "🏒",
    "WNBA": "🏀", "MLS": "⚽", "NFL": "🏈", "IPL": "🏏",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def today_pacific() -> str:
    try:
        return datetime.datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:
        return datetime.date.today().isoformat()


def load_state(today: str) -> dict:
    path = REPO_ROOT / f"state/picks_{today}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"No state file at {path}")
        return {}


def collect_picks(state: dict) -> tuple[list, list]:
    """Returns (budget_picks, watchlist_picks) from state."""
    budget = state.get("singles", [])

    # All display picks from main pool (MLB/NBA/NFL/NHL)
    display = state.get("singles_display", [])
    budget_keys = {
        (p.get("sport"), p.get("home_team"), p.get("away_team"), p.get("pick"))
        for p in budget
    }

    watchlist_pool = [p for p in display if
                      (p.get("sport"), p.get("home_team"), p.get("away_team"), p.get("pick"))
                      not in budget_keys]

    # Own-tile watchlist sports
    for slug in ("wnba", "mls", "ipl"):
        watchlist_pool.extend(state.get(f"{slug}_display", []))

    return budget, watchlist_pool


# ── ESPN score fetching ───────────────────────────────────────────────────────

def _fetch_espn(sport: str, date_str: str) -> list:
    path = ESPN_SPORT_PATH.get(sport.upper(), "")
    if not path:
        return []
    url = f"https://site.api.espn.com/apis/site/v2/sports/{path}/scoreboard?dates={date_str}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=12) as r:
            data = json.loads(r.read())
        return data.get("events", [])
    except Exception as e:
        print(f"  ESPN {sport} error: {e}")
        return []


def fetch_all_espn_scores(today: str) -> dict:
    """Returns {sport: [event, ...]} for all active sports."""
    date_str = today.replace("-", "")
    active = set()

    # Only fetch sports that have picks today
    return {sport: _fetch_espn(sport, date_str) for sport in ESPN_SPORT_PATH}


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (s or "").lower())


def find_event(events: list, home_team: str, away_team: str) -> dict | None:
    hn = _norm(home_team)
    an = _norm(away_team)

    def hits(a: str, b: str) -> bool:
        return bool(a) and bool(b) and (a in b or b in a)

    for event in events:
        comps = (event.get("competitions") or [{}])[0].get("competitors", [])
        names = [_norm(c.get("team", {}).get("displayName") or
                       c.get("team", {}).get("name") or "")
                 for c in comps]
        if len(names) >= 2 and (
            (hits(names[0], hn) or hits(names[0], an)) and
            (hits(names[1], hn) or hits(names[1], an))
        ):
            scores = {}
            abbrs = {}
            for c in comps:
                side = c.get("homeAway", "")
                scores[side] = int(c.get("score") or 0)
                abbrs[side] = (c.get("team", {}).get("abbreviation") or
                               c.get("team", {}).get("shortDisplayName") or side[:3].upper())
            state = (event.get("status") or {}).get("type", {}).get("state", "pre")
            detail = (event.get("status") or {}).get("type", {}).get("shortDetail", "")
            return {
                "home_score": scores.get("home", 0),
                "away_score": scores.get("away", 0),
                "home_abbr":  abbrs.get("home", "HM"),
                "away_abbr":  abbrs.get("away", "AW"),
                "status":     state,   # pre | in | post
                "detail":     detail,
            }
    return None


def determine_result(pick: dict, score: dict) -> str:
    """Returns 'WON', 'LOST', 'PUSH', or 'PENDING'."""
    if score["status"] != "post":
        return "PENDING"

    p  = (pick.get("pick") or "").strip()
    bt = (pick.get("bet_type") or "").lower()
    hs = score["home_score"]
    aw = score["away_score"]
    ht = pick.get("home_team", "")

    def is_home_pick(team_str: str) -> bool:
        t = team_str.replace(" ML", "").strip()
        return _norm(ht) in _norm(t) or _norm(t) in _norm(ht)

    if bt == "moneyline":
        if hs == aw:
            return "PUSH"
        home_won = hs > aw
        return "WON" if (home_won == is_home_pick(p)) else "LOST"

    if bt == "total":
        over = p.lower().startswith("over")
        try:
            line = float(re.sub(r"^(over|under)\s*", "", p, flags=re.IGNORECASE))
        except ValueError:
            return "PENDING"
        total = hs + aw
        if total == line:
            return "PUSH"
        return "WON" if ((over and total > line) or (not over and total < line)) else "LOST"

    if bt == "spread":
        m = re.match(r"^(.+?)\s+([-+]?\d+\.?\d*)\s*$", p)
        if not m:
            return "PENDING"
        spread = float(m.group(2))
        diff = (hs - aw) if is_home_pick(m.group(1)) else (aw - hs)
        if diff + spread == 0:
            return "PUSH"
        return "WON" if diff + spread > 0 else "LOST"

    return "PENDING"


# ── Build prompt context ──────────────────────────────────────────────────────

def _pick_summary(pick: dict, score: dict | None, result: str | None) -> dict:
    score_str = "No ESPN data"
    if score:
        if score["status"] == "post":
            score_str = f"{score['away_abbr']} {score['away_score']} – {score['home_abbr']} {score['home_score']} (Final)"
        elif score["status"] == "in":
            score_str = f"{score['away_abbr']} {score['away_score']} – {score['home_abbr']} {score['home_score']} (Live: {score['detail']})"
        else:
            score_str = f"Not started ({score['detail']})"

    return {
        "game":          pick.get("game", f"{pick.get('away_team','')} @ {pick.get('home_team','')}"),
        "sport":         pick.get("sport", ""),
        "pick":          pick.get("pick", ""),
        "bet_type":      pick.get("bet_type", ""),
        "confidence":    pick.get("confidence", ""),
        "model_prob":    pick.get("model_prob_pct", ""),
        "market_prob":   pick.get("market_prob_pct", ""),
        "edge_pct":      pick.get("edge_pct", ""),
        "signals":       pick.get("signals", []),
        "research":      pick.get("research", []),
        "score":         score_str,
        "result":        result or "PENDING",
        "pnl":           (
            f"+${pick.get('profit_if_win', 0):.2f}" if result == "WON"
            else f"-${pick.get('total_cost', 0):.2f}" if result == "LOST"
            else ""
        ),
    }


SYSTEM_PROMPT = """You are an analytical assistant for a sports betting model called "the System."
Your job: analyze today's picks with their results, search the web for game context,
identify what the model got right and wrong, and flag useful patterns.

Rules:
- Be concise and analytical — no hype, no filler
- Focus on WHY signals held or failed (connect results back to specific signals)
- Use web_search to find context you need: lineup changes, key plays, injuries, weather, pitcher performance, etc.
- Report both successes and failures honestly — model improvement requires accurate diagnosis
- Search for games you need context on, but don't search for obvious results

Return a single JSON object with this exact schema (no markdown, pure JSON):
{
  "headline": "One-sentence summary of today's results (e.g. '3-2 on budget; ERA trap 2-for-2')",
  "budget_record": "W-L",
  "watchlist_record": "W-L",
  "picks": [
    {
      "game": "Away @ Home",
      "sport": "MLB",
      "pick": "Giants ML",
      "bet_type": "Moneyline",
      "confidence": "HIGH",
      "result": "WON",
      "score": "Giants 6 – Dodgers 3 (Final)",
      "pnl": "+$8.20",
      "why_picked": "Brief: what model signal drove this pick",
      "what_happened": "What actually occurred in the game (from web search)",
      "signal_verdict": "CORRECT / INCORRECT / MIXED — explain briefly",
      "key_insight": "One observation worth remembering for model improvement (or empty string)"
    }
  ],
  "patterns": [
    "Underdog ML picks: 3-0 today",
    "HIGH confidence: 2-1, MEDIUM confidence: 1-1 today",
    "MLB totals: 2-0 today"
  ],
  "model_observations": [
    "ERA trap signal fired on 2 picks, both WON — signal performing well",
    "TBD pitcher cap may be suppressing correct totals: both TBD-capped totals went Over tonight"
  ]
}"""


def build_user_prompt(today: str, budget_picks: list, watchlist_picks: list,
                      espn: dict) -> str:
    lines = [f"Date: {today}", ""]

    def format_pick(p: dict) -> str:
        sport = p["sport"].upper()
        score_data = find_event(espn.get(sport, []), p.get("home_team", ""), p.get("away_team", ""))
        result = determine_result(p, score_data) if score_data else "PENDING"
        ps = _pick_summary(p, score_data, result)

        out = [f"  Game: {ps['game']}  |  Sport: {ps['sport']}  |  Result: {ps['result']}"]
        out.append(f"  Pick: {ps['pick']} ({ps['bet_type']}, {ps['confidence']} confidence)")
        out.append(f"  Model: {ps['model_prob']}% vs Market: {ps['market_prob']}% | Edge: {ps['edge_pct']}%")
        out.append(f"  Score: {ps['score']}")
        if ps["pnl"]:
            out.append(f"  P&L: {ps['pnl']}")
        if ps["signals"]:
            out.append("  Key signals: " + " | ".join(str(s) for s in ps["signals"][:4]))
        return "\n".join(out)

    lines.append("=== BUDGET PICKS (Today's Card — real money) ===")
    if budget_picks:
        for p in budget_picks:
            lines.append(format_pick(p))
            lines.append("")
    else:
        lines.append("  (No budget picks today)")
        lines.append("")

    lines.append("=== WATCHLIST PICKS (all sport tabs — no money, model tracking) ===")
    if watchlist_picks:
        for p in watchlist_picks[:20]:  # cap at 20 to keep prompt manageable
            lines.append(format_pick(p))
            lines.append("")
    else:
        lines.append("  (No watchlist picks today)")
        lines.append("")

    lines.append("Please search for context on any games you need more detail on,")
    lines.append("then return the JSON analysis as described.")
    return "\n".join(lines)


# ── Claude API call ───────────────────────────────────────────────────────────

def call_claude(user_prompt: str) -> dict:
    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    messages = [{"role": "user", "content": user_prompt}]
    system   = [{"type": "text", "text": SYSTEM_PROMPT,
                 "cache_control": {"type": "ephemeral"}}]

    max_turns = 15
    for turn in range(max_turns):
        response = client.messages.create(
            model=MODEL,
            max_tokens=8192,
            system=system,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=messages,
        )

        if response.stop_reason == "end_turn":
            text = "".join(
                getattr(b, "text", "") for b in response.content
                if getattr(b, "type", "") == "text"
            )
            try:
                m = re.search(r"\{.*\}", text, re.DOTALL)
                return json.loads(m.group(0)) if m else {}
            except (json.JSONDecodeError, AttributeError) as e:
                print(f"JSON parse error: {e}\nRaw: {text[:500]}")
                return {}

        if response.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": response.content})
            tool_results = []
            for block in response.content:
                btype = getattr(block, "type", "")
                if btype == "tool_use":
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     "",
                    })
                elif btype == "server_tool_use":
                    # web_search results handled server-side — just acknowledge
                    tool_results.append({
                        "type":        "tool_result",
                        "tool_use_id": block.id,
                        "content":     "",
                    })
            if tool_results:
                messages.append({"role": "user", "content": tool_results})
            print(f"  Turn {turn + 1}: tool use → continuing")
            continue

        print(f"Unexpected stop_reason: {response.stop_reason}")
        break

    return {}


# ── HTML rendering ────────────────────────────────────────────────────────────

RESULT_STYLE = {
    "WON":     ("✅", "#22c55e", "#052213"),
    "LOST":    ("❌", "#ef4444", "#250808"),
    "PUSH":    ("↔️", "#94a3b8", "#1a1d27"),
    "PENDING": ("⏳", "#f59e0b", "#1a120a"),
}

VERDICT_COLOR = {
    "CORRECT":   "#22c55e",
    "INCORRECT": "#ef4444",
    "MIXED":     "#f59e0b",
}


def _result_badge(result: str) -> str:
    emoji, color, _ = RESULT_STYLE.get(result, ("❓", "#94a3b8", "#1a1d27"))
    return f'<span style="color:{color};font-weight:700">{emoji} {result}</span>'


def _verdict_badge(verdict: str) -> str:
    v = (verdict or "").split("—")[0].strip().upper()
    for key, color in VERDICT_COLOR.items():
        if key in v:
            return f'<span style="color:{color};font-size:0.82em;font-weight:600">{verdict}</span>'
    return f'<span style="color:#94a3b8;font-size:0.82em">{verdict}</span>'


def _pick_cards_html(picks: list, analysis_map: dict) -> str:
    parts = []
    for p in picks:
        key = (p.get("sport", ""), p.get("home_team", ""), p.get("away_team", ""))
        ana = analysis_map.get(key, {})
        result = ana.get("result") or p.get("_result", "PENDING")
        _, bg_accent, bg_card = RESULT_STYLE.get(result, ("", "#94a3b8", "#1a1d27"))
        sport_emoji = SPORT_EMOJI.get(p.get("sport", "").upper(), "🏆")
        game_str = ana.get("game") or f"{p.get('away_team','')} @ {p.get('home_team','')}"
        score_str = ana.get("score") or "—"
        pnl = ana.get("pnl", "")
        why = ana.get("why_picked", "")
        what = ana.get("what_happened", "")
        verdict = ana.get("signal_verdict", "")
        insight = ana.get("key_insight", "")

        pnl_html = (f'<span style="color:{bg_accent};font-weight:700;font-size:1.05em">'
                    f'{pnl}</span>') if pnl else ""

        parts.append(f"""
<div style="background:#1a1d27;border-radius:12px;padding:16px 18px;margin-bottom:12px;
            border-left:3px solid {bg_accent}">
  <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:8px">
    <div>
      <span style="font-size:0.78em;color:#64748b;letter-spacing:.05em;text-transform:uppercase">
        {sport_emoji} {p.get('sport','')} · {p.get('bet_type','').title()}
      </span><br>
      <span style="font-weight:700;font-size:1.05em;color:#e2e8f0">{game_str}</span>
    </div>
    <div style="text-align:right;flex-shrink:0;margin-left:12px">
      {_result_badge(result)}<br>
      {pnl_html}
    </div>
  </div>

  <div style="background:#0f1117;border-radius:8px;padding:10px 12px;margin-bottom:10px;
              font-size:0.9em">
    <span style="color:#94a3b8">Pick: </span>
    <span style="color:#38bdf8;font-weight:600">{p.get('pick','')}</span>
    &nbsp;·&nbsp;
    <span style="color:#94a3b8">Score: </span>
    <span style="color:#e2e8f0">{score_str}</span>
    <br>
    <span style="color:#64748b;font-size:0.85em">
      Model {p.get('model_prob_pct','')}% vs Market {p.get('market_prob_pct','')}%
      · Edge {p.get('edge_pct','')}%
      · {p.get('confidence','')} confidence
    </span>
  </div>

  {"<div style='margin-bottom:8px'><span style='color:#94a3b8;font-size:0.83em'>WHY PICKED: </span><span style='color:#cbd5e1;font-size:0.88em'>" + why + "</span></div>" if why else ""}
  {"<div style='margin-bottom:8px'><span style='color:#94a3b8;font-size:0.83em'>WHAT HAPPENED: </span><span style='color:#cbd5e1;font-size:0.88em'>" + what + "</span></div>" if what else ""}
  {"<div style='margin-bottom:6px'><span style='color:#94a3b8;font-size:0.83em'>SIGNAL: </span>" + _verdict_badge(verdict) + "</div>" if verdict else ""}
  {"<div style='background:#141720;border-radius:6px;padding:8px 10px;font-size:0.83em;color:#f59e0b'>💡 " + insight + "</div>" if insight else ""}
</div>""")
    return "\n".join(parts)


def render_html(today: str, analysis: dict, budget_picks: list,
                watchlist_picks: list, generated_at: str) -> str:
    # Build lookup map: (sport, home_team, away_team) → analysis pick dict
    analysis_map: dict = {}
    for ap in analysis.get("picks", []):
        game = ap.get("game", "")
        sport = ap.get("sport", "")
        m = re.match(r"^(.+?)\s+@\s+(.+)$", game)
        if m:
            away_norm = _norm(m.group(1))
            home_norm = _norm(m.group(2))
            for p in budget_picks + watchlist_picks:
                if (p.get("sport", "").upper() == sport.upper() and
                        _norm(p.get("home_team", "")) == home_norm and
                        _norm(p.get("away_team", "")) == away_norm):
                    key = (p.get("sport", ""), p.get("home_team", ""), p.get("away_team", ""))
                    analysis_map[key] = ap

    budget_html    = _pick_cards_html(budget_picks, analysis_map)
    watchlist_html = _pick_cards_html(watchlist_picks[:20], analysis_map)

    headline = analysis.get("headline", "Nightly Debrief")
    br       = analysis.get("budget_record", "—")
    wr       = analysis.get("watchlist_record", "—")

    patterns_html = ""
    for pat in analysis.get("patterns", []):
        patterns_html += f'<li style="color:#cbd5e1;margin-bottom:6px">{pat}</li>'

    obs_html = ""
    for obs in analysis.get("model_observations", []):
        obs_html += f'<li style="color:#cbd5e1;margin-bottom:6px">{obs}</li>'

    # Overall P&L from budget picks
    total_pnl = 0.0
    for p in budget_picks:
        key = (p.get("sport", ""), p.get("home_team", ""), p.get("away_team", ""))
        ap = analysis_map.get(key, {})
        result = ap.get("result", "PENDING")
        if result == "WON":
            total_pnl += float(p.get("profit_if_win", 0) or 0)
        elif result == "LOST":
            total_pnl -= float(p.get("total_cost", 0) or 0)

    pnl_color = "#22c55e" if total_pnl >= 0 else "#ef4444"
    pnl_str   = f"+${total_pnl:.2f}" if total_pnl >= 0 else f"-${abs(total_pnl):.2f}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1">
  <meta name="theme-color" content="#050507">
  <title>Nightly Debrief · {today}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    html, body {{
      background: #050507;
      color: #e2e8f0;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
      font-size: 16px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .container {{ max-width: 600px; margin: 0 auto; padding: 16px; }}
    .header {{
      background: #0f1117;
      border-radius: 14px;
      padding: 20px;
      margin-bottom: 20px;
      border-bottom: 2px solid #1e2433;
    }}
    .header-date {{ color: #64748b; font-size: 0.82em; margin-bottom: 4px; }}
    .header-headline {{ font-size: 1.15em; font-weight: 700; color: #e2e8f0; margin-bottom: 12px; }}
    .header-stats {{ display: flex; gap: 16px; flex-wrap: wrap; }}
    .stat-chip {{
      background: #1a1d27;
      border-radius: 8px;
      padding: 8px 14px;
      text-align: center;
    }}
    .stat-label {{ font-size: 0.72em; color: #64748b; text-transform: uppercase; letter-spacing: .05em; }}
    .stat-value {{ font-size: 1.1em; font-weight: 700; }}
    h2 {{
      font-size: 0.82em;
      color: #64748b;
      text-transform: uppercase;
      letter-spacing: .1em;
      margin: 20px 0 10px;
    }}
    .section {{ margin-bottom: 24px; }}
    ul {{ padding-left: 18px; }}
    .patterns-card, .obs-card {{
      background: #1a1d27;
      border-radius: 12px;
      padding: 16px 18px;
    }}
    .footer {{
      text-align: center;
      color: #374151;
      font-size: 0.75em;
      padding: 24px 0 32px;
    }}
    .empty {{ color: #4b5563; font-style: italic; font-size: 0.9em; padding: 12px 0; }}
  </style>
</head>
<body>
  <div class="container">

    <div class="header">
      <div class="header-date">📊 Nightly Debrief · {today}</div>
      <div class="header-headline">{headline}</div>
      <div class="header-stats">
        <div class="stat-chip">
          <div class="stat-label">Today's Card</div>
          <div class="stat-value">{br}</div>
        </div>
        <div class="stat-chip">
          <div class="stat-label">Watchlist</div>
          <div class="stat-value">{wr}</div>
        </div>
        <div class="stat-chip">
          <div class="stat-label">Budget P&L</div>
          <div class="stat-value" style="color:{pnl_color}">{pnl_str}</div>
        </div>
      </div>
    </div>

    <div class="section">
      <h2>Today's Card — Budget Picks</h2>
      {budget_html if budget_html.strip() else '<p class="empty">No budget picks today.</p>'}
    </div>

    <div class="section">
      <h2>Watchlist — All Sport Tabs</h2>
      {watchlist_html if watchlist_html.strip() else '<p class="empty">No watchlist picks today.</p>'}
    </div>

    {"<div class='section'><h2>Patterns Today</h2><div class='patterns-card'><ul>" + patterns_html + "</ul></div></div>" if patterns_html else ""}

    {"<div class='section'><h2>Model Observations</h2><div class='obs-card'><ul>" + obs_html + "</ul></div></div>" if obs_html else ""}

    <div class="footer">
      Generated {generated_at} · <a href="/index_spa.html" style="color:#38bdf8;text-decoration:none">← Back to Picks</a>
    </div>

  </div>
</body>
</html>"""


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    today = today_pacific()
    print(f"Nightly debrief for {today}")

    # Load state
    state = load_state(today)
    if not state:
        print("No state file found — writing empty debrief page")
        html = render_html(today, {}, [], [], datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        (REPO_ROOT / "docs/debrief_latest.html").write_text(html)
        sys.exit(0)

    budget_picks, watchlist_picks = collect_picks(state)
    total_picks = len(budget_picks) + len(watchlist_picks)
    print(f"  Budget picks: {len(budget_picks)}, Watchlist picks: {len(watchlist_picks)}")

    if total_picks == 0:
        print("No picks found in state — writing empty debrief page")
        html = render_html(today, {}, [], [], datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"))
        (REPO_ROOT / "docs/debrief_latest.html").write_text(html)
        sys.exit(0)

    # Fetch ESPN scores
    print("Fetching ESPN scores...")
    espn = fetch_all_espn_scores(today)
    for sport, events in espn.items():
        if events:
            print(f"  {sport}: {len(events)} game(s)")

    # Build prompt
    user_prompt = build_user_prompt(today, budget_picks, watchlist_picks, espn)
    print(f"Built prompt ({len(user_prompt)} chars). Calling Claude...")

    # Call Claude with web search
    analysis = call_claude(user_prompt)

    if not analysis:
        print("Claude returned empty analysis — writing ESPN-only debrief")
        analysis = {
            "headline": "Analysis unavailable — ESPN results only",
            "budget_record": "—",
            "watchlist_record": "—",
            "picks": [],
            "patterns": [],
            "model_observations": [],
        }

    print(f"Analysis received: {len(analysis.get('picks', []))} picks analyzed")

    # Render HTML
    generated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    html = render_html(today, analysis, budget_picks, watchlist_picks, generated_at)

    out_path = REPO_ROOT / "docs/debrief_latest.html"
    out_path.write_text(html, encoding="utf-8")
    print(f"Wrote {out_path} ({len(html):,} bytes)")


if __name__ == "__main__":
    main()
