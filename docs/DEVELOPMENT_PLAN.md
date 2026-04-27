# System Development Plan

This file tracks the implementation roadmap for the sports betting analysis system.
It exists so any future session can pick up context without relying on conversation history.

---

## Architecture Overview

- **Data sources**: The Odds API (market odds), MLB Stats API, ESPN (live scores + settlement), wttr.in (weather), no-key public APIs throughout
- **State**: `state/picks_YYYY-MM-DD.json` — locked morning picks, merged on re-runs
- **History**: `state/history.json` — all-time settled bet outcomes, appended daily
- **Report**: `docs/index.html` — rendered by Jinja2, served via GitHub Pages, auto-refreshes live scores every 60s
- **CI**: GitHub Actions runs daily at ~9am PDT, commits report + state to repo

---

## Phase 1 — Outcome Settlement ✅ Complete

**What it does:**
- `src/data/outcome_checker.py` runs at the start of each daily run
- Loads yesterday's `state/picks_YYYY-MM-DD.json`
- Fetches final scores from ESPN scoreboard API (no key needed)
- Determines WON / LOST / PUSH for each single and parlay
- Appends settled records to `state/history.json`
- `load_performance_summary()` aggregates history into totals, win rate, ROI, broken down by confidence tier (HIGH/MEDIUM) and sport (NBA/MLB/PARLAY)

**Performance table in report:**
- All-time Record, Win Rate, Total PnL, ROI shown in report footer
- Broken down by confidence tier and sport
- Updates live in the browser as today's games finish (JS reads `data-live-result` from settled bet cards)
- "● includes today" badge appears when live outcomes are reflected

**Key files:**
- `src/data/outcome_checker.py` — settlement logic
- `src/state/manager.py` — `bet_to_dict`, `parlay_to_dict`, `prop_to_dict`
- `src/report/generator.py` — calls `load_performance_summary()`
- `src/report/templates/report.html` — performance section + JS `updatePerformance()`

---

## Phase 2 — Trend Visualization ✅ Complete

**Trigger:** Build once ~20 settled bets exist in `history.json` (enough for charts to be meaningful).

**What to build:**

1. **Cumulative PnL curve** — line chart of running PnL over time (x = bet number or date, y = cumulative $PnL). Shows whether the model is on a run or grinding steadily.

2. **Rolling win rate (last 20 bets)** — line chart showing win % over a rolling window. Flat/rising = model is stable; sharp drop = possible drift or bad stretch.

3. **Calibration table** — group bets by model edge bucket (3–5%, 5–8%, 8%+) and show actual win rate per bucket. If HIGH edge bets win at the same rate as LOW edge bets, the edge estimates are not predictive.

**Implementation notes:**
- All data is already in `history.json` — no new data collection needed
- Charts: use lightweight inline SVG or Chart.js (CDN, no build step) in the report template
- Add a new `build_chart_data()` helper in `src/data/outcome_checker.py` that returns bucketed and time-series data
- Render charts in the performance section of `report.html` below the existing stat tiles
- Keep it collapsed behind a `<details>` toggle so it doesn't dominate the report on mobile

---

## Phase 3 — Model Self-Calibration 🔲 Not yet built

**Trigger:** Build once 50+ settled bets exist in `history.json`. Do NOT implement earlier — sample size too small for statistically meaningful patterns.

**What it does (read-only suggestions, no auto-apply):**

The system analyzes `history.json` for systematic biases and surfaces them as flagged suggestions in the report. A human reviews and manually updates `src/config.py` if warranted.

**Bias checks to run:**

| Check | Signal | Suggested adjustment |
|---|---|---|
| HIGH confidence win rate < 50% over 30+ bets | Threshold too loose | Raise `MIN_EDGE` or tighten `_confidence_label` |
| MLB totals: Unders winning > 60% | Run expectations too high | Reduce `league_avg_runs` in `edge_finder.py` |
| NBA totals: Overs winning > 60% | Pace assumptions too low | Increase `NBA_RECENT_FORM_WEIGHT` toward recent scoring |
| Home ML winning >> model prediction | Home advantage underweighted | Increase `NBA_HOME_ADVANTAGE` / `MLB_HOME_ADVANTAGE` |
| Playoff bets consistently wrong direction | Playoff factors miscalibrated | Adjust `NBA_PLAYOFF_SCORING_FACTOR` / `MLB_PLAYOFF_SCORING_FACTOR` |

**Rules to enforce (conservative calibration):**
- Minimum 30 bets in a category before flagging a bias (not 50 overall, 30 per bucket)
- Only flag if win rate deviation is > 10 percentage points from expected (not noise)
- Never auto-apply — always surface as "Suggested adjustment: ..." in the report
- Re-check after each adjustment with fresh data; don't chain multiple changes at once

**Implementation notes:**
- Add `analyze_calibration(records)` to `src/data/outcome_checker.py`
- Add a `calibration_warnings` list to the report context (similar to `change_warnings`)
- Render in the performance section with an orange/yellow callout style
- Config constants to potentially adjust: `MIN_EDGE`, `NBA_HOME_ADVANTAGE`, `MLB_HOME_ADVANTAGE`, `NBA_PLAYOFF_SCORING_FACTOR`, `MLB_PLAYOFF_SCORING_FACTOR`, `NBA_BACK_TO_BACK_PENALTY`, `KELLY_FRACTION`

---

## Completed Features (reference)

### Model inputs
- Market odds via The Odds API (bookmaker consensus — no-vig average across all books)
- NBA: ESPN team stats, recent form (14-day), rest days, schedule load (7-day), injuries, team leaders for props
- MLB: MLB Stats API pitcher stats (ERA/FIP/K9/BB9/HR9), team batting (OPS/AVG/SLG), bullpen ERA, schedule load, injuries, park factors
- Umpire tendencies: MLB Stats API `hydrate=officials` → built-in tendency table (~40 umpires)
- Weather: wttr.in (free, no key) → temp, wind speed/direction, precip % → run adjustment + signals
- Playoff context: date-based detection, sport-specific scoring/pace/IP adjustments

### State management
- Morning picks locked; subsequent runs merge (signal refresh + edge-based replacement)
- Line movement signal: 3%+ probability shift triggers sharp-money or fade signal
- Uncapped fresh singles pool ensures signal refresh works even for bets outside top 5

### Report features
- Live score updates (ESPN, every 60s) with win probability model (Normal CDF)
- WON/LOST/PUSH display on settled games
- Allocation table with live score/prob in score cell
- Bidirectional navigation (↑/↓) between bet cards and allocation table
- Performance table with live updates as today's games settle
- Props section (NBA: points/rebounds/assists; MLB: strikeouts/hits)
- Change warnings when morning picks are replaced on subsequent runs
- GitHub Actions manual trigger with `reset_state` and `league` options

---

## How to continue in a future session

Tell Claude: **"Continue work on the sports betting system. Check `docs/DEVELOPMENT_PLAN.md` for the roadmap."**

For Phase 2: **"We now have ~20+ settled bets. Implement Phase 2 trend visualization from the development plan."**

For Phase 3: **"We now have 50+ settled bets. Implement Phase 3 model self-calibration from the development plan."**
