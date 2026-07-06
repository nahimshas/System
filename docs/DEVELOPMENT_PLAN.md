# System Development Plan

This file tracks the **implementation roadmap** (phases, model changes, architectural decisions) for the sports betting analysis system.

> **Note:** This is NOT the Claude project memory file. The canonical memory file is:
> `/Users/nahim/.claude/projects/-Users-nahim-Desktop-Claude-Projects-Code-Projects/memory/project_sports_betting.md`
> Always update that file when asked to "update the project memory."

---

## Architecture Overview

- **Data sources**: The Odds API (market odds), MLB Stats API, ESPN (live scores + settlement), wttr.in (weather), no-key public APIs throughout
- **State**: `state/picks_YYYY-MM-DD.json` — locked morning picks, merged on re-runs
- **History**: `state/history.json` — all-time settled bet outcomes (budget sports only)
- **Watchlist history**: `state/watchlist_history.json` — settled watchlist picks (NHL/IPL/WNBA/MLS)
- **Report**: `docs/index.html` — rendered by Jinja2, served via GitHub Pages, auto-refreshes live scores every 60s
- **CI**: GitHub Actions runs daily at ~9am PDT, commits report + state to repo

### Sport Module Architecture (Phases 1–5, May 2026)
Each sport has its own module (`src/sports/{sport}.py`) implementing a 5-method protocol: `fetch_games`, `fetch_context`, `analyze_games`, `fetch_props`, `settle`. Sports are registered in `src/sports/registry.py` with capability flags that control all routing (budget pool, parlay eligibility, display section, props, settlement). `src/main.py` runs a single loop over the registry — no per-sport if/else blocks. **Adding a new sport or graduating a watchlist sport to budget requires zero changes to `main.py`.**

See `CLAUDE.md` in the repo root for a quick-start guide, and the project memory file for the full architecture reference.

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

The system analyzes `history.json` for systematic biases and surfaces them as flagged suggestions in the report. A human reviews and manually updates the relevant `src/config/{sport}.py` if warranted.

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

## Recent Model Changes (May 2026)

### MLB: tanh soft cap on run differential
**File:** `src/models/edge_finder.py` — `analyze_mlb_game()`
**Problem:** Model was overconfident at high probability levels (70–90% predicted → ~45% actual win rate). Changing `MLB_SPREAD_STD` moved all probabilities and broke calibration in the mid-range.
**Fix:** `tanh` soft cap applied to the raw run differential before it enters the Normal CDF. Formula: `run_diff_capped = MLB_RUN_DIFF_CAP * tanh(run_diff / MLB_RUN_DIFF_CAP)` where `MLB_RUN_DIFF_CAP = 1.8`. Applies diminishing returns only to stacked large edges without affecting well-calibrated mid-range picks. `MLB_SPREAD_STD` stays at 1.8.

### MLB: ERA trap severity rework
**File:** `src/models/edge_finder.py` — `_era_trap_severity()`
**Problem (1):** Old `ip_weight = 1 - ip/80` counter-intuitively amplified small samples (treating uncertainty as danger), pushing 10-19 IP pitchers to MODERATE/SEVERE on noise.
**Problem (2):** No K/9 adjustment — high-strikeout pitchers legitimately outperform xFIP via fewer balls in play, but were flagged as traps anyway (e.g. Imanaga K/9 10.1 scored SEVERE).
**Fix:**
- `ip_weight` replaced with `ip_conf = min(1.0, max(0.0, (ip - 10) / 20.0))` — grows from 0 at 10 IP to full confidence at 30+ IP. Larger samples = more trust in the gap.
- K/9 guard added: `k9_factor = max(0.5, 1.0 - max(0.0, (k9 - 9.0) * 0.25))` — each K/9 above 9.0 reduces severity 25%, floored at 50%.
- Absolute minimum stays at 10 IP. ERA < FIP prerequisite gate and elite-pitcher guard (xFIP < 3.20 → cap at 0.79) unchanged.
**Downstream effect:** ERA trap severity only affects the confidence label (HIGH/MEDIUM) and which edge threshold triggers HIGH (5%/6%/7%). It does NOT affect projected scores or edge — those use blended xFIP independently.

### MLB: ERA trap terminology alignment
**File:** `src/report/card_context.py` — `_mlb_narrative()`
**Fix:** Narrative was mapping MODERATE severity to "notable" while the context section showed the raw word "MODERATE" — confusing and contradictory. Changed to "moderate" to match.

### NBA: B2B double-counting fix
**File:** `src/models/edge_finder.py` — `analyze_nba_game()`
**Problem:** Back-to-back penalty (`NBA_BACK_TO_BACK_PENALTY`) was applied to a team, then the rest_diff calculation on top of it double-counted the same fatigue signal.
**Fix:** `rest_diff` calculation now only fires when both teams have rest > 0 (i.e. neither is on a true back-to-back). The B2B penalty and rest_diff adjustment are now mutually exclusive.

### NHL: ESPN standings + missing return
**File:** `src/data/nhl_stats.py`
**Problem (1):** ESPN playoffs standings structure changed: `conference → standings.entries` directly (no division children layer). All team stats were returning 0, silently killing all NHL picks.
**Problem (2):** `analyze_nhl_game()` was missing a `return recs` statement — always returned `None`.
**Fix:** Dual-path parsing handles both regular season (division children) and playoff (no division layer) structures. `otLosses` capitalization fixed. `pointsFor/gamesPlayed` fallback added for playoff mode where `avgGoalsFor` is absent.

### Props: HRR target line fix
**File:** `src/data/odds_client.py`
**Problem:** Hits + Runs + RBI (HRR) props were showing "Over 1.5" when the system was configured to target "Over 0.5". The fallback silently used 1.5 when no book offered 0.5.
**Fix:** When `target_found` is populated, `pref_lines` is completely replaced with only target-line entries. When `target_found` is empty (no book offers the target line), `pref_lines` is cleared entirely so the market is skipped rather than falling back to a wrong line.

---

## WNBA Integration (May 2026)

**Watchlist-only sport (never enters budget pool or parlays). Active May–September.**

**Files:** `src/data/wnba_stats.py` (new), `src/config/wnba.py`, `src/models/edge_finder.py`, `src/data/outcome_checker.py`, `src/report/generator.py`, `src/main.py`, `src/report/templates/report.html`, `.github/workflows/daily_report.yml`

**Model:** Moneyline-only. Normal CDF on blended net ratings (season + recent form weighted 45/55). Home advantage +2%, B2B penalty −4%. Compound lineup penalty from injured players using points-share formula: `player_weight = max(player_ppg/team_ppg, mpg/40 * 0.6)` → continuously differentiates stars from role players. Capped at 30% total lineup penalty.

**Data sources:** All ESPN public APIs (no key). Team stats, schedule-based rest days, recent form from last 8 games, injury report with per-player PPG/MPG fetched from `sports.core.api.espn.com`.

**Constants in `src/config/wnba.py`:** `WNBA_HOME_ADVANTAGE=0.020`, `WNBA_BACK_TO_BACK_PENALTY=0.040`, `WNBA_RECENT_WEIGHT=0.45`, `WNBA_SPREAD_STD=8.5`, `WNBA_REPLACEMENT_RATE=0.55`, `WNBA_MAX_LINEUP_PENALTY=0.30`, `WNBA_STATUS_WEIGHTS`.

**Settlement:** Date-based via `check_and_settle_watchlist()` (same pattern as NHL/IPL). Live scores via 4th ESPN fetch in the browser (`basketball/wnba` scoreboard). Dedicated tile in the watchlist performance section.

---

## Bug Fixes (May 2026)

### IPL live score fix
**File:** `src/report/templates/report.html`
**Problem:** `core.espnuk.org` API was dead (connection refused), silently killing all IPL live score updates.
**Fix:** Replaced with `site.api.espn.com/apis/site/v2/sports/cricket/8048/scoreboard` — CORS-open, returns linescores (runs/wickets/overs) and match result in a single call. New `calcIplWinProb()` is directional (checks which team we picked). Server-settled IPL cards get `data-live-result` and `data-espn-done` baked in at render time.

### Today's Profit mismatch on re-runs
**File:** `src/report/generator.py` — `_tag_alloc()`
**Problem:** Subsequent intra-day runs refresh pick sizing with updated odds for games that haven't started yet. Bet cards in the league sections pulled from `singles_display` (refreshed prices) while the allocation table used `all_singles` (morning prices). The JS reads `data-cost`/`data-profit` from cards, not the table, so "Today's Profit" diverged from what the user saw.
**Fix:** `_tag_alloc()` now snaps `total_cost`, `profit_if_win`, `num_contracts`, and `edge_pct` on any card with an `alloc_rank` to the locked `all_singles` values. Cards and table always agree. Safe because users only buy contracts once the game is locked (started), at which point pricing is already frozen by the lock mechanism.

### Props by-sport tiles not updating live
**Files:** `src/data/outcome_checker.py`, `src/report/templates/report.html`
**Problem:** The NBA/MLB by-sport accuracy tiles were Python-rendered only — `updatePropAccuracy()` updated the overall hit rate, by-type rows, and by-conf blocks live, but never touched the by-sport tiles. A good day for NBA props had no visible effect on the NBA tile.
**Fix:** `load_prop_accuracy()` now computes `hist_by_sport` (pre-today baseline per sport). Each by-sport tile gets `data-hist-sport-total`/`data-hist-sport-hits` attributes. `updatePropAccuracy()` now includes a by-sport update block that reads today's settled prop cards grouped by sport and updates each tile in real time.

---

## Card Narrative + Context System (May 2026)

**Files:** `src/report/card_context.py` (new), `src/state/manager.py`, `src/report/templates/report.html`

**What it does:** Replaces the separate "Key Signals" and "Research & Stats" card sections with:
1. A plain-English narrative paragraph ("Why this pick") explaining the primary driver
2. A single merged/deduplicated context list (signals + research, with superseded lines removed)

Both sections are wrapped in a collapsible `card-insight` panel (clickable "Why this pick ▾" header).

**Key design decision:** Confidence labels are computed upstream in `edge_finder` before `bet_to_dict` is called. The card context system runs at display time (`bet_to_dict` / `prop_to_dict`) and never affects pick selection or sizing.

**Narrative logic per sport:**
- MLB: detects ERA trap (primary driver) → pitcher xFIP mismatch → injury → generic fallback. Totals get a separate projection-vs-line narrative.
- NBA: detects B2B → injury → generic net rating edge. Totals get over/under projection narrative.
- NHL: detects B2B → injury → generic season net rating narrative.
- Props: covers Hits/HRR/TB/Strikeouts/Points/Rebounds/Assists.

**Signal deduplication:** Pitcher stat lines, park factor, weather signals, and umpire signal lines are dropped from signals when research has the fuller version. Internal model debug lines (tanh cap, expected runs, form weights) are suppressed from the context display.

**Hydration for state-loaded picks:** `_hydrate_bet()` and `_hydrate_prop()` in `src/main.py` recompute `narrative`/`context` for any pick dict loaded from a state file that was saved before this feature was deployed. Applied in both the normal run path and the `code_only` (re-render) path.

---

## Jul 4 2026 — MLB Optimization Package (data-driven constants reset)

Derived from a full reconstruction sweep over the decision log (June, 231 games, exact formula re-run per candidate with daily top-5 budget re-selection), validated out-of-sample and against actual recorded P&L. Current constants ran −12% ROI at realistic costs; the package simulated ~breakeven-to-positive (+10pp relative).

**Changes (all effective from the Jul 5 2026 morning run):**
1. `_INJ_RUNS_PER_PCT` 0.08 → **0.0** (`edge_finder.py`) — injury-driven picks won 38% over 126 samples with flat CLV: the market already prices injuries; subtracting runs double-counted them. Injury credibility cap KEPT (anchors toward market, never creates edge). Injury data still logged/displayed as context.
2. Offense weight 0.6 → **0.4** (`_OFF_W`, `edge_finder.py`) — season-OPS deviations over-projected run differences.
3. `MLB_SPREAD_STD` 1.8 → **2.2** — flatter run_diff→probability mapping (mid-range probs were systematically overconfident: model 55–65% buckets realized 45–47%).
4. `MLB_CRED_CAP` 0.15 → **0.10** + `state/cap_state.json` mlb.credibility* current/default values — disagreements beyond 10pp were noise.
5. **`BUDGET_MIN_EDGE` = 0.05** (new, `config/base.py`) — real-money entry now requires ≥5% effective edge (display/watchlist/logs keep `MIN_EDGE` 0.03). Applied in `main.py` budget routing; parlays inherit it via `all_singles_raw`.
6. **Dog-with-better-starter confidence promotion** (`edge_finder.py` MLB spread section): run-line dog whose starter outscores the favorite's by >0.1 composite sp_score → promoted to HIGH (skipped when injury/TBD-capped) → floats to top of confidence-first slot ranking. Validated 39-18 (68%) Jun 13–30. New `dog_better_starter` flag on recs → shadow log field + decision-log features (`home/away_dog_better_sp`) so the pattern keeps being measured. Exempt from the `_recalibrate_confidence` edge-based downgrade.
7. Card narratives (`card_context.py`): injuries reworded as market-priced context (never "primary driver"); new lead narrative for promoted pattern picks.

**Expected visible effects:** fewer budget singles on thin days (floor), spread-dog-heavy card, smaller displayed edges (5–10% range), promoted picks labeled HIGH with the "validated pattern" signal. **Review checkpoint: ~2 weeks (mid-July)** — budget CLV should be ≥ 0; promoted picks ≥ ~58% → keep; ≤ ~55% → demote pattern.

---

## Jul 6 2026 — Analysis toolkit + weekly health loop

Turned the manual "user notices → session investigates" pattern into a scheduled loop.

**Files:**
- `tools/analysis/backtest.py` — canonical decision-log backtest engine (reconstructs MLB probs from logged features, daily top-5 sim with promotion + conf-first ranking, 4.5% vig). CLI: `--set KEY=VAL` variants, `--since/--until`, `--pattern-only`. `LIVE` dict at top must be kept in sync with shipped constants.
- `tools/analysis/health_report.py` — read-only weekly health checks: bankroll-vs-ledger drift, budget record/P&L/CLV, promoted-pattern record, governor/calibration/cap state, and automatic evaluation of every due checkpoint. `--json` for machines.
- `tools/analysis/checkpoints.json` — pre-registered evaluations with explicit pass/fail rules + dates (package_health, ha_zero, rlsig_30, pattern_card due Jul 18; era_trap_fip_bonus Sep 1). Convention: every future model change registers a checkpoint here; resolved ones get status+resolution, never deleted.
- `docs/health_routine_prompt.md` — canonical prompt for the weekly `model-health-weekly` Claude routine (Sundays 8am PT; git-clone I/O like the debrief; publishes docs/health_latest.html + health_history.json; notifies via KV only when ACTION NEEDED).

**Contract:** the routine measures and recommends; it never changes model code/constants/state. Implementation of a PASSed checkpoint happens in a normal session.

---

## How to continue in a future session

Tell Claude: **"Continue work on the sports betting system. Check `docs/DEVELOPMENT_PLAN.md` for the roadmap."**

For Phase 2: **"We now have ~20+ settled bets. Implement Phase 2 trend visualization from the development plan."**

For Phase 3: **"We now have 50+ settled bets. Implement Phase 3 model self-calibration from the development plan."**
