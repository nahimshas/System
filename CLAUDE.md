# Sports Betting Analysis System — Claude Session Guide

## Who you're working with
The user does not write code. All requests come conversationally ("add MLS", "make WNBA a budget sport", "add NFL props", "why isn't the pick showing up"). Implement changes fully, commit, and summarize. Never ask the user to edit a file or run a command — tell them what workflow button to press if anything is needed on their end.

## Project location
`/Users/nahim/Documents/GitHub/System`

## Full project memory
The canonical deep-reference file is:
`/Users/nahim/.claude/projects/-Users-nahim-Desktop-Claude-Projects-Code-Projects/memory/project_sports_betting.md`

Read it at the start of any session that involves model changes, new sport integration, bug fixing, or anything beyond simple template edits. It contains the complete architecture, all design decisions, all known bugs and their fixes, and step-by-step guides for common tasks.

---

## What the system does
Runs daily at 9am PDT (GitHub Actions). Fetches odds for NBA/MLB/NFL/NHL/IPL/WNBA/MLS, runs edge models, picks top-5 singles + parlays + props, saves state, generates an HTML report, deploys to GitHub Pages, and sends an email.

Report URL: `https://nahimshas.github.io/System/`

---

## Architecture in one paragraph
Every sport has a **module** (`src/sports/{sport}.py`) that implements `fetch_games`, `fetch_context`, `analyze_games`, `fetch_props`, and `settle`. Modules are registered in `src/sports/registry.py` with **capability flags** (`enters_budget`, `in_main_display_pool`, `has_props`, etc.). `src/main.py` runs **one loop** over the registry — no per-sport if/else blocks exist anywhere in main.py. The flags tell the loop where to route each sport's picks automatically.

---

## Common user requests → what to do

| User says | What it means technically |
|---|---|
| "Add [sport]" | Create stats module + edge model + sport module + registry entry + report tile. See memory file "HOW TO ADD A NEW SPORT". |
| "Make [sport] a budget sport" | Flip 3 flags in `registry.py`: `enters_budget=True`, `enters_parlays=True`, `track_in_main_history=True`. See memory file "HOW TO GRADUATE". |
| "Add props to NFL" | Implement `fetch_props()` in `src/sports/nfl.py` + the model in `props_analyzer.py` + flip `has_props=True` in registry. No main.py changes. |
| "The pick isn't showing up" | Check: is the sport active this month? Is it in `leagues`? Did `fetch_games` return games? Did `analyze_games` return picks above `min_edge`? Is it a subsequent run overwriting morning picks? |
| "Add [sport] to the watchlist" | Same as "add [sport]" but watchlist-only: `enters_budget=False`, `in_main_display_pool=False`. |
| "Update the result for [game]" | Add the settled record to `state/watchlist_history.json` with correct `result: "WON"` or `"LOST"`. |
| "Run the code-only deploy" | Tell them: Actions → Daily Betting Report → Run workflow → check "code_only" → Run workflow. |
| "Reset today's picks" | Tell them: Actions → Daily Betting Report → Run workflow → check "Reset state" → Run workflow. |

---

## Key files to know immediately

| File | What's in it |
|---|---|
| `src/sports/registry.py` | REGISTRY dict — the single source of truth for all sports and their flags |
| `src/sports/base.py` | SportCapabilities dataclass + Sport Protocol |
| `src/sports/{sport}.py` | One module per sport — fetch/context/analyze/props/settle |
| `src/main.py` | `run()` function — registry-driven loop, state management, report build, shadow log + calibration + cap state hooks |
| `src/config/base.py` | Shared constants: `MIN_EDGE`, `DAILY_BUDGET`, sport API keys, etc. |
| `src/config/{sport}.py` | Per-sport model constants (home advantage, spread std, weights, etc.) |
| `src/models/edge_finder.py` | All `analyze_{sport}_game()` functions + `_apply_credibility_cap_dispatched()` (mode-aware) + `_stamp_recs_calibration()` |
| `src/state/shadow_log.py` | Append/update-only pick log → calibration data (month-sharded). `record_picks()`, `settle_from_history()`. |
| `src/state/calibration.py` | Auto-phase calibration engine. `effective_edge(rec)` is the slot-ranking chokepoint. `persist_state()` writes the panel snapshot. |
| `src/state/cap_state.py` | Self-tuning cap values + mode upgrades (hard clip → tanh saturation). `get_current_cap()`, `get_current_cap_mode()`, `evaluate_and_adjust_caps()`. |
| `src/report/templates/report.html` | ~3200-line template — read with offset/limit, never all at once. Includes "Model Calibration Status" + "Credibility Cap Auto-Adjustment" panels at bottom. |
| `state/picks_YYYY-MM-DD.json` | Today's locked picks |
| `state/shadow_log/YYYY-MM.json` | Monthly sharded log of every produced pick + cap flags + outcome |
| `state/calibration_state.json` | Per-(sport, market_type) phase + ratios snapshot (panel reads this) |
| `state/cap_state.json` | Per-sport current cap value, current mode, adjustment history |
| `state/watchlist_history.json` | All-time watchlist results (NHL/IPL/WNBA/MLS) |
| `state/watchlist_pending.json` | Rolling IPL picks in-progress |
| `backfill_shadow_log.py` | One-time root script — imports historical settled picks into shadow log |
| `docs/DEVELOPMENT_PLAN.md` | Feature roadmap + model change log |

---

## Config file — which one to use

There are TWO config locations. Always use `src/config/{sport}.py`, never `src/config.py`:

| Path | Status | What's in it |
|---|---|---|
| `src/config/` (folder) | ✅ **Active** | Per-sport constant files (`nba.py`, `mlb.py`, etc.) + `base.py` for shared constants. `__init__.py` re-exports everything so `from src.config import X` works everywhere. |
| `src/config.py` (file) | ❌ **Dead / ignored** | Original single-file config kept from before the refactor. Python silently ignores it because the `src/config/` package takes priority. Do NOT edit this file — changes here have no effect. |

**Rule:** When adding a new constant for an existing sport, add it to `src/config/{sport}.py`. When adding a new sport, create `src/config/{sport}.py` and add the import to `src/config/__init__.py`. Never touch `src/config.py`.

---

## Self-calibration system (added May 2026)

A fully autonomous feedback loop. **Do not work around it manually** — it self-tunes within hard safety bounds. The user expects "set and forget".

| Concept | Where | Quick reference |
|---|---|---|
| Shadow log | `src/state/shadow_log.py` + `state/shadow_log/YYYY-MM.json` | Stable key = `date\|sport\|game\|market_type\|pick_side`. Idempotent. Game-locked after `commence_time`. |
| Calibration phases | `src/state/calibration.py` | Phase 0 (n<100, no adjustment) → A (single ratio) → B (4-bucket) → C (reserved). Auto-promoted per `(sport, market_type)`. |
| Effective edge | `effective_edge(rec)` | The slot-ranking chokepoint in `_slot_sort_key`. `raw_edge × calibration_ratio`. Always 1.0 ratio in Phase 0. |
| Cap auto-tuning | `src/state/cap_state.py` | Counterfactual: cap-fired entries → `raw_mae` vs `capped_mae`. Widen / tighten / keep. Bounded [0.05, 0.30]. Throttled 30 days per cap. |
| Cap mode upgrades | `_apply_credibility_cap_dispatched()` | Mode 0 = hard clip, Mode 1 = tanh saturation. Promoted to Mode 1 when ≥ 200 firings show tanh MAE < hard MAE by ≥ 1.5%. Throttled 60 days. |
| Analyzer integration | `_stamp_recs_calibration()` in `edge_finder.py` | Called at end of every `analyze_*_game()` to stamp raw_prob + cap trigger flags onto each pick. |
| Backfill | `backfill_shadow_log.py` | One-time root script. Already run once — re-running is idempotent (skips existing keys). |

**Key invariants — do not break:**
- Every shadow log / calibration / cap call site in `main.py` is wrapped in try/except — failure must never block the report.
- `get_current_cap()` always falls back to the constant defined in `edge_finder.py` (e.g. `NBA_CRED_CAP`) if cap_state is unavailable.
- Hard safety bounds (`CAP_MIN=0.05`, `CAP_MAX=0.30`) override any counterfactual recommendation.
- Phase 0 behaviour must be byte-identical to pre-deployment (identity adjustment, raw edges used).

Full deep reference: see "Self-Calibration System" section in the project memory file.

---

## Critical rules
- `report.html` is ~3100 lines. Always use `offset` and `limit` when reading it — never read the whole file.
- The `today` parameter in sport modules is always a **string** `"YYYY-MM-DD"`. Stats functions need `datetime.date` — always convert with `date.fromisoformat(today)` inside the module.
- Never change main.py's analysis loop to add per-sport special cases — add a capability flag to the registry instead.
- `morning_singles_display` and `morning_{slug}_display` are **write-once** — never update them on subsequent runs.
- IPL game count is set by the pending file, not the odds API. After the IPL pending section, `game_counts["ipl"]` is synced explicitly.
- Watchlist tile (`wl-tile-IPL` etc.) `data-hist-won/lost` already includes **all** settled picks including today's. The JS `updateWatchlistTiles()` skips cards with `data-espn-done="1"` (server-settled) to avoid double-counting. Only live ESPN-polled resolutions (new results during the user's session) increment the tile beyond the baseline.
- `load_watchlist_performance()` dedup key is `(date, game, pick)` — three picks on the same game (ML + spread + total) all count separately. Do not change it to `(date, game)`.
