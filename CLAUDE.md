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
| `src/main.py` | `run()` function — registry-driven loop, state management, report build |
| `src/models/edge_finder.py` | All `analyze_{sport}_game()` functions |
| `src/report/templates/report.html` | ~3100-line template — read with offset/limit, never all at once |
| `state/picks_YYYY-MM-DD.json` | Today's locked picks |
| `state/watchlist_history.json` | All-time watchlist results (NHL/IPL/WNBA/MLS) |
| `state/watchlist_pending.json` | Rolling IPL picks in-progress |
| `docs/DEVELOPMENT_PLAN.md` | Feature roadmap + model change log |

---

## Critical rules
- `report.html` is ~3100 lines. Always use `offset` and `limit` when reading it — never read the whole file.
- The `today` parameter in sport modules is always a **string** `"YYYY-MM-DD"`. Stats functions need `datetime.date` — always convert with `date.fromisoformat(today)` inside the module.
- Never change main.py's analysis loop to add per-sport special cases — add a capability flag to the registry instead.
- `morning_singles_display` and `morning_{slug}_display` are **write-once** — never update them on subsequent runs.
- IPL game count is set by the pending file, not the odds API. After the IPL pending section, `game_counts["ipl"]` is synced explicitly.
