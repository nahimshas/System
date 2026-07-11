# Sports Betting Analysis System — Claude Session Guide

## ⚠️ Only one real system exists — read this first

The sole source of truth is `/Users/nahim/Documents/GitHub/System/`.

**Stale copies that no longer exist (do not look for them):**
- `/Users/nahim/Documents/GitHub/System/sports-betting-system/` — old NBA+MLB-only prototype, deleted from git May 2026
- `/Users/nahim/Desktop/Claude Projects/Code Projects/sports-betting-system/` — same old prototype, deleted May 2026

If you are running from a working directory other than `/Users/nahim/Documents/GitHub/System`, stop and redirect to the correct path before doing anything.

---

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
Runs daily at 9am PDT (GitHub Actions). Fetches odds for NBA/MLB/NFL/NHL/IPL/WNBA/MLS/WC, runs edge models, picks top-5 singles + parlays + props, saves state, generates an HTML report, deploys to GitHub Pages, and sends an email. A nightly Results Snapshot workflow (dispatched by the user's debrief routine at ~10:50pm) resolves results AND captures closing lines (CLV) for every pick; a CLV governor automatically gates negative-CLV markets out of the budget pool.

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
| `src/data/closing_lines.py` | CLV capture — historical Odds API fetch, self-healing shadow-log stamping (`market_prob_at_close`, `clv`), credit-budgeted. See "CLV system" below. |
| `src/state/clv_governor.py` | Phase-gated budget gating per (sport, market_type) by average CLV. `clv_gate(rec)`, `persist_state()` → `state/clv_state.json` |
| `backfill_clv.py` + `clv_backfill.yml` | Throttled historical CLV backfill (workflow button, ~900 credits/run, idempotent — run until "Nothing left to backfill") |
| `state/clv_state.json` | Per-market CLV snapshot — read by the CLV panels (classic report + PWA Analytics tab) |
| `src/state/decision_log.py` | Full candidate + feature archive (both sides of every market, made + rejected) → `state/decision_log/YYYY-MM.json`. Pure analysis layer, separate from the shadow log. `record_candidates()`, `update_decision_log_clv` (in closing_lines), `settle_decision_from_scores()`. See "Decision log" below — **has a maintenance rule**. |
| `docs/debrief_routine_prompt.md` | Canonical CLV-aware nightly debrief routine prompt (credential placeholders — real tokens live only in the user's routine) |
| `tools/analysis/backtest.py` | **Canonical backtest engine** — reconstructs MLB probs from decision-log features, sims daily budget selection at realistic 4.5% vig. `--set KEY=VAL` variants, `--pattern-only`. ⚠️ Its `LIVE` dict must match shipped constants (guarded by `tests/test_backtest_live_sync.py`). NEVER rebuild backtests in scratchpads — use/extend this. |
| `tools/analysis/health_report.py` | Read-only health checks + automatic checkpoint evaluation + deterministic alerts (drawdown, log liveness, bankroll drift). Run any time: `python3 -m tools.analysis.health_report` |
| `tools/analysis/checkpoints.json` | **Pre-registered model-change evaluations** with explicit pass/fail rules + dates. ⚠️ RULE: every model change ships with a checkpoint entry here; resolved ones get `status`+`resolution`, never deleted. |
| `src/report/debrief_builder.py` | Deterministic nightly debrief page (docs/debrief_latest.html + debrief_history.json) built from the results snapshot inside `results_snapshot.yml` — verdict templates, canonical-signal regex map, ESPN-headline narratives, no-downgrade guard. The Claude routines are PAUSED (lost write access Jul 2026); workflows are the entire automation surface. |
| `docs/health_routine_prompt.md` | Canonical prompt for the (PAUSED) weekly Claude routine — publishing now lives in `.github/workflows/health_report.yml` (Sundays 8am PT + external cron) |
| `docs/DEVELOPMENT_PLAN.md` | Feature roadmap + model change log |

---

## Config file — which one to use

All config lives in the `src/config/` package: per-sport constant files (`nba.py`, `mlb.py`, etc.) + `base.py` for shared constants. `__init__.py` re-exports everything so `from src.config import X` works everywhere.

**Rule:** When adding a new constant for an existing sport, add it to `src/config/{sport}.py` and re-export it from `src/config/__init__.py`. When adding a new sport, create `src/config/{sport}.py` and add the import to `src/config/__init__.py`.

> Note: a legacy single-file `src/config.py` existed before the package refactor and was deleted May 2026 (it was shadowed by the package and never loaded). Do not recreate it.

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

## CLV system (June 2026) — closing-line value capture + governor

CLV = `market_prob_at_close − market_prob_at_first_pick` (positive = beat the close). The fastest reliable skill signal: ~50 graded picks per (sport, market) gives a verdict win/loss needs 500+ for. **Set and forget — do not work around it manually.**

| Concept | Where | Quick reference |
|---|---|---|
| Capture | `src/data/closing_lines.py` | `update_shadow_log_clv(max_credits, lookback_days/since)` — scans shadow log for entries missing `market_prob_at_close`, fetches Odds API **historical** snapshots at each game's commence_time (10 credits/market/region; one snapshot covers a whole sport wave), stamps `clv`. Self-healing + idempotent; `clv_fetch_attempts` capped at 3. |
| Nightly run | `results_snapshot.py` (+ workflow) | Capture runs inside the snapshot the debrief routine dispatches; each resolved pick carries `clv`/`clv_pct`. Workflow needs `ODDS_API_KEY` secret and commits `state/shadow_log/`. |
| Morning self-heal | `main.py` (before cap auto-adjust) | 400-credit budget, 7-day lookback — repairs failed nights. Late data, never lost data. |
| Governor | `src/state/clv_governor.py` + `_clv_gate_safe()` in `main.py` | Phase 0 (n<30) observe → Phase 1 (30–49) gate avg ≤ −2% → Phase 2 (≥50) gate avg ≤ −1%. Gates BUDGET routing only — display/watchlist/shadow log untouched, so markets recover automatically. Fail-open everywhere. First gate fired Jun 11 2026: MLB Total (−2.28% over 37). |
| Panels | classic `#sec-clv` + PWA Analytics `sect-clv` | Both read `state/clv_state.json` via `_load_clv_panel()` in `generator.py`. |

**Consistency rule:** closing probs use the SAME no-vig consensus math as the morning pipeline. Never source closing lines from web search or any other feed — apples-to-apples or the CLV signal is corrupted.

**Workflow commit pattern:** in any workflow that commits state, the order MUST be `git add` → `git commit` → `git pull --rebase` → `git push`. Pull-before-add fails with "unstaged changes" whenever the debrief routine commits concurrently (bit both new workflows on day one).

---

## Decision log (June 2026) — full candidate + feature archive

The "log everything so we can improve later" layer. `src/state/decision_log.py` + `state/decision_log/YYYY-MM.json`. **SEPARATE from the shadow log** (which feeds calibration/caps and must stay clean) — the decision log is a pure analysis archive that NEVER touches display or calibration. Month-sharded, idempotent (stable key `date|sport|game|market|side`), game-locked, exception-safe — same safety patterns as the shadow log.

Captures, for EVERY analyzed game, BOTH sides of EVERY market (made **and rejected**) + the model inputs behind each probability. This is what makes exhaustive CLV analysis possible: measuring CLV/outcome for picks the model REJECTED, segmented by any input.

| Concept | Where | Quick reference |
|---|---|---|
| Capture | `edge_finder._stamp_decision(game, min_edge, features, markets, recs)` | Called before `return recs` in every analyzer. `markets` = 6-tuples `(market_type, side, model_prob_POST, model_prob_RAW, market_prob, line)`. Stamps `game["_decision"]={"features":{...},"candidates":[...]}`. |
| Record | `main.py` loop → `decision_log.record_candidates(...)` | Per analyzed game; preserves `market_prob_at_first_pick` across re-runs. |
| Candidate CLV | `closing_lines.update_decision_log_clv()` | Mirrors `update_shadow_log_clv` over the decision log. `_SNAPSHOT_CACHE` reuses the shadow pass's snapshots → overlapping waves cost 0 extra credits. Runs after the shadow CLV pass in `main.py` + `results_snapshot.py`. |
| Candidate outcomes | `decision_log.settle_decision_from_scores(today)` | Grades both sides from ESPN final scores. `_grade_candidate()` uses structured market/side/line; soccer ML draw = LOSS; soccer uses 90-min score. IPL skipped (no reliable historical score). Wired in `main.py` after shadow settlement. |
| Row schema | `state/decision_log/*.json` | market_prob (open/last/close), model_prob (post-cap), model_prob_raw (pre-cap, both sides), edge, raw_edge, made, line, final_confidence_label (made only), clv, outcome, + features dict. |

**⚠️ MAINTENANCE RULE — part of "done" for ANY model change:** whenever you add or change a model input/variable/signal for any sport or bet type, ALSO add it to that sport's `_stamp_decision(...)` `features` dict. Any NEW market/bet type must be added to that sport's `candidates` list (both sides) INCLUDING its pre-cap raw prob as the 4th tuple element. Skip this and the input/market becomes permanently invisible to future analysis (no backfill possible — a feature only has history from the day it's wired in). Decision-log data starts Jun 13 2026; nothing before exists.

**Re-run safety:** stable keys mean every mode (reset / code-only / full re-run) UPDATES rows, never duplicates. `code_only` returns before any analysis → touches neither log. `reset_state` deletes only `state/picks_TODAY.json` (not the logs). `market_prob_at_first_pick` is preserved across all re-runs, and started games are frozen (game-lock + the odds API dropping them), so the at-pick-time snapshot is never corrupted.

---

## Calibrated sizing + game types + soccer knockouts (June 2026)

- **Calibrated Kelly sizing** — `_resize_with_calibrated_probs()` in `main.py` (runs right after `_recalibrate_confidence`): Kelly stakes are recomputed on `market_prob + effective_edge` (the calibration-corrected belief), stamped as `rec.model_prob_calibrated`. Parlay legs use it too (`parlay_builder.py`). Phase 0 = byte-identical sizing. Watchlist zero-sizing recs untouched. Budget routing additionally requires `sizing.num_contracts > 0` (mirrors analyzer creation gate). Cards still DISPLAY the raw model prob — only stakes change.
- **Per-game season types** — `src/data/game_types.py` stamps `season_game_type` (`exhibition`/`regular`/`play_in`/`postseason`/`superbowl`) onto game dicts from ESPN's per-event `season.type` (verified: 1 pre, 2 regular, 3 post, 5 play-in; Super Bowl via competition notes headline). Wired in `main.py` loop after `fetch_games`; exhibition games are SKIPPED entirely. Analyzers use `_game_playoff(game, _is_x_playoff)` — per-game flag with the legacy calendar windows as FALLBACK (do not delete them). Super Bowl zeroes NFL home advantage. `rec.game_type` flows into shadow log entries (`game_type` field) so CLV/calibration can segment by type — finer play-in/round distinctions wait for that data. IPL/WC keep their existing date-based stage logic.
- **Soccer 90-minute settlement** — `_soccer_90min_scores()` in `outcome_checker.py`, applied to MLS/WC in `_fetch_watchlist_final_scores` AND `results_snapshot._fetch_events`: knockout games decided in ET/pens are graded on the 90-minute score (sum of the first two half linescores). If a game went past regulation and halves are unavailable → pick stays PENDING (never grade against the post-ET final). WC knockouts start Jun 28 2026; same code covers MLS Cup playoffs in November.

---

## Critical rules
- `report.html` is ~3100 lines. Always use `offset` and `limit` when reading it — never read the whole file.
- The `today` parameter in sport modules is always a **string** `"YYYY-MM-DD"`. Stats functions need `datetime.date` — always convert with `date.fromisoformat(today)` inside the module.
- Never change main.py's analysis loop to add per-sport special cases — add a capability flag to the registry instead.
- `morning_singles_display` and `morning_{slug}_display` are **write-once** — never update them on subsequent runs.
- IPL game count is set by the pending file, not the odds API. After the IPL pending section, `game_counts["ipl"]` is synced explicitly.
- Watchlist tile (`wl-tile-IPL` etc.) `data-hist-won/lost` already includes **all** settled picks including today's. The JS `updateWatchlistTiles()` skips cards with `data-espn-done="1"` (server-settled) to avoid double-counting. Only live ESPN-polled resolutions (new results during the user's session) increment the tile beyond the baseline.
- `load_watchlist_performance()` dedup key is `(date, game, pick)` — three picks on the same game (ML + spread + total) all count separately. Do not change it to `(date, game)`.
- **Model-change protocol (Jul 2026):** any model constant/input/signal change ALSO requires (1) a pre-registered entry in `tools/analysis/checkpoints.json` with explicit pass/fail rules + evaluation date, (2) updating the `LIVE` dict in `tools/analysis/backtest.py` (the sync test fails otherwise), and (3) the decision-log feature stamping per the existing maintenance rule. Backtests/analyses always go through `tools/analysis/backtest.py` — never rebuild the methodology in scratchpads.
