"""
Snapshot tests for core model functions.

Purpose: verify that refactoring phases do not change model behavior.
These tests use fixed inputs and assert fixed outputs. If a test fails
after a refactor, behavior changed — investigate before merging.

Run with: python -m pytest tests/test_snapshots.py -v
"""

import math
import pytest


# ---------------------------------------------------------------------------
# Pure math — edge_finder internals
# ---------------------------------------------------------------------------

class TestScheduleLoadPenalty:
    def test_no_penalty_below_threshold(self):
        from src.models.edge_finder import _schedule_load_penalty
        assert _schedule_load_penalty(0) == 0.0
        assert _schedule_load_penalty(4) == 0.0

    def test_penalty_at_5_games(self):
        from src.models.edge_finder import _schedule_load_penalty
        assert _schedule_load_penalty(5) == 0.01

    def test_penalty_at_6_games(self):
        from src.models.edge_finder import _schedule_load_penalty
        assert _schedule_load_penalty(6) == 0.02

    def test_penalty_at_7_games(self):
        from src.models.edge_finder import _schedule_load_penalty
        assert _schedule_load_penalty(7) == 0.03

    def test_penalty_above_7_games(self):
        from src.models.edge_finder import _schedule_load_penalty
        assert _schedule_load_penalty(10) == 0.03


class TestConfidenceLabel:
    def test_high_requires_all_three(self):
        from src.models.edge_finder import _confidence_label
        # All three conditions met
        assert _confidence_label(0.07, 3, True) == "HIGH"
        assert _confidence_label(0.10, 5, True) == "HIGH"

    def test_medium_if_edge_too_low(self):
        from src.models.edge_finder import _confidence_label
        assert _confidence_label(0.06, 3, True) == "MEDIUM"

    def test_medium_if_signals_too_few(self):
        from src.models.edge_finder import _confidence_label
        assert _confidence_label(0.07, 2, True) == "MEDIUM"

    def test_medium_if_no_stats(self):
        from src.models.edge_finder import _confidence_label
        assert _confidence_label(0.07, 3, False) == "MEDIUM"


class TestNbaMarginToProb:
    def test_zero_margin_is_50_pct(self):
        from src.models.edge_finder import _nba_margin_to_prob
        result = _nba_margin_to_prob(0.0)
        assert abs(result - 0.5) < 1e-9

    def test_positive_margin_above_50(self):
        from src.models.edge_finder import _nba_margin_to_prob
        result = _nba_margin_to_prob(5.0)
        assert result > 0.5

    def test_negative_margin_below_50(self):
        from src.models.edge_finder import _nba_margin_to_prob
        result = _nba_margin_to_prob(-5.0)
        assert result < 0.5

    def test_known_value(self):
        # NBA_SPREAD_STD = 12.0; norm.cdf(6, 0, 12) ≈ 0.6915
        from src.models.edge_finder import _nba_margin_to_prob
        result = _nba_margin_to_prob(6.0)
        assert abs(result - 0.6915) < 0.001


class TestPitcherQualityScore:
    def test_league_average_pitcher_scores_zero(self):
        from src.models.edge_finder import _pitcher_quality_score
        # ERA = league average (4.20), xFIP = None → base = fip = 4.20
        stats = {"innings_pitched": 60.0, "fip": 4.20, "xfip": None}
        result = _pitcher_quality_score(stats)
        assert abs(result) < 1e-6

    def test_elite_pitcher_scores_positive(self):
        from src.models.edge_finder import _pitcher_quality_score
        stats = {"innings_pitcher": 80.0, "fip": 2.80, "xfip": 2.90, "innings_pitched": 80.0}
        result = _pitcher_quality_score(stats)
        assert result > 0

    def test_bad_pitcher_scores_negative(self):
        from src.models.edge_finder import _pitcher_quality_score
        stats = {"innings_pitched": 60.0, "fip": 5.50, "xfip": 5.20}
        result = _pitcher_quality_score(stats)
        assert result < 0

    def test_small_sample_blends_toward_average(self):
        from src.models.edge_finder import _pitcher_quality_score
        # Small-sample xFIP regression is now handled by _blend_xfip before
        # _pitcher_quality_score is called. Test the contract: when xFIP has
        # already been regressed to league average (4.20) at 0 IP, score = 0.
        stats = {"innings_pitched": 0.0, "fip": 4.20, "xfip": 4.20}
        result = _pitcher_quality_score(stats)
        assert abs(result) < 1e-6

    def test_small_sample_k9_bb9_regressed(self):
        from src.models.edge_finder import _pitcher_quality_score
        # At tiny IP, effective K/9 and BB/9 should be pulled toward league avg,
        # so a pitcher with elite peripherals from 9 IP scores below full-confidence.
        tiny_ip = {"era": 6.00, "xfip": 3.80, "fip": 2.00,
                   "bb_per_9": 2.0, "k_per_9": 9.0, "innings_pitched": 9.0}
        full_ip  = {"era": 6.00, "xfip": 3.80, "fip": 2.00,
                    "bb_per_9": 2.0, "k_per_9": 9.0, "innings_pitched": 80.0}
        assert _pitcher_quality_score(tiny_ip) < _pitcher_quality_score(full_ip)


class TestEraTrapSeverity:
    def test_no_era_returns_zero(self):
        from src.models.edge_finder import _era_trap_severity
        stats = {"fip": 4.0, "innings_pitched": 40.0}
        assert _era_trap_severity(stats) == 0.0

    def test_below_10_ip_returns_zero(self):
        from src.models.edge_finder import _era_trap_severity
        stats = {"era": 2.80, "fip": 4.20, "xfip": 4.10, "innings_pitched": 8.0}
        assert _era_trap_severity(stats) == 0.0

    def test_era_above_fip_returns_zero(self):
        # ERA >= FIP means pitcher is underperforming, not outperforming — no trap
        from src.models.edge_finder import _era_trap_severity
        stats = {"era": 4.50, "fip": 4.00, "xfip": 4.10, "innings_pitched": 40.0}
        assert _era_trap_severity(stats) == 0.0

    def test_clear_trap_returns_nonzero(self):
        from src.models.edge_finder import _era_trap_severity
        stats = {
            "era": 2.50, "fip": 4.00, "xfip": 4.20,
            "innings_pitched": 40.0, "babip": 0.250, "k_per_9": 7.0
        }
        result = _era_trap_severity(stats)
        assert result > 0.15

    def test_high_k9_reduces_severity(self):
        from src.models.edge_finder import _era_trap_severity
        base_stats = {
            "era": 2.50, "fip": 4.00, "xfip": 4.20,
            "innings_pitched": 40.0, "babip": 0.270
        }
        low_k9 = {**base_stats, "k_per_9": 7.0}
        high_k9 = {**base_stats, "k_per_9": 12.0}
        assert _era_trap_severity(high_k9) < _era_trap_severity(low_k9)

    def test_elite_pitcher_capped_at_moderate(self):
        # xFIP < 3.20 → severity capped at 0.79
        from src.models.edge_finder import _era_trap_severity
        stats = {
            "era": 1.50, "fip": 3.00, "xfip": 3.00,
            "innings_pitched": 60.0, "babip": 0.200, "k_per_9": 7.0
        }
        result = _era_trap_severity(stats)
        assert result <= 0.79


class TestMlbConf:
    def test_high_with_no_traps(self):
        from src.models.edge_finder import _mlb_conf
        assert _mlb_conf(0.07, 3, True, 0.0, 0.0) == "HIGH"

    def test_medium_if_own_severe_trap(self):
        from src.models.edge_finder import _mlb_conf
        # Own pitcher is in an ERA trap — cap at MEDIUM
        assert _mlb_conf(0.07, 3, True, 0.85, 0.0) == "MEDIUM"

    def test_medium_if_edge_too_low(self):
        from src.models.edge_finder import _mlb_conf
        assert _mlb_conf(0.04, 3, True, 0.0, 0.0) == "MEDIUM"


class TestPoissonProb:
    def test_zero_goals(self):
        from src.models.edge_finder import _poisson_prob
        # P(k=0 | lam=1.5) = e^-1.5 ≈ 0.2231
        result = _poisson_prob(1.5, 0)
        assert abs(result - math.exp(-1.5)) < 1e-9

    def test_known_value(self):
        # P(k=2 | lam=2.0) = e^-2 * 2^2 / 2! = e^-2 * 2 ≈ 0.2707
        from src.models.edge_finder import _poisson_prob
        result = _poisson_prob(2.0, 2)
        expected = math.exp(-2.0) * (2.0 ** 2) / math.factorial(2)
        assert abs(result - expected) < 1e-9


class TestMlsProbMatrix:
    """
    _mls_prob_matrix returns {(i, j): probability} where i = home goals,
    j = away goals. Derived metrics (home_win, draw, etc.) are computed
    by summing over the matrix in analyze_mls_game().
    """
    def test_probabilities_sum_to_one(self):
        from src.models.edge_finder import _mls_prob_matrix
        matrix = _mls_prob_matrix(1.35, 1.15)
        total = sum(matrix.values())
        assert abs(total - 1.0) < 1e-6

    def test_all_keys_are_tuples(self):
        from src.models.edge_finder import _mls_prob_matrix
        matrix = _mls_prob_matrix(1.35, 1.15)
        assert all(isinstance(k, tuple) and len(k) == 2 for k in matrix)

    def test_draw_probability_in_range(self):
        # Draw = sum of (i, i) cells; league-typical ~25-30%
        from src.models.edge_finder import _mls_prob_matrix
        matrix = _mls_prob_matrix(1.35, 1.15)
        draw_prob = sum(v for (i, j), v in matrix.items() if i == j)
        assert 0.22 < draw_prob < 0.32

    def test_strong_home_team_favored(self):
        from src.models.edge_finder import _mls_prob_matrix
        matrix = _mls_prob_matrix(2.5, 0.8)
        home_win = sum(v for (i, j), v in matrix.items() if i > j)
        away_win = sum(v for (i, j), v in matrix.items() if j > i)
        assert home_win > away_win

    def test_equal_teams_near_even(self):
        from src.models.edge_finder import _mls_prob_matrix
        matrix = _mls_prob_matrix(1.5, 1.5)
        home_win = sum(v for (i, j), v in matrix.items() if i > j)
        away_win = sum(v for (i, j), v in matrix.items() if j > i)
        assert abs(home_win - away_win) < 0.01


# ---------------------------------------------------------------------------
# Kelly model
# ---------------------------------------------------------------------------

class TestKelly:
    def test_no_edge_returns_zero_size(self):
        from src.models.kelly import robinhood_kelly
        # market_prob = model_prob → no edge → zero bet
        result = robinhood_kelly(0.55, 0.55, 100.0)
        assert result.num_contracts == 0

    def test_positive_edge_returns_nonzero(self):
        from src.models.kelly import robinhood_kelly
        result = robinhood_kelly(0.60, 0.50, 100.0)
        assert result.num_contracts > 0

    def test_negative_edge_returns_zero(self):
        from src.models.kelly import robinhood_kelly
        # model_prob < market_prob → negative edge → skip
        result = robinhood_kelly(0.40, 0.55, 100.0)
        assert result.num_contracts == 0

    def test_budget_constraint_respected(self):
        from src.models.kelly import robinhood_kelly
        result = robinhood_kelly(0.75, 0.50, 10.0)
        # total cost must not exceed budget
        assert result.total_cost <= 10.0 + 0.01  # small float tolerance


# ---------------------------------------------------------------------------
# Performance summary — load from fixture, verify structure
# ---------------------------------------------------------------------------

class TestLoadPerformanceSummaryStructure:
    """
    Verifies load_performance_summary() returns the expected shape.
    Does not assert exact values (those depend on history.json content)
    but does assert every key the report template reads is present.
    """
    def test_returns_expected_keys(self, tmp_path, monkeypatch):
        import json

        # Write a minimal history fixture matching the real schema:
        # "result" (not "outcome"), "actual_pnl" (not "pnl"), "cost" (not "total_cost")
        history = [
            {
                "date": "2025-01-01", "sport": "NBA", "game": "Team A @ Team B",
                "bet_type": "Moneyline", "pick": "Team A",
                "market_prob": 0.55, "model_prob": 0.62,
                "edge": 0.07, "confidence": "HIGH",
                "contract_price": 0.55, "num_contracts": 3,
                "cost": 1.65, "profit_if_win": 1.35,
                "result": "WON", "actual_pnl": 1.35
            },
            {
                "date": "2025-01-02", "sport": "MLB", "game": "Team C @ Team D",
                "bet_type": "Moneyline", "pick": "Team C",
                "market_prob": 0.52, "model_prob": 0.59,
                "edge": 0.07, "confidence": "MEDIUM",
                "contract_price": 0.52, "num_contracts": 2,
                "cost": 1.04, "profit_if_win": 0.96,
                "result": "LOST", "actual_pnl": -1.04
            },
        ]

        history_path = tmp_path / "history.json"
        history_path.write_text(json.dumps(history))

        # Patch HISTORY_PATH (computed at import time from HISTORY_FILE)
        from pathlib import Path
        import src.data.outcome_checker as oc
        monkeypatch.setattr(oc, "HISTORY_PATH", history_path)

        result = oc.load_performance_summary()

        # Must have the top-level keys the report template reads
        assert "total" in result
        assert "won" in result
        assert "lost" in result
        assert "win_rate_pct" in result
        assert "total_pnl" in result
        assert "roi_pct" in result
        assert "by_confidence" in result
        assert "by_sport" in result
        assert "all_records" in result

        # Spot-check values match our fixture (1 WON, 1 LOST)
        assert result["total"] == 2
        assert result["won"] == 1
        assert result["lost"] == 1


class TestInjuryValueWeighting:
    """Injuries weight by player value (value_mult), not position alone."""

    def test_star_injury_docks_more_than_bench(self):
        from src.data.injuries import injury_adjustment, INJURY_DRAG_CAP
        star = {"T": [{"player": "Star", "position": "RF", "status": "out", "value_mult": 3.0}]}
        bench = {"T": [{"player": "Depth", "position": "RF", "status": "out", "value_mult": 0.4}]}
        a_star = injury_adjustment("T", star, "mlb")
        a_bench = injury_adjustment("T", bench, "mlb")
        assert a_star > a_bench
        assert abs(a_star - 0.030) < 1e-9   # trips _INJURY_GATE (0.030)
        assert INJURY_DRAG_CAP == 0.10

    def test_unenriched_injury_matches_prior_behavior(self):
        from src.data.injuries import injury_adjustment
        # No value_mult key → defaults to 1.0 → position-only (old) behaviour
        old = {"T": [{"player": "X", "position": "RF", "status": "out"}]}
        assert abs(injury_adjustment("T", old, "mlb") - 0.010) < 1e-9


class TestInjuryCompoundCombination:
    """Multiple injuries combine via compound (1-∏(1-p)), not naive sum."""

    def test_single_injury_unchanged_by_compound(self):
        from src.data.injuries import injury_adjustment
        one = {"T": [{"player": "A", "position": "RF", "status": "out", "value_mult": 3.0}]}
        assert abs(injury_adjustment("T", one, "mlb") - 0.030) < 1e-9

    def test_two_injuries_compound_below_sum(self):
        from src.data.injuries import injury_adjustment
        two = {"T": [
            {"player": "A", "position": "RF", "status": "out", "value_mult": 3.0},  # 0.030
            {"player": "B", "position": "RF", "status": "out", "value_mult": 3.0},  # 0.030
        ]}
        adj = injury_adjustment("T", two, "mlb")
        # compound = 1-(1-0.03)^2 = 0.0591, below naive sum 0.060
        assert abs(adj - 0.0591) < 1e-4
        assert adj < 0.060


class TestInjuryStatusNormalization:
    """MLB IL designations must map to 'out' (were being dropped entirely)."""

    def test_il_maps_to_out(self):
        from src.data.injuries import _normalize_status
        assert _normalize_status("10-Day-IL") == "out"
        assert _normalize_status("15-Day-IL") == "out"
        assert _normalize_status("IR") == "out"          # NFL injured reserve

    def test_60day_il_skipped(self):
        # 60-day IL is fully priced into team odds — skip to avoid false signal
        from src.data.injuries import _normalize_status
        assert _normalize_status("60-Day-IL") is None

    def test_known_statuses_passthrough(self):
        from src.data.injuries import _normalize_status
        assert _normalize_status("Day-To-Day") == "day-to-day"
        assert _normalize_status("Questionable") == "questionable"
        assert _normalize_status("Out") == "out"

    def test_active_or_unknown_skipped(self):
        from src.data.injuries import _normalize_status
        assert _normalize_status("Active") is None
        assert _normalize_status("") is None

    def test_il_injury_now_counts(self):
        from src.data.injuries import _parse_injuries
        raw = [{"displayName": "Team A", "injuries": [
            {"status": "60-Day-IL", "athlete": {"displayName": "Star",
             "position": {"abbreviation": "SS"}}},
        ]}]
        parsed = _parse_injuries(raw)
        assert parsed["Team A"][0]["status"] == "out"   # was dropped before
