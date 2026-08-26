"""
Health-report pulse for the exception-guarded measurement layers.

Aug 26 2026: a date-vs-ISO-string mismatch made the execution log match 0 of 5
picks and both Kalshi passes raise on entry. All three subsystems never ran in
production — and because they are deliberately wrapped in try/except so they
can never block the daily card, NOTHING surfaced it. It was found by reading a
workflow log by chance. These tests keep that failure loud.
"""
from tools.analysis import health_report as hr


def test_zero_execution_rows_is_not_ok():
    """The signature of a silent failure: the log simply never grows."""
    report = {"subsystem_liveness": {"ok": False, "execution_rows_recent": 0,
                                     "execution_rows_total": 0,
                                     "kalshi_clv_coverage_pct": 80.0},
              "bankroll": {"ok": True},
              "budget_performance": {"last_7d": {}},
              "log_liveness": {"ok": True}}
    alerts = hr.compute_alerts(report)
    assert any("execution log stalled" in a for a in alerts), alerts


def test_healthy_subsystems_raise_no_alert():
    report = {"subsystem_liveness": {"ok": True, "execution_rows_recent": 12,
                                     "execution_rows_total": 300,
                                     "kalshi_clv_coverage_pct": 80.0},
              "bankroll": {"ok": True},
              "budget_performance": {"last_7d": {}},
              "log_liveness": {"ok": True}}
    assert hr.compute_alerts(report) == []


def test_kalshi_clv_going_dark_alerts():
    """Kalshi is the PRIMARY CLV feed since Aug 25 — losing it is not a nicety."""
    report = {"subsystem_liveness": {"ok": True, "execution_rows_recent": 5,
                                     "execution_rows_total": 50,
                                     "kalshi_clv_coverage_pct": 3.0},
              "bankroll": {"ok": True},
              "budget_performance": {"last_7d": {}},
              "log_liveness": {"ok": True}}
    assert any("Kalshi CLV coverage" in a for a in hr.compute_alerts(report))


def test_clv_coverage_counts_either_feed():
    """After the Aug 25 switch the paid feed stops writing `clv`. Counting only
    that field would show coverage collapsing and fire a false alarm weekly."""
    import inspect
    src = inspect.getsource(hr.check_log_liveness)
    assert 'r.get("kalshi_clv") is not None' in src, \
        "coverage must accept the Kalshi feed too"


def test_subsystem_check_is_wired_into_the_report():
    import inspect
    src = inspect.getsource(hr.main)
    assert '"subsystem_liveness": check_subsystem_liveness()' in src


def test_check_survives_a_missing_execution_log_dir():
    """A fresh clone has no execution_log/ — that must report zero, not crash."""
    out = hr.check_subsystem_liveness()
    assert "execution_rows_recent" in out or out.get("error")
    assert isinstance(out.get("ok"), bool)
