"""The PWA showed yesterday's card after today's notification (Oct 21 2026).

User report: "I got the notification that 7 bets were found today but the PWA
still shows yesterday's results."

The server side was fine — verified live: index_spa.html published at 15:32 with
TODAY_DATE 2026-10-21 and 30 picks, and the service worker's fetch handler was
already network-first. Two client-side causes compounded:

1. PUSH NOTIFICATIONS ARE HANDLED BY THE SERVICE WORKER, which runs
   independently of any open page. So the alert fires while the installed app,
   suspended in the background since yesterday, still holds yesterday's DOM in
   memory. Resuming it does not re-navigate and nothing re-fetches — the only
   refresh anywhere in the app was the manual pull-to-refresh gesture.

2. GitHub Pages serves the report with `cache-control: max-age=600`, so even a
   genuine reload could be answered from the browser's HTTP cache without
   reaching the network. The service worker looked network-first while actually
   returning a stale page.
"""
import re

import pytest


def _js():
    with open("src/report/templates/_report_js.html") as f:
        return f.read()


def _sw():
    with open("docs/sw.js") as f:
        return f.read()


class TestResumeGuard:
    def test_reloads_on_resume(self):
        js = _js()
        assert "visibilitychange" in js
        assert "_reloadIfStale" in js

    def test_also_covers_bfcache_restores(self):
        """iOS back-swipe / app resume does not fire visibilitychange."""
        js = _js()
        assert "pageshow" in js
        assert "ev.persisted" in js

    def test_compares_against_pacific_date(self):
        """The run date is Pacific; comparing against UTC would fire a spurious
        reload every evening."""
        js = _js()
        assert "America/Los_Angeles" in js
        assert "en-CA" in js            # yields YYYY-MM-DD to match TODAY_DATE

    def test_has_a_loop_guard(self):
        """Opening at 6am, before the 9am run, is legitimately 'yesterday' and
        reloading cannot fix it — so it must not spin."""
        js = _js()
        block = js.split("function _reloadIfStale")[1][:800]
        assert "sessionStorage" in block
        assert "5 * 60 * 1000" in block

    def test_never_throws(self):
        """Storage can be blocked; the guard must not break the page."""
        block = _js().split("function _reloadIfStale")[1][:800]
        assert "try {" in block and "catch" in block

    def test_unsupported_timezone_does_not_reload(self):
        block = _js().split("function _pacificToday")[1][:400]
        assert "return TODAY_DATE" in block    # equal to page date -> no reload


class TestServiceWorkerDoesNotRewrapNavigations:
    """⚠️ A no-store rewrap of navigations BROKE THE PWA (Oct 21-22 2026).

    Safari refused every launch with "Response served by service worker has
    redirections". Cause: Cloudflare Pages 301-redirects /index_spa.html ->
    /index_spa, and the notification opens the .html URL. fetch(url, ...)
    defaults to redirect:'follow', so the response carries redirected === true —
    and the spec forbids a service worker returning a redirected response to a
    NAVIGATION request. Passing e.request through preserves the navigation's own
    redirect mode.

    The premise was wrong too: this host already sends
    `cache-control: public, max-age=0, must-revalidate`, so there was no HTTP
    cache staleness to defeat. The real fix is the in-page resume guard above.
    """

    def test_navigations_are_not_refetched_by_url(self):
        sw = _sw()
        assert "cache: 'no-store'" not in sw
        assert "fetch(e.request.url" not in sw

    def test_the_original_request_is_passed_through(self):
        assert "fetch(e.request)" in _sw()

    def test_the_reason_is_recorded_so_it_is_not_retried(self):
        sw = _sw()
        assert "redirections" in sw
        assert "redirect" in sw

    def test_still_network_first_with_offline_fallback(self):
        sw = _sw()
        assert sw.index("fetch(e.request)") < sw.index("caches.match(e.request)")
        assert ".catch(" in sw


class TestCacheKeyStaysRunUnique:
    def test_key_carries_a_timestamp(self):
        """A date-only key is republished identically by later runs, so the
        activate handler never clears the earlier copy."""
        assert re.search(r"const CACHE = 'picks-[\d-]+", _sw())

    def test_writer_still_stamps_it(self):
        src = open("src/main.py").read()
        assert "def _write_sw(" in src
        assert "%H%M%S" in src
