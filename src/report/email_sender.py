"""Sends the daily HTML report via Gmail SMTP."""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date
from src.config import EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587

# ── CSS variable → actual hex ─────────────────────────────────────────────────
# Email clients don't support CSS custom properties, so we replace every
# var(--x) with its real value before sending.
_CSS_VARS = {
    "var(--bg)":         "#0f1117",
    "var(--card)":       "#1a1d27",
    "var(--border)":     "#2d3148",
    "var(--text)":       "#e2e8f0",
    "var(--muted)":      "#94a3b8",
    "var(--green)":      "#16a34a",
    "var(--green-dim)":  "#166534",
    "var(--yellow)":     "#eab308",
    "var(--yellow-dim)": "#713f12",
    "var(--red)":        "#ef4444",
    "var(--blue)":       "#3b82f6",
    "var(--blue-dim)":   "#1e3a5f",
    "var(--purple)":     "#a855f7",
    "var(--purple-dim)": "#3b0764",
    "var(--teal)":       "#14b8a6",
    "var(--teal-dim)":   "#134e4a",
}

# ── Inline-style injections ───────────────────────────────────────────────────
# Gmail strips <style> blocks, so class-based rules don't apply.
# We surgically add style="" to every element that has contrast-sensitive colors.
# Format: (exact_string_in_rendered_html, replacement_with_inline_style)
_INLINE_PATCHES = [

    # ── Body / container ──────────────────────────────────────────────────────
    (
        "<body>",
        '<body bgcolor="#0f1117" style="background-color:#0f1117;color:#e2e8f0;'
        'margin:0;padding:0;font-family:-apple-system,BlinkMacSystemFont,\'Segoe UI\',sans-serif;">',
    ),
    (
        'class="container"',
        'class="container" style="max-width:680px;margin:0 auto;padding:16px;'
        'background-color:#0f1117;color:#e2e8f0;"',
    ),

    # ── Cards ─────────────────────────────────────────────────────────────────
    (
        'class="card"',
        'class="card" style="background-color:#1a1d27;border:1px solid #2d3148;'
        'border-radius:12px;padding:16px;margin-bottom:12px;"',
    ),
    (
        'class="card parlay-card"',
        'class="card parlay-card" style="background-color:#1a1d27;border:1px solid #a855f7;'
        'border-radius:12px;padding:16px;margin-bottom:12px;"',
    ),
    (
        'class="card prop-card"',
        'class="card prop-card" style="background-color:#1a1d27;'
        'border-top:1px solid #2d3148;border-right:1px solid #2d3148;'
        'border-bottom:1px solid #2d3148;border-left:3px solid #eab308;'
        'border-radius:12px;padding:16px;margin-bottom:12px;"',
    ),

    # ── Confidence badges — white text for guaranteed contrast ────────────────
    (
        'class="confidence-badge HIGH"',
        'class="confidence-badge HIGH" style="display:inline-block;'
        'background-color:#166534;color:#ffffff;font-size:11px;font-weight:700;'
        'padding:3px 10px;border-radius:99px;white-space:nowrap;"',
    ),
    (
        'class="confidence-badge MEDIUM"',
        'class="confidence-badge MEDIUM" style="display:inline-block;'
        'background-color:#92400e;color:#ffffff;font-size:11px;font-weight:700;'
        'padding:3px 10px;border-radius:99px;white-space:nowrap;"',
    ),

    # ── Stat boxes (where "Over", model prob, edge etc. live) ─────────────────
    (
        'class="stat-box"',
        'class="stat-box" style="background-color:#0f1117;border-radius:8px;'
        'padding:8px 10px;text-align:center;"',
    ),

    # ── Action box (buy contracts / profit block) ─────────────────────────────
    (
        'class="action-box"',
        'class="action-box" style="background-color:#1e3a5f;border:1px solid #3b82f6;'
        'border-radius:8px;padding:12px;margin-top:10px;"',
    ),

    # ── Prop note — white text on deeper amber; was yellow-on-yellow ──────────
    (
        'class="prop-note"',
        'class="prop-note" style="background-color:#92400e;border-radius:6px;'
        'padding:8px 10px;font-size:13px;margin-top:8px;color:#ffffff;"',
    ),

    # ── Parlay action box ─────────────────────────────────────────────────────
    (
        'class="action-box" style="background:var(--purple-dim);border-color:var(--purple);"',
        'class="action-box" style="background-color:#3b0764;border:1px solid #a855f7;'
        'border-radius:8px;padding:12px;margin-top:10px;"',
    ),

    # ── Parlay leg boxes ──────────────────────────────────────────────────────
    (
        'class="parlay-leg"',
        'class="parlay-leg" style="background-color:#0f1117;border-radius:6px;'
        'padding:8px 10px;margin-bottom:6px;font-size:14px;"',
    ),

    # ── Signal items ──────────────────────────────────────────────────────────
    (
        'class="signal-item"',
        'class="signal-item" style="font-size:13px;padding:3px 0 3px 12px;'
        'border-left:2px solid #3b82f6;margin-bottom:4px;line-height:1.4;color:#e2e8f0;"',
    ),

    # ── Research items ────────────────────────────────────────────────────────
    (
        'class="research-item"',
        'class="research-item" style="font-size:13px;padding:3px 0 3px 12px;'
        'border-left:2px solid #134e4a;margin-bottom:4px;line-height:1.4;color:#94a3b8;"',
    ),

    # ── Game time (teal) — ensure readable on dark card ───────────────────────
    (
        'class="card-time"',
        'class="card-time" style="color:#14b8a6;font-size:12px;font-weight:600;margin-top:3px;"',
    ),

    # ── Error cards ───────────────────────────────────────────────────────────
    (
        'class="error-card"',
        'class="error-card" style="background-color:#1a0a0a;border:1px solid #7f1d1d;'
        'border-radius:8px;padding:12px;margin-bottom:10px;font-size:13px;color:#fca5a5;"',
    ),
]


def _prepare_email_html(html: str) -> str:
    """
    Prepares the report HTML for email delivery:
    1. Replaces all CSS custom properties with actual hex values.
    2. Injects inline style= attributes on every contrast-sensitive element
       so Gmail and other clients that strip <style> blocks still render correctly.
    """
    # Step 1: resolve CSS variables
    for var, value in _CSS_VARS.items():
        html = html.replace(var, value)

    # Step 2: inject inline styles (order matters — more specific patches first)
    for old, new in _INLINE_PATCHES:
        html = html.replace(old, new)

    return html


def send_report(html_content: str, report_date: date, bet_count: int) -> bool:
    if not EMAIL_PASSWORD:
        logger.warning("EMAIL_PASSWORD not set — skipping email delivery")
        return False

    subject = (
        f"Betting Report {report_date.strftime('%b %d')} — "
        f"{bet_count} bet{'s' if bet_count != 1 else ''} found"
    )

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = EMAIL_FROM
    msg["To"]      = ", ".join(EMAIL_TO)

    plain = (
        f"Daily Betting Report — {report_date.strftime('%B %d, %Y')}\n\n"
        f"{bet_count} bet(s) found today.\n\n"
        "Open the HTML version for full details and Robinhood actions.\n"
        "View online: https://system-83q.pages.dev"
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(_prepare_email_html(html_content), "html"))

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.sendmail(EMAIL_FROM, EMAIL_TO, msg.as_string())
        logger.info(f"Email sent to {EMAIL_TO}")
        return True
    except smtplib.SMTPAuthenticationError:
        logger.error("SMTP auth failed — check EMAIL_PASSWORD secret")
        return False
    except Exception as e:
        logger.error(f"Email send error: {e}")
        return False
