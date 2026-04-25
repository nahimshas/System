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

# CSS variable → actual hex value.
# Email clients don't support CSS custom properties so we inline them before sending.
_CSS_VARS = {
    "var(--bg)":         "#0f1117",
    "var(--card)":       "#1a1d27",
    "var(--border)":     "#2d3148",
    "var(--text)":       "#e2e8f0",
    "var(--muted)":      "#94a3b8",
    "var(--green)":      "#22c55e",
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


def _prepare_email_html(html: str) -> str:
    """
    Makes the HTML safe for email clients by:
    1. Replacing all CSS custom properties with their actual hex values.
    2. Adding explicit bgcolor + style to <body> so even basic clients show dark background.
    """
    for var, value in _CSS_VARS.items():
        html = html.replace(var, value)

    # Force dark background on body — belt-and-suspenders for Outlook/Gmail
    html = html.replace(
        "<body>",
        '<body bgcolor="#0f1117" style="background-color:#0f1117;color:#e2e8f0;margin:0;padding:0;">',
    )
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
        f"View online: https://system-83q.pages.dev"
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
