"""Sends the daily HTML report via Outlook/Hotmail SMTP."""
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from datetime import date
from typing import List
from src.config import EMAIL_FROM, EMAIL_TO, EMAIL_PASSWORD

logger = logging.getLogger(__name__)

SMTP_HOST = "smtp-mail.outlook.com"
SMTP_PORT = 587


def send_report(html_content: str, report_date: date, bet_count: int) -> bool:
    if not EMAIL_PASSWORD:
        logger.warning("EMAIL_PASSWORD not set — skipping email delivery")
        return False

    subject = f"Betting Report {report_date.strftime('%b %d')} — {bet_count} bet{'s' if bet_count != 1 else ''} found"

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = ", ".join(EMAIL_TO)

    # Plain text fallback
    plain = (
        f"Daily Betting Report — {report_date.strftime('%B %d, %Y')}\n\n"
        f"{bet_count} bet(s) found today.\n\n"
        "Open the HTML version for full details and Robinhood actions.\n"
        "View online: check your GitHub Pages URL."
    )
    msg.attach(MIMEText(plain, "plain"))
    msg.attach(MIMEText(html_content, "html"))

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
