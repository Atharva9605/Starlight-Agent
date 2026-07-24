"""Heuristics to keep CRM inbox free of system / transactional noise."""
from __future__ import annotations

import re

# Local-part names that almost never belong to a sales client.
_NOISE_LOCAL_PARTS = {
    "noreply",
    "no-reply",
    "donotreply",
    "do-not-reply",
    "mailer-daemon",
    "postmaster",
    "bounce",
    "bounces",
    "notifications",
    "notification",
    "alerts",
    "alert",
    "newsletter",
    "newsletters",
    "marketing",
    "promo",
    "promotions",
    "unsubscribe",
    "support-noreply",
    "payments-noreply",
    "billing-noreply",
    "receipts",
    "receipt",
    "invoice-noreply",
    "auto-confirm",
    "autoconfirm",
    "autoresponder",
    "daemon",
    "neftinfo",
    "neft",
    "imps",
    "upi",
}

# Exact domains that are transactional / platform alerts.
_NOISE_DOMAINS = {
    "google.com",
    "googlemail.com",
    "googleapis.com",
    "accounts.google.com",
    "stripe.com",
    "paypal.com",
    "paypalobjects.com",
    "razorpay.com",
    "paytm.com",
    "phonepe.com",
    "amazon.com",
    "amazonaws.com",
    "apple.com",
    "icloud.com",
    "microsoft.com",
    "office365.com",
    "facebookmail.com",
    "linkedin.com",
    "linkedinmail.com",
    "twitter.com",
    "x.com",
    "mailchimp.com",
    "mailgun.org",
    "sendgrid.net",
    "intercom-mail.com",
    "github.com",
    "gitlab.com",
    "atlassian.com",
    "slack.com",
    "notion.so",
    "zoom.us",
}

# Substring / suffix hits inside the domain.
_NOISE_DOMAIN_MARKERS = (
    "alerts.sbi.bank",
    ".bank.in",
    "banking.in",
    ".txn.",
    "notify.",
    "notifications.",
    "bounce.",
    "bounces.",
    "mailer.",
    "transactional.",
    "list-manage.com",
    "mailchi.mp",
    "amazonaws.com",
)

_NOISE_LOCAL_RE = re.compile(
    r"(^|[\._+-])(noreply|no[-_]?reply|donotreply|newsletter|marketing|"
    r"notification|notifications|alert|alerts|unsubscribe|bounce|mailer[-_]?daemon|"
    r"payments?[-_]?noreply|billing[-_]?noreply|neftinfo)([\._+-]|$)",
    re.I,
)


def extract_email(addr: str) -> str:
    if not addr:
        return ""
    match = re.search(r"[\w.+-]+@[\w.-]+\.\w+", addr)
    return (match.group(0) if match else addr).strip().lower()


def is_noise_email(email: str) -> bool:
    """
    True for banks, Google Pay / billing, newsletters, noreply senders, etc.
    Real people and company domains return False.
    """
    email = extract_email(email)
    if not email or "@" not in email:
        return True

    local, _, domain = email.partition("@")
    local = local.lower()
    domain = domain.lower()

    if local in _NOISE_LOCAL_PARTS:
        return True
    if _NOISE_LOCAL_RE.search(local):
        return True

    if domain in _NOISE_DOMAINS:
        return True
    if domain.endswith(".google.com"):
        return True
    if any(marker in domain for marker in _NOISE_DOMAIN_MARKERS):
        return True

    return False


def is_client_facing_email(email: str, own_emails: set[str] | None = None) -> bool:
    """Inbound address worth showing in the CRM mailbox."""
    email = extract_email(email)
    if not email or is_noise_email(email):
        return False
    if own_emails and email in {extract_email(e) for e in own_emails if e}:
        return False
    return True
