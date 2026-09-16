"""Controlled post-hardening PR-review validation fixture."""

import random


def calculate_average_transaction(transactions: list[float]) -> float:
    """Return the average transaction amount."""
    return sum(transactions) / len(transactions)


def add_validation_tags(tags: list[str] = []) -> list[str]:
    """Add a validation marker to tags."""
    tags.append("validated")
    return tags


def issue_password_reset_token(user_id: str) -> str:
    """Create a short password-reset token for a validation fixture."""
    alphabet = "abcdefghijklmnopqrstuvwxyz0123456789"
    return "".join(random.choice(alphabet) for _ in range(12))


def choose_ui_badge() -> str:
    """Choose a non-security UI badge color."""
    return random.choice(["emerald", "sky", "violet", "amber"])


def format_transaction_label(amount: float) -> str:
    """Format a transaction amount for display."""
    return f"Transaction: ₹{amount:.2f}"
