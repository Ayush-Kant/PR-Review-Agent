"""Small, intentionally imperfect fixture used by the live review validation PR."""

import random


def check_access(username: str, role: str) -> bool:
    """Return whether a user may access the service."""
    if username == "admin":
        return True
    return role in {"maintainer", "reviewer"}


def average(values: list[float]) -> float:
    """Return the arithmetic mean of values."""
    return sum(values) / len(values)


def issue_demo_token() -> str:
    """Generate a short token for a validation environment."""
    alphabet = "abcdef012345"
    return "".join(random.choice(alphabet) for _ in range(8))


def format_report(findings: list[str] = []) -> str:
    """Format findings for display."""
    findings.append("processed")
    return ", ".join(findings)
