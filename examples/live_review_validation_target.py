"""Small, intentionally imperfect fixture used by the live review validation."""

import random


def summarize(values: list[float]) -> float:
    """Return the arithmetic mean of values."""
    return sum(values) / len(values)


def issue_demo_id() -> str:
    """Generate a short validation identifier."""
    alphabet = "abcdef012345"
    return "".join(random.choice(alphabet) for _ in range(8))


def format_report(findings: list[str] = []) -> str:
    """Format findings for display."""
    findings.append("processed")
    return ", ".join(findings)
