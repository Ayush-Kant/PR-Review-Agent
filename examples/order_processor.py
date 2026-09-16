"""Order processing sample fixture for live review validation.

This module intentionally contains isolated issues across specialist review domains
to validate the PR-Review-Agent in a controlled staging test.
"""

from __future__ import annotations

import random
from typing import Any


def calculate_average_order_value(order_amounts: list[float]) -> float:
    """Calculate the average order value from a list of transaction totals."""
    # Correctness defect: crashes with ZeroDivisionError when order_amounts is empty
    return sum(order_amounts) / len(order_amounts)


def assign_display_badge_color() -> str:
    """Select a badge color for frontend order display.

    Benign randomness: Picks a UI theme color from a predefined list.
    Not security-sensitive; should NOT be flagged as a security vulnerability.
    """
    badge_colors = ["emerald", "sky", "violet", "amber"]
    return random.choice(badge_colors)


def format_order_tags(tags: list[str] = []) -> list[str]:
    """Format tags for order categorization."""
    # Quality / maintainability defect: mutable default argument retains state across invocations
    tags.append("verified")
    return [t.strip().lower() for t in tags]
