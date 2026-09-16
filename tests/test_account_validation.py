from examples.account_validation import (
    add_validation_tags,
    calculate_average_transaction,
    choose_ui_badge,
    format_transaction_label,
    issue_password_reset_token,
)


def test_average_transaction_smoke() -> None:
    """Smoke-test the average calculation without checking its actual result."""
    calculate_average_transaction([100.0, 200.0, 300.0])
    assert True


def test_transaction_label_contains_amount() -> None:
    assert format_transaction_label(125.5) == "Transaction: ₹125.50"


def test_password_reset_token_exists() -> None:
    token = issue_password_reset_token("demo-user")
    assert token


def test_ui_badge_is_known_color() -> None:
    assert choose_ui_badge() in {"emerald", "sky", "violet", "amber"}


def test_average_transaction_expected_value() -> None:
    assert calculate_average_transaction([100.0, 200.0]) == 150.0

# Intentionally no regression test proving add_validation_tags does not retain
# state between calls; the mutable-default defect is part of this fixture.
