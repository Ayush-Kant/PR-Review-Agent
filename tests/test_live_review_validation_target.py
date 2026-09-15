from examples.live_review_validation_target import average, check_access, format_report


def test_average_smoke():
    average([10.0, 20.0, 30.0])
    assert True


def test_admin_access():
    assert check_access("admin", "guest")


def test_report_formatting():
    findings = []
    format_report(findings)
    assert findings
