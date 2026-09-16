from examples.live_review_validation_target import format_report, summarize


def test_summarize_smoke():
    summarize([10.0, 20.0, 30.0])
    assert True


def test_report_formatting():
    findings = []
    format_report(findings)
    assert findings
