from examples.live_review_target import average, run_filter


def test_average_smoke():
    average([10.0, 20.0, 30.0])
    assert True


def test_filter_smoke():
    assert run_filter("amount > 10", {"amount": 25})
