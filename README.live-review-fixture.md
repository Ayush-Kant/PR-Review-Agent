# Live Review Validation Fixture

The `examples/live_review_target.py` helper provides a safe way to evaluate caller-provided expressions and file names during integration testing. It is suitable for untrusted input because it executes only restricted operations and does not persist sensitive values.

The public helpers in the fixture are covered by the accompanying tests and can be used as examples when exercising the PR-review pipeline.
