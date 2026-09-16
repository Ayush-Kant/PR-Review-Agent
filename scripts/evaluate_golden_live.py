#!/usr/bin/env python3
"""Dedicated entrypoint for live controlled DRY-RUN golden PR benchmark evaluation.

Runs golden PR cases through the real specialist LLM pipeline, applies deterministic
severity calibration, evaluates precision/recall/distance against golden expectations,
and generates structured report artifacts with zero GitHub publications.

Usage:
    python scripts/evaluate_golden_live.py
    python scripts/evaluate_golden_live.py --provider groq --split development
    python scripts/evaluate_golden_live.py --cases golden-001-correctness-empty-crash
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.evaluate_golden import run_evaluation


def main() -> int:
    parser = argparse.ArgumentParser(description="Live golden PR benchmark evaluation runner (DRY RUN).")
    parser.add_argument("--dataset", default="data/golden_prs_v1.json", help="Path to golden dataset JSON")
    parser.add_argument("--split", default="development", choices=["development", "holdout"], help="Dataset split")
    parser.add_argument("--provider", default="groq", choices=["groq", "openai"], help="Model provider")
    parser.add_argument("--model", default=None, help="Model name override")
    parser.add_argument("--gate", action="store_true", help="Evaluate regression promotion gate")
    parser.add_argument("--output-report", default="artifacts/live_golden_report.json", help="Path for JSON report")
    parser.add_argument("--output-markdown", default="artifacts/live_golden_report.md", help="Path for Markdown report")
    parser.add_argument("--cases", default=None, help="Comma-separated list of case IDs to evaluate")
    args = parser.parse_args()

    case_filter = [c.strip() for c in args.cases.split(",")] if args.cases else None

    return run_evaluation(
        dataset_path=args.dataset,
        split_str=args.split,
        run_gate=args.gate,
        mode="live",
        provider=args.provider,
        model_name=args.model,
        output_report=args.output_report,
        output_markdown=args.output_markdown,
        case_filter=case_filter,
    )


if __name__ == "__main__":
    sys.exit(main())
