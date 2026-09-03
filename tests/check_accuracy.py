"""
Accuracy floor gate for CI: reads a saved benchmark results file and fails
if mean judge accuracy (excluding known-defective questions) drops below
the configured minimum.

Usage:
  python tests/check_accuracy.py --label ci_benchmark --min-accuracy 0.80
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

DEFECTIVE_QIDS = {32, 33}
ARTIFACTS = Path(__file__).resolve().parents[1] / "evaluation" / "artifacts"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--min-accuracy", type=float, default=0.80)
    args = ap.parse_args()

    results_file = ARTIFACTS / f"mod01_results_{args.label}.json"
    if not results_file.exists():
        print(f"ERROR: results file not found: {results_file}")
        return 1

    rows = json.loads(results_file.read_text())
    valid = [r for r in rows if r["qid"] not in DEFECTIVE_QIDS]
    acc = statistics.mean(r["judge_score"] for r in valid)
    print(f"Accuracy: {acc*100:.1f}% ({len(valid)} valid questions, floor {args.min_accuracy*100:.0f}%)")

    if acc < args.min_accuracy:
        print("FAIL: accuracy below floor — pipeline regression.")
        return 1
    print("PASS: accuracy floor met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
