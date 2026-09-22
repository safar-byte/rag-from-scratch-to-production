"""Lesson 03 - the evaluation harness.

    python lessons/03-evaluation-harness/run.py

A thin wrapper over `python -m ragkit.eval.run --retrieval-only`, kept so every lesson
has the same shape. The real entry point is the module; use it directly for anything
beyond a first look.
"""

from __future__ import annotations

import argparse

from ragkit.eval.run import print_report, run_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--generate", action="store_true", help="Also generate and grade answers.")
    args = parser.parse_args()

    print_report(run_eval(top_k=args.top_k, retrieval_only=not args.generate))


if __name__ == "__main__":
    main()
