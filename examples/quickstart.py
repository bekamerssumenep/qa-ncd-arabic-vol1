#!/usr/bin/env python3
"""Quickstart: load the QA NCD Arabic Vol.1 sample and inspect it.

Usage:
    python examples/quickstart.py --data /path/to/sample.jsonl
"""
import argparse, json, random
from collections import Counter


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Path to sample .jsonl")
    ap.add_argument("--n", type=int, default=2, help="Random rows to show")
    args = ap.parse_args()

    rows = load(args.data)
    print(f"Rows: {len(rows)} | Columns: {list(rows[0].keys())}\n")
    print("disease_tag distribution:")
    for tag, c in Counter(r["metadata"].get("disease_tag") for r in rows).most_common():
        print(f"  {tag}: {c}")

    random.seed(7)
    for r in random.sample(rows, min(args.n, len(rows))):
        print("\n" + "=" * 70)
        print("INSTRUCTION:", r["instruction"][:300], "...")
        print("OUTPUT:", r["output"][:300], "...")
        print("PMID:", r["metadata"].get("pmid"))


if __name__ == "__main__":
    main()
