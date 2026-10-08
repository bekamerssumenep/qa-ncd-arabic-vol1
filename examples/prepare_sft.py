#!/usr/bin/env python3
"""Format QA NCD Arabic Vol.1 rows into SFT training texts.

Usage:
    python examples/prepare_sft.py --data sample.jsonl --out sft_texts.jsonl
"""
import argparse, json


def format_sft(row):
    return (
        f"### Instruction:\n{row['instruction']}\n\n"
        f"### Response:\n{row['output']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        rows = [json.loads(l) for l in f if l.strip()]
    texts = [format_sft(r) for r in rows]
    with open(args.out, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(json.dumps({"text": t}, ensure_ascii=False) + "\n")
    print(f"Wrote {len(texts)} SFT texts -> {args.out}")
    print("Preview:\n", texts[0][:400], "...")


if __name__ == "__main__":
    main()
