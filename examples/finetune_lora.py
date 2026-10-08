#!/usr/bin/env python3
"""LoRA fine-tuning on QA NCD Arabic Vol.1 (Transformers + PEFT + TRL).

Usage:
    pip install -r requirements.txt
    python examples/finetune_lora.py --data sft_texts.jsonl --model Qwen/Qwen2.5-0.5B

For a full run, point --data at the formatted FULL dataset (3,480 rows).
"""
import argparse, json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="JSONL with {'text': ...} rows")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--out", default="./qa-ncd-arabic-lora")
    ap.add_argument("--max_steps", type=int, default=100)
    args = ap.parse_args()

    with open(args.data, encoding="utf-8") as f:
        texts = [json.loads(l)["text"] for l in f if l.strip()]
    print(f"Training texts: {len(texts)}")

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True)
    ds = Dataset.from_list([{"text": t} for t in texts])

    trainer = SFTTrainer(
        model=model,
        train_dataset=ds,
        args=TrainingArguments(
            output_dir=args.out, per_device_train_batch_size=2,
            gradient_accumulation_steps=4, max_steps=args.max_steps,
            logging_steps=10, save_steps=200, learning_rate=2e-4,
            fp16=True, report_to="none"),
        peft_config=LoraConfig(
            r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05, task_type="CAUSAL_LM"),
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"Done. Adapter saved to {args.out}")


if __name__ == "__main__":
    main()
