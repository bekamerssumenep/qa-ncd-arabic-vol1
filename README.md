# QA NCD Arabic Vol.1 — Sample & Code Examples

Free **150-row sample** + ready-to-run code for **IndoHealth-NLP QA NCD Arabic Vol.1** — 3,480 Arabic medical instruction-tuning pairs (NCD-focused: diabetes, hypertension, cardiovascular, chronic care).

- 📦 **Full dataset (3,480 pairs, $39):** https://3929431511879.gumroad.com/l/QANCDArabicVol1
- 🤗 **Sample on Hugging Face:** https://huggingface.co/datasets/IndoHealth-NLP/qa-ncd-arabic-vol1-sample
- 📊 **Sample on Kaggle:** https://www.kaggle.com/datasets/indohealthnlp/qa-ncd-arabic-vol-1-sample

## What's inside

Each row (JSONL): `system`, `instruction` (White Arabic — natural Educated Spoken Arabic), `output` (clinical Modern Standard Arabic / Fusha), `metadata` (`pmid` + `disease_tag`).

## Quickstart

```bash
pip install -r requirements.txt
# Download the free sample first:
# https://huggingface.co/datasets/IndoHealth-NLP/qa-ncd-arabic-vol1-sample
python examples/quickstart.py --data sample.jsonl
```

## Examples

| File | Description |
|---|---|
| `examples/quickstart.py` | Load the sample, print stats & random rows |
| `examples/prepare_sft.py` | Format rows into SFT training texts |
| `examples/finetune_lora.py` | LoRA fine-tuning with 🤗 Transformers + PEFT + TRL |

## Licenses

- **Code** in this repo: MIT — free to use.
- **Data** (sample & full dataset): [IndoHealth-NLP Commercial License](https://3929431511879.gumroad.com/l/QANCDArabicVol1) — commercial use allowed (including training commercial models); redistribution prohibited.

## About

**IndoHealth-NLP** — Medical Knowledge • AI • Better Health.
Premium bilingual medical datasets for AI training.
