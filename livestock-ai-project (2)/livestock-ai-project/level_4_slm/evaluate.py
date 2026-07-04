"""
Evaluate the fine-tuned SLM advisory generator with ROUGE + qualitative samples.

Usage:
    python evaluate.py --model ./outputs/slm-livestock-advisor --val ./data/val.jsonl
"""
import argparse
import json

import evaluate
from datasets import load_dataset
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)

    val_ds = load_dataset("json", data_files={"validation": args.val})["validation"]
    rouge = evaluate.load("rouge")

    preds, refs = [], []
    samples = []
    for i, row in enumerate(val_ds):
        inputs = tokenizer(row["input"], return_tensors="pt", truncation=True, max_length=128)
        out = model.generate(**inputs, max_new_tokens=128)
        pred_text = tokenizer.decode(out[0], skip_special_tokens=True)
        preds.append(pred_text)
        refs.append(row["target"])
        if i < 20:
            samples.append({"query": row["input"], "reference": row["target"], "generated": pred_text})

    scores = rouge.compute(predictions=preds, references=refs)
    with open("outputs/rouge_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    with open("outputs/qualitative_samples.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("ROUGE scores:", scores)
    print("Wrote outputs/rouge_scores.json and outputs/qualitative_samples.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--val", required=True)
    args = parser.parse_args()
    main(args)
