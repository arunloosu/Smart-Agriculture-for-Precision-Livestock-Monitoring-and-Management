"""
Reshape KCC farmer-query CSV into (query -> advisory answer) JSONL pairs
for SLM fine-tuning.

Usage:
    python prepare_data.py --data ../level_3_nlp/data/KCC_animal_husbandry.csv
"""
import argparse
import json
import os

import pandas as pd
from sklearn.model_selection import train_test_split

COLUMN_MAP = {"text": ["QueryText", "Query"], "answer": ["KccAns", "Answer"]}


def resolve(df):
    resolved = {}
    for key, cands in COLUMN_MAP.items():
        for c in cands:
            if c in df.columns:
                resolved[key] = c
                break
    return resolved


def main(args):
    os.makedirs("data", exist_ok=True)
    df = pd.read_csv(args.data, on_bad_lines="skip")
    cols = resolve(df)
    df = df.dropna(subset=[cols["text"], cols["answer"]])
    df = df[(df[cols["text"]].str.len() > 5) & (df[cols["answer"]].str.len() > 5)]

    train_df, val_df = train_test_split(df, test_size=0.1, random_state=42)

    def write_jsonl(d, path):
        with open(path, "w") as f:
            for _, row in d.iterrows():
                rec = {
                    "input": f"Provide farming advice for this livestock query: {row[cols['text']]}",
                    "target": str(row[cols["answer"]]),
                }
                f.write(json.dumps(rec) + "\n")

    write_jsonl(train_df, "data/train.jsonl")
    write_jsonl(val_df, "data/val.jsonl")
    print(f"Wrote {len(train_df)} train / {len(val_df)} val examples")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    args = parser.parse_args()
    main(args)
