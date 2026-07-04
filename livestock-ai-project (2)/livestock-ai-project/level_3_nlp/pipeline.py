"""
Level NLP — classification, NER, and summarization over real farmer queries.

Dataset: Kisan Call Centre (KCC) query dataset, filtered to Animal
Husbandry / Dairy / Veterinary categories (data.gov.in).

Expected columns (KCC standard schema — rename in COLUMN_MAP if your dump differs):
    QueryText, KccAns, StateName, DistrictName, CreatedOn, Category

Usage:
    python pipeline.py --data ./data/KCC_animal_husbandry.csv
"""
import argparse
import json
import os
import re

import pandas as pd
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.text_rank import TextRankSummarizer

COLUMN_MAP = {
    "text": ["QueryText", "Query", "query_text"],
    "answer": ["KccAns", "Answer", "answer"],
    "state": ["StateName", "State"],
    "district": ["DistrictName", "District"],
    "date": ["CreatedOn", "Date", "QueryDate"],
}

# Seed veterinary vocabulary for a lightweight rule-based issue tagger used to
# BOOTSTRAP labels (KCC data has no ready-made "issue type" column — this
# mirrors how a real triage system would be cold-started before enough
# human-labeled data exists for a supervised model).
ISSUE_KEYWORDS = {
    "Disease/Symptom": ["fever", "lumpy", "fmd", "foot and mouth", "mastitis", "bloat",
                         "diarrhea", "cough", "swelling", "lameness", "tick", "worm"],
    "Feeding/Nutrition": ["feed", "fodder", "diet", "silage", "mineral mixture", "grazing"],
    "Breeding/Calving": ["calving", "pregnan", "breeding", "insemination", "heat cycle"],
    "Vaccination": ["vaccine", "vaccination", "immuniz"],
}

DISEASE_TERMS = ["lumpy skin disease", "foot and mouth disease", "fmd", "mastitis",
                  "bloat", "tick fever", "black quarter", "anthrax", "brucellosis"]


def resolve_columns(df):
    resolved = {}
    for key, candidates in COLUMN_MAP.items():
        for c in candidates:
            if c in df.columns:
                resolved[key] = c
                break
    return resolved


def bootstrap_label(text: str) -> str:
    text_l = str(text).lower()
    for label, kws in ISSUE_KEYWORDS.items():
        if any(kw in text_l for kw in kws):
            return label
    return "Other"


def build_entity_ruler(nlp):
    ruler = nlp.add_pipe("entity_ruler", before="ner")
    patterns = [{"label": "DISEASE", "pattern": term} for term in DISEASE_TERMS]
    ruler.add_patterns(patterns)
    return nlp


def extract_entities(nlp, texts):
    rows = []
    for doc in nlp.pipe(texts, batch_size=64):
        rows.append({
            "diseases": [e.text for e in doc.ents if e.label_ == "DISEASE"],
            "locations": [e.text for e in doc.ents if e.label_ in ("GPE", "LOC")],
            "orgs": [e.text for e in doc.ents if e.label_ == "ORG"],
        })
    return rows


def summarize(text_block: str, sentence_count=3) -> str:
    if not text_block.strip():
        return ""
    parser = PlaintextParser.from_string(text_block, Tokenizer("english"))
    summarizer = TextRankSummarizer()
    sentences = summarizer(parser.document, sentence_count)
    return " ".join(str(s) for s in sentences)


def main(args):
    os.makedirs("outputs", exist_ok=True)
    df = pd.read_csv(args.data, encoding="utf-8", on_bad_lines="skip")
    cols = resolve_columns(df)
    df = df.dropna(subset=[cols["text"]]).reset_index(drop=True)
    df["clean_text"] = df[cols["text"]].astype(str).str.lower().str.replace(r"[^a-z0-9\s]", " ", regex=True)

    # 1. Bootstrap labels, then train a supervised classifier on top of them
    df["issue_type"] = df["clean_text"].apply(bootstrap_label)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["issue_type"], test_size=0.2, stratify=df["issue_type"], random_state=42
    )
    clf_pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ("clf", SGDClassifier(loss="log_loss", random_state=42)),
    ])
    clf_pipeline.fit(X_train, y_train)
    preds = clf_pipeline.predict(X_test)
    report = classification_report(y_test, preds, output_dict=True, zero_division=0)
    with open("outputs/classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    import joblib
    joblib.dump(clf_pipeline, "outputs/query_classifier.pkl")

    # 2. NER
    nlp = spacy.load("en_core_web_sm")
    nlp = build_entity_ruler(nlp)
    ents = extract_entities(nlp, df[cols["text"]].astype(str).tolist())
    ent_df = pd.DataFrame(ents)
    ent_df["query"] = df[cols["text"]]
    if "district" in cols:
        ent_df["district"] = df[cols["district"]]
    ent_df.to_csv("outputs/entities_extracted.csv", index=False)

    # 3. Summarization per district/day -> advisory digest
    digest = {}
    if "district" in cols and "date" in cols:
        for (district, date), group in df.groupby([cols["district"], cols["date"]]):
            block = " ".join(group[cols["text"]].astype(str).tolist())
            digest.setdefault(str(district), {})[str(date)] = summarize(block)
    else:
        digest["overall"] = summarize(" ".join(df[cols["text"]].astype(str).tolist()), sentence_count=5)
    with open("outputs/daily_digest.json", "w") as f:
        json.dump(digest, f, indent=2)

    # 4. Emerging keyword extraction (simple TF-IDF top terms, most recent slice
    #    vs. overall — proxy for outbreak early-warning)
    tfidf = TfidfVectorizer(max_features=30, stop_words="english", ngram_range=(1, 2))
    tfidf.fit(df["clean_text"])
    keywords = tfidf.get_feature_names_out().tolist()
    with open("outputs/emerging_keywords.json", "w") as f:
        json.dump({"top_terms": keywords}, f, indent=2)

    print("Issue-type classification report:")
    print(json.dumps(report, indent=2))
    print(f"\nExtracted entities for {len(ent_df)} queries -> outputs/entities_extracted.csv")
    print("Artifacts written to ./outputs/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to KCC CSV filtered to livestock queries")
    args = parser.parse_args()
    main(args)
