# Level NLP — Natural Language Processing
## Mining Farmer Queries & Advisory Text for Livestock Insights

### Real Dataset
**Kisan Call Centre (KCC) Query Dataset** — real, anonymized transcripts of
farmer queries to India's national agricultural helpline, each tagged with
Crop/Category, State/District, Query Type, and the officer's answer. A large
share of records fall under `Category = "Animal Husbandry"` (cattle health,
feeding, calving, vaccination, disease symptoms) — a direct real-world analog
of the "farmer logs" the Phase-2 upgrade specifies.

- Portal: **data.gov.in** → search "Kisan Call Centre" (published by
  Ministry of Agriculture & Farmers Welfare / ICAR; monthly CSV dumps,
  millions of rows nationally).
```bash
# Download the relevant state/month CSV(s) from data.gov.in (API key required,
# free to obtain at data.gov.in), then:
mkdir -p ./data
mv KCC_*.csv ./data/
```
Filter to livestock-relevant rows: `Category` values such as
`"Animal Husbandry"`, `"Dairy"`, `"Veterinary"`.

### Tasks Implemented (`pipeline.py`)
1. **Preprocessing** — lowercase, remove boilerplate, tokenize (spaCy).
2. **Text classification** — categorize each query into an issue type
   (Disease/Symptom, Feeding/Nutrition, Breeding/Calving, Vaccination, Other)
   using TF-IDF + Linear SVM (fast, interpretable baseline for a helpline
   triage system).
3. **Named Entity Recognition** — extract disease names, symptoms, drug/vaccine
   names, and locations using spaCy's `en_core_web_sm` plus a small custom
   `EntityRuler` seeded with real veterinary terms (mastitis, FMD, lumpy skin
   disease, bloat, tick fever, etc.).
4. **Summarization/keyword extraction** — TextRank (via `sumy`) to produce a
   one-line digest of each day's queries per district, and TF-IDF keyword
   extraction to surface emerging terms (early signal of a local outbreak).

### Run
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python pipeline.py --data ./data/KCC_animal_husbandry.csv
```

### Deliverables
- `outputs/query_classifier.pkl`, `outputs/classification_report.json`
- `outputs/entities_extracted.csv` — disease/symptom/location entities per query
- `outputs/daily_digest.json` — summarized advisory digest per district/day
- `outputs/emerging_keywords.json` — top rising terms (outbreak early-warning signal)
