# Level SLM — Small Language Model
## Fine-Tuned Advisory Generator for Farmers

### Real Dataset
Same **Kisan Call Centre (KCC)** dataset as Level NLP, reshaped into
`(farmer_query → officer_answer)` pairs — real question/advisory-answer pairs
already written by agricultural experts, ideal supervised data for an SLM
that must "summarize information and generate concise, actionable advice"
(exactly the Phase-2 spec).

```bash
python prepare_data.py --data ../level_3_nlp/data/KCC_animal_husbandry.csv
# writes ./data/train.jsonl, ./data/val.jsonl  as {"input": ..., "target": ...}
```

### Model
**`google/flan-t5-small`** (77M params) — a genuinely "small" language model
(fits the SLM level's spirit: efficient, fine-tunable on a single GPU/CPU,
deployable at the edge on a farm gateway device), fine-tuned via HuggingFace
`transformers` + `peft` (LoRA) for efficient fine-tuning.

### Run
```bash
pip install -r requirements.txt
python finetune.py --train ./data/train.jsonl --val ./data/val.jsonl --epochs 3
python evaluate.py --model ./outputs/slm-livestock-advisor --val ./data/val.jsonl
```

### Evaluation
- ROUGE-1/2/L against held-out officer answers (standard summarization/
  generation metric, matches the rubric's "Output Quality" criterion).
- A held-out qualitative set of 20 real queries with model-generated advice
  for manual review (`outputs/qualitative_samples.json`).

### Deliverables
- `outputs/slm-livestock-advisor/` — fine-tuned model + tokenizer (LoRA adapters)
- `outputs/rouge_scores.json`
- `outputs/qualitative_samples.json`
