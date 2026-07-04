"""
LoRA fine-tuning of google/flan-t5-small as a livestock advisory generator.

Usage:
    python finetune.py --train ./data/train.jsonl --val ./data/val.jsonl --epochs 3
"""
import argparse

from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

MODEL_NAME = "google/flan-t5-small"
MAX_IN, MAX_OUT = 128, 128


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM, r=8, lora_alpha=32, lora_dropout=0.1,
        target_modules=["q", "v"],
    )
    model = get_peft_model(base_model, lora_config)
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files={"train": args.train, "validation": args.val})

    def preprocess(batch):
        model_inputs = tokenizer(batch["input"], max_length=MAX_IN, truncation=True)
        labels = tokenizer(text_target=batch["target"], max_length=MAX_OUT, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    tokenized = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)

    training_args = Seq2SeqTrainingArguments(
        output_dir="outputs/slm-livestock-advisor",
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=1e-3,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        predict_with_generate=True,
        logging_steps=25,
        load_best_model_at_end=True,
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained("outputs/slm-livestock-advisor")
    tokenizer.save_pretrained("outputs/slm-livestock-advisor")
    print("Saved fine-tuned SLM to outputs/slm-livestock-advisor")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    main(args)
