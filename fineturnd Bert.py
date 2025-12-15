import os
import json
import random
from typing import List, Dict

import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

# config
DATA_PATH = "dataset.jsonl"         # your path
SEED = 42

MODEL_NAME = "bert-base-uncased"        # "distilbert-base-uncased"
MAX_LENGTH = 384                        # 256

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

EPOCHS = 5
BATCH_SIZE = 16                         #  32
LR = 2e-5
WEIGHT_DECAY = 0.01

OUTPUT_DIR = "bert_out"
SAVE_DIR = "bert_finetuned"





def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_jsonl(path: str) -> List[Dict]:
    data = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text" not in rec or "label" not in rec:
                continue
            try:
                label = int(rec["label"])
            except Exception:
                continue
            if label not in (0, 1):
                continue

            text = rec["text"]
            if not isinstance(text, str):
                text = "" if text is None else str(text)

            data.append({"text": text, "label": label})
    return data


def split_data(data: List[Dict]):
    random.shuffle(data)
    n = len(data)
    n_train = int(n * TRAIN_RATIO)
    n_val = int(n * VAL_RATIO)
    train = data[:n_train]
    val = data[n_train:n_train + n_val]
    test = data[n_train + n_val:]
    return train, val, test


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", pos_label=1)

    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(labels, preds, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main():
    set_seed(SEED)

    print("=== Environment check ===")
    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print("CUDA version:", torch.version.cuda)
    print()

    data = load_jsonl(DATA_PATH)
    if not data:
        raise RuntimeError(f"No records loaded from {DATA_PATH}")

    n0 = sum(1 for r in data if r["label"] == 0)
    n1 = len(data) - n0
    print(f"Loaded: {len(data)} records | label0={n0} label1={n1}")

    train, val, test = split_data(data)
    print(f"Split: train={len(train)} val={len(val)} test={len(test)}")

    ds_train = Dataset.from_list(train)
    ds_val = Dataset.from_list(val)
    ds_test = Dataset.from_list(test)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=MAX_LENGTH)

    ds_train = ds_train.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_val = ds_val.map(tokenize_fn, batched=True, remove_columns=["text"])
    ds_test = ds_test.map(tokenize_fn, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        learning_rate=LR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        weight_decay=WEIGHT_DECAY,

        eval_strategy="epoch",          
        save_strategy="epoch",
        save_total_limit=1,  

        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,

        logging_steps=50,
        seed=SEED,
        report_to="none",

        fp16=True,  
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds_train,
        eval_dataset=ds_val,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("\n=== Training ===")
    print("cuda available:", torch.cuda.is_available())
    print("torch cuda device:", torch.cuda.get_device_name(0))
    print("model device:", next(model.parameters()).device)
    trainer.train()

    print("\n=== Test evaluation (best checkpoint by F1) ===")
    test_metrics = trainer.evaluate(ds_test)

    acc = test_metrics.get("eval_accuracy")
    p = test_metrics.get("eval_precision")
    r = test_metrics.get("eval_recall")
    f1 = test_metrics.get("eval_f1")
    tp = test_metrics.get("eval_tp")
    fp = test_metrics.get("eval_fp")
    fn = test_metrics.get("eval_fn")
    tn = test_metrics.get("eval_tn")

    print(f"test_acc={acc:.4f} | test_p={p:.4f} | test_r={r:.4f} | test_f1={f1:.4f}")
    print(f"Confusion (pos=1): TP={int(tp)} FP={int(fp)} FN={int(fn)} TN={int(tn)}")

    os.makedirs(SAVE_DIR, exist_ok=True)
    trainer.save_model(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Saved: {SAVE_DIR}/")


if __name__ == "__main__":
    main()