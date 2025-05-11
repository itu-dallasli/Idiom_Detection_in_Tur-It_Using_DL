
"""Training script for MWE detection (BIO tags, CRF) using
dataset_fixed.py and model_fixed.py.

Usage:
  python run_train.py \
       --train_csv data/train.csv \
       --output_dir runs/mwe_bert_crf \
       --model_name dbmdz/bert-base-turkish-cased \
       --epochs 5 \
       --batch_size 16
"""

import argparse, os, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from dataset_fixed import preprocess_dataframe, IdiomDataset
from model_fixed import EnhancedBertForIdiomDetection

# ---------------------------------------------------------------------------
# Sequence-level F1 used only for monitoring during training (token-level).
# ---------------------------------------------------------------------------
def bio_f1(preds, labels, ignore_index=-100):
    # both are (B, T) tensors
    preds = preds.reshape(-1)
    labels = labels.reshape(-1)

    mask = labels != ignore_index
    preds = preds[mask]
    labels = labels[mask]

    tp = ((preds == labels) & (labels > 0)).sum()
    fp = ((preds != labels) & (preds > 0)).sum()
    fn = ((preds != labels) & (labels > 0)).sum()

    precision = tp / (tp + fp + 1e-8)
    recall    = tp / (tp + fn + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    return f1.item()


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits  # model_fixed returns decoded indices already
    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels)
    return {"token_f1": bio_f1(preds_t, labels_t)}


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="dbmdz/bert-base-turkish-cased")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--freeze_layers", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    return ap


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    # ------------------ data prep ------------------
    df = pd.read_csv(args.train_csv)
    sentences, labels = preprocess_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Train / dev split (stratify by language)
    train_sent, dev_sent, train_lab, dev_lab = train_test_split(
        sentences, labels, test_size=0.1, random_state=args.seed,
        stratify=df["language"]
    )

    train_ds = IdiomDataset(train_sent, train_lab, tokenizer)
    dev_ds   = IdiomDataset(dev_sent,   dev_lab, tokenizer)

    # ------------------ model ------------------
    model = EnhancedBertForIdiomDetection(
        model_name_or_path=args.model_name,
        num_labels=3,                       # BIO
        lstm_hidden_size=384,
        lstm_layers=3,
        use_bilstm=True,
        freeze_bert_layers=args.freeze_layers,
        dropout_prob=0.1,
        use_crf=True,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # ------------------ train ------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        learning_rate=3e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        gradient_accumulation_steps=2,      # effective batch = 32 if bs=16
        fp16=True,
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(os.path.join(args.output_dir, "best_model"))

    print("Training complete. Best checkpoint saved to", args.output_dir)


if __name__ == "__main__":
    main()
