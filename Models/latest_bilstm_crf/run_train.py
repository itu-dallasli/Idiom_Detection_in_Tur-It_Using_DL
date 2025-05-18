
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


def main():
    # ------------------ data prep ------------------
    df = pd.read_csv(r"dataset/train.csv")
    sentences, labels = preprocess_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Train / dev split (stratify by language)
    train_sent, dev_sent, train_lab, dev_lab = train_test_split(
        sentences, labels, test_size=0.1,
        stratify=df["language"]
    )

    train_ds = IdiomDataset(train_sent, train_lab, tokenizer)
    dev_ds   = IdiomDataset(dev_sent,   dev_lab, tokenizer)

    # ------------------ model ------------------
    model = EnhancedBertForIdiomDetection(
        model_name_or_path="xlm-roberta-base",
        num_labels=3,                       # BIO
        lstm_hidden_size=384,
        lstm_layers=3,
        use_bilstm=True,
        freeze_bert_layers= 0,
        dropout_prob=0.1,
        use_crf=True,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # ------------------ train ------------------

    trainer = Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model("best_model")

    print("Training complete. Best checkpoint saved to")


if __name__ == "__main__":
    main()
