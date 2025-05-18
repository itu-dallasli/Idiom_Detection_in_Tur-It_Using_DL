"""Training script for MWE detection (BIO tags, CRF) using
dataset_fixed.py and model_fixed.py.

Usage:
  python run_train.py \
       --train_csv data/train.csv \
       --output_dir runs/mwe_bert_crf \
       --model_name dbmdz/bert-base-turkish-cased \
       --epochs 5 \
       --batch_size 16 \
       --seed 42
"""

import argparse, os, random, numpy as np, pandas as pd, torch
from sklearn.model_selection import train_test_split
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          Trainer, TrainingArguments, set_seed)

from dataset_fixed import preprocess_dataframe, IdiomDataset
from model_fixed import EnhancedBertForIdiomDetection

def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_seed(seed)

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

def parse_args():
    parser = argparse.ArgumentParser(description="Train the idiom detection model")
    parser.add_argument("--train_csv", type=str, required=True,
                      help="Path to training data CSV")
    parser.add_argument("--output_dir", type=str, required=True,
                      help="Directory to save model checkpoints")
    parser.add_argument("--model_name", type=str, default="bert-base-multilingual-cased",
                      help="Pre-trained model name")
    parser.add_argument("--epochs", type=int, default=5,
                      help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--lstm_hidden_size", type=int, default=384,
                      help="Hidden size for BiLSTM")
    parser.add_argument("--lstm_layers", type=int, default=3,
                      help="Number of BiLSTM layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                      help="Dropout probability")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save training configuration
    with open(os.path.join(args.output_dir, "training_config.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")

    # ------------------ data prep ------------------
    df = pd.read_csv(args.train_csv)
    sentences, labels = preprocess_dataframe(df)

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Train / dev split (stratify by language)
    train_sent, dev_sent, train_lab, dev_lab = train_test_split(
        sentences, labels, test_size=0.1,
        stratify=df["language"],
        random_state=args.seed
    )

    train_ds = IdiomDataset(train_sent, train_lab, tokenizer)
    dev_ds   = IdiomDataset(dev_sent,   dev_lab, tokenizer)

    # ------------------ model ------------------
    model = EnhancedBertForIdiomDetection(
        model_name_or_path=args.model_name,
        num_labels=3,                       # BIO
        lstm_hidden_size=args.lstm_hidden_size,
        lstm_layers=args.lstm_layers,
        use_bilstm=True,
        freeze_bert_layers=0,
        dropout_prob=args.dropout,
        use_crf=True,
    )

    collator = DataCollatorForTokenClassification(tokenizer, pad_to_multiple_of=8)

    # ------------------ training args ------------------
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="token_f1",
        greater_is_better=True,
        seed=args.seed,
    )

    # ------------------ train ------------------
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
    
    print(f"Training complete. Best checkpoint saved to {os.path.join(args.output_dir, 'best_model')}")

if __name__ == "__main__":
    main()
