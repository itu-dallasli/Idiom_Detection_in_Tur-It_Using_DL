"""Generate prediction.csv for Kaggle‑style evaluation.

Usage:
  python run_predict.py \
      --model_dir runs/mwe_bert_crf/best_model \
      --test_csv data/test.csv \
      --output_csv prediction.csv \
      --seed 42
"""

import argparse, os, json
import pandas as pd, torch
from transformers import AutoTokenizer
from tqdm import tqdm
import random
import numpy as np

from dataset_fixed import IdiomDataset, preprocess_dataframe
from model_fixed import EnhancedBertForIdiomDetection

def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def decode_indices(sentence_words, predicted_tags, tokenizer):
    """Convert token‑level BIO predictions back to WORD indices list."""
    encoding = tokenizer(sentence_words,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         padding=False,
                         truncation=False,
                         return_tensors="pt")
    word_ids = encoding.word_ids(batch_index=0)

    mwes = set()
    for tok_pos, (tag, wid) in enumerate(zip(predicted_tags, word_ids)):
        if wid is None:
            continue
        if tag > 0:                       # B or I
            mwes.add(wid)

    if not mwes:
        return [-1]
    else:
        return sorted(list(mwes))

def parse_args():
    parser = argparse.ArgumentParser(description="Generate predictions using trained model")
    parser.add_argument("--model_dir", type=str, required=True,
                      help="Path to saved model directory")
    parser.add_argument("--test_csv", type=str, required=True,
                      help="Path to test data CSV")
    parser.add_argument("--output_csv", type=str, required=True,
                      help="Path to save predictions")
    parser.add_argument("--seed", type=int, default=42,
                      help="Random seed for reproducibility")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for inference")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Set seeds for reproducibility
    set_seeds(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model configuration
    config_path = os.path.join(args.model_dir, "training_config.txt")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            config = dict(line.strip().split(": ") for line in f)
        model_name = config.get("model_name", "bert-base-multilingual-cased")
    else:
        model_name = "bert-base-multilingual-cased"
        print(f"Warning: Could not find config file at {config_path}, using default model name")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = EnhancedBertForIdiomDetection.from_pretrained(
        args.model_dir,
        model_name_or_path=model_name,
        num_labels=3,
        use_bilstm=True,
        use_crf=True,
    ).to(device)
    model.eval()

    # Load and preprocess test data
    test_df = pd.read_csv(args.test_csv)
    sentences, labels_dummy = preprocess_dataframe(test_df)
    dataset = IdiomDataset(sentences, labels_dummy, tokenizer)

    # Generate predictions
    predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            item = dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention = item["attention_mask"].unsqueeze(0).to(device)

            out = model(input_ids=input_ids, attention_mask=attention)
            pred_tags = out["logits"].squeeze().cpu().tolist()

            # Remove padding part
            seq_len = int(attention.sum())
            pred_tags_trim = pred_tags[:seq_len]

            indices = decode_indices(sentences[i], pred_tags_trim, tokenizer)

            predictions.append({
                "id": int(test_df.loc[i, "id"]),
                "indices": str(indices),   # keep as string for CSV
                "language": test_df.loc[i, "language"]
            })

    # Save predictions
    pd.DataFrame(predictions).to_csv(args.output_csv, index=False)
    print(f"Predictions saved to {args.output_csv}")

if __name__ == "__main__":
    main()
