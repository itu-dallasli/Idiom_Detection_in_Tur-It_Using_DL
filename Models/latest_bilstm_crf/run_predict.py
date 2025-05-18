
"""Generate prediction.csv for Kaggle‑style evaluation.

Usage:
  python run_predict.py \
      --model_dir runs/mwe_bert_crf/best_model \
      --test_csv data/test.csv \
      --output_csv prediction.csv
"""

import argparse, os, json
import pandas as pd, torch
from transformers import AutoTokenizer
from tqdm import tqdm

from dataset_fixed import IdiomDataset, preprocess_dataframe
from model_fixed import EnhancedBertForIdiomDetection

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



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')
    model = EnhancedBertForIdiomDetection.from_pretrained(
        'bert-base-multilingual-cased',
        model_name_or_path='bert-base-multilingual-cased',
        num_labels=3,
        use_bilstm=True,
        use_crf=True,
    ).to(device)
    model.eval()

    # Raw test DF is needed for IDs, language, and original words
    test_df = pd.read_csv("dataset/test_w_o_labels.csv")
    sentences, labels_dummy = preprocess_dataframe(test_df)
    dataset = IdiomDataset(sentences, labels_dummy, tokenizer)

    predictions = []
    with torch.no_grad():
        for i in tqdm(range(len(dataset)), desc="Predicting"):
            item = dataset[i]
            input_ids = item["input_ids"].unsqueeze(0).to(device)
            attention = item["attention_mask"].unsqueeze(0).to(device)

            out = model(input_ids=input_ids, attention_mask=attention)
            pred_tags = out["logits"].squeeze().cpu().tolist()

            # Remove padding part (word_ids==None gives length inc. specials)
            # For decode_indices we need only the first seq_len outputs before padding
            seq_len = int(attention.sum())
            pred_tags_trim = pred_tags[:seq_len]

            indices = decode_indices(sentences[i], pred_tags_trim, tokenizer)

            predictions.append({
                "id": int(test_df.loc[i, "id"]),
                "indices": str(indices),   # keep as string for CSV
                "language": test_df.loc[i, "language"]
            })

    pd.DataFrame(predictions).to_csv("prediction.csv", index=False)
    print("Saved")

if __name__ == "__main__":
    main()
