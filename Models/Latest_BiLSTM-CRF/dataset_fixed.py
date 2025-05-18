
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# -----------------------------------------------------------------------------
# Tag mapping: 0=O, 1=B‑IDIOM, 2=I‑IDIOM  (BIO scheme kept for compatibility)
# -----------------------------------------------------------------------------

class IdiomDataset(Dataset):
    """Dataset that takes WORD‑level tokens & BIO labels and aligns them to
    WordPiece sub‑tokens on‑the‑fly so labels stay in sync with the model input.
    """

    def __init__(self, sentences, bio_labels, tokenizer: BertTokenizer,
                 max_length: int = 128):
        self.sentences = sentences          # list[list[str]]  – word tokens
        self.bio_labels = bio_labels        # list[list[int]]  – BIO per word
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        words  = self.sentences[idx]        # e.g. ["Ben", "de", "geldim"]
        labels = self.bio_labels[idx]

        # ---------------------------------------------------------------------
        # 1) Tokenise with `is_split_into_words=True` so the tokenizer keeps
        #    the *word boundaries*. Special tokens ([CLS]/[SEP]) are added.
        # ---------------------------------------------------------------------
        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        # ---------------------------------------------------------------------
        # 2) Align BIO labels to the resulting WordPiece sequence.
        #    * `word_ids()` returns None for special/pad tokens.
        #    * We propagate the word's BIO tag to **all** its sub‑pieces.
        # ---------------------------------------------------------------------
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        for word_id in word_ids:
            if word_id is None:          # special / padding
                aligned_labels.append(0) # treat as "O" (ignored by CRF mask)
            else:
                aligned_labels.append(labels[word_id])

        labels_tensor = torch.tensor(
            aligned_labels,
            dtype=torch.long
        )

        return {
            "input_ids"     : encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels"        : labels_tensor
        }

# -----------------------------------------------------------------------------
# Helper: build (inputs, labels) from the original CSV provided by the user
# -----------------------------------------------------------------------------

def preprocess_dataframe(df: pd.DataFrame):
    """Convert the competition CSV into
       – sentences: list[list[str]]  (words)
       – labels   : list[list[int]]  (BIO per word)
    """
    sentences_out, labels_out = [], []

    for _, row in df.iterrows():
        sentence_words  = eval(row["tokenized_sentence"])       # list[str]
        idiom_indices   = eval(row["indices"])                  # list[int]

        # ---------------------------------------------------------------------
        # Build BIO tags **at word level**
        # ---------------------------------------------------------------------
        bio = [0] * len(sentence_words)  # start with "O" everywhere

        if idiom_indices != [-1]:        # -1 means "no MWE in sentence"
            idiom_indices_sorted = sorted(idiom_indices)
            for j, w_idx in enumerate(idiom_indices_sorted):
                if j == 0:
                    bio[w_idx] = 1       # B‑IDIOM
                else:
                    # Gap → start a new idiom segment with "B"
                    if w_idx - idiom_indices_sorted[j-1] > 1:
                        bio[w_idx] = 1
                    else:
                        bio[w_idx] = 2   # I‑IDIOM

        sentences_out.append(sentence_words)
        labels_out.append(bio)

    return sentences_out, labels_out
