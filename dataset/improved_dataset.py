
import pandas as pd
import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader

class IdiomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        label = [0] + label + [0]  # Add labels for [CLS] and [SEP]
        if len(label) > self.max_length:
            label = label[:self.max_length]
        else:
            label += [-100] * (self.max_length - len(label))

        label = torch.tensor(label, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }

def preprocess_data(df, tokenizer):
    inputs, labels = [], []

    for _, row in df.iterrows():
        sentence = row["sentence"]
        idiom_indices = eval(row["indices"])
        tokenized_words = eval(row["tokenized_sentence"])

        bio_tags = [0] * len(tokenized_words)
        for i, idx in enumerate(sorted(idiom_indices)):
            if idx < 0 or idx >= len(tokenized_words):
                continue
            if i == 0 or (idx - idiom_indices[i-1]) > 1:
                bio_tags[idx] = 1  # B-IDIOM
            else:
                bio_tags[idx] = 2  # I-IDIOM

        wordpiece_tokens = []
        label_list = []

        for word_idx, word in enumerate(tokenized_words):
            subwords = tokenizer.tokenize(word)
            if not subwords:
                continue
            wordpiece_tokens.extend(subwords)
            label_list.append(bio_tags[word_idx])
            label_list.extend([-100] * (len(subwords) - 1))  # Ignore subwords

        inputs.append(" ".join(wordpiece_tokens))  # Join for tokenizer.encode_plus
        labels.append(label_list)

    return inputs, labels

def get_dataloaders(train_path="public_data/train.csv", val_path="public_data/eval.csv", batch_size=8, max_length=128):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_inputs, train_labels = preprocess_data(df_train, tokenizer)
    val_inputs, val_labels = preprocess_data(df_val, tokenizer)

    train_dataset = IdiomDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = IdiomDataset(val_inputs, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer
