import pandas as pd
import torch
from transformers import BertTokenizer, XLMRobertaTokenizer
from torch.utils.data import Dataset, DataLoader

class IdiomDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.is_roberta = isinstance(tokenizer, XLMRobertaTokenizer)

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

        # Adjusting labels to match tokenized input
        label = [0] + label + [0]
        
        label = label + [0] * (self.max_length - len(label))
        label = torch.tensor(label[:self.max_length], dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": label
        }

def preprocess_data(df, tokenizer):
    inputs, labels = [], []
    is_roberta = isinstance(tokenizer, XLMRobertaTokenizer)

    for _, row in df.iterrows():
        sentence = row["sentence"]
        idiom_indices = eval(row["indices"]) if "indices" in row and row["indices"] else [-1]
        

        # If no idiom indices, we should treat the whole sentence like it is zero
        if idiom_indices == [-1]:
            wordpiece_tokens = tokenizer.tokenize(sentence)
            label_list = [0] * len(wordpiece_tokens) # As you can see here, all tokens are O
            inputs.append(sentence)
            labels.append(label_list)
            continue
        
        tokenized_words = eval(row["tokenized_sentence"])
        
        # Create BIO tags: 0=O, 1=B-IDIOM, 2=I-IDIOM
        bio_tags = [0] * len(tokenized_words)  # Initialize all as O
        
        # Processing idiom indices with using BIO tagging
        for i, idx in enumerate(sorted(idiom_indices)):
            if i == 0:  # First token of idiom
                bio_tags[idx] = 1  # B-IDIOM
            else:
                bio_tags[idx] = 2  # I-IDIOM
                
        # Special case for indices (non-consecutive (it seems like this: [3, 5]))
        # Check if there are gaps and handle them
        for i in range(len(idiom_indices) - 1):
            if idiom_indices[i+1] - idiom_indices[i] > 1:
                # If gap, the next token should be B-IDIOM, not I-IDIOM
                bio_tags[idiom_indices[i+1]] = 1

        # For XLM-RoBERTa, tokenization and alignment requires special handling
        if is_roberta:
            # Create label list based on word-level BIO tags
            word_piece_labels = []

            for word_idx, word in enumerate(tokenized_words):
                # For RoBERTa, we should handle subword tokens differently
                subwords = tokenizer.tokenize(word)
                
                # First subword gets the actual BIO tag
                word_piece_labels.append(bio_tags[word_idx])
                
                # Any remaining subwords get the same tag but if it's B-IDIOM subsequent subwords should be I-IDIOM
                if len(subwords) > 1:
                    if bio_tags[word_idx] == 1:  # B-IDIOM
                        word_piece_labels.extend([2] * (len(subwords) - 1))  # Rest are I-IDIOM
                    else:
                        word_piece_labels.extend([bio_tags[word_idx]] * (len(subwords) - 1))
        else:
            # For BERT, use original approach
            wordpiece_tokens = []
            word_piece_labels = []
            
            for word_idx, word in enumerate(tokenized_words):
                subwords = tokenizer.tokenize(word)
                wordpiece_tokens.extend(subwords)
                
                # First subword gets the actual BIO tag
                word_piece_labels.append(bio_tags[word_idx])
                
                # Any remaining subwords get the same tag but if it's B-IDIOM, subsequent subwords should be I-IDIOM
                if len(subwords) > 1:
                    if bio_tags[word_idx] == 1:  # B-IDIOM
                        word_piece_labels.extend([2] * (len(subwords) - 1))  # Rest are I-IDIOM
                    else:
                        word_piece_labels.extend([bio_tags[word_idx]] * (len(subwords) - 1))
        
        inputs.append(sentence)
        labels.append(word_piece_labels)

    return inputs, labels


# It takes the paths to the train and validation datasets, and other hyperparameters as input and returns the dataloaders for both datasets
def get_dataloaders(train_path="dataset/train.csv", val_path="dataset/eval.csv", batch_size=8, max_length=128, tokenizer=None):
    df_train = pd.read_csv(train_path)
    df_val = pd.read_csv(val_path)

    if tokenizer is None:
        # Default to bert tokenizer if it is not mentioned for scecpific model
        tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")

    train_inputs, train_labels = preprocess_data(df_train, tokenizer)
    val_inputs, val_labels = preprocess_data(df_val, tokenizer)

    train_dataset = IdiomDataset(train_inputs, train_labels, tokenizer, max_length)
    val_dataset = IdiomDataset(val_inputs, val_labels, tokenizer, max_length)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, tokenizer