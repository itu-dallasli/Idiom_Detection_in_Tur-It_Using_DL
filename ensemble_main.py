import os
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from dataset.dataset import get_dataloaders
from models.plain_bert import PlainBertClassifier
from models.roberta_classifier import RobertaClassifier
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np
import pandas as pd
from tqdm import tqdm


def train_model(model, train_loader, val_loader, tokenizer, model_name, device, epochs=10, lr=2e-5):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f"Training {model_name} Epoch {epoch+1}"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1} Loss: {total_loss / len(train_loader):.4f}")

        # Save model
        torch.save(model.state_dict(), f"{model_name}_best.pt")


def predict(model, dataloader, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(F.softmax(logits, dim=-1), dim=-1)

            predictions.extend(preds.cpu().tolist())

    return predictions


def ensemble_predictions(preds_list):
    # Majority vote ensemble
    final_preds = []
    for tokens in zip(*preds_list):
        batch_preds = []
        for token_preds in zip(*tokens):
            vote = torch.mode(torch.tensor(token_preds)).values.item()
            batch_preds.append(vote)
        final_preds.append(batch_preds)
    return final_preds

def compute_metrics(predictions, labels, mask):
    # Flatten predictions and labels with mask applied
    all_preds = []
    all_labels = []

    for pred, label, attn in zip(predictions, labels, mask):
        for p, l, m in zip(pred, label, attn):
            if m == 1:
                all_preds.append(p)
                all_labels.append(l)

    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def write_submission(predictions, filename="submission/prediction.csv"):
    df = pd.read_csv("dataset/eval_w_o_labels.csv")
    result = []

    for i, row in df.iterrows():
        tokenized = eval(row['tokenized_sentence'])
        labels = predictions[i][:len(tokenized)]

        idiom_indices = []
        current = []
        for j, label in enumerate(labels):
            if label == 1:  # B-IDIOM
                if current:
                    idiom_indices.extend(current)
                    current = []
                current = [j]
            elif label == 2:  # I-IDIOM
                if current:
                    current.append(j)
            else:
                if current:
                    idiom_indices.extend(current)
                    current = []
        if current:
            idiom_indices.extend(current)

        result.append({"id": row['id'], "idiom_indices": idiom_indices if idiom_indices else [-1]})

    pd.DataFrame(result).to_csv(filename, index=False)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data
    train_loader, val_loader, tokenizer = get_dataloaders(
        train_path="dataset/train.csv",
        val_path="dataset/eval.csv",
        batch_size=8
    )

    # Initialize models
    model_a = PlainBertClassifier().to(device)
    model_b = RobertaClassifier().to(device)

    # Train models
    train_model(model_a, train_loader, val_loader, tokenizer, "plain_bert", device)
    train_model(model_b, train_loader, val_loader, tokenizer, "roberta", device)

    # Reload models for prediction
    model_a.load_state_dict(torch.load("plain_bert_best.pt"))
    model_b.load_state_dict(torch.load("roberta_best.pt"))

    val_loader_nolabels = DataLoader(val_loader.dataset, batch_size=8)

    preds_a = predict(model_a, val_loader_nolabels, device)
    preds_b = predict(model_b, val_loader_nolabels, device)

    # Ensemble
    final_preds = ensemble_predictions([preds_a, preds_b])

    # Write submission
    write_submission(final_preds)


if __name__ == '__main__':
    main()
