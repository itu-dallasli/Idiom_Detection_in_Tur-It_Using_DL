import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForTokenClassification, AdamW
from sklearn.metrics import classification_report
from tqdm import tqdm

class XLMRForIdiomDetection(nn.Module):
    def __init__(self, num_labels=3):  # 0:O, 1:B, 2:I
        super(XLMRForIdiomDetection, self).__init__()
        self.model = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=num_labels)

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

def train_model(model, train_loader, val_loader, tokenizer, epochs=5, lr=3e-5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)

    best_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask, labels)
            loss = outputs.loss

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Training Loss: {avg_loss:.4f}")

    return model

def evaluate_model(model, val_loader, tokenizer, device):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids, attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            for pred, label in zip(predictions, labels):
                all_preds.extend(pred.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

    return classification_report(all_labels, all_preds, output_dict=True)

def predict_idiom_indices(model, tokenizer, sentence, device, max_length=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        sentence,
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1).squeeze().cpu().numpy()

    # Convert label predictions to indices of idioms (label 1 or 2)
    idiom_indices = [i for i, label in enumerate(preds) if label in [1, 2]]
    return preds, idiom_indices
