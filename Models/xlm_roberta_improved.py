import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import XLMRobertaModel
import numpy as np
from tqdm import tqdm


class XLMRobertaForIdiomDetection(nn.Module):
    def __init__(self, model_name="xlm-roberta-large", num_labels=3):  # 3 labels: O, B-IDIOM, I-IDIOM
        super(XLMRobertaForIdiomDetection, self).__init__()
        
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        
        # It is added a BiLSTM layer to capture context information for longrange informations
        self.lstm = nn.LSTM(
            input_size=self.roberta.config.hidden_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.2
        )
        
        # Classification layers
        self.dropout = nn.Dropout(0.2)
        self.dense = nn.Linear(512, 256)  # 512 = 2*256 (bidirectional)
        self.activation = nn.ReLU()
        self.norm = nn.LayerNorm(256)
        self.classifier = nn.Linear(256, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get xlmroberta outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get token level representations
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply BiLSTM
        lstm_output, _ = self.lstm(sequence_output)  # [batch_size, seq_len, 2*hidden_size]
        
        # Apply classification layers
        x = self.dropout(lstm_output)
        x = self.dense(x)
        x = self.activation(x)
        x = self.norm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)  # [batch_size, seq_len, num_labels]
        
        loss = None
        if labels is not None:
            # Create mask for CRF
            crf_mask = attention_mask.bool()
            
            # CRF loss (negative log-likelihood)
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
        
        # CRF decoding for predictions (remove redundant conditional)
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())

        # Convert list of lists to tensor with padding
        pred_tensor = torch.zeros_like(input_ids)
        for i, pred_seq in enumerate(predictions):
            pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        
        return {
            'loss': loss,
            'logits': emissions,
            'predictions': pred_tensor
        }


def evaluate(model, val_loader, tokenizer, device):
    model.eval()
    val_loss = 0.0
    predictions = []
    ground_truth = []
    total_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            if batch['input_ids'].size(0) == 0:
                continue
                
            total_batches += 1
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass with loss calculation
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if outputs['loss'] is not None:
                val_loss += outputs['loss'].item()
            
            preds = outputs['predictions']
            
            # Process each sequence in batch
            for seq_preds, seq_mask, seq_labels, seq_ids in zip(preds, attention_mask, labels, input_ids):
                tokens = tokenizer.convert_ids_to_tokens(seq_ids)
                
                # Extract idiom indices based on BIO tags
                # For ground truth
                word_idx = -1
                true_idiom_indices = []
                current_idiom_indices = []
                previous_tag = 0  # O tag
                
                for i, (token, mask, label) in enumerate(zip(tokens, seq_mask, seq_labels)):
                    if mask == 0 or token in ['<s>', '</s>', '<pad>']:
                        continue
                        
                    if not token.startswith('▁'):  # Not part of a word
                        continue
                        
                    word_idx += 1
                    
                    # Handle end of previous idiom
                    if previous_tag in [1, 2] and label.item() not in [1, 2]:
                        if current_idiom_indices:
                            true_idiom_indices.extend(current_idiom_indices)
                            current_idiom_indices = []
                    
                    # Handle new idiom
                    if label.item() == 1:  # B-IDIOM
                        current_idiom_indices = [word_idx]
                    elif label.item() == 2:  # I-IDIOM
                        if previous_tag in [1, 2]:  # Continue idiom
                            current_idiom_indices.append(word_idx)
                    
                    previous_tag = label.item()
                
                # We don't forget last idiom
                if current_idiom_indices:
                    true_idiom_indices.extend(current_idiom_indices)
                
                # For predictions
                word_idx = -1
                pred_idiom_indices = []
                current_idiom_indices = []
                previous_tag = 0  # O tag
                
                for i, (token, mask, pred) in enumerate(zip(tokens, seq_mask, seq_preds)):
                    if mask == 0 or token in ['<s>', '</s>', '<pad>']:
                        continue
                        
                    if not token.startswith('▁'):  # Not part of a word
                        continue
                        
                    word_idx += 1
                    
                    # Handle end of previous idiom
                    if previous_tag in [1, 2] and pred.item() not in [1, 2]:
                        if current_idiom_indices:
                            pred_idiom_indices.extend(current_idiom_indices)
                            current_idiom_indices = []
                    
                    # Handle new idiom
                    if pred.item() == 1:  # B-IDIOM
                        current_idiom_indices = [word_idx]
                    elif pred.item() == 2:  # I-IDIOM
                        if previous_tag in [1, 2]:  # Continue idiom
                            current_idiom_indices.append(word_idx)
                    
                    previous_tag = pred.item()
                
                # Don't forget last idiom
                if current_idiom_indices:
                    pred_idiom_indices.extend(current_idiom_indices)
                
                # Store results
                predictions.append(pred_idiom_indices)
                ground_truth.append(true_idiom_indices)
    
    # Calculate metrics
    val_loss = val_loss / total_batches if total_batches > 0 else 0
    
    # Calculate F1 score like scoring.py file calculation style
    f1_scores = []
    precision_values = []
    recall_values = []
    
    for pred, gold in zip(predictions, ground_truth):
        # Special case for ground truth [-1]
        if gold == [-1]:
            # Prediction must also be exactly [-1]
            if pred == [-1]:
                f1_scores.append(1.0)
                precision_values.append(1.0)
                recall_values.append(1.0)
            else:
                f1_scores.append(0.0)
                precision_values.append(0.0)
                recall_values.append(0.0)
        else:
            # Convert indices into sets for comparison
            pred_set = set(pred) if isinstance(pred, list) else set()
            gold_set = set(gold) if isinstance(gold, list) else set()
            
            # Calculate precision, recall, and F1 score
            intersection = len(pred_set & gold_set)
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
            recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            
            f1_scores.append(f1)
            precision_values.append(precision)
            recall_values.append(recall)
    
    # Compute mean scores
    f1 = np.mean(f1_scores) if f1_scores else 0
    precision = np.mean(precision_values) if precision_values else 0
    recall = np.mean(recall_values) if recall_values else 0
    
    metrics = {
        'val_loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics


def predict_idioms(model, tokenizer, sentence, device):
    model.eval()
    
    # Tokenize sentence
    encoding = tokenizer(
        sentence,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

    # Map predictions back to words
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    
    # Process the sentence to get idiom indices
    word_idx = -1
    idiom_indices = []
    current_idiom_indices = []
    previous_tag = 0
    
    for i, (token, mask, pred) in enumerate(zip(tokens, attention_mask[0], outputs['predictions'][0])):
        if mask == 0 or token in ['<s>', '</s>', '<pad>']:
            continue
            
        if token.startswith('▁'):  # New word
            word_idx += 1
            
            # Handle end of previous idiom
            if previous_tag in [1, 2] and pred.item() not in [1, 2]:
                # End of idiom
                if current_idiom_indices:
                    idiom_indices.extend(current_idiom_indices)
                    current_idiom_indices = []
            
            # Handle new idiom
            if pred.item() == 1:  # B-IDIOM
                current_idiom_indices = [word_idx]
            elif pred.item() == 2:  # I-IDIOM
                if previous_tag in [1, 2]:  # Continue idiom
                    current_idiom_indices.append(word_idx)
            
            previous_tag = pred.item()
    
    # Don't forget last idiom
    if current_idiom_indices:
        idiom_indices.extend(current_idiom_indices)
    
    return sentence, idiom_indices