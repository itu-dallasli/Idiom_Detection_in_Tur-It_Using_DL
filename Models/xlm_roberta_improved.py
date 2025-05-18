import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm


class XLMRobertaForIdiomDetection(nn.Module):
    def __init__(self, model_name="xlm-roberta-large", num_labels=3):  # 3 labels: O, B-IDIOM, I-IDIOM
        super(XLMRobertaForIdiomDetection, self).__init__()
        
        # Pre-trained XLM-RoBERTa model
        self.roberta = XLMRobertaModel.from_pretrained(model_name)
        
        # Add a BiLSTM layer to capture context
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

    # Add this method to your XLMRobertaForIdiomDetection class
    def increase_dropout(self, dropout_rate=0.3):
        """Increase dropout rate in all applicable layers"""
        self.dropout = nn.Dropout(dropout_rate)
        # Modify LSTM dropout if it exists
        if hasattr(self, 'lstm'):
            # Save original hidden size and bidirectional settings
            hidden_size = self.lstm.hidden_size
            bidirectional = self.lstm.bidirectional
            num_layers = self.lstm.num_layers
            
            # Recreate LSTM with higher dropout
            self.lstm = nn.LSTM(
                input_size=self.roberta.config.hidden_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout_rate
            )
        return self
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Get XLM-RoBERTa outputs
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get token-level representations
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
        
        # CRF decoding for predictions
        if self.training or labels is None:
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            # Convert list of lists to tensor with padding
            max_len = emissions.size(1)
            pred_tensor = torch.zeros_like(input_ids)
            for i, pred_seq in enumerate(predictions):
                pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        else:
            # During evaluation, use CRF decoding
            predictions = self.crf.decode(emissions, mask=attention_mask.bool())
            # Convert list of lists to tensor with padding
            max_len = emissions.size(1)
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
                        # End of idiom
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
                
                # Don't forget last idiom
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
                        # End of idiom
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
    
    # Calculate F1 score using the same logic as in scoring.py
    f1_scores = []
    
    for pred, gold in zip(predictions, ground_truth):
        # Special case for ground truth [-1]
        if gold == [-1]:
            # Prediction must also be exactly [-1]
            if pred == [-1]:
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
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
    
    # Compute mean F1 score
    f1 = np.mean(f1_scores) if f1_scores else 0
    
    # Calculate precision and recall for overall metrics
    total_predictions = sum(len(set(pred)) for pred in predictions)
    total_ground_truth = sum(len(set(gold)) for gold in ground_truth)
    
    total_correct = 0
    for pred, gold in zip(predictions, ground_truth):
        pred_set = set(pred)
        gold_set = set(gold)
        total_correct += len(pred_set & gold_set)
    
    precision = total_correct / total_predictions if total_predictions > 0 else 0
    recall = total_correct / total_ground_truth if total_ground_truth > 0 else 0
    
    metrics = {
        'val_loss': val_loss,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    
    return metrics



def train_model(train_loader, val_loader, tokenizer, epochs=10, lr=2e-5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Set seeds for reproducibility
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    model = XLMRobertaForIdiomDetection()
    
    # Increase dropout in the model
    model.dropout = nn.Dropout(0.2)  # Increased from default 0.2
    
    model.to(device)
    
    # Optimizer with weight decay (L2 regularization)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    from transformers import get_scheduler
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),  # 10% warmup
        num_training_steps=total_steps
    )
    
    best_f1 = 0.0
    patience = 3  # For early stopping
    no_improvement = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Training]")
        for batch in progress_bar:
            if batch['input_ids'].size(0) == 0:
                continue
                
            train_batches += 1
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Validation
        metrics = evaluate(model, val_loader, tokenizer, device)
        print(f"Validation loss: {metrics['val_loss']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), 'best_xlm_roberta_idiom_model.pt')
            print(f"New best model saved with F1: {best_f1:.4f}")
            no_improvement = 0  # Reset counter
        else:
            no_improvement += 1
            print(f"No improvement for {no_improvement} epochs")
            
            # Early stopping
            if no_improvement >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    return model


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
        
    preds = outputs['predictions'][0].tolist()  # Get predictions for the single sentence
    
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


def debug_predictions(model, tokenizer, test_sentences, device):
    for sentence in test_sentences:
        processed_sentence, idiom_indices = predict_idioms(model, tokenizer, sentence, device)
        
        print("\nSentence:", sentence)
        print("Detected Idiom Indices:", idiom_indices)
        
        if idiom_indices:
            words = sentence.split()
            idiom_words = [words[idx] for idx in idiom_indices]
            print("Idiom Words:", " ".join(idiom_words))
        else:
            print("No idiom detected.")