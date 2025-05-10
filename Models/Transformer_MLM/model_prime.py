import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torchcrf import CRF
import numpy as np
from tqdm import tqdm

class MWEMaskedAttentionModel(nn.Module):
    def __init__(self, 
                 model_name="xlm-roberta-base",
                 num_labels=3,
                 hidden_size=768,
                 num_attention_heads=8,
                 attention_dropout=0.2,
                 hidden_dropout=0.4,
                 freeze_bert_layers=6,
                 mlm_probability=0.15):
        super(MWEMaskedAttentionModel, self).__init__()
        
        # Base transformer model (XLM-RoBERTa)
        self.transformer = AutoModel.from_pretrained(model_name)
        
        # Freeze specified number of transformer layers
        if freeze_bert_layers > 0:
            modules = [self.transformer.embeddings]
            modules.extend(self.transformer.encoder.layer[:freeze_bert_layers])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        # MLM head for masked language modeling
        self.mlm_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, self.transformer.config.vocab_size)
        )
        
        # Multi-head attention for MWE detection
        self.mwe_attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Dropout layers
        self.dropout = nn.Dropout(hidden_dropout)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LayerNorm(hidden_size // 2),
            nn.GELU(),
            nn.Dropout(hidden_dropout),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # CRF layer for sequence labeling
        self.crf = CRF(num_labels, batch_first=True)
        
        # MLM probability
        self.mlm_probability = mlm_probability
        
    def get_mlm_mask(self, input_ids, attention_mask):
        """Generate mask for MLM training"""
        mask = torch.zeros_like(input_ids, dtype=torch.bool)
        for i in range(input_ids.size(0)):
            # Get indices of tokens that can be masked (not special tokens)
            valid_indices = torch.where(
                (input_ids[i] != self.transformer.config.pad_token_id) &
                (input_ids[i] != self.transformer.config.cls_token_id) &
                (input_ids[i] != self.transformer.config.sep_token_id)
            )[0]
            
            # Randomly select tokens to mask
            num_tokens_to_mask = int(len(valid_indices) * self.mlm_probability)
            if num_tokens_to_mask > 0:
                indices_to_mask = valid_indices[torch.randperm(len(valid_indices))[:num_tokens_to_mask]]
                mask[i, indices_to_mask] = True
        
        return mask
    
    def forward(self, input_ids, attention_mask, labels=None):
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Generate MLM mask
        mlm_mask = self.get_mlm_mask(input_ids, attention_mask)
        
        # MLM loss
        mlm_loss = None
        if labels is not None:
            mlm_logits = self.mlm_head(sequence_output)
            mlm_loss = nn.CrossEntropyLoss()(
                mlm_logits[mlm_mask].view(-1, self.transformer.config.vocab_size),
                input_ids[mlm_mask].view(-1)
            )
        
        # Apply multi-head attention
        attn_output, _ = self.mwe_attention(
            sequence_output,
            sequence_output,
            sequence_output,
            key_padding_mask=~attention_mask.bool()
        )
        
        # Add & Norm
        x = self.layer_norm1(sequence_output + attn_output)
        
        # Feed-forward network
        x = self.dropout(x)
        x = self.layer_norm2(x)
        
        # Classification
        emissions = self.classifier(x)
        
        # CRF logic
        crf_loss = None
        if labels is not None:
            crf_mask = attention_mask.bool()
            crf_loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
        
        # Combine losses if training
        loss = None
        if labels is not None:
            loss = crf_loss + 0.1 * mlm_loss  # Weight MLM loss less than CRF loss
        
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        max_len = emissions.size(1)
        pred_tensor = torch.zeros_like(input_ids)
        for i, pred_seq in enumerate(predictions):
            pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        
        return {
            'loss': loss,
            'logits': emissions,
            'predictions': pred_tensor,
            'mlm_loss': mlm_loss,
            'crf_loss': crf_loss
        }

def train_model(model, train_loader, val_loader, tokenizer, 
                epochs=10, lr=5e-6, weight_decay=0.05, 
                lr_multiplier=5, patience=3, min_loss_change=0.0005):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Differential learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'transformer' in n and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'transformer' in n and p.requires_grad],
            'weight_decay': 0.0,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'transformer' not in n],
            'weight_decay': weight_decay,
            'lr': lr * lr_multiplier
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'transformer' not in n],
            'weight_decay': 0.0,
            'lr': lr * lr_multiplier
        }
    ]
    
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
    
    # Learning rate scheduler with warmup
    total_steps = len(train_loader) * epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=[lr, lr, lr*lr_multiplier, lr*lr_multiplier],
        total_steps=total_steps,
        pct_start=0.1,
        div_factor=25.0,
        final_div_factor=1000.0
    )
    
    best_f1 = 0
    best_loss = float('inf')
    no_improve_epochs = 0
    no_loss_change_epochs = 0
    previous_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        total_mlm_loss = 0
        total_crf_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if batch['input_ids'].size(0) == 0:
                continue
                
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs['loss']
            mlm_loss = outputs['mlm_loss']
            crf_loss = outputs['crf_loss']
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            if mlm_loss is not None:
                total_mlm_loss += mlm_loss.item()
            if crf_loss is not None:
                total_crf_loss += crf_loss.item()
        
        avg_loss = total_loss / len(train_loader)
        avg_mlm_loss = total_mlm_loss / len(train_loader)
        avg_crf_loss = total_crf_loss / len(train_loader)
        
        print(f"\nEpoch {epoch+1} Training Loss: {avg_loss:.4f}")
        print(f"MLM Loss: {avg_mlm_loss:.4f}")
        print(f"CRF Loss: {avg_crf_loss:.4f}")
        
        # Validation
        model.eval()
        val_loss = 0
        predictions = []
        ground_truth = []
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                if batch['input_ids'].size(0) == 0:
                    continue
                    
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                if outputs['loss'] is not None:
                    val_loss += outputs['loss'].item()
                
                preds = outputs['predictions']
                
                # Process predictions
                for seq_preds, seq_mask, seq_labels in zip(preds, attention_mask, labels):
                    valid_preds = seq_preds[seq_mask.bool()].cpu().numpy()
                    valid_labels = seq_labels[seq_mask.bool()].cpu().numpy()
                    predictions.extend(valid_preds)
                    ground_truth.extend(valid_labels)
        
        val_loss /= len(val_loader)
        print(f"Validation Loss: {val_loss:.4f}")
        
        # Calculate metrics
        metrics = calculate_metrics(ground_truth, predictions)
        print(f"Validation Metrics:")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Early stopping based on F1 score
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_loss = val_loss
            torch.save(model.state_dict(), "best_mwe_model.pt")
            print("New best model saved!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No F1 improvement for {no_improve_epochs} epochs")
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
        
        # Early stopping based on loss change
        if abs(previous_loss - val_loss) < min_loss_change:
            no_loss_change_epochs += 1
            print(f"No significant loss change for {no_loss_change_epochs} epochs")
            
            if no_loss_change_epochs >= patience:
                print(f"Early stopping due to minimal loss change at epoch {epoch+1}")
                break
        else:
            no_loss_change_epochs = 0
        
        previous_loss = val_loss
    
    # Load the best model weights
    model.load_state_dict(torch.load("best_mwe_model.pt"))
    return model

def calculate_metrics(ground_truth, predictions):
    """Calculate precision, recall, and F1 score for MWE detection"""
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    # Convert to numpy arrays if they aren't already
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    
    # Count true positives, false positives, and false negatives
    for gt, pred in zip(ground_truth, predictions):
        if gt == 1 and pred == 1:  # B-IDIOM
            true_positives += 1
        elif gt == 2 and pred == 2:  # I-IDIOM
            true_positives += 1
        elif gt == 0 and pred in [1, 2]:  # False positive
            false_positives += 1
        elif gt in [1, 2] and pred == 0:  # False negative
            false_negatives += 1
    
    # Calculate metrics
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def predict_mwe(model, tokenizer, sentence, device, max_length=128):
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
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        predictions = outputs['predictions']
    
    # Convert predictions to tokens and their labels
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    preds = predictions[0].cpu().numpy()
    
    # Process predictions
    words = []
    word_labels = []
    current_word = []
    current_label = None
    
    for token, pred in zip(tokens, preds):
        if token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if not token.startswith('##'):
            if current_word:
                words.append(''.join(current_word))
                word_labels.append(current_label)
            current_word = [token]
            current_label = pred
        else:
            current_word.append(token[2:])
    
    if current_word:
        words.append(''.join(current_word))
        word_labels.append(current_label)
    
    return words, word_labels
