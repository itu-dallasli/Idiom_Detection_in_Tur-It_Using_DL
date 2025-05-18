import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
import re
from collections import defaultdict
 


class EnhancedBertForIdiomDetection(nn.Module):
    def __init__(self, 
                 model_name="bert-base-multilingual-cased", 
                 num_labels=3,
                 lstm_hidden_size=384,
                 lstm_layers=2,
                 lstm_dropout=0.3,
                 hidden_dropout=0.3,
                 use_layer_norm=True,
                 freeze_bert_layers=0,
                 use_char_embeddings=True,  # New: character-level embeddings
                 use_pos_embeddings=True,   # New: POS tag embeddings
                 char_embedding_dim=32,     # New: dimension for char embeddings
                 pos_embedding_dim=32):     # New: dimension for POS embeddings
        super(EnhancedBertForIdiomDetection, self).__init__()
        
        # Pre-trained BERT model
        self.bert = BertModel.from_pretrained(model_name)
        
        # Freeze specified number of BERT layers
        if freeze_bert_layers > 0:
            modules = [self.bert.embeddings]
            modules.extend(self.bert.encoder.layer[:freeze_bert_layers])
            for module in modules:
                for param in module.parameters():
                    param.requires_grad = False
        
        # Character-level embeddings (if enabled)
        self.use_char_embeddings = use_char_embeddings
        if use_char_embeddings:
            self.char_embeddings = nn.Embedding(128, char_embedding_dim)  # ASCII characters
            self.char_cnn = nn.Sequential(
                nn.Conv1d(char_embedding_dim, char_embedding_dim, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2)
            )
        
        # POS tag embeddings (if enabled)
        self.use_pos_embeddings = use_pos_embeddings
        if use_pos_embeddings:
            self.pos_embeddings = nn.Embedding(50, pos_embedding_dim)  # Assuming max 50 POS tags
        
        # Calculate input size for LSTM
        lstm_input_size = self.bert.config.hidden_size
        if use_char_embeddings:
            lstm_input_size += char_embedding_dim
        if use_pos_embeddings:
            lstm_input_size += pos_embedding_dim
        
        # Add a BiLSTM layer to capture context
        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        
        # Classification layers
        self.dropout = nn.Dropout(hidden_dropout)
        self.dense = nn.Linear(lstm_hidden_size*2, lstm_hidden_size)
        self.activation = nn.GELU()  # Changed from ReLU to GELU
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(lstm_hidden_size)
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
    def get_char_embeddings(self, tokens):
        """Convert tokens to character-level embeddings"""
        batch_size = len(tokens)
        max_word_len = max(len(token) for token in tokens)
        char_ids = torch.zeros(batch_size, max_word_len, dtype=torch.long)
        
        for i, token in enumerate(tokens):
            for j, char in enumerate(token):
                if j < max_word_len:
                    char_ids[i, j] = ord(char)
        
        char_embeds = self.char_embeddings(char_ids)
        char_embeds = char_embeds.transpose(1, 2)  # [batch, char_dim, seq_len]
        char_embeds = self.char_cnn(char_embeds)
        char_embeds = char_embeds.transpose(1, 2)  # [batch, seq_len, char_dim]
        
        return char_embeds
    
    def forward(self, input_ids, attention_mask, labels=None, pos_tags=None):
        # BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Get additional features
        batch_size, seq_len, _ = sequence_output.size()
        additional_features = []
        
        if self.use_char_embeddings:
            tokens = [self.tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]
            char_embeds = self.get_char_embeddings(tokens)
            additional_features.append(char_embeds)
        
        if self.use_pos_embeddings and pos_tags is not None:
            pos_embeds = self.pos_embeddings(pos_tags)
            additional_features.append(pos_embeds)
        
        # Concatenate additional features
        if additional_features:
            sequence_output = torch.cat([sequence_output] + additional_features, dim=-1)
        
        # BiLSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # Classification layers
        x = self.dropout(lstm_output)
        x = self.dense(x)
        x = self.activation(x)
        if self.use_layer_norm:
            x = self.norm(x)
        x = self.dropout(x)
        emissions = self.classifier(x)
        
        # CRF logic
        loss = None
        if labels is not None:
            crf_mask = attention_mask.bool()
            loss = -self.crf(emissions, labels, mask=crf_mask, reduction='mean')
        
        predictions = self.crf.decode(emissions, mask=attention_mask.bool())
        max_len = emissions.size(1)
        pred_tensor = torch.zeros_like(input_ids)
        for i, pred_seq in enumerate(predictions):
            pred_tensor[i, :len(pred_seq)] = torch.tensor(pred_seq, device=pred_tensor.device)
        
        return {
            'loss': loss,
            'logits': emissions,
            'predictions': pred_tensor
        }

def preprocess_text(text):
    """Enhanced text preprocessing"""
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    # Handle special characters
    text = re.sub(r'[^\w\s]', ' ', text)
    
    # Handle numbers
    text = re.sub(r'\d+', 'NUM', text)
    
    return text

def post_process_bio_tags(tokens, tags, token_is_first_subword):
    """
    Enhanced post-processing of BIO tags with additional rules
    """
    corrected_tags = tags.copy()
    
    # Rule 1: Fix I-IDIOM without preceding B-IDIOM
    for i in range(len(tags)):
        if not token_is_first_subword[i]:
            continue
            
        if i > 0 and tags[i] == 2 and tags[i-1] == 0:
            if i < len(tags)-1 and tags[i+1] == 2:
                corrected_tags[i] = 1
            else:
                corrected_tags[i] = 0
    
    # Rule 2: Fix consecutive B-IDIOM tags
    for i in range(len(tags)-1):
        if not token_is_first_subword[i] or not token_is_first_subword[i+1]:
            continue
            
        if tags[i] == 1 and tags[i+1] == 1:
            corrected_tags[i+1] = 2
    
    # Rule 4: Fix broken MWEs
    for i in range(len(tags)-2):
        if not all(token_is_first_subword[j] for j in range(i, i+3)):
            continue
            
        if tags[i] == 1 and tags[i+1] == 0 and tags[i+2] == 2:
            # Likely a broken MWE, fix middle token
            corrected_tags[i+1] = 2
    
    # Rule 5: Handle common MWE patterns
    for i in range(len(tags)-1):
        if not token_is_first_subword[i] or not token_is_first_subword[i+1]:
            continue
            
        # Check for common MWE patterns (e.g., "take into account")
        if tokens[i].lower() in ['take', 'make', 'put', 'get'] and tokens[i+1].lower() in ['into', 'up', 'down', 'out']:
            if tags[i] == 0 and tags[i+1] == 0:
                corrected_tags[i] = 1
                corrected_tags[i+1] = 2
    
    return corrected_tags

def apply_post_processing(model, tokenizer, input_ids, attention_mask, device):
    """Enhanced post-processing"""
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    preds = outputs['predictions'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    masks = attention_mask[0]
    
    # Identify which tokens are first subwords
    token_is_first_subword = [not token.startswith('##') for token in tokens]
    
    # Apply enhanced post-processing
    corrected_preds = post_process_bio_tags(tokens, preds.tolist(), token_is_first_subword)
    
    # Map back to tensor
    corrected_tensor = torch.tensor(corrected_preds, device=preds.device)
    
    return corrected_tensor, tokens, masks

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
                    if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    if not token.startswith('##'):  # New word
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
                    if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
                        continue
                        
                    if not token.startswith('##'):  # New word
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
                
                # Debug print for a few examples
                if len(predictions) <= 5:
                    print("\nExample:")
                    print("Tokens:", tokens)
                    print("True BIO tags:", seq_labels.tolist())
                    print("Pred BIO tags:", seq_preds.tolist())
                    print("True idiom indices:", true_idiom_indices)
                    print("Pred idiom indices:", pred_idiom_indices)
    
    # Calculate average loss
    avg_val_loss = val_loss / max(1, total_batches)
    
    # Calculate F1 scores using competition method
    f1_scores = []
    for pred, gold in zip(predictions, ground_truth):
        # Handle special case for no idiom
        if not gold:  # Empty gold = no idiom
            if not pred:  # Empty pred = correctly predicted no idiom
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
            continue
            
        # Normal case - set comparison
        pred_set = set(pred)
        gold_set = set(gold)
        
        intersection = len(pred_set & gold_set)
        precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
        recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        f1_scores.append(f1)
    
    mean_f1 = sum(f1_scores) / max(1, len(f1_scores))
    
    print(f"\nValidation Loss: {avg_val_loss:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")
    
    return {
        'loss': avg_val_loss,
        'f1': mean_f1,
        'predictions': predictions,
        'ground_truth': ground_truth
    }

def train_model(train_loader, val_loader, tokenizer, model=None, epochs=10, lr=2e-5, 
                weight_decay=0.01, lr_multiplier=10, patience=3):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create model if not provided
    if model is None:
        model = EnhancedBertForIdiomDetection().to(device)
    
    # Differential learning rates
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' in n and p.requires_grad],
            'weight_decay': weight_decay,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' in n and p.requires_grad],
            'weight_decay': 0.0,
            'lr': lr
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and 'bert' not in n],
            'weight_decay': weight_decay,
            'lr': lr * lr_multiplier
        },
        {
            'params': [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and 'bert' not in n],
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
        pct_start=0.1  # 10% warmup
    )
    
    best_f1 = 0
    no_improve_epochs = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        
        # Training
        model.train()
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs['loss']
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Evaluation
        metrics = evaluate(model, val_loader, tokenizer, device)
        
        print(f"Epoch {epoch+1}:")
        print(f"Training Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"F1 Score: {metrics['f1']:.4f}")
        
        # Save best model and check for early stopping
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            torch.save(model.state_dict(), "best_idiom_model.pt")
            print("New best model saved!")
            no_improve_epochs = 0
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs} epochs")
            
            if no_improve_epochs >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load the best model weights
    model.load_state_dict(torch.load("best_idiom_model.pt"))
    return model

def predict_idioms_with_postprocessing(model, tokenizer, sentence, device):
    model.eval()
    
    # Tokenize
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        truncation=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt"
    )
    
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    
    # Apply model and post-processing
    corrected_preds, tokens, masks = apply_post_processing(
        model, tokenizer, input_ids, attention_mask, device
    )
    
    # Convert to word-level predictions (same as your original function, but using corrected_preds)
    words = []
    bio_tags = []
    word_idx = -1
    idiom_indices = []
    current_idiom = []
    current_word = ""
    previous_tag = 0  # O tag
    
    for token, mask, pred in zip(tokens, masks, corrected_preds):
        if mask == 0 or token in ['[CLS]', '[SEP]', '[PAD]']:
            continue
            
        if not token.startswith('##'):  # New word
            # Save previous word
            if current_word:
                words.append(current_word)
                bio_tags.append(previous_tag)
                word_idx += 1
                
                # Handle idiom tracking
                if previous_tag in [1, 2] and pred.item() not in [1, 2]:  # End of idiom
                    if current_idiom:
                        idiom_indices.extend(current_idiom)
                        current_idiom = []
            
            # Start new word
            current_word = token
            previous_tag = pred.item()
            
            # Track idioms
            if pred.item() == 1:  # B-IDIOM
                current_idiom = [word_idx + 1]  # +1 because we haven't incremented yet
            elif pred.item() == 2:  # I-IDIOM
                if previous_tag in [1, 2]:  # Continue idiom
                    current_idiom.append(word_idx + 1)
        else:
            # Continue current word
            current_word += token[2:]  # Remove ## prefix
    
    # Don't forget last word
    if current_word:
        words.append(current_word)
        bio_tags.append(previous_tag)
        
        # Handle last idiom
        if previous_tag in [1, 2] and current_idiom:
            idiom_indices.extend(current_idiom)
    
    # Format results with BIO tags
    results = []
    for i, (word, tag) in enumerate(zip(words, bio_tags)):
        if tag == 0:
            results.append((word, "O"))
        elif tag == 1:
            results.append((word, "B-IDIOM"))
        elif tag == 2:
            results.append((word, "I-IDIOM"))
    
    return results, idiom_indices
        
def debug_predictions(model, tokenizer, test_sentences, device):
    """
    Debug function to show the complete pipeline of tokenization, prediction, and remapping
    """
    model.eval()
    
    for sentence in test_sentences:
        print("\n" + "="*80)
        print(f"Original sentence: {sentence}")
        
        # Tokenize
        encoding = tokenizer.encode_plus(
            sentence,
            truncation=True,
            padding="max_length",
            max_length=128,
            return_tensors="pt"
        )
        
        # Move to device
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs['logits']
            preds = torch.argmax(logits, dim=2)
        
        # Get tokens
        tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
        
        # Map predictions back to words
        word_idx = -1  # Start at -1 to account for [CLS]
        current_word_preds = []
        word_level_preds = []
        words = []
        current_word = []
        
        print("\nDetailed token analysis:")
        print(f"{'Token':<15} {'Is Subword':<12} {'Prediction':<10} {'Word Index':<10}")
        print("-" * 50)
        
        for token, pred in zip(tokens, preds[0]):
            if token == '[CLS]' or token == '[SEP]' or token == '[PAD]':
                print(f"{token:<15} {'N/A':<12} {pred.item():<10} {'N/A':<10}")
                continue
                
            is_subword = token.startswith('##')
            
            if not is_subword:  # New word
                # Save prediction for previous word
                if current_word_preds:
                    if 1 in current_word_preds:
                        word_level_preds.append(word_idx)
                    words.append(''.join(current_word))
                word_idx += 1
                current_word_preds = [pred.item()]
                current_word = [token]
            else:  # Subword
                current_word_preds.append(pred.item())
                current_word.append(token[2:])  # Remove ## prefix
                
            print(f"{token:<15} {str(is_subword):<12} {pred.item():<10} {word_idx:<10}")
        
        # Handle last word
        if current_word_preds and 1 in current_word_preds:
            word_level_preds.append(word_idx)
        if current_word:
            words.append(''.join(current_word))
            
        print("\nFinal Analysis:")
        print("Reconstructed words:", words)
        print("Word-level predictions:", word_level_preds)
        print("Predicted idiom words:", [words[i] for i in word_level_preds])
