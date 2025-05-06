import torch
from torchcrf import CRF
import torch.nn as nn
from transformers import BertModel
from sklearn.metrics import classification_report, accuracy_score, f1_score
import numpy as np
from tqdm import tqdm
 


class EnhancedBertForIdiomDetection(nn.Module):
    def __init__(self, 
                 model_name="bert-base-multilingual-cased", 
                 num_labels=3,
                 lstm_hidden_size=384,
                 lstm_layers=2,
                 lstm_dropout=0.3,
                 hidden_dropout=0.3,
                 use_layer_norm=True,
                 freeze_bert_layers=0,  # Number of BERT layers to freeze
                 cnn_filters=[128, 256],  # Number of filters for each CNN layer
                 cnn_kernel_sizes=[2, 3],  # Kernel sizes for each CNN layer
                 cnn_dropout=0.2):  # Dropout for CNN layers
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
        
        # CNN layers
        self.cnn_layers = nn.ModuleList()
        input_channels = self.bert.config.hidden_size
        
        for i in range(len(cnn_filters)):
            self.cnn_layers.append(nn.Sequential(
                nn.Conv1d(
                    in_channels=input_channels if i == 0 else cnn_filters[i-1],
                    out_channels=cnn_filters[i],
                    kernel_size=cnn_kernel_sizes[i],
                    padding=cnn_kernel_sizes[i]//2  # Same padding
                ),
                nn.ReLU(),
                nn.Dropout(cnn_dropout)
            ))
        
        # Add a BiLSTM layer to capture context
        self.lstm = nn.LSTM(
            input_size=cnn_filters[-1],  # Use last CNN layer's output size
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=lstm_dropout if lstm_layers > 1 else 0
        )
        
        # Classification layers
        self.dropout = nn.Dropout(hidden_dropout)
        self.dense = nn.Linear(lstm_hidden_size*2, lstm_hidden_size)
        self.activation = nn.ReLU()
        self.use_layer_norm = use_layer_norm
        if use_layer_norm:
            self.norm = nn.LayerNorm(lstm_hidden_size)
        self.classifier = nn.Linear(lstm_hidden_size, num_labels)
        
        # CRF layer
        self.crf = CRF(num_labels, batch_first=True)
        
    def forward(self, input_ids, attention_mask, labels=None):
        # BERT outputs
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Apply CNN layers
        x = sequence_output.transpose(1, 2)  # [batch_size, hidden_size, seq_len]
        for cnn_layer in self.cnn_layers:
            x = cnn_layer(x)
        x = x.transpose(1, 2)  # [batch_size, seq_len, cnn_filters[-1]]
        
        # BiLSTM
        lstm_output, _ = self.lstm(x)
        
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

def post_process_bio_tags(tokens, tags, token_is_first_subword):
    """
    Apply linguistic rules to fix common errors in BIO tag sequences
    
    Parameters:
    - tokens: List of tokens (including subtokens)
    - tags: Predicted BIO tags (0=O, 1=B-IDIOM, 2=I-IDIOM)
    - token_is_first_subword: Boolean list indicating if a token is the first subword of a word
    
    Returns:
    - Corrected BIO tags
    """
    corrected_tags = tags.copy()
    
    # Rule 1: Fix I-IDIOM without preceding B-IDIOM
    for i in range(len(tags)):
        if not token_is_first_subword[i]:
            continue  # Skip subtokens
            
        if i > 0 and tags[i] == 2 and tags[i-1] == 0:  # I-IDIOM after O
            # Either correct to B-IDIOM or O
            if i < len(tags)-1 and tags[i+1] == 2:  # If followed by I-IDIOM
                corrected_tags[i] = 1  # Convert to B-IDIOM
            else:
                corrected_tags[i] = 0  # Convert to O if isolated
    
    # Rule 2: Fix B-IDIOM followed by O (when it should likely be followed by I-IDIOM)
    for i in range(len(tags)-1):
        if not token_is_first_subword[i] or not token_is_first_subword[i+1]:
            continue  # Skip subtokens
            
        if tags[i] == 1 and tags[i+1] == 0:  # B-IDIOM followed by O
            # Check if this is likely part of a longer expression
            # For now, keep as is (could be a single-token MWE)
            pass
    
    # Rule 3: Fix consecutive B-IDIOM tags (should usually be B-IDIOM followed by I-IDIOM)
    for i in range(len(tags)-1):
        if not token_is_first_subword[i] or not token_is_first_subword[i+1]:
            continue  # Skip subtokens
            
        if tags[i] == 1 and tags[i+1] == 1:  # B-IDIOM followed by B-IDIOM
            corrected_tags[i+1] = 2  # Convert second to I-IDIOM
    
    # Rule 4: Handle very short MWEs (could be false positives)
    # This requires more context and depends on your specific data
    
    return corrected_tags

def apply_post_processing(model, tokenizer, input_ids, attention_mask, device):
    """Apply model prediction and post-processing to input sequence"""
    # Get model predictions
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    
    preds = outputs['predictions'][0]
    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    masks = attention_mask[0]
    
    # Identify which tokens are first subwords
    token_is_first_subword = [not token.startswith('##') for token in tokens]
    
    # Apply post-processing
    corrected_preds = post_process_bio_tags(tokens, preds.tolist(), token_is_first_subword)
    
    # Map back to tensor
    corrected_tensor = torch.tensor(corrected_preds, device=preds.device)
    
    return corrected_tensor, tokens, masks


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

def hyperparameter_search(train_loader, val_loader, tokenizer, language='italian', n_trials=10):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    best_f1 = 0
    best_config = None
    
    # Choose appropriate pre-trained model based on language
    if language == 'italian':
        base_model = "dbmdz/bert-base-italian-xxl-cased"  # Or "dbmdz/bert-base-italian-cased" if memory is an issue
    elif language == 'turkish':
        base_model = "dbmdz/bert-base-turkish-cased"
    else:
        base_model = "bert-base-multilingual-cased"
    
    # Define hyperparameter search space
    param_grid = {
        'learning_rate': [1e-5, 2e-5, 3e-5, 5e-5],
        'lstm_hidden_size': [256, 384, 512],
        'lstm_layers': [1, 2, 3],
        'lstm_dropout': [0.2, 0.3, 0.4],
        'hidden_dropout': [0.1, 0.2, 0.3, 0.4],
        'weight_decay': [0.0, 0.01, 0.05],
        'batch_size': [8, 16, 32],
        'lr_multiplier': [5, 10, 15, 20],
        'use_layer_norm': [True, False],
        'freeze_bert_layers': [0, 3, 6],
        'cnn_filters': [[128, 256], [256, 512], [128, 256, 512]],
        'cnn_kernel_sizes': [[2, 3], [3, 5], [2, 3, 5]],
        'cnn_dropout': [0.2, 0.3, 0.4]
    }
    
    results = []
    
    for trial in range(n_trials):
        # Sample hyperparameters
        config = {
            'learning_rate': np.random.choice(param_grid['learning_rate']),
            'lstm_hidden_size': np.random.choice(param_grid['lstm_hidden_size']),
            'lstm_layers': np.random.choice(param_grid['lstm_layers']),
            'lstm_dropout': np.random.choice(param_grid['lstm_dropout']),
            'hidden_dropout': np.random.choice(param_grid['hidden_dropout']),
            'weight_decay': np.random.choice(param_grid['weight_decay']),
            'batch_size': np.random.choice(param_grid['batch_size']),
            'lr_multiplier': np.random.choice(param_grid['lr_multiplier']),
            'use_layer_norm': np.random.choice(param_grid['use_layer_norm']),
            'freeze_bert_layers': np.random.choice(param_grid['freeze_bert_layers']),
            'cnn_filters': np.random.choice(param_grid['cnn_filters']),
            'cnn_kernel_sizes': np.random.choice(param_grid['cnn_kernel_sizes']),
            'cnn_dropout': np.random.choice(param_grid['cnn_dropout']),
            'model_name': base_model
        }
        
        print(f"\nTrial {trial+1}/{n_trials}")
        print("Configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        
        # Create model with the selected hyperparameters
        model = EnhancedBertForIdiomDetection(
            model_name=config['model_name'],
            lstm_hidden_size=config['lstm_hidden_size'],
            lstm_layers=config['lstm_layers'],
            lstm_dropout=config['lstm_dropout'],
            hidden_dropout=config['hidden_dropout'],
            use_layer_norm=config['use_layer_norm'],
            freeze_bert_layers=config['freeze_bert_layers'],
            cnn_filters=config['cnn_filters'],
            cnn_kernel_sizes=config['cnn_kernel_sizes'],
            cnn_dropout=config['cnn_dropout']
        ).to(device)
        
        # Train for a few epochs to see potential
        model = train_model(
            train_loader=train_loader,
            val_loader=val_loader,
            tokenizer=tokenizer,
            model=model,
            epochs=5,  # Reduced epochs for hyperparameter search
            lr=config['learning_rate'],
            weight_decay=config['weight_decay'],
            lr_multiplier=config['lr_multiplier'],
            patience=2  # Early stopping for efficiency
        )
        
        # Evaluate on validation set
        metrics = evaluate(model, val_loader, tokenizer, device)
        f1_score = metrics['f1']
        
        # Save results
        config['f1_score'] = f1_score
        results.append(config)
        
        print(f"Trial {trial+1} F1 Score: {f1_score:.4f}")
        
        # Update best configuration
        if f1_score > best_f1:
            best_f1 = f1_score
            best_config = config
            print(f"New best configuration! F1: {best_f1:.4f}")
    
    # Sort results by F1 score
    results.sort(key=lambda x: x['f1_score'], reverse=True)
    
    print("\nTop 3 Configurations:")
    for i, config in enumerate(results[:3]):
        print(f"Rank {i+1} (F1: {config['f1_score']:.4f}):")
        for k, v in config.items():
            if k != 'f1_score':
                print(f"  {k}: {v}")
    
    return best_config