import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class HardNegativeDataset(Dataset):
    def __init__(self, original_dataset, model, tokenizer, device, num_hard_negatives=2):
        self.original_dataset = original_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.num_hard_negatives = num_hard_negatives
        self.hard_negatives = self._generate_hard_negatives()

    def _generate_hard_negatives(self):
        hard_negatives = []
        self.model.eval()
        
        with torch.no_grad():
            for item in tqdm(self.original_dataset, desc="Generating hard negatives"):
                input_ids = item['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = item['attention_mask'].unsqueeze(0).to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                sentence_logits = outputs['sentence_logits']
                
                # If model predicts high probability for idiomatic but it's actually literal
                if item['sentence_label'] == 0 and torch.softmax(sentence_logits, dim=1)[0][1] > 0.7:
                    hard_negatives.append(item)
                    
                    if len(hard_negatives) >= self.num_hard_negatives:
                        break
        
        return hard_negatives

    def __len__(self):
        return len(self.original_dataset) + len(self.hard_negatives)

    def __getitem__(self, idx):
        if idx < len(self.original_dataset):
            return self.original_dataset[idx]
        else:
            return self.hard_negatives[idx - len(self.original_dataset)]

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

def get_learning_rate_schedule(optimizer, num_training_steps, warmup_steps=0.1):
    """
    Create a learning rate schedule with warmup and cosine decay
    """
    from torch.optim.lr_scheduler import LambdaLR
    import math

    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * (current_step - warmup_steps) / (num_training_steps - warmup_steps)))

    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, train_loader, optimizer, scheduler, device, focal_loss=None):
    model.train()
    total_loss = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs['loss']
        
        if focal_loss is not None and 'sentence_labels' in batch:
            # Apply focal loss to sentence-level classification
            sentence_loss = focal_loss(outputs['sentence_logits'], batch['sentence_labels'])
            loss = model.token_loss_weight * loss + model.sentence_loss_weight * sentence_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device, focal_loss=None):
    model.eval()
    total_loss = 0
    token_correct = 0
    token_total = 0
    sentence_correct = 0
    sentence_total = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs['loss']
            
            if focal_loss is not None and 'sentence_labels' in batch:
                sentence_loss = focal_loss(outputs['sentence_logits'], batch['sentence_labels'])
                loss = model.token_loss_weight * loss + model.sentence_loss_weight * sentence_loss
            
            total_loss += loss.item()
            
            # Token-level accuracy
            preds = outputs['predictions']
            labels = batch['labels']
            mask = batch['attention_mask'].bool()
            token_correct += ((preds == labels) & mask).sum().item()
            token_total += mask.sum().item()
            
            # Sentence-level accuracy
            if 'sentence_labels' in batch:
                sentence_preds = torch.argmax(outputs['sentence_logits'], dim=1)
                sentence_correct += (sentence_preds == batch['sentence_labels']).sum().item()
                sentence_total += len(batch['sentence_labels'])
    
    metrics = {
        'loss': total_loss / len(val_loader),
        'token_accuracy': token_correct / token_total if token_total > 0 else 0,
        'sentence_accuracy': sentence_correct / sentence_total if sentence_total > 0 else 0
    }
    
    return metrics 