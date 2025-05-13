import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from enhanced_model import EnhancedBertForIdiomDetection
from training_utils import (
    HardNegativeDataset,
    FocalLoss,
    get_learning_rate_schedule,
    train_epoch,
    evaluate
)
import argparse
import json
from tqdm import tqdm

def load_data(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def prepare_batch(batch, tokenizer, max_length=128):
    texts = [item['text'] for item in batch]
    tokenized = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # Convert BIO to BIOES tags
    labels = []
    sentence_labels = []
    for item in batch:
        # Convert BIO tags to BIOES
        bioes_tags = convert_bio_to_bioes(item['tags'])
        # Pad or truncate labels
        if len(bioes_tags) < max_length:
            bioes_tags.extend([0] * (max_length - len(bioes_tags)))
        else:
            bioes_tags = bioes_tags[:max_length]
        labels.append(bioes_tags)
        sentence_labels.append(1 if any(tag != 0 for tag in item['tags']) else 0)
    
    tokenized['labels'] = torch.tensor(labels)
    tokenized['sentence_labels'] = torch.tensor(sentence_labels)
    return tokenized

def main(args):
    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = EnhancedBertForIdiomDetection.from_pretrained(
        'bert-base-uncased',
        num_labels=5,  # BIOES tags
        lstm_hidden_size=256,
        lstm_layers=2,
        dropout=0.1,
        token_loss_weight=0.7,
        sentence_loss_weight=0.3
    )
    
    # Load data
    train_data = load_data(args.train_data)
    val_data = load_data(args.val_data)
    
    # Create datasets
    train_dataset = HardNegativeDataset(
        train_data,
        model,
        tokenizer,
        args.device,
        num_hard_negatives=args.num_hard_negatives
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda x: prepare_batch(x, tokenizer)
    )
    
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=lambda x: prepare_batch(x, tokenizer)
    )
    
    # Initialize optimizer and scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    num_training_steps = len(train_loader) * args.num_epochs
    scheduler = get_learning_rate_schedule(
        optimizer,
        num_training_steps,
        warmup_steps=int(num_training_steps * args.warmup_ratio)
    )
    
    # Initialize focal loss for sentence-level classification
    focal_loss = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    
    # Training loop
    best_val_loss = float('inf')
    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            args.device,
            focal_loss
        )
        print(f"Training Loss: {train_loss:.4f}")
        
        # Evaluate
        metrics = evaluate(model, val_loader, args.device, focal_loss)
        print(f"Validation Loss: {metrics['loss']:.4f}")
        print(f"Token Accuracy: {metrics['token_accuracy']:.4f}")
        print(f"Sentence Accuracy: {metrics['sentence_accuracy']:.4f}")
        
        # Save best model
        if metrics['loss'] < best_val_loss:
            best_val_loss = metrics['loss']
            torch.save(model.state_dict(), args.output_dir + '/best_model.pt')
            print("Saved best model!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--num_hard_negatives", type=int, default=2)
    parser.add_argument("--focal_alpha", type=float, default=0.25)
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    main(args) 