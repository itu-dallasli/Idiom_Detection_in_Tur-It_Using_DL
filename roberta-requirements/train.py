#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for training the XLM-RoBERTa model for idiom detection.
"""

import os
import sys
import torch
import pandas as pd
import argparse
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import XLMRobertaTokenizer, get_scheduler
from pathlib import Path

# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Import project modules
try:
    from dataset.dataset_es import get_dataloaders
    from Models.xlm_roberta_improved import XLMRobertaForIdiomDetection, predict_idioms, evaluate
except ImportError:
    # Add the project root to Python path for imports if run from different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from dataset.dataset_es import get_dataloaders
    from Models.xlm_roberta_improved import XLMRobertaForIdiomDetection, predict_idioms, evaluate

def train_model(args):
    """
    Train the XLM-RoBERTa model for idiom detection.
    
    Args:
        args: Command line arguments
    
    Returns:
        dict: Training results
    """
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "results"), exist_ok=True)
    
    # Define experiment name and paths
    experiment_name = f"xlm_bs{args.batch_size}_lr{args.learning_rate}_ml{args.max_length}_ep{args.epochs}"
    if args.experiment_name:
        experiment_name = args.experiment_name
    
    model_save_path = os.path.join(args.models_dir, "saved_models", f"{experiment_name}.pt")
    results_save_path = os.path.join(args.models_dir, "results", f"{experiment_name}_metrics.csv")
    fig_save_path = os.path.join(args.models_dir, "results", f"{experiment_name}_plot.png")
    
    # Initialize results
    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    
    print(f"\nRunning experiment: {experiment_name}")
    print("Loading data...")
    
    # Get data loaders from dataset.py
    train_loader, val_loader, _ = get_dataloaders(
        train_path=os.path.join(args.data_dir, 'train.csv'),
        val_path=os.path.join(args.data_dir, 'eval.csv'),
        batch_size=args.batch_size,
        max_length=args.max_length,
        tokenizer=tokenizer
    )
    
    # Initialize model
    model = XLMRobertaForIdiomDetection(
        model_name=args.model_name, 
        num_labels=args.num_labels
    )
    model.to(device)
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    total_steps = len(train_loader) * args.epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(args.warmup_ratio * total_steps),
        num_training_steps=total_steps
    )
    
    best_f1 = 0.0
    best_epoch = 0
    no_improve_epochs = 0
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        train_batches = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]")
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        avg_train_loss = train_loss / train_batches if train_batches > 0 else 0
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # Evaluation
        print("Evaluating...")
        metrics = evaluate(model, val_loader, tokenizer, device)
        
        # Store metrics
        results['epoch'].append(epoch + 1)
        results['train_loss'].append(avg_train_loss)
        results['val_loss'].append(metrics['val_loss'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f1'].append(metrics['f1'])
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}")
        print(f"  Val Loss: {metrics['val_loss']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1 Score: {metrics['f1']:.4f}")
        
        # Early stopping logic
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_epoch = epoch + 1
            no_improve_epochs = 0
            torch.save(model.state_dict(), model_save_path)
            print(f"New best model saved with F1: {best_f1:.4f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement in F1 for {no_improve_epochs} epoch(s).")
            
            if no_improve_epochs >= args.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs.")
                break
                
    # Save metrics to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(results_save_path, index=False)
    print(f"Results saved to {results_save_path}")
    
    # Plot results
    plt.figure(figsize=(12, 10))
    
    # Plot losses
    plt.subplot(2, 1, 1)
    plt.plot(results['epoch'], results['train_loss'], 'b-', marker='o', label='Training Loss')
    plt.plot(results['epoch'], results['val_loss'], 'r-', marker='o', label='Validation Loss')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    
    # Plot metrics
    plt.subplot(2, 1, 2)
    plt.plot(results['epoch'], results['precision'], 'g-', marker='o', label='Precision')
    plt.plot(results['epoch'], results['recall'], 'b-', marker='o', label='Recall')
    plt.plot(results['epoch'], results['f1'], 'r-', marker='o', label='F1 Score')
    plt.axvline(x=best_epoch, color='g', linestyle='--', label=f'Best Model (Epoch {best_epoch})')
    plt.title('Model Metrics')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(fig_save_path)
    print(f"Plot saved to {fig_save_path}")
    
    return {
        'model_path': model_save_path,
        'results_path': results_save_path,
        'fig_path': fig_save_path,
        'best_f1': best_f1,
        'best_epoch': best_epoch,
        'final_metrics': {
            'precision': results['precision'][-1],
            'recall': results['recall'][-1],
            'f1': results['f1'][-1]
        }
    }

def main():
    parser = argparse.ArgumentParser(description='XLM-RoBERTa for Idiom Detection - Training')
    
    # Path arguments
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Directory containing the datasets')
    parser.add_argument('--models_dir', type=str, default='Models', 
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='Submissions', 
                        help='Directory to save outputs')
    
    # Model parameters
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large', 
                        help='Base model name')
    parser.add_argument('--num_labels', type=int, default=3, 
                        help='Number of labels for classification')
    parser.add_argument('--hidden_size', type=int, default=256, 
                        help='Size of the LSTM hidden layer')
    parser.add_argument('--dropout', type=float, default=0.2, 
                        help='Dropout rate')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, 
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                        help='Weight decay for AdamW optimizer')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, 
                        help='Ratio of warmup steps')
    parser.add_argument('--patience', type=int, default=2, 
                        help='Patience for early stopping')
    parser.add_argument('--max_grad_norm', type=float, default=1.0, 
                        help='Maximum gradient norm for clipping')
    parser.add_argument('--cpu', action='store_true', 
                        help='Use CPU even if CUDA is available')
    parser.add_argument('--experiment_name', type=str, default='', 
                        help='Custom name for the experiment')
    
    args = parser.parse_args()
    
    # Train the model
    result = train_model(args)
    
    print("\nTraining complete!")
    print(f"Best F1 Score: {result['best_f1']:.4f} (Epoch {result['best_epoch']})")
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['results_path']}")
    print(f"Plot saved to: {result['fig_path']}")

if __name__ == "__main__":
    main() 