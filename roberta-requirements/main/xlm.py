import os
import sys
import torch
import pandas as pd
from transformers import XLMRobertaTokenizer
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from transformers import get_scheduler
import argparse
import random
from pathlib import Path

# Note: Install pytorch-crf if not already installed
# pip install pytorch-crf

# Add the project root to Python path for imports
PROJECT_DIR = "/content/drive/MyDrive/Colab Notebooks/nlp_f1_cleaned"
sys.path.append(PROJECT_DIR)

from dataset.dataset_es import get_dataloaders
from Models.xlm_roberta_improved import XLMRobertaForIdiomDetection, predict_idioms, evaluate

# Define paths
DATA_DIR = os.path.join(PROJECT_DIR, "dataset")
MODELS_DIR = os.path.join(PROJECT_DIR, "Models")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "Submissions")

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "Models/saved_models"), exist_ok=True)
os.makedirs(os.path.join(PROJECT_DIR, "Models/results"), exist_ok=True)

# Ensure reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def run_organized_experiment(data_dir, models_dir, output_dir, batch_size=16, lr=1e-5, max_length=128, epochs=3, seed=42):
    # Set seed for reproducibility
    set_seed(seed)
    
    experiment_name = f"xlm_bs{batch_size}_lr{lr}_ml{max_length}_ep{epochs}"
    model_save_path = os.path.join(models_dir, "saved_models", f"{experiment_name}.pt")
    results_save_path = os.path.join(models_dir, "results", f"{experiment_name}_metrics.csv")
    fig_save_path = os.path.join(models_dir, "results", f"{experiment_name}_plot.png")

    # Initialize results
    results = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'precision': [],
        'recall': [],
        'f1': []
    }

    # Initialize tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-large')

    print(f"\nRunning experiment: {experiment_name}")
    print("Loading data...")

    # Get data loaders from dataset.py
    train_loader, val_loader, _ = get_dataloaders(
        train_path=os.path.join(data_dir, 'train.csv'),
        val_path=os.path.join(data_dir, 'eval.csv'),
        batch_size=batch_size,
        max_length=max_length,
        tokenizer=tokenizer
    )

    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XLMRobertaForIdiomDetection()
    model.to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    total_steps = len(train_loader) * epochs
    scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    best_f1 = 0.0
    best_epoch = 0

    # Added these line to control earlystopping
    patience = 2
    no_improve_epochs = 0

    # Training loop
    for epoch in range(epochs):
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
        print(f"Epoch {epoch+1}/{epochs}:")
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

            if no_improve_epochs >= patience:
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
    plt.show()

    # Load best model for predictions
    best_model = XLMRobertaForIdiomDetection(model_name="xlm-roberta-large", num_labels=3)
    best_model.load_state_dict(torch.load(model_save_path))
    best_model.to(device)

    # Make predictions
    prediction_file = os.path.join(models_dir, "results", f"{experiment_name}_predictions.csv")
    generate_predictions(best_model, tokenizer, device, os.path.join(data_dir, 'test_w_o_labels.csv'), prediction_file)

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

def generate_predictions(model, tokenizer, device, input_file, output_file):
    print(f"Generating predictions from {input_file}...")
    model.eval()

    # Read test data
    test_df = pd.read_csv(input_file)
    ids = test_df['id'].tolist()
    languages = test_df['language'].tolist()
    sentences = test_df['sentence'].tolist()

    # Generate predictions
    results = []
    for idx, sentence, lang in tqdm(zip(ids, sentences, languages), total=len(ids)):
        _, idiom_indices = predict_idioms(model, tokenizer, sentence, device)
        # If no idiom, output [-1] as in training
        if not idiom_indices:
            idiom_indices = [-1]
        results.append({'id': idx, 'indices': str(idiom_indices), 'language': lang})

    # Save predictions
    out_df = pd.DataFrame(results)
    out_df.to_csv(output_file, index=False)
    print(f'Predictions saved to {output_file}')

    return out_df

def main():
    parser = argparse.ArgumentParser(description='XLM-RoBERTa for Idiom Detection')
    
    # Path arguments
    parser.add_argument('--data_dir', type=str, default='dataset', 
                        help='Directory containing the datasets')
    parser.add_argument('--models_dir', type=str, default='Models', 
                        help='Directory to save models')
    parser.add_argument('--output_dir', type=str, default='Submissions', 
                        help='Directory to save outputs')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=16, 
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=3e-5, 
                        help='Learning rate')
    parser.add_argument('--max_length', type=int, default=128, 
                        help='Maximum sequence length')
    parser.add_argument('--epochs', type=int, default=15, 
                        help='Number of training epochs')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Create directories if they don't exist
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "saved_models"), exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "results"), exist_ok=True)
    
    result = run_organized_experiment(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        lr=args.learning_rate,
        max_length=args.max_length,
        epochs=args.epochs,
        seed=args.seed
    )

    print("\nExperiment complete!")
    print(f"Best F1 Score: {result['best_f1']:.4f} (Epoch {result['best_epoch']})")
    print(f"Model saved to: {result['model_path']}")
    print(f"Metrics saved to: {result['results_path']}")
    print(f"Plot saved to: {result['fig_path']}")

if __name__ == "__main__":
    main()



