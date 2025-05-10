import torch
from transformers import AutoTokenizer
from dataset.dataset import get_dataloaders
from model_from_scratch import MWEMaskedAttentionModel, train_model
import pandas as pd
import numpy as np

def analyze_predictions(model, val_loader, tokenizer, device):
    model.eval()
    all_predictions = []
    all_labels = []
    all_sentences = []
    
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            predictions = outputs['predictions']
            
            # Store predictions and labels
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(batch['labels'].cpu().numpy())
            
            # Get original sentences
            input_ids = batch['input_ids'].cpu().numpy()
            for ids in input_ids:
                tokens = tokenizer.convert_ids_to_tokens(ids)
                sentence = tokenizer.convert_tokens_to_string(tokens)
                all_sentences.append(sentence)
    
    # Calculate statistics
    total_sentences = len(all_sentences)
    total_mwes_predicted = 0
    total_mwes_actual = 0
    
    print("\nDetailed Validation Analysis:")
    print(f"Total sentences: {total_sentences}")
    
    # Analyze first 5 sentences in detail
    print("\nSample Predictions (first 5 sentences):")
    for i in range(min(5, total_sentences)):
        print(f"\nSentence {i+1}:")
        print(f"Text: {all_sentences[i]}")
        
        # Get predictions and labels for this sentence
        preds = all_predictions[i]
        labels = all_labels[i]
        
        # Count MWEs
        pred_mwes = sum(1 for p in preds if p in [1, 2])  # B-IDIOM or I-IDIOM
        actual_mwes = sum(1 for l in labels if l in [1, 2])
        
        total_mwes_predicted += pred_mwes
        total_mwes_actual += actual_mwes
        
        print(f"Predicted MWEs: {pred_mwes}")
        print(f"Actual MWEs: {actual_mwes}")
        
        # Show token-level predictions
        tokens = tokenizer.tokenize(all_sentences[i])
        print("\nToken-level analysis:")
        for token, pred, label in zip(tokens, preds, labels):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                pred_tag = 'B-IDIOM' if pred == 1 else 'I-IDIOM' if pred == 2 else 'O'
                true_tag = 'B-IDIOM' if label == 1 else 'I-IDIOM' if label == 2 else 'O'
                print(f"{token:<20} Pred: {pred_tag:<10} True: {true_tag}")
    
    # Print overall statistics
    print("\nOverall Statistics:")
    print(f"Average predicted MWEs per sentence: {total_mwes_predicted/total_sentences:.2f}")
    print(f"Average actual MWEs per sentence: {total_mwes_actual/total_sentences:.2f}")
    print(f"Percentage of sentences with predicted MWEs: {(sum(1 for p in all_predictions if any(x in [1,2] for x in p))/total_sentences)*100:.2f}%")
    print(f"Percentage of sentences with actual MWEs: {(sum(1 for l in all_labels if any(x in [1,2] for x in l))/total_sentences)*100:.2f}%")

def main():
    # Data parameters
    train_path = 'dataset/train.csv'
    val_path = 'dataset/eval.csv'
    batch_size = 16
    max_length = 128
    
    # Training parameters
    epochs = 10
    learning_rate = 1e-5  # Reduced learning rate
    weight_decay = 0.01
    lr_multiplier = 5  # Reduced multiplier
    patience = 3
    
    # Model parameters
    model_name = 'xlm-roberta-base'
    hidden_size = 768
    num_attention_heads = 8
    attention_dropout = 0.2  # Increased dropout
    hidden_dropout = 0.4  # Increased dropout
    freeze_bert_layers = 6  # Freeze first 6 layers
    mlm_probability = 0.15  # MLM masking probability
    
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Get dataloaders
    print("Loading data...")
    train_loader, val_loader, _ = get_dataloaders(
        train_path=train_path,
        val_path=val_path,
        batch_size=batch_size,
        max_length=max_length
    )
    
    # Print dataset statistics
    train_df = pd.read_csv(train_path)
    val_df = pd.read_csv(val_path)
    
    print("\nDataset Statistics:")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    
    # Count MWEs in training set
    train_mwes = sum(1 for idx in train_df['indices'] if eval(idx) != [-1])
    val_mwes = sum(1 for idx in val_df['indices'] if eval(idx) != [-1])
    
    print(f"Training set MWEs: {train_mwes}")
    print(f"Validation set MWEs: {val_mwes}")
    print(f"Training set MWE percentage: {(train_mwes/len(train_df))*100:.2f}%")
    print(f"Validation set MWE percentage: {(val_mwes/len(val_df))*100:.2f}%")
    
    # Initialize model
    print("\nInitializing model...")
    model = MWEMaskedAttentionModel(
        model_name=model_name,
        num_labels=3,  # O, B-IDIOM, I-IDIOM
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout,
        freeze_bert_layers=freeze_bert_layers,
        mlm_probability=mlm_probability
    )
    
    # Train model
    print("Starting training...")
    model = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        tokenizer=tokenizer,
        epochs=epochs,
        lr=learning_rate,
        weight_decay=weight_decay,
        lr_multiplier=lr_multiplier,
        patience=patience
    )
    
    # Analyze validation predictions
    print("\nAnalyzing validation predictions...")
    analyze_predictions(model, val_loader, tokenizer, model.device)
    
    print("\nTraining completed!")

if __name__ == "__main__":
    main() 