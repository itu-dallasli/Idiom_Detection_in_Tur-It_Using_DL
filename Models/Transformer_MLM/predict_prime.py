import torch
import pandas as pd
from transformers import AutoTokenizer
import argparse
import os
import json
from model_prime import MWEMaskedAttentionModel, predict_mwe

def parse_args():
    parser = argparse.ArgumentParser(description='Make predictions using trained MWE detection model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to input data CSV')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model weights')
    parser.add_argument('--output_path', type=str, required=True, help='Path to save predictions')
    parser.add_argument('--config_path', type=str, help='Path to model configuration JSON')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for inference')
    parser.add_argument('--max_length', type=int, default=128, help='Maximum sequence length')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model configuration
    if args.config_path:
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {
            'model_name': 'xlm-roberta-base',
            'hidden_size': 768,
            'num_attention_heads': 8,
            'attention_dropout': 0.1,
            'hidden_dropout': 0.3
        }
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config['model_name'])
    
    # Initialize model
    print("Loading model...")
    model = MWEMaskedAttentionModel(
        model_name=config['model_name'],
        num_labels=3,  # O, B-IDIOM, I-IDIOM
        hidden_size=config['hidden_size'],
        num_attention_heads=config['num_attention_heads'],
        attention_dropout=config['attention_dropout'],
        hidden_dropout=config['hidden_dropout']
    )
    
    # Load trained weights
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_df = pd.read_csv(args.input_path)
    
    # Prepare results
    results = []
    
    print("\nMaking predictions...")
    total_sentences = len(eval_df)
    total_mwes = 0
    total_no_mwes = 0
    
    for idx, row in eval_df.iterrows():
        sentence = row["sentence"]
        language = row["language"]
        
        # Get predictions
        words, labels = predict_mwe(model, tokenizer, sentence, device, args.max_length)
        
        # Debug print for first few sentences
        if idx < 5:
            print(f"\nSentence {idx}: {sentence}")
            print(f"Language: {language}")
            print("Words and their labels:")
            for w, l in zip(words, labels):
                print(f"{w}: {l}")
        
        # Find MWE indices
        mwe_indices = []
        current_mwe = []
        word_idx = 0
        
        for word, label in zip(words, labels):
            if label == 1:  # B-IDIOM
                if current_mwe:  # Save previous MWE if exists
                    mwe_indices.extend(current_mwe)
                current_mwe = [word_idx]
            elif label == 2:  # I-IDIOM
                current_mwe.append(word_idx)
            else:  # O
                if current_mwe:  # Save previous MWE if exists
                    mwe_indices.extend(current_mwe)
                current_mwe = []
            word_idx += 1
        
        # Don't forget the last MWE if exists
        if current_mwe:
            mwe_indices.extend(current_mwe)
        
        # If no MWEs found, use [-1] as per scoring.py requirements
        if not mwe_indices:
            mwe_indices = [-1]
            total_no_mwes += 1
        else:
            total_mwes += 1
        
        # Add to results
        results.append({
            'id': row['id'],
            'indices': str(mwe_indices),  # Convert list to string representation
            'language': language
        })
    
    # Create prediction DataFrame
    prediction_df = pd.DataFrame(results)
    
    # Print statistics
    print("\nPrediction Statistics:")
    print(f"Total sentences: {total_sentences}")
    print(f"Sentences with MWEs: {total_mwes}")
    print(f"Sentences without MWEs: {total_no_mwes}")
    print(f"Percentage of sentences with MWEs: {(total_mwes/total_sentences)*100:.2f}%")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Save predictions
    print(f"\nSaving predictions to {args.output_path}...")
    prediction_df.to_csv(args.output_path, index=False)
    print("Predictions saved successfully!")

if __name__ == "__main__":
    main() 