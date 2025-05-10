import torch
import pandas as pd
from transformers import AutoTokenizer
from model_from_scratch import MWEAttentionModel, predict_mwe

def main():
    # Model parameters (should match training parameters)
    model_name = 'xlm-roberta-base'
    hidden_size = 768
    num_attention_heads = 8
    attention_dropout = 0.1
    hidden_dropout = 0.3
    
    # Initialize device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize tokenizer
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize model
    print("Loading model...")
    model = MWEAttentionModel(
        model_name=model_name,
        num_labels=3,  # O, B-IDIOM, I-IDIOM
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        attention_dropout=attention_dropout,
        hidden_dropout=hidden_dropout
    )
    
    # Load trained weights
    model.load_state_dict(torch.load("best_mwe_model.pt"))
    model = model.to(device)
    model.eval()
    
    # Load evaluation data
    print("Loading evaluation data...")
    eval_df = pd.read_csv("dataset/eval.csv")
    
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
        words, labels = predict_mwe(model, tokenizer, sentence, device)
        
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
    
    # Save predictions
    print("\nSaving predictions to prediction.csv...")
    prediction_df.to_csv("prediction.csv", index=False)
    print("Predictions saved successfully!")

if __name__ == "__main__":
    main() 