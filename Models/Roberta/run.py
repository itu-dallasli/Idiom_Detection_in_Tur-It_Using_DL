#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for generating predictions from a trained XLM-RoBERTa model.
"""

import os
import sys
import torch
import pandas as pd
import argparse
from tqdm import tqdm
from transformers import XLMRobertaTokenizer
import random
import numpy as np

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
    from Models.xlm_roberta_improved import XLMRobertaForIdiomDetection, predict_idioms
except ImportError:
    # Add the project root to Python path for imports if run from different directory
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from Models.xlm_roberta_improved import XLMRobertaForIdiomDetection, predict_idioms

def generate_predictions(model, tokenizer, device, input_file, output_file):
    """
    Generate predictions using a trained model.
    
    Args:
        model: The trained model
        tokenizer: Tokenizer for the model
        device: Device to run inference on (cuda/cpu)
        input_file: Path to the input CSV file
        output_file: Path to save the predictions CSV
    """
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
    parser = argparse.ArgumentParser(description='XLM-RoBERTa Idiom Detection Inference')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the trained model weights')
    parser.add_argument('--input_file', type=str, required=True, 
                        help='Path to the input test file (CSV format)')
    parser.add_argument('--output_file', type=str, required=True, 
                        help='Path to save predictions (CSV format)')
    parser.add_argument('--model_name', type=str, default='xlm-roberta-large',
                        help='Base model name to use')
    parser.add_argument('--num_labels', type=int, default=3,
                        help='Number of labels for classification')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default=None,
                        help='Device to run on (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize tokenizer and model
    tokenizer = XLMRobertaTokenizer.from_pretrained(args.model_name)
    model = XLMRobertaForIdiomDetection(model_name=args.model_name, num_labels=args.num_labels)
    
    # Load model weights
    print(f"Loading model from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    
    # Generate predictions
    generate_predictions(model, tokenizer, device, args.input_file, args.output_file)
    print("Inference complete!")

if __name__ == "__main__":
    main() 