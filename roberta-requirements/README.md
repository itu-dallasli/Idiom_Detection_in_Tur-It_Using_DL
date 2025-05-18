# XLM-RoBERTa for Idiom Detection

This repository contains code for detecting idioms in multilingual text using XLM-RoBERTa with BiLSTM and CRF layers.

## Setup

### Requirements
Install all required dependencies:
```bash
pip install -r requirements.txt
```

### Model Weights
The pre-trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1Ye8jMhutMpNkB-JPU5LCxYAhUkkoAQ0j). Place the downloaded weights in the `Models/saved_models/` directory. # You can download the model which name is xlm_bs16_lr3e-05_ml128_ep15.pt

## How to Run

### Training
To train the model from scratch:
```bash
python train.py --data_dir dataset --models_dir Models --output_dir Submissions --batch_size 16 --learning_rate 3e-5 --max_length 128 --epochs 15
```

Optional arguments:
```
Path arguments:
  --data_dir DATA_DIR     Directory containing the datasets (default: dataset)
  --models_dir MODELS_DIR Directory to save models (default: Models)
  --output_dir OUTPUT_DIR Directory to save outputs (default: Submissions)

Model parameters:
  --model_name MODEL_NAME Base model name (default: xlm-roberta-large)
  --num_labels NUM_LABELS Number of labels for classification (default: 3)
  --hidden_size HIDDEN_SIZE Size of the LSTM hidden layer (default: 256)
  --dropout DROPOUT       Dropout rate (default: 0.2)

Training parameters:
  --batch_size BATCH_SIZE Batch size for training (default: 16)
  --learning_rate LEARNING_RATE Learning rate (default: 3e-5)
  --max_length MAX_LENGTH Maximum sequence length (default: 128)
  --epochs EPOCHS         Number of training epochs (default: 15)
  --seed SEED             Random seed for reproducibility (default: 42)
  --weight_decay WEIGHT_DECAY Weight decay for AdamW optimizer (default: 0.01)
  --warmup_ratio WARMUP_RATIO Ratio of warmup steps (default: 0.1)
  --patience PATIENCE     Patience for early stopping (default: 2)
  --max_grad_norm MAX_GRAD_NORM Maximum gradient norm for clipping (default: 1.0)
  --cpu                   Use CPU even if CUDA is available
  --experiment_name EXPERIMENT_NAME Custom name for the experiment
```

### Inference
To generate predictions on new data:
```bash
python run.py --model_path Models/saved_models/MODEL_NAME.pt --input_file dataset/test_w_o_labels.csv --output_file Submissions/predictions.csv
```

Required arguments:
```
--model_path MODEL_PATH   Path to the trained model weights
--input_file INPUT_FILE   Path to the input test file (CSV format)
--output_file OUTPUT_FILE Path to save predictions (CSV format)
```

Optional arguments:
```
--model_name MODEL_NAME   Base model name to use (default: xlm-roberta-large)
--num_labels NUM_LABELS   Number of labels for classification (default: 3)
--seed SEED               Random seed for reproducibility (default: 42)
--device {cuda,cpu}       Device to run on (default: auto-detect)
```

## Input/Output Format

### Input Format
The input data should be a CSV file with the following columns:
- `id`: Unique identifier for each example
- `sentence`: The text to be analyzed for idioms
- `language`: The language code of the text
- `tokenized_sentence`: (For training data) List of tokenized words
- `indices`: (For training data) List of indices where idioms occur

Example:
```
id,sentence,language,tokenized_sentence,indices
1,"The cat is out of the bag.",en,"['The', 'cat', 'is', 'out', 'of', 'the', 'bag', '.']","[1, 2, 3, 4, 5, 6]"
```

### Output Format
The output file is a CSV with the following columns:
- `id`: The identifier from the input
- `indices`: The predicted indices of idioms, formatted as a string representation of a list
- `language`: The language code from the input

Example:
```
id,indices,language
1,"[1, 2, 3, 4, 5, 6]",en
```

## Assumptions and Limitations

- The model is designed for multilingual idiom detection.
- Input text must be correctly formatted with proper tokenization for accurate predictions.
- The model has been trained on a specific set of languages; performance may vary for unsupported languages.
- The idiom detection is based on BIO tagging (B-IDIOM for beginning of idiom, I-IDIOM for inside of idiom, O for non-idiom).
- The model outputs the word indices that are part of idioms, not the actual idiom phrases.
- A special index of [-1] is used to indicate no idioms were found in the sentence.

## Reproducibility

To ensure reproducible results:
- The code sets random seeds for Python, NumPy, and PyTorch
- CUDA operations are set to be deterministic (may impact performance)
- For exact reproducibility across runs, use the `--seed` parameter with the same value
- When running on the same hardware with the same dependencies and seeds, results should be consistent

## Project Structure

```
.
├── dataset/                     # Dataset directory
│   ├── dataset_es.py            # Data loading and preprocessing utilities
│   ├── train.csv                # Training data
│   ├── eval.csv                 # Evaluation data
│   └── test_w_o_labels.csv      # Test data without labels
│
├── Models/                      # Models directory
│   ├── saved_models/            # Saved model weights
│   ├── results/                 # Training results and metrics
│   └── xlm_roberta_improved.py  # XLM-RoBERTa model with BiLSTM and CRF layers
│
├── Submissions/                 # Output predictions
│
├── main/                        # Main implementation code
│   └── xlm.py                   # Original implementation
│
├── train.py                     # Training script
├── run.py                       # Inference script for generating predictions
├── requirements.txt             # Dependencies
└── README.md                    # This documentation file
```

## Model Architecture

The model architecture consists of:
1. XLM-RoBERTa large as the base transformer model for token embeddings
2. BiLSTM layers for capturing contextual information
3. CRF layer for BIO tagging (ensuring valid label sequences)

This architecture allows for effective detection of idioms in multilingual text by leveraging both the cross-lingual capabilities of XLM-RoBERTa and the sequence modeling strengths of BiLSTM-CRF. 