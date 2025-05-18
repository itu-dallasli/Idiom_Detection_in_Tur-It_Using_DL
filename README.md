# 🎯 Idiom Detection Project

This repository contains a machine learning and deep learning hybrid project focused on idiom detection in multiple languages, specifically Turkish and Italian. The project uses advanced NLP techniques and transformer-based models to identify idioms in text.

## 📁 Project Structure

```
.
├── 📂 Config/                 # Configuration files
├── 📂 Dataset/                # Dataset files
├── 📂 Models/                 # Model implementations and saved models
│   ├── 📂 Transformer/    # Transformer
│   ├── 📂 Latest_BiLSTM-CRF/  # BiLSTM-CRF model implementation
│   ├── 📂 Roberta/       # Roberta model implementation
│   └── 📄 xlm_roberta_improved.py  # Enhanced XLM-RoBERTa model
├── 📂 Submissions/            # Submission files
├── 📂 Running/                # IPYNB files and main running files
│   ├── 📄 xlm_earlystop_main.ipynb  # Main training notebook
│   └── 📄 hyperparameter_search.ipynb  # Hyperparameter optimization
└── 📄 requirements.txt        # Project dependencies
```

## 🚀 Quick Start

### 1. Environment Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Model Weights
The pre-trained model weights can be downloaded from [Google Drive](https://drive.google.com/drive/u/0/folders/1Ye8jMhutMpNkB-JPU5LCxYAhUkkoAQ0j). Place the downloaded weights in the `Models/saved_models/` directory. You can download the model which name is `xlm_bs16_lr3e-05_ml128_ep5.pt`.

## 📊 Data Format

### Input Format 📥
The models expect CSV files with the following columns:
| Column | Description |
|--------|-------------|
| `id` | Unique identifier for each sentence |
| `text`/`sentence` | The input text |
| `language` | Language code |
| `labels` | BIO tags for training data |
| `indices` | List of indices where MWEs are located |
| `tokenized_sentence` | (For training data) List of tokenized words |

### Output Format 📤
The models produce a CSV file with:
| Column | Description |
|--------|-------------|
| `id` | Original sentence ID |
| `indices` | List of word indices that form idioms |
| `language` | Original language code |

## 🤖 Model Architectures

### 1. 🧠 XLM-RoBERTa Model (Enhanced)
- Pre-trained XLM-RoBERTa large model
- BiLSTM layers for capturing long-range context information
- Conditional Random Fields (CRF) for optimal sequence labeling
- Layer normalization and configurable dropout
- Detailed evaluation metrics calculation

#### Training Parameters
```bash
python train.py --data_dir dataset --models_dir Models --output_dir Submissions --batch_size 16 --learning_rate 3e-5 --max_length 128 --epochs 5
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

#### Inference
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

### 2. 🔄 Transformer MLM Model
- Transformer-based architecture with Masked Language Modeling
- Enhanced context understanding through MLM pre-training
- Multi-head attention for MWE detection
- CRF layer for sequence labeling
- Flexible architecture for different language pairs

### 3. 📈 BiLSTM-CRF Model
- BERT/RoBERTa as the base encoder
- Bidirectional LSTM for sequence modeling
- CRF layer for structured prediction
- Focal Loss for handling class imbalance
- Configurable architecture parameters

## 🏃‍♂️ Running the Models

### Training

1. "Running/" folder contains example ipynb files to show how to use the models:  
```bash
jupyter notebook Running/xlm_earlystop_main.ipynb    # For XLM-RoBERTa model
jupyter notebook Running/hyperparameter_search.ipynb  # Plug-in function definition for hyperparameter optimization in the training step.
```

### Transformer MLM Model Usage

The repository provides a unified `run.py` script that can handle both training and prediction:

#### 🎯 Training Mode
```bash
python run.py --mode train \
    --train_path path/to/train.csv \
    --val_path path/to/val.csv \
    --output_dir path/to/output
```

#### 🔮 Prediction Mode
```bash
python run.py --mode predict \
    --test_path path/to/test.csv \
    --output_dir path/to/output \
    --model_path path/to/model.pt
```

#### 🔄 Both Training and Prediction
```bash
python run.py --mode both \
    --train_path path/to/train.csv \
    --val_path path/to/val.csv \
    --test_path path/to/test.csv \
    --output_dir path/to/output
```

### BiLSTM-CRF Model Usage

#### 🎯 Training
```bash
python run_train.py \
    --train_csv path/to/train.csv \
    --output_dir path/to/save/model \
    --model_name bert-base-multilingual-cased \
    --epochs 5 \
    --batch_size 16 \
    --seed 42
```

#### 🔮 Inference
```bash
python run_predict.py \
    --model_dir path/to/saved/model \
    --test_csv path/to/test.csv \
    --output_csv predictions.csv \
    --seed 42
```

## ⚙️ Hyperparameter Optimization

The project includes a comprehensive hyperparameter search framework that optimizes:
- 📊 Learning rate
- 🧠 LSTM hidden size and layers
- 💧 Dropout rates
- 📐 Layer normalization
- 🔒 BERT layer freezing
- ⚖️ Weight decay
- 📈 Learning rate multiplier

## 📊 Evaluation

Consider metrics and structure by the main competition files.
Use the scoring script to evaluate model predictions:
```bash
python scoring.py
```

The model is evaluated using F1-score, calculated separately for:
- 🇹🇷 Turkish language (f1-score-tr)
- 🇮🇹 Italian language (f1-score-it)
- 🌍 Average score across languages (f1-score-avg)

## 🔄 Reproducibility

The code includes:
- 🎲 Fixed random seeds for reproducibility
- 🔍 Deterministic operations where possible
- 📦 Version-controlled dependencies
- 🔄 Consistent data preprocessing
- CUDA operations are set to be deterministic (may impact performance)
- For exact reproducibility across runs, use the `--seed` parameter with the same value
- When running on the same hardware with the same dependencies and seeds, results should be consistent

## 📚 External Resources

The models use the following pre-trained models:
- 🤖 XLM-RoBERTa base/large model
- 🌐 bert-base-multilingual-cased
- 🔤 xlm-roberta-base

These are automatically downloaded when first used.

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b YZV405_2425_150220331_150210310`)
3. Commit your changes (`git commit -m 'Add some YZV405_2425_150220331_150210310'`)
4. Push to the branch (`git push origin YZV405_2425_150220331_150210310`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- 🤗 Hugging Face Transformers library
