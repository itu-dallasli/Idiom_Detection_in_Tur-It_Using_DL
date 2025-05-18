# Idiom Detection Project

This repository contains a machine learning and deep learning hybrid project focused on idiom detection in multiple languages, specifically Turkish and Italian. The project uses advanced NLP techniques and transformer-based models to identify idioms in text.

## Project Structure

```
.
├── Config/                 # Configuration files
├── Dataset/                # Dataset files
├── Models/                 # Model implementations and saved models
├── Submissions/            # Submission files
├── Running/                # IPYNB files and main running files
└── requirements.txt        # Project dependencies
```

## Features

- Multi-language support (Turkish and Italian)
- Multiple model implementations:
  - XLM-RoBERTa with BiLSTM and CRF 
  - BERT-based model architecture

- LSTM enhancement for better sequence understanding
- Hyperparameter optimization framework
- Support for both training and inference
- Different model structure approaches (BiLSTM, Transformer, Bert-based etc.)

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training

1. Create an "ipynb" file with respect to a model that you wish to use. Use training, evaluation, prediction functions and run them consequently.
2. "Running/" folder contains example ipynb files to show how to use the models:  
```bash
jupyter notebook xlm_earlystop_main.ipynb    # For XLM-RoBERTa model
```

### Evaluation
Consider metrics and structure by the main competition files.
Use the scoring script to evaluate model predictions:
```bash
python scoring.py
```

## Model Architectures

### XLM-RoBERTa Model
The project includes an XLM-RoBERTa-based model with the following features:
- Pre-trained XLM-RoBERTa large model
- BiLSTM layers for capturing long-range context information
- Conditional Random Fields (CRF) for optimal sequence labeling
- Layer normalization and configurable dropout
- Detailed evaluation metrics calculation

### BERT-based Model
The project also includes an enhanced BERT-based model with the following features:
- Multi-lingual BERT base model
- LSTM layers for improved sequence understanding
- Layer normalization options
- Configurable dropout rates
- Flexible BERT layer freezing

## Hyperparameter Optimization

The project includes a comprehensive hyperparameter search framework that optimizes:
- Learning rate
- LSTM hidden size and layers
- Dropout rates
- Layer normalization
- BERT layer freezing
- Weight decay
- Learning rate multiplier

## Evaluation Metrics

The model is evaluated using F1-score, calculated separately for:
- Turkish language (f1-score-tr)
- Italian language (f1-score-it)
- Average score across languages (f1-score-avg)

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b YZV405_2425_150220331_150210310`)
3. Commit your changes (`git commit -m 'Add some YZV405_2425_150220331_150210310'`)
4. Push to the branch (`git push origin YZV405_2425_150220331_150210310`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Hugging Face Transformers library
- PyTorch team
- Contributors and maintainers 
