#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Configuration settings for the XLM-RoBERTa Idiom Detection project.
These are the default settings that can be overridden via command line arguments.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, "dataset")
MODELS_DIR = os.path.join(BASE_DIR, "Models")
OUTPUT_DIR = os.path.join(BASE_DIR, "Submissions")

# Model settings
MODEL_NAME = "xlm-roberta-large"
NUM_LABELS = 3
HIDDEN_SIZE = 256
LSTM_LAYERS = 2
DROPOUT = 0.2
BIDIRECTIONAL = True

# Training settings
BATCH_SIZE = 16
LEARNING_RATE = 3e-5
MAX_LENGTH = 128
EPOCHS = 3
WARMUP_RATIO = 0.1
PATIENCE = 2
WEIGHT_DECAY = 0.01
MAX_GRAD_NORM = 1.0
SEED = 42

# Data paths
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
EVAL_PATH = os.path.join(DATA_DIR, "eval.csv")
TEST_PATH = os.path.join(DATA_DIR, "test_w_o_labels.csv")

# Label mapping
LABELS = {
    0: "O",
    1: "B-IDIOM",
    2: "I-IDIOM"
}

# Model paths
MODEL_SAVE_DIR = os.path.join(MODELS_DIR, "saved_models")
RESULTS_SAVE_DIR = os.path.join(MODELS_DIR, "results")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_SAVE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True) 