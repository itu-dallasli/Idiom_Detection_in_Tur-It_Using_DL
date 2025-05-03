import json
import os
import numpy as np
import pandas as pd
import ast  # For safely evaluating the string representation of lists

# Define input and output directories
reference_dir = os.path.join('/app/input', 'ref')
prediction_dir = os.path.join('/app/input', 'res')
score_dir = '/app/output'

print("Reading prediction.csv")

prediction_file = os.path.join(prediction_dir, 'prediction.csv')
truth_file = os.path.join(reference_dir, 'ground_truth.csv')

# Load CSV files
prediction_df = pd.read_csv(prediction_file)
truth_df = pd.read_csv(truth_file)

print("prediction.csv file is successfully read")

# Ensure 'indices' and 'language' columns exist
required_columns = ['id', 'indices', 'language']
for col in required_columns:
    if col not in prediction_df.columns or col not in truth_df.columns:
        raise ValueError(f"prediction.csv file must have a '{col}' column")

# Length check: Ensure the dataframes have the same number of rows
if len(prediction_df) != len(truth_df):
    raise ValueError(f"prediction.csv and ground truth files have different number of rows. "
                     f"Predictions: {len(prediction_df)}, Ground Truth: {len(truth_df)}")

# Check for missing or empty values in the 'indices' column
if prediction_df['indices'].isnull().any() or truth_df['indices'].isnull().any():
    raise ValueError("There are missing values in the 'indices' column.")

# Check for missing or empty language entries
if prediction_df['language'].isnull().any() or truth_df['language'].isnull().any():
    raise ValueError("There are missing language entries in the 'language' column.")

# Check for invalid or empty indices (should not be empty strings)
if (prediction_df['indices'] == '').any() or (truth_df['indices'] == '').any():
    raise ValueError("Some rows have empty 'indices' values.")


# Convert 'indices' column from string representation of lists to actual lists
def parse_indices(indices):
    try:
        return ast.literal_eval(indices)  # Safely evaluate the string to a list
    except (ValueError, SyntaxError):
        raise ValueError(f"Invalid indices format: {indices}")


prediction_df['indices'] = prediction_df['indices'].apply(parse_indices)
truth_df['indices'] = truth_df['indices'].apply(parse_indices)

# Separate data by language (e.g., Turkish 'tr' and other language)
prediction_tr = prediction_df[prediction_df["language"] == "tr"]
truth_tr = truth_df[truth_df["language"] == "tr"]

prediction_it = prediction_df[prediction_df["language"] == "it"]
truth_it = truth_df[truth_df["language"] == "it"]


# Function to calculate F1 score
def calculate_f1_score(prediction_df, truth_df):
    f1_scores = []

    # Compute F1 scores for each row
    for pred, gold in zip(prediction_df["indices"], truth_df["indices"]):
        # Special case for ground truth [-1]
        if gold == [-1]:
            # Prediction must also be exactly [-1]
            if pred == [-1]:
                f1_scores.append(1.0)
            else:
                f1_scores.append(0.0)
        else:
            # Convert indices into sets for comparison
            pred_set = set(pred) if isinstance(pred, list) else set()
            gold_set = set(gold) if isinstance(gold, list) else set()

            # Calculate precision, recall, and F1 score
            intersection = len(pred_set & gold_set)
            precision = intersection / len(pred_set) if len(pred_set) > 0 else 0
            recall = intersection / len(gold_set) if len(gold_set) > 0 else 0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            f1_scores.append(f1)

    # Compute mean F1 score
    return np.mean(f1_scores)


# Calculate F1 scores separately for Turkish and other languages
print("Calculating F1 scores...")

print("Calculating F1 score for Turkish")
f1_tr = calculate_f1_score(prediction_tr, truth_tr)

print("Calculating F1 score for Italian")
f1_it = calculate_f1_score(prediction_it, truth_it)

print("F1 scores are calculated successfully")
f1_avg = (f1_tr + f1_it) / 2

print("Scores:")
scores = {
    "f1-score-tr": f1_tr,
    "f1-score-it": f1_it,
    "f1-score-avg": f1_avg
}

print(scores)

# Save scores to JSON file
with open(os.path.join(score_dir, 'scores.json'), 'w') as score_file:
    json.dump(scores, score_file)
