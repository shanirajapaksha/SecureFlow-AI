"""
Script to export trained model predictions and results to CSV files
"""

import numpy as np
import pandas as pd
import pickle
import sys
from pathlib import Path

# Get the latest model file
train_data_dir = Path(__file__).parent
model_files = list(train_data_dir.glob('random_forest_model_*.pkl'))

if not model_files:
    print("Error: No trained model found in train_data directory")
    sys.exit(1)

# Use the most recently modified model
model_path = max(model_files, key=lambda p: p.stat().st_mtime)
print(f"Loading model from: {model_path}")

# Load the trained model
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Load the preprocessed data
data_dir = Path(__file__).parent.parent / 'data' / 'processed'
print("Loading preprocessed data...")

X_train = np.load(data_dir / 'X_train.npy')
y_train = np.load(data_dir / 'y_train.npy')
X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"Training data shape: {X_train.shape}")
print(f"Test data shape: {X_test.shape}")

# Make predictions
print("\nGenerating predictions on training set...")
y_train_pred = model.predict(X_train)
y_train_pred_proba = model.predict_proba(X_train)

print("Generating predictions on test set...")
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

# Create DataFrames for training set predictions
print("\nCreating training predictions CSV...")
train_df = pd.DataFrame({
    'True_Label': y_train,
    'Predicted_Label': y_train_pred,
    'Pred_Probability_Class_0': y_train_pred_proba[:, 0],
    'Pred_Probability_Class_1': y_train_pred_proba[:, 1] if y_train_pred_proba.shape[1] > 1 else np.zeros(len(y_train))
})

# Add features to training dataframe
for i in range(X_train.shape[1]):
    train_df[f'Feature_{i}'] = X_train[:, i]

train_csv_path = train_data_dir / 'training_predictions.csv'
train_df.to_csv(train_csv_path, index=False)
print(f"Training predictions saved to: {train_csv_path}")
print(f"Training CSV shape: {train_df.shape}")

# Create DataFrame for test set predictions
print("\nCreating test predictions CSV...")
test_df = pd.DataFrame({
    'True_Label': y_test,
    'Predicted_Label': y_test_pred,
    'Pred_Probability_Class_0': y_test_pred_proba[:, 0],
    'Pred_Probability_Class_1': y_test_pred_proba[:, 1] if y_test_pred_proba.shape[1] > 1 else np.zeros(len(y_test))
})

# Add features to test dataframe
for i in range(X_test.shape[1]):
    test_df[f'Feature_{i}'] = X_test[:, i]

test_csv_path = train_data_dir / 'test_predictions.csv'
test_df.to_csv(test_csv_path, index=False)
print(f"Test predictions saved to: {test_csv_path}")
print(f"Test CSV shape: {test_df.shape}")

# Create a summary statistics file
print("\nCreating summary statistics...")
summary_stats = {
    'Dataset': ['Training', 'Test'],
    'Total_Samples': [len(y_train), len(y_test)],
    'Correct_Predictions': [
        (y_train_pred == y_train).sum(),
        (y_test_pred == y_test).sum()
    ],
    'Incorrect_Predictions': [
        (y_train_pred != y_train).sum(),
        (y_test_pred != y_test).sum()
    ],
    'Accuracy': [
        (y_train_pred == y_train).sum() / len(y_train),
        (y_test_pred == y_test).sum() / len(y_test)
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_csv_path = train_data_dir / 'prediction_summary.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"Summary statistics saved to: {summary_csv_path}")
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("All CSV files saved successfully to train_data folder!")
print("="*60)
