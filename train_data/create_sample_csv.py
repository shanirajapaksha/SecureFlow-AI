"""
Create sample CSV files from predictions for GitHub upload
This creates smaller sample files instead of the full multi-GB files
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
    print("No trained model found. Running training first...")
    # Import and run training
    import train_random_forest
    train_random_forest.main()
    model_files = list(train_data_dir.glob('random_forest_model_*.pkl'))

if not model_files:
    print("Error: Failed to create model")
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

X_test = np.load(data_dir / 'X_test.npy')
y_test = np.load(data_dir / 'y_test.npy')

print(f"Test data shape: {X_test.shape}")

# Make predictions on test set only (smaller than training set)
print("\nGenerating predictions on test set...")
y_test_pred = model.predict(X_test)
y_test_pred_proba = model.predict_proba(X_test)

# Create sample - first 10000 rows for GitHub
SAMPLE_SIZE = 10000
print(f"\nCreating sample CSV with first {SAMPLE_SIZE} rows...")

test_sample_df = pd.DataFrame({
    'True_Label': y_test[:SAMPLE_SIZE],
    'Predicted_Label': y_test_pred[:SAMPLE_SIZE],
    'Pred_Probability_Class_0': y_test_pred_proba[:SAMPLE_SIZE, 0],
    'Pred_Probability_Class_1': y_test_pred_proba[:SAMPLE_SIZE, 1] if y_test_pred_proba.shape[1] > 1 else np.zeros(SAMPLE_SIZE)
})

# Add features
for i in range(X_test.shape[1]):
    test_sample_df[f'Feature_{i}'] = X_test[:SAMPLE_SIZE, i]

# Save sample
sample_csv_path = train_data_dir / 'test_predictions_sample.csv'
test_sample_df.to_csv(sample_csv_path, index=False)
print(f"Sample CSV saved to: {sample_csv_path}")
print(f"Sample CSV shape: {test_sample_df.shape}")

# Create summary statistics for full dataset
print("\nCreating summary statistics for full test set...")
summary_stats = {
    'Metric': ['Total Samples', 'Correct Predictions', 'Incorrect Predictions', 'Accuracy', 'Class 0 Count', 'Class 1 Count'],
    'Value': [
        len(y_test),
        (y_test_pred == y_test).sum(),
        (y_test_pred != y_test).sum(),
        f"{(y_test_pred == y_test).sum() / len(y_test):.6f}",
        (y_test == 0).sum(),
        (y_test == 1).sum()
    ]
}

summary_df = pd.DataFrame(summary_stats)
summary_csv_path = train_data_dir / 'test_predictions_summary.csv'
summary_df.to_csv(summary_csv_path, index=False)
print(f"\nSummary saved to: {summary_csv_path}")
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

print("\n" + "="*60)
print("CSV files created successfully!")
print(f"Sample file ({SAMPLE_SIZE} rows) suitable for GitHub upload")
print("="*60)
