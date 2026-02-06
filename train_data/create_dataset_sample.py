"""
Create a sample test dataset CSV for GitHub upload (without model predictions)
This allows uploading data structure without waiting for full model training
"""

import numpy as np
import pandas as pd
from pathlib import Path

# Load the preprocessed test data
data_dir = Path(__file__).parent.parent / 'data' / 'processed'
print("Loading preprocessed test data...")

try:
    X_test = np.load(data_dir / 'X_test.npy')
    y_test = np.load(data_dir / 'y_test.npy')
    
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    # Create sample - first 10000 rows
    SAMPLE_SIZE = 10000
    print(f"\nCreating sample CSV with first {SAMPLE_SIZE} rows...")
    
    # Create DataFrame with features and labels
    test_sample_df = pd.DataFrame()
    test_sample_df['Label'] = y_test[:SAMPLE_SIZE]
    
    # Add all features
    for i in range(X_test.shape[1]):
        test_sample_df[f'Feature_{i}'] = X_test[:SAMPLE_SIZE, i]
    
    # Save sample
    sample_csv_path = Path(__file__).parent / 'test_data_sample.csv'
    test_sample_df.to_csv(sample_csv_path, index=False)
    print(f"\nSample CSV saved to: {sample_csv_path}")
    print(f"Sample CSV shape: {test_sample_df.shape}")
    print(f"File size: {sample_csv_path.stat().st_size / (1024*1024):.2f} MB")
    
    # Create summary statistics
    print("\nCreating dataset summary...")
    summary_stats = {
        'Metric': [
            'Total Test Samples',
            'Number of Features', 
            'Class 0 (Benign) Count',
            'Class 1 (Attack) Count',
            'Class Balance'
        ],
        'Value': [
            len(y_test),
            X_test.shape[1],
            (y_test == 0).sum(),
            (y_test == 1).sum(),
            f"{(y_test == 0).sum() / len(y_test) * 100:.2f}% / {(y_test == 1).sum() / len(y_test) * 100:.2f}%"
        ]
    }
    
    summary_df = pd.DataFrame(summary_stats)
    summary_csv_path = Path(__file__).parent / 'dataset_summary.csv'
    summary_df.to_csv(summary_csv_path, index=False)
    print(f"Summary saved to: {summary_csv_path}")
    print("\nDataset Summary:")
    print(summary_df.to_string(index=False))
    
    # Create feature statistics
    print("\nCreating feature statistics...")
    feature_stats = {
        'Feature': [f'Feature_{i}' for i in range(min(10, X_test.shape[1]))],
        'Mean': [X_test[:, i].mean() for i in range(min(10, X_test.shape[1]))],
        'Std': [X_test[:, i].std() for i in range(min(10, X_test.shape[1]))],
        'Min': [X_test[:, i].min() for i in range(min(10, X_test.shape[1]))],
        'Max': [X_test[:, i].max() for i in range(min(10, X_test.shape[1]))]
    }
    
    feature_stats_df = pd.DataFrame(feature_stats)
    feature_stats_csv_path = Path(__file__).parent / 'feature_statistics.csv'
    feature_stats_df.to_csv(feature_stats_csv_path, index=False)
    print(f"Feature statistics saved to: {feature_stats_csv_path}")
    
    print("\n" + "="*60)
    print("CSV files created successfully!")
    print("These files are suitable for GitHub upload")
    print("="*60)
    
except FileNotFoundError as e:
    print(f"Error: Could not find preprocessed data files")
    print(f"Please ensure X_test.npy and y_test.npy exist in {data_dir}")
    print(f"Error details: {e}")
