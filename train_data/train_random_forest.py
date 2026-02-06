"""
Random Forest Training Script for Network Intrusion Detection System
This script trains a Random Forest model on preprocessed network traffic data.
It handles batch processing if the dataset is too large to fit in memory at once.
"""

import numpy as np
import pickle
import os
import sys
from pathlib import Path
from datetime import datetime
import json

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score
    )
except ImportError as e:
    print(f"Error importing scikit-learn: {e}")
    print("Please ensure scikit-learn is installed.")
    sys.exit(1)


class RandomForestTrainer:
    def __init__(self, data_dir, output_dir, batch_size=10000, random_state=42):
        """
        Initialize the trainer with paths and hyperparameters.
        
        Args:
            data_dir: Directory containing preprocessed data
            output_dir: Directory to save results
            batch_size: Number of samples to process at once for memory efficiency
            random_state: Random state for reproducibility
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.batch_size = batch_size
        self.random_state = random_state
        self.model = None
        self.results = {}
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_data(self):
        """Load the preprocessed numpy arrays."""
        print("Loading preprocessed data...")
        try:
            self.X_train = np.load(self.data_dir / 'X_train.npy')
            self.y_train = np.load(self.data_dir / 'y_train.npy')
            self.X_test = np.load(self.data_dir / 'X_test.npy')
            self.y_test = np.load(self.data_dir / 'y_test.npy')
            
            print(f"Training data shape: {self.X_train.shape}")
            print(f"Training labels shape: {self.y_train.shape}")
            print(f"Test data shape: {self.X_test.shape}")
            print(f"Test labels shape: {self.y_test.shape}")
            
            # Check class distribution
            unique_train, counts_train = np.unique(self.y_train, return_counts=True)
            print("\nTraining set class distribution:")
            for label, count in zip(unique_train, counts_train):
                print(f"  Class {label}: {count} samples ({count/len(self.y_train)*100:.2f}%)")
                
            return True
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            return False
    
    def train_batch(self, n_estimators=100, max_depth=20, min_samples_split=5):
        """
        Train Random Forest model. If data is too large, uses warm_start for batch training.
        
        Args:
            n_estimators: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples required to split a node
        """
        print("\nTraining Random Forest model...")
        print(f"Hyperparameters: n_estimators={n_estimators}, max_depth={max_depth}, min_samples_split={min_samples_split}")
        
        try:
            # Check if we can fit everything in memory
            total_size = self.X_train.nbytes / (1024**3)  # Convert to GB
            print(f"\nTraining data size: {total_size:.2f} GB")
            
            if total_size < 2:  # If less than 2GB, train all at once
                print("Training on full dataset...")
                self.model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    min_samples_split=min_samples_split,
                    n_jobs=-1,  # Use all available cores
                    random_state=self.random_state,
                    verbose=1
                )
                self.model.fit(self.X_train, self.y_train)
                print("Training completed successfully!")
            else:
                print(f"Dataset size exceeds 2GB, using batch training...")
                self._train_batch_processing(n_estimators, max_depth, min_samples_split)
                
        except MemoryError:
            print("Memory error encountered, switching to batch processing...")
            self._train_batch_processing(n_estimators, max_depth, min_samples_split)
    
    def _train_batch_processing(self, n_estimators, max_depth, min_samples_split):
        """
        Train using batch processing with warm_start.
        """
        print(f"Using batch processing with batch size: {self.batch_size}")
        
        # Initialize model with warm_start
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            n_jobs=-1,
            random_state=self.random_state,
            warm_start=True,
            verbose=1
        )
        
        # Process data in batches
        n_samples = len(self.X_train)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        print(f"Processing {n_batches} batches of {self.batch_size} samples...")
        
        for batch_idx in range(0, n_samples, self.batch_size):
            batch_end = min(batch_idx + self.batch_size, n_samples)
            batch_num = (batch_idx // self.batch_size) + 1
            
            X_batch = self.X_train[batch_idx:batch_end]
            y_batch = self.y_train[batch_idx:batch_end]
            
            print(f"\nBatch {batch_num}/{n_batches}: Training on samples {batch_idx}-{batch_end}...")
            
            if batch_idx == 0:
                # First batch: fit the model
                self.model.fit(X_batch, y_batch)
            else:
                # Subsequent batches: use warm_start
                # Increase n_estimators for warm_start
                self.model.n_estimators += n_estimators // n_batches
                self.model.fit(X_batch, y_batch)
        
        print("\nBatch training completed!")
    
    def evaluate(self):
        """Evaluate the model on test set."""
        print("\nEvaluating model on test set...")
        
        # Make predictions
        y_pred = self.model.predict(self.X_test)
        y_pred_proba = self.model.predict_proba(self.X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        precision = precision_score(self.y_test, y_pred, average='weighted', zero_division=0)
        recall = recall_score(self.y_test, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average='weighted', zero_division=0)
        
        self.results['metrics'] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print("\nPerformance Metrics:")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        self.results['confusion_matrix'] = cm.tolist()
        
        print("\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        report = classification_report(self.y_test, y_pred, output_dict=True)
        self.results['classification_report'] = report
        
        print("\nClassification Report:")
        print(classification_report(self.y_test, y_pred))
        
        # Try to calculate ROC-AUC if binary or multiclass
        try:
            if len(np.unique(self.y_test)) == 2:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            else:
                roc_auc = roc_auc_score(self.y_test, y_pred_proba, multi_class='ovr', average='weighted')
            self.results['roc_auc_score'] = float(roc_auc)
            print(f"\nROC-AUC Score: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC-AUC: {e}")
        
        # Feature importance
        if hasattr(self.model, 'feature_importances_'):
            feature_importance = self.model.feature_importances_
            self.results['feature_importances'] = feature_importance.tolist()
            
            print("\nTop 10 Most Important Features:")
            top_indices = np.argsort(feature_importance)[-10:][::-1]
            for rank, idx in enumerate(top_indices, 1):
                print(f"  {rank}. Feature {idx}: {feature_importance[idx]:.6f}")
    
    def save_model(self):
        """Save the trained model and results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model
        model_path = self.output_dir / f'random_forest_model_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"\nModel saved to: {model_path}")
        
        # Save results as JSON
        results_path = self.output_dir / f'training_results_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=4)
        print(f"Results saved to: {results_path}")
        
        # Save model info
        info_path = self.output_dir / f'model_info_{timestamp}.txt'
        with open(info_path, 'w') as f:
            f.write("Random Forest Model Training Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Training Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Model Type: Random Forest Classifier\n")
            f.write(f"Number of Estimators: {self.model.n_estimators}\n")
            f.write(f"Max Depth: {self.model.max_depth}\n")
            f.write(f"Min Samples Split: {self.model.min_samples_split}\n\n")
            f.write("Performance Metrics:\n")
            f.write("-" * 50 + "\n")
            for metric, value in self.results.get('metrics', {}).items():
                f.write(f"{metric}: {value:.4f}\n")
        
        print(f"Model info saved to: {info_path}")
        
        return model_path


def main():
    """Main training pipeline."""
    # Set up paths
    base_dir = Path(__file__).parent.parent
    data_dir = base_dir / 'data' / 'processed'
    output_dir = Path(__file__).parent
    
    print("=" * 60)
    print("Random Forest Network Intrusion Detection System Training")
    print("=" * 60)
    
    # Initialize trainer
    trainer = RandomForestTrainer(
        data_dir=data_dir,
        output_dir=output_dir,
        batch_size=10000,
        random_state=42
    )
    
    # Load data
    if not trainer.load_data():
        sys.exit(1)
    
    # Train model
    trainer.train_batch(
        n_estimators=50,  # Reduced for faster training
        max_depth=20,
        min_samples_split=10
    )
    
    # Evaluate
    trainer.evaluate()
    
    # Save results
    trainer.save_model()
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
