"""
train.py
End-to-end training script for the Car Auction Risk Prediction model.
Handles time-series data splitting, pipeline execution, evaluation, and model saving.
"""

import sys
import logging
from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import (
    accuracy_score, 
    roc_auc_score, 
    average_precision_score, 
    classification_report
)

# Add src directory to path to allow importing pipeline modules
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.pipeline import build_model_pipeline

# ==========================================
# 1. Configuration & Logging
# ==========================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

RANDOM_STATE = 42
TARGET_COL = "IsBadBuy"
DATE_COL = "PurchDate"

# Paths (Assuming you run this from the root of the repository)
DATA_DIR = Path("../data/raw")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Columns to drop before modeling (IDs, leaky info, or dates)
DROP_COLS = ["RefId", "PRIMEUNIT", "AUCGUART", "year", "IsOnlineSale", DATE_COL]


# ==========================================
# 2. Data Loading & Time-Series Splitting
# ==========================================
def load_and_split_data(data_path: Path) -> tuple:
    """
    Loads the dataset and performs a strict chronological split.
    This prevents data leakage from future auctions into past predictions.
    """
    logging.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Parse dates
    df[DATE_COL] = pd.to_datetime(df[DATE_COL], format="%m/%d/%Y")
    
    # Sort by date just to be safe
    df = df.sort_values(DATE_COL)
    
    # Chronological 3-way split (Train / Val / Test)
    unique_dates = sorted(df[DATE_COL].unique())
    n_dates = len(unique_dates)
    split_size = int(n_dates / 3)
    
    train_dates = unique_dates[:split_size]
    val_dates = unique_dates[split_size : split_size * 2]
    test_dates = unique_dates[split_size * 2 :]
    
    train_df = df[df[DATE_COL].isin(train_dates)]
    val_df = df[df[DATE_COL].isin(val_dates)]
    test_df = df[df[DATE_COL].isin(test_dates)]
    
    logging.info(f"Split complete -> Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Separate features and target, drop irrelevant columns
    def prepare_df(df_subset):
        X = df_subset.drop(columns=[TARGET_COL] + DROP_COLS, errors='ignore')
        y = df_subset[TARGET_COL]
        return X, y

    X_train, y_train = prepare_df(train_df)
    X_val, y_val = prepare_df(val_df)
    X_test, y_test = prepare_df(test_df)
    
    return X_train, y_train, X_val, y_val, X_test, y_test


# ==========================================
# 3. Evaluation Metrics
# ==========================================
def evaluate_model(model, X, y, dataset_name: str):
    """Calculates and prints key metrics for a given dataset."""
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]
    
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    gini = 2 * auc - 1
    pr_auc = average_precision_score(y, y_proba)
    
    logging.info(f"\n--- {dataset_name} Results ---")
    logging.info(f"Accuracy : {acc:.4f}")
    logging.info(f"ROC-AUC  : {auc:.4f}")
    logging.info(f"Gini     : {gini:.4f}  <-- Primary Metric")
    logging.info(f"PR-AUC   : {pr_auc:.4f}")
    
    return {"accuracy": acc, "auc": auc, "gini": gini, "pr_auc": pr_auc}


# ==========================================
# 4. Main Execution Pipeline
# ==========================================
def main():
    # 1. Load and split data
    data_path = DATA_DIR / "training.csv"
    if not data_path.exists():
        logging.error(f"Data not found at {data_path}. Please download the Kaggle dataset and place it in data/raw/")
        sys.exit(1)
        
    X_train, y_train, X_val, y_val, X_test, y_test = load_and_split_data(data_path)
    
    # 2. Build the Pipeline
    # Note: build_model_pipeline returns a factory function because the preprocessor 
    # needs to dynamically inspect X_train to determine cardinality thresholds.
    logging.info("Building preprocessing and modeling pipeline...")
    pipeline_factory = build_model_pipeline()
    full_pipeline = pipeline_factory(X_train)
    
    # 3. Train the Model
    logging.info("Training the model (Logistic Regression with L1 penalty)...")
    full_pipeline.fit(X_train, y_train)
    
    # 4. Evaluate on all sets
    logging.info("Evaluating model performance...")
    train_metrics = evaluate_model(full_pipeline, X_train, y_train, "Train Set")
    val_metrics = evaluate_model(full_pipeline, X_val, y_val, "Validation Set")
    test_metrics = evaluate_model(full_pipeline, X_test, y_test, "Test Set")
    
    # 5. Save the final model
    model_save_path = MODEL_DIR / "best_pipeline.joblib"
    joblib.dump(full_pipeline, model_save_path)
    logging.info(f"\n✅ Model successfully saved to {model_save_path}")
    
    # 6. Print Final Summary
    logging.info("\n" + "="*40)
    logging.info("FINAL SUMMARY")
    logging.info("="*40)
    logging.info(f"Train Gini : {train_metrics['gini']:.4f}")
    logging.info(f"Val Gini   : {val_metrics['gini']:.4f}")
    logging.info(f"Test Gini  : {test_metrics['gini']:.4f}")
    logging.info("="*40 + "\n")


if __name__ == "__main__":
    main()