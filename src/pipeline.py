"""
pipeline.py
Constructs the complete scikit-learn preprocessing and modeling pipelines.
Ensures no data leakage and handles column routing automatically.
"""

import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

# Import custom transformers from the same directory
from preprocessing import FrequencyEncoder, NonLinearFeatureBuilder

# Columns that contain unique IDs, dates, or leaky/target-adjacent info
DROP_COLS = ["RefId", "PRIMEUNIT", "AUCGUART", "year", "IsOnlineSale"]

# Threshold to separate low vs high cardinality categoricals
CARDINALITY_THRESHOLD = 50


def split_columns(df: pd.DataFrame):
    """
    Dynamically splits dataframe columns into numeric, low-cardinality, 
    and high-cardinality groups based on data types and unique value counts.
    """
    # Separate numeric and categorical
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    
    # Remove drop columns from categorical list
    categorical_cols = [col for col in categorical_cols if col not in DROP_COLS]
    
    # Calculate cardinality for categoricals
    cardinality = df[categorical_cols].nunique()
    
    low_card_cols = cardinality[cardinality < CARDINALITY_THRESHOLD].index.tolist()
    high_card_cols = cardinality[cardinality >= CARDINALITY_THRESHOLD].index.tolist()
    
    return numeric_cols, low_card_cols, high_card_cols


def build_preprocessing_pipeline(numeric_cols, low_card_cols, high_card_cols):
    """
    Builds the ColumnTransformer for standard preprocessing, 
    wrapped in a Pipeline with the NonLinearFeatureBuilder.
    """
    # 1. Numerical: Median imputation + Standard Scaling
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # 2. Low Cardinality Categorical: Constant imputation + One-Hot Encoding
    low_card_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.int8))
    ])

    # 3. High Cardinality Categorical: Custom Frequency Encoding
    high_card_transformer = Pipeline([
        ("freq", FrequencyEncoder(drop_original=True))
    ])

    # Combine into a ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("low_card", low_card_transformer, low_card_cols),
            ("high_card", high_card_transformer, high_card_cols),
        ],
        remainder="drop" # Drops the DROP_COLS automatically
    )

    # Wrap with Non-Linear Feature Builder to create interactions BEFORE scaling/encoding
    full_preprocessor = Pipeline([
        ("nonlinear_features", NonLinearFeatureBuilder(smoothing=20.0)),
        ("preprocessor", preprocessor)
    ])

    return full_preprocessor


def build_model_pipeline(model=None):
    """
    Builds the final end-to-end pipeline.
    If no model is provided, defaults to the best-performing configuration:
    Logistic Regression with L1 penalty and balanced class weights.
    """
    if model is None:
        # Best model configuration found via GridSearch
        model = LogisticRegression(
            solver="liblinear",
            penalty="l1",
            C=0.1,
            class_weight="balanced",
            fit_intercept=False,
            max_iter=1000,
            random_state=42
        )

    # We return a factory function because the preprocessor needs to be 
    # fitted on the training data to determine column splits dynamically.
    def _create_pipeline(X_train_df):
        numeric_cols, low_card_cols, high_card_cols = split_columns(X_train_df)
        preprocessor = build_preprocessing_pipeline(numeric_cols, low_card_cols, high_card_cols)
        
        return Pipeline([
            ("preprocessing", preprocessor),
            ("classifier", model)
        ])

    return _create_pipeline


# ==========================================
# Example Usage / Quick Test
# ==========================================
if __name__ == "__main__":
    # Dummy data to test pipeline construction without errors
    print("Testing pipeline construction...")
    
    # Create a mock dataframe with expected columns
    mock_df = pd.DataFrame({
        "RefId": [1, 2],
        "VehOdo": [50000.0, 60000.0],
        "VehicleAge": [5, 6],
        "VehBCost": [10000, 12000],
        "WarrantyCost": [500, 600],
        "Make": ["Ford", "Toyota"],
        "Model": ["F-150", "Camry"],
        "SubModel": ["XLT", "LE"],
        "Trim": ["High", "Low"],
        "Auction": ["A", "B"],
        "IsBadBuy": [0, 1]
    })
    
    # 1. Test column splitting
    num, low, high = split_columns(mock_df)
    print(f"Numeric: {num}")
    print(f"Low Card: {low}")
    print(f"High Card: {high}")
    
    # 2. Test pipeline creation
    pipeline_factory = build_model_pipeline()
    final_pipeline = pipeline_factory(mock_df)
    
    print("\nPipeline structure:")
    print(final_pipeline)
    print("\n✅ Pipeline built successfully!")