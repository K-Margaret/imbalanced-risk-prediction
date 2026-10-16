"""
preprocessing.py
Custom scikit-learn transformers for handling high-cardinality categoricals 
and creating non-linear domain-specific features.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """
    Encodes high-cardinality categorical features by replacing categories 
    with their relative frequency (proportion) in the training set.
    
    This prevents the dimensionality explosion caused by One-Hot Encoding 
    while preserving the statistical weight of each category.
    """
    
    def __init__(self, columns=None, drop_original=True, 
                 missing_label='__MISSING__', unknown_value=0.0):
        """
        Args:
            columns (list, optional): Columns to encode. If None, encodes all object/category cols.
            drop_original (bool): Whether to drop the original categorical columns after encoding.
            missing_label (str): Temporary label used to calculate frequency of missing values.
            unknown_value (float): Value assigned to unseen categories during transform.
        """
        self.columns = columns
        self.drop_original = drop_original
        self.missing_label = missing_label
        self.unknown_value = unknown_value

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder expects a pandas DataFrame.")
            
        self.columns_ = self.columns if self.columns else X.select_dtypes(include=['object', 'category']).columns.tolist()
        self.freq_maps_ = {}
        
        for col in self.columns_:
            # Fill NaNs temporarily to include them in the frequency calculation
            filled = X[col].fillna(self.missing_label)
            # Calculate relative frequencies (proportions)
            self.freq_maps_[col] = filled.value_counts(normalize=True).to_dict()
            
        return self

    def transform(self, X):
        check_is_fitted(self, ['columns_', 'freq_maps_'])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("FrequencyEncoder expects a pandas DataFrame.")
            
        X = X.copy()
        
        for col in self.columns_:
            # Fill NaNs in test/val data
            X[col] = X[col].fillna(self.missing_label)
            
            # Map to frequencies; unseen categories automatically become NaN, which we fill with unknown_value
            freq_col_name = f"{col}_freq"
            X[freq_col_name] = X[col].map(self.freq_maps_[col]).fillna(self.unknown_value)
            
        if self.drop_original:
            X = X.drop(columns=self.columns_)
            
        # Store output feature names for sklearn compatibility
        self.feature_names_out_ = X.columns.to_numpy()
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['feature_names_out_'])
        return self.feature_names_out_


class NonLinearFeatureBuilder(BaseEstimator, TransformerMixin):
    """
    Creates non-linear interaction features and smoothed group means.
    Helps linear models (like Logistic Regression) capture complex, non-linear 
    relationships without relying on tree-based models.
    """
    
    def __init__(self, smoothing=20.0):
        """
        Args:
            smoothing (float): Smoothing factor for group means. Higher values pull 
                               rare categories closer to the global mean (prevents overfitting).
        """
        self.smoothing = smoothing
        self.groupby_maps_ = {}
        self.global_means_ = {}

    def _get_groupby_specs(self):
        # Define which categorical features to group by, and which numerical features to aggregate
        return [
            ("Make", "VehOdo"),
            ("Make", "VehBCost"),
        ]

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NonLinearFeatureBuilder expects a pandas DataFrame.")
            
        self.groupby_specs_ = self._get_groupby_specs()
        
        for cat_col, num_col in self.groupby_specs_:
            if cat_col in X.columns and num_col in X.columns:
                grouped = X.groupby(cat_col)[num_col]
                counts = grouped.count()
                means = grouped.mean()
                global_mean = X[num_col].mean()
                
                # Bayesian/Maddens smoothing: prevents extreme values for rare categories
                smoothed_mean = (counts * means + self.smoothing * global_mean) / (counts + self.smoothing)
                
                map_name = f"{cat_col}_mean_{num_col}"
                self.groupby_maps_[map_name] = smoothed_mean.to_dict()
                self.global_means_[map_name] = global_mean
                
        return self

    def transform(self, X):
        check_is_fitted(self, ['groupby_specs_'])
        if not isinstance(X, pd.DataFrame):
            raise TypeError("NonLinearFeatureBuilder expects a pandas DataFrame.")
            
        X = X.copy()
        
        # 1. Basic non-linear ratio features (adding +1 to denominators to prevent division by zero)
        if 'VehOdo' in X.columns and 'VehicleAge' in X.columns:
            X['odo_per_age'] = X['VehOdo'] / (X['VehicleAge'] + 1)
            
        if 'VehBCost' in X.columns and 'VehOdo' in X.columns:
            X['price_per_odo'] = X['VehBCost'] / (X['VehOdo'] + 1)
            
        if 'WarrantyCost' in X.columns and 'VehicleAge' in X.columns:
            X['warranty_per_age'] = X['WarrantyCost'] / (X['VehicleAge'] + 1)

        # 2. Smoothed group means
        for feature_name, mapping in self.groupby_maps_.items():
            cat_col = feature_name.split('_mean_')[0]
            if cat_col in X.columns:
                # Map values, fill unseen/missing with the global smoothed mean
                X[feature_name] = X[cat_col].map(mapping).fillna(self.global_means_[feature_name])

        # Store output feature names for sklearn compatibility
        self.feature_names_out_ = X.columns.to_numpy()
        return X

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, ['feature_names_out_'])
        return self.feature_names_out_