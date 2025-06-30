"""
Time series preprocessing utilities for anomaly detection
Handles missing values, normalization, feature engineering
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any, List
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Comprehensive preprocessing for multivariate time series data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.scaler = None
        self.imputer = None
        self.pca = None
        self.feature_columns = None
        self.timestamp_column = None
        
    def fit_transform(self, df: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit preprocessor and transform the data
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (processed_features, timestamps)
        """
        try:
            logger.info("Starting data preprocessing...")
            
            # Identify feature and timestamp columns
            self._identify_columns(df)
            
            # Extract features and timestamps
            features = df[self.feature_columns].copy()
            timestamps = self._extract_timestamps(df)
            
            # Handle missing values
            features = self._handle_missing_values(features)
            
            # Feature engineering
            features = self._engineer_features(features, df)
            
            # Normalize features
            features_scaled = self._normalize_features(features)
            
            # Dimensionality reduction if specified
            if self.config.get('apply_pca', False):
                features_scaled = self._apply_pca(features_scaled)
            
            # Convert to numpy array
            X = features_scaled.values if isinstance(features_scaled, pd.DataFrame) else features_scaled
            
            logger.info(f"Preprocessing completed. Final shape: {X.shape}")
            
            return X, timestamps
            
        except Exception as e:
            logger.error(f"Error in preprocessing: {str(e)}")
            raise
    
    def _identify_columns(self, df: pd.DataFrame):
        """
        Identify feature columns and timestamp column
        """
        # Identify timestamp column
        timestamp_candidates = ['timestamp', 'time', 'datetime', 'date_time', 'ts']
        self.timestamp_column = None
        
        for col in timestamp_candidates:
            if col in df.columns:
                self.timestamp_column = col
                break
        
        # If no standard timestamp column, look for datetime-like columns
        if not self.timestamp_column:
            for col in df.columns:
                if df[col].dtype in ['datetime64[ns]', 'datetime64[ns, UTC]'] or \
                   (df[col].dtype == 'object' and self._is_datetime_like(df[col])):
                    self.timestamp_column = col
                    break
        
        # Identify feature columns (numeric columns excluding timestamp)
        numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove timestamp column if it's numeric
        if self.timestamp_column and self.timestamp_column in numeric_columns:
            numeric_columns.remove(self.timestamp_column)
        
        # Remove ID-like columns
        id_columns = [col for col in numeric_columns if 
                     any(keyword in col.lower() for keyword in ['id', 'index', '_id'])]
        numeric_columns = [col for col in numeric_columns if col not in id_columns]
        
        self.feature_columns = numeric_columns
        
        logger.info(f"Identified {len(self.feature_columns)} feature columns")
        if self.timestamp_column:
            logger.info(f"Timestamp column: {self.timestamp_column}")
    
    def _is_datetime_like(self, series: pd.Series) -> bool:
        """
        Check if a series contains datetime-like strings
        """
        try:
            sample = series.dropna().iloc[:5]
            for val in sample:
                pd.to_datetime(val)
            return True
        except (ValueError, TypeError):
            return False
    
    def _extract_timestamps(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Extract timestamps as numpy array
        """
        if not self.timestamp_column:
            return None
        
        try:
            timestamps = pd.to_datetime(df[self.timestamp_column])
            return timestamps.values
        except Exception as e:
            logger.warning(f"Error extracting timestamps: {str(e)}")
            return None
    
    def _handle_missing_values(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in features
        """
        missing_strategy = self.config.get('missing_strategy', 'mean')
        
        if features.isnull().sum().sum() == 0:
            logger.info("No missing values found")
            return features
        
        logger.info(f"Handling missing values using strategy: {missing_strategy}")
        
        if missing_strategy == 'drop':
            return features.dropna()
        elif missing_strategy == 'knn':
            self.imputer = KNNImputer(n_neighbors=5)
        else:
            self.imputer = SimpleImputer(strategy=missing_strategy)
        
        # Fit and transform
        features_imputed = self.imputer.fit_transform(features)
        return pd.DataFrame(features_imputed, columns=features.columns, index=features.index)
    
    def _engineer_features(self, features: pd.DataFrame, original_df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer additional features for anomaly detection
        """
        if not self.config.get('feature_engineering', True):
            return features
        
        logger.info("Engineering additional features...")
        
        # Add statistical features
        if self.config.get('add_statistical_features', True):
            # Rolling statistics
            window_size = self.config.get('rolling_window', 10)
            
            for col in features.columns:
                if features[col].dtype in [np.float64, np.int64]:
                    # Rolling mean and std
                    features[f'{col}_rolling_mean'] = features[col].rolling(
                        window=window_size, min_periods=1).mean()
                    features[f'{col}_rolling_std'] = features[col].rolling(
                        window=window_size, min_periods=1).std().fillna(0)
                    
                    # Lag features
                    for lag in [1, 2, 3]:
                        features[f'{col}_lag_{lag}'] = features[col].shift(lag).fillna(0)
        
        # Add time-based features if timestamp is available
        if self.timestamp_column and self.timestamp_column in original_df.columns:
            try:
                ts = pd.to_datetime(original_df[self.timestamp_column])
                
                # Time-based features
                features['hour_of_day'] = ts.dt.hour
                features['day_of_week'] = ts.dt.dayofweek
                features['is_weekend'] = (ts.dt.dayofweek >= 5).astype(int)
                features['month'] = ts.dt.month
                
                # Cyclical encoding for time features
                features['hour_sin'] = np.sin(2 * np.pi * features['hour_of_day'] / 24)
                features['hour_cos'] = np.cos(2 * np.pi * features['hour_of_day'] / 24)
                features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
                features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
                
            except Exception as e:
                logger.warning(f"Error adding time-based features: {str(e)}")
        
        # Add interaction features
        if self.config.get('add_interaction_features', False):
            numeric_cols = features.select_dtypes(include=[np.number]).columns[:5]  # Limit to first 5 to avoid explosion
            for i, col1 in enumerate(numeric_cols):
                for col2 in numeric_cols[i+1:]:
                    features[f'{col1}_x_{col2}'] = features[col1] * features[col2]
        
        logger.info(f"Feature engineering completed. New shape: {features.shape}")
        return features
    
    def _normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize features using specified scaler
        """
        scaling_method = self.config.get('scaling_method', 'standard')
        
        logger.info(f"Normalizing features using {scaling_method} scaling")
        
        if scaling_method == 'standard':
            self.scaler = StandardScaler()
        elif scaling_method == 'minmax':
            self.scaler = MinMaxScaler()
        elif scaling_method == 'robust':
            self.scaler = RobustScaler()
        else:
            logger.warning(f"Unknown scaling method: {scaling_method}. Using standard scaling.")
            self.scaler = StandardScaler()
        
        # Fit and transform
        features_scaled = self.scaler.fit_transform(features)
        return pd.DataFrame(features_scaled, columns=features.columns, index=features.index)
    
    def _apply_pca(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Apply PCA for dimensionality reduction
        """
        n_components = self.config.get('pca_components', 0.95)
        
        logger.info(f"Applying PCA with {n_components} components")
        
        self.pca = PCA(n_components=n_components)
        features_pca = self.pca.fit_transform(features)
        
        # Create column names for PCA components
        pca_columns = [f'PC_{i+1}' for i in range(features_pca.shape[1])]
        
        logger.info(f"PCA explained variance ratio: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        return pd.DataFrame(features_pca, columns=pca_columns, index=features.index)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores if PCA was applied
        """
        if self.pca is None:
            return {}
        
        # Get absolute values of PCA components
        components = np.abs(self.pca.components_)
        
        # Calculate feature importance as the sum of absolute loadings
        feature_importance = {}
        for i, feature in enumerate(self.feature_columns):
            importance = np.sum(components[:, i])
            feature_importance[feature] = importance
        
        return dict(sorted(feature_importance.items(), key=lambda x: x[1], reverse=True))
    
    def get_preprocessing_summary(self) -> Dict[str, Any]:
        """
        Get summary of preprocessing steps applied
        """
        summary = {
            'feature_columns': self.feature_columns,
            'timestamp_column': self.timestamp_column,
            'scaling_method': self.config.get('scaling_method', 'standard'),
            'missing_strategy': self.config.get('missing_strategy', 'mean'),
            'pca_applied': self.pca is not None,
            'feature_engineering': self.config.get('feature_engineering', True)
        }
        
        if self.pca is not None:
            summary['pca_components'] = self.pca.n_components_
            summary['explained_variance'] = float(self.pca.explained_variance_ratio_.sum())
        
        return summary