"""
Meta-learning model selector for automatic model selection
Implements MetaOD, FMMS, and Orthus approaches
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
import logging
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class MetaModelSelector:
    """
    Meta-learning based model selector for anomaly detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.enabled = config.get('enabled', True)
        self.methods = config.get('methods', ['metaod', 'fmms', 'orthus'])
        self.selection_criteria = config.get('selection_criteria', ['auc_roc', 'f1_score', 'isolation_score'])
        self.max_models = config.get('max_models', 5)
        self.min_models = config.get('min_models', 2)
        
        # Meta-features for model selection
        self.meta_features = {}
        
    def select_models(self, X: np.ndarray, 
                     available_models: Optional[List[str]] = None) -> List[str]:
        """
        Select best models based on meta-learning approaches
        
        Args:
            X: Input data
            available_models: List of available model names
            
        Returns:
            List of selected model names
        """
        if not self.enabled:
            return available_models or []
        
        try:
            logger.info("Starting meta-learning model selection...")
            
            # Extract meta-features from data
            self.meta_features = self._extract_meta_features(X)
            
            # Get available models if not provided
            if available_models is None:
                from .model_factory import ModelFactory
                factory = ModelFactory()
                available_models = factory.get_available_models()
            
            # Apply different selection methods
            selected_models = set()
            
            for method in self.methods:
                if method == 'metaod':
                    models = self._metaod_selection(X, available_models)
                elif method == 'fmms':
                    models = self._fmms_selection(X, available_models)
                elif method == 'orthus':
                    models = self._orthus_selection(X, available_models)
                else:
                    logger.warning(f"Unknown selection method: {method}")
                    continue
                
                selected_models.update(models)
            
            # Convert to list and apply constraints
            final_selection = list(selected_models)
            
            # Ensure we have at least min_models
            if len(final_selection) < self.min_models:
                # Add default models
                default_models = ['isolation_forest', 'pca', 'local_outlier_factor']
                for model in default_models:
                    if model in available_models and model not in final_selection:
                        final_selection.append(model)
                        if len(final_selection) >= self.min_models:
                            break
            
            # Limit to max_models
            if len(final_selection) > self.max_models:
                # Rank models and select top ones
                ranked_models = self._rank_models(final_selection, X)
                final_selection = ranked_models[:self.max_models]
            
            logger.info(f"Selected {len(final_selection)} models: {final_selection}")
            return final_selection
            
        except Exception as e:
            logger.error(f"Error in model selection: {str(e)}")
            # Return default selection
            return ['isolation_forest', 'pca', 'local_outlier_factor']
    
    def _extract_meta_features(self, X: np.ndarray) -> Dict[str, float]:
        """
        Extract meta-features from the dataset
        """
        try:
            n_samples, n_features = X.shape
            
            meta_features = {
                # Basic statistics
                'n_samples': n_samples,
                'n_features': n_features,
                'dimensionality_ratio': n_features / n_samples,
                
                # Data distribution features
                'mean_correlation': self._calculate_mean_correlation(X),
                'skewness': self._calculate_skewness(X),
                'kurtosis': self._calculate_kurtosis(X),
                
                # Outlier-related features
                'outlier_fraction': self._estimate_outlier_fraction(X),
                'isolation_score': self._calculate_isolation_score(X),
                
                # Complexity features
                'intrinsic_dimensionality': self._estimate_intrinsic_dimensionality(X),
                'clustering_tendency': self._calculate_clustering_tendency(X),
                
                # Noise features
                'noise_level': self._estimate_noise_level(X),
                'signal_to_noise_ratio': self._calculate_snr(X)
            }
            
            logger.debug(f"Extracted {len(meta_features)} meta-features")
            return meta_features
            
        except Exception as e:
            logger.error(f"Error extracting meta-features: {str(e)}")
            return {}
    
    def _metaod_selection(self, X: np.ndarray, available_models: List[str]) -> List[str]:
        """
        MetaOD-based model selection
        """
        try:
            selected = []
            
            # Rule-based selection based on meta-features
            n_samples = self.meta_features.get('n_samples', 0)
            n_features = self.meta_features.get('n_features', 0)
            dimensionality_ratio = self.meta_features.get('dimensionality_ratio', 0)
            outlier_fraction = self.meta_features.get('outlier_fraction', 0.1)
            
            # High-dimensional data
            if dimensionality_ratio > 0.1:
                if 'pca' in available_models:
                    selected.append('pca')
                if 'autoencoder' in available_models:
                    selected.append('autoencoder')
            
            # Large datasets
            if n_samples > 10000:
                if 'isolation_forest' in available_models:
                    selected.append('isolation_forest')
                if 'local_outlier_factor' in available_models:
                    selected.append('local_outlier_factor')
            
            # High outlier fraction
            if outlier_fraction > 0.15:
                if 'dbscan' in available_models:
                    selected.append('dbscan')
                if 'one_class_svm' in available_models:
                    selected.append('one_class_svm')
            
            # Time series characteristics
            if self._is_time_series_like(X):
                if 'lstm_autoencoder' in available_models:
                    selected.append('lstm_autoencoder')
                if 'prophet' in available_models:
                    selected.append('prophet')
            
            # Ensure at least one model is selected
            if not selected and available_models:
                selected.append(available_models[0])
            
            logger.debug(f"MetaOD selected: {selected}")
            return selected
            
        except Exception as e:
            logger.error(f"Error in MetaOD selection: {str(e)}")
            return []
    
    def _fmms_selection(self, X: np.ndarray, available_models: List[str]) -> List[str]:
        """
        Fast Model Selection Strategy (FMMS)
        """
        try:
            selected = []
            
            # Quick performance estimation for each model
            model_scores = {}
            
            from .model_factory import ModelFactory
            factory = ModelFactory()
            
            # Sample data for quick evaluation
            sample_size = min(1000, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            for model_name in available_models[:6]:  # Limit to avoid long computation
                try:
                    # Quick model evaluation
                    model = factory.get_model(model_name)
                    scores = model.fit_predict(X_sample)
                    
                    # Calculate simple metrics
                    score_variance = np.var(scores)
                    score_range = np.max(scores) - np.min(scores)
                    
                    # Combined score
                    combined_score = score_variance * score_range
                    model_scores[model_name] = combined_score
                    
                except Exception as e:
                    logger.debug(f"FMMS evaluation failed for {model_name}: {str(e)}")
                    continue
            
            # Select top performing models
            if model_scores:
                sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
                selected = [model for model, _ in sorted_models[:3]]
            
            logger.debug(f"FMMS selected: {selected}")
            return selected
            
        except Exception as e:
            logger.error(f"Error in FMMS selection: {str(e)}")
            return []
    
    def _orthus_selection(self, X: np.ndarray, available_models: List[str]) -> List[str]:
        """
        Orthus-based model selection (diversity-based)
        """
        try:
            selected = []
            
            # Select models based on diversity and complementarity
            model_categories = {
                'statistical': ['isolation_forest', 'pca', 'one_class_svm', 'local_outlier_factor'],
                'clustering': ['dbscan', 'gaussian_mixture'],
                'deep_learning': ['autoencoder', 'lstm_autoencoder', 'transformer'],
                'time_series': ['prophet', 'arima', 'seasonal_decompose']
            }
            
            # Select at least one model from each available category
            for category, models in model_categories.items():
                available_in_category = [m for m in models if m in available_models]
                
                if available_in_category:
                    # Select the most suitable model from this category
                    best_model = self._select_best_from_category(X, available_in_category, category)
                    if best_model:
                        selected.append(best_model)
            
            logger.debug(f"Orthus selected: {selected}")
            return selected
            
        except Exception as e:
            logger.error(f"Error in Orthus selection: {str(e)}")
            return []
    
    def _select_best_from_category(self, X: np.ndarray, models: List[str], category: str) -> Optional[str]:
        """
        Select the best model from a specific category
        """
        try:
            # Category-specific selection logic
            if category == 'statistical':
                # Prefer isolation forest for general use
                if 'isolation_forest' in models:
                    return 'isolation_forest'
                elif 'pca' in models:
                    return 'pca'
                else:
                    return models[0]
            
            elif category == 'clustering':
                # Choose based on data characteristics
                if self.meta_features.get('clustering_tendency', 0) > 0.5:
                    return 'gaussian_mixture' if 'gaussian_mixture' in models else models[0]
                else:
                    return 'dbscan' if 'dbscan' in models else models[0]
            
            elif category == 'deep_learning':
                # Choose based on data size and complexity
                n_samples = self.meta_features.get('n_samples', 0)
                if n_samples > 1000:
                    if self._is_time_series_like(X) and 'lstm_autoencoder' in models:
                        return 'lstm_autoencoder'
                    elif 'autoencoder' in models:
                        return 'autoencoder'
                return models[0] if models else None
            
            elif category == 'time_series':
                # Choose based on time series characteristics
                if self._is_time_series_like(X):
                    return 'prophet' if 'prophet' in models else models[0]
                return None
            
            return models[0] if models else None
            
        except Exception:
            return models[0] if models else None
    
    def _rank_models(self, models: List[str], X: np.ndarray) -> List[str]:
        """
        Rank models based on expected performance
        """
        try:
            model_scores = {}
            
            # Simple ranking based on meta-features
            for model in models:
                score = 0
                
                # Model-specific scoring
                if model == 'isolation_forest':
                    score += 3  # Generally robust
                elif model == 'pca':
                    score += 2 + (1 if self.meta_features.get('dimensionality_ratio', 0) > 0.1 else 0)
                elif model == 'local_outlier_factor':
                    score += 2 + (1 if self.meta_features.get('clustering_tendency', 0) > 0.5 else 0)
                elif model == 'autoencoder':
                    score += 1 + (2 if self.meta_features.get('n_samples', 0) > 1000 else 0)
                elif model in ['lstm_autoencoder', 'transformer']:
                    score += 1 + (2 if self._is_time_series_like(X) else 0)
                else:
                    score += 1
                
                model_scores[model] = score
            
            # Sort by score
            ranked = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            return [model for model, _ in ranked]
            
        except Exception as e:
            logger.error(f"Error ranking models: {str(e)}")
            return models
    
    # Helper methods for meta-feature calculation
    
    def _calculate_mean_correlation(self, X: np.ndarray) -> float:
        """Calculate mean absolute correlation between features"""
        try:
            if X.shape[1] < 2:
                return 0.0
            
            corr_matrix = np.corrcoef(X.T)
            # Remove diagonal and get upper triangle
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            correlations = corr_matrix[mask]
            
            return float(np.mean(np.abs(correlations)))
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, X: np.ndarray) -> float:
        """Calculate mean skewness across features"""
        try:
            from scipy.stats import skew
            skewness_values = [skew(X[:, i]) for i in range(X.shape[1])]
            return float(np.mean(np.abs(skewness_values)))
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, X: np.ndarray) -> float:
        """Calculate mean kurtosis across features"""
        try:
            from scipy.stats import kurtosis
            kurtosis_values = [kurtosis(X[:, i]) for i in range(X.shape[1])]
            return float(np.mean(kurtosis_values))
        except Exception:
            return 0.0
    
    def _estimate_outlier_fraction(self, X: np.ndarray) -> float:
        """Estimate fraction of outliers using IQR method"""
        try:
            outlier_counts = []
            
            for i in range(X.shape[1]):
                q75, q25 = np.percentile(X[:, i], [75, 25])
                iqr = q75 - q25
                
                if iqr > 0:
                    lower_bound = q25 - 1.5 * iqr
                    upper_bound = q75 + 1.5 * iqr
                    
                    outliers = np.sum((X[:, i] < lower_bound) | (X[:, i] > upper_bound))
                    outlier_counts.append(outliers)
            
            if outlier_counts:
                return float(np.mean(outlier_counts) / X.shape[0])
            else:
                return 0.1  # Default
                
        except Exception:
            return 0.1
    
    def _calculate_isolation_score(self, X: np.ndarray) -> float:
        """Calculate isolation-based score"""
        try:
            from sklearn.ensemble import IsolationForest
            
            # Use small sample for speed
            sample_size = min(500, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            iso_forest = IsolationForest(n_estimators=50, random_state=42)
            scores = iso_forest.fit(X_sample).decision_function(X_sample)
            
            return float(np.std(scores))
            
        except Exception:
            return 0.0
    
    def _estimate_intrinsic_dimensionality(self, X: np.ndarray) -> float:
        """Estimate intrinsic dimensionality using PCA"""
        try:
            from sklearn.decomposition import PCA
            
            pca = PCA()
            pca.fit(X)
            
            # Find number of components for 95% variance
            cumsum = np.cumsum(pca.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= 0.95) + 1
            
            return float(n_components / X.shape[1])
            
        except Exception:
            return 1.0
    
    def _calculate_clustering_tendency(self, X: np.ndarray) -> float:
        """Calculate clustering tendency using Hopkins statistic"""
        try:
            from sklearn.neighbors import NearestNeighbors
            
            # Sample for efficiency
            sample_size = min(200, X.shape[0])
            indices = np.random.choice(X.shape[0], sample_size, replace=False)
            X_sample = X[indices]
            
            # Hopkins statistic approximation
            nbrs = NearestNeighbors(n_neighbors=2).fit(X_sample)
            distances, _ = nbrs.kneighbors(X_sample)
            
            # Use second nearest neighbor (first is the point itself)
            nn_distances = distances[:, 1]
            
            # Generate random points
            random_points = np.random.uniform(
                X_sample.min(axis=0), X_sample.max(axis=0), X_sample.shape
            )
            
            random_distances, _ = nbrs.kneighbors(random_points)
            random_nn_distances = random_distances[:, 0]  # Nearest neighbor to random points
            
            # Hopkins statistic
            hopkins = np.sum(random_nn_distances) / (np.sum(nn_distances) + np.sum(random_nn_distances))
            
            return float(hopkins)
            
        except Exception:
            return 0.5
    
    def _estimate_noise_level(self, X: np.ndarray) -> float:
        """Estimate noise level in the data"""
        try:
            # Use difference between consecutive points as noise estimate
            if X.shape[0] < 2:
                return 0.0
            
            diffs = np.diff(X, axis=0)
            noise_level = np.mean(np.std(diffs, axis=0))
            
            return float(noise_level)
            
        except Exception:
            return 0.0
    
    def _calculate_snr(self, X: np.ndarray) -> float:
        """Calculate signal-to-noise ratio"""
        try:
            signal_power = np.mean(np.var(X, axis=0))
            noise_level = self._estimate_noise_level(X)
            
            if noise_level > 0:
                snr = signal_power / (noise_level ** 2)
                return float(snr)
            else:
                return float('inf')
                
        except Exception:
            return 1.0
    
    def _is_time_series_like(self, X: np.ndarray) -> bool:
        """Check if data has time series characteristics"""
        try:
            # Simple heuristic: check for temporal patterns
            if X.shape[0] < 10:
                return False
            
            # Check for autocorrelation in first feature
            from scipy.stats import pearsonr
            
            y = X[:, 0]
            if len(y) > 1:
                # Check correlation with lagged version
                lagged = np.roll(y, 1)[1:]
                original = y[1:]
                
                corr, p_value = pearsonr(original, lagged)
                
                # If significant autocorrelation exists, likely time series
                return abs(corr) > 0.3 and p_value < 0.05
            
            return False
            
        except Exception:
            return False
    
    def get_meta_features(self) -> Dict[str, float]:
        """Get extracted meta-features"""
        return self.meta_features.copy()
    
    def get_selection_summary(self) -> Dict[str, Any]:
        """Get summary of the selection process"""
        return {
            'enabled': self.enabled,
            'methods_used': self.methods,
            'selection_criteria': self.selection_criteria,
            'constraints': {
                'max_models': self.max_models,
                'min_models': self.min_models
            },
            'meta_features': self.meta_features
        }