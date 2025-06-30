"""
Model factory for creating and managing anomaly detection models
Supports statistical, deep learning, and time series models
"""

import numpy as np
from typing import Dict, Any, List, Optional, Union
import logging
from abc import ABC, abstractmethod

from .statistical_models import (
    IsolationForestDetector, PCADetector, OneClassSVMDetector,
    LocalOutlierFactorDetector, DBSCANDetector, GaussianMixtureDetector
)
from .deep_learning_models import (
    AutoencoderDetector, LSTMAutoencoderDetector, TransformerDetector
)
from .time_series_models import (
    ProphetDetector, ARIMADetector, SeasonalDecomposeDetector
)

logger = logging.getLogger(__name__)

class BaseAnomalyDetector(ABC):
    """
    Base class for all anomaly detectors
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.model = None
        self.is_fitted = False
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model and predict anomaly scores
        
        Args:
            X: Input data
            
        Returns:
            Array of anomaly scores
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the model
        """
        return {
            'name': self.__class__.__name__,
            'config': self.config,
            'is_fitted': self.is_fitted
        }

class ModelFactory:
    """
    Factory class for creating anomaly detection models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self._available_models = self._initialize_available_models()
    
    def _initialize_available_models(self) -> Dict[str, type]:
        """
        Initialize dictionary of available models
        """
        return {
            # Statistical models
            'isolation_forest': IsolationForestDetector,
            'pca': PCADetector,
            'one_class_svm': OneClassSVMDetector,
            'local_outlier_factor': LocalOutlierFactorDetector,
            'dbscan': DBSCANDetector,
            'gaussian_mixture': GaussianMixtureDetector,
            
            # Deep learning models
            'autoencoder': AutoencoderDetector,
            'lstm_autoencoder': LSTMAutoencoderDetector,
            'transformer': TransformerDetector,
            
            # Time series models
            'prophet': ProphetDetector,
            'arima': ARIMADetector,
            'seasonal_decompose': SeasonalDecomposeDetector
        }
    
    def get_available_models(self) -> List[str]:
        """
        Get list of available model names
        """
        return list(self._available_models.keys())
    
    def get_model(self, model_name: str) -> BaseAnomalyDetector:
        """
        Create and return a model instance
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            Instantiated model
        """
        if model_name not in self._available_models:
            raise ValueError(f"Unknown model: {model_name}. Available models: {self.get_available_models()}")
        
        try:
            model_class = self._available_models[model_name]
            model_config = self.config.get(model_name, {})
            
            logger.debug(f"Creating model: {model_name}")
            model = model_class(model_config)
            
            return model
            
        except Exception as e:
            logger.error(f"Error creating model {model_name}: {str(e)}")
            raise
    
    def get_models_by_category(self) -> Dict[str, List[str]]:
        """
        Get models grouped by category
        """
        categories = {
            'statistical': [
                'isolation_forest', 'pca', 'one_class_svm', 
                'local_outlier_factor', 'dbscan', 'gaussian_mixture'
            ],
            'deep_learning': [
                'autoencoder', 'lstm_autoencoder', 'transformer'
            ],
            'time_series': [
                'prophet', 'arima', 'seasonal_decompose'
            ]
        }
        
        # Filter out unavailable models
        available_models = self.get_available_models()
        filtered_categories = {}
        
        for category, models in categories.items():
            filtered_categories[category] = [
                model for model in models if model in available_models
            ]
        
        return filtered_categories
    
    def get_model_requirements(self, model_name: str) -> Dict[str, Any]:
        """
        Get requirements and constraints for a specific model
        """
        requirements = {
            # Statistical models
            'isolation_forest': {
                'min_samples': 10,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['scikit-learn']
            },
            'pca': {
                'min_samples': 50,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['scikit-learn']
            },
            'one_class_svm': {
                'min_samples': 20,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': False,
                'dependencies': ['scikit-learn']
            },
            'local_outlier_factor': {
                'min_samples': 20,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['scikit-learn']
            },
            'dbscan': {
                'min_samples': 10,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['scikit-learn']
            },
            'gaussian_mixture': {
                'min_samples': 50,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['scikit-learn']
            },
            
            # Deep learning models
            'autoencoder': {
                'min_samples': 100,
                'supports_multivariate': True,
                'supports_streaming': False,
                'memory_efficient': False,
                'dependencies': ['tensorflow', 'keras']
            },
            'lstm_autoencoder': {
                'min_samples': 200,
                'supports_multivariate': True,
                'supports_streaming': True,
                'memory_efficient': False,
                'dependencies': ['tensorflow', 'keras']
            },
            'transformer': {
                'min_samples': 500,
                'supports_multivariate': True,
                'supports_streaming': True,
                'memory_efficient': False,
                'dependencies': ['tensorflow', 'keras']
            },
            
            # Time series models
            'prophet': {
                'min_samples': 100,
                'supports_multivariate': False,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['prophet']
            },
            'arima': {
                'min_samples': 50,
                'supports_multivariate': False,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['statsmodels']
            },
            'seasonal_decompose': {
                'min_samples': 100,
                'supports_multivariate': False,
                'supports_streaming': False,
                'memory_efficient': True,
                'dependencies': ['statsmodels']
            }
        }
        
        return requirements.get(model_name, {
            'min_samples': 10,
            'supports_multivariate': True,
            'supports_streaming': False,
            'memory_efficient': True,
            'dependencies': []
        })
    
    def validate_model_compatibility(self, model_name: str, X: np.ndarray) -> bool:
        """
        Validate if a model is compatible with the given data
        
        Args:
            model_name: Name of the model
            X: Input data
            
        Returns:
            True if compatible, False otherwise
        """
        try:
            requirements = self.get_model_requirements(model_name)
            
            # Check minimum samples
            if X.shape[0] < requirements.get('min_samples', 10):
                logger.warning(f"Model {model_name} requires at least {requirements['min_samples']} samples, got {X.shape[0]}")
                return False
            
            # Check multivariate support
            if X.shape[1] > 1 and not requirements.get('supports_multivariate', True):
                logger.warning(f"Model {model_name} does not support multivariate data")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error validating model compatibility: {str(e)}")
            return False
    
    def get_recommended_models(self, X: np.ndarray, 
                             max_models: int = 5) -> List[str]:
        """
        Get recommended models based on data characteristics
        
        Args:
            X: Input data
            max_models: Maximum number of models to recommend
            
        Returns:
            List of recommended model names
        """
        try:
            n_samples, n_features = X.shape
            
            # Get all compatible models
            compatible_models = []
            for model_name in self.get_available_models():
                if self.validate_model_compatibility(model_name, X):
                    compatible_models.append(model_name)
            
            # Score models based on data characteristics
            model_scores = {}
            
            for model_name in compatible_models:
                score = 0
                requirements = self.get_model_requirements(model_name)
                
                # Prefer memory-efficient models for large datasets
                if n_samples > 10000 and requirements.get('memory_efficient', False):
                    score += 2
                
                # Prefer multivariate models for high-dimensional data
                if n_features > 10 and requirements.get('supports_multivariate', False):
                    score += 2
                
                # Prefer fast models for large datasets
                if n_samples > 50000:
                    if model_name in ['isolation_forest', 'local_outlier_factor', 'pca']:
                        score += 1
                
                # Prefer robust models
                if model_name in ['isolation_forest', 'pca', 'one_class_svm']:
                    score += 1
                
                model_scores[model_name] = score
            
            # Sort by score and return top recommendations
            recommended = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            return [model_name for model_name, _ in recommended[:max_models]]
            
        except Exception as e:
            logger.error(f"Error getting recommended models: {str(e)}")
            # Return default models
            return ['isolation_forest', 'pca', 'local_outlier_factor'][:max_models]
    
    def create_ensemble(self, model_names: List[str]) -> 'EnsembleDetector':
        """
        Create an ensemble of multiple models
        
        Args:
            model_names: List of model names to include in ensemble
            
        Returns:
            Ensemble detector
        """
        models = []
        for model_name in model_names:
            try:
                model = self.get_model(model_name)
                models.append((model_name, model))
            except Exception as e:
                logger.warning(f"Failed to create model {model_name} for ensemble: {str(e)}")
        
        if not models:
            raise ValueError("No valid models for ensemble")
        
        return EnsembleDetector(models, self.config.get('ensemble', {}))

class EnsembleDetector(BaseAnomalyDetector):
    """
    Ensemble anomaly detector that combines multiple models
    """
    
    def __init__(self, models: List[tuple], config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.models = models  # List of (name, model) tuples
        self.combination_method = config.get('combination_method', 'average')
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit all models and combine their predictions
        """
        try:
            all_scores = []
            successful_models = []
            
            for model_name, model in self.models:
                try:
                    logger.debug(f"Running ensemble model: {model_name}")
                    scores = model.fit_predict(X)
                    all_scores.append(scores)
                    successful_models.append(model_name)
                except Exception as e:
                    logger.warning(f"Model {model_name} failed in ensemble: {str(e)}")
                    continue
            
            if not all_scores:
                raise RuntimeError("All models in ensemble failed")
            
            # Combine scores
            scores_array = np.array(all_scores)
            
            if self.combination_method == 'average':
                combined_scores = np.mean(scores_array, axis=0)
            elif self.combination_method == 'max':
                combined_scores = np.max(scores_array, axis=0)
            elif self.combination_method == 'median':
                combined_scores = np.median(scores_array, axis=0)
            elif self.combination_method == 'weighted':
                # Simple weighted average (can be enhanced)
                weights = np.ones(len(all_scores)) / len(all_scores)
                combined_scores = np.average(scores_array, axis=0, weights=weights)
            else:
                combined_scores = np.mean(scores_array, axis=0)
            
            logger.info(f"Ensemble completed with {len(successful_models)} models: {successful_models}")
            self.is_fitted = True
            
            return combined_scores
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {str(e)}")
            raise