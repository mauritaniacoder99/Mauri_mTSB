"""
Statistical anomaly detection models
Includes PCA, Isolation Forest, One-Class SVM, LOF, DBSCAN, Gaussian Mixture
"""

import numpy as np
from typing import Dict, Any, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import MinMaxScaler
import warnings

from .model_factory import BaseAnomalyDetector

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

class IsolationForestDetector(BaseAnomalyDetector):
    """
    Isolation Forest anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.contamination = config.get('contamination', 0.1)
        self.n_estimators = config.get('n_estimators', 100)
        self.random_state = config.get('random_state', 42)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Isolation Forest and predict anomaly scores
        """
        try:
            self.model = IsolationForest(
                contamination=self.contamination,
                n_estimators=self.n_estimators,
                random_state=self.random_state,
                n_jobs=-1
            )
            
            # Fit and predict
            predictions = self.model.fit_predict(X)
            
            # Get anomaly scores (decision function returns negative scores for outliers)
            anomaly_scores = self.model.decision_function(X)
            
            # Normalize scores to [0, 1] range
            normalized_scores = self._normalize_scores(anomaly_scores)
            
            self.is_fitted = True
            logger.debug(f"Isolation Forest: {np.sum(predictions == -1)} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Isolation Forest: {str(e)}")
            # Return default scores
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize anomaly scores to [0, 1] range
        """
        scores = np.array(scores)
        # Isolation Forest returns negative values for outliers
        # Convert to positive and normalize
        scores = -scores  # Now higher values indicate more anomalous
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

class PCADetector(BaseAnomalyDetector):
    """
    PCA-based anomaly detector using reconstruction error
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_components = config.get('n_components', 0.95)
        self.contamination = config.get('contamination', 0.1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and predict anomaly scores based on reconstruction error
        """
        try:
            # Standardize data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            # Fit PCA
            self.model = PCA(n_components=self.n_components)
            X_transformed = self.model.fit_transform(X_scaled)
            
            # Reconstruct data
            X_reconstructed = self.model.inverse_transform(X_transformed)
            
            # Calculate reconstruction error
            reconstruction_errors = np.mean((X_scaled - X_reconstructed) ** 2, axis=1)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(reconstruction_errors)
            
            self.is_fitted = True
            logger.debug(f"PCA: Explained variance ratio: {self.model.explained_variance_ratio_.sum():.3f}")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in PCA detector: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize reconstruction errors to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use quantile-based normalization to handle outliers
        q99 = np.percentile(scores, 99)
        q1 = np.percentile(scores, 1)
        
        if q99 - q1 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q1) / (q99 - q1), 0, 1)
        return normalized

class OneClassSVMDetector(BaseAnomalyDetector):
    """
    One-Class SVM anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.nu = config.get('nu', 0.1)
        self.kernel = config.get('kernel', 'rbf')
        self.gamma = config.get('gamma', 'scale')
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit One-Class SVM and predict anomaly scores
        """
        try:
            # Standardize data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.model = OneClassSVM(
                nu=self.nu,
                kernel=self.kernel,
                gamma=self.gamma
            )
            
            # Fit and predict
            predictions = self.model.fit_predict(X_scaled)
            
            # Get decision scores
            decision_scores = self.model.decision_function(X_scaled)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(decision_scores)
            
            self.is_fitted = True
            logger.debug(f"One-Class SVM: {np.sum(predictions == -1)} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in One-Class SVM: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize decision scores to [0, 1] range
        """
        scores = np.array(scores)
        # SVM returns negative values for outliers
        scores = -scores
        
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if max_score - min_score == 0:
            return np.zeros_like(scores)
        
        normalized = (scores - min_score) / (max_score - min_score)
        return normalized

class LocalOutlierFactorDetector(BaseAnomalyDetector):
    """
    Local Outlier Factor (LOF) anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_neighbors = config.get('n_neighbors', 20)
        self.contamination = config.get('contamination', 0.1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit LOF and predict anomaly scores
        """
        try:
            # Ensure n_neighbors is not larger than number of samples
            n_neighbors = min(self.n_neighbors, X.shape[0] - 1)
            
            self.model = LocalOutlierFactor(
                n_neighbors=n_neighbors,
                contamination=self.contamination,
                n_jobs=-1
            )
            
            # Fit and predict
            predictions = self.model.fit_predict(X)
            
            # Get negative outlier factor scores
            outlier_scores = self.model.negative_outlier_factor_
            
            # Normalize scores
            normalized_scores = self._normalize_scores(outlier_scores)
            
            self.is_fitted = True
            logger.debug(f"LOF: {np.sum(predictions == -1)} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in LOF: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize LOF scores to [0, 1] range
        """
        scores = np.array(scores)
        # LOF returns negative values, convert to positive
        scores = -scores
        
        # Values > 1 indicate outliers
        normalized = np.clip((scores - 1) / np.max([np.max(scores) - 1, 1e-8]), 0, 1)
        
        return normalized

class DBSCANDetector(BaseAnomalyDetector):
    """
    DBSCAN-based anomaly detector (noise points as anomalies)
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.eps = config.get('eps', 0.5)
        self.min_samples = config.get('min_samples', 5)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit DBSCAN and predict anomaly scores
        """
        try:
            # Standardize data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.model = DBSCAN(
                eps=self.eps,
                min_samples=self.min_samples,
                n_jobs=-1
            )
            
            # Fit and predict
            cluster_labels = self.model.fit_predict(X_scaled)
            
            # Points with label -1 are noise (anomalies)
            anomaly_scores = np.zeros(X.shape[0])
            
            # Calculate distance-based scores for noise points
            noise_mask = cluster_labels == -1
            
            if np.any(noise_mask):
                # For noise points, calculate distance to nearest core point
                core_samples = self.model.core_sample_indices_
                
                if len(core_samples) > 0:
                    from sklearn.metrics.pairwise import pairwise_distances
                    
                    # Calculate distances from noise points to core samples
                    noise_points = X_scaled[noise_mask]
                    core_points = X_scaled[core_samples]
                    
                    distances = pairwise_distances(noise_points, core_points)
                    min_distances = np.min(distances, axis=1)
                    
                    # Normalize distances
                    if len(min_distances) > 0:
                        max_dist = np.max(min_distances)
                        if max_dist > 0:
                            anomaly_scores[noise_mask] = min_distances / max_dist
                        else:
                            anomaly_scores[noise_mask] = 1.0
                else:
                    # If no core samples, all noise points get max score
                    anomaly_scores[noise_mask] = 1.0
            
            self.is_fitted = True
            logger.debug(f"DBSCAN: {np.sum(noise_mask)} anomalies detected")
            
            return anomaly_scores
            
        except Exception as e:
            logger.error(f"Error in DBSCAN: {str(e)}")
            return np.zeros(X.shape[0])

class GaussianMixtureDetector(BaseAnomalyDetector):
    """
    Gaussian Mixture Model anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.n_components = config.get('n_components', 2)
        self.contamination = config.get('contamination', 0.1)
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Gaussian Mixture Model and predict anomaly scores
        """
        try:
            # Standardize data
            scaler = MinMaxScaler()
            X_scaled = scaler.fit_transform(X)
            
            self.model = GaussianMixture(
                n_components=self.n_components,
                random_state=42
            )
            
            # Fit model
            self.model.fit(X_scaled)
            
            # Calculate log-likelihood scores
            log_likelihood = self.model.score_samples(X_scaled)
            
            # Convert to anomaly scores (lower likelihood = higher anomaly score)
            anomaly_scores = -log_likelihood
            
            # Normalize scores
            normalized_scores = self._normalize_scores(anomaly_scores)
            
            self.is_fitted = True
            
            # Determine anomalies based on contamination parameter
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            
            logger.debug(f"Gaussian Mixture: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Gaussian Mixture: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize log-likelihood scores to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use robust normalization
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized