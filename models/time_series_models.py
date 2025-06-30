"""
Time series specific anomaly detection models
Includes Prophet, ARIMA, and Seasonal Decomposition
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging
import warnings

warnings.filterwarnings('ignore')

from .model_factory import BaseAnomalyDetector

logger = logging.getLogger(__name__)

class ProphetDetector(BaseAnomalyDetector):
    """
    Facebook Prophet-based anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.changepoint_prior_scale = config.get('changepoint_prior_scale', 0.05)
        self.seasonality_prior_scale = config.get('seasonality_prior_scale', 10.0)
        self.contamination = config.get('contamination', 0.1)
        
        try:
            from prophet import Prophet
            self.Prophet = Prophet
            self.prophet_available = True
        except ImportError:
            logger.warning("Prophet not available. Install with: pip install prophet")
            self.prophet_available = False
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit Prophet model and predict anomaly scores
        """
        if not self.prophet_available:
            logger.error("Prophet is not available")
            return np.zeros(X.shape[0])
        
        try:
            # Prophet works with univariate time series
            # Use first column or mean of all columns
            if X.shape[1] == 1:
                y = X[:, 0]
            else:
                y = np.mean(X, axis=1)
            
            # Create timestamps (assuming regular intervals)
            timestamps = pd.date_range(start='2020-01-01', periods=len(y), freq='H')
            
            # Prepare data for Prophet
            df = pd.DataFrame({
                'ds': timestamps,
                'y': y
            })
            
            # Initialize and fit Prophet model
            self.model = self.Prophet(
                changepoint_prior_scale=self.changepoint_prior_scale,
                seasonality_prior_scale=self.seasonality_prior_scale,
                daily_seasonality=True,
                weekly_seasonality=True,
                yearly_seasonality=False
            )
            
            # Suppress Prophet's verbose output
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.model.fit(df)
            
            # Make predictions
            forecast = self.model.predict(df)
            
            # Calculate residuals
            residuals = np.abs(y - forecast['yhat'].values)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(residuals)
            
            self.is_fitted = True
            
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"Prophet: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Prophet detector: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize residual scores to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use robust normalization
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized

class ARIMADetector(BaseAnomalyDetector):
    """
    ARIMA-based anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.order = tuple(config.get('order', (1, 1, 1)))
        self.contamination = config.get('contamination', 0.1)
        
        try:
            from statsmodels.tsa.arima.model import ARIMA
            self.ARIMA = ARIMA
            self.arima_available = True
        except ImportError:
            logger.warning("statsmodels not available. Install with: pip install statsmodels")
            self.arima_available = False
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit ARIMA model and predict anomaly scores
        """
        if not self.arima_available:
            logger.error("ARIMA is not available")
            return np.zeros(X.shape[0])
        
        try:
            # ARIMA works with univariate time series
            if X.shape[1] == 1:
                y = X[:, 0]
            else:
                y = np.mean(X, axis=1)
            
            # Fit ARIMA model
            self.model = self.ARIMA(y, order=self.order)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitted_model = self.model.fit()
            
            # Get fitted values and residuals
            fitted_values = fitted_model.fittedvalues
            residuals = np.abs(y - fitted_values)
            
            # Handle NaN values (common in ARIMA)
            if np.any(np.isnan(residuals)):
                # Fill NaN with median residual
                median_residual = np.nanmedian(residuals)
                residuals = np.nan_to_num(residuals, nan=median_residual)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(residuals)
            
            self.is_fitted = True
            
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"ARIMA: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in ARIMA detector: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize residual scores to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use robust normalization
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized

class SeasonalDecomposeDetector(BaseAnomalyDetector):
    """
    Seasonal decomposition-based anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.model_type = config.get('model', 'additive')  # 'additive' or 'multiplicative'
        self.period = config.get('period', None)  # Auto-detect if None
        self.contamination = config.get('contamination', 0.1)
        
        try:
            from statsmodels.tsa.seasonal import seasonal_decompose
            self.seasonal_decompose = seasonal_decompose
            self.decompose_available = True
        except ImportError:
            logger.warning("statsmodels not available. Install with: pip install statsmodels")
            self.decompose_available = False
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Perform seasonal decomposition and predict anomaly scores
        """
        if not self.decompose_available:
            logger.error("Seasonal decomposition is not available")
            return np.zeros(X.shape[0])
        
        try:
            # Works with univariate time series
            if X.shape[1] == 1:
                y = X[:, 0]
            else:
                y = np.mean(X, axis=1)
            
            # Auto-detect period if not specified
            if self.period is None:
                period = self._detect_period(y)
            else:
                period = self.period
            
            # Ensure minimum length for decomposition
            if len(y) < 2 * period:
                logger.warning(f"Time series too short for period {period}. Using simple residuals.")
                return self._simple_residual_detection(y)
            
            # Create time series
            ts = pd.Series(y, index=pd.date_range(start='2020-01-01', periods=len(y), freq='H'))
            
            # Perform seasonal decomposition
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                decomposition = self.seasonal_decompose(
                    ts, 
                    model=self.model_type, 
                    period=period
                )
            
            # Use residuals as anomaly indicators
            residuals = np.abs(decomposition.resid.values)
            
            # Handle NaN values
            if np.any(np.isnan(residuals)):
                median_residual = np.nanmedian(residuals)
                residuals = np.nan_to_num(residuals, nan=median_residual)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(residuals)
            
            self.is_fitted = True
            
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"Seasonal Decompose: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Seasonal Decomposition: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _detect_period(self, y: np.ndarray) -> int:
        """
        Auto-detect period using autocorrelation
        """
        try:
            from scipy import signal
            
            # Calculate autocorrelation
            autocorr = signal.correlate(y, y, mode='full')
            autocorr = autocorr[autocorr.size // 2:]
            
            # Find peaks in autocorrelation
            peaks, _ = signal.find_peaks(autocorr[1:], height=0.1 * np.max(autocorr))
            
            if len(peaks) > 0:
                # Return the first significant peak as period
                return peaks[0] + 1
            else:
                # Default period based on data length
                return max(2, len(y) // 10)
                
        except Exception:
            # Fallback to simple heuristic
            return max(2, len(y) // 10)
    
    def _simple_residual_detection(self, y: np.ndarray) -> np.ndarray:
        """
        Simple residual-based detection when decomposition fails
        """
        try:
            # Use moving average as baseline
            window_size = max(2, len(y) // 20)
            
            # Calculate moving average
            moving_avg = pd.Series(y).rolling(window=window_size, center=True).mean()
            
            # Fill NaN values
            moving_avg = moving_avg.fillna(method='bfill').fillna(method='ffill')
            
            # Calculate residuals
            residuals = np.abs(y - moving_avg.values)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(residuals)
            
            return normalized_scores
            
        except Exception:
            # Ultimate fallback
            return np.zeros(len(y))
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize residual scores to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use robust normalization
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized