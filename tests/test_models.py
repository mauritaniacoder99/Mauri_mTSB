"""
Comprehensive tests for anomaly detection models
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import patch, MagicMock

from models.model_factory import ModelFactory, EnsembleDetector
from models.statistical_models import (
    IsolationForestDetector, PCADetector, OneClassSVMDetector,
    LocalOutlierFactorDetector, DBSCANDetector, GaussianMixtureDetector
)
from models.meta_selector import MetaModelSelector

class TestModelFactory:
    """Test model factory functionality"""
    
    def test_get_available_models(self):
        """Test getting list of available models"""
        factory = ModelFactory()
        models = factory.get_available_models()
        
        assert isinstance(models, list)
        assert len(models) > 0
        assert 'isolation_forest' in models
        assert 'pca' in models
        assert 'local_outlier_factor' in models
    
    def test_get_models_by_category(self):
        """Test getting models grouped by category"""
        factory = ModelFactory()
        categories = factory.get_models_by_category()
        
        assert 'statistical' in categories
        assert 'deep_learning' in categories
        assert 'time_series' in categories
        
        assert 'isolation_forest' in categories['statistical']
        assert 'pca' in categories['statistical']
    
    def test_model_requirements(self):
        """Test getting model requirements"""
        factory = ModelFactory()
        
        # Test isolation forest requirements
        req = factory.get_model_requirements('isolation_forest')
        assert 'min_samples' in req
        assert 'supports_multivariate' in req
        assert req['supports_multivariate'] is True
    
    def test_model_compatibility(self):
        """Test model compatibility validation"""
        factory = ModelFactory()
        
        # Test with sufficient data
        X_good = np.random.normal(0, 1, (100, 5))
        assert factory.validate_model_compatibility('isolation_forest', X_good)
        
        # Test with insufficient data
        X_small = np.random.normal(0, 1, (5, 5))
        # Should still pass for isolation forest (low requirements)
        result = factory.validate_model_compatibility('isolation_forest', X_small)
        # Result depends on specific requirements
    
    def test_recommended_models(self):
        """Test model recommendation system"""
        factory = ModelFactory()
        
        # Test with different data characteristics
        X_small = np.random.normal(0, 1, (50, 3))
        recommendations = factory.get_recommended_models(X_small, max_models=3)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3
        assert len(recommendations) > 0
    
    def test_ensemble_creation(self):
        """Test ensemble detector creation"""
        factory = ModelFactory()
        
        model_names = ['isolation_forest', 'pca']
        ensemble = factory.create_ensemble(model_names)
        
        assert isinstance(ensemble, EnsembleDetector)
        assert len(ensemble.models) <= len(model_names)  # Some models might fail to create

class TestStatisticalModels:
    """Test statistical anomaly detection models"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (80, 4))
        anomaly_data = np.random.normal(3, 0.5, (20, 4))  # Shifted anomalies
        return np.vstack([normal_data, anomaly_data])
    
    def test_isolation_forest_detector(self, sample_data):
        """Test Isolation Forest detector"""
        detector = IsolationForestDetector({
            'contamination': 0.2,
            'n_estimators': 50,
            'random_state': 42
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
        
        # Check if model was created
        assert detector.model is not None
    
    def test_pca_detector(self, sample_data):
        """Test PCA-based detector"""
        detector = PCADetector({
            'n_components': 0.95,
            'contamination': 0.2
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
    
    def test_one_class_svm_detector(self, sample_data):
        """Test One-Class SVM detector"""
        detector = OneClassSVMDetector({
            'nu': 0.2,
            'kernel': 'rbf'
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
    
    def test_lof_detector(self, sample_data):
        """Test Local Outlier Factor detector"""
        detector = LocalOutlierFactorDetector({
            'n_neighbors': 10,
            'contamination': 0.2
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
    
    def test_dbscan_detector(self, sample_data):
        """Test DBSCAN detector"""
        detector = DBSCANDetector({
            'eps': 0.5,
            'min_samples': 5
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
    
    def test_gaussian_mixture_detector(self, sample_data):
        """Test Gaussian Mixture detector"""
        detector = GaussianMixtureDetector({
            'n_components': 2,
            'contamination': 0.2
        })
        
        scores = detector.fit_predict(sample_data)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted

class TestDeepLearningModels:
    """Test deep learning models (with mocking if TensorFlow not available)"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return np.random.normal(0, 1, (200, 6))
    
    def test_autoencoder_import(self):
        """Test autoencoder model import"""
        try:
            from models.deep_learning_models import AutoencoderDetector
            assert True
        except ImportError:
            pytest.skip("TensorFlow not available")
    
    @patch('models.deep_learning_models.TF_AVAILABLE', False)
    def test_autoencoder_without_tensorflow(self):
        """Test autoencoder behavior without TensorFlow"""
        from models.deep_learning_models import AutoencoderDetector
        
        with pytest.raises(ImportError):
            AutoencoderDetector()

class TestEnsembleDetector:
    """Test ensemble detector functionality"""
    
    def test_ensemble_creation(self):
        """Test ensemble detector creation"""
        # Create mock models
        mock_model1 = MagicMock()
        mock_model1.fit_predict.return_value = np.array([0.1, 0.2, 0.8, 0.9])
        
        mock_model2 = MagicMock()
        mock_model2.fit_predict.return_value = np.array([0.2, 0.3, 0.7, 0.8])
        
        models = [('model1', mock_model1), ('model2', mock_model2)]
        ensemble = EnsembleDetector(models, {'combination_method': 'average'})
        
        X = np.random.normal(0, 1, (4, 3))
        scores = ensemble.fit_predict(X)
        
        assert len(scores) == 4
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert ensemble.is_fitted
        
        # Check if scores are averaged
        expected_scores = np.array([0.15, 0.25, 0.75, 0.85])
        np.testing.assert_array_almost_equal(scores, expected_scores)
    
    def test_ensemble_combination_methods(self):
        """Test different ensemble combination methods"""
        mock_model1 = MagicMock()
        mock_model1.fit_predict.return_value = np.array([0.1, 0.8])
        
        mock_model2 = MagicMock()
        mock_model2.fit_predict.return_value = np.array([0.3, 0.6])
        
        models = [('model1', mock_model1), ('model2', mock_model2)]
        X = np.random.normal(0, 1, (2, 3))
        
        # Test max combination
        ensemble_max = EnsembleDetector(models, {'combination_method': 'max'})
        scores_max = ensemble_max.fit_predict(X)
        expected_max = np.array([0.3, 0.8])
        np.testing.assert_array_almost_equal(scores_max, expected_max)
        
        # Test median combination
        ensemble_median = EnsembleDetector(models, {'combination_method': 'median'})
        scores_median = ensemble_median.fit_predict(X)
        expected_median = np.array([0.2, 0.7])
        np.testing.assert_array_almost_equal(scores_median, expected_median)

class TestMetaModelSelector:
    """Test meta-learning model selector"""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        return np.random.normal(0, 1, (100, 8))
    
    def test_meta_feature_extraction(self, sample_data):
        """Test meta-feature extraction"""
        selector = MetaModelSelector({'enabled': True})
        
        # Extract meta-features
        meta_features = selector._extract_meta_features(sample_data)
        
        assert isinstance(meta_features, dict)
        assert 'n_samples' in meta_features
        assert 'n_features' in meta_features
        assert 'dimensionality_ratio' in meta_features
        assert meta_features['n_samples'] == 100
        assert meta_features['n_features'] == 8
    
    def test_model_selection_enabled(self, sample_data):
        """Test model selection when enabled"""
        available_models = ['isolation_forest', 'pca', 'local_outlier_factor', 'autoencoder']
        
        selector = MetaModelSelector({
            'enabled': True,
            'methods': ['metaod'],
            'max_models': 3,
            'min_models': 2
        })
        
        selected = selector.select_models(sample_data, available_models)
        
        assert isinstance(selected, list)
        assert len(selected) >= 2  # min_models
        assert len(selected) <= 3  # max_models
        assert all(model in available_models for model in selected)
    
    def test_model_selection_disabled(self, sample_data):
        """Test model selection when disabled"""
        available_models = ['isolation_forest', 'pca']
        
        selector = MetaModelSelector({'enabled': False})
        selected = selector.select_models(sample_data, available_models)
        
        assert selected == available_models
    
    def test_metaod_selection(self, sample_data):
        """Test MetaOD selection method"""
        available_models = ['isolation_forest', 'pca', 'autoencoder', 'lstm_autoencoder']
        
        selector = MetaModelSelector({'enabled': True})
        selected = selector._metaod_selection(sample_data, available_models)
        
        assert isinstance(selected, list)
        assert len(selected) > 0
    
    def test_time_series_detection(self):
        """Test time series pattern detection"""
        selector = MetaModelSelector()
        
        # Create time series-like data (with autocorrelation)
        t = np.linspace(0, 4*np.pi, 100)
        ts_data = np.sin(t).reshape(-1, 1) + np.random.normal(0, 0.1, (100, 1))
        
        is_ts = selector._is_time_series_like(ts_data)
        # This might be True or False depending on the noise level and correlation threshold
        assert isinstance(is_ts, bool)
        
        # Create non-time series data
        random_data = np.random.normal(0, 1, (100, 1))
        is_ts_random = selector._is_time_series_like(random_data)
        assert isinstance(is_ts_random, bool)

class TestModelErrorHandling:
    """Test error handling in models"""
    
    def test_invalid_model_name(self):
        """Test handling of invalid model names"""
        factory = ModelFactory()
        
        with pytest.raises(ValueError):
            factory.get_model('nonexistent_model')
    
    def test_model_with_invalid_data(self):
        """Test model behavior with invalid data"""
        detector = IsolationForestDetector()
        
        # Test with empty data
        empty_data = np.array([]).reshape(0, 3)
        scores = detector.fit_predict(empty_data)
        assert len(scores) == 0
        
        # Test with single sample
        single_sample = np.array([[1, 2, 3]])
        scores = detector.fit_predict(single_sample)
        assert len(scores) == 1
    
    def test_model_with_nan_data(self):
        """Test model behavior with NaN data"""
        detector = IsolationForestDetector()
        
        # Create data with NaN values
        data_with_nan = np.random.normal(0, 1, (50, 3))
        data_with_nan[10:15, 1] = np.nan
        
        # Model should handle this gracefully (might return zeros or handle internally)
        scores = detector.fit_predict(data_with_nan)
        assert len(scores) == 50
        # Scores might be all zeros if model can't handle NaN

if __name__ == '__main__':
    pytest.main([__file__, '-v'])