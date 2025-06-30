"""
Basic tests for Mauri-mTSB functionality
"""

import pytest
import numpy as np
import pandas as pd
import tempfile
import os
from pathlib import Path

# Import modules to test
from utils.data_loader import DataLoader
from utils.preprocessor import TimeSeriesPreprocessor
from utils.config_manager import ConfigManager
from models.model_factory import ModelFactory
from models.statistical_models import IsolationForestDetector, PCADetector

class TestDataLoader:
    """Test data loading functionality"""
    
    def test_csv_loading(self):
        """Test CSV file loading"""
        # Create sample CSV data
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=100, freq='H'),
            'feature1': np.random.normal(0, 1, 100),
            'feature2': np.random.normal(0, 1, 100),
            'feature3': np.random.normal(0, 1, 100)
        }
        df = pd.DataFrame(data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Test loading
            loader = DataLoader()
            loaded_df = loader.load_csv(temp_file)
            
            assert len(loaded_df) == 100
            assert 'feature1' in loaded_df.columns
            assert 'feature2' in loaded_df.columns
            assert 'feature3' in loaded_df.columns
            
        finally:
            os.unlink(temp_file)
    
    def test_netflow_loading(self):
        """Test NetFlow CSV loading"""
        # Create sample NetFlow data
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=50, freq='5min'),
            'src_ip': ['192.168.1.1', '192.168.1.2'] * 25,
            'dst_ip': ['10.0.0.1', '10.0.0.2'] * 25,
            'src_port': np.random.randint(1024, 65535, 50),
            'dst_port': np.random.randint(80, 8080, 50),
            'bytes': np.random.randint(100, 10000, 50),
            'packets': np.random.randint(1, 100, 50)
        }
        df = pd.DataFrame(data)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            loader = DataLoader()
            loaded_df = loader.load_netflow_csv(temp_file)
            
            assert len(loaded_df) == 50
            assert 'src_ip_numeric' in loaded_df.columns
            assert 'dst_ip_numeric' in loaded_df.columns
            
        finally:
            os.unlink(temp_file)

class TestPreprocessor:
    """Test data preprocessing functionality"""
    
    def test_basic_preprocessing(self):
        """Test basic preprocessing pipeline"""
        # Create sample data
        np.random.seed(42)
        data = np.random.normal(0, 1, (100, 5))
        df = pd.DataFrame(data, columns=[f'feature_{i}' for i in range(5)])
        df['timestamp'] = pd.date_range('2023-01-01', periods=100, freq='H')
        
        preprocessor = TimeSeriesPreprocessor()
        X_processed, timestamps = preprocessor.fit_transform(df)
        
        assert X_processed.shape[0] == 100
        assert X_processed.shape[1] >= 5  # Should have at least original features
        assert timestamps is not None
        assert len(timestamps) == 100
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        # Create data with missing values
        data = np.random.normal(0, 1, (50, 3))
        data[10:15, 1] = np.nan  # Add missing values
        df = pd.DataFrame(data, columns=['f1', 'f2', 'f3'])
        
        config = {'missing_strategy': 'mean'}
        preprocessor = TimeSeriesPreprocessor(config)
        X_processed, _ = preprocessor.fit_transform(df)
        
        assert not np.any(np.isnan(X_processed))
        assert X_processed.shape[0] == 50

class TestConfigManager:
    """Test configuration management"""
    
    def test_default_config(self):
        """Test default configuration loading"""
        config_manager = ConfigManager()
        config = config_manager.get_config()
        
        assert 'data' in config
        assert 'preprocessing' in config
        assert 'models' in config
        assert 'evaluation' in config
    
    def test_config_creation(self):
        """Test configuration file creation"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / 'test_config.yaml'
            
            config_manager = ConfigManager()
            config_manager.create_default_config(str(config_path))
            
            assert config_path.exists()
            
            # Test loading the created config
            new_config_manager = ConfigManager(str(config_path))
            config = new_config_manager.get_config()
            
            assert 'data' in config

class TestModelFactory:
    """Test model factory functionality"""
    
    def test_available_models(self):
        """Test getting available models"""
        factory = ModelFactory()
        models = factory.get_available_models()
        
        assert len(models) > 0
        assert 'isolation_forest' in models
        assert 'pca' in models
    
    def test_model_creation(self):
        """Test model creation"""
        factory = ModelFactory()
        
        # Test creating isolation forest
        model = factory.get_model('isolation_forest')
        assert model is not None
        assert hasattr(model, 'fit_predict')
    
    def test_model_compatibility(self):
        """Test model compatibility checking"""
        factory = ModelFactory()
        
        # Create sample data
        X = np.random.normal(0, 1, (100, 5))
        
        # Test compatibility
        is_compatible = factory.validate_model_compatibility('isolation_forest', X)
        assert is_compatible
        
        # Test with insufficient data
        X_small = np.random.normal(0, 1, (5, 5))
        is_compatible = factory.validate_model_compatibility('isolation_forest', X_small)
        # Should still be compatible as isolation forest has low minimum requirements

class TestStatisticalModels:
    """Test statistical anomaly detection models"""
    
    def test_isolation_forest(self):
        """Test Isolation Forest detector"""
        # Create sample data with anomalies
        np.random.seed(42)
        normal_data = np.random.normal(0, 1, (90, 3))
        anomaly_data = np.random.normal(5, 1, (10, 3))  # Shifted anomalies
        X = np.vstack([normal_data, anomaly_data])
        
        detector = IsolationForestDetector({'contamination': 0.1})
        scores = detector.fit_predict(X)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted
        
        # Check if anomalies have higher scores
        normal_scores = scores[:90]
        anomaly_scores = scores[90:]
        assert np.mean(anomaly_scores) > np.mean(normal_scores)
    
    def test_pca_detector(self):
        """Test PCA-based detector"""
        # Create sample data
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 5))
        
        detector = PCADetector({'n_components': 0.95})
        scores = detector.fit_predict(X)
        
        assert len(scores) == 100
        assert np.all(scores >= 0) and np.all(scores <= 1)
        assert detector.is_fitted

class TestIntegration:
    """Integration tests"""
    
    def test_full_pipeline(self):
        """Test complete anomaly detection pipeline"""
        # Create sample data
        np.random.seed(42)
        data = {
            'timestamp': pd.date_range('2023-01-01', periods=200, freq='H'),
            'cpu_usage': np.random.normal(50, 10, 200),
            'memory_usage': np.random.normal(60, 15, 200),
            'network_traffic': np.random.normal(1000, 200, 200)
        }
        
        # Add some anomalies
        data['cpu_usage'][50:55] = 95  # High CPU usage anomaly
        data['memory_usage'][100:105] = 95  # High memory usage anomaly
        
        df = pd.DataFrame(data)
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f.name, index=False)
            temp_file = f.name
        
        try:
            # Load data
            loader = DataLoader()
            loaded_df = loader.load_csv(temp_file)
            
            # Preprocess
            preprocessor = TimeSeriesPreprocessor()
            X_processed, timestamps = preprocessor.fit_transform(loaded_df)
            
            # Run anomaly detection
            factory = ModelFactory()
            model = factory.get_model('isolation_forest')
            scores = model.fit_predict(X_processed)
            
            # Verify results
            assert len(scores) == 200
            assert np.all(scores >= 0) and np.all(scores <= 1)
            
            # Check if anomalies are detected
            anomaly_indices = np.where(scores > 0.7)[0]
            assert len(anomaly_indices) > 0
            
        finally:
            os.unlink(temp_file)

def test_imports():
    """Test that all modules can be imported"""
    try:
        from main import cli
        from utils.data_loader import DataLoader
        from utils.preprocessor import TimeSeriesPreprocessor
        from utils.config_manager import ConfigManager
        from models.model_factory import ModelFactory
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")

if __name__ == '__main__':
    pytest.main([__file__])