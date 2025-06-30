"""
Configuration management for Mauri-mTSB
Handles YAML/JSON configuration files with validation
"""

import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ConfigManager:
    """
    Manages configuration for the anomaly detection tool
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path
        self.config = self._load_config() if config_path else self._get_default_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from file
        """
        try:
            config_path = Path(self.config_path)
            
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}. Using defaults.")
                return self._get_default_config()
            
            if config_path.suffix.lower() == '.yaml' or config_path.suffix.lower() == '.yml':
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Validate and merge with defaults
            default_config = self._get_default_config()
            merged_config = self._merge_configs(default_config, config)
            
            logger.info(f"Configuration loaded from: {config_path}")
            return merged_config
            
        except Exception as e:
            logger.error(f"Error loading config: {str(e)}. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """
        Get default configuration
        """
        return {
            'data': {
                'csv_params': {
                    'parse_dates': True,
                    'index_col': None,
                    'low_memory': False
                },
                'auto_detect_timestamp': True,
                'supported_formats': ['.csv', '.tsv', '.json', '.parquet']
            },
            'preprocessing': {
                'missing_strategy': 'mean',  # 'mean', 'median', 'drop', 'knn'
                'scaling_method': 'standard',  # 'standard', 'minmax', 'robust'
                'feature_engineering': True,
                'add_statistical_features': True,
                'add_interaction_features': False,
                'rolling_window': 10,
                'apply_pca': False,
                'pca_components': 0.95
            },
            'models': {
                'isolation_forest': {
                    'contamination': 0.1,
                    'n_estimators': 100,
                    'random_state': 42
                },
                'pca': {
                    'n_components': 0.95,
                    'contamination': 0.1
                },
                'autoencoder': {
                    'encoding_dim': [64, 32, 16],
                    'epochs': 50,
                    'batch_size': 32,
                    'contamination': 0.1
                },
                'lstm_autoencoder': {
                    'sequence_length': 10,
                    'encoding_dim': 32,
                    'epochs': 50,
                    'batch_size': 32,
                    'contamination': 0.1
                },
                'transformer': {
                    'sequence_length': 10,
                    'd_model': 64,
                    'nhead': 8,
                    'epochs': 50,
                    'batch_size': 32,
                    'contamination': 0.1
                },
                'one_class_svm': {
                    'nu': 0.1,
                    'kernel': 'rbf',
                    'gamma': 'scale'
                },
                'local_outlier_factor': {
                    'n_neighbors': 20,
                    'contamination': 0.1
                },
                'dbscan': {
                    'eps': 0.5,
                    'min_samples': 5
                },
                'gaussian_mixture': {
                    'n_components': 2,
                    'contamination': 0.1
                },
                'prophet': {
                    'changepoint_prior_scale': 0.05,
                    'seasonality_prior_scale': 10.0,
                    'contamination': 0.1
                },
                'arima': {
                    'order': (1, 1, 1),
                    'contamination': 0.1
                },
                'seasonal_decompose': {
                    'model': 'additive',
                    'period': None,
                    'contamination': 0.1
                }
            },
            'meta_selection': {
                'enabled': True,
                'methods': ['metaod', 'fmms', 'orthus'],
                'selection_criteria': ['auc_roc', 'f1_score', 'isolation_score'],
                'max_models': 5,
                'min_models': 2
            },
            'evaluation': {
                'threshold': 0.5,
                'metrics': [
                    'auc_roc', 'auc_pr', 'f1_score', 'precision', 'recall',
                    'vus_pr', 'isolation_score', 'consistency'
                ],
                'cross_validation': False,
                'cv_folds': 5
            },
            'visualization': {
                'style': 'seaborn-v0_8-whitegrid',
                'figsize': [12, 8],
                'dpi': 300,
                'generate_interactive': True,
                'color_palette': {
                    'normal': '#1f77b4',
                    'anomaly': '#d62728',
                    'threshold': '#ff7f0e'
                }
            },
            'output': {
                'save_models': False,
                'save_preprocessor': False,
                'export_detailed_results': True,
                'compression': None  # 'gzip', 'bz2', 'xz'
            },
            'performance': {
                'n_jobs': -1,  # Use all available cores
                'memory_limit': '4GB',
                'batch_processing': False,
                'batch_size': 10000
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file_logging': False,
                'log_file': 'mauri_mtsb.log'
            }
        }
    
    def _merge_configs(self, default: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
        """
        Recursively merge user config with default config
        """
        merged = default.copy()
        
        for key, value in user.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration
        """
        return self.config.copy()
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """
        Get a specific configuration section
        """
        return self.config.get(section, {})
    
    def update_config(self, updates: Dict[str, Any]):
        """
        Update configuration with new values
        """
        self.config = self._merge_configs(self.config, updates)
    
    def save_config(self, output_path: str):
        """
        Save current configuration to file
        """
        try:
            output_path = Path(output_path)
            
            if output_path.suffix.lower() in ['.yaml', '.yml']:
                with open(output_path, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif output_path.suffix.lower() == '.json':
                with open(output_path, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported output format: {output_path.suffix}")
            
            logger.info(f"Configuration saved to: {output_path}")
            
        except Exception as e:
            logger.error(f"Error saving config: {str(e)}")
            raise
    
    def create_default_config(self, output_path: str):
        """
        Create a default configuration file with comments
        """
        try:
            output_path = Path(output_path)
            
            config_with_comments = self._add_config_comments()
            
            with open(output_path, 'w') as f:
                f.write(config_with_comments)
            
            logger.info(f"Default configuration created at: {output_path}")
            
        except Exception as e:
            logger.error(f"Error creating default config: {str(e)}")
            raise
    
    def _add_config_comments(self) -> str:
        """
        Create configuration file with explanatory comments
        """
        config_template = """# Mauri-mTSB Configuration File
# Professional CLI-based Anomaly Detection Tool
# Author: Mohamed lemine Ahmed Jidou

# Data loading configuration
data:
  csv_params:
    parse_dates: true
    index_col: null
    low_memory: false
  auto_detect_timestamp: true
  supported_formats: ['.csv', '.tsv', '.json', '.parquet']

# Data preprocessing configuration
preprocessing:
  missing_strategy: 'mean'  # Options: 'mean', 'median', 'drop', 'knn'
  scaling_method: 'standard'  # Options: 'standard', 'minmax', 'robust'
  feature_engineering: true
  add_statistical_features: true
  add_interaction_features: false
  rolling_window: 10
  apply_pca: false
  pca_components: 0.95

# Model configurations
models:
  # Statistical models
  isolation_forest:
    contamination: 0.1
    n_estimators: 100
    random_state: 42
  
  pca:
    n_components: 0.95
    contamination: 0.1
  
  one_class_svm:
    nu: 0.1
    kernel: 'rbf'
    gamma: 'scale'
  
  local_outlier_factor:
    n_neighbors: 20
    contamination: 0.1
  
  # Deep learning models
  autoencoder:
    encoding_dim: [64, 32, 16]
    epochs: 50
    batch_size: 32
    contamination: 0.1
  
  lstm_autoencoder:
    sequence_length: 10
    encoding_dim: 32
    epochs: 50
    batch_size: 32
    contamination: 0.1
  
  transformer:
    sequence_length: 10
    d_model: 64
    nhead: 8
    epochs: 50
    batch_size: 32
    contamination: 0.1
  
  # Clustering models
  dbscan:
    eps: 0.5
    min_samples: 5
  
  gaussian_mixture:
    n_components: 2
    contamination: 0.1
  
  # Time series models
  prophet:
    changepoint_prior_scale: 0.05
    seasonality_prior_scale: 10.0
    contamination: 0.1
  
  arima:
    order: [1, 1, 1]
    contamination: 0.1
  
  seasonal_decompose:
    model: 'additive'
    period: null  # Auto-detect
    contamination: 0.1

# Automatic model selection
meta_selection:
  enabled: true
  methods: ['metaod', 'fmms', 'orthus']
  selection_criteria: ['auc_roc', 'f1_score', 'isolation_score']
  max_models: 5
  min_models: 2

# Evaluation configuration
evaluation:
  threshold: 0.5
  metrics:
    - 'auc_roc'
    - 'auc_pr'
    - 'f1_score'
    - 'precision'
    - 'recall'
    - 'vus_pr'
    - 'isolation_score'
    - 'consistency'
  cross_validation: false
  cv_folds: 5

# Visualization settings
visualization:
  style: 'seaborn-v0_8-whitegrid'
  figsize: [12, 8]
  dpi: 300
  generate_interactive: true
  color_palette:
    normal: '#1f77b4'
    anomaly: '#d62728'
    threshold: '#ff7f0e'

# Output configuration
output:
  save_models: false
  save_preprocessor: false
  export_detailed_results: true
  compression: null  # Options: 'gzip', 'bz2', 'xz'

# Performance settings
performance:
  n_jobs: -1  # Use all available cores
  memory_limit: '4GB'
  batch_processing: false
  batch_size: 10000

# Logging configuration
logging:
  level: 'INFO'
  format: '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
  file_logging: false
  log_file: 'mauri_mtsb.log'
"""
        return config_template
    
    def validate_config(self) -> bool:
        """
        Validate the current configuration
        """
        try:
            # Check required sections
            required_sections = ['data', 'preprocessing', 'models', 'evaluation']
            
            for section in required_sections:
                if section not in self.config:
                    logger.error(f"Missing required configuration section: {section}")
                    return False
            
            # Validate specific parameters
            if not isinstance(self.config['evaluation'].get('threshold'), (int, float)):
                logger.error("Evaluation threshold must be a number")
                return False
            
            if not 0 <= self.config['evaluation']['threshold'] <= 1:
                logger.error("Evaluation threshold must be between 0 and 1")
                return False
            
            logger.info("Configuration validation passed")
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {str(e)}")
            return False