# 🔥 Mauri-mTSB: Professional CLI Anomaly Detection Tool

**Author:** Mohamed lemine Ahmed Jidou  
**Platform:** Kali Linux (Python 3.10+)  
**Version:** 1.0.0

A production-ready, CLI-based anomaly detection tool designed specifically for cybersecurity professionals. Built on the concepts of the mTSBench benchmark, Mauri-mTSB supports multivariate time series analysis of system/network logs with advanced machine learning models and automatic model selection.

## 🚀 Features

### Core Capabilities
- **🔍 Multi-Model Detection**: 12+ anomaly detection algorithms including statistical, deep learning, and time series models
- **🤖 Automatic Model Selection**: MetaOD, FMMS, and Orthus-based intelligent model selection
- **📊 Comprehensive Evaluation**: AUC-ROC, F1-Score, VUS-PR, Event-based F1, and custom metrics
- **⚡ High Performance**: Multiprocessing support and optimized for large datasets
- **🎨 Rich Visualizations**: Interactive plots with matplotlib, seaborn, and plotly
- **📁 Flexible Data Support**: CSV, TSV, JSON, Parquet with auto-detection of timestamps

### Supported Models

#### Statistical Models
- **Isolation Forest**: Efficient outlier detection for large datasets
- **PCA**: Principal Component Analysis with reconstruction error
- **One-Class SVM**: Support Vector Machine for novelty detection
- **Local Outlier Factor (LOF)**: Density-based local outlier detection
- **DBSCAN**: Clustering-based anomaly detection
- **Gaussian Mixture**: Probabilistic anomaly scoring

#### Deep Learning Models
- **Autoencoder**: Neural network reconstruction-based detection
- **LSTM Autoencoder**: Recurrent neural networks for time series
- **Transformer**: Attention-based sequence modeling

#### Time Series Models
- **Prophet**: Facebook's time series forecasting with anomaly detection
- **ARIMA**: Classical time series analysis
- **Seasonal Decompose**: Trend and seasonality-based detection

## 🛠️ Installation

### Prerequisites
- **Operating System**: Kali Linux (recommended) or any Linux distribution
- **Python**: 3.10 or higher
- **Memory**: 4GB+ RAM recommended
- **Storage**: 2GB+ free space

### Quick Install
```bash
# Clone the repository
git clone https://github.com/your-repo/mauri-mtsb.git
cd mauri-mtsb

# Install dependencies
pip install -r requirements.txt

# Verify installation
python main.py --help
```

### Development Install
```bash
# Install in development mode
pip install -e .

# Install additional development dependencies
pip install pytest pytest-cov flake8 black

# Run tests
python -m pytest tests/
```

## 🎯 Quick Start

### 1. Initialize Configuration
```bash
python main.py init-config --output ./config
```

### 2. Basic Anomaly Detection
```bash
# Run with default settings
python main.py detect --input data/network_logs.csv --output ./results

# With specific models
python main.py detect -i data/system_logs.csv -m isolation_forest -m pca --visualize

# With automatic model selection
python main.py detect -i data/auth_logs.csv --auto-select --export-format json
```

### 3. Advanced Usage
```bash
# Custom configuration
python main.py detect \
    --input data/netflow.csv \
    --config config/custom_config.yaml \
    --output ./results \
    --models isolation_forest pca lstm_autoencoder \
    --visualize \
    --export-format both \
    --verbose
```

## 📋 Command Reference

### Main Commands

#### `detect` - Run Anomaly Detection
```bash
python main.py detect [OPTIONS]

Options:
  -i, --input PATH          Input CSV file (required)
  -c, --config PATH         Configuration YAML file
  -o, --output PATH         Output directory [default: ./results]
  -m, --models TEXT         Specific models to run (multiple allowed)
  --auto-select            Use automatic model selection
  --visualize              Generate visualization plots
  --export-format CHOICE   Export format: json|csv|both [default: both]
  -v, --verbose            Enable verbose output
```

#### `list-models` - Show Available Models
```bash
python main.py list-models
```

#### `init-config` - Generate Configuration
```bash
python main.py init-config [--output PATH]
```

### Model Selection Examples
```bash
# Statistical models only
python main.py detect -i data.csv -m isolation_forest -m pca -m local_outlier_factor

# Deep learning models
python main.py detect -i data.csv -m autoencoder -m lstm_autoencoder

# Time series models
python main.py detect -i data.csv -m prophet -m arima -m seasonal_decompose

# Automatic selection (recommended)
python main.py detect -i data.csv --auto-select
```

## 📊 Data Format Support

### Supported File Formats
- **CSV**: Comma-separated values (primary)
- **TSV**: Tab-separated values
- **JSON**: JavaScript Object Notation
- **Parquet**: Columnar storage format

### Data Types

#### NetFlow Logs
```csv
timestamp,src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets,duration
2023-01-01 00:00:00,192.168.1.1,10.0.0.1,12345,80,TCP,1024,10,5.2
2023-01-01 00:01:00,192.168.1.2,10.0.0.2,12346,443,TCP,2048,15,3.1
```

#### System Logs
```csv
timestamp,event_id,level,source,message,user,process
2023-01-01 00:00:00,4624,INFO,Security,Successful logon,admin,winlogon.exe
2023-01-01 00:01:00,4625,WARNING,Security,Failed logon,guest,winlogon.exe
```

#### Authentication Events
```csv
timestamp,user,action,src_ip,success,session_id
2023-01-01 00:00:00,admin,login,192.168.1.100,1,sess_001
2023-01-01 00:01:00,guest,login,192.168.1.101,0,sess_002
```

### Automatic Feature Detection
- **Timestamp Columns**: Auto-detected and parsed
- **IP Addresses**: Converted to numerical features
- **Categorical Data**: Automatically encoded
- **Missing Values**: Handled with configurable strategies

## ⚙️ Configuration

### Configuration File Structure
```yaml
# Data loading configuration
data:
  csv_params:
    parse_dates: true
    index_col: null
  auto_detect_timestamp: true

# Preprocessing settings
preprocessing:
  missing_strategy: 'mean'  # mean, median, drop, knn
  scaling_method: 'standard'  # standard, minmax, robust
  feature_engineering: true
  rolling_window: 10

# Model configurations
models:
  isolation_forest:
    contamination: 0.1
    n_estimators: 100
  
  autoencoder:
    encoding_dim: [64, 32, 16]
    epochs: 50
    batch_size: 32

# Automatic model selection
meta_selection:
  enabled: true
  methods: ['metaod', 'fmms', 'orthus']
  max_models: 5
  min_models: 2

# Evaluation metrics
evaluation:
  threshold: 0.5
  metrics:
    - 'auc_roc'
    - 'f1_score'
    - 'precision'
    - 'recall'
```

### Environment Variables
```bash
# Performance tuning
export MAURI_MTSB_N_JOBS=-1          # Use all CPU cores
export MAURI_MTSB_MEMORY_LIMIT=4GB   # Memory limit
export MAURI_MTSB_BATCH_SIZE=10000   # Batch processing size

# Logging
export MAURI_MTSB_LOG_LEVEL=INFO     # DEBUG, INFO, WARNING, ERROR
export MAURI_MTSB_LOG_FILE=mauri.log # Log file path
```

## 📈 Output and Results

### Result Files
```
results/
├── results.json              # Detailed results in JSON format
├── results_summary.csv       # Summary table in CSV format
├── anomaly_scores_comparison.png
├── time_series_anomalies.png
├── model_performance_comparison.png
├── feature_space_visualization.png
├── anomaly_distribution.png
├── interactive_time_series.html
├── interactive_3d_scatter.html
└── interactive_dashboard.html
```

### JSON Result Structure
```json
{
  "isolation_forest": {
    "anomaly_scores": [0.1, 0.2, 0.8, 0.9, ...],
    "metrics": {
      "auc_roc": 0.85,
      "f1_score": 0.78,
      "precision": 0.82,
      "recall": 0.75,
      "anomalies_detected": 23
    },
    "timestamps": ["2023-01-01T00:00:00", ...]
  }
}
```

### Evaluation Metrics

#### Supervised Metrics (when ground truth available)
- **AUC-ROC**: Area Under ROC Curve
- **AUC-PR**: Area Under Precision-Recall Curve
- **F1-Score**: Harmonic mean of precision and recall
- **VUS-PR**: Volume Under Surface for PR curve
- **Event-based F1**: Time-aware F1 score

#### Unsupervised Metrics
- **Isolation Score**: Quality of anomaly isolation
- **Consistency Score**: Temporal consistency of predictions
- **Score Variance**: Distribution of anomaly scores
- **Synthetic Metrics**: Statistical-based evaluation

## 🔧 Advanced Usage

### Custom Model Configuration
```python
# Custom model parameters
config = {
    'models': {
        'isolation_forest': {
            'contamination': 0.05,
            'n_estimators': 200,
            'max_samples': 'auto',
            'random_state': 42
        },
        'autoencoder': {
            'encoding_dim': [128, 64, 32, 16],
            'epochs': 100,
            'batch_size': 64,
            'validation_split': 0.2
        }
    }
}
```

### Batch Processing
```bash
# Process multiple files
for file in data/*.csv; do
    python main.py detect -i "$file" -o "results/$(basename "$file" .csv)" --auto-select
done

# Parallel processing
find data/ -name "*.csv" | xargs -P 4 -I {} python main.py detect -i {} --auto-select
```

### Integration with Security Tools
```bash
# Process Zeek/Bro logs
python main.py detect -i /opt/zeek/logs/conn.log --config config/zeek_config.yaml

# Process Suricata EVE logs
python main.py detect -i /var/log/suricata/eve.json --config config/suricata_config.yaml

# Process Windows Event logs (converted to CSV)
python main.py detect -i security_events.csv --config config/windows_config.yaml
```

## 🧪 Testing

### Run Test Suite
```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/test_models.py -v
python -m pytest tests/test_basic.py -v

# Run with coverage
python -m pytest tests/ --cov=. --cov-report=html
```

### Performance Testing
```bash
# Test with large dataset
python main.py detect -i large_dataset.csv --auto-select --verbose

# Memory profiling
python -m memory_profiler main.py detect -i data.csv
```

## 🚨 Security Considerations

### Data Privacy
- **Local Processing**: All data processing happens locally
- **No External Calls**: No data sent to external services
- **Secure Logging**: Sensitive information filtered from logs

### Recommended Security Practices
```bash
# Run with restricted permissions
sudo -u limited_user python main.py detect -i sensitive_data.csv

# Use encrypted storage for results
python main.py detect -i data.csv -o /encrypted/results/

# Clear temporary files
export TMPDIR=/secure/tmp
python main.py detect -i data.csv
```

## 🔍 Troubleshooting

### Common Issues

#### Memory Issues
```bash
# Reduce memory usage
export MAURI_MTSB_MEMORY_LIMIT=2GB
python main.py detect -i large_file.csv --config config/low_memory.yaml
```

#### Performance Issues
```bash
# Enable batch processing
python main.py detect -i huge_file.csv --config config/batch_config.yaml

# Use fewer models
python main.py detect -i data.csv -m isolation_forest -m pca
```

#### Missing Dependencies
```bash
# Install optional dependencies
pip install prophet tensorflow torch

# Check installation
python -c "import tensorflow; print('TensorFlow:', tensorflow.__version__)"
```

### Debug Mode
```bash
# Enable debug logging
python main.py detect -i data.csv --verbose

# Check system information
python -c "from utils.logger import log_system_info; log_system_info()"
```

## 🤝 Contributing

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/your-username/mauri-mtsb.git
cd mauri-mtsb

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install -e .

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

### Code Style
```bash
# Format code
python -m black .

# Lint code
python -m flake8 .

# Type checking
python -m mypy main.py
```

### Adding New Models
1. Create model class inheriting from `BaseAnomalyDetector`
2. Implement `fit_predict` method
3. Add to `ModelFactory`
4. Write tests
5. Update documentation

## 📚 References

- **mTSBench**: Multivariate Time Series Anomaly Detection Benchmark
- **MetaOD**: Meta-learning for Outlier Detection
- **FMMS**: Fast Model Selection Strategy
- **Orthus**: Diversity-based Model Selection

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **mTSBench** team for benchmark concepts
- **PyOD** library for anomaly detection algorithms
- **scikit-learn** for machine learning utilities
- **TensorFlow/PyTorch** for deep learning models
- **Kali Linux** community for security-focused tools

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/mauri-mtsb/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/mauri-mtsb/discussions)
- **Email**: mohamed.jidou@example.com

---

**🔥 Mauri-mTSB** - Professional anomaly detection for cybersecurity professionals on Kali Linux.