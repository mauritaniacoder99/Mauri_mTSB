"""
Data loading utilities for multivariate time series data
Supports CSV files from various cybersecurity data sources
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class DataLoader:
    """
    Handles loading of multivariate time series data from various sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.supported_formats = ['.csv', '.tsv', '.json', '.parquet']
    
    def load_csv(self, file_path: str, **kwargs) -> pd.DataFrame:
        """
        Load CSV file with cybersecurity time series data
        
        Args:
            file_path: Path to CSV file
            **kwargs: Additional pandas read_csv parameters
            
        Returns:
            DataFrame with loaded data
        """
        try:
            # Default CSV loading parameters
            csv_params = {
                'parse_dates': True,
                'index_col': None,
                'low_memory': False
            }
            
            # Update with config parameters
            csv_params.update(self.config.get('csv_params', {}))
            csv_params.update(kwargs)
            
            logger.info(f"Loading CSV file: {file_path}")
            df = pd.read_csv(file_path, **csv_params)
            
            # Basic validation
            if df.empty:
                raise ValueError("Loaded DataFrame is empty")
            
            logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")
            
            # Auto-detect timestamp column
            timestamp_col = self._detect_timestamp_column(df)
            if timestamp_col:
                logger.info(f"Detected timestamp column: {timestamp_col}")
                df = self._process_timestamps(df, timestamp_col)
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV file {file_path}: {str(e)}")
            raise
    
    def load_netflow_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load NetFlow CSV data with specific preprocessing
        
        Common NetFlow fields: timestamp, src_ip, dst_ip, src_port, dst_port,
        protocol, bytes, packets, duration, flags
        """
        try:
            df = self.load_csv(file_path)
            
            # NetFlow specific preprocessing
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Convert IP addresses to numerical features if present
            for col in ['src_ip', 'dst_ip']:
                if col in df.columns:
                    df[f'{col}_numeric'] = df[col].apply(self._ip_to_numeric)
            
            # Ensure numeric columns are properly typed
            numeric_cols = ['bytes', 'packets', 'duration', 'src_port', 'dst_port']
            for col in numeric_cols:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            logger.info("NetFlow data preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error loading NetFlow CSV: {str(e)}")
            raise
    
    def load_system_logs_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load system logs CSV data with specific preprocessing
        
        Common fields: timestamp, event_id, level, source, message, user, process
        """
        try:
            df = self.load_csv(file_path)
            
            # System logs specific preprocessing
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Encode categorical features
            categorical_cols = ['level', 'source', 'user', 'process']
            for col in categorical_cols:
                if col in df.columns:
                    df[f'{col}_encoded'] = pd.Categorical(df[col]).codes
            
            # Extract features from message field if present
            if 'message' in df.columns:
                df['message_length'] = df['message'].str.len()
                df['has_error'] = df['message'].str.contains('error|fail|exception', case=False, na=False).astype(int)
            
            logger.info("System logs data preprocessing completed")
            return df
            
        except Exception as e:
            logger.error(f"Error loading system logs CSV: {str(e)}")
            raise
    
    def _detect_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Auto-detect timestamp column in DataFrame
        """
        # Common timestamp column names
        timestamp_candidates = [
            'timestamp', 'time', 'datetime', 'date_time', 'ts',
            'created_at', 'event_time', 'log_time', 'occurrence_time'
        ]
        
        for col in timestamp_candidates:
            if col in df.columns:
                return col
        
        # Check for datetime-like columns
        for col in df.columns:
            if df[col].dtype == 'object':
                try:
                    # Try to parse a sample as datetime
                    sample = df[col].dropna().iloc[0]
                    pd.to_datetime(sample)
                    return col
                except (ValueError, TypeError):
                    continue
        
        return None
    
    def _process_timestamps(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """
        Process timestamp column for time series analysis
        """
        try:
            # Convert to datetime
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Sort by timestamp
            df = df.sort_values(timestamp_col)
            
            # Create time-based features
            df['hour'] = df[timestamp_col].dt.hour
            df['day_of_week'] = df[timestamp_col].dt.dayofweek
            df['is_weekend'] = (df[timestamp_col].dt.dayofweek >= 5).astype(int)
            
            return df
            
        except Exception as e:
            logger.warning(f"Error processing timestamps: {str(e)}")
            return df
    
    def _ip_to_numeric(self, ip_str: str) -> int:
        """
        Convert IP address string to numeric value
        """
        try:
            if pd.isna(ip_str):
                return 0
            
            parts = str(ip_str).split('.')
            if len(parts) != 4:
                return 0
            
            return sum(int(part) * (256 ** (3 - i)) for i, part in enumerate(parts))
            
        except (ValueError, AttributeError):
            return 0
    
    def get_sample_data_info(self) -> Dict[str, Any]:
        """
        Return information about supported data formats and sample structures
        """
        return {
            'supported_formats': self.supported_formats,
            'netflow_sample': {
                'required_columns': ['timestamp', 'src_ip', 'dst_ip', 'bytes', 'packets'],
                'optional_columns': ['src_port', 'dst_port', 'protocol', 'duration', 'flags'],
                'example': 'timestamp,src_ip,dst_ip,src_port,dst_port,protocol,bytes,packets'
            },
            'system_logs_sample': {
                'required_columns': ['timestamp', 'event_id', 'level'],
                'optional_columns': ['source', 'message', 'user', 'process'],
                'example': 'timestamp,event_id,level,source,message,user'
            },
            'auth_logs_sample': {
                'required_columns': ['timestamp', 'user', 'action'],
                'optional_columns': ['src_ip', 'success', 'session_id'],
                'example': 'timestamp,user,action,src_ip,success,session_id'
            }
        }