"""
Logging utilities for Mauri-mTSB
Provides colorized console output and file logging
"""

import logging
import colorama
from colorama import Fore, Style, Back
from rich.logging import RichHandler
from rich.console import Console
import sys
from pathlib import Path

# Initialize colorama for cross-platform colored output
colorama.init(autoreset=True)

console = Console()

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter for colored console output
    """
    
    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Back.WHITE + Style.BRIGHT,
    }
    
    def format(self, record):
        log_color = self.COLORS.get(record.levelname, '')
        record.levelname = f"{log_color}{record.levelname}{Style.RESET_ALL}"
        return super().format(record)

def setup_logger(verbose: bool = False, log_file: str = None):
    """
    Setup logging configuration for Mauri-mTSB
    
    Args:
        verbose: Enable verbose (DEBUG) logging
        log_file: Optional log file path
    """
    try:
        # Determine log level
        log_level = logging.DEBUG if verbose else logging.INFO
        
        # Clear existing handlers
        root_logger = logging.getLogger()
        root_logger.handlers.clear()
        
        # Console handler with rich formatting
        console_handler = RichHandler(
            console=console,
            show_time=True,
            show_path=verbose
        )
        console_handler.setLevel(log_level)
        
        # Console format
        console_format = "%(message)s"
        console_handler.setFormatter(logging.Formatter(console_format))
        
        # Add console handler
        root_logger.addHandler(console_handler)
        root_logger.setLevel(log_level)
        
        # File handler if specified
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_path)
            file_handler.setLevel(logging.DEBUG)  # Always DEBUG for files
            
            # File format (more detailed)
            file_format = "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
            file_handler.setFormatter(logging.Formatter(file_format))
            
            root_logger.addHandler(file_handler)
        
        # Set specific logger levels
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        logging.getLogger('tensorflow').setLevel(logging.ERROR)
        logging.getLogger('torch').setLevel(logging.WARNING)
        
        # Welcome message
        if verbose:
            console.print("[dim]🔧 Verbose logging enabled[/dim]")
        
    except Exception as e:
        print(f"Error setting up logger: {e}")
        # Fallback to basic logging
        logging.basicConfig(level=log_level)

class SecurityLogger:
    """
    Specialized logger for security-related events
    """
    
    def __init__(self, name: str = "mauri-mtsb-security"):
        self.logger = logging.getLogger(name)
        
        # Security events should always be logged
        self.logger.setLevel(logging.INFO)
    
    def log_anomaly_detected(self, model_name: str, timestamp: str, 
                           score: float, details: str = ""):
        """
        Log detected anomaly with security context
        """
        message = (
            f"🚨 ANOMALY DETECTED | Model: {model_name} | "
            f"Time: {timestamp} | Score: {score:.3f}"
        )
        
        if details:
            message += f" | Details: {details}"
        
        self.logger.warning(message)
    
    def log_model_performance(self, model_name: str, metrics: dict):
        """
        Log model performance metrics
        """
        message = f"📊 MODEL PERFORMANCE | {model_name} | "
        message += " | ".join([f"{k}: {v:.3f}" for k, v in metrics.items()])
        
        self.logger.info(message)
    
    def log_data_processed(self, file_path: str, rows: int, features: int):
        """
        Log data processing information
        """
        message = f"📁 DATA PROCESSED | File: {file_path} | Rows: {rows} | Features: {features}"
        
        self.logger.info(message)
    
    def log_configuration_loaded(self, config_path: str):
        """
        Log configuration loading
        """
        message = f"⚙️ CONFIG LOADED | Path: {config_path}"
        
        self.logger.info(message)
    
    def log_security_event(self, event_type: str, details: str, severity: str = "INFO"):
        """
        Log general security events
        """
        emoji_map = {
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌",
            "CRITICAL": "🔥"
        }
        
        emoji = emoji_map.get(severity.upper(), "ℹ️")
        message = f"{emoji} SECURITY EVENT | Type: {event_type} | Details: {details}"
        
        log_method = getattr(self.logger, severity.lower(), self.logger.info)
        log_method(message)

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with proper configuration
    
    Args:
        name: Logger name
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)

def log_system_info():
    """
    Log system information for debugging
    """
    import platform
    import psutil
    import sys
    
    logger = get_logger("system-info")
    
    try:
        # System information
        logger.info(f"🖥️  System: {platform.system()} {platform.release()}")
        logger.info(f"🐍 Python: {sys.version.split()[0]}")
        logger.info(f"💾 Memory: {psutil.virtual_memory().total // (1024**3)} GB")
        logger.info(f"🔧 CPU Cores: {psutil.cpu_count()}")
        
        # Check for GPU availability
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"🚀 GPU Available: {len(gpus)} device(s)")
            else:
                logger.info("🔋 GPU: Not available (using CPU)")
        except ImportError:
            logger.info("🔋 TensorFlow not available")
        
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"🚀 PyTorch CUDA: Available ({torch.cuda.device_count()} device(s))")
            else:
                logger.info("🔋 PyTorch CUDA: Not available")
        except ImportError:
            logger.info("🔋 PyTorch not available")
            
    except Exception as e:
        logger.error(f"Error logging system info: {e}")

# Export main functions
__all__ = [
    'setup_logger',
    'get_logger', 
    'SecurityLogger',
    'log_system_info',
    'ColoredFormatter'
]