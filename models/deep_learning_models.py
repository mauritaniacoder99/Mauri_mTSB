"""
Deep learning anomaly detection models
Includes Autoencoder, LSTM Autoencoder, and Transformer-based detectors
"""

import numpy as np
from typing import Dict, Any, Optional, List
import logging
import warnings

# Suppress TensorFlow warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
warnings.filterwarnings('ignore')

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    keras = None
    layers = None
    Model = None

from .model_factory import BaseAnomalyDetector

logger = logging.getLogger(__name__)

class AutoencoderDetector(BaseAnomalyDetector):
    """
    Autoencoder-based anomaly detector using reconstruction error
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.encoding_dim = config.get('encoding_dim', [64, 32, 16])
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.contamination = config.get('contamination', 0.1)
        self.validation_split = config.get('validation_split', 0.2)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Autoencoder detector")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit autoencoder and predict anomaly scores
        """
        try:
            # Normalize input data
            X_normalized = self._normalize_data(X)
            
            # Build autoencoder
            self.model = self._build_autoencoder(X_normalized.shape[1])
            
            # Train autoencoder
            self._train_model(X_normalized)
            
            # Calculate reconstruction errors
            X_reconstructed = self.model.predict(X_normalized, verbose=0)
            reconstruction_errors = np.mean((X_normalized - X_reconstructed) ** 2, axis=1)
            
            # Normalize scores
            normalized_scores = self._normalize_scores(reconstruction_errors)
            
            self.is_fitted = True
            
            # Log anomaly statistics
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"Autoencoder: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Autoencoder: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _build_autoencoder(self, input_dim: int) -> Model:
        """
        Build autoencoder architecture
        """
        # Input layer
        input_layer = keras.Input(shape=(input_dim,))
        
        # Encoder
        encoded = input_layer
        for dim in self.encoding_dim:
            encoded = layers.Dense(dim, activation='relu')(encoded)
            encoded = layers.Dropout(0.2)(encoded)
        
        # Decoder
        decoded = encoded
        for dim in reversed(self.encoding_dim[:-1]):
            decoded = layers.Dense(dim, activation='relu')(decoded)
            decoded = layers.Dropout(0.2)(decoded)
        
        # Output layer
        decoded = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _train_model(self, X: np.ndarray):
        """
        Train the autoencoder model
        """
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train model
        self.model.fit(
            X, X,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=self.validation_split,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize input data to [0, 1] range
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize reconstruction errors to [0, 1] range
        """
        scores = np.array(scores)
        
        # Use robust normalization
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized

class LSTMAutoencoderDetector(BaseAnomalyDetector):
    """
    LSTM Autoencoder for time series anomaly detection
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 10)
        self.encoding_dim = config.get('encoding_dim', 32)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.contamination = config.get('contamination', 0.1)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTM Autoencoder detector")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit LSTM autoencoder and predict anomaly scores
        """
        try:
            # Normalize data
            X_normalized = self._normalize_data(X)
            
            # Create sequences
            X_sequences = self._create_sequences(X_normalized)
            
            if X_sequences.shape[0] == 0:
                logger.warning("Not enough data to create sequences")
                return np.zeros(X.shape[0])
            
            # Build LSTM autoencoder
            self.model = self._build_lstm_autoencoder(
                X_sequences.shape[1], X_sequences.shape[2]
            )
            
            # Train model
            self._train_model(X_sequences)
            
            # Calculate reconstruction errors
            X_reconstructed = self.model.predict(X_sequences, verbose=0)
            reconstruction_errors = np.mean((X_sequences - X_reconstructed) ** 2, axis=(1, 2))
            
            # Pad scores to match original length
            full_scores = self._pad_scores(reconstruction_errors, X.shape[0])
            
            # Normalize scores
            normalized_scores = self._normalize_scores(full_scores)
            
            self.is_fitted = True
            
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"LSTM Autoencoder: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in LSTM Autoencoder: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sequences for LSTM input
        """
        sequences = []
        
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def _build_lstm_autoencoder(self, sequence_length: int, n_features: int) -> Model:
        """
        Build LSTM autoencoder architecture
        """
        # Encoder
        encoder_inputs = keras.Input(shape=(sequence_length, n_features))
        encoder_lstm = layers.LSTM(self.encoding_dim, return_state=True)
        encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
        encoder_states = [state_h, state_c]
        
        # Decoder
        decoder_inputs = layers.RepeatVector(sequence_length)(encoder_outputs)
        decoder_lstm = layers.LSTM(self.encoding_dim, return_sequences=True)
        decoder_outputs = decoder_lstm(decoder_inputs, initial_state=encoder_states)
        decoder_dense = layers.Dense(n_features, activation='linear')
        decoder_outputs = decoder_dense(decoder_outputs)
        
        # Create model
        autoencoder = Model(encoder_inputs, decoder_outputs)
        autoencoder.compile(optimizer='adam', loss='mse')
        
        return autoencoder
    
    def _train_model(self, X_sequences: np.ndarray):
        """
        Train the LSTM autoencoder
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_sequences, X_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def _pad_scores(self, scores: np.ndarray, original_length: int) -> np.ndarray:
        """
        Pad reconstruction error scores to match original data length
        """
        if len(scores) == original_length:
            return scores
        
        # Pad beginning with first score
        padded_scores = np.zeros(original_length)
        
        # Fill the beginning with the first score
        padded_scores[:self.sequence_length-1] = scores[0] if len(scores) > 0 else 0
        
        # Fill the rest with actual scores
        padded_scores[self.sequence_length-1:] = scores
        
        return padded_scores
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize input data
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize reconstruction errors
        """
        scores = np.array(scores)
        
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized

class TransformerDetector(BaseAnomalyDetector):
    """
    Transformer-based anomaly detector
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.sequence_length = config.get('sequence_length', 10)
        self.d_model = config.get('d_model', 64)
        self.nhead = config.get('nhead', 8)
        self.epochs = config.get('epochs', 50)
        self.batch_size = config.get('batch_size', 32)
        self.contamination = config.get('contamination', 0.1)
        
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for Transformer detector")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """
        Fit transformer and predict anomaly scores
        """
        try:
            # Normalize data
            X_normalized = self._normalize_data(X)
            
            # Create sequences
            X_sequences = self._create_sequences(X_normalized)
            
            if X_sequences.shape[0] == 0:
                logger.warning("Not enough data to create sequences")
                return np.zeros(X.shape[0])
            
            # Build transformer
            self.model = self._build_transformer(
                X_sequences.shape[1], X_sequences.shape[2]
            )
            
            # Train model
            self._train_model(X_sequences)
            
            # Calculate reconstruction errors
            X_reconstructed = self.model.predict(X_sequences, verbose=0)
            reconstruction_errors = np.mean((X_sequences - X_reconstructed) ** 2, axis=(1, 2))
            
            # Pad scores to match original length
            full_scores = self._pad_scores(reconstruction_errors, X.shape[0])
            
            # Normalize scores
            normalized_scores = self._normalize_scores(full_scores)
            
            self.is_fitted = True
            
            threshold = np.percentile(normalized_scores, (1 - self.contamination) * 100)
            n_anomalies = np.sum(normalized_scores > threshold)
            logger.debug(f"Transformer: {n_anomalies} anomalies detected")
            
            return normalized_scores
            
        except Exception as e:
            logger.error(f"Error in Transformer: {str(e)}")
            return np.zeros(X.shape[0])
    
    def _create_sequences(self, X: np.ndarray) -> np.ndarray:
        """
        Create sequences for transformer input
        """
        sequences = []
        
        for i in range(len(X) - self.sequence_length + 1):
            sequences.append(X[i:i + self.sequence_length])
        
        return np.array(sequences)
    
    def _build_transformer(self, sequence_length: int, n_features: int) -> Model:
        """
        Build transformer architecture for anomaly detection
        """
        # Input
        inputs = keras.Input(shape=(sequence_length, n_features))
        
        # Positional encoding
        x = self._add_positional_encoding(inputs)
        
        # Multi-head attention
        attention_output = layers.MultiHeadAttention(
            num_heads=self.nhead,
            key_dim=self.d_model // self.nhead
        )(x, x)
        
        # Add & Norm
        x = layers.Add()([x, attention_output])
        x = layers.LayerNormalization()(x)
        
        # Feed forward
        ff_output = layers.Dense(self.d_model * 2, activation='relu')(x)
        ff_output = layers.Dense(self.d_model)(ff_output)
        
        # Add & Norm
        x = layers.Add()([x, ff_output])
        x = layers.LayerNormalization()(x)
        
        # Output projection
        outputs = layers.Dense(n_features, activation='linear')(x)
        
        # Create model
        model = Model(inputs, outputs)
        model.compile(optimizer='adam', loss='mse')
        
        return model
    
    def _add_positional_encoding(self, inputs):
        """
        Add positional encoding to inputs
        """
        sequence_length = tf.shape(inputs)[1]
        d_model = tf.shape(inputs)[2]
        
        # Create positional encoding
        position = tf.range(sequence_length, dtype=tf.float32)[:, tf.newaxis]
        div_term = tf.exp(tf.range(0, d_model, 2, dtype=tf.float32) * 
                         -(tf.math.log(10000.0) / tf.cast(d_model, tf.float32)))
        
        pos_encoding = tf.zeros((sequence_length, d_model))
        pos_encoding = tf.tensor_scatter_nd_update(
            pos_encoding,
            tf.stack([tf.range(sequence_length), tf.range(0, d_model, 2)], axis=1),
            tf.sin(position * div_term)
        )
        
        if d_model > 1:
            pos_encoding = tf.tensor_scatter_nd_update(
                pos_encoding,
                tf.stack([tf.range(sequence_length), tf.range(1, d_model, 2)], axis=1),
                tf.cos(position * div_term)
            )
        
        return inputs + pos_encoding
    
    def _train_model(self, X_sequences: np.ndarray):
        """
        Train the transformer model
        """
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        self.model.fit(
            X_sequences, X_sequences,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
    
    def _pad_scores(self, scores: np.ndarray, original_length: int) -> np.ndarray:
        """
        Pad reconstruction error scores to match original data length
        """
        if len(scores) == original_length:
            return scores
        
        padded_scores = np.zeros(original_length)
        padded_scores[:self.sequence_length-1] = scores[0] if len(scores) > 0 else 0
        padded_scores[self.sequence_length-1:] = scores
        
        return padded_scores
    
    def _normalize_data(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize input data
        """
        from sklearn.preprocessing import MinMaxScaler
        
        scaler = MinMaxScaler()
        return scaler.fit_transform(X)
    
    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """
        Normalize reconstruction errors
        """
        scores = np.array(scores)
        
        q95 = np.percentile(scores, 95)
        q5 = np.percentile(scores, 5)
        
        if q95 - q5 == 0:
            return np.zeros_like(scores)
        
        normalized = np.clip((scores - q5) / (q95 - q5), 0, 1)
        return normalized