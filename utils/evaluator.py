"""
Model evaluation utilities for anomaly detection
Includes various metrics: AUC-ROC, F1, VUS-PR, Event-based F1, etc.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, roc_curve
)
import logging

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive evaluation for anomaly detection models
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.threshold = self.config.get('threshold', 0.5)
    
    def evaluate(self, anomaly_scores: np.ndarray, 
                timestamps: Optional[np.ndarray] = None,
                true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Comprehensive evaluation of anomaly detection results
        
        Args:
            anomaly_scores: Anomaly scores from model
            timestamps: Optional timestamps
            true_labels: Optional true anomaly labels (if available)
            
        Returns:
            Dictionary of evaluation metrics
        """
        try:
            metrics = {}
            
            # Convert scores to binary predictions
            predictions = (anomaly_scores > self.threshold).astype(int)
            
            # Basic statistics
            metrics.update(self._calculate_basic_stats(anomaly_scores, predictions))
            
            # If true labels are available, calculate supervised metrics
            if true_labels is not None:
                metrics.update(self._calculate_supervised_metrics(
                    anomaly_scores, predictions, true_labels))
            else:
                # Use unsupervised evaluation methods
                metrics.update(self._calculate_unsupervised_metrics(
                    anomaly_scores, predictions))
            
            # Time-based metrics if timestamps available
            if timestamps is not None:
                metrics.update(self._calculate_temporal_metrics(
                    anomaly_scores, predictions, timestamps))
            
            logger.info(f"Evaluation completed. Metrics calculated: {len(metrics)}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in evaluation: {str(e)}")
            return {}
    
    def _calculate_basic_stats(self, scores: np.ndarray, 
                             predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate basic statistics about the anomaly detection
        """
        return {
            'total_samples': len(scores),
            'anomalies_detected': int(np.sum(predictions)),
            'anomaly_rate': float(np.mean(predictions)),
            'mean_anomaly_score': float(np.mean(scores)),
            'std_anomaly_score': float(np.std(scores)),
            'min_anomaly_score': float(np.min(scores)),
            'max_anomaly_score': float(np.max(scores)),
            'median_anomaly_score': float(np.median(scores)),
            'threshold_used': self.threshold
        }
    
    def _calculate_supervised_metrics(self, scores: np.ndarray, 
                                    predictions: np.ndarray,
                                    true_labels: np.ndarray) -> Dict[str, float]:
        """
        Calculate supervised evaluation metrics when true labels are available
        """
        try:
            metrics = {}
            
            # ROC-AUC
            try:
                metrics['auc_roc'] = roc_auc_score(true_labels, scores)
            except ValueError:
                metrics['auc_roc'] = 0.0
            
            # Precision-Recall AUC
            try:
                metrics['auc_pr'] = average_precision_score(true_labels, scores)
            except ValueError:
                metrics['auc_pr'] = 0.0
            
            # Classification metrics
            metrics['accuracy'] = accuracy_score(true_labels, predictions)
            metrics['precision'] = precision_score(true_labels, predictions, zero_division=0)
            metrics['recall'] = recall_score(true_labels, predictions, zero_division=0)
            metrics['f1_score'] = f1_score(true_labels, predictions, zero_division=0)
            
            # Confusion matrix metrics
            tn, fp, fn, tp = confusion_matrix(true_labels, predictions).ravel()
            metrics['true_positives'] = int(tp)
            metrics['false_positives'] = int(fp)
            metrics['true_negatives'] = int(tn)
            metrics['false_negatives'] = int(fn)
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            # Volume Under Surface for PR curve (VUS-PR)
            metrics['vus_pr'] = self._calculate_vus_pr(true_labels, scores)
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating supervised metrics: {str(e)}")
            return {}
    
    def _calculate_unsupervised_metrics(self, scores: np.ndarray, 
                                      predictions: np.ndarray) -> Dict[str, float]:
        """
        Calculate unsupervised evaluation metrics
        """
        try:
            metrics = {}
            
            # Score-based metrics
            metrics['score_variance'] = float(np.var(scores))
            metrics['score_entropy'] = self._calculate_entropy(scores)
            
            # Isolation metrics
            metrics['isolation_score'] = self._calculate_isolation_score(scores)
            
            # Consistency metrics
            metrics['prediction_consistency'] = self._calculate_consistency(predictions)
            
            # Distribution-based metrics
            metrics['score_skewness'] = float(self._calculate_skewness(scores))
            metrics['score_kurtosis'] = float(self._calculate_kurtosis(scores))
            
            # Synthetic supervised metrics (using statistical properties)
            metrics.update(self._synthetic_supervised_metrics(scores, predictions))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating unsupervised metrics: {str(e)}")
            return {}
    
    def _calculate_temporal_metrics(self, scores: np.ndarray, 
                                  predictions: np.ndarray,
                                  timestamps: np.ndarray) -> Dict[str, float]:
        """
        Calculate time-based evaluation metrics
        """
        try:
            metrics = {}
            
            # Convert timestamps to pandas datetime if needed
            if not isinstance(timestamps[0], pd.Timestamp):
                timestamps = pd.to_datetime(timestamps)
            
            # Temporal clustering of anomalies
            anomaly_indices = np.where(predictions == 1)[0]
            
            if len(anomaly_indices) > 0:
                # Time gaps between anomalies
                time_gaps = np.diff(timestamps[anomaly_indices])
                if len(time_gaps) > 0:
                    metrics['avg_time_between_anomalies'] = float(
                        np.mean([gap.total_seconds() for gap in time_gaps]))
                    metrics['min_time_between_anomalies'] = float(
                        np.min([gap.total_seconds() for gap in time_gaps]))
                    metrics['max_time_between_anomalies'] = float(
                        np.max([gap.total_seconds() for gap in time_gaps]))
                
                # Anomaly duration (consecutive anomalies)
                anomaly_durations = self._calculate_anomaly_durations(
                    predictions, timestamps)
                if anomaly_durations:
                    metrics['avg_anomaly_duration'] = float(np.mean(anomaly_durations))
                    metrics['max_anomaly_duration'] = float(np.max(anomaly_durations))
                    metrics['min_anomaly_duration'] = float(np.min(anomaly_durations))
                    metrics['num_anomaly_episodes'] = len(anomaly_durations)
                
                # Time-based distribution
                metrics['anomaly_temporal_spread'] = self._calculate_temporal_spread(
                    anomaly_indices, len(predictions))
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating temporal metrics: {str(e)}")
            return {}
    
    def _calculate_vus_pr(self, true_labels: np.ndarray, scores: np.ndarray) -> float:
        """
        Calculate Volume Under Surface for Precision-Recall curve
        """
        try:
            precision, recall, thresholds = precision_recall_curve(true_labels, scores)
            
            # Calculate area under PR curve
            auc_pr = np.trapz(precision, recall)
            
            # Normalize to get VUS-PR
            vus_pr = auc_pr / np.sum(true_labels) if np.sum(true_labels) > 0 else 0
            
            return float(vus_pr)
            
        except Exception:
            return 0.0
    
    def _calculate_entropy(self, scores: np.ndarray) -> float:
        """
        Calculate entropy of anomaly scores
        """
        try:
            # Discretize scores for entropy calculation
            bins = np.linspace(scores.min(), scores.max(), 20)
            hist, _ = np.histogram(scores, bins=bins)
            
            # Calculate probability distribution
            probs = hist / np.sum(hist)
            probs = probs[probs > 0]  # Remove zero probabilities
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log2(probs))
            return float(entropy)
            
        except Exception:
            return 0.0
    
    def _calculate_isolation_score(self, scores: np.ndarray) -> float:
        """
        Calculate how well anomalies are isolated from normal points
        """
        try:
            # Use quantile-based approach
            q75 = np.percentile(scores, 75)
            q25 = np.percentile(scores, 25)
            iqr = q75 - q25
            
            # Calculate isolation as ratio of high scores to IQR
            high_scores = scores[scores > q75 + 1.5 * iqr]
            isolation = len(high_scores) / len(scores) if len(scores) > 0 else 0
            
            return float(isolation)
            
        except Exception:
            return 0.0
    
    def _calculate_consistency(self, predictions: np.ndarray) -> float:
        """
        Calculate consistency of predictions (less frequent changes indicate better consistency)
        """
        try:
            if len(predictions) <= 1:
                return 1.0
            
            # Count transitions (changes in prediction)
            transitions = np.sum(np.diff(predictions) != 0)
            
            # Consistency = 1 - (transitions / possible_transitions)
            consistency = 1 - (transitions / (len(predictions) - 1))
            
            return float(consistency)
            
        except Exception:
            return 0.0
    
    def _calculate_skewness(self, data: np.ndarray) -> float:
        """
        Calculate skewness of data
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                return 0.0
            
            skewness = np.mean(((data - mean) / std) ** 3)
            return float(skewness)
            
        except Exception:
            return 0.0
    
    def _calculate_kurtosis(self, data: np.ndarray) -> float:
        """
        Calculate kurtosis of data
        """
        try:
            mean = np.mean(data)
            std = np.std(data)
            
            if std == 0:
                return 0.0
            
            kurtosis = np.mean(((data - mean) / std) ** 4) - 3
            return float(kurtosis)
            
        except Exception:
            return 0.0
    
    def _synthetic_supervised_metrics(self, scores: np.ndarray, 
                                    predictions: np.ndarray) -> Dict[str, float]:
        """
        Create synthetic supervised metrics using statistical properties
        """
        try:
            metrics = {}
            
            # Create synthetic "ground truth" based on statistical outliers
            z_scores = np.abs((scores - np.mean(scores)) / np.std(scores))
            synthetic_labels = (z_scores > 2.0).astype(int)  # 2-sigma rule
            
            if np.sum(synthetic_labels) > 0 and np.sum(synthetic_labels) < len(synthetic_labels):
                # Calculate metrics against synthetic labels
                metrics['synthetic_precision'] = precision_score(
                    synthetic_labels, predictions, zero_division=0)
                metrics['synthetic_recall'] = recall_score(
                    synthetic_labels, predictions, zero_division=0)
                metrics['synthetic_f1'] = f1_score(
                    synthetic_labels, predictions, zero_division=0)
                
                try:
                    metrics['synthetic_auc'] = roc_auc_score(synthetic_labels, scores)
                except ValueError:
                    metrics['synthetic_auc'] = 0.0
            else:
                # Fallback metrics
                metrics['synthetic_precision'] = 0.0
                metrics['synthetic_recall'] = 0.0
                metrics['synthetic_f1'] = 0.0
                metrics['synthetic_auc'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error calculating synthetic supervised metrics: {str(e)}")
            return {}
    
    def _calculate_anomaly_durations(self, predictions: np.ndarray, 
                                   timestamps: np.ndarray) -> List[float]:
        """
        Calculate durations of anomaly episodes
        """
        durations = []
        
        try:
            in_anomaly = False
            anomaly_start = None
            
            for i, (pred, ts) in enumerate(zip(predictions, timestamps)):
                if pred == 1 and not in_anomaly:
                    # Start of anomaly episode
                    in_anomaly = True
                    anomaly_start = ts
                elif pred == 0 and in_anomaly:
                    # End of anomaly episode
                    in_anomaly = False
                    if anomaly_start is not None:
                        duration = (ts - anomaly_start).total_seconds()
                        durations.append(duration)
            
            # Handle case where anomaly continues to the end
            if in_anomaly and anomaly_start is not None:
                duration = (timestamps[-1] - anomaly_start).total_seconds()
                durations.append(duration)
            
            return durations
            
        except Exception:
            return []
    
    def _calculate_temporal_spread(self, anomaly_indices: np.ndarray, 
                                 total_length: int) -> float:
        """
        Calculate how spread out anomalies are in time
        """
        try:
            if len(anomaly_indices) <= 1:
                return 0.0
            
            # Calculate spread as ratio of time span to total time
            time_span = anomaly_indices[-1] - anomaly_indices[0]
            spread = time_span / (total_length - 1) if total_length > 1 else 0
            
            return float(spread)
            
        except Exception:
            return 0.0
    
    def get_evaluation_summary(self, metrics: Dict[str, float]) -> str:
        """
        Generate a human-readable summary of evaluation results
        """
        try:
            summary_lines = []
            
            # Basic stats
            summary_lines.append("=== Anomaly Detection Evaluation Summary ===")
            summary_lines.append(f"Total Samples: {metrics.get('total_samples', 0)}")
            summary_lines.append(f"Anomalies Detected: {metrics.get('anomalies_detected', 0)}")
            summary_lines.append(f"Anomaly Rate: {metrics.get('anomaly_rate', 0):.2%}")
            summary_lines.append("")
            
            # Performance metrics
            if 'auc_roc' in metrics:
                summary_lines.append("=== Supervised Metrics ===")
                summary_lines.append(f"AUC-ROC: {metrics['auc_roc']:.3f}")
                summary_lines.append(f"AUC-PR: {metrics.get('auc_pr', 0):.3f}")
                summary_lines.append(f"F1-Score: {metrics.get('f1_score', 0):.3f}")
                summary_lines.append(f"Precision: {metrics.get('precision', 0):.3f}")
                summary_lines.append(f"Recall: {metrics.get('recall', 0):.3f}")
                summary_lines.append("")
            
            # Unsupervised metrics
            if 'synthetic_f1' in metrics:
                summary_lines.append("=== Unsupervised Metrics ===")
                summary_lines.append(f"Synthetic F1: {metrics['synthetic_f1']:.3f}")
                summary_lines.append(f"Isolation Score: {metrics.get('isolation_score', 0):.3f}")
                summary_lines.append(f"Consistency: {metrics.get('prediction_consistency', 0):.3f}")
                summary_lines.append("")
            
            # Temporal metrics
            if 'num_anomaly_episodes' in metrics:
                summary_lines.append("=== Temporal Analysis ===")
                summary_lines.append(f"Anomaly Episodes: {metrics['num_anomaly_episodes']}")
                if 'avg_anomaly_duration' in metrics:
                    summary_lines.append(f"Avg Duration: {metrics['avg_anomaly_duration']:.1f}s")
                summary_lines.append("")
            
            return "\n".join(summary_lines)
            
        except Exception as e:
            logger.error(f"Error generating evaluation summary: {str(e)}")
            return "Error generating summary"