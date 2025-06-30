"""
Visualization utilities for anomaly detection results
Supports matplotlib, seaborn, and plotly for comprehensive plotting
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, Any, Optional, List
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AnomalyVisualizer:
    """
    Comprehensive visualization for anomaly detection results
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        # Set style
        plt.style.use(self.config.get('style', 'seaborn-v0_8-whitegrid'))
        
        # Color palette for anomalies
        self.colors = {
            'normal': '#1f77b4',
            'anomaly': '#d62728',
            'threshold': '#ff7f0e'
        }
    
    def plot_results(self, X: np.ndarray, results: Dict[str, Any], 
                    timestamps: Optional[np.ndarray] = None, 
                    output_dir: Path = None):
        """
        Generate comprehensive visualizations for anomaly detection results
        
        Args:
            X: Input data
            results: Dictionary containing model results
            timestamps: Optional timestamps
            output_dir: Directory to save plots
        """
        try:
            if output_dir:
                output_dir = Path(output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
            
            logger.info("Generating visualizations...")
            
            # 1. Anomaly scores comparison
            self._plot_anomaly_scores_comparison(results, timestamps, output_dir)
            
            # 2. Time series with anomalies
            if timestamps is not None:
                self._plot_time_series_anomalies(X, results, timestamps, output_dir)
            
            # 3. Model performance comparison
            self._plot_model_performance(results, output_dir)
            
            # 4. Feature space visualization (2D projection)
            self._plot_feature_space(X, results, output_dir)
            
            # 5. Anomaly distribution
            self._plot_anomaly_distribution(results, output_dir)
            
            # 6. Interactive plots (Plotly)
            if self.config.get('generate_interactive', True):
                self._generate_interactive_plots(X, results, timestamps, output_dir)
            
            logger.info("Visualization generation completed")
            
        except Exception as e:
            logger.error(f"Error generating visualizations: {str(e)}")
            raise
    
    def _plot_anomaly_scores_comparison(self, results: Dict[str, Any], 
                                      timestamps: Optional[np.ndarray] = None,
                                      output_dir: Optional[Path] = None):
        """
        Plot anomaly scores comparison across models
        """
        try:
            n_models = len(results)
            if n_models == 0:
                return
            
            fig, axes = plt.subplots(min(n_models, 4), 1, figsize=(12, 3 * min(n_models, 4)))
            if n_models == 1:
                axes = [axes]
            
            for i, (model_name, result) in enumerate(results.items()):
                if i >= 4:  # Limit to 4 subplots
                    break
                
                ax = axes[i] if n_models > 1 else axes[0]
                scores = result['anomaly_scores']
                
                if timestamps is not None:
                    x_axis = timestamps
                    ax.set_xlabel('Time')
                else:
                    x_axis = range(len(scores))
                    ax.set_xlabel('Sample Index')
                
                ax.plot(x_axis, scores, label=f'{model_name} scores', alpha=0.7)
                ax.axhline(y=0.5, color=self.colors['threshold'], linestyle='--', 
                          label='Threshold (0.5)')
                
                # Highlight anomalies
                anomalies = np.array(scores) > 0.5
                if np.any(anomalies):
                    ax.scatter(np.array(x_axis)[anomalies], np.array(scores)[anomalies], 
                             color=self.colors['anomaly'], s=30, alpha=0.8, 
                             label='Anomalies')
                
                ax.set_title(f'{model_name} - Anomaly Scores')
                ax.set_ylabel('Anomaly Score')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'anomaly_scores_comparison.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting anomaly scores: {str(e)}")
    
    def _plot_time_series_anomalies(self, X: np.ndarray, results: Dict[str, Any],
                                   timestamps: np.ndarray, output_dir: Optional[Path] = None):
        """
        Plot time series data with detected anomalies
        """
        try:
            # Select first few features for visualization
            n_features = min(X.shape[1], 4)
            
            fig, axes = plt.subplots(n_features, 1, figsize=(15, 3 * n_features))
            if n_features == 1:
                axes = [axes]
            
            # Get anomalies from the first model for visualization
            first_model = list(results.keys())[0]
            anomaly_scores = results[first_model]['anomaly_scores']
            anomalies = np.array(anomaly_scores) > 0.5
            
            for i in range(n_features):
                ax = axes[i]
                
                # Plot time series
                ax.plot(timestamps, X[:, i], color=self.colors['normal'], 
                       alpha=0.7, label=f'Feature {i+1}')
                
                # Highlight anomalies
                if np.any(anomalies):
                    ax.scatter(timestamps[anomalies], X[anomalies, i], 
                             color=self.colors['anomaly'], s=50, alpha=0.8,
                             label='Anomalies')
                
                ax.set_title(f'Feature {i+1} - Time Series with Anomalies')
                ax.set_ylabel('Value')
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            axes[-1].set_xlabel('Time')
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'time_series_anomalies.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting time series anomalies: {str(e)}")
    
    def _plot_model_performance(self, results: Dict[str, Any], 
                              output_dir: Optional[Path] = None):
        """
        Plot model performance comparison
        """
        try:
            # Extract metrics
            models = []
            metrics_data = {
                'AUC-ROC': [],
                'F1-Score': [],
                'Precision': [],
                'Recall': []
            }
            
            for model_name, result in results.items():
                models.append(model_name)
                metrics = result.get('metrics', {})
                
                metrics_data['AUC-ROC'].append(metrics.get('auc_roc', 0.0))
                metrics_data['F1-Score'].append(metrics.get('f1_score', 0.0))
                metrics_data['Precision'].append(metrics.get('precision', 0.0))
                metrics_data['Recall'].append(metrics.get('recall', 0.0))
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            axes = axes.flatten()
            
            for i, (metric_name, values) in enumerate(metrics_data.items()):
                ax = axes[i]
                bars = ax.bar(models, values, alpha=0.7)
                
                # Color bars based on performance
                for j, (bar, value) in enumerate(zip(bars, values)):
                    if value >= 0.8:
                        bar.set_color('green')
                    elif value >= 0.6:
                        bar.set_color('orange')
                    else:
                        bar.set_color('red')
                
                ax.set_title(f'{metric_name} Comparison')
                ax.set_ylabel(metric_name)
                ax.set_ylim(0, 1)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'model_performance_comparison.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting model performance: {str(e)}")
    
    def _plot_feature_space(self, X: np.ndarray, results: Dict[str, Any],
                           output_dir: Optional[Path] = None):
        """
        Plot 2D projection of feature space with anomalies
        """
        try:
            from sklearn.decomposition import PCA
            from sklearn.manifold import TSNE
            
            # Apply PCA for 2D visualization
            if X.shape[1] > 2:
                pca = PCA(n_components=2)
                X_2d = pca.fit_transform(X)
            else:
                X_2d = X[:, :2]
            
            # Get anomalies from the first model
            first_model = list(results.keys())[0]
            anomaly_scores = results[first_model]['anomaly_scores']
            anomalies = np.array(anomaly_scores) > 0.5
            
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            
            # PCA plot
            ax1 = axes[0]
            normal_mask = ~anomalies
            
            ax1.scatter(X_2d[normal_mask, 0], X_2d[normal_mask, 1], 
                       c=self.colors['normal'], alpha=0.6, s=30, label='Normal')
            
            if np.any(anomalies):
                ax1.scatter(X_2d[anomalies, 0], X_2d[anomalies, 1], 
                           c=self.colors['anomaly'], alpha=0.8, s=50, label='Anomalies')
            
            ax1.set_title('Feature Space (PCA Projection)')
            ax1.set_xlabel('PC1')
            ax1.set_ylabel('PC2')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Anomaly score heatmap
            ax2 = axes[1]
            
            # Create a grid for interpolation
            xx, yy = np.meshgrid(np.linspace(X_2d[:, 0].min(), X_2d[:, 0].max(), 50),
                                np.linspace(X_2d[:, 1].min(), X_2d[:, 1].max(), 50))
            
            # Interpolate anomaly scores
            from scipy.interpolate import griddata
            
            scores_interp = griddata(X_2d, anomaly_scores, (xx, yy), method='cubic', fill_value=0)
            
            im = ax2.contourf(xx, yy, scores_interp, levels=20, cmap='YlOrRd', alpha=0.7)
            ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=anomaly_scores, cmap='YlOrRd', s=30)
            
            plt.colorbar(im, ax=ax2, label='Anomaly Score')
            ax2.set_title('Anomaly Score Heatmap')
            ax2.set_xlabel('PC1')
            ax2.set_ylabel('PC2')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'feature_space_visualization.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting feature space: {str(e)}")
    
    def _plot_anomaly_distribution(self, results: Dict[str, Any],
                                 output_dir: Optional[Path] = None):
        """
        Plot distribution of anomaly scores and detected anomalies
        """
        try:
            n_models = len(results)
            fig, axes = plt.subplots(2, min(n_models, 3), figsize=(5 * min(n_models, 3), 8))
            
            if n_models == 1:
                axes = axes.reshape(-1, 1)
            elif min(n_models, 3) == 1:
                axes = axes.reshape(-1, 1)
            
            for i, (model_name, result) in enumerate(results.items()):
                if i >= 3:  # Limit to 3 models
                    break
                
                scores = result['anomaly_scores']
                anomalies = np.array(scores) > 0.5
                
                col_idx = i if n_models > 1 else 0
                
                # Distribution of anomaly scores
                ax1 = axes[0, col_idx] if axes.ndim > 1 else axes[0]
                ax1.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                ax1.axvline(x=0.5, color=self.colors['threshold'], linestyle='--', 
                           label='Threshold')
                ax1.set_title(f'{model_name}\nAnomaly Score Distribution')
                ax1.set_xlabel('Anomaly Score')
                ax1.set_ylabel('Frequency')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Anomaly detection summary
                ax2 = axes[1, col_idx] if axes.ndim > 1 else axes[1]
                
                normal_count = np.sum(~anomalies)
                anomaly_count = np.sum(anomalies)
                
                categories = ['Normal', 'Anomaly']
                counts = [normal_count, anomaly_count]
                colors = [self.colors['normal'], self.colors['anomaly']]
                
                bars = ax2.bar(categories, counts, color=colors, alpha=0.7)
                ax2.set_title(f'{model_name}\nDetection Summary')
                ax2.set_ylabel('Count')
                
                # Add percentage labels
                total = normal_count + anomaly_count
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    percentage = (count / total) * 100
                    ax2.text(bar.get_x() + bar.get_width()/2., height + total * 0.01,
                           f'{count}\n({percentage:.1f}%)', ha='center', va='bottom')
            
            plt.tight_layout()
            
            if output_dir:
                plt.savefig(output_dir / 'anomaly_distribution.png', 
                           dpi=300, bbox_inches='tight')
            plt.show()
            
        except Exception as e:
            logger.error(f"Error plotting anomaly distribution: {str(e)}")
    
    def _generate_interactive_plots(self, X: np.ndarray, results: Dict[str, Any],
                                  timestamps: Optional[np.ndarray] = None,
                                  output_dir: Optional[Path] = None):
        """
        Generate interactive plots using Plotly
        """
        try:
            # Interactive time series plot
            if timestamps is not None:
                self._create_interactive_time_series(X, results, timestamps, output_dir)
            
            # Interactive 3D scatter plot
            self._create_interactive_3d_scatter(X, results, output_dir)
            
            # Interactive model comparison dashboard
            self._create_interactive_dashboard(results, output_dir)
            
        except Exception as e:
            logger.error(f"Error generating interactive plots: {str(e)}")
    
    def _create_interactive_time_series(self, X: np.ndarray, results: Dict[str, Any],
                                       timestamps: np.ndarray, output_dir: Optional[Path] = None):
        """
        Create interactive time series plot with Plotly
        """
        try:
            fig = make_subplots(
                rows=min(X.shape[1], 3), cols=1,
                subplot_titles=[f'Feature {i+1}' for i in range(min(X.shape[1], 3))],
                shared_xaxes=True
            )
            
            # Get anomalies from first model
            first_model = list(results.keys())[0]
            anomaly_scores = results[first_model]['anomaly_scores']
            anomalies = np.array(anomaly_scores) > 0.5
            
            for i in range(min(X.shape[1], 3)):
                # Add time series
                fig.add_trace(
                    go.Scatter(
                        x=timestamps,
                        y=X[:, i],
                        mode='lines',
                        name=f'Feature {i+1}',
                        line=dict(color='blue', width=1),
                        opacity=0.7
                    ),
                    row=i+1, col=1
                )
                
                # Add anomalies
                if np.any(anomalies):
                    fig.add_trace(
                        go.Scatter(
                            x=timestamps[anomalies],
                            y=X[anomalies, i],
                            mode='markers',
                            name=f'Anomalies (Feature {i+1})',
                            marker=dict(color='red', size=8, symbol='x'),
                            showlegend=(i == 0)
                        ),
                        row=i+1, col=1
                    )
            
            fig.update_layout(
                title='Interactive Time Series with Anomalies',
                height=300 * min(X.shape[1], 3),
                showlegend=True
            )
            
            if output_dir:
                fig.write_html(output_dir / 'interactive_time_series.html')
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive time series: {str(e)}")
    
    def _create_interactive_3d_scatter(self, X: np.ndarray, results: Dict[str, Any],
                                     output_dir: Optional[Path] = None):
        """
        Create interactive 3D scatter plot
        """
        try:
            if X.shape[1] < 3:
                return
            
            # Get anomalies from first model
            first_model = list(results.keys())[0]
            anomaly_scores = results[first_model]['anomaly_scores']
            anomalies = np.array(anomaly_scores) > 0.5
            
            # Create 3D scatter plot
            colors = ['red' if anom else 'blue' for anom in anomalies]
            
            fig = go.Figure(data=[go.Scatter3d(
                x=X[:, 0],
                y=X[:, 1],
                z=X[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors,
                    opacity=0.7
                ),
                text=[f'Score: {score:.3f}' for score in anomaly_scores],
                hovertemplate='<b>Point %{pointNumber}</b><br>' +
                             'X: %{x:.3f}<br>' +
                             'Y: %{y:.3f}<br>' +
                             'Z: %{z:.3f}<br>' +
                             '%{text}<extra></extra>'
            )])
            
            fig.update_layout(
                title='3D Feature Space with Anomalies',
                scene=dict(
                    xaxis_title='Feature 1',
                    yaxis_title='Feature 2',
                    zaxis_title='Feature 3'
                )
            )
            
            if output_dir:
                fig.write_html(output_dir / 'interactive_3d_scatter.html')
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating 3D scatter plot: {str(e)}")
    
    def _create_interactive_dashboard(self, results: Dict[str, Any],
                                    output_dir: Optional[Path] = None):
        """
        Create interactive model comparison dashboard
        """
        try:
            # Prepare data
            models = list(results.keys())
            metrics_data = {
                'Model': [],
                'AUC-ROC': [],
                'F1-Score': [],
                'Precision': [],
                'Recall': [],
                'Anomalies': []
            }
            
            for model_name, result in results.items():
                metrics = result.get('metrics', {})
                anomaly_count = sum(1 for score in result['anomaly_scores'] if score > 0.5)
                
                metrics_data['Model'].append(model_name)
                metrics_data['AUC-ROC'].append(metrics.get('auc_roc', 0.0))
                metrics_data['F1-Score'].append(metrics.get('f1_score', 0.0))
                metrics_data['Precision'].append(metrics.get('precision', 0.0))
                metrics_data['Recall'].append(metrics.get('recall', 0.0))
                metrics_data['Anomalies'].append(anomaly_count)
            
            # Create subplots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Performance Metrics', 'Anomalies Detected', 
                              'Precision vs Recall', 'Score Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                      [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Performance metrics bar chart
            for metric in ['AUC-ROC', 'F1-Score', 'Precision', 'Recall']:
                fig.add_trace(
                    go.Bar(
                        x=metrics_data['Model'],
                        y=metrics_data[metric],
                        name=metric,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Anomalies detected
            fig.add_trace(
                go.Bar(
                    x=metrics_data['Model'],
                    y=metrics_data['Anomalies'],
                    name='Anomalies',
                    marker_color='red',
                    showlegend=False
                ),
                row=1, col=2
            )
            
            # Precision vs Recall scatter
            fig.add_trace(
                go.Scatter(
                    x=metrics_data['Precision'],
                    y=metrics_data['Recall'],
                    mode='markers+text',
                    text=metrics_data['Model'],
                    textposition='top center',
                    marker=dict(size=10),
                    name='Models',
                    showlegend=False
                ),
                row=2, col=1
            )
            
            # Score distribution (first model as example)
            if results:
                first_model = list(results.keys())[0]
                scores = results[first_model]['anomaly_scores']
                
                fig.add_trace(
                    go.Histogram(
                        x=scores,
                        nbinsx=30,
                        name='Score Distribution',
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            fig.update_layout(
                title='Anomaly Detection Dashboard',
                height=800,
                showlegend=True
            )
            
            if output_dir:
                fig.write_html(output_dir / 'interactive_dashboard.html')
            
            fig.show()
            
        except Exception as e:
            logger.error(f"Error creating interactive dashboard: {str(e)}")