#!/usr/bin/env python3
"""
Mauri-mTSB: Professional CLI-based Anomaly Detection Tool
Author: Mohamed lemine Ahmed Jidou
Platform: Kali Linux (Python 3.10+)

A production-ready anomaly detection tool for cybersecurity professionals
based on mTSBench benchmark concepts.
"""

import os
import sys
import click
import yaml
import json
import logging
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import track
from rich.logging import RichHandler

from utils.data_loader import DataLoader
from utils.preprocessor import TimeSeriesPreprocessor
from utils.visualizer import AnomalyVisualizer
from utils.evaluator import ModelEvaluator
from models.model_factory import ModelFactory
from models.meta_selector import MetaModelSelector
from utils.config_manager import ConfigManager
from utils.logger import setup_logger

# Global console for rich output
console = Console()

# Setup logging
logging.basicConfig(
    level="INFO",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)]
)
logger = logging.getLogger("mauri-mtsb")

@click.group()
@click.version_option(version="1.0.0")
@click.pass_context
def cli(ctx):
    """
    🔥 Mauri-mTSB: Professional Anomaly Detection Tool for Cybersecurity
    
    A production-ready CLI tool for detecting anomalies in multivariate time series
    data from system/network logs, designed for cybersecurity professionals.
    """
    ctx.ensure_object(dict)
    
    # Display banner
    banner = Panel.fit(
        "[bold red]🔥 Mauri-mTSB v1.0.0[/bold red]\n"
        "[cyan]Professional Anomaly Detection Tool[/cyan]\n"
        "[dim]Author: Mohamed lemine Ahmed Jidou[/dim]\n"
        "[dim]Platform: Kali Linux | Python 3.10+[/dim]",
        border_style="red"
    )
    console.print(banner)

@cli.command()
@click.option('--input', '-i', required=True, type=click.Path(exists=True),
              help='Input CSV file with multivariate time series data')
@click.option('--config', '-c', type=click.Path(exists=True),
              help='Configuration YAML file (optional)')
@click.option('--output', '-o', default='./results',
              help='Output directory for results')
@click.option('--models', '-m', multiple=True,
              help='Specific models to run (optional, runs all if not specified)')
@click.option('--auto-select', is_flag=True,
              help='Use automatic model selection')
@click.option('--visualize', is_flag=True,
              help='Generate visualization plots')
@click.option('--export-format', type=click.Choice(['json', 'csv', 'both']),
              default='both', help='Export format for results')
@click.option('--verbose', '-v', is_flag=True,
              help='Enable verbose output')
def detect(input, config, output, models, auto_select, visualize, export_format, verbose):
    """
    🎯 Run anomaly detection on multivariate time series data
    
    Supports various data sources: NetFlow logs, system logs, authentication events, etc.
    """
    try:
        # Setup logger
        setup_logger(verbose)
        
        # Load configuration
        config_manager = ConfigManager(config)
        cfg = config_manager.get_config()
        
        console.print(f"[green]🔍 Starting anomaly detection on:[/green] {input}")
        
        # Create output directory
        output_path = Path(output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Step 1: Load data
        console.print("[blue]📊 Loading and preprocessing data...[/blue]")
        data_loader = DataLoader(cfg['data'])
        df = data_loader.load_csv(input)
        
        # Step 2: Preprocess data
        preprocessor = TimeSeriesPreprocessor(cfg['preprocessing'])
        X_processed, timestamps = preprocessor.fit_transform(df)
        
        console.print(f"[green]✓ Data shape:[/green] {X_processed.shape}")
        
        # Step 3: Initialize models
        model_factory = ModelFactory(cfg['models'])
        
        if models:
            # Use specified models
            selected_models = list(models)
        elif auto_select:
            # Use automatic model selection
            console.print("[blue]🤖 Running automatic model selection...[/blue]")
            meta_selector = MetaModelSelector(cfg['meta_selection'])
            selected_models = meta_selector.select_models(X_processed)
            console.print(f"[green]✓ Selected models:[/green] {', '.join(selected_models)}")
        else:
            # Use all available models
            selected_models = model_factory.get_available_models()
        
        # Step 4: Run anomaly detection
        results = {}
        evaluator = ModelEvaluator(cfg['evaluation'])
        
        for model_name in track(selected_models, description="Running models..."):
            try:
                console.print(f"[yellow]🔄 Running {model_name}...[/yellow]")
                
                # Get model
                model = model_factory.get_model(model_name)
                
                # Fit and predict
                anomaly_scores = model.fit_predict(X_processed)
                
                # Evaluate
                metrics = evaluator.evaluate(anomaly_scores, timestamps)
                
                results[model_name] = {
                    'anomaly_scores': anomaly_scores.tolist(),
                    'metrics': metrics,
                    'timestamps': timestamps.tolist() if timestamps is not None else None
                }
                
                console.print(f"[green]✓ {model_name} completed[/green]")
                
            except Exception as e:
                logger.error(f"Error running {model_name}: {str(e)}")
                continue
        
        # Step 5: Display results
        _display_results(results)
        
        # Step 6: Export results
        _export_results(results, output_path, export_format)
        
        # Step 7: Generate visualizations
        if visualize:
            console.print("[blue]📈 Generating visualizations...[/blue]")
            visualizer = AnomalyVisualizer(cfg['visualization'])
            visualizer.plot_results(X_processed, results, timestamps, output_path)
            console.print(f"[green]✓ Visualizations saved to:[/green] {output_path}")
        
        console.print(f"[bold green]🎉 Anomaly detection completed! Results saved to: {output_path}[/bold green]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
@click.option('--output', '-o', default='./config',
              help='Output directory for configuration files')
def init_config(output):
    """
    📝 Generate default configuration files
    """
    try:
        config_manager = ConfigManager()
        config_path = Path(output)
        config_path.mkdir(parents=True, exist_ok=True)
        
        config_manager.create_default_config(config_path / 'config.yaml')
        
        console.print(f"[green]✓ Default configuration created at:[/green] {config_path / 'config.yaml'}")
        console.print("[dim]Edit the configuration file to customize your analysis[/dim]")
        
    except Exception as e:
        console.print(f"[bold red]❌ Error creating config: {str(e)}[/bold red]")
        sys.exit(1)

@cli.command()
def list_models():
    """
    📋 List all available anomaly detection models
    """
    try:
        model_factory = ModelFactory()
        models = model_factory.get_available_models()
        
        table = Table(title="Available Anomaly Detection Models")
        table.add_column("Model", style="cyan", no_wrap=True)
        table.add_column("Type", style="magenta")
        table.add_column("Description", style="green")
        
        model_info = {
            'isolation_forest': ('Statistical', 'Isolation Forest for outlier detection'),
            'pca': ('Statistical', 'Principal Component Analysis'),
            'autoencoder': ('Deep Learning', 'Neural network autoencoder'),
            'lstm_autoencoder': ('Deep Learning', 'LSTM-based autoencoder'),
            'transformer': ('Deep Learning', 'Transformer-based detection'),
            'one_class_svm': ('Statistical', 'One-Class Support Vector Machine'),
            'local_outlier_factor': ('Statistical', 'Local Outlier Factor'),
            'dbscan': ('Clustering', 'Density-based clustering'),
            'gaussian_mixture': ('Statistical', 'Gaussian Mixture Model'),
            'prophet': ('Time Series', 'Facebook Prophet for time series'),
            'arima': ('Time Series', 'ARIMA-based detection'),
            'seasonal_decompose': ('Time Series', 'Seasonal decomposition')
        }
        
        for model in models:
            model_type, description = model_info.get(model, ('Unknown', 'Model description'))
            table.add_row(model, model_type, description)
        
        console.print(table)
        
    except Exception as e:
        console.print(f"[bold red]❌ Error listing models: {str(e)}[/bold red]")

def _display_results(results):
    """Display results in a formatted table"""
    if not results:
        console.print("[yellow]⚠️  No results to display[/yellow]")
        return
    
    table = Table(title="Anomaly Detection Results")
    table.add_column("Model", style="cyan", no_wrap=True)
    table.add_column("Anomalies", style="red", justify="right")
    table.add_column("AUC-ROC", style="green", justify="right")
    table.add_column("F1-Score", style="blue", justify="right")
    table.add_column("Precision", style="magenta", justify="right")
    table.add_column("Recall", style="yellow", justify="right")
    
    for model_name, result in results.items():
        metrics = result.get('metrics', {})
        anomaly_count = sum(1 for score in result['anomaly_scores'] if score > 0.5)
        
        table.add_row(
            model_name,
            str(anomaly_count),
            f"{metrics.get('auc_roc', 0.0):.3f}",
            f"{metrics.get('f1_score', 0.0):.3f}",
            f"{metrics.get('precision', 0.0):.3f}",
            f"{metrics.get('recall', 0.0):.3f}"
        )
    
    console.print(table)

def _export_results(results, output_path, export_format):
    """Export results to specified format"""
    try:
        if export_format in ['json', 'both']:
            json_path = output_path / 'results.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=2)
            console.print(f"[green]✓ JSON results saved to:[/green] {json_path}")
        
        if export_format in ['csv', 'both']:
            # Create a summary CSV
            import pandas as pd
            
            summary_data = []
            for model_name, result in results.items():
                metrics = result.get('metrics', {})
                anomaly_count = sum(1 for score in result['anomaly_scores'] if score > 0.5)
                
                summary_data.append({
                    'model': model_name,
                    'anomalies_detected': anomaly_count,
                    'auc_roc': metrics.get('auc_roc', 0.0),
                    'f1_score': metrics.get('f1_score', 0.0),
                    'precision': metrics.get('precision', 0.0),
                    'recall': metrics.get('recall', 0.0)
                })
            
            df_summary = pd.DataFrame(summary_data)
            csv_path = output_path / 'results_summary.csv'
            df_summary.to_csv(csv_path, index=False)
            console.print(f"[green]✓ CSV results saved to:[/green] {csv_path}")
            
    except Exception as e:
        logger.error(f"Error exporting results: {str(e)}")

if __name__ == '__main__':
    cli()