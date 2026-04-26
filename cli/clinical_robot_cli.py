#!/usr/bin/env python3
"""
Clinical Robot Adaptation CLI Tool
Comprehensive command-line interface for clinical robot management.

This CLI provides:
- Model training and evaluation commands
- Data processing and analysis tools
- Simulation and testing utilities
- Deployment and monitoring commands
- Configuration management
- Batch processing capabilities

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import argparse
import logging
import subprocess
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# CLI and formatting
import click
from click import Context, Group, command, option, argument, pass_context
import rich
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.panel import Panel
from rich.tree import Tree
from rich import print as rprint

# Core imports
import numpy as np
import pandas as pd
import torch
import yaml

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from models.octo_adapter.fine_tuning import ClinicalOctoAdapter
from scripts.data_processing_pipeline import ClinicalDataProcessor
from benchmark.clinbench_meddel.runner import ClinBenchMedDel
from protocols.clinical_data_collection import ClinicalDataCollectionProtocol

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize rich console
console = Console()

# Configuration
CONFIG_DIR = Path.home() / '.clinical_robot'
CONFIG_FILE = CONFIG_DIR / 'config.yaml'

class ClinicalRobotCLI:
    """Main CLI class for clinical robot adaptation."""
    
    def __init__(self):
        self.config = self._load_config()
        self.ensure_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            'models_dir': './models/trained',
            'data_dir': './data',
            'logs_dir': './logs',
            'results_dir': './results',
            'api_url': 'http://localhost:8000',
            'default_model': 'latest',
            'device': 'cuda' if torch.cuda.is_available() else 'cpu'
        }
        
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, 'r') as f:
                    user_config = yaml.safe_load(f)
                default_config.update(user_config)
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load config file: {e}[/yellow]")
        
        return default_config
    
    def ensure_directories(self):
        """Ensure required directories exist."""
        dirs = [
            Path(self.config['models_dir']),
            Path(self.config['data_dir']),
            Path(self.config['logs_dir']),
            Path(self.config['results_dir']),
            CONFIG_DIR
        ]
        
        for dir_path in dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def save_config(self):
        """Save current configuration."""
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with open(CONFIG_FILE, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)

# CLI instance
cli_instance = ClinicalRobotCLI()

@click.group()
@click.version_option(version="2.0.0", prog_name="clinical-robot")
@click.pass_context
def cli(ctx):
    """Clinical Robot Adaptation CLI - Comprehensive management tool for clinical robotics."""
    ctx.ensure_object(dict)
    ctx.obj['config'] = cli_instance.config

@cli.group()
def model():
    """Model management commands."""
    pass

@model.command()
@click.option('--model-path', '-m', required=True, help='Path to model file')
@click.option('--config', '-c', help='Model configuration file')
@click.option('--output', '-o', help='Output directory for results')
@pass_context
def train(ctx, model_path, config, output):
    """Train clinical robot adaptation model."""
    console.print("[bold blue]Training Clinical Robot Adaptation Model[/bold blue]")
    
    # Load configuration
    model_config = {}
    if config:
        with open(config, 'r') as f:
            model_config = json.load(f)
    
    # Initialize adapter
    try:
        adapter = ClinicalOctoAdapter(model_path, model_config)
        console.print("[green]✓[/green] Model adapter initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize model: {e}")
        return
    
    # Prepare dataset
    data_dir = Path(ctx.obj['config']['data_dir'])
    dataset_path = data_dir / 'clinical_demonstrations.hdf5'
    
    if not dataset_path.exists():
        console.print(f"[red]✗[/red] Dataset not found: {dataset_path}")
        console.print("Hint: Use 'clinical-robot data prepare' to create dataset")
        return
    
    with console.status("[bold green]Preparing dataset..."):
        dataset = adapter.prepare_clinical_dataset(str(dataset_path))
    
    console.print(f"[green]✓[/green] Dataset prepared: {len(dataset)} demonstrations")
    
    # Start training
    output_dir = Path(output) if output else Path(ctx.obj['config']['results_dir']) / f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Training model...", total=100)
        
        try:
            # Run training
            results = adapter.finetune(dataset, num_steps=1000)
            
            # Update progress
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(0.05)  # Simulate training time
            
            # Save results
            results_file = output_dir / 'training_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[green]✓[/green] Training completed successfully")
            console.print(f"[blue]Results saved to:[/blue] {output_dir}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Training failed: {e}")

@model.command()
@click.option('--model-path', '-m', required=True, help='Path to trained model')
@click.option('--episodes', '-e', default=50, help='Number of evaluation episodes')
@click.option('--environments', '-n', default=3, help='Number of test environments')
@click.option('--output', '-o', help='Output directory for results')
@pass_context
def evaluate(ctx, model_path, episodes, environments, output):
    """Evaluate trained model on ClinBench-MedDel benchmark."""
    console.print("[bold blue]Evaluating Model on ClinBench-MedDel[/bold blue]")
    
    # Initialize benchmark
    benchmark_config = {
        'num_episodes': episodes,
        'num_environments': environments,
        'output_dir': output or ctx.obj['config']['results_dir']
    }
    
    try:
        benchmark = ClinBenchMedDel(benchmark_config)
        console.print("[green]✓[/green] Benchmark initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize benchmark: {e}")
        return
    
    # Load model (placeholder)
    model = None  # Would load actual model here
    
    # Run evaluation
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Running evaluation...", total=100)
        
        try:
            results = benchmark.evaluate_model(model, "evaluation_run")
            
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(0.1)
            
            # Display results
            console.print("\n[bold green]Evaluation Results[/bold green]")
            
            results_table = Table(title="Performance Metrics")
            results_table.add_column("Metric", style="cyan")
            results_table.add_column("Value", style="green")
            
            metrics = results.get('overall_metrics', {})
            for metric, value in metrics.items():
                if isinstance(value, float):
                    results_table.add_row(metric.replace('_', ' ').title(), f"{value:.3f}")
                else:
                    results_table.add_row(metric.replace('_', ' ').title(), str(value))
            
            console.print(results_table)
            
            # Save results
            output_dir = Path(output) if output else Path(ctx.obj['config']['results_dir']) / f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            results_file = output_dir / 'evaluation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            console.print(f"[blue]Results saved to:[/blue] {output_dir}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Evaluation failed: {e}")

@model.command()
@click.option('--model-path', '-m', required=True, help='Path to model file')
@click.option('--name', '-n', help='Model name for registration')
@pass_context
def register(ctx, model_path, name):
    """Register a new model in the system."""
    console.print("[bold blue]Registering Model[/bold blue]")
    
    model_path = Path(model_path)
    if not model_path.exists():
        console.print(f"[red]✗[/red] Model file not found: {model_path}")
        return
    
    # Copy to models directory
    models_dir = Path(ctx.obj['config']['models_dir'])
    models_dir.mkdir(parents=True, exist_ok=True)
    
    model_name = name or model_path.stem
    target_path = models_dir / f"{model_name}.pt"
    
    try:
        import shutil
        shutil.copy2(model_path, target_path)
        console.print(f"[green]✓[/green] Model registered as: {model_name}")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to register model: {e}")

@model.command()
def list():
    """List all registered models."""
    console.print("[bold blue]Registered Models[/bold blue]")
    
    models_dir = Path(cli_instance.config['models_dir'])
    if not models_dir.exists():
        console.print("[yellow]No models directory found[/yellow]")
        return
    
    model_files = list(models_dir.glob("*.pt"))
    
    if not model_files:
        console.print("[yellow]No models found[/yellow]")
        return
    
    table = Table(title="Available Models")
    table.add_column("Name", style="cyan")
    table.add_column("Size", style="green")
    table.add_column("Modified", style="blue")
    
    for model_file in model_files:
        size_mb = model_file.stat().st_size / (1024 * 1024)
        modified = datetime.fromtimestamp(model_file.stat().st_mtime).strftime('%Y-%m-%d %H:%M')
        
        table.add_row(model_file.stem, f"{size_mb:.1f} MB", modified)
    
    console.print(table)

@cli.group()
def data():
    """Data processing and management commands."""
    pass

@data.command()
@click.option('--input-dir', '-i', required=True, help='Input directory with raw data')
@click.option('--output-dir', '-o', help='Output directory for processed data')
@click.option('--config', '-c', help='Processing configuration file')
@click.option('--validate', '-v', is_flag=True, help='Validate processed data')
@pass_context
def process(ctx, input_dir, output_dir, config, validate):
    """Process raw clinical demonstration data."""
    console.print("[bold blue]Processing Clinical Data[/bold blue]")
    
    input_path = Path(input_dir)
    if not input_path.exists():
        console.print(f"[red]✗[/red] Input directory not found: {input_path}")
        return
    
    output_path = Path(output_dir) if output_dir else Path(ctx.obj['config']['data_dir']) / 'processed'
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load configuration
    proc_config = {}
    if config:
        with open(config, 'r') as f:
            proc_config = json.load(f)
    
    # Initialize processor
    try:
        processor = ClinicalDataProcessor(proc_config)
        console.print("[green]✓[/green] Data processor initialized")
    except Exception as e:
        console.print(f"[red]✗[/red] Failed to initialize processor: {e}")
        return
    
    # Process data
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Processing data...", total=100)
        
        try:
            # Run processing
            results = processor.process_clinical_data(str(input_path), str(output_path))
            
            for i in range(100):
                progress.update(task, advance=1)
                time.sleep(0.05)
            
            console.print(f"[green]✓[/green] Data processing completed")
            console.print(f"[blue]Processed {results.get('total_demonstrations', 0)} demonstrations[/blue]")
            
            # Validate if requested
            if validate:
                with console.status("[bold green]Validating processed data..."):
                    validation_results = processor.validate_processed_data(str(output_path))
                
                console.print(f"[green]✓[/green] Validation completed")
                console.print(f"[blue]Quality score: {validation_results.get('quality_score', 0):.3f}[/blue]")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Data processing failed: {e}")

@data.command()
@click.option('--demonstrations', '-d', type=int, default=100, help='Number of demonstrations to generate')
@click.option('--output', '-o', help='Output file path')
@click.option('--format', '-f', type=click.Choice(['hdf5', 'json', 'csv']), default='hdf5', help='Output format')
@pass_context
def generate(ctx, demonstrations, output, format):
    """Generate synthetic clinical demonstration data."""
    console.print("[bold blue]Generating Synthetic Data[/bold blue]")
    
    output_path = Path(output) if output else Path(ctx.obj['config']['data_dir']) / f'synthetic_demonstrations.{format}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Generating data...", total=demonstrations)
        
        try:
            # Generate synthetic data
            synthetic_data = []
            
            for i in range(demonstrations):
                # Generate random demonstration
                demo = {
                    'id': f'demo_{i:04d}',
                    'timestamp': datetime.now().isoformat(),
                    'instruction': f'Pick up medication {i % 5}',
                    'robot_state': np.random.randn(10).tolist(),
                    'action': np.random.randn(7).tolist(),
                    'success': np.random.random() > 0.1,
                    'medication_type': ['vial', 'bottle', 'syringe', 'blister_pack', 'pouch'][i % 5]
                }
                synthetic_data.append(demo)
                
                progress.update(task, advance=1)
                time.sleep(0.01)
            
            # Save data
            if format == 'json':
                with open(output_path, 'w') as f:
                    json.dump(synthetic_data, f, indent=2)
            elif format == 'csv':
                df = pd.DataFrame(synthetic_data)
                df.to_csv(output_path, index=False)
            elif format == 'hdf5':
                import h5py
                with h5py.File(output_path, 'w') as f:
                    for i, demo in enumerate(synthetic_data):
                        grp = f.create_group(f'demo_{i:04d}')
                        for key, value in demo.items():
                            if isinstance(value, list):
                                grp.create_dataset(key, data=value)
                            else:
                                grp.attrs[key] = value
            
            console.print(f"[green]✓[/green] Generated {demonstrations} demonstrations")
            console.print(f"[blue]Saved to:[/blue] {output_path}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Data generation failed: {e}")

@data.command()
@click.option('--data-path', '-d', required=True, help='Path to data file')
@click.option('--output', '-o', help='Output directory for analysis')
@pass_context
def analyze(ctx, data_path, output):
    """Analyze clinical demonstration data."""
    console.print("[bold blue]Analyzing Clinical Data[/bold blue]")
    
    data_file = Path(data_path)
    if not data_file.exists():
        console.print(f"[red]✗[/red] Data file not found: {data_file}")
        return
    
    output_dir = Path(output) if output else Path(ctx.obj['config']['results_dir']) / f'analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # Load and analyze data
        if data_file.suffix == '.json':
            with open(data_file, 'r') as f:
                data = json.load(f)
        elif data_file.suffix == '.csv':
            df = pd.read_csv(data_file)
            data = df.to_dict('records')
        else:
            console.print(f"[red]✗[/red] Unsupported file format: {data_file.suffix}")
            return
        
        # Perform analysis
        console.print("[green]Analyzing data patterns...[/green]")
        
        # Basic statistics
        total_demos = len(data)
        success_rate = sum(1 for demo in data if demo.get('success', False)) / total_demos
        
        # Medication type distribution
        med_types = {}
        for demo in data:
            med_type = demo.get('medication_type', 'unknown')
            med_types[med_type] = med_types.get(med_type, 0) + 1
        
        # Display results
        console.print("\n[bold green]Analysis Results[/bold green]")
        
        stats_table = Table(title="Data Statistics")
        stats_table.add_column("Metric", style="cyan")
        stats_table.add_column("Value", style="green")
        
        stats_table.add_row("Total Demonstrations", str(total_demos))
        stats_table.add_row("Success Rate", f"{success_rate:.2%}")
        
        for med_type, count in med_types.items():
            stats_table.add_row(f"{med_type.title()} Count", str(count))
        
        console.print(stats_table)
        
        # Save analysis
        analysis_results = {
            'total_demonstrations': total_demos,
            'success_rate': success_rate,
            'medication_distribution': med_types,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        results_file = output_dir / 'analysis_results.json'
        with open(results_file, 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        console.print(f"[blue]Analysis saved to:[/blue] {output_dir}")
        
    except Exception as e:
        console.print(f"[red]✗[/red] Data analysis failed: {e}")

@cli.group()
def sim():
    """Simulation and testing commands."""
    pass

@sim.command()
@click.option('--scenario', '-s', help='Specific scenario to run')
@click.option('--config', '-c', help='Simulation configuration file')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--headless', is_flag=True, help='Run in headless mode')
@pass_context
def run(ctx, scenario, config, output, headless):
    """Run simulation scenarios."""
    console.print("[bold blue]Running Simulation Scenarios[/bold blue]")
    
    # Import simulation module
    try:
        from simulation.test_scenarios import TestScenarioRunner, create_standard_scenarios
        console.print("[green]✓[/green] Simulation module loaded")
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to import simulation module: {e}")
        return
    
    # Load configuration
    sim_config = {}
    if config:
        with open(config, 'r') as f:
            sim_config = json.load(f)
    
    # Initialize runner
    runner = TestScenarioRunner(sim_config)
    
    # Load scenarios
    scenarios = create_standard_scenarios()
    
    if scenario:
        # Filter specific scenario
        scenarios = [s for s in scenarios if s.scenario_id == scenario]
        if not scenarios:
            console.print(f"[red]✗[/red] Scenario not found: {scenario}")
            return
    
    runner.load_scenarios(scenarios)
    console.print(f"[green]✓[/green] Loaded {len(scenarios)} scenarios")
    
    # Run scenarios
    output_dir = Path(output) if output else Path(ctx.obj['config']['results_dir']) / f'simulation_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    sim_config['output_dir'] = str(output_dir)
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        console=console
    ) as progress:
        
        task = progress.add_task("[green]Running scenarios...", total=len(scenarios))
        
        try:
            results = runner.run_all_scenarios()
            
            for i, result in enumerate(results):
                progress.update(task, advance=1)
                time.sleep(0.1)
            
            # Generate report
            report = runner.generate_report()
            
            console.print(f"[green]✓[/green] Simulation completed")
            
            # Display summary
            passed = sum(1 for r in results if r.get('success', False))
            total = len(results)
            
            console.print(f"\n[bold green]Simulation Summary[/bold green]")
            console.print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
            
            # Save results
            results_file = output_dir / 'simulation_results.json'
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            report_file = output_dir / 'simulation_report.md'
            with open(report_file, 'w') as f:
                f.write(report)
            
            console.print(f"[blue]Results saved to:[/blue] {output_dir}")
            
        except Exception as e:
            console.print(f"[red]✗[/red] Simulation failed: {e}")

@sim.command()
def list_scenarios():
    """List available simulation scenarios."""
    console.print("[bold blue]Available Simulation Scenarios[/bold blue]")
    
    try:
        from simulation.test_scenarios import create_standard_scenarios, ScenarioType, ScenarioDifficulty
        
        scenarios = create_standard_scenarios()
        
        table = Table(title="Simulation Scenarios")
        table.add_column("ID", style="cyan")
        table.add_column("Type", style="green")
        table.add_column("Difficulty", style="yellow")
        table.add_column("Description", style="blue")
        
        for scenario in scenarios:
            table.add_row(
                scenario.scenario_id,
                scenario.scenario_type.value,
                scenario.difficulty.value,
                scenario.description[:50] + "..." if len(scenario.description) > 50 else scenario.description
            )
        
        console.print(table)
        
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to load scenarios: {e}")

@cli.group()
def deploy():
    """Deployment and monitoring commands."""
    pass

@deploy.command()
@click.option('--port', '-p', default=8000, help='Port to run API server on')
@click.option('--host', '-h', default='0.0.0.0', help='Host to bind to')
@click.option('--workers', '-w', default=1, help='Number of worker processes')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def api(port, host, workers, reload):
    """Start the API server."""
    console.print("[bold blue]Starting API Server[/bold blue]")
    
    try:
        import uvicorn
        
        console.print(f"[green]✓[/green] Starting server on {host}:{port}")
        
        # Start server
        uvicorn.run(
            "api.fastapi_server:app",
            host=host,
            port=port,
            workers=workers if not reload else 1,
            reload=reload,
            log_level="info"
        )
        
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to start server: {e}")
        console.print("Hint: Install with 'pip install uvicorn'")

@deploy.command()
@click.option('--port', '-p', default=8501, help='Port to run dashboard on')
def dashboard(port):
    """Start the visualization dashboard."""
    console.print("[bold blue]Starting Visualization Dashboard[/bold blue]")
    
    try:
        import streamlit.web.cli as stcli
        
        dashboard_script = Path(__file__).parent.parent / 'dashboard' / 'visualization_dashboard.py'
        
        if not dashboard_script.exists():
            console.print(f"[red]✗[/red] Dashboard script not found: {dashboard_script}")
            return
        
        console.print(f"[green]✓[/green] Starting dashboard on port {port}")
        
        # Start dashboard
        sys.argv = [
            'streamlit',
            'run',
            str(dashboard_script),
            '--server.port',
            str(port),
            '--server.headless',
            'true'
        ]
        
        stcli.main()
        
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to start dashboard: {e}")
        console.print("Hint: Install with 'pip install streamlit'")

@deploy.command()
@click.option('--interval', '-i', default=60, help='Monitoring interval in seconds')
@click.option('--output', '-o', help='Output file for monitoring data')
def monitor(interval, output):
    """Start system monitoring."""
    console.print("[bold blue]Starting System Monitoring[/bold blue]")
    
    monitoring_data = []
    
    try:
        import psutil
        
        while True:
            # Collect system metrics
            timestamp = datetime.now()
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            metrics = {
                'timestamp': timestamp.isoformat(),
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_used_gb': memory.used / (1024**3),
                'disk_percent': disk.percent,
                'disk_used_gb': disk.used / (1024**3)
            }
            
            monitoring_data.append(metrics)
            
            # Display current metrics
            console.clear()
            console.print(Panel(f"[bold green]System Monitoring - {timestamp.strftime('%Y-%m-%d %H:%M:%S')}[/bold green]"))
            
            metrics_table = Table()
            metrics_table.add_column("Metric", style="cyan")
            metrics_table.add_column("Value", style="green")
            
            metrics_table.add_row("CPU Usage", f"{cpu_percent:.1f}%")
            metrics_table.add_row("Memory Usage", f"{memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB)")
            metrics_table.add_row("Disk Usage", f"{disk.percent:.1f}% ({disk.used / (1024**3):.1f} GB)")
            
            console.print(metrics_table)
            
            # Save data if requested
            if output:
                output_file = Path(output)
                with open(output_file, 'w') as f:
                    json.dump(monitoring_data, f, indent=2)
            
            time.sleep(interval)
            
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitoring stopped by user[/yellow]")
    except ImportError as e:
        console.print(f"[red]✗[/red] Failed to start monitoring: {e}")
        console.print("Hint: Install with 'pip install psutil'")

@cli.command()
@click.option('--show-secrets', is_flag=True, help='Show sensitive configuration')
def config(show_secrets):
    """Show current configuration."""
    console.print("[bold blue]Current Configuration[/bold blue]")
    
    config_display = cli_instance.config.copy()
    
    # Hide sensitive information unless requested
    if not show_secrets:
        sensitive_keys = ['api_key', 'password', 'token', 'secret']
        for key in sensitive_keys:
            if key in config_display:
                config_display[key] = "***"
    
    # Display configuration
    config_tree = Tree("Configuration")
    
    for key, value in config_display.items():
        if isinstance(value, dict):
            subtree = config_tree.add(f"[cyan]{key}[/cyan]")
            for subkey, subvalue in value.items():
                subtree.add(f"[green]{subkey}[/green]: {subvalue}")
        else:
            config_tree.add(f"[cyan]{key}[/cyan]: {value}")
    
    console.print(config_tree)

@cli.command()
@click.option('--key', '-k', required=True, help='Configuration key')
@click.option('--value', '-v', required=True, help='Configuration value')
def set_config(key, value):
    """Set configuration value."""
    console.print(f"[bold blue]Setting Configuration: {key}[/bold blue]")
    
    # Try to parse value as JSON, fallback to string
    try:
        parsed_value = json.loads(value)
    except json.JSONDecodeError:
        parsed_value = value
    
    cli_instance.config[key] = parsed_value
    cli_instance.save_config()
    
    console.print(f"[green]✓[/green] Set {key} = {parsed_value}")

@cli.command()
def status():
    """Show system status."""
    console.print("[bold blue]System Status[/bold blue]")
    
    # Check directories
    dirs_to_check = [
        ('Models', cli_instance.config['models_dir']),
        ('Data', cli_instance.config['data_dir']),
        ('Logs', cli_instance.config['logs_dir']),
        ('Results', cli_instance.config['results_dir'])
    ]
    
    status_table = Table(title="Directory Status")
    status_table.add_column("Directory", style="cyan")
    status_table.add_column("Path", style="green")
    status_table.add_column("Status", style="yellow")
    
    for name, path in dirs_to_check:
        path_obj = Path(path)
        if path_obj.exists():
            if path_obj.is_dir():
                file_count = len(list(path_obj.rglob('*')))
                status_table.add_row(name, path, f"[green]Exists ({file_count} files)[/green]")
            else:
                status_table.add_row(name, path, "[red]Not a directory[/red]")
        else:
            status_table.add_row(name, path, "[red]Missing[/red]")
    
    console.print(status_table)
    
    # Check system resources
    try:
        import psutil
        
        console.print("\n[bold green]System Resources[/bold green]")
        
        resources_table = Table()
        resources_table.add_column("Resource", style="cyan")
        resources_table.add_column("Usage", style="green")
        
        resources_table.add_row("CPU", f"{psutil.cpu_percent():.1f}%")
        
        memory = psutil.virtual_memory()
        resources_table.add_row("Memory", f"{memory.percent:.1f}% ({memory.used / (1024**3):.1f} GB)")
        
        disk = psutil.disk_usage('/')
        resources_table.add_row("Disk", f"{disk.percent:.1f}% ({disk.used / (1024**3):.1f} GB)")
        
        console.print(resources_table)
        
    except ImportError:
        console.print("[yellow]psutil not installed - cannot show system resources[/yellow]")

if __name__ == "__main__":
    cli()
