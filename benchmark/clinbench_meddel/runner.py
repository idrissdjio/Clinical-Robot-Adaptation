#!/usr/bin/env python3
"""
ClinBench-MedDel: Clinical Medication Delivery Benchmark Runner
Comprehensive evaluation framework for clinical robot adaptation.

This script implements ClinBench-MedDel benchmark for evaluating
few-shot adapted robotic medication delivery systems. It provides:
- Standardized evaluation protocols
- Clinical-relevant performance metrics
- Cross-environment generalization testing
- Human-aware safety evaluation
- Comprehensive reporting and visualization

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import argparse
import logging
import json
import time
import pickle
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

# Core imports
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Robotics and simulation
import gymnasium as gym
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import trimesh

# Visualization and analysis
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image

# Metrics and evaluation
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from scipy import stats
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('benchmark.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Performance monitoring imports
import psutil
import threading
from concurrent.futures import ThreadPoolExecutor


class ClinBenchMedDel:
    """
    Clinical Medication Delivery Benchmark (ClinBench-MedDel).
    
    This benchmark evaluates robotic medication delivery systems across multiple
    dimensions critical for clinical deployment:
    - Task success and medication recognition accuracy
    - Cross-environment generalization
    - Human-aware safety and interaction
    - Grasp quality and manipulation precision
    - Trajectory efficiency and smoothness
    - Clinical workflow integration
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize ClinBench-MedDel benchmark.
        
        Args:
            config: Benchmark configuration
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './benchmark_results'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Benchmark parameters
        self.num_episodes = config.get('num_episodes', 50)
        self.num_environments = config.get('num_environments', 5)
        self.max_steps_per_episode = config.get('max_steps_per_episode', 100)
        self.safety_checks_enabled = config.get('safety_checks', True)
        
        # Clinical parameters
        self.medication_types = config.get('medication_types', [
            'vial', 'blister_pack', 'syringe', 'bottle', 'pouch'
        ])
        self.hospital_layouts = config.get('hospital_layouts', ['A', 'B', 'C'])
        self.workflow_scenarios = config.get('workflow_scenarios', [
            'routine', 'urgent', 'emergency', 'sterile_procedure'
        ])
        
        # Evaluation components
        self.environments = []
        self.metrics_calculator = ClinicalMetricsCalculator(config)
        self.safety_monitor = SafetyMonitor(config)
        self.human_simulator = HumanSimulator(config)
        
        # Results storage
        self.benchmark_results = {
            'episodes': [],
            'environment_results': {},
            'overall_metrics': {},
            'safety_analysis': {},
            'generalization_analysis': {}
        }
        
        # Initialize benchmark environments
        self._initialize_environments()
        
        logger.info(f"ClinBench-MedDel initialized")
        logger.info(f"Episodes per environment: {self.num_episodes}")
        logger.info(f"Number of environments: {self.num_environments}")
        logger.info(f"Output directory: {self.output_dir}")
    
    def _initialize_environments(self):
        """Initialize benchmark environments."""
        logger.info("Initializing benchmark environments")
        
        for i in range(self.num_environments):
            layout_type = self.hospital_layouts[i % len(self.hospital_layouts)]
            
            env_config = {
                'layout_type': layout_type,
                'medication_types': self.medication_types,
                'safety_checks': self.safety_checks_enabled,
                'human_simulation': True,
                'workspace_bounds': {
                    'x': [-1.0, 1.0],
                    'y': [-0.8, 0.8],
                    'z': [0.0, 1.5]
                }
            }
            
            # Create environment
            env = ClinicalPharmacyEnvironment(env_config)
            self.environments.append(env)
            
            logger.info(f"Environment {i+1}: Layout {layout_type}")
    
    def evaluate_model(self, model, model_name: str = "test_model") -> Dict[str, Any]:
        """
        Evaluate a model on ClinBench-MedDel benchmark.
        
        Args:
            model: Trained model to evaluate
            model_name: Name of model for reporting
            
        Returns:
            Comprehensive evaluation results
        """
        logger.info(f"Evaluating model: {model_name}")
        logger.info(f"Total episodes: {self.num_episodes * self.num_environments}")
        
        start_time = time.time()
        
        # Evaluate across all environments
        for env_idx, env in enumerate(self.environments):
            logger.info(f"Evaluating on environment {env_idx+1}/{self.num_environments}")
            
            env_results = self._evaluate_on_environment(model, env, env_idx)
            self.benchmark_results['environment_results'][f'env_{env_idx}'] = env_results
            
            # Log intermediate results
            env_metrics = env_results['summary_metrics']
            logger.info(f"  Success Rate: {env_metrics['success_rate']:.3f}")
            logger.info(f"  Grasp Success: {env_metrics['grasp_success_rate']:.3f}")
            logger.info(f"  Safety Score: {env_metrics['safety_score']:.3f}")
        
        # Compute overall metrics
        self._compute_overall_metrics()
        
        # Analyze generalization
        self._analyze_generalization()
        
        # Analyze safety
        self._analyze_safety()
        
        # Compute execution time
        total_time = time.time() - start_time
        self.benchmark_results['execution_info'] = {
            'total_time_seconds': total_time,
            'total_time_formatted': self._format_time(total_time),
            'episodes_per_second': (self.num_episodes * self.num_environments) / total_time,
            'model_name': model_name,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save results
        self._save_results(model_name)
        
        # Generate report
        report_path = self._generate_report(model_name)
        
        logger.info(f"Benchmark evaluation completed in {self._format_time(total_time)}")
        logger.info(f"Results saved to: {self.output_dir}")
        logger.info(f"Report generated: {report_path}")
        
        return self.benchmark_results
    
    def _evaluate_on_environment(self, model, env, env_idx: int) -> Dict[str, Any]:
        """Evaluate model on a single environment."""
        env_results = {
            'environment_index': env_idx,
            'layout_type': self.hospital_layouts[env_idx % len(self.hospital_layouts)],
            'episodes': [],
            'summary_metrics': {}
        }
        
        for episode in tqdm(range(self.num_episodes), 
                           desc=f"Env {env_idx+1} Episodes"):
            
            # Reset environment with random configuration
            obs = env.reset()
            
            # Run episode
            episode_result = self._run_episode(model, env, obs, episode)
            episode_result['environment_index'] = env_idx
            episode_result['episode_number'] = episode
            
            env_results['episodes'].append(episode_result)
        
        # Compute environment summary metrics
        env_results['summary_metrics'] = self.metrics_calculator.compute_environment_metrics(
            env_results['episodes']
        )
        
        return env_results
    
    def _run_episode(self, model, env, obs, episode_num: int) -> Dict[str, Any]:
        """Run a single evaluation episode."""
        episode_result = {
            'episode_number': episode_num,
            'success': False,
            'steps': 0,
            'rewards': [],
            'actions': [],
            'observations': [],
            'grasp_successes': [],
            'medication_recognitions': [],
            'safety_violations': [],
            'human_interactions': [],
            'trajectory_data': [],
            'clinical_metrics': {}
        }
        
        # Get episode configuration
        episode_config = env.get_episode_config()
        episode_result['episode_config'] = episode_config
        
        for step in range(self.max_steps_per_episode):
            # Record observation
            episode_result['observations'].append(obs.copy())
            
            # Get model prediction
            action = self._get_model_action(model, obs, episode_config)
            episode_result['actions'].append(action)
            
            # Execute action
            next_obs, reward, done, info = env.step(action)
            
            # Record step information
            episode_result['rewards'].append(reward)
            episode_result['steps'] += 1
            
            # Record trajectory data
            if 'robot_state' in obs:
                episode_result['trajectory_data'].append({
                    'step': step,
                    'position': obs['robot_state'][:3],
                    'orientation': obs['robot_state'][3:7],
                    'action': action,
                    'timestamp': time.time()
                })
            
            # Check grasp success
            if info.get('grasp_success', False):
                episode_result['grasp_successes'].append(step)
            
            # Check medication recognition
            if info.get('medication_recognized', False):
                episode_result['medication_recognitions'].append(step)
            
            # Check safety violations
            safety_violation = self.safety_monitor.check_safety(obs, action, info)
            if safety_violation['violation']:
                episode_result['safety_violations'].append({
                    'step': step,
                    'type': safety_violation['type'],
                    'severity': safety_violation['severity'],
                    'description': safety_violation['description']
                })
            
            # Check human interactions
            human_interaction = self.human_simulator.check_interaction(obs, action, info)
            if human_interaction['interaction']:
                episode_result['human_interactions'].append({
                    'step': step,
                    'type': human_interaction['type'],
                    'distance': human_interaction['distance'],
                    'appropriate_response': human_interaction['appropriate_response']
                })
            
            obs = next_obs
            
            if done:
                episode_result['success'] = reward > 0
                break
        
        # Compute episode clinical metrics
        episode_result['clinical_metrics'] = self.metrics_calculator.compute_episode_metrics(
            episode_result
        )
        
        return episode_result
    
    def _get_model_action(self, model, obs, episode_config) -> np.ndarray:
        """Get action from model for current observation."""
        # This would interface with the actual model
        # For now, return a placeholder action
        
        # Extract relevant observation components
        if 'image' in obs and 'robot_state' in obs:
            # Real model inference would happen here
            # action = model.predict(obs['image'], obs['robot_state'], episode_config['instruction'])
            
            # Placeholder: simple heuristic action
            target_pos = episode_config.get('target_position', [0.5, 0.0, 0.8])
            current_pos = obs['robot_state'][:3]
            
            # Simple proportional control toward target
            direction = target_pos - current_pos
            distance = np.linalg.norm(direction)
            
            if distance > 0.05:
                action = np.concatenate([
                    direction / distance * 0.1,  # Linear velocity
                    np.zeros(4)  # Orientation (placeholder)
                ])
            else:
                action = np.zeros(7)  # Stop
            
            return action
        else:
            return np.zeros(7)  # Default action
    
    def _compute_overall_metrics(self):
        """Compute overall benchmark metrics across all environments."""
        all_episodes = []
        
        # Collect all episodes
        for env_results in self.benchmark_results['environment_results'].values():
            all_episodes.extend(env_results['episodes'])
        
        # Compute overall metrics
        self.benchmark_results['overall_metrics'] = self.metrics_calculator.compute_overall_metrics(
            all_episodes
        )
        
        logger.info(f"Overall success rate: {self.benchmark_results['overall_metrics']['success_rate']:.3f}")
        logger.info(f"Overall grasp success: {self.benchmark_results['overall_metrics']['grasp_success_rate']:.3f}")
        logger.info(f"Overall safety score: {self.benchmark_results['overall_metrics']['safety_score']:.3f}")
    
    def _analyze_generalization(self):
        """Analyze cross-environment generalization."""
        env_metrics = {}
        
        # Collect metrics per environment
        for env_name, env_results in self.benchmark_results['environment_results'].items():
            env_metrics[env_name] = env_results['summary_metrics']
        
        # Compute generalization metrics
        generalization = self.metrics_calculator.compute_generalization_metrics(env_metrics)
        
        self.benchmark_results['generalization_analysis'] = generalization
        
        logger.info(f"Generalization variance: {generalization['performance_variance']:.4f}")
        logger.info(f"Worst environment: {generalization['worst_environment']}")
        logger.info(f"Best environment: {generalization['best_environment']}")
    
    def _analyze_safety(self):
        """Analyze safety performance across all episodes."""
        all_safety_violations = []
        all_human_interactions = []
        
        # Collect safety data
        for env_results in self.benchmark_results['environment_results'].values():
            for episode in env_results['episodes']:
                all_safety_violations.extend(episode['safety_violations'])
                all_human_interactions.extend(episode['human_interactions'])
        
        # Compute safety analysis
        safety_analysis = self.safety_monitor.analyze_safety_performance(
            all_safety_violations, all_human_interactions
        )
        
        self.benchmark_results['safety_analysis'] = safety_analysis
        
        logger.info(f"Total safety violations: {len(all_safety_violations)}")
        logger.info(f"Critical violations: {safety_analysis['critical_violations']}")
        logger.info(f"Human awareness score: {safety_analysis['human_awareness_score']:.3f}")
    
    def _save_results(self, model_name: str):
        """Save benchmark results to files."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save complete results as JSON
        results_path = self.output_dir / f'benchmark_results_{model_name}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(self.benchmark_results, f, indent=2, default=str)
        
        # Save episode data as CSV
        episodes_path = self.output_dir / f'episodes_{model_name}_{timestamp}.csv'
        self._save_episodes_csv(episodes_path)
        
        # Save metrics summary
        metrics_path = self.output_dir / f'metrics_summary_{model_name}_{timestamp}.json'
        with open(metrics_path, 'w') as f:
            json.dump(self.benchmark_results['overall_metrics'], f, indent=2, default=str)
        
        logger.info(f"Results saved to: {results_path}")
        logger.info(f"Episodes CSV saved to: {episodes_path}")
        logger.info(f"Metrics summary saved to: {metrics_path}")
    
    def _save_episodes_csv(self, episodes_path: Path):
        """Save episode data as CSV."""
        episodes_data = []
        
        for env_results in self.benchmark_results['environment_results'].values():
            for episode in env_results['episodes']:
                row = {
                    'environment_index': episode['environment_index'],
                    'episode_number': episode['episode_number'],
                    'success': episode['success'],
                    'steps': episode['steps'],
                    'total_reward': sum(episode['rewards']),
                    'grasp_successes': len(episode['grasp_successes']),
                    'medication_recognitions': len(episode['medication_recognitions']),
                    'safety_violations': len(episode['safety_violations']),
                    'human_interactions': len(episode['human_interactions']),
                    'trajectory_smoothness': episode['clinical_metrics'].get('trajectory_smoothness', 0),
                    'task_efficiency': episode['clinical_metrics'].get('task_efficiency', 0)
                }
                
                # Add episode config
                if 'episode_config' in episode:
                    config = episode['episode_config']
                    row['medication_type'] = config.get('medication_type', 'unknown')
                    row['grasp_type'] = config.get('grasp_type', 'unknown')
                    row['layout_type'] = config.get('layout_type', 'unknown')
                
                episodes_data.append(row)
        
        # Save to CSV
        df = pd.DataFrame(episodes_data)
        df.to_csv(episodes_path, index=False)
    
    def _generate_report(self, model_name: str) -> str:
        """Generate comprehensive HTML report."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f'benchmark_report_{model_name}_{timestamp}.html'
        
        html_content = self._generate_html_report(model_name)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        return str(report_path)
    
    def _generate_html_report(self, model_name: str) -> str:
        """Generate HTML content for benchmark report."""
        overall = self.benchmark_results['overall_metrics']
        safety = self.benchmark_results['safety_analysis']
        gen = self.benchmark_results['generalization_analysis']
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ClinBench-MedDel Report - {model_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
                .container {{ max-width: 1200px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 20px rgba(0,0,0,0.1); }}
                .header {{ text-align: center; margin-bottom: 40px; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 10px; }}
                .section {{ margin: 30px 0; padding: 20px; border: 1px solid #e0e0e0; border-radius: 8px; }}
                .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
                .metric {{ background: #f8f9fa; padding: 20px; border-radius: 8px; text-align: center; border-left: 4px solid #007bff; }}
                .metric-value {{ font-size: 2em; font-weight: bold; color: #007bff; }}
                .metric-label {{ color: #666; margin-top: 5px; }}
                .success {{ border-left-color: #28a745; }}
                .success .metric-value {{ color: #28a745; }}
                .warning {{ border-left-color: #ffc107; }}
                .warning .metric-value {{ color: #ffc107; }}
                .danger {{ border-left-color: #dc3545; }}
                .danger .metric-value {{ color: #dc3545; }}
                table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
                th {{ background-color: #f8f9fa; font-weight: bold; }}
                .chart-container {{ margin: 30px 0; text-align: center; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>ClinBench-MedDel Benchmark Report</h1>
                    <h2>Model: {model_name}</h2>
                    <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                </div>
                
                <div class="section">
                    <h2>Overall Performance Metrics</h2>
                    <div class="metric-grid">
                        <div class="metric {'success' if overall['success_rate'] > 0.8 else 'warning' if overall['success_rate'] > 0.6 else 'danger'}">
                            <div class="metric-value">{overall['success_rate']:.3f}</div>
                            <div class="metric-label">Success Rate</div>
                        </div>
                        <div class="metric {'success' if overall['grasp_success_rate'] > 0.8 else 'warning' if overall['grasp_success_rate'] > 0.6 else 'danger'}">
                            <div class="metric-value">{overall['grasp_success_rate']:.3f}</div>
                            <div class="metric-label">Grasp Success Rate</div>
                        </div>
                        <div class="metric {'success' if overall['medication_recognition_accuracy'] > 0.8 else 'warning' if overall['medication_recognition_accuracy'] > 0.6 else 'danger'}">
                            <div class="metric-value">{overall['medication_recognition_accuracy']:.3f}</div>
                            <div class="metric-label">Medication Recognition</div>
                        </div>
                        <div class="metric {'success' if overall['safety_score'] > 0.8 else 'warning' if overall['safety_score'] > 0.6 else 'danger'}">
                            <div class="metric-value">{overall['safety_score']:.3f}</div>
                            <div class="metric-label">Safety Score</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{overall['trajectory_smoothness']:.3f}</div>
                            <div class="metric-label">Trajectory Smoothness</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{overall['task_efficiency']:.3f}</div>
                            <div class="metric-label">Task Efficiency</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Safety Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric {'danger' if safety['critical_violations'] > 0 else 'success'}">
                            <div class="metric-value">{safety['critical_violations']}</div>
                            <div class="metric-label">Critical Violations</div>
                        </div>
                        <div class="metric {'warning' if safety['total_violations'] > 10 else 'success'}">
                            <div class="metric-value">{safety['total_violations']}</div>
                            <div class="metric-label">Total Violations</div>
                        </div>
                        <div class="metric {'success' if safety['human_awareness_score'] > 0.8 else 'warning'}">
                            <div class="metric-value">{safety['human_awareness_score']:.3f}</div>
                            <div class="metric-label">Human Awareness Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Generalization Analysis</h2>
                    <div class="metric-grid">
                        <div class="metric {'success' if gen['performance_variance'] < 0.1 else 'warning' if gen['performance_variance'] < 0.2 else 'danger'}">
                            <div class="metric-value">{gen['performance_variance']:.4f}</div>
                            <div class="metric-label">Performance Variance</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{gen['worst_environment']}</div>
                            <div class="metric-label">Worst Environment</div>
                        </div>
                        <div class="metric">
                            <div class="metric-value">{gen['best_environment']}</div>
                            <div class="metric-label">Best Environment</div>
                        </div>
                        <div class="metric {'success' if gen['generalization_score'] > 0.8 else 'warning'}">
                            <div class="metric-value">{gen['generalization_score']:.3f}</div>
                            <div class="metric-label">Generalization Score</div>
                        </div>
                    </div>
                </div>
                
                <div class="section">
                    <h2>Execution Information</h2>
                    <table>
                        <tr><th>Metric</th><th>Value</th></tr>
                        <tr><td>Total Episodes</td><td>{self.benchmark_results['execution_info']['episodes_per_second'] * self.benchmark_results['execution_info']['total_time_seconds']:.0f}</td></tr>
                        <tr><td>Total Time</td><td>{self.benchmark_results['execution_info']['total_time_formatted']}</td></tr>
                        <tr><td>Episodes/Second</td><td>{self.benchmark_results['execution_info']['episodes_per_second']:.2f}</td></tr>
                        <tr><td>Environments</td><td>{len(self.benchmark_results['environment_results'])}</td></tr>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return html
    
    def _format_time(self, seconds: float) -> str:
        """Format time in human-readable format."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        
        if hours > 0:
            return f"{hours}h {minutes}m {secs}s"
        elif minutes > 0:
            return f"{minutes}m {secs}s"
        else:
            return f"{secs}s"


class ClinicalPharmacyEnvironment:
    """Simulated clinical pharmacy environment for benchmarking."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.layout_type = config.get('layout_type', 'A')
        self.workspace_bounds = config.get('workspace_bounds', {})
        
        # Initialize PyBullet simulation
        self.physics_client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
        # Load environment
        self._load_environment()
        
        # Episode tracking
        self.current_episode = 0
        self.episode_step = 0
        
        # Human simulation
        self.human_simulator = HumanSimulator(config)
    
    def _load_environment(self):
        """Load hospital pharmacy environment."""
        # Load robot
        self.robot_id = p.loadURDF("franka_panda/panda.urdf", 
                                  useFixedBase=True,
                                  basePosition=[0, 0, 0])
        
        # Load pharmacy layout based on type
        if self.layout_type == 'A':
            self._load_layout_a()
        elif self.layout_type == 'B':
            self._load_layout_b()
        elif self.layout_type == 'C':
            self._load_layout_c()
        
        # Set gravity and physics parameters
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(1/240)
    
    def _load_layout_a(self):
        """Load pharmacy layout type A."""
        # Central medication carousel
        carousel_pos = [0.5, 0, 0.8]
        self.carousel_id = p.loadURDF("carousel.urdf", 
                                     basePosition=carousel_pos)
        
        # Shelving units
        shelf_positions = [
            [-0.6, 0.3, 0.9],
            [0.6, 0.3, 0.9],
            [0, -0.4, 0.9]
        ]
        
        self.shelf_ids = []
        for pos in shelf_positions:
            shelf_id = p.loadURDF("shelf.urdf", basePosition=pos)
            self.shelf_ids.append(shelf_id)
    
    def _load_layout_b(self):
        """Load pharmacy layout type B."""
        # Vertical storage columns
        column_positions = [[-0.4, 0, 0], [0.4, 0, 0]]
        
        self.storage_ids = []
        for pos in column_positions:
            storage_id = p.loadURDF("vertical_storage.urdf", basePosition=pos)
            self.storage_ids.append(storage_id)
        
        # Conveyor system
        self.conveyor_id = p.loadURDF("conveyor.urdf", basePosition=[0, -0.6, 0.8])
    
    def _load_layout_c(self):
        """Load pharmacy layout type C."""
        # Simple shelving arrangement
        shelf_config = [
            [-0.7, 0, 0.8, 1.0],
            [0.7, 0, 0.8, 1.0],
            [0, -0.5, 0.8, 1.2]
        ]
        
        self.simple_shelf_ids = []
        for config in shelf_config:
            shelf_id = p.loadURDF("simple_shelf.urdf", 
                                   basePosition=config[:3])
            self.simple_shelf_ids.append(shelf_id)
    
    def reset(self) -> Dict[str, Any]:
        """Reset environment for new episode."""
        self.current_episode += 1
        self.episode_step = 0
        
        # Reset robot to home position
        home_pos = [0, 0, 0.5, 0, 0, 0, 1, 0]  # x,y,z,qx,qy,qz,qw,gripper
        p.resetBasePositionAndOrientation(self.robot_id, home_pos[:3], home_pos[3:])
        
        # Generate episode configuration
        episode_config = self._generate_episode_config()
        
        # Place medication object
        self._place_medication_object(episode_config)
        
        # Reset human simulator
        self.human_simulator.reset(episode_config)
        
        # Get initial observation
        obs = self._get_observation()
        obs['episode_config'] = episode_config
        
        return obs
    
    def _generate_episode_config(self) -> Dict[str, Any]:
        """Generate random episode configuration."""
        medication_types = ['vial', 'blister_pack', 'syringe', 'bottle', 'pouch']
        grasp_types = ['precision', 'power', 'pinch', 'lateral', 'cylindrical']
        
        config = {
            'medication_type': np.random.choice(medication_types),
            'grasp_type': np.random.choice(grasp_types),
            'target_position': self._generate_target_position(),
            'difficulty': np.random.uniform(0.1, 0.9),
            'human_present': np.random.random() < 0.3,
            'layout_type': self.layout_type
        }
        
        return config
    
    def _generate_target_position(self) -> List[float]:
        """Generate random target position within workspace."""
        bounds = self.workspace_bounds
        x = np.random.uniform(bounds['x'][0], bounds['x'][1])
        y = np.random.uniform(bounds['y'][0], bounds['y'][1])
        z = np.random.uniform(bounds['z'][0], bounds['z'][1])
        
        return [x, y, z]
    
    def _place_medication_object(self, config: Dict[str, Any]):
        """Place medication object in environment."""
        # Select object based on medication type
        object_urdf = f"{config['medication_type']}.urdf"
        
        # Place at target position
        self.medication_id = p.loadURDF(object_urdf, 
                                      basePosition=config['target_position'])
    
    def _get_observation(self) -> Dict[str, Any]:
        """Get current environment observation."""
        # Get robot state
        robot_pos, robot_orn = p.getBasePositionAndOrientation(self.robot_id)
        robot_joints = [p.getJointState(self.robot_id, i)[0] for i in range(7)]
        
        # Get camera image
        # This would get camera image from simulation
        # For now, return placeholder
        image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        
        obs = {
            'robot_state': np.concatenate([robot_pos, robot_orn, robot_joints]),
            'image': image,
            'medication_position': config['target_position'] if 'config' in locals() else [0, 0, 0],
            'human_state': self.human_simulator.get_state()
        }
        
        return obs
    
    def step(self, action: np.ndarray) -> Tuple[Dict[str, Any], float, bool, Dict[str, Any]]:
        """Execute action and return next observation."""
        self.episode_step += 1
        
        # Apply action to robot
        # This would convert action to robot commands
        # For now, simple position control
        if len(action) >= 3:
            target_pos = action[:3]
            p.resetBasePositionAndOrientation(self.robot_id, target_pos, [0, 0, 0, 1])
        
        # Update human simulation
        self.human_simulator.step()
        
        # Get new observation
        obs = self._get_observation()
        
        # Compute reward
        reward, done, info = self._compute_reward_and_done(obs)
        
        return obs, reward, done, info
    
    def _compute_reward_and_done(self, obs: Dict[str, Any]) -> Tuple[float, bool, Dict[str, Any]]:
        """Compute reward and termination condition."""
        reward = 0.0
        done = False
        info = {}
        
        # Distance to target
        if 'medication_position' in obs and 'robot_state' in obs:
            robot_pos = obs['robot_state'][:3]
            target_pos = obs['medication_position']
            distance = np.linalg.norm(robot_pos - target_pos)
            
            # Reward for getting closer
            reward = -distance
            
            # Success if close enough
            if distance < 0.05:
                reward = 10.0
                done = True
                info['grasp_success'] = True
                info['medication_recognized'] = True
            else:
                info['grasp_success'] = False
                info['medication_recognized'] = False
        
        # Timeout
        if self.episode_step >= 100:
            done = True
            info['timeout'] = True
        
        return reward, done, info
    
    def get_episode_config(self) -> Dict[str, Any]:
        """Get current episode configuration."""
        return getattr(self, '_current_episode_config', {})


class ClinicalMetricsCalculator:
    """Calculate clinical-relevant performance metrics."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def compute_episode_metrics(self, episode_result: Dict[str, Any]) -> Dict[str, float]:
        """Compute metrics for a single episode."""
        metrics = {}
        
        # Basic success metrics
        metrics['success'] = float(episode_result['success'])
        metrics['steps'] = float(episode_result['steps'])
        metrics['total_reward'] = float(sum(episode_result['rewards']))
        
        # Grasp metrics
        metrics['grasp_success_rate'] = float(len(episode_result['grasp_successes']) / max(episode_result['steps'], 1))
        metrics['medication_recognition_rate'] = float(len(episode_result['medication_recognitions']) / max(episode_result['steps'], 1))
        
        # Safety metrics
        metrics['safety_violation_rate'] = float(len(episode_result['safety_violations']) / max(episode_result['steps'], 1))
        critical_violations = [v for v in episode_result['safety_violations'] if v.get('severity') == 'critical']
        metrics['critical_violation_count'] = float(len(critical_violations))
        
        # Human interaction metrics
        appropriate_responses = [i for i in episode_result['human_interactions'] if i.get('appropriate_response', False)]
        metrics['human_awareness_rate'] = float(len(appropriate_responses) / max(len(episode_result['human_interactions']), 1))
        
        # Trajectory metrics
        if episode_result['trajectory_data']:
            metrics['trajectory_smoothness'] = self._compute_trajectory_smoothness(episode_result['trajectory_data'])
            metrics['task_efficiency'] = self._compute_task_efficiency(episode_result['trajectory_data'], episode_result['success'])
        else:
            metrics['trajectory_smoothness'] = 0.0
            metrics['task_efficiency'] = 0.0
        
        return metrics
    
    def compute_environment_metrics(self, episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute metrics for a specific environment."""
        if not episodes:
            return {}
        
        # Aggregate episode metrics
        env_metrics = {}
        
        for metric_name in ['success', 'grasp_success_rate', 'medication_recognition_rate', 
                          'safety_violation_rate', 'human_awareness_rate', 
                          'trajectory_smoothness', 'task_efficiency']:
            values = [ep['clinical_metrics'].get(metric_name, 0) for ep in episodes]
            env_metrics[f'mean_{metric_name}'] = np.mean(values)
            env_metrics[f'std_{metric_name}'] = np.std(values)
            env_metrics[f'min_{metric_name}'] = np.min(values)
            env_metrics[f'max_{metric_name}'] = np.max(values)
        
        # Additional environment-level metrics
        env_metrics['success_rate'] = env_metrics['mean_success']
        env_metrics['grasp_success_rate'] = env_metrics['mean_grasp_success_rate']
        env_metrics['medication_recognition_accuracy'] = env_metrics['mean_medication_recognition_rate']
        env_metrics['safety_score'] = 1.0 - env_metrics['mean_safety_violation_rate']
        env_metrics['human_awareness_score'] = env_metrics['mean_human_awareness_rate']
        env_metrics['trajectory_smoothness'] = env_metrics['mean_trajectory_smoothness']
        env_metrics['task_efficiency'] = env_metrics['mean_task_efficiency']
        
        return env_metrics
    
    def compute_overall_metrics(self, all_episodes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Compute overall benchmark metrics."""
        if not all_episodes:
            return {}
        
        # Collect all episode metrics
        all_episode_metrics = [ep['clinical_metrics'] for ep in all_episodes]
        
        overall = {}
        
        for metric_name in ['success', 'grasp_success_rate', 'medication_recognition_rate', 
                          'safety_violation_rate', 'human_awareness_rate', 
                          'trajectory_smoothness', 'task_efficiency']:
            values = [ep.get(metric_name, 0) for ep in all_episode_metrics]
            overall[metric_name] = np.mean(values)
            overall[f'{metric_name}_std'] = np.std(values)
        
        # Composite scores
        overall['clinical_score'] = (
            overall['success'] * 0.3 +
            overall['grasp_success_rate'] * 0.2 +
            overall['medication_recognition_accuracy'] * 0.2 +
            overall['safety_score'] * 0.2 +
            overall['human_awareness_score'] * 0.1
        )
        
        return overall
    
    def compute_generalization_metrics(self, env_metrics: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Compute generalization metrics across environments."""
        if not env_metrics:
            return {}
        
        # Extract success rates per environment
        success_rates = [metrics.get('success_rate', 0) for metrics in env_metrics.values()]
        
        generalization = {
            'performance_variance': np.var(success_rates),
            'performance_std': np.std(success_rates),
            'mean_performance': np.mean(success_rates),
            'worst_performance': np.min(success_rates),
            'best_performance': np.max(success_rates),
            'generalization_score': 1.0 - np.var(success_rates)
        }
        
        # Find best and worst environments
        env_names = list(env_metrics.keys())
        if success_rates:
            worst_idx = np.argmin(success_rates)
            best_idx = np.argmax(success_rates)
            generalization['worst_environment'] = env_names[worst_idx]
            generalization['best_environment'] = env_names[best_idx]
        
        return generalization
    
    def _compute_trajectory_smoothness(self, trajectory_data: List[Dict]) -> float:
        """Compute trajectory smoothness metric."""
        if len(trajectory_data) < 3:
            return 0.0
        
        positions = np.array([data['position'] for data in trajectory_data])
        
        # Compute accelerations
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Smoothness = 1 / (1 + variance of accelerations)
        smoothness = 1.0 / (1.0 + np.var(accelerations))
        
        return smoothness
    
    def _compute_task_efficiency(self, trajectory_data: List[Dict], success: bool) -> float:
        """Compute task efficiency metric."""
        if not trajectory_data:
            return 0.0
        
        # Path efficiency
        positions = np.array([data['position'] for data in trajectory_data])
        
        if len(positions) > 1:
            path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
            direct_distance = np.linalg.norm(positions[-1] - positions[0])
            
            path_efficiency = direct_distance / path_length if path_length > 0 else 0.0
        else:
            path_efficiency = 0.0
        
        # Time efficiency (penalize long episodes)
        time_efficiency = 1.0 / (1.0 + len(trajectory_data) / 100.0)
        
        # Combine with success
        overall_efficiency = path_efficiency * time_efficiency * (1.0 if success else 0.0)
        
        return overall_efficiency


class SafetyMonitor:
    """Monitor safety during robot operation."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_thresholds = config.get('safety_thresholds', {
            'min_human_distance': 0.5,
            'max_velocity': 0.3,
            'max_acceleration': 2.0,
            'workspace_bounds': {'x': [-1.0, 1.0], 'y': [-0.8, 0.8], 'z': [0.0, 1.5]}
        })
    
    def check_safety(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Check for safety violations."""
        violations = []
        
        # Check human distance
        if 'human_state' in obs and 'robot_state' in obs:
            human_pos = obs['human_state'].get('position', [10, 10, 1.0])  # Far away if not present
            robot_pos = obs['robot_state'][:3]
            
            distance = np.linalg.norm(robot_pos - human_pos)
            if distance < self.safety_thresholds['min_human_distance']:
                violations.append({
                    'type': 'human_distance',
                    'severity': 'critical' if distance < 0.3 else 'warning',
                    'distance': distance,
                    'threshold': self.safety_thresholds['min_human_distance'],
                    'description': f'Too close to human: {distance:.3f}m'
                })
        
        # Check velocity limits
        if len(action) >= 3:
            velocity = np.linalg.norm(action[:3])
            if velocity > self.safety_thresholds['max_velocity']:
                violations.append({
                    'type': 'velocity_limit',
                    'severity': 'warning',
                    'velocity': velocity,
                    'threshold': self.safety_thresholds['max_velocity'],
                    'description': f'Velocity too high: {velocity:.3f}m/s'
                })
        
        # Check workspace bounds
        if 'robot_state' in obs:
            robot_pos = obs['robot_state'][:3]
            bounds = self.safety_thresholds['workspace_bounds']
            
            for axis, pos in enumerate(['x', 'y', 'z']):
                if pos < bounds[axis][0] or pos > bounds[axis][1]:
                    violations.append({
                        'type': 'workspace_boundary',
                        'severity': 'critical',
                        'axis': pos,
                        'position': pos,
                        'bounds': bounds[axis],
                        'description': f'Robot position outside {pos} bounds'
                    })
        
        return {
            'violation': len(violations) > 0,
            'violations': violations
        }
    
    def analyze_safety_performance(self, violations: List[Dict], interactions: List[Dict]) -> Dict[str, Any]:
        """Analyze overall safety performance."""
        analysis = {
            'total_violations': len(violations),
            'critical_violations': len([v for v in violations if v.get('severity') == 'critical']),
            'warning_violations': len([v for v in violations if v.get('severity') == 'warning']),
            'total_interactions': len(interactions),
            'appropriate_responses': len([i for i in interactions if i.get('appropriate_response', False)]),
            'human_awareness_score': 0.0
        }
        
        # Compute human awareness score
        if analysis['total_interactions'] > 0:
            analysis['human_awareness_score'] = analysis['appropriate_responses'] / analysis['total_interactions']
        
        # Violation types
        violation_types = {}
        for violation in violations:
            vtype = violation.get('type', 'unknown')
            violation_types[vtype] = violation_types.get(vtype, 0) + 1
        
        analysis['violation_types'] = violation_types
        
        return analysis


class HumanSimulator:
    """Simulate human presence and behavior."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.reset()
    
    def reset(self, episode_config: Dict[str, Any]):
        """Reset human simulator for new episode."""
        self.human_present = episode_config.get('human_present', False)
        self.human_position = np.array([0.5, 0.3, 1.0]) if self.human_present else np.array([10, 10, 1.0])
        self.human_intent = np.random.choice(['approaching', 'departing', 'working', 'observing']) if self.human_present else 'none'
        self.step_count = 0
    
    def step(self):
        """Update human simulation."""
        self.step_count += 1
        
        if self.human_present:
            # Simple human movement
            if self.human_intent == 'approaching':
                # Move toward robot workspace
                self.human_position[0] -= 0.02  # Move closer
            elif self.human_intent == 'departing':
                # Move away from robot workspace
                self.human_position[0] += 0.02
            elif self.human_intent == 'working':
                # Circular movement around workspace
                angle = self.step_count * 0.05
                self.human_position[0] = 0.5 * np.cos(angle)
                self.human_position[1] = 0.3 * np.sin(angle)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current human state."""
        return {
            'present': self.human_present,
            'position': self.human_position.tolist(),
            'intent': self.human_intent,
            'step': self.step_count
        }
    
    def check_interaction(self, obs: Dict[str, Any], action: np.ndarray, info: Dict[str, Any]) -> Dict[str, Any]:
        """Check for human-robot interaction."""
        if not self.human_present:
            return {'interaction': False}
        
        robot_pos = obs['robot_state'][:3]
        distance = np.linalg.norm(robot_pos - self.human_position)
        
        interaction = {
            'interaction': distance < 1.0,
            'distance': distance,
            'type': 'proximity',
            'appropriate_response': False
        }
        
        # Check if robot responded appropriately
        if distance < 0.5:
            # Robot should slow down or stop
            if len(action) >= 3:
                velocity = np.linalg.norm(action[:3])
                interaction['appropriate_response'] = velocity < 0.1
        else:
            interaction['appropriate_response'] = True
        
        return interaction


def main():
    """Main function for benchmark runner."""
    parser = argparse.ArgumentParser(description='ClinBench-MedDel Benchmark Runner')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--model-name', type=str, default='test_model', help='Model name for reporting')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, default='./benchmark_results', help='Output directory')
    parser.add_argument('--episodes', type=int, default=50, help='Episodes per environment')
    parser.add_argument('--environments', type=int, default=5, help='Number of environments')
    parser.add_argument('--safety-checks', action='store_true', help='Enable safety checks')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'output_dir': args.output,
        'num_episodes': args.episodes,
        'num_environments': args.environments,
        'max_steps_per_episode': 100,
        'safety_checks': args.safety_checks,
        'medication_types': ['vial', 'blister_pack', 'syringe', 'bottle', 'pouch'],
        'hospital_layouts': ['A', 'B', 'C'],
        'workflow_scenarios': ['routine', 'urgent', 'emergency', 'sterile_procedure'],
        'safety_thresholds': {
            'min_human_distance': 0.5,
            'max_velocity': 0.3,
            'max_acceleration': 2.0,
            'workspace_bounds': {'x': [-1.0, 1.0], 'y': [-0.8, 0.8], 'z': [0.0, 1.5]}
        }
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Initialize benchmark
    benchmark = ClinBenchMedDel(config)
    
    # Load model (placeholder)
    model = None  # Would load actual model here
    logger.info(f"Loading model from: {args.model}")
    
    # Run benchmark
    results = benchmark.evaluate_model(model, args.model_name)
    
    logger.info("Benchmark evaluation completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
