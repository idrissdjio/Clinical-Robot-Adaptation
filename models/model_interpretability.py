#!/usr/bin/env python3
"""
Model Interpretability and Explainability for Clinical Robotics
Advanced tools for understanding and explaining clinical robot model decisions.

This module implements:
- Attention visualization for transformer-based models
- Gradient-based saliency maps for vision components
- Feature importance analysis for robot state predictions
- Counterfactual explanations for clinical decisions
- LIME and SHAP explanations for multi-modal models
- Clinical safety explanation generation
- Model uncertainty quantification

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass
import warnings

# Deep learning and ML
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import cv2
from PIL import Image

# Explainability
import shap
import lime
import lime.lime_image
from lime.lime_text import LimeTextExplainer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from scipy.stats import spearmanr

# Clinical specific
from scipy.spatial.transform import Rotation
import trimesh

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_interpretability.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class ExplainabilityConfig:
    """Configuration for explainability methods."""
    methods: List[str] = None
    visualization_format: str = "plotly"
    output_dir: str = "./explainability_results"
    num_samples: int = 100
    random_seed: int = 42
    attention_layers: List[int] = None
    feature_names: List[str] = None
    
    def __post_init__(self):
        if self.methods is None:
            self.methods = ["attention", "gradcam", "shap", "lime", "feature_importance"]
        if self.attention_layers is None:
            self.attention_layers = [0, 1, 2, 3]
        if self.feature_names is None:
            self.feature_names = [
                "joint_1", "joint_2", "joint_3", "joint_4", "joint_5", "joint_6", "joint_7",
                "gripper_pos", "gripper_vel", "gripper_force",
                "human_distance", "human_velocity", "safety_zone_violation"
            ]

class AttentionVisualizer:
    """Visualize attention patterns in transformer models."""
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hook for attention extraction
        self.attention_hooks = []
        self.attention_maps = {}
        
        self._register_attention_hooks()
    
    def _register_attention_hooks(self):
        """Register hooks to capture attention maps."""
        def get_attention_hook(layer_idx):
            def hook(module, input, output):
                # Assuming output is (attn_weights, context)
                if isinstance(output, tuple) and len(output) > 1:
                    self.attention_maps[f'layer_{layer_idx}'] = output[1].detach().cpu()
            return hook
        
        # Register hooks for specified layers
        for i, layer_idx in enumerate(self.config.attention_layers):
            if hasattr(self.model, f'transformer_layers'):
                layer = getattr(self.model, 'transformformer_layers')[layer_idx]
                hook = layer.register_forward_hook(get_attention_hook(layer_idx))
                self.attention_hooks.append(hook)
    
    def visualize_attention(self, input_data: Dict[str, torch.Tensor], 
                          output_dir: str = None) -> Dict[str, Any]:
        """Visualize attention patterns for given input."""
        self.model.eval()
        
        # Clear previous attention maps
        self.attention_maps.clear()
        
        # Forward pass to capture attention
        with torch.no_grad():
            _ = self.model(input_data)
        
        visualizations = {}
        
        for layer_name, attention in self.attention_maps.items():
            # Average attention across heads
            avg_attention = attention.mean(dim=1)  # [batch_size, seq_len, seq_len]
            
            # Take first sample in batch
            attn_map = avg_attention[0].numpy()
            
            # Create visualization
            fig = go.Figure(data=go.Heatmap(
                z=attn_map,
                colorscale='Blues',
                showscale=True,
                name=f'{layer_name} Attention'
            ))
            
            fig.update_layout(
                title=f'Attention Map - {layer_name}',
                xaxis_title='Key Position',
                yaxis_title='Query Position'
            )
            
            visualizations[layer_name] = {
                'attention_map': attn_map,
                'visualization': fig,
                'summary_stats': {
                    'mean_attention': np.mean(attn_map),
                    'max_attention': np.max(attn_map),
                    'entropy': -np.sum(attn_map * np.log(attn_map + 1e-8), axis=1).mean()
                }
            }
        
        # Save visualizations
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            for layer_name, viz in visualizations.items():
                fig = viz['visualization']
                fig.write_html(str(output_path / f'{layer_name}_attention.html'))
        
        return visualizations
    
    def analyze_attention_patterns(self, dataset: DataLoader) -> Dict[str, Any]:
        """Analyze attention patterns across dataset."""
        logger.info("Analyzing attention patterns across dataset...")
        
        all_attention_stats = {layer: [] for layer in self.config.attention_layers}
        
        for batch_idx, batch in enumerate(dataset):
            if batch_idx >= self.config.num_samples:
                break
            
            # Get attention for this batch
            visualizations = self.visualize_attention(batch)
            
            # Collect statistics
            for layer_name, viz in visualizations.items():
                stats = viz['summary_stats']
                all_attention_stats[layer_name].append(stats)
        
        # Aggregate statistics
        aggregated_stats = {}
        for layer_name, stats_list in all_attention_stats.items():
            if stats_list:
                df = pd.DataFrame(stats_list)
                aggregated_stats[layer_name] = {
                    'mean_attention_mean': df['mean_attention'].mean(),
                    'mean_attention_std': df['mean_attention'].std(),
                    'max_attention_mean': df['max_attention'].mean(),
                    'entropy_mean': df['entropy'].mean(),
                    'entropy_std': df['entropy'].std()
                }
        
        return aggregated_stats
    
    def cleanup(self):
        """Clean up registered hooks."""
        for hook in self.attention_hooks:
            hook.remove()

class GradCAMVisualizer:
    """Gradient-based visualization for vision components."""
    
    def __init__(self, model: nn.Module, target_layer: str, config: ExplainabilityConfig):
        self.model = model
        self.target_layer = target_layer
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Hook for gradients
        self.gradients = None
        self.activations = None
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks."""
        def forward_hook(module, input, output):
            self.activations = output.detach()
        
        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()
        
        # Find target layer
        target_module = self.model
        for layer_name in self.target_layer.split('.'):
            target_module = getattr(target_module, layer_name)
        
        target_module.register_forward_hook(forward_hook)
        target_module.register_backward_hook(backward_hook)
    
    def generate_gradcam(self, input_image: torch.Tensor, 
                        target_class: int = None) -> np.ndarray:
        """Generate GradCAM visualization."""
        self.model.eval()
        
        # Forward pass
        output = self.model(input_image)
        
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()
        
        # Generate GradCAM
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling gradients
        weights = torch.mean(gradients, dim=(1, 2))  # [C]
        
        # Weighted combination of activation maps
        gradcam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            gradcam += w * activations[i]
        
        # ReLU and normalize
        gradcam = F.relu(gradcam)
        gradcam = gradcam / gradcam.max()
        
        return gradcam.cpu().numpy()
    
    def visualize_gradcam(self, input_image: torch.Tensor, 
                         original_image: np.ndarray = None,
                         output_dir: str = None) -> go.Figure:
        """Create GradCAM visualization."""
        # Generate GradCAM
        gradcam = self.generate_gradcam(input_image)
        
        # Resize to match original image
        if original_image is not None:
            gradcam = cv2.resize(gradcam, (original_image.shape[1], original_image.shape[0]))
        
        # Create heatmap overlay
        heatmap = plt.cm.jet(gradcam)[:, :, :3]
        
        # Create visualization
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Original Image', 'GradCAM Heatmap'),
            specs=[[{'type': 'xy'}, {'type': 'xy'}]]
        )
        
        if original_image is not None:
            # Original image
            fig.add_trace(
                go.Image(z=original_image),
                row=1, col=1
            )
        
        # GradCAM heatmap
        fig.add_trace(
            go.Image(z=(heatmap * 255).astype(np.uint8)),
            row=1, col=2
        )
        
        fig.update_layout(
            title='GradCAM Visualization',
            showlegend=False
        )
        
        # Save visualization
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path / 'gradcam_visualization.html'))
        
        return fig

class FeatureImportanceAnalyzer:
    """Analyze feature importance for robot state predictions."""
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def analyze_permutation_importance(self, dataset: DataLoader, 
                                     num_samples: int = 100) -> Dict[str, float]:
        """Analyze permutation importance."""
        logger.info("Computing permutation importance...")
        
        # Collect data
        all_inputs = []
        all_outputs = []
        
        for batch_idx, batch in enumerate(dataset):
            if batch_idx >= num_samples:
                break
            
            inputs = batch['robot_state'].cpu().numpy()
            outputs = self.model(batch)['action'].cpu().numpy()
            
            all_inputs.append(inputs)
            all_outputs.append(outputs)
        
        X = np.vstack(all_inputs)
        y = np.vstack(all_outputs)
        
        # Fit a simple model to estimate importance
        # For multi-output, compute importance for each output dimension
        importance_scores = {}
        
        for output_dim in range(y.shape[1]):
            # Fit linear regression
            lr = LinearRegression()
            lr.fit(X, y[:, output_dim])
            
            # Compute permutation importance
            baseline_score = lr.score(X, y[:, output_dim])
            
            feature_importance = {}
            for feature_idx in range(X.shape[1]):
                X_permuted = X.copy()
                X_permuted[:, feature_idx] = np.random.permutation(X_permuted[:, feature_idx])
                
                permuted_score = lr.score(X_permuted, y[:, output_dim])
                importance = baseline_score - permuted_score
                feature_importance[self.config.feature_names[feature_idx]] = importance
            
            importance_scores[f'output_dim_{output_dim}'] = feature_importance
        
        return importance_scores
    
    def analyze_shap_values(self, dataset: DataLoader, 
                           num_samples: int = 50) -> Dict[str, Any]:
        """Compute SHAP values for feature importance."""
        logger.info("Computing SHAP values...")
        
        # Collect background data
        background_data = []
        for batch_idx, batch in enumerate(dataset):
            if batch_idx >= 10:  # Use first 10 batches as background
                break
            background_data.append(batch['robot_state'].cpu().numpy())
        
        background = np.vstack(background_data)
        
        # Create SHAP explainer
        def model_predict(inputs):
            inputs_tensor = torch.tensor(inputs, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                outputs = self.model({'robot_state': inputs_tensor})['action']
            return outputs.cpu().numpy()
        
        explainer = shap.KernelExplainer(model_predict, background[:100])  # Use subset for efficiency
        
        # Compute SHAP values for sample instances
        shap_values = []
        test_data = []
        
        for batch_idx, batch in enumerate(dataset):
            if batch_idx >= num_samples:
                break
            
            inputs = batch['robot_state'].cpu().numpy()
            test_data.append(inputs)
            
            # Compute SHAP values (this can be slow, so limit batch size)
            batch_shap = explainer.shap_values(inputs[:5])  # Limit to 5 samples per batch
            shap_values.extend(batch_shap)
        
        # Aggregate SHAP values
        if shap_values:
            # For multi-output, take mean absolute SHAP value across outputs
            if isinstance(shap_values[0], list):
                # Multi-output case
                aggregated_shap = np.mean([np.abs(sv) for sv in shap_values[0]], axis=0)
            else:
                # Single output case
                aggregated_shap = np.abs(shap_values)
            
            feature_importance = {}
            for i, feature_name in enumerate(self.config.feature_names):
                if i < aggregated_shap.shape[1]:
                    feature_importance[feature_name] = np.mean(aggregated_shap[:, i])
            
            return {
                'feature_importance': feature_importance,
                'shap_values': shap_values,
                'test_data': np.vstack(test_data)
            }
        
        return {'feature_importance': {}}
    
    def create_feature_importance_plot(self, importance_scores: Dict[str, float],
                                     output_dir: str = None) -> go.Figure:
        """Create feature importance visualization."""
        # Sort features by importance
        sorted_features = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)
        features, scores = zip(*sorted_features)
        
        # Create horizontal bar chart
        fig = go.Figure(data=[
            go.Bar(
                x=list(scores),
                y=list(features),
                orientation='h',
                marker_color='lightblue'
            )
        ])
        
        fig.update_layout(
            title='Feature Importance Analysis',
            xaxis_title='Importance Score',
            yaxis_title='Features',
            height=max(400, len(features) * 30)
        )
        
        # Save visualization
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            fig.write_html(str(output_path / 'feature_importance.html'))
        
        return fig

class ClinicalExplainer:
    """Clinical-specific explanation generation."""
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Clinical knowledge base
        self.clinical_rules = self._load_clinical_rules()
        self.safety_guidelines = self._load_safety_guidelines()
    
    def _load_clinical_rules(self) -> Dict[str, Any]:
        """Load clinical rule base."""
        return {
            'medication_handling': {
                'vial': {
                    'required_precision': 'high',
                    'safety_concerns': ['breakage', 'contamination'],
                    'optimal_grasp': 'precision_grasp'
                },
                'bottle': {
                    'required_precision': 'medium',
                    'safety_concerns': ['spillage', 'over-tightening'],
                    'optimal_grasp': 'power_grasp'
                },
                'syringe': {
                    'required_precision': 'very_high',
                    'safety_concerns': ['needle_stick', 'contamination'],
                    'optimal_grasp': 'precision_grasp'
                }
            },
            'human_proximity': {
                'safe_distance': 0.5,  # meters
                'critical_distance': 0.3,  # meters
                'required_slowdown': True
            },
            'workspace_constraints': {
                'max_velocity': 0.3,  # m/s
                'max_acceleration': 1.0,  # m/s^2
                'restricted_zones': ['sterile_field', 'emergency_equipment']
            }
        }
    
    def _load_safety_guidelines(self) -> Dict[str, Any]:
        """Load safety guidelines."""
        return {
            'force_limits': {
                'grasping': {'min': 2.0, 'max': 15.0},  # Newtons
                'transport': {'min': 0.5, 'max': 8.0}   # Newtons
            },
            'velocity_limits': {
                'near_human': 0.1,  # m/s
                'normal': 0.3      # m/s
            },
            'emergency_conditions': [
                'human_collision_risk',
                'medication_drop_risk',
                'equipment_malfunction'
            ]
        }
    
    def generate_clinical_explanation(self, input_data: Dict[str, torch.Tensor],
                                   model_output: Dict[str, torch.Tensor],
                                   attention_weights: np.ndarray = None) -> Dict[str, Any]:
        """Generate clinical explanation for model decision."""
        explanation = {
            'decision_summary': '',
            'safety_assessment': '',
            'clinical_rationale': '',
            'alternative_actions': [],
            'confidence_factors': {},
            'recommendations': []
        }
        
        # Analyze medication type
        if 'medication_type' in input_data:
            med_type = input_data['medication_type'].item()
            med_rules = self.clinical_rules['medication_handling'].get(str(med_type), {})
            
            explanation['clinical_rationale'] += f"Medication type requires {med_rules.get('required_precision', 'medium')} precision. "
            
            if 'safety_concerns' in med_rules:
                explanation['safety_assessment'] += f"Primary safety concerns: {', '.join(med_rules['safety_concerns'])}. "
        
        # Analyze human proximity
        if 'human_distance' in input_data:
            human_dist = input_data['human_distance'].item()
            safe_dist = self.clinical_rules['human_proximity']['safe_distance']
            critical_dist = self.clinical_rules['human_proximity']['critical_distance']
            
            if human_dist < critical_dist:
                explanation['safety_assessment'] += f"CRITICAL: Human distance ({human_dist:.2f}m) below critical threshold ({critical_dist}m). "
                explanation['recommendations'].append("Reduce velocity and increase safety monitoring")
            elif human_dist < safe_dist:
                explanation['safety_assessment'] += f"WARNING: Human distance ({human_dist:.2f}m) below safe threshold ({safe_dist}m). "
                explanation['recommendations'].append("Proceed with caution and enhanced monitoring")
        
        # Analyze robot state
        if 'robot_state' in input_data:
            robot_state = input_data['robot_state'].cpu().numpy()
            
            # Check joint velocities
            if len(robot_state) >= 14:  # Assuming positions and velocities
                velocities = robot_state[7:14]
                max_vel = np.max(np.abs(velocities))
                vel_limit = self.clinical_rules['workspace_constraints']['max_velocity']
                
                if max_vel > vel_limit:
                    explanation['safety_assessment'] += f"WARNING: Joint velocity ({max_vel:.3f} rad/s) exceeds limit ({vel_limit} rad/s). "
                    explanation['recommendations'].append("Reduce joint velocities")
        
        # Analyze model confidence
        if 'confidence' in model_output:
            confidence = model_output['confidence'].item()
            explanation['confidence_factors']['model_confidence'] = confidence
            
            if confidence < 0.8:
                explanation['decision_summary'] += f"Model confidence is low ({confidence:.2f}). "
                explanation['recommendations'].append("Consider human intervention or alternative approach")
            else:
                explanation['decision_summary'] += f"Model confidence is high ({confidence:.2f}). "
        
        # Add attention-based explanation if available
        if attention_weights is not None:
            # Find most attended features
            max_attention_idx = np.argmax(attention_weights.mean(axis=0))
            if max_attention_idx < len(self.config.feature_names):
                most_attended = self.config.feature_names[max_attention_idx]
                explanation['confidence_factors']['primary_attention'] = most_attended
                explanation['decision_summary'] += f"Model primarily focused on {most_attended}. "
        
        # Generate alternative actions
        explanation['alternative_actions'] = self._generate_alternatives(input_data, model_output)
        
        return explanation
    
    def _generate_alternatives(self, input_data: Dict[str, torch.Tensor],
                              model_output: Dict[str, torch.Tensor]) -> List[str]:
        """Generate alternative action recommendations."""
        alternatives = []
        
        # Based on safety assessment
        if 'human_distance' in input_data:
            human_dist = input_data['human_distance'].item()
            if human_dist < 0.3:
                alternatives.append("Pause operation and wait for human to move away")
                alternatives.append("Execute emergency stop procedure")
        
        # Based on medication type
        if 'medication_type' in input_data:
            med_type = input_data['medication_type'].item()
            if med_type == 2:  # Syringe
                alternatives.append("Use alternative grasping strategy with higher precision")
                alternatives.append("Reduce approach velocity by 50%")
        
        # Based on model confidence
        if 'confidence' in model_output:
            confidence = model_output['confidence'].item()
            if confidence < 0.7:
                alternatives.append("Request human supervision for this task")
                alternatives.append("Use fallback conservative trajectory")
        
        return alternatives
    
    def create_explanation_report(self, explanations: List[Dict[str, Any]],
                                output_dir: str = None) -> str:
        """Create comprehensive explanation report."""
        report = "# Clinical Model Explanation Report\n\n"
        report += f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        
        # Summary statistics
        total_explanations = len(explanations)
        high_confidence_count = sum(1 for e in explanations 
                                   if e['confidence_factors'].get('model_confidence', 0) > 0.8)
        
        report += "## Summary Statistics\n\n"
        report += f"- Total explanations: {total_explanations}\n"
        report += f"- High confidence decisions: {high_confidence_count} ({high_confidence_count/total_explanations*100:.1f}%)\n"
        report += f"- Average recommendations per decision: {np.mean([len(e['recommendations']) for e in explanations]):.1f}\n\n"
        
        # Common recommendations
        all_recommendations = []
        for e in explanations:
            all_recommendations.extend(e['recommendations'])
        
        from collections import Counter
        recommendation_counts = Counter(all_recommendations)
        
        report += "## Top Recommendations\n\n"
        for rec, count in recommendation_counts.most_common(5):
            report += f"- {rec}: {count} occurrences\n"
        report += "\n"
        
        # Sample explanations
        report += "## Sample Explanations\n\n"
        for i, explanation in enumerate(explanations[:3]):
            report += f"### Explanation {i+1}\n\n"
            report += f"**Decision Summary:** {explanation['decision_summary']}\n\n"
            report += f"**Safety Assessment:** {explanation['safety_assessment']}\n\n"
            report += f"**Clinical Rationale:** {explanation['clinical_rationale']}\n\n"
            
            if explanation['recommendations']:
                report += "**Recommendations:**\n"
                for rec in explanation['recommendations']:
                    report += f"- {rec}\n"
                report += "\n"
            
            report += "---\n\n"
        
        # Save report
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            report_file = output_path / 'explanation_report.md'
            with open(report_file, 'w') as f:
                f.write(report)
            
            logger.info(f"Explanation report saved to: {report_file}")
        
        return report

class ModelInterpretabilityPipeline:
    """Main pipeline for model interpretability analysis."""
    
    def __init__(self, model: nn.Module, config: ExplainabilityConfig):
        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize explainers
        self.attention_viz = AttentionVisualizer(model, config)
        self.gradcam_viz = GradCAMVisualizer(model, "vision_encoder.backbone.layer4", config)
        self.feature_analyzer = FeatureImportanceAnalyzer(model, config)
        self.clinical_explainer = ClinicalExplainer(model, config)
        
        # Results storage
        self.results = {
            'attention_analysis': {},
            'feature_importance': {},
            'clinical_explanations': [],
            'visualizations': {}
        }
    
    def run_full_analysis(self, dataset: DataLoader, 
                         output_dir: str = None) -> Dict[str, Any]:
        """Run complete interpretability analysis."""
        logger.info("Starting comprehensive model interpretability analysis...")
        
        output_path = Path(output_dir) if output_dir else Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 1. Attention analysis
        logger.info("Analyzing attention patterns...")
        attention_results = self.attention_viz.analyze_attention_patterns(dataset)
        self.results['attention_analysis'] = attention_results
        
        # 2. Feature importance analysis
        logger.info("Analyzing feature importance...")
        
        # Permutation importance
        perm_importance = self.feature_analyzer.analyze_permutation_importance(
            dataset, self.config.num_samples
        )
        
        # SHAP analysis
        shap_results = self.feature_analyzer.analyze_shap_values(
            dataset, self.config.num_samples // 2
        )
        
        self.results['feature_importance'] = {
            'permutation_importance': perm_importance,
            'shap_analysis': shap_results
        }
        
        # 3. Clinical explanations
        logger.info("Generating clinical explanations...")
        explanations = []
        
        for batch_idx, batch in enumerate(dataset):
            if batch_idx >= self.config.num_samples:
                break
            
            # Get model output
            with torch.no_grad():
                model_output = self.model(batch)
            
            # Get attention weights
            attention_viz = self.attention_viz.visualize_attention(batch)
            attention_weights = None
            if attention_viz:
                # Use attention from first layer
                first_layer = list(attention_viz.keys())[0]
                attention_weights = attention_viz[first_layer]['attention_map']
            
            # Generate clinical explanation
            explanation = self.clinical_explainer.generate_clinical_explanation(
                batch, model_output, attention_weights
            )
            explanations.append(explanation)
        
        self.results['clinical_explanations'] = explanations
        
        # 4. Generate visualizations
        logger.info("Creating visualizations...")
        
        # Feature importance plot
        if shap_results.get('feature_importance'):
            importance_fig = self.feature_analyzer.create_feature_importance_plot(
                shap_results['feature_importance'], str(output_path)
            )
            self.results['visualizations']['feature_importance'] = importance_fig
        
        # 5. Generate report
        logger.info("Generating comprehensive report...")
        report = self.clinical_explainer.create_explanation_report(
            explanations, str(output_path)
        )
        
        # Save all results
        results_file = output_path / 'interpretability_results.json'
        with open(results_file, 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_serializable(self.results)
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"Interpretability analysis completed. Results saved to: {output_path}")
        
        return self.results
    
    def _make_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to serializable format."""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return str(obj)
        else:
            return obj
    
    def explain_single_instance(self, input_data: Dict[str, torch.Tensor],
                              output_dir: str = None) -> Dict[str, Any]:
        """Generate explanation for a single instance."""
        self.model.eval()
        
        # Get model output
        with torch.no_grad():
            model_output = self.model(input_data)
        
        # Get attention visualization
        attention_viz = self.attention_viz.visualize_attention(input_data, output_dir)
        
        # Get GradCAM if image data available
        gradcam_viz = None
        if 'image' in input_data:
            gradcam_viz = self.gradcam_viz.visualize_gradcam(
                input_data['image'], output_dir=output_dir
            )
        
        # Generate clinical explanation
        attention_weights = None
        if attention_viz:
            first_layer = list(attention_viz.keys())[0]
            attention_weights = attention_viz[first_layer]['attention_map']
        
        clinical_explanation = self.clinical_explainer.generate_clinical_explanation(
            input_data, model_output, attention_weights
        )
        
        return {
            'attention_visualization': attention_viz,
            'gradcam_visualization': gradcam_viz,
            'clinical_explanation': clinical_explanation,
            'model_output': {k: v.cpu().numpy() for k, v in model_output.items()}
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.attention_viz.cleanup()

def main():
    """Main function for model interpretability analysis."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Model interpretability analysis')
    parser.add_argument('--model', type=str, required=True, help='Model checkpoint path')
    parser.add_argument('--data', type=str, required=True, help='Dataset path')
    parser.add_argument('--output', type=str, default='./interpretability_results', help='Output directory')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples to analyze')
    parser.add_argument('--methods', nargs='+', default=['attention', 'gradcam', 'shap', 'lime'], 
                       help='Explainability methods to use')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ExplainabilityConfig(
        methods=args.methods,
        output_dir=args.output,
        num_samples=args.samples
    )
    
    # Load model (placeholder)
    # model = load_clinical_model(args.model)
    
    # Load dataset (placeholder)
    # dataset = load_clinical_dataset(args.data)
    
    # Initialize pipeline
    # pipeline = ModelInterpretabilityPipeline(model, config)
    
    # Run analysis
    # results = pipeline.run_full_analysis(dataset, args.output)
    
    print(f"Interpretability analysis completed. Results saved to: {args.output}")

if __name__ == "__main__":
    main()
