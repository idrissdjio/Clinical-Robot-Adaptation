#!/usr/bin/env python3
"""
Test suite for Model Interpretability and Explainability
Comprehensive unit tests for clinical robotics model explainability.

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import pytest
import torch
import numpy as np
import tempfile
import shutil
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Import modules to test
import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.model_interpretability import (
    ExplainabilityConfig,
    AttentionVisualizer,
    GradCAMVisualizer,
    FeatureImportanceAnalyzer,
    ClinicalExplainer,
    ModelInterpretabilityPipeline
)

class TestExplainabilityConfig:
    """Test cases for ExplainabilityConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = ExplainabilityConfig()
        
        assert "attention" in config.methods
        assert "gradcam" in config.methods
        assert "shap" in config.methods
        assert config.visualization_format == "plotly"
        assert config.output_dir == "./explainability_results"
        assert config.num_samples == 100
        assert config.random_seed == 42
        assert config.attention_layers == [0, 1, 2, 3]
        assert len(config.feature_names) > 0
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = ExplainabilityConfig(
            methods=["attention", "shap"],
            num_samples=50,
            visualization_format="matplotlib"
        )
        
        assert config.methods == ["attention", "shap"]
        assert config.num_samples == 50
        assert config.visualization_format == "matplotlib"

class TestAttentionVisualizer:
    """Test cases for AttentionVisualizer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with transformer layers."""
        model = Mock()
        
        # Mock transformer layers
        transformer_layers = []
        for i in range(4):
            layer = Mock()
            layer.register_forward_hook = Mock()
            transformer_layers.append(layer)
        
        model.transformformer_layers = transformer_layers
        return model
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExplainabilityConfig(
            attention_layers=[0, 1],
            num_samples=10
        )
    
    @pytest.fixture
    def attention_viz(self, mock_model, config):
        """Create attention visualizer."""
        return AttentionVisualizer(mock_model, config)
    
    def test_initialization(self, attention_viz, mock_model, config):
        """Test attention visualizer initialization."""
        assert attention_viz.model == mock_model
        assert attention_viz.config == config
        assert len(attention_viz.attention_hooks) == len(config.attention_layers)
    
    def test_visualize_attention(self, attention_viz):
        """Test attention visualization."""
        # Mock model forward pass
        mock_output = (torch.randn(2, 4, 10, 10), torch.randn(2, 8, 10, 10))
        attention_viz.model.return_value = mock_output
        
        # Create mock input
        input_data = {
            'image': torch.randn(2, 3, 224, 224),
            'robot_state': torch.randn(2, 14)
        }
        
        visualizations = attention_viz.visualize_attention(input_data)
        
        assert len(visualizations) == len(attention_viz.config.attention_layers)
        
        for layer_name, viz in visualizations.items():
            assert 'attention_map' in viz
            assert 'visualization' in viz
            assert 'summary_stats' in viz
            
            assert isinstance(viz['attention_map'], np.ndarray)
            assert 'mean_attention' in viz['summary_stats']
            assert 'max_attention' in viz['summary_stats']
            assert 'entropy' in viz['summary_stats']
    
    def test_analyze_attention_patterns(self, attention_viz):
        """Test attention pattern analysis."""
        # Mock data loader
        mock_batch = {
            'image': torch.randn(2, 3, 224, 224),
            'robot_state': torch.randn(2, 14)
        }
        
        mock_loader = [mock_batch] * 5  # 5 batches
        
        # Mock visualize_attention to return test data
        test_viz = {
            'layer_0': {
                'summary_stats': {
                    'mean_attention': 0.5,
                    'max_attention': 0.9,
                    'entropy': 2.0
                }
            }
        }
        
        with patch.object(attention_viz, 'visualize_attention', return_value=test_viz):
            patterns = attention_viz.analyze_attention_patterns(mock_loader)
        
        assert len(patterns) == len(attention_viz.config.attention_layers)
        
        for layer_name, stats in patterns.items():
            assert 'mean_attention_mean' in stats
            assert 'mean_attention_std' in stats
            assert 'entropy_mean' in stats
    
    def test_cleanup(self, attention_viz):
        """Test hook cleanup."""
        # Verify hooks are registered
        initial_hooks = len(attention_viz.attention_hooks)
        assert initial_hooks > 0
        
        # Cleanup
        attention_viz.cleanup()
        
        # All hooks should be removed
        assert len(attention_viz.attention_hooks) == 0

class TestGradCAMVisualizer:
    """Test cases for GradCAMVisualizer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model with target layer."""
        model = Mock()
        
        # Mock target layer
        target_layer = Mock()
        target_layer.register_forward_hook = Mock()
        target_layer.register_backward_hook = Mock()
        
        # Set up nested attribute access
        model.vision_encoder.backbone.layer4 = target_layer
        
        return model
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExplainabilityConfig()
    
    @pytest.fixture
    def gradcam_viz(self, mock_model, config):
        """Create GradCAM visualizer."""
        return GradCAMVisualizer(mock_model, "vision_encoder.backbone.layer4", config)
    
    def test_initialization(self, gradcam_viz, mock_model):
        """Test GradCAM visualizer initialization."""
        assert gradcam_viz.model == mock_model
        assert gradcam_viz.target_layer == "vision_encoder.backbone.layer4"
        assert gradcam_viz.gradients is None
        assert gradcam_viz.activations is None
    
    def test_generate_gradcam(self, gradcam_viz):
        """Test GradCAM generation."""
        # Mock model output
        mock_output = torch.randn(1, 10)  # 10 classes
        gradcam_viz.model.return_value = mock_output
        
        # Mock gradients and activations
        gradcam_viz.gradients = [torch.randn(64, 7, 7)]
        gradcam_viz.activations = [torch.randn(64, 7, 7)]
        
        input_image = torch.randn(1, 3, 224, 224)
        
        gradcam = gradcam_viz.generate_gradcam(input_image)
        
        assert isinstance(gradcam, np.ndarray)
        assert gradcam.shape == (7, 7)
        assert 0 <= gradcam.min() <= gradcam.max() <= 1
    
    def test_visualize_gradcam(self, gradcam_viz):
        """Test GradCAM visualization."""
        # Mock generate_gradcam
        test_gradcam = np.random.rand(7, 7)
        
        with patch.object(gradcam_viz, 'generate_gradcam', return_value=test_gradcam):
            original_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            input_image = torch.randn(1, 3, 224, 224)
            
            fig = gradcam_viz.visualize_gradcam(input_image, original_image)
            
            assert fig is not None
            # Check that figure has the expected structure
            assert hasattr(fig, 'data')  # Plotly figure attribute

class TestFeatureImportanceAnalyzer:
    """Test cases for FeatureImportanceAnalyzer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        mock_output = {'action': torch.randn(2, 7)}  # 7-dimensional action
        model.return_value = mock_output
        return model
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExplainabilityConfig(
            num_samples=10,
            feature_names=['joint_1', 'joint_2', 'joint_3', 'gripper_pos', 'human_distance']
        )
    
    @pytest.fixture
    def analyzer(self, mock_model, config):
        """Create feature importance analyzer."""
        return FeatureImportanceAnalyzer(mock_model, config)
    
    def test_initialization(self, analyzer, mock_model, config):
        """Test analyzer initialization."""
        assert analyzer.model == mock_model
        assert analyzer.config == config
    
    def test_analyze_permutation_importance(self, analyzer):
        """Test permutation importance analysis."""
        # Mock data loader
        mock_batch = {
            'robot_state': torch.randn(4, 14),  # 4 samples, 14 features
            'image': torch.randn(4, 3, 224, 224)
        }
        
        mock_loader = [mock_batch] * 3  # 3 batches
        
        importance = analyzer.analyze_permutation_importance(mock_loader, num_samples=3)
        
        assert isinstance(importance, dict)
        
        # Check that we have importance for each output dimension
        for output_dim in range(7):  # 7 action dimensions
            assert f'output_dim_{output_dim}' in importance
            
            # Check that each output has feature importance
            feature_imp = importance[f'output_dim_{output_dim}']
            assert isinstance(feature_imp, dict)
            
            # Check that we have importance for configured features
            for feature_name in analyzer.config.feature_names[:5]:  # Only first 5 in this test
                if feature_name in feature_imp:
                    assert isinstance(feature_imp[feature_name], (int, float))
    
    def test_analyze_shap_values(self, analyzer):
        """Test SHAP value analysis."""
        # Mock data loader
        mock_batch = {
            'robot_state': torch.randn(4, 14),
            'image': torch.randn(4, 3, 224, 224)
        }
        
        mock_loader = [mock_batch] * 2  # 2 batches
        
        # Mock SHAP explainer
        with patch('models.model_interpretability.shap.KernelExplainer') as mock_shap:
            mock_explainer = Mock()
            mock_shap.return_value = mock_explainer
            
            # Mock SHAP values
            mock_shap_values = [np.random.randn(4, 14) for _ in range(7)]  # 7 outputs
            mock_explainer.shap_values.return_value = mock_shap_values
            
            results = analyzer.analyze_shap_values(mock_loader, num_samples=2)
        
        assert 'feature_importance' in results
        assert 'shap_values' in results
        assert 'test_data' in results
        
        # Check feature importance
        feature_imp = results['feature_importance']
        assert isinstance(feature_imp, dict)
        
        # Should have importance for at least some features
        assert len(feature_imp) > 0
    
    def test_create_feature_importance_plot(self, analyzer):
        """Test feature importance plot creation."""
        # Create test importance scores
        importance_scores = {
            'joint_1': 0.8,
            'joint_2': 0.6,
            'gripper_pos': 0.4,
            'human_distance': 0.9
        }
        
        fig = analyzer.create_feature_importance_plot(importance_scores)
        
        assert fig is not None
        assert hasattr(fig, 'data')  # Plotly figure attribute

class TestClinicalExplainer:
    """Test cases for ClinicalExplainer."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        mock_output = {
            'action': torch.randn(1, 7),
            'confidence': torch.tensor([0.85])
        }
        model.return_value = mock_output
        return model
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExplainabilityConfig()
    
    @pytest.fixture
    def explainer(self, mock_model, config):
        """Create clinical explainer."""
        return ClinicalExplainer(mock_model, config)
    
    def test_initialization(self, explainer, config):
        """Test explainer initialization."""
        assert explainer.model is not None
        assert explainer.config == config
        assert 'medication_handling' in explainer.clinical_rules
        assert 'human_proximity' in explainer.clinical_rules
        assert 'force_limits' in explainer.safety_guidelines
    
    def test_load_clinical_rules(self, explainer):
        """Test clinical rules loading."""
        rules = explainer.clinical_rules
        
        assert 'medication_handling' in rules
        assert 'vial' in rules['medication_handling']
        assert 'bottle' in rules['medication_handling']
        assert 'syringe' in rules['medication_handling']
        
        # Check medication rules structure
        vial_rules = rules['medication_handling']['vial']
        assert 'required_precision' in vial_rules
        assert 'safety_concerns' in vial_rules
        assert 'optimal_grasp' in vial_rules
    
    def test_load_safety_guidelines(self, explainer):
        """Test safety guidelines loading."""
        guidelines = explainer.safety_guidelines
        
        assert 'force_limits' in guidelines
        assert 'velocity_limits' in guidelines
        assert 'emergency_conditions' in guidelines
        
        # Check force limits structure
        force_limits = guidelines['force_limits']
        assert 'grasping' in force_limits
        assert 'transport' in force_limits
        assert 'min' in force_limits['grasping']
        assert 'max' in force_limits['grasping']
    
    def test_generate_clinical_explanation(self, explainer):
        """Test clinical explanation generation."""
        # Create input data
        input_data = {
            'medication_type': torch.tensor([0]),  # vial
            'human_distance': torch.tensor([0.4]),  # Below safe distance
            'robot_state': torch.randn(1, 14)
        }
        
        # Create model output
        model_output = {
            'action': torch.randn(1, 7),
            'confidence': torch.tensor([0.75])  # Low confidence
        }
        
        # Create attention weights
        attention_weights = np.random.rand(10, 10)
        
        explanation = explainer.generate_clinical_explanation(
            input_data, model_output, attention_weights
        )
        
        # Check explanation structure
        assert 'decision_summary' in explanation
        assert 'safety_assessment' in explanation
        assert 'clinical_rationale' in explanation
        assert 'alternative_actions' in explanation
        assert 'confidence_factors' in explanation
        assert 'recommendations' in explanation
        
        # Check content
        assert 'vial' in explanation['clinical_rationale']
        assert 'human distance' in explanation['safety_assessment']
        assert 'model confidence' in explanation['confidence_factors']
        assert len(explanation['alternative_actions']) > 0
        assert len(explanation['recommendations']) > 0
    
    def test_generate_alternatives(self, explainer):
        """Test alternative action generation."""
        # Test with critical human distance
        input_data = {
            'human_distance': torch.tensor([0.2]),  # Critical
            'medication_type': torch.tensor([2]),  # Syringe
            'robot_state': torch.randn(1, 14)
        }
        
        model_output = {
            'confidence': torch.tensor([0.6])  # Low confidence
        }
        
        alternatives = explainer._generate_alternatives(input_data, model_output)
        
        assert isinstance(alternatives, list)
        assert len(alternatives) > 0
        
        # Should have human-related alternatives
        human_alts = [alt for alt in alternatives if 'human' in alt.lower()]
        assert len(human_alts) > 0
    
    def test_create_explanation_report(self, explainer):
        """Test explanation report generation."""
        # Create sample explanations
        explanations = [
            {
                'decision_summary': 'Test summary 1',
                'safety_assessment': 'Test safety 1',
                'clinical_rationale': 'Test rationale 1',
                'confidence_factors': {'model_confidence': 0.8},
                'recommendations': ['Recommendation 1', 'Recommendation 2']
            },
            {
                'decision_summary': 'Test summary 2',
                'safety_assessment': 'Test safety 2',
                'clinical_rationale': 'Test rationale 2',
                'confidence_factors': {'model_confidence': 0.9},
                'recommendations': ['Recommendation 3']
            }
        ]
        
        report = explainer.create_explanation_report(explanations)
        
        assert isinstance(report, str)
        assert 'Clinical Model Explanation Report' in report
        assert 'Summary Statistics' in report
        assert 'Top Recommendations' in report
        assert 'Sample Explanations' in report
        
        # Check statistics
        assert 'Total explanations: 2' in report
        assert 'High confidence decisions: 1' in report  # One has >0.8 confidence
    
    def test_critical_distance_explanation(self, explainer):
        """Test explanation generation for critical human distance."""
        input_data = {
            'human_distance': torch.tensor([0.25]),  # Critical distance
            'robot_state': torch.randn(1, 14)
        }
        
        model_output = {'action': torch.randn(1, 7)}
        
        explanation = explainer.generate_clinical_explanation(input_data, model_output)
        
        assert 'CRITICAL' in explanation['safety_assessment']
        assert 'emergency stop' in str(explanation['alternative_actions']).lower()
    
    def test_medication_specific_explanation(self, explainer):
        """Test medication-specific explanations."""
        # Test syringe (high precision required)
        input_data = {
            'medication_type': torch.tensor([2]),  # Syringe
            'robot_state': torch.randn(1, 14)
        }
        
        model_output = {'action': torch.randn(1, 7)}
        
        explanation = explainer.generate_clinical_explanation(input_data, model_output)
        
        assert 'very_high precision' in explanation['clinical_rationale']
        assert 'needle stick' in explanation['safety_assessment']

class TestModelInterpretabilityPipeline:
    """Test cases for ModelInterpretabilityPipeline."""
    
    @pytest.fixture
    def mock_model(self):
        """Create mock model."""
        model = Mock()
        
        # Mock transformer layers
        transformer_layers = []
        for i in range(4):
            layer = Mock()
            layer.register_forward_hook = Mock()
            transformer_layers.append(layer)
        model.transformformer_layers = transformer_layers
        
        # Mock vision encoder
        model.vision_encoder = Mock()
        model.vision_encoder.backbone = Mock()
        model.vision_encoder.backbone.layer4 = Mock()
        model.vision_encoder.backbone.layer4.register_forward_hook = Mock()
        model.vision_encoder.backbone.layer4.register_backward_hook = Mock()
        
        # Mock output
        model.return_value = {
            'action': torch.randn(2, 7),
            'confidence': torch.tensor([0.8, 0.9])
        }
        
        return model
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return ExplainabilityConfig(
            num_samples=5,
            methods=['attention', 'feature_importance']
        )
    
    @pytest.fixture
    def pipeline(self, mock_model, config):
        """Create interpretability pipeline."""
        return ModelInterpretabilityPipeline(mock_model, config)
    
    def test_initialization(self, pipeline, mock_model, config):
        """Test pipeline initialization."""
        assert pipeline.model == mock_model
        assert pipeline.config == config
        assert pipeline.attention_viz is not None
        assert pipeline.gradcam_viz is not None
        assert pipeline.feature_analyzer is not None
        assert pipeline.clinical_explainer is not None
        
        # Check results structure
        assert 'attention_analysis' in pipeline.results
        assert 'feature_importance' in pipeline.results
        assert 'clinical_explanations' in pipeline.results
        assert 'visualizations' in pipeline.results
    
    def test_explain_single_instance(self, pipeline):
        """Test single instance explanation."""
        # Create input data
        input_data = {
            'image': torch.randn(1, 3, 224, 224),
            'robot_state': torch.randn(1, 14),
            'medication_type': torch.tensor([0]),
            'human_distance': torch.tensor([0.6])
        }
        
        # Mock attention visualization
        test_attention = {
            'layer_0': {
                'attention_map': np.random.rand(10, 10)
            }
        }
        
        with patch.object(pipeline.attention_viz, 'visualize_attention', return_value=test_attention):
            with patch.object(pipeline.gradcam_viz, 'visualize_gradcam') as mock_gradcam:
                mock_gradcam.return_value = Mock()  # Mock plotly figure
                
                result = pipeline.explain_single_instance(input_data)
        
        # Check result structure
        assert 'attention_visualization' in result
        assert 'gradcam_visualization' in result
        assert 'clinical_explanation' in result
        assert 'model_output' in result
        
        # Check clinical explanation
        explanation = result['clinical_explanation']
        assert 'decision_summary' in explanation
        assert 'safety_assessment' in explanation
        assert 'recommendations' in explanation
    
    def test_cleanup(self, pipeline):
        """Test pipeline cleanup."""
        # Verify attention hooks are registered
        initial_hooks = len(pipeline.attention_viz.attention_hooks)
        assert initial_hooks > 0
        
        # Cleanup
        pipeline.cleanup()
        
        # All hooks should be removed
        assert len(pipeline.attention_viz.attention_hooks) == 0

class TestIntegration:
    """Integration tests for the interpretability pipeline."""
    
    def test_end_to_end_explanation(self):
        """Test end-to-end explanation generation."""
        # Create mock model
        model = Mock()
        model.return_value = {
            'action': torch.randn(1, 7),
            'confidence': torch.tensor([0.85])
        }
        
        config = ExplainabilityConfig(
            num_samples=2,
            methods=['attention', 'feature_importance']
        )
        
        pipeline = ModelInterpretabilityPipeline(model, config)
        
        # Create test input
        input_data = {
            'image': torch.randn(1, 3, 224, 224),
            'robot_state': torch.randn(1, 14),
            'medication_type': torch.tensor([1]),
            'human_distance': torch.tensor([0.4])
        }
        
        # Mock attention visualization
        test_attention = {
            'layer_0': {
                'attention_map': np.random.rand(10, 10),
                'summary_stats': {'mean_attention': 0.5, 'entropy': 2.0}
            }
        }
        
        with patch.object(pipeline.attention_viz, 'visualize_attention', return_value=test_attention):
            result = pipeline.explain_single_instance(input_data)
        
        # Verify all components worked together
        assert 'clinical_explanation' in result
        assert 'attention_visualization' in result
        assert 'model_output' in result
        
        # Check that clinical explanation is comprehensive
        explanation = result['clinical_explanation']
        assert len(explanation['decision_summary']) > 0
        assert len(explanation['safety_assessment']) > 0
        assert len(explanation['recommendations']) > 0

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_missing_model_components(self):
        """Test handling of missing model components."""
        model = Mock()
        # Don't set up transformer layers
        
        config = ExplainabilityConfig()
        
        # Should handle gracefully
        attention_viz = AttentionVisualizer(model, config)
        assert attention_viz.attention_hooks == []  # No hooks registered
    
    def test_invalid_input_data(self):
        """Test handling of invalid input data."""
        model = Mock()
        model.return_value = {'action': torch.randn(1, 7)}
        
        config = ExplainabilityConfig()
        explainer = ClinicalExplainer(model, config)
        
        # Test with empty input
        input_data = {}
        model_output = {'action': torch.randn(1, 7)}
        
        explanation = explainer.generate_clinical_explanation(input_data, model_output)
        
        # Should still generate explanation structure
        assert 'decision_summary' in explanation
        assert 'safety_assessment' in explanation
        assert 'recommendations' in explanation
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        model = Mock()
        model.return_value = {'confidence': torch.tensor([0.01])}  # Very low confidence
        
        config = ExplainabilityConfig()
        explainer = ClinicalExplainer(model, config)
        
        # Test with extreme human distance
        input_data = {
            'human_distance': torch.tensor([0.01]),  # Very close
            'robot_state': torch.randn(1, 14) * 1000  # Extreme values
        }
        
        model_output = {'action': torch.randn(1, 7)}
        
        explanation = explainer.generate_clinical_explanation(input_data, model_output)
        
        # Should handle extreme values and generate appropriate warnings
        assert 'CRITICAL' in explanation['safety_assessment']
        assert 'low confidence' in explanation['decision_summary'].lower()
    
    def test_empty_attention_weights(self):
        """Test handling of empty attention weights."""
        model = Mock()
        model.return_value = {'action': torch.randn(1, 7)}
        
        config = ExplainabilityConfig()
        explainer = ClinicalExplainer(model, config)
        
        input_data = {'robot_state': torch.randn(1, 14)}
        model_output = {'action': torch.randn(1, 7)}
        
        # Test with None attention weights
        explanation = explainer.generate_clinical_explanation(input_data, model_output, None)
        
        # Should still generate explanation without attention-based insights
        assert 'decision_summary' in explanation
        assert 'primary_attention' not in explanation['confidence_factors']

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
