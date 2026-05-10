#!/usr/bin/env python3
"""
Test suite for GAN-based synthetic data generation
Comprehensive unit tests for clinical robotics synthetic data pipeline.

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

from models.gan_synthetic_data import (
    SyntheticDataConfig,
    ClinicalConditionalGAN,
    PhysicsAwareGenerator,
    SyntheticDataValidator,
    ClinicalSyntheticDataPipeline
)

class TestSyntheticDataConfig:
    """Test cases for SyntheticDataConfig."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SyntheticDataConfig()
        
        assert config.generator_type == "conditional_gan"
        assert config.latent_dim == 100
        assert config.image_size == (224, 224)
        assert config.trajectory_length == 50
        assert config.num_medication_types == 5
        assert config.num_grasp_types == 4
        assert config.physics_aware is True
        assert config.quality_threshold == 0.8
        assert config.batch_size == 32
        assert config.epochs == 1000
        assert config.learning_rate == 2e-4
    
    def test_config_to_dict(self):
        """Test configuration serialization."""
        config = SyntheticDataConfig()
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict['generator_type'] == "conditional_gan"
        assert config_dict['latent_dim'] == 100
        assert config_dict['image_size'] == [224, 224]
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = SyntheticDataConfig(
            latent_dim=128,
            batch_size=64,
            quality_threshold=0.9
        )
        
        assert config.latent_dim == 128
        assert config.batch_size == 64
        assert config.quality_threshold == 0.9

class TestClinicalConditionalGAN:
    """Test cases for ClinicalConditionalGAN."""
    
    @pytest.fixture
    def config(self):
        """Fixture for test configuration."""
        return SyntheticDataConfig(
            latent_dim=32,
            image_size=(64, 64),
            trajectory_length=10,
            batch_size=4
        )
    
    @pytest.fixture
    def gan(self, config):
        """Fixture for GAN instance."""
        return ClinicalConditionalGAN(config)
    
    def test_gan_initialization(self, gan, config):
        """Test GAN initialization."""
        assert gan.config == config
        assert gan.generator is not None
        assert gan.discriminator is not None
        assert gan.optimizer_G is not None
        assert gan.optimizer_D is not None
        assert gan.real_label == 1.0
        assert gan.fake_label == 0.0
    
    def test_generator_architecture(self, gan):
        """Test generator architecture."""
        # Test that generator can process input
        batch_size = 2
        z = torch.randn(batch_size, gan.config.latent_dim)
        med_type = torch.randint(0, gan.config.num_medication_types, (batch_size,))
        grasp_type = torch.randint(0, gan.config.num_grasp_types, (batch_size,))
        instruction_tokens = torch.randint(0, 1000, (batch_size, 20))
        
        img, traj, ft = gan.generator(z, med_type, grasp_type, instruction_tokens)
        
        assert img.shape == (batch_size, 3, 224, 224)
        assert traj.shape == (batch_size, gan.config.trajectory_length, 7)
        assert ft.shape == (batch_size, gan.config.trajectory_length, 6)
    
    def test_discriminator_architecture(self, gan):
        """Test discriminator architecture."""
        batch_size = 2
        img = torch.randn(batch_size, 3, 224, 224)
        traj = torch.randn(batch_size, gan.config.trajectory_length, 7)
        ft = torch.randn(batch_size, gan.config.trajectory_length, 6)
        
        validity = gan.discriminator(img, traj, ft)
        
        assert validity.shape == (batch_size, 1)
        assert 0 <= validity.min() <= 1
        assert 0 <= validity.max() <= 1
    
    def test_generate_sample(self, gan):
        """Test single sample generation."""
        sample = gan.generate_sample(
            medication_type=0,
            grasp_type=1,
            instruction="Pick up the medication vial"
        )
        
        assert 'image' in sample
        assert 'trajectory' in sample
        assert 'force_torque' in sample
        assert 'medication_type' in sample
        assert 'grasp_type' in sample
        assert 'instruction' in sample
        
        assert sample['image'].shape == (1, 3, 224, 224)
        assert sample['trajectory'].shape == (1, gan.config.trajectory_length, 7)
        assert sample['force_torque'].shape == (1, gan.config.trajectory_length, 6)
    
    def test_train_step(self, gan):
        """Test training step."""
        # Create fake real batch
        batch_size = 2
        real_batch = {
            'image': torch.randn(batch_size, 3, 224, 224),
            'trajectory': torch.randn(batch_size, gan.config.trajectory_length, 7),
            'force_torque': torch.randn(batch_size, gan.config.trajectory_length, 6),
            'medication_type': torch.randint(0, gan.config.num_medication_types, (batch_size,)),
            'grasp_type': torch.randint(0, gan.config.num_grasp_types, (batch_size,)),
            'instruction_tokens': torch.randint(0, 1000, (batch_size, 20))
        }
        
        losses = gan.train_step(real_batch)
        
        assert 'g_loss' in losses
        assert 'd_loss' in losses
        assert 'real_loss' in losses
        assert 'fake_loss' in losses
        
        assert all(isinstance(loss, float) for loss in losses.values())
        assert all(loss >= 0 for loss in losses.values())

class TestPhysicsAwareGenerator:
    """Test cases for PhysicsAwareGenerator."""
    
    @pytest.fixture
    def config(self):
        """Fixture for test configuration."""
        return SyntheticDataConfig(trajectory_length=20)
    
    @pytest.fixture
    def physics_gen(self, config):
        """Fixture for physics generator."""
        return PhysicsAwareGenerator(config)
    
    def test_physics_initialization(self, physics_gen, config):
        """Test physics generator initialization."""
        assert physics_gen.config == config
        assert 'gravity' in physics_gen.physics_params
        assert 'friction_coefficient' in physics_gen.physics_params
        assert 'max_joint_velocity' in physics_gen.physics_params
        assert 'workspace_bounds' in physics_gen.physics_params
    
    def test_smooth_trajectory(self, physics_gen):
        """Test trajectory smoothing."""
        # Create noisy trajectory
        trajectory = torch.randn(1, 20, 7)
        
        smoothed = physics_gen._smooth_trajectory(trajectory)
        
        assert smoothed.shape == trajectory.shape
        # Smoothed trajectory should have lower variance
        assert torch.var(smoothed) <= torch.var(trajectory)
    
    def test_apply_velocity_constraints(self, physics_gen):
        """Test velocity constraint application."""
        # Create trajectory with high velocities
        trajectory = torch.randn(1, 20, 7) * 10  # High velocities
        
        constrained = physics_gen._apply_velocity_constraints(trajectory)
        
        assert constrained.shape == trajectory.shape
        # Check that velocities are within bounds
        velocities = torch.diff(constrained, dim=1)
        max_vel = physics_gen.physics_params['max_joint_velocity']
        assert torch.all(torch.abs(velocities) <= max_vel + 1e-6)
    
    def test_ensure_workspace_bounds(self, physics_gen):
        """Test workspace bounds enforcement."""
        # Create trajectory outside bounds
        trajectory = torch.randn(1, 20, 7) * 5  # Large values
        
        bounded = physics_gen._ensure_workspace_bounds(trajectory)
        
        assert bounded.shape == trajectory.shape
        
        # Check position bounds (first 3 dimensions)
        bounds = physics_gen.physics_params['workspace_bounds']
        pos = bounded[0, :, :3]
        
        assert torch.all(pos[:, 0] >= bounds['x'][0])
        assert torch.all(pos[:, 0] <= bounds['x'][1])
        assert torch.all(pos[:, 1] >= bounds['y'][0])
        assert torch.all(pos[:, 1] <= bounds['y'][1])
        assert torch.all(pos[:, 2] >= bounds['z'][0])
        assert torch.all(pos[:, 2] <= bounds['z'][1])
    
    def test_apply_physics_constraints(self, physics_gen):
        """Test full physics constraint application."""
        trajectory = torch.randn(1, 20, 7)
        
        constrained = physics_gen.apply_physics_constraints(trajectory)
        
        assert constrained.shape == trajectory.shape
        # Trajectory should be smoother
        velocities = torch.diff(constrained, dim=1)
        assert torch.var(velocities) <= torch.var(torch.diff(trajectory, dim=1))

class TestSyntheticDataValidator:
    """Test cases for SyntheticDataValidator."""
    
    @pytest.fixture
    def config(self):
        """Fixture for test configuration."""
        return SyntheticDataConfig()
    
    @pytest.fixture
    def validator(self, config):
        """Fixture for validator."""
        return SyntheticDataValidator(config)
    
    def test_validator_initialization(self, validator, config):
        """Test validator initialization."""
        assert validator.config == config
        assert validator.quality_metrics == []
    
    def test_validate_sample_complete(self, validator):
        """Test validation of complete sample."""
        sample = {
            'image': torch.randn(1, 3, 224, 224),
            'trajectory': torch.randn(1, 50, 7),
            'force_torque': torch.randn(1, 50, 6)
        }
        
        metrics = validator.validate_sample(sample)
        
        assert 'image_quality' in metrics
        assert 'trajectory_quality' in metrics
        assert 'force_torque_quality' in metrics
        assert 'overall_quality' in metrics
        
        assert all(0 <= score <= 1 for score in metrics.values())
    
    def test_validate_sample_partial(self, validator):
        """Test validation of partial sample."""
        sample = {
            'image': torch.randn(1, 3, 224, 224)
        }
        
        metrics = validator.validate_sample(sample)
        
        assert 'image_quality' in metrics
        assert 'overall_quality' in metrics
        # Missing components should not be in metrics
    
    def test_validate_image_quality(self, validator):
        """Test image quality validation."""
        # Create a reasonable quality image
        image = torch.randn(1, 3, 224, 224)
        
        quality = validator._validate_image_quality(image)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_validate_trajectory_quality(self, validator):
        """Test trajectory quality validation."""
        # Create smooth trajectory
        trajectory = torch.randn(1, 20, 7)
        
        quality = validator._validate_trajectory_quality(trajectory)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_validate_force_torque_quality(self, validator):
        """Test force/torque quality validation."""
        # Create reasonable force/torque data
        ft = torch.randn(1, 20, 6) * 10  # Scale to reasonable range
        
        quality = validator._validate_force_torque_quality(ft)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_validate_joint_limits(self, validator):
        """Test joint limits validation."""
        # Create trajectory within limits
        trajectory = torch.randn(1, 20, 7) * 0.5  # Small values within limits
        
        quality = validator._validate_joint_limits(trajectory.numpy())
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1

class TestClinicalSyntheticDataPipeline:
    """Test cases for ClinicalSyntheticDataPipeline."""
    
    @pytest.fixture
    def config(self):
        """Fixture for test configuration."""
        return SyntheticDataConfig(
            latent_dim=16,
            batch_size=2,
            epochs=2,
            quality_threshold=0.5
        )
    
    @pytest.fixture
    def pipeline(self, config):
        """Fixture for pipeline."""
        return ClinicalSyntheticDataPipeline(config)
    
    def test_pipeline_initialization(self, pipeline, config):
        """Test pipeline initialization."""
        assert pipeline.config == config
        assert pipeline.gan is not None
        assert pipeline.physics_generator is not None
        assert pipeline.validator is not None
        assert 'g_losses' in pipeline.training_history
        assert 'd_losses' in pipeline.training_history
        assert 'validation_scores' in pipeline.training_history
    
    @patch('torch.utils.data.DataLoader')
    def test_train(self, mock_dataloader, pipeline):
        """Test training process."""
        # Mock data loader
        mock_batch = {
            'image': torch.randn(2, 3, 224, 224),
            'trajectory': torch.randn(2, pipeline.config.trajectory_length, 7),
            'force_torque': torch.randn(2, pipeline.config.trajectory_length, 6),
            'medication_type': torch.randint(0, 5, (2,)),
            'grasp_type': torch.randint(0, 4, (2,)),
            'instruction_tokens': torch.randint(0, 1000, (2, 20))
        }
        
        mock_loader = [mock_batch, mock_batch]  # 2 batches
        mock_dataloader.return_value = mock_loader
        
        # Train for 1 epoch
        pipeline.train(mock_loader, num_epochs=1)
        
        # Check that training history was updated
        assert len(pipeline.training_history['g_losses']) == 1
        assert len(pipeline.training_history['d_losses']) == 1
    
    def test_generate_synthetic_dataset(self, pipeline):
        """Test synthetic dataset generation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = pipeline.generate_synthetic_dataset(
                num_samples=10,
                output_dir=temp_dir
            )
            
            assert Path(dataset_path).exists()
            assert dataset_path.endswith('.hdf5')
            
            # Verify dataset structure
            import h5py
            with h5py.File(dataset_path, 'r') as f:
                assert len(f.keys()) > 0  # Should have some demonstrations
                
                # Check first demonstration
                first_demo = list(f.keys())[0]
                demo_group = f[first_demo]
                
                assert 'image' in demo_group
                assert 'trajectory' in demo_group
                assert 'force_torque' in demo_group
                assert 'medication_type' in demo_group.attrs
                assert 'grasp_type' in demo_group.attrs
                assert 'instruction' in demo_group.attrs
                assert 'quality_score' in demo_group.attrs
    
    def test_validate_model(self, pipeline):
        """Test model validation."""
        # Mock data loader
        mock_batch = {
            'image': torch.randn(2, 3, 224, 224),
            'trajectory': torch.randn(2, pipeline.config.trajectory_length, 7),
            'force_torque': torch.randn(2, pipeline.config.trajectory_length, 6),
            'medication_type': torch.randint(0, 5, (2,)),
            'grasp_type': torch.randint(0, 4, (2,)),
            'instruction_tokens': torch.randint(0, 1000, (2, 20))
        }
        
        mock_loader = [mock_batch]
        
        validation_score = pipeline._validate_model(mock_loader)
        
        assert isinstance(validation_score, float)
        assert 0 <= validation_score <= 1
    
    def test_save_and_load_model(self, pipeline):
        """Test model saving and loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save model
            pipeline.save_model(temp_dir)
            
            # Check files were created
            assert Path(temp_dir / 'generator.pth').exists()
            assert Path(temp_dir / 'discriminator.pth').exists()
            assert Path(temp_dir / 'config.json').exists()
            assert Path(temp_dir / 'training_history.json').exists()
            
            # Create new pipeline and load model
            new_pipeline = ClinicalSyntheticDataPipeline(pipeline.config)
            new_pipeline.load_model(temp_dir)
            
            # Check that training history was loaded
            assert new_pipeline.training_history == pipeline.training_history

class TestIntegration:
    """Integration tests for the synthetic data pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test end-to-end pipeline functionality."""
        config = SyntheticDataConfig(
            latent_dim=8,
            batch_size=1,
            epochs=1,
            quality_threshold=0.3,
            trajectory_length=5
        )
        
        pipeline = ClinicalSyntheticDataPipeline(config)
        
        # Test sample generation
        sample = pipeline.gan.generate_sample(0, 1, "Test instruction")
        
        # Test physics constraints
        constrained_traj = pipeline.physics_generator.apply_physics_constraints(sample['trajectory'])
        
        # Test validation
        metrics = pipeline.validator.validate_sample(sample)
        
        # Test dataset generation
        with tempfile.TemporaryDirectory() as temp_dir:
            dataset_path = pipeline.generate_synthetic_dataset(
                num_samples=5,
                output_dir=temp_dir
            )
            
            assert Path(dataset_path).exists()
        
        # Verify all components worked
        assert 'image' in sample
        assert constrained_traj.shape == sample['trajectory'].shape
        assert 'overall_quality' in metrics
        assert metrics['overall_quality'] >= config.quality_threshold

class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_invalid_config(self):
        """Test handling of invalid configuration."""
        # Test with invalid quality threshold
        config = SyntheticDataConfig(quality_threshold=1.5)
        assert config.quality_threshold == 1.5  # Should accept but may cause issues
    
    def test_empty_dataset(self):
        """Test handling of empty dataset."""
        config = SyntheticDataConfig()
        pipeline = ClinicalSyntheticDataPipeline(config)
        
        # Mock empty data loader
        mock_loader = []
        
        # Should handle gracefully
        validation_score = pipeline._validate_model(mock_loader)
        assert isinstance(validation_score, float)
    
    def test_malformed_sample(self):
        """Test validation of malformed samples."""
        config = SyntheticDataConfig()
        validator = SyntheticDataValidator(config)
        
        # Test with completely empty sample
        metrics = validator.validate_sample({})
        assert 'overall_quality' in metrics
        assert metrics['overall_quality'] == 0.0  # Should be zero for empty sample
    
    def test_extreme_values(self):
        """Test handling of extreme values."""
        config = SyntheticDataConfig()
        validator = SyntheticDataValidator(config)
        
        # Test with extreme image values
        extreme_image = torch.zeros(1, 3, 224, 224)  # All black
        quality = validator._validate_image_quality(extreme_image)
        assert 0 <= quality <= 1
        
        # Test with extreme trajectory values
        extreme_traj = torch.randn(1, 20, 7) * 1000  # Very large values
        quality = validator._validate_trajectory_quality(extreme_traj)
        assert 0 <= quality <= 1

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
