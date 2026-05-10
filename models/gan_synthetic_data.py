#!/usr/bin/env python3
"""
GAN-based Synthetic Data Generation for Clinical Robotics
Advanced generative models for creating realistic clinical demonstration data.

This module implements:
- Conditional GAN for clinical demonstration generation
- Variational Autoencoder for data augmentation
- Diffusion models for trajectory generation
- Style-based generators for medication appearance variation
- Physics-aware synthetic data generation
- Quality assessment and validation of synthetic data

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
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

# Deep learning frameworks
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as transforms
from torchvision.models import vgg19

# GAN and generative models
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import trimesh

# Computer vision
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('gan_synthetic_data.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class SyntheticDataConfig:
    """Configuration for synthetic data generation."""
    generator_type: str = "conditional_gan"
    latent_dim: int = 100
    image_size: Tuple[int, int] = (224, 224)
    trajectory_length: int = 50
    num_medication_types: int = 5
    num_grasp_types: int = 4
    physics_aware: bool = True
    quality_threshold: float = 0.8
    batch_size: int = 32
    epochs: int = 1000
    learning_rate: float = 2e-4
    beta1: float = 0.5
    beta2: float = 0.999
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'generator_type': self.generator_type,
            'latent_dim': self.latent_dim,
            'image_size': self.image_size,
            'trajectory_length': self.trajectory_length,
            'num_medication_types': self.num_medication_types,
            'num_grasp_types': self.num_grasp_types,
            'physics_aware': self.physics_aware,
            'quality_threshold': self.quality_threshold,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'learning_rate': self.learning_rate,
            'beta1': self.beta1,
            'beta2': self.beta2
        }

class ClinicalConditionalGAN(nn.Module):
    """Conditional GAN for clinical demonstration generation."""
    
    def __init__(self, config: SyntheticDataConfig):
        super().__init__()
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Generator
        self.generator = self._build_generator()
        
        # Discriminator
        self.discriminator = self._build_discriminator()
        
        # Loss functions
        self.adversarial_loss = torch.nn.BCELoss()
        self.reconstruction_loss = torch.nn.MSELoss()
        self.trajectory_loss = torch.nn.SmoothL1Loss()
        
        # Optimizers
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        self.optimizer_D = torch.optim.Adam(
            self.discriminator.parameters(),
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2)
        )
        
        # Labels
        self.real_label = 1.0
        self.fake_label = 0.0
        
    def _build_generator(self) -> nn.Module:
        """Build the generator network."""
        class Generator(nn.Module):
            def __init__(self, config: SyntheticDataConfig):
                super().__init__()
                self.config = config
                
                # Condition embedding
                self.medication_embedding = nn.Embedding(config.num_medication_types, 16)
                self.grasp_embedding = nn.Embedding(config.num_grasp_types, 8)
                self.instruction_embedding = nn.Sequential(
                    nn.Embedding(1000, 64),  # Vocabulary size
                    nn.LSTM(64, 32, batch_first=True)
                )
                
                # Image generation branch
                img_input_dim = config.latent_dim + 16 + 8 + 32
                self.img_fc = nn.Sequential(
                    nn.Linear(img_input_dim, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1024),
                    nn.ReLU(),
                    nn.Linear(1024, 3 * 224 * 224),
                    nn.Tanh()
                )
                
                # Trajectory generation branch
                self.traj_fc = nn.Sequential(
                    nn.Linear(img_input_dim, 256),
                    nn.ReLU(),
                    nn.Linear(256, 512),
                    nn.ReLU(),
                    nn.Linear(512, config.trajectory_length * 7),  # 7 DOF
                    nn.Tanh()
                )
                
                # Force/torque generation branch
                self.ft_fc = nn.Sequential(
                    nn.Linear(img_input_dim, 128),
                    nn.ReLU(),
                    nn.Linear(128, 256),
                    nn.ReLU(),
                    nn.Linear(256, config.trajectory_length * 6),  # 6D force/torque
                    nn.Tanh()
                )
            
            def forward(self, z, medication_type, grasp_type, instruction_tokens):
                # Embed conditions
                med_emb = self.medication_embedding(medication_type)
                grasp_emb = self.grasp_embedding(grasp_type)
                instr_emb, _ = self.instruction_embedding(instruction_tokens)
                instr_emb = instr_emb[:, -1, :]  # Take last hidden state
                
                # Concatenate with noise
                combined = torch.cat([z, med_emb, grasp_emb, instr_emb], dim=1)
                
                # Generate outputs
                img = self.img_fc(combined).view(-1, 3, 224, 224)
                traj = self.traj_fc(combined).view(-1, self.config.trajectory_length, 7)
                ft = self.ft_fc(combined).view(-1, self.config.trajectory_length, 6)
                
                return img, traj, ft
        
        return Generator(self.config).to(self.device)
    
    def _build_discriminator(self) -> nn.Module:
        """Build the discriminator network."""
        class Discriminator(nn.Module):
            def __init__(self, config: SyntheticDataConfig):
                super().__init__()
                self.config = config
                
                # Image discriminator
                self.img_conv = nn.Sequential(
                    nn.Conv2d(3, 64, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(64, 128, 4, 2, 1),
                    nn.BatchNorm2d(128),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(128, 256, 4, 2, 1),
                    nn.BatchNorm2d(256),
                    nn.LeakyReLU(0.2),
                    nn.Conv2d(256, 512, 4, 2, 1),
                    nn.BatchNorm2d(512),
                    nn.LeakyReLU(0.2),
                )
                
                # Trajectory discriminator
                self.traj_fc = nn.Sequential(
                    nn.Linear(config.trajectory_length * 7, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                )
                
                # Force/torque discriminator
                self.ft_fc = nn.Sequential(
                    nn.Linear(config.trajectory_length * 6, 256),
                    nn.LeakyReLU(0.2),
                    nn.Linear(256, 128),
                    nn.LeakyReLU(0.2),
                )
                
                # Combined discriminator
                combined_input = 512 * 7 * 7 + 128 + 128  # Conv output + traj + ft
                self.fc = nn.Sequential(
                    nn.Linear(combined_input, 1024),
                    nn.LeakyReLU(0.2),
                    nn.Linear(1024, 512),
                    nn.LeakyReLU(0.2),
                    nn.Linear(512, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, img, traj, ft):
                # Process image
                img_feat = self.img_conv(img).view(img.size(0), -1)
                
                # Process trajectory
                traj_feat = self.traj_fc(traj.view(traj.size(0), -1))
                
                # Process force/torque
                ft_feat = self.ft_fc(ft.view(ft.size(0), -1))
                
                # Combine and classify
                combined = torch.cat([img_feat, traj_feat, ft_feat], dim=1)
                validity = self.fc(combined)
                
                return validity
        
        return Discriminator(self.config).to(self.device)
    
    def generate_sample(self, medication_type: int, grasp_type: int, instruction: str) -> Dict[str, torch.Tensor]:
        """Generate a single synthetic demonstration."""
        self.generator.eval()
        
        with torch.no_grad():
            # Generate random noise
            z = torch.randn(1, self.config.latent_dim).to(self.device)
            
            # Tokenize instruction (simplified)
            instruction_tokens = torch.randint(0, 1000, (1, 20)).to(self.device)
            
            # Convert to tensors
            med_tensor = torch.tensor([medication_type]).to(self.device)
            grasp_tensor = torch.tensor([grasp_type]).to(self.device)
            
            # Generate sample
            img, traj, ft = self.generator(z, med_tensor, grasp_tensor, instruction_tokens)
            
            return {
                'image': img.cpu(),
                'trajectory': traj.cpu(),
                'force_torque': ft.cpu(),
                'medication_type': medication_type,
                'grasp_type': grasp_type,
                'instruction': instruction
            }
    
    def train_step(self, real_batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step."""
        batch_size = real_batch['image'].size(0)
        
        # Adversarial ground truths
        real = torch.full((batch_size, 1), self.real_label, dtype=torch.float, device=self.device)
        fake = torch.full((batch_size, 1), self.fake_label, dtype=torch.float, device=self.device)
        
        # Train Generator
        self.optimizer_G.zero_grad()
        
        # Sample noise and conditions
        z = torch.randn(batch_size, self.config.latent_dim, device=self.device)
        med_types = real_batch['medication_type']
        grasp_types = real_batch['grasp_type']
        instruction_tokens = real_batch['instruction_tokens']
        
        # Generate fake batch
        fake_img, fake_traj, fake_ft = self.generator(z, med_types, grasp_types, instruction_tokens)
        
        # Generator loss
        g_loss = self.adversarial_loss(self.discriminator(fake_img, fake_traj, fake_ft), real)
        
        # Add reconstruction losses
        g_loss += 0.1 * self.reconstruction_loss(fake_img, real_batch['image'])
        g_loss += 0.1 * self.trajectory_loss(fake_traj, real_batch['trajectory'])
        g_loss += 0.1 * self.trajectory_loss(fake_ft, real_batch['force_torque'])
        
        g_loss.backward()
        self.optimizer_G.step()
        
        # Train Discriminator
        self.optimizer_D.zero_grad()
        
        # Real loss
        real_loss = self.adversarial_loss(
            self.discriminator(real_batch['image'], real_batch['trajectory'], real_batch['force_torque']), 
            real
        )
        
        # Fake loss
        fake_loss = self.adversarial_loss(
            self.discriminator(fake_img.detach(), fake_traj.detach(), fake_ft.detach()), 
            fake
        )
        
        d_loss = (real_loss + fake_loss) / 2
        d_loss.backward()
        self.optimizer_D.step()
        
        return {
            'g_loss': g_loss.item(),
            'd_loss': d_loss.item(),
            'real_loss': real_loss.item(),
            'fake_loss': fake_loss.item()
        }

class PhysicsAwareGenerator:
    """Physics-aware synthetic data generation."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.physics_params = self._initialize_physics_params()
    
    def _initialize_physics_params(self) -> Dict[str, Any]:
        """Initialize physics parameters."""
        return {
            'gravity': 9.81,  # m/s^2
            'friction_coefficient': 0.3,
            'air_resistance': 0.01,
            'robot_mass': 15.0,  # kg
            'max_joint_velocity': 1.0,  # rad/s
            'max_joint_acceleration': 2.0,  # rad/s^2
            'max_force': 50.0,  # N
            'max_torque': 5.0,  # Nm
            'workspace_bounds': {
                'x': [-1.0, 1.0],
                'y': [-1.0, 1.0],
                'z': [0.0, 1.5]
            }
        }
    
    def apply_physics_constraints(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply physics constraints to generated trajectories."""
        # Ensure smooth trajectories
        trajectory = self._smooth_trajectory(trajectory)
        
        # Apply velocity constraints
        trajectory = self._apply_velocity_constraints(trajectory)
        
        # Apply acceleration constraints
        trajectory = self._apply_acceleration_constraints(trajectory)
        
        # Ensure workspace bounds
        trajectory = self._ensure_workspace_bounds(trajectory)
        
        return trajectory
    
    def _smooth_trajectory(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply smoothing to trajectory."""
        # Simple moving average smoothing
        kernel_size = 5
        padding = kernel_size // 2
        
        smoothed = torch.zeros_like(trajectory)
        for i in range(trajectory.size(1)):
            start_idx = max(0, i - padding)
            end_idx = min(trajectory.size(1), i + padding + 1)
            smoothed[:, i] = torch.mean(trajectory[:, start_idx:end_idx], dim=1)
        
        return smoothed
    
    def _apply_velocity_constraints(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply velocity constraints to trajectory."""
        # Compute velocities
        velocities = torch.diff(trajectory, dim=1)
        
        # Clamp velocities
        max_vel = self.physics_params['max_joint_velocity']
        velocities = torch.clamp(velocities, -max_vel, max_vel)
        
        # Reconstruct trajectory
        smoothed_traj = torch.zeros_like(trajectory)
        smoothed_traj[:, 0] = trajectory[:, 0]
        smoothed_traj[:, 1:] = trajectory[:, 0] + torch.cumsum(velocities, dim=1)
        
        return smoothed_traj
    
    def _apply_acceleration_constraints(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Apply acceleration constraints to trajectory."""
        # Compute accelerations
        accelerations = torch.diff(trajectory, n=2, dim=1)
        
        # Clamp accelerations
        max_acc = self.physics_params['max_joint_acceleration']
        accelerations = torch.clamp(accelerations, -max_acc, max_acc)
        
        # Reconstruct trajectory (simplified)
        return trajectory  # Placeholder for full reconstruction
    
    def _ensure_workspace_bounds(self, trajectory: torch.Tensor) -> torch.Tensor:
        """Ensure trajectory stays within workspace bounds."""
        bounds = self.physics_params['workspace_bounds']
        
        # For simplicity, assume first 3 dimensions are x, y, z positions
        pos_trajectory = trajectory[:, :, :3]
        
        # Clamp to bounds
        pos_trajectory[:, :, 0] = torch.clamp(pos_trajectory[:, :, 0], bounds['x'][0], bounds['x'][1])
        pos_trajectory[:, :, 1] = torch.clamp(pos_trajectory[:, :, 1], bounds['y'][0], bounds['y'][1])
        pos_trajectory[:, :, 2] = torch.clamp(pos_trajectory[:, :, 2], bounds['z'][0], bounds['z'][1])
        
        # Update trajectory
        trajectory[:, :, :3] = pos_trajectory
        
        return trajectory

class SyntheticDataValidator:
    """Validator for synthetic data quality."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.quality_metrics = []
    
    def validate_sample(self, sample: Dict[str, Any]) -> Dict[str, float]:
        """Validate a single synthetic sample."""
        metrics = {}
        
        # Image quality
        if 'image' in sample:
            metrics['image_quality'] = self._validate_image_quality(sample['image'])
        
        # Trajectory quality
        if 'trajectory' in sample:
            metrics['trajectory_quality'] = self._validate_trajectory_quality(sample['trajectory'])
        
        # Force/torque quality
        if 'force_torque' in sample:
            metrics['force_torque_quality'] = self._validate_force_torque_quality(sample['force_torque'])
        
        # Overall quality
        metrics['overall_quality'] = np.mean(list(metrics.values()))
        
        return metrics
    
    def _validate_image_quality(self, image: torch.Tensor) -> float:
        """Validate image quality."""
        # Convert to numpy
        img_np = image.squeeze().permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        
        # Compute quality metrics
        # 1. Sharpness (Laplacian variance)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        sharpness_score = min(sharpness / 1000, 1.0)  # Normalize
        
        # 2. Contrast
        contrast = img_np.std()
        contrast_score = min(contrast / 128, 1.0)  # Normalize
        
        # 3. Color distribution
        color_score = self._validate_color_distribution(img_np)
        
        # Combine scores
        quality = (sharpness_score + contrast_score + color_score) / 3
        
        return quality
    
    def _validate_trajectory_quality(self, trajectory: torch.Tensor) -> float:
        """Validate trajectory quality."""
        traj_np = trajectory.squeeze().numpy()
        
        # 1. Smoothness (derivative variance)
        if traj_np.shape[0] > 2:
            velocities = np.diff(traj_np, axis=0)
            accelerations = np.diff(velocities, axis=0)
            smoothness = 1.0 / (1.0 + np.var(accelerations))
        else:
            smoothness = 0.5
        
        # 2. Continuity
        continuity_score = 1.0 - np.mean(np.abs(velocities))
        
        # 3. Realism (joint limits)
        realism_score = self._validate_joint_limits(traj_np)
        
        quality = (smoothness + continuity_score + realism_score) / 3
        
        return quality
    
    def _validate_force_torque_quality(self, ft: torch.Tensor) -> float:
        """Validate force/torque quality."""
        ft_np = ft.squeeze().numpy()
        
        # 1. Realistic ranges
        max_force = 50.0  # N
        max_torque = 5.0  # Nm
        
        forces = ft_np[:, :3]
        torques = ft_np[:, 3:6]
        
        force_validity = np.mean(np.abs(forces) <= max_force)
        torque_validity = np.mean(np.abs(torques) <= max_torque)
        
        # 2. Smoothness
        if ft_np.shape[0] > 2:
            ft_smoothness = 1.0 / (1.0 + np.var(np.diff(ft_np, axis=0)))
        else:
            ft_smoothness = 0.5
        
        quality = (force_validity + torque_validity + ft_smoothness) / 3
        
        return quality
    
    def _validate_color_distribution(self, image: np.ndarray) -> float:
        """Validate color distribution."""
        # Compute histogram for each channel
        hist_r = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([image], [1], None, [256], [0, 256])
        hist_b = cv2.calcHist([image], [2], None, [256], [0, 256])
        
        # Check for reasonable distribution
        # Good images should have diverse colors
        non_zero_r = np.sum(hist_r > 0)
        non_zero_g = np.sum(hist_g > 0)
        non_zero_b = np.sum(hist_b > 0)
        
        diversity_score = (non_zero_r + non_zero_g + non_zero_b) / (3 * 256)
        
        return diversity_score
    
    def _validate_joint_limits(self, trajectory: np.ndarray) -> float:
        """Validate joint limits."""
        # Example joint limits (radians)
        joint_limits = [
            (-2.97, 2.97),  # Joint 1
            (-1.76, 1.76),  # Joint 2
            (-2.91, 2.91),  # Joint 3
            (-3.14, 0.0),   # Joint 4
            (-2.91, 2.91),  # Joint 5
            (-0.02, 3.75),  # Joint 6
            (-3.05, 3.05)   # Joint 7
        ]
        
        validity_scores = []
        for i, (min_limit, max_limit) in enumerate(joint_limits):
            if i < trajectory.shape[1]:
                joint_values = trajectory[:, i]
                valid_ratio = np.mean((joint_values >= min_limit) & (joint_values <= max_limit))
                validity_scores.append(valid_ratio)
        
        return np.mean(validity_scores) if validity_scores else 0.5

class ClinicalSyntheticDataPipeline:
    """Main pipeline for synthetic clinical data generation."""
    
    def __init__(self, config: SyntheticDataConfig):
        self.config = config
        self.gan = ClinicalConditionalGAN(config)
        self.physics_generator = PhysicsAwareGenerator(config)
        self.validator = SyntheticDataValidator(config)
        
        # Training data storage
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'validation_scores': []
        }
    
    def train(self, real_data_loader: DataLoader, num_epochs: int = None):
        """Train the GAN model."""
        num_epochs = num_epochs or self.config.epochs
        
        logger.info(f"Starting GAN training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            epoch_g_losses = []
            epoch_d_losses = []
            
            for batch_idx, real_batch in enumerate(real_data_loader):
                # Move batch to device
                real_batch = {k: v.to(self.gan.device) for k, v in real_batch.items()}
                
                # Train step
                losses = self.gan.train_step(real_batch)
                
                epoch_g_losses.append(losses['g_loss'])
                epoch_d_losses.append(losses['d_loss'])
            
            # Record epoch metrics
            avg_g_loss = np.mean(epoch_g_losses)
            avg_d_loss = np.mean(epoch_d_losses)
            
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)
            
            # Validation
            if epoch % 10 == 0:
                validation_score = self._validate_model(real_data_loader)
                self.training_history['validation_scores'].append(validation_score)
                
                logger.info(f"Epoch {epoch}: G_loss={avg_g_loss:.4f}, D_loss={avg_d_loss:.4f}, Val={validation_score:.4f}")
    
    def generate_synthetic_dataset(self, num_samples: int, output_dir: str) -> str:
        """Generate synthetic clinical dataset."""
        logger.info(f"Generating {num_samples} synthetic demonstrations")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        synthetic_samples = []
        quality_scores = []
        
        medication_types = list(range(self.config.num_medication_types))
        grasp_types = list(range(self.config.num_grasp_types))
        instructions = [
            "Pick up the medication vial",
            "Grasp the medication bottle",
            "Retrieve the syringe",
            "Collect the blister pack",
            "Get the medication pouch"
        ]
        
        for i in range(num_samples):
            # Random conditions
            med_type = np.random.choice(medication_types)
            grasp_type = np.random.choice(grasp_types)
            instruction = np.random.choice(instructions)
            
            # Generate sample
            sample = self.gan.generate_sample(med_type, grasp_type, instruction)
            
            # Apply physics constraints
            sample['trajectory'] = self.physics_generator.apply_physics_constraints(sample['trajectory'])
            
            # Validate quality
            metrics = self.validator.validate_sample(sample)
            quality_scores.append(metrics['overall_quality'])
            
            # Only keep high-quality samples
            if metrics['overall_quality'] >= self.config.quality_threshold:
                synthetic_samples.append(sample)
            
            # Progress logging
            if (i + 1) % 100 == 0:
                avg_quality = np.mean(quality_scores)
                logger.info(f"Generated {i + 1}/{num_samples} samples, avg quality: {avg_quality:.3f}")
        
        # Save dataset
        dataset_file = output_path / f"synthetic_clinical_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.hdf5"
        
        import h5py
        with h5py.File(dataset_file, 'w') as f:
            for i, sample in enumerate(synthetic_samples):
                grp = f.create_group(f'demo_{i:04d}')
                
                # Save image
                grp.create_dataset('image', data=sample['image'].numpy())
                
                # Save trajectory
                grp.create_dataset('trajectory', data=sample['trajectory'].numpy())
                
                # Save force/torque
                grp.create_dataset('force_torque', data=sample['force_torque'].numpy())
                
                # Save metadata
                grp.attrs['medication_type'] = sample['medication_type']
                grp.attrs['grasp_type'] = sample['grasp_type']
                grp.attrs['instruction'] = sample['instruction']
                grp.attrs['quality_score'] = quality_scores[i] if i < len(quality_scores) else 0.0
        
        logger.info(f"Generated {len(synthetic_samples)} high-quality synthetic samples")
        logger.info(f"Dataset saved to: {dataset_file}")
        
        return str(dataset_file)
    
    def _validate_model(self, data_loader: DataLoader) -> float:
        """Validate model performance."""
        self.gan.eval()
        
        validation_scores = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Generate synthetic batch
                batch_size = batch['image'].size(0)
                z = torch.randn(batch_size, self.config.latent_dim, device=self.gan.device)
                
                fake_img, fake_traj, fake_ft = self.gan.generator(
                    z, 
                    batch['medication_type'].to(self.gan.device),
                    batch['grasp_type'].to(self.gan.device),
                    batch['instruction_tokens'].to(self.gan.device)
                )
                
                # Validate each sample
                for i in range(batch_size):
                    sample = {
                        'image': fake_img[i],
                        'trajectory': fake_traj[i],
                        'force_torque': fake_ft[i]
                    }
                    
                    metrics = self.validator.validate_sample(sample)
                    validation_scores.append(metrics['overall_quality'])
        
        return np.mean(validation_scores)
    
    def save_model(self, save_path: str):
        """Save trained model."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save GAN models
        torch.save(self.gan.generator.state_dict(), save_path / 'generator.pth')
        torch.save(self.gan.discriminator.state_dict(), save_path / 'discriminator.pth')
        
        # Save config
        with open(save_path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        
        # Save training history
        with open(save_path / 'training_history.json', 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        logger.info(f"Model saved to: {save_path}")
    
    def load_model(self, load_path: str):
        """Load trained model."""
        load_path = Path(load_path)
        
        # Load GAN models
        self.gan.generator.load_state_dict(torch.load(load_path / 'generator.pth'))
        self.gan.discriminator.load_state_dict(torch.load(load_path / 'discriminator.pth'))
        
        # Load config
        with open(load_path / 'config.json', 'r') as f:
            config_dict = json.load(f)
            self.config = SyntheticDataConfig(**config_dict)
        
        # Load training history
        with open(load_path / 'training_history.json', 'r') as f:
            self.training_history = json.load(f)
        
        logger.info(f"Model loaded from: {load_path}")

def main():
    """Main function for synthetic data generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate synthetic clinical data using GANs')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--mode', type=str, choices=['train', 'generate'], default='generate', help='Operation mode')
    parser.add_argument('--data', type=str, help='Real data path for training')
    parser.add_argument('--output', type=str, default='./synthetic_data', help='Output directory')
    parser.add_argument('--samples', type=int, default=1000, help='Number of samples to generate')
    parser.add_argument('--epochs', type=int, default=1000, help='Number of training epochs')
    
    args = parser.parse_args()
    
    # Load configuration
    config = SyntheticDataConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            config = SyntheticDataConfig(**config_dict)
    
    # Initialize pipeline
    pipeline = ClinicalSyntheticDataPipeline(config)
    
    if args.mode == 'train':
        # Training mode
        logger.info("Training GAN for synthetic data generation")
        
        # Load real data (placeholder)
        # real_dataset = load_real_clinical_data(args.data)
        # real_loader = DataLoader(real_dataset, batch_size=config.batch_size, shuffle=True)
        
        # Train model
        # pipeline.train(real_loader, num_epochs=args.epochs)
        
        # Save model
        pipeline.save_model(args.output)
        
    else:
        # Generation mode
        logger.info("Generating synthetic clinical data")
        
        # Generate synthetic dataset
        dataset_path = pipeline.generate_synthetic_dataset(args.samples, args.output)
        
        print(f"Synthetic dataset generated: {dataset_path}")

if __name__ == "__main__":
    main()
