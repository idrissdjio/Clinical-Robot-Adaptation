"""
ClinAdapt: Octo Foundation Model Few-Shot Fine-Tuning Pipeline
Domain adaptation for clinical medication delivery environments.

Based on:
    Octo: An Open-Source Generalist Robot Policy (arXiv:2405.12213)
    Etukuru et al., Robot Utility Models, IEEE ICRA 2025
    Hwang et al., AI Implementation in U.S. Hospitals, Stanford 2025

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.experimental import optimizers, stax
from flax import linen as nn_flax
from flax.training import train_state

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import json
import pickle
import h5py
from pathlib import Path
import logging
from datetime import datetime
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split

# Performance optimization imports
import gc
from torch.profiler import profile, record_function, ProfilerActivity

# Octo-specific imports
from transformers import AutoTokenizer, AutoModel
from einops import rearrange, repeat
from torchvision import transforms
from PIL import Image
import cv2

# Clinical robotics imports
from scipy.spatial.transform import Rotation
import pybullet as p
import pybullet_data

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ClinicalOctoAdapter:
    """
    Advanced Octo foundation model adapter for clinical medication delivery robots.
    
    This class implements sophisticated few-shot learning techniques for adapting
    Octo generalist robot policy to clinical environments. It incorporates:
    - Multi-modal fusion (vision, language, robot state)
    - Sample-efficient fine-tuning with layer freezing
    - Clinical data augmentation
    - Human-aware constraint integration
    - Performance monitoring and evaluation
    
    Key Innovation: Reduces deployment data requirements from thousands of 
    demonstrations to 50-200 demonstrations per new hospital environment,
    enabling national-scale adoption of robotic medication systems.
    """

    def __init__(self, base_model_path=None, config=None):
        """
        Initialize Clinical Octo Adapter.
        
        Args:
            base_model_path: Path to pre-trained Octo model
            config: Configuration dictionary containing model parameters,
                   training settings, and clinical environment specifications.
        """
        self.base_model_path = base_model_path
        self.config = config or self._create_default_config()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Model configuration
        self.model_name = self.config.get('model_name', 'octo-clinical-v1')
        self.sequence_length = self.config.get('sequence_length', 32)
        self.image_size = self.config.get('image_size', (224, 224))
        self.action_dim = self.config.get('action_dim', 7)  # 7-DOF robot arm
        
        # Training configuration
        self.batch_size = self.config.get('batch_size', 16)
        self.learning_rate = self.config.get('learning_rate', 1e-4)
        self.num_epochs = self.config.get('num_epochs', 100)
        self.patience = self.config.get('patience', 10)
        
        # Clinical-specific parameters
        self.min_demonstrations = self.config.get('min_demonstrations', 50)
        self.max_demonstrations = self.config.get('max_demonstrations', 200)
        self.quality_threshold = self.config.get('quality_threshold', 0.85)
        self.safety_margin = self.config.get('safety_margin', 0.1)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler()
        
        # Data management
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        
        # Performance tracking
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'adaptation_progress': []
        }
        
        # Clinical metrics
        self.clinical_metrics = {
            'grasp_success_rate': [],
            'medication_recognition': [],
            'safety_violations': [],
            'human_awareness': []
        }
        
        # Initialize model and components
        self._initialize_model()
        self._initialize_training_components()
        self._setup_logging()
        
        logger.info(f"Clinical Octo Adapter initialized with model: {self.model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Target demonstration range: {self.min_demonstrations}-{self.max_demonstrations}")
    
    def _create_default_config(self) -> Dict[str, Any]:
        """Create default configuration for clinical robot adaptation."""
        return {
            # Model configuration
            'model_name': 'octo-clinical-v1',
            'image_size': (224, 224),
            'sequence_length': 32,
            'action_dim': 7,
            'hidden_dim': 512,
            'num_heads': 8,
            'num_layers': 6,
            'dropout': 0.1,
            'patch_size': 16,
            'vision_layers': 4,
            'language_layers': 4,
            'state_dim': 14,
            
            # Training configuration
            'batch_size': 16,
            'learning_rate': 1e-4,
            'num_epochs': 100,
            'patience': 10,
            'weight_decay': 1e-4,
            'T_0': 10,
            'T_mult': 2,
            
            # Loss weights
            'action_loss_weight': 1.0,
            'grasp_loss_weight': 0.5,
            'human_loss_weight': 0.3,
            'safety_loss_weight': 2.0,
            
            # Clinical parameters
            'min_demonstrations': 50,
            'max_demonstrations': 200,
            'quality_threshold': 0.85,
            'safety_margin': 0.1,
            
            # Layer freezing for few-shot learning
            'frozen_layers': [
                'vision_encoder.backbone',
                'language_encoder.embeddings',
                'transformer.layers.0',
                'transformer.layers.1',
                'transformer.layers.2'
            ],
            
            # Data configuration
            'val_split': 0.2,
            'use_wandb': False,
            'save_dir': './checkpoints'
        }
    
    def _initialize_model(self):
        """Initialize Octo foundation model and tokenizer."""
        try:
            # Initialize tokenizer for language instructions
            self.tokenizer = AutoTokenizer.from_pretrained(
                'microsoft/DialoGPT-medium',
                trust_remote_code=True
            )
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Initialize Octo model architecture
            self.model = OctoTransformer(
                vocab_size=self.tokenizer.vocab_size,
                image_size=self.image_size,
                sequence_length=self.sequence_length,
                action_dim=self.action_dim,
                **self.config.get('model_params', {})
            ).to(self.device)
            
            # Load pre-trained weights if available
            if self.base_model_path and Path(self.base_model_path).exists():
                self._load_pretrained_weights(self.base_model_path)
            
            # Setup layer freezing for few-shot learning
            self._setup_layer_freezing()
            
            logger.info("Model and tokenizer initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise
    
    def _initialize_training_components(self):
        """Initialize training components: optimizer, scheduler, loss functions."""
        # Setup optimizer with different learning rates for different components
        param_groups = self._create_parameter_groups()
        
        self.optimizer = optim.AdamW(
            param_groups,
            lr=self.learning_rate,
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=(0.9, 0.999)
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2),
            eta_min=self.learning_rate * 0.1
        )
        
        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
        
        logger.info("Training components initialized")
    
    def _create_parameter_groups(self) -> List[Dict]:
        """Create parameter groups with different learning rates."""
        param_groups = []
        
        # Frozen layers (backbone)
        frozen_params = []
        for name, param in self.model.named_parameters():
            if any(freeze_layer in name for freeze_layer in self.config.get('frozen_layers', [])):
                param.requires_grad = False
                frozen_params.append(param)
        
        # Trainable layers (adaptation heads)
        trainable_params = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
        
        if trainable_params:
            param_groups.append({
                'params': trainable_params,
                'lr': self.learning_rate,
                'name': 'trainable'
            })
        
        logger.info(f"Parameter groups: {len(frozen_params)} frozen, {len(trainable_params)} trainable")
        return param_groups
    
    def _setup_layer_freezing(self):
        """Setup layer freezing strategy for few-shot learning."""
        frozen_layers = self.config.get('frozen_layers', [
            'vision_encoder.backbone',
            'language_encoder.embeddings',
            'transformer.layers.0',
            'transformer.layers.1',
            'transformer.layers.2'
        ])
        
        for name, param in self.model.named_parameters():
            if any(freeze_layer in name for freeze_layer in frozen_layers):
                param.requires_grad = False
        
        logger.info(f"Frozen layers: {frozen_layers}")
    
    def _load_pretrained_weights(self, path: str):
        """Load pre-trained weights from checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Load weights with partial matching
        model_dict = self.model.state_dict()
        pretrained_dict = {k: v for k, v in state_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.model.load_state_dict(model_dict)
        
        logger.info(f"Loaded {len(pretrained_dict)}/{len(model_dict)} parameters from {path}")
    
    def _setup_logging(self):
        """Setup experiment logging with Weights & Biases."""
        if self.config.get('use_wandb', False):
            wandb.init(
                project="clinical-robot-adaptation",
                name=f"octo-clinical-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
                config=self.config
            )
    
    def prepare_clinical_dataset(self, demonstration_path):
        """
        Load and preprocess clinical demonstration data for few-shot adaptation.
        
        Args:
            demonstration_path: Path to demonstration data (HDF5, pickle, or directory)
        
        Returns:
            Tuple of (train_dataset, val_dataset)
        """
        logger.info(f"Loading clinical demonstrations from {demonstration_path}")
        
        # Load demonstrations
        demonstrations = self._load_demonstrations(demonstration_path)
        
        # Filter demonstrations by quality
        filtered_demos = [
            demo for demo in demonstrations 
            if demo.get('quality_score', 0) >= self.quality_threshold
        ]
        
        logger.info(f"Filtered to {len(filtered_demos)} high-quality demonstrations")
        
        # Apply clinical data augmentation
        augmented_demos = self._apply_clinical_augmentation(filtered_demos)
        
        # Split into train/validation
        train_demos, val_demos = train_test_split(
            augmented_demos, 
            test_size=self.config.get('val_split', 0.2),
            random_state=42
        )
        
        # Create datasets
        self.train_dataset = ClinicalRobotDataset(
            train_demos, 
            self.tokenizer,
            self.image_size,
            self.sequence_length
        )
        
        self.val_dataset = ClinicalRobotDataset(
            val_demos,
            self.tokenizer,
            self.image_size,
            self.sequence_length
        )
        
        logger.info(f"Created datasets: {len(self.train_dataset)} train, {len(self.val_dataset)} val")
        
        return self.train_dataset, self.val_dataset
    
    def _load_demonstrations(self, demonstration_path: str) -> List[Dict]:
        """Load demonstrations from various formats."""
        demonstrations = []
        data_path = Path(demonstration_path)
        
        if data_path.is_file() and data_path.suffix == '.h5':
            demonstrations = self._load_hdf5_demonstrations(data_path)
        elif data_path.is_file() and data_path.suffix == '.pkl':
            with open(data_path, 'rb') as f:
                demonstrations = pickle.load(f)
        elif data_path.is_dir():
            demonstrations = self._load_directory_demonstrations(data_path)
        else:
            raise ValueError(f"Unsupported data format: {demonstration_path}")
        
        return demonstrations
    
    def _load_hdf5_demonstrations(self, file_path: Path) -> List[Dict]:
        """Load demonstrations from HDF5 file."""
        demonstrations = []
        
        with h5py.File(file_path, 'r') as f:
            demo_groups = list(f.keys())
            
            for demo_name in demo_groups:
                demo_group = f[demo_name]
                
                demo = {
                    'images': demo_group['images'][:],
                    'robot_state': demo_group['robot_state'][:],
                    'actions': demo_group['actions'][:],
                    'instruction': demo_group.attrs.get('instruction', ''),
                    'grasp_type': demo_group.attrs.get('grasp_type', 'precision'),
                    'medication_type': demo_group.attrs.get('medication_type', 'vial'),
                    'quality_score': demo_group.attrs.get('quality_score', 1.0),
                    'human_present': demo_group.attrs.get('human_present', False)
                }
                
                demonstrations.append(demo)
        
        return demonstrations
    
    def _load_directory_demonstrations(self, dir_path: Path) -> List[Dict]:
        """Load demonstrations from directory structure."""
        demonstrations = []
        
        for demo_file in dir_path.glob('*.json'):
            with open(demo_file, 'r') as f:
                demo = json.load(f)
                
            # Load associated image if exists
            image_path = demo_file.with_suffix('.jpg')
            if image_path.exists():
                demo['images'] = str(image_path)
            
            demonstrations.append(demo)
        
        return demonstrations
    
    def _apply_clinical_augmentation(self, demonstrations: List[Dict]) -> List[Dict]:
        """Apply clinical-specific data augmentation."""
        augmented = demonstrations.copy()
        
        for demo in demonstrations:
            # Lighting variations (hospital lighting conditions)
            if np.random.random() < 0.3:
                aug_demo = self._augment_lighting(demo)
                augmented.append(aug_demo)
            
            # Medication object variations
            if np.random.random() < 0.2:
                aug_demo = self._augment_medication_objects(demo)
                augmented.append(aug_demo)
            
            # Human presence simulation
            if np.random.random() < 0.15:
                aug_demo = self._augment_human_presence(demo)
                augmented.append(aug_demo)
        
        return augmented
    
    def _augment_lighting(self, demo: Dict) -> Dict:
        """Augment demonstration with lighting variations."""
        aug_demo = demo.copy()
        
        # Apply brightness and contrast adjustments
        brightness_factor = np.random.uniform(0.8, 1.2)
        contrast_factor = np.random.uniform(0.9, 1.1)
        
        # Process images
        for i, image in enumerate(demo['images']):
            if isinstance(image, np.ndarray):
                # Convert to PIL for augmentation
                pil_image = Image.fromarray(image.astype(np.uint8))
                
                # Apply transforms
                transform = transforms.Compose([
                    transforms.ColorJitter(
                        brightness=brightness_factor - 1.0,
                        contrast=contrast_factor - 1.0,
                        saturation=0.1,
                        hue=0.05
                    ),
                    transforms.RandomAdjustSharpness(0.5),
                ])
                
                aug_image = transform(pil_image)
                aug_demo['images'][i] = np.array(aug_image)
        
        return aug_demo
    
    def _augment_medication_objects(self, demo: Dict) -> Dict:
        """Augment demonstration with medication object variations."""
        aug_demo = demo.copy()
        
        # Simulate different medication packaging
        medication_types = ['vial', 'blister_pack', 'syringe', 'bottle', 'pouch']
        current_type = demo.get('medication_type', 'vial')
        
        # Randomly change medication type
        new_type = np.random.choice([t for t in medication_types if t != current_type])
        aug_demo['medication_type'] = new_type
        
        # Adjust grasp parameters based on medication type
        grasp_params = self._get_grasp_parameters(new_type)
        aug_demo['grasp_parameters'] = grasp_params
        
        return aug_demo
    
    def _get_grasp_parameters(self, medication_type: str) -> Dict:
        """Get grasp parameters for different medication types."""
        grasp_params = {
            'vial': {
                'grasp_type': 'precision',
                'force_range': (5.0, 15.0),  # Newtons
                'approach_angle': np.random.uniform(-15, 15),  # degrees
                'grasp_width': np.random.uniform(0.02, 0.04)  # meters
            },
            'blister_pack': {
                'grasp_type': 'power',
                'force_range': (8.0, 20.0),
                'approach_angle': np.random.uniform(-30, 30),
                'grasp_width': np.random.uniform(0.08, 0.12)
            },
            'syringe': {
                'grasp_type': 'precision',
                'force_range': (3.0, 10.0),
                'approach_angle': np.random.uniform(-10, 10),
                'grasp_width': np.random.uniform(0.008, 0.015)
            },
            'bottle': {
                'grasp_type': 'power',
                'force_range': (10.0, 25.0),
                'approach_angle': np.random.uniform(-45, 45),
                'grasp_width': np.random.uniform(0.03, 0.08)
            },
            'pouch': {
                'grasp_type': 'pinch',
                'force_range': (5.0, 15.0),
                'approach_angle': np.random.uniform(-20, 20),
                'grasp_width': np.random.uniform(0.12, 0.15)
            }
        }
        
        return grasp_params.get(medication_type, grasp_params['vial'])
    
    def _augment_human_presence(self, demo: Dict) -> Dict:
        """Augment demonstration with human presence simulation."""
        aug_demo = demo.copy()
        
        # Simulate human in workspace
        human_present = True
        human_position = np.random.uniform(-0.5, 0.5, 2)  # x, y position
        human_intent = np.random.choice([
            'approaching_robot', 'departing_robot', 'crossing_path',
            'working_nearby', 'observing'
        ])
        
        aug_demo['human_present'] = human_present
        aug_demo['human_position'] = human_position
        aug_demo['human_intent'] = human_intent
        
        # Adjust robot trajectory for human awareness
        aug_demo['actions'] = self._adjust_trajectory_for_human(
            demo['actions'], human_position, human_intent
        )
        
        return aug_demo
    
    def _adjust_trajectory_for_human(self, actions: np.ndarray, 
                                   human_position: np.ndarray, 
                                   human_intent: str) -> np.ndarray:
        """Adjust robot trajectory based on human presence and intent."""
        adjusted_actions = actions.copy()
        
        # Safety distance based on intent
        safety_distances = {
            'approaching_robot': 0.8,
            'departing_robot': 0.5,
            'crossing_path': 1.0,
            'working_nearby': 0.6,
            'observing': 0.4
        }
        
        min_distance = safety_distances.get(human_intent, 0.5)
        
        # Adjust trajectory points that are too close to human
        for i, action in enumerate(actions):
            # Assuming action contains end-effector position
            ee_pos = action[:3] if len(action) >= 3 else np.array([0, 0, 0])
            human_pos_3d = np.array([human_position[0], human_position[1], 0.8])  # Approximate human height
            
            distance = np.linalg.norm(ee_pos - human_pos_3d)
            
            if distance < min_distance:
                # Adjust trajectory away from human
                direction = ee_pos - human_pos_3d
                if np.linalg.norm(direction) > 0:
                    direction = direction / np.linalg.norm(direction)
                    new_pos = human_pos_3d + direction * min_distance
                    adjusted_actions[i, :3] = new_pos
        
        return adjusted_actions
    
    def finetune(self, dataset=None, num_steps=1000):
        """
        Run few-shot fine-tuning on clinical demonstration data.
        
        Args:
            dataset: Training dataset (if None, uses self.train_dataset)
            num_steps: Number of training steps
        
        Returns:
            Training results and metrics
        """
        if dataset is None:
            dataset = self.train_dataset
        
        if dataset is None:
            raise ValueError("No training dataset available. Call prepare_clinical_dataset() first.")
        
        logger.info(f"Starting few-shot fine-tuning with {len(dataset)} demonstrations")
        
        # Create data loader
        train_loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4
        )
        
        # Validation loader
        val_loader = None
        if self.val_dataset:
            val_loader = DataLoader(
                self.val_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=4
            )
        
        # Training loop
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.num_epochs):
            # Training phase
            train_metrics = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_metrics = {}
            if val_loader:
                val_metrics = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            
            # Record metrics
            self.training_history['train_loss'].append(train_metrics['loss'])
            if val_metrics:
                self.training_history['val_loss'].append(val_metrics['loss'])
            self.training_history['train_acc'].append(train_metrics['accuracy'])
            if val_metrics:
                self.training_history['val_acc'].append(val_metrics['accuracy'])
            
            # Log to wandb
            if self.config.get('use_wandb', False):
                log_dict = {
                    'epoch': epoch,
                    'train_loss': train_metrics['loss'],
                    'train_acc': train_metrics['accuracy'],
                    'lr': self.optimizer.param_groups[0]['lr']
                }
                if val_metrics:
                    log_dict.update({
                        'val_loss': val_metrics['loss'],
                        'val_acc': val_metrics['accuracy']
                    })
                wandb.log(log_dict)
            
            # Early stopping check
            if val_metrics and val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                patience_counter = 0
                self._save_checkpoint(epoch, 'best_model.pth')
            else:
                patience_counter += 1
                
            if patience_counter >= self.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Log progress
            if epoch % 10 == 0:
                log_msg = f"Epoch {epoch}: Train Loss={train_metrics['loss']:.4f}, Train Acc={train_metrics['accuracy']:.4f}"
                if val_metrics:
                    log_msg += f", Val Loss={val_metrics['loss']:.4f}, Val Acc={val_metrics['accuracy']:.4f}"
                logger.info(log_msg)
        
        # Final evaluation
        final_metrics = self._evaluate_model(val_loader or train_loader)
        
        results = {
            'training_history': self.training_history,
            'clinical_metrics': self.clinical_metrics,
            'final_metrics': final_metrics,
            'best_val_loss': best_val_loss if val_metrics else train_metrics['loss']
        }
        
        logger.info("Few-shot fine-tuning completed successfully")
        return results
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = self._prepare_batch(batch)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            with autocast():
                outputs = self.model(
                    images=batch['images'],
                    language_instructions=batch['instructions'],
                    robot_states=batch['robot_states'],
                    actions=batch['actions']
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
            
            # Backward pass
            self.scaler.scale(loss).backward()
            
            # Gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Optimizer step
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            # Update metrics
            total_loss += loss.item()
            
            # Compute accuracy (for action prediction)
            if 'predicted_actions' in outputs:
                predicted_actions = outputs['predicted_actions']
                target_actions = batch['actions']
                
                # Compute action accuracy (within tolerance)
                action_error = torch.norm(predicted_actions - target_actions, dim=-1)
                correct = (action_error < 0.1).sum().item()
                total_correct += correct
                total_samples += target_actions.size(0)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss / (batch_idx + 1):.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        accuracy = total_correct / max(total_samples, 1)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Validation Epoch {epoch}"):
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    images=batch['images'],
                    language_instructions=batch['instructions'],
                    robot_states=batch['robot_states'],
                    actions=batch['actions']
                )
                
                # Compute loss
                loss = self._compute_loss(outputs, batch)
                total_loss += loss.item()
                
                # Compute accuracy
                if 'predicted_actions' in outputs:
                    predicted_actions = outputs['predicted_actions']
                    target_actions = batch['actions']
                    
                    action_error = torch.norm(predicted_actions - target_actions, dim=-1)
                    correct = (action_error < 0.1).sum().item()
                    total_correct += correct
                    total_samples += target_actions.size(0)
        
        avg_loss = total_loss / len(val_loader)
        accuracy = total_correct / max(total_samples, 1)
        
        return {'loss': avg_loss, 'accuracy': accuracy}
    
    def _prepare_batch(self, batch: Dict) -> Dict[str, torch.Tensor]:
        """Prepare batch for model input."""
        # Tokenize language instructions
        if isinstance(batch['instructions'], list):
            tokenized_instructions = self.tokenizer(
                batch['instructions'],
                padding=True,
                truncation=True,
                max_length=self.sequence_length,
                return_tensors='pt'
            )
            batch['instructions'] = tokenized_instructions
        
        # Move tensors to device
        batch = {k: v.to(self.device) if torch.is_tensor(v) else v 
                 for k, v in batch.items()}
        
        return batch
    
    def _compute_loss(self, outputs: Dict[str, torch.Tensor], 
                     batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss with clinical weighting."""
        losses = {}
        
        # Action prediction loss
        if 'predicted_actions' in outputs:
            action_loss = self.mse_loss(
                outputs['predicted_actions'], 
                batch['actions']
            )
            losses['action'] = action_loss * self.config.get('action_loss_weight', 1.0)
        
        # Grasp type classification loss
        if 'grasp_logits' in outputs and 'grasp_types' in batch:
            grasp_loss = self.ce_loss(
                outputs['grasp_logits'],
                batch['grasp_types']
            )
            losses['grasp'] = grasp_loss * self.config.get('grasp_loss_weight', 0.5)
        
        # Human awareness loss
        if 'human_awareness' in outputs and 'human_present' in batch:
            human_loss = self.focal_loss(
                outputs['human_awareness'],
                batch['human_present'].float()
            )
            losses['human'] = human_loss * self.config.get('human_loss_weight', 0.3)
        
        # Safety constraint loss
        if 'safety_violations' in outputs:
            safety_loss = torch.mean(outputs['safety_violations'])
            losses['safety'] = safety_loss * self.config.get('safety_loss_weight', 2.0)
        
        # Total loss
        total_loss = sum(losses.values())
        
        return total_loss
    
    def _evaluate_model(self, data_loader: DataLoader) -> Dict[str, float]:
        """Comprehensive model evaluation with clinical metrics."""
        self.model.eval()
        
        all_predictions = []
        all_targets = []
        all_grasp_predictions = []
        all_grasp_targets = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Move batch to device
                batch = self._prepare_batch(batch)
                
                # Forward pass
                outputs = self.model(
                    images=batch['images'],
                    language_instructions=batch['instructions'],
                    robot_states=batch['robot_states'],
                    actions=batch['actions']
                )
                
                # Collect predictions
                if 'predicted_actions' in outputs:
                    all_predictions.append(outputs['predicted_actions'].cpu())
                    all_targets.append(batch['actions'].cpu())
                
                if 'grasp_logits' in outputs:
                    all_grasp_predictions.append(torch.argmax(outputs['grasp_logits'], dim=1).cpu())
                    all_grasp_targets.append(batch['grasp_types'].cpu())
        
        # Concatenate all predictions
        if all_predictions:
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Compute action metrics
            action_mse = F.mse_loss(all_predictions, all_targets).item()
            action_mae = F.l1_loss(all_predictions, all_targets).item()
            
            # Compute grasp accuracy
            grasp_accuracy = 0.0
            if all_grasp_predictions:
                all_grasp_predictions = torch.cat(all_grasp_predictions, dim=0)
                all_grasp_targets = torch.cat(all_grasp_targets, dim=0)
                grasp_accuracy = accuracy_score(
                    all_grasp_targets.numpy(),
                    all_grasp_predictions.numpy()
                )
        
        metrics = {
            'action_mse': action_mse if all_predictions else 0.0,
            'action_mae': action_mae if all_predictions else 0.0,
            'grasp_accuracy': grasp_accuracy
        }
        
        return metrics
    
    def _save_checkpoint(self, epoch: int, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'training_history': self.training_history,
            'clinical_metrics': self.clinical_metrics,
            'config': self.config
        }
        
        save_path = Path(self.config.get('save_dir', './checkpoints')) / filename
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, save_path)
        logger.info(f"Checkpoint saved to {save_path}")
    
    def evaluate(self, env, num_episodes=50):
        """
        Evaluate adapted model on ClinBench-MedDel metrics.
        
        Args:
            env: Clinical environment for evaluation
            num_episodes: Number of evaluation episodes
        
        Returns:
            Dictionary containing evaluation metrics
        """
        logger.info(f"Evaluating model on {num_episodes} episodes")
        
        self.model.eval()
        
        evaluation_metrics = {
            'success_rate': 0,
            'grasp_success_rate': 0,
            'medication_recognition_accuracy': 0,
            'safety_violations': 0,
            'human_awareness_score': 0,
            'average_completion_time': 0,
            'trajectory_smoothness': 0
        }
        
        total_successes = 0
        total_grasp_successes = 0
        total_medication_recognitions = 0
        total_safety_violations = 0
        total_human_awareness = 0
        completion_times = []
        trajectory_smoothness_scores = []
        
        for episode in tqdm(range(num_episodes), desc="Evaluating"):
            # Reset environment
            obs = env.reset()
            
            episode_success = False
            grasp_success = False
            medication_recognized = False
            safety_violations = 0
            human_awareness_score = 0
            episode_start_time = datetime.now()
            trajectory_data = []
            
            for step in range(100):  # Max steps per episode
                # Get language instruction
                instruction = obs.get('instruction', 'Retrieve medication from shelf')
                
                # Prepare observation for model
                image = torch.FloatTensor(obs['image']).unsqueeze(0).to(self.device)
                robot_state = torch.FloatTensor(obs['robot_state']).unsqueeze(0).to(self.device)
                
                # Get model prediction
                with torch.no_grad():
                    outputs = self.predict(image, [instruction], robot_state)
                    action = outputs['predicted_actions'][0].cpu().numpy()
                
                # Execute action in environment
                next_obs, reward, done, info = env.step(action)
                
                # Record trajectory data
                trajectory_data.append({
                    'position': obs['robot_state'][:3],
                    'action': action,
                    'timestamp': step
                })
                
                # Update metrics
                if info.get('grasp_success', False):
                    grasp_success = True
                
                if info.get('medication_recognized', False):
                    medication_recognized = True
                
                if info.get('safety_violation', False):
                    safety_violations += 1
                
                if info.get('human_awareness', 0) > 0:
                    human_awareness_score = max(human_awareness_score, info['human_awareness'])
                
                obs = next_obs
                
                if done:
                    episode_success = reward > 0
                    break
            
            # Update episode metrics
            if episode_success:
                total_successes += 1
            if grasp_success:
                total_grasp_successes += 1
            if medication_recognized:
                total_medication_recognitions += 1
            
            total_safety_violations += safety_violations
            total_human_awareness += human_awareness_score
            
            # Calculate completion time
            completion_time = (datetime.now() - episode_start_time).total_seconds()
            completion_times.append(completion_time)
            
            # Calculate trajectory smoothness
            smoothness = self._calculate_trajectory_smoothness(trajectory_data)
            trajectory_smoothness_scores.append(smoothness)
        
        # Calculate final metrics
        evaluation_metrics['success_rate'] = total_successes / num_episodes
        evaluation_metrics['grasp_success_rate'] = total_grasp_successes / num_episodes
        evaluation_metrics['medication_recognition_accuracy'] = total_medication_recognitions / num_episodes
        evaluation_metrics['safety_violations'] = total_safety_violations / num_episodes
        evaluation_metrics['human_awareness_score'] = total_human_awareness / num_episodes
        evaluation_metrics['average_completion_time'] = np.mean(completion_times)
        evaluation_metrics['trajectory_smoothness'] = np.mean(trajectory_smoothness_scores)
        
        logger.info(f"Evaluation completed: Success Rate={evaluation_metrics['success_rate']:.3f}")
        
        return evaluation_metrics
    
    def _calculate_trajectory_smoothness(self, trajectory_data: List[Dict]) -> float:
        """Calculate trajectory smoothness metric."""
        if len(trajectory_data) < 3:
            return 1.0
        
        positions = np.array([data['position'] for data in trajectory_data])
        
        # Calculate accelerations
        velocities = np.diff(positions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Smoothness = 1 / (1 + variance of accelerations)
        smoothness = 1.0 / (1.0 + np.var(accelerations))
        
        return smoothness
    
    def predict(self, images: torch.Tensor, language_instructions: List[str], 
               robot_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Make predictions using the adapted model.
        
        Args:
            images: Batch of RGB images
            language_instructions: List of language instructions
            robot_states: Robot state tensors
        
        Returns:
            Dictionary containing predictions
        """
        self.model.eval()
        
        with torch.no_grad():
            # Tokenize language instructions
            tokenized_instructions = self.tokenizer(
                language_instructions,
                padding=True,
                truncation=True,
                max_length=self.sequence_length,
                return_tensors='pt'
            )
            
            # Move to device
            images = images.to(self.device)
            robot_states = robot_states.to(self.device)
            tokenized_instructions = {
                k: v.to(self.device) for k, v in tokenized_instructions.items()
            }
            
            # Forward pass
            outputs = self.model(
                images=images,
                language_instructions=tokenized_instructions,
                robot_states=robot_states
            )
            
            return outputs


class OctoTransformer(nn.Module):
    """
    Octo Transformer architecture for clinical robot adaptation.
    
    This model implements the Octo foundation model with modifications for
    clinical environments and few-shot learning capabilities.
    """

    def __init__(self, vocab_size: int, image_size: Tuple[int, int], 
                 sequence_length: int, action_dim: int, **kwargs):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.image_size = image_size
        self.sequence_length = sequence_length
        self.action_dim = action_dim
        
        # Model dimensions
        self.hidden_dim = kwargs.get('hidden_dim', 512)
        self.num_heads = kwargs.get('num_heads', 8)
        self.num_layers = kwargs.get('num_layers', 6)
        self.dropout = kwargs.get('dropout', 0.1)
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(
            image_size=image_size,
            patch_size=kwargs.get('patch_size', 16),
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=kwargs.get('vision_layers', 4)
        )
        
        # Language encoder
        self.language_encoder = LanguageEncoder(
            vocab_size=vocab_size,
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=kwargs.get('language_layers', 4),
            max_length=sequence_length
        )
        
        # Robot state encoder
        self.robot_state_encoder = RobotStateEncoder(
            state_dim=kwargs.get('state_dim', 14),  # 7 joints + 7 velocities
            embed_dim=self.hidden_dim
        )
        
        # Multimodal fusion transformer
        self.fusion_transformer = MultimodalFusion(
            embed_dim=self.hidden_dim,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dropout=self.dropout
        )
        
        # Action prediction heads
        self.action_head = ActionPredictionHead(
            embed_dim=self.hidden_dim,
            action_dim=action_dim
        )
        
        self.grasp_head = GraspClassificationHead(
            embed_dim=self.hidden_dim,
            num_classes=5  # precision, power, pinch, etc.
        )
        
        self.human_awareness_head = HumanAwarenessHead(
            embed_dim=self.hidden_dim
        )
        
        self.safety_head = SafetyConstraintHead(
            embed_dim=self.hidden_dim
        )
    
    def forward(self, images: torch.Tensor, language_instructions: Dict[str, torch.Tensor],
                robot_states: torch.Tensor, actions: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the Octo model.
        
        Args:
            images: Batch of RGB images [B, C, H, W]
            language_instructions: Tokenized language instructions
            robot_states: Robot state tensors [B, state_dim]
            actions: Target actions for training [B, action_dim]
        
        Returns:
            Dictionary containing model outputs
        """
        batch_size = images.size(0)
        
        # Encode modalities
        vision_features = self.vision_encoder(images)  # [B, N_vision, hidden_dim]
        language_features = self.language_encoder(
            language_instructions['input_ids'],
            language_instructions['attention_mask']
        )  # [B, N_lang, hidden_dim]
        
        robot_features = self.robot_state_encoder(robot_states)  # [B, 1, hidden_dim]
        
        # Multimodal fusion
        fused_features = self.fusion_transformer(
            vision_features, language_features, robot_features
        )  # [B, N_total, hidden_dim]
        
        # Global pooling
        pooled_features = fused_features.mean(dim=1)  # [B, hidden_dim]
        
        # Prediction heads
        outputs = {}
        
        # Action prediction
        predicted_actions = self.action_head(pooled_features)
        outputs['predicted_actions'] = predicted_actions
        
        # Grasp type classification
        grasp_logits = self.grasp_head(pooled_features)
        outputs['grasp_logits'] = grasp_logits
        
        # Human awareness prediction
        human_awareness = self.human_awareness_head(pooled_features)
        outputs['human_awareness'] = human_awareness
        
        # Safety constraint prediction
        safety_violations = self.safety_head(pooled_features)
        outputs['safety_violations'] = safety_violations
        
        return outputs


class VisionEncoder(nn.Module):
    """Vision encoder using Vision Transformer architecture."""
    
    def __init__(self, image_size: Tuple[int, int], patch_size: int, 
                 embed_dim: int, num_heads: int, num_layers: int):
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        # Patch embedding
        self.patch_embed = nn.Conv2d(
            3, embed_dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Position embeddings
        num_patches = (image_size[0] // patch_size) * (image_size[1] // patch_size)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Patch embedding
        x = self.patch_embed(x)  # [B, embed_dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, embed_dim]
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class LanguageEncoder(nn.Module):
    """Language encoder using Transformer architecture."""
    
    def __init__(self, vocab_size: int, embed_dim: int, num_heads: int, 
                 num_layers: int, max_length: int):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Token embedding
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        
        # Position embeddings
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Token embedding
        x = self.token_embed(input_ids)
        
        # Add position embeddings
        seq_len = x.size(1)
        x = x + self.pos_embed[:, :seq_len, :]
        
        # Apply attention mask
        x = x.masked_fill(~attention_mask.bool().unsqueeze(-1), 0)
        
        # Transformer encoding
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class RobotStateEncoder(nn.Module):
    """Encoder for robot joint states and velocities."""
    
    def __init__(self, state_dim: int, embed_dim: int):
        super().__init__()
        
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # State embedding layers
        self.state_embed = nn.Sequential(
            nn.Linear(state_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [B, state_dim]
        x = self.state_embed(x)  # [B, embed_dim]
        x = x.unsqueeze(1)  # [B, 1, embed_dim]
        return x


class MultimodalFusion(nn.Module):
    """Multimodal fusion using cross-attention."""
    
    def __init__(self, embed_dim: int, num_heads: int, num_layers: int, dropout: float):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Cross-attention layers
        self.cross_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        # Feed-forward networks
        self.ffn_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim * 4, embed_dim),
                nn.LayerNorm(embed_dim)
            )
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, vision_features: torch.Tensor, language_features: torch.Tensor,
                robot_features: torch.Tensor) -> torch.Tensor:
        # Concatenate features
        x = torch.cat([vision_features, language_features, robot_features], dim=1)
        
        # Apply cross-attention and feed-forward layers
        for attn_layer, ffn_layer in zip(self.cross_attention_layers, self.ffn_layers):
            # Self-attention
            attn_out, _ = attn_layer(x, x, x)
            x = self.dropout(attn_out) + x
            
            # Feed-forward
            ffn_out = ffn_layer(x)
            x = x + ffn_out
        
        return x


class ActionPredictionHead(nn.Module):
    """Action prediction head for robot control."""
    
    def __init__(self, embed_dim: int, action_dim: int):
        super().__init__()
        
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, action_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.action_head(x)


class GraspClassificationHead(nn.Module):
    """Grasp type classification head."""
    
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        
        self.grasp_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.grasp_head(x)


class HumanAwarenessHead(nn.Module):
    """Human awareness prediction head."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.human_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.human_head(x).squeeze(-1)


class SafetyConstraintHead(nn.Module):
    """Safety constraint prediction head."""
    
    def __init__(self, embed_dim: int):
        super().__init__()
        
        self.safety_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, 1),
            nn.ReLU()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.safety_head(x).squeeze(-1)


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ClinicalRobotDataset(Dataset):
    """Dataset for clinical robot demonstrations."""
    
    def __init__(self, demonstrations: List[Dict], tokenizer, image_size: Tuple[int, int], 
                 sequence_length: int):
        self.demonstrations = demonstrations
        self.tokenizer = tokenizer
        self.image_size = image_size
        self.sequence_length = sequence_length
        
        # Image transforms
        self.image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Grasp type mapping
        self.grasp_type_map = {
            'precision': 0,
            'power': 1,
            'pinch': 2,
            'lateral': 3,
            'cylindrical': 4
        }
    
    def __len__(self) -> int:
        return len(self.demonstrations)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        demo = self.demonstrations[idx]
        
        # Process images
        if isinstance(demo['images'], list):
            # Take the first image from the sequence
            image = demo['images'][0]
        else:
            image = demo['images']
        
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image).convert('RGB')
        
        image = self.image_transform(image)
        
        # Process language instruction
        instruction = demo.get('instruction', 'Retrieve medication from shelf')
        
        # Process robot state
        robot_state = np.array(demo.get('robot_state', np.zeros(14)))
        
        # Process actions
        actions = np.array(demo.get('actions', np.zeros(7)))
        
        # Process grasp type
        grasp_type_str = demo.get('grasp_type', 'precision')
        grasp_type = self.grasp_type_map.get(grasp_type_str, 0)
        
        # Process human presence
        human_present = demo.get('human_present', False)
        
        return {
            'images': image,
            'instructions': instruction,
            'robot_states': torch.FloatTensor(robot_state),
            'actions': torch.FloatTensor(actions),
            'grasp_types': torch.LongTensor([grasp_type]),
            'human_present': torch.FloatTensor([float(human_present)])
        }


# Utility functions for clinical robot adaptation
def create_clinical_config() -> Dict[str, Any]:
    """Create default configuration for clinical robot adaptation."""
    return {
        # Model configuration
        'model_name': 'octo-clinical-v1',
        'image_size': (224, 224),
        'sequence_length': 32,
        'action_dim': 7,
        'hidden_dim': 512,
        'num_heads': 8,
        'num_layers': 6,
        'dropout': 0.1,
        'patch_size': 16,
        'vision_layers': 4,
        'language_layers': 4,
        'state_dim': 14,
        
        # Training configuration
        'batch_size': 16,
        'learning_rate': 1e-4,
        'num_epochs': 100,
        'patience': 10,
        'weight_decay': 1e-4,
        'T_0': 10,
        'T_mult': 2,
        
        # Loss weights
        'action_loss_weight': 1.0,
        'grasp_loss_weight': 0.5,
        'human_loss_weight': 0.3,
        'safety_loss_weight': 2.0,
        
        # Clinical parameters
        'min_demonstrations': 50,
        'max_demonstrations': 200,
        'quality_threshold': 0.85,
        'safety_margin': 0.1,
        
        # Layer freezing for few-shot learning
        'frozen_layers': [
            'vision_encoder.backbone',
            'language_encoder.embeddings',
            'transformer.layers.0',
            'transformer.layers.1',
            'transformer.layers.2'
        ],
        
        # Data configuration
        'val_split': 0.2,
        'use_wandb': False,
        'save_dir': './checkpoints'
    }


if __name__ == "__main__":
    # Example usage
    config = create_clinical_config()
    
    # Initialize adapter
    adapter = ClinicalOctoAdapter(config=config)
    
    # Load demonstration data
    demonstrations = adapter._load_demonstrations('./data/demonstrations')
    
    # Prepare data
    train_dataset, val_dataset = adapter.prepare_clinical_dataset('./data/demonstrations')
    
    # Train model
    results = adapter.finetune()
    
    # Save results
    with open('training_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("Clinical robot adaptation completed successfully!")
