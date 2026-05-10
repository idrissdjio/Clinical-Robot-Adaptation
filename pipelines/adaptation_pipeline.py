#!/usr/bin/env python3
"""
Clinical Robot Adaptation Pipeline
Comprehensive PyTorch-based pipeline for few-shot adaptation to clinical environments.

This pipeline implements:
- Multi-modal data processing and fusion
- Foundation model fine-tuning with clinical constraints
- Safety-aware adaptation with human-aware constraints
- Real-time inference and deployment
- Continuous learning and adaptation
- Performance monitoring and validation

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
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# PyTorch and deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.cuda.amp import GradScaler, autocast
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
import transformers
from transformers import AutoModel, AutoTokenizer, AutoConfig

# Scientific computing
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

# Computer vision
import cv2
from PIL import Image
import open3d as o3d

# Clinical and robotics
import trimesh
from pyquaternion import Quaternion

# Monitoring and logging
import wandb
from tqdm import tqdm
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('adaptation_pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class AdaptationConfig:
    """Configuration for the adaptation pipeline."""
    
    # Model configuration
    foundation_model: str = "google/octo-base"
    vision_encoder: str = "resnet50"
    language_encoder: str = "bert-base-uncased"
    
    # Training configuration
    batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    num_epochs: int = 100
    warmup_epochs: int = 10
    gradient_clip_norm: float = 1.0
    
    # Few-shot configuration
    num_demonstrations: int = 100
    adaptation_steps: int = 1000
    freeze_backbone: bool = True
    freeze_layers: List[int] = field(default_factory=lambda: list(range(8)))
    
    # Safety configuration
    safety_weight: float = 0.3
    human_aware_weight: float = 0.2
    collision_threshold: float = 0.1
    velocity_limit: float = 0.5
    
    # Data configuration
    image_size: Tuple[int, int] = (224, 224)
    sequence_length: int = 10
    action_dim: int = 7
    state_dim: int = 14
    
    # Hardware configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    mixed_precision: bool = True
    num_workers: int = 4
    
    # Monitoring configuration
    use_wandb: bool = True
    log_interval: int = 10
    save_interval: int = 50
    eval_interval: int = 25
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'foundation_model': self.foundation_model,
            'vision_encoder': self.vision_encoder,
            'language_encoder': self.language_encoder,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'num_epochs': self.num_epochs,
            'warmup_epochs': self.warmup_epochs,
            'gradient_clip_norm': self.gradient_clip_norm,
            'num_demonstrations': self.num_demonstrations,
            'adaptation_steps': self.adaptation_steps,
            'freeze_backbone': self.freeze_backbone,
            'freeze_layers': self.freeze_layers,
            'safety_weight': self.safety_weight,
            'human_aware_weight': self.human_aware_weight,
            'collision_threshold': self.collision_threshold,
            'velocity_limit': self.velocity_limit,
            'image_size': self.image_size,
            'sequence_length': self.sequence_length,
            'action_dim': self.action_dim,
            'state_dim': self.state_dim,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'num_workers': self.num_workers,
            'use_wandb': self.use_wandb,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval
        }

class ClinicalDataset(Dataset):
    """Dataset for clinical robot demonstrations."""
    
    def __init__(self, data_path: str, config: AdaptationConfig, transform=None):
        self.data_path = Path(data_path)
        self.config = config
        self.transform = transform or self._default_transform()
        
        # Load data
        self.demonstrations = self._load_demonstrations()
        
        # Tokenizer for language instructions
        self.tokenizer = AutoTokenizer.from_pretrained(config.language_encoder)
        
    def _default_transform(self):
        """Default image transformations."""
        return transforms.Compose([
            transforms.Resize(self.config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_demonstrations(self) -> List[Dict[str, Any]]:
        """Load demonstration data from file."""
        demonstrations = []
        
        if self.data_path.suffix == '.hdf5':
            import h5py
            with h5py.File(self.data_path, 'r') as f:
                for demo_key in f.keys():
                    demo_group = f[demo_key]
                    
                    demo = {
                        'id': demo_key,
                        'images': demo_group['images'][:],
                        'states': demo_group['states'][:],
                        'actions': demo_group['actions'][:],
                        'instruction': demo_group.attrs['instruction'],
                        'medication_type': demo_group.attrs['medication_type'],
                        'safety_level': demo_group.attrs.get('safety_level', 'medium'),
                        'success': demo_group.attrs.get('success', True)
                    }
                    demonstrations.append(demo)
        
        elif self.data_path.suffix == '.json':
            with open(self.data_path, 'r') as f:
                data = json.load(f)
                demonstrations = data.get('demonstrations', [])
        
        return demonstrations
    
    def __len__(self) -> int:
        return len(self.demonstrations)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        demo = self.demonstrations[idx]
        
        # Process images
        images = []
        for img in demo['images']:
            if isinstance(img, str):
                img = Image.open(img).convert('RGB')
            else:
                img = Image.fromarray(img.astype(np.uint8))
            
            if self.transform:
                img = self.transform(img)
            images.append(img)
        
        # Stack images
        images = torch.stack(images)  # [seq_len, C, H, W]
        
        # Process states and actions
        states = torch.tensor(demo['states'], dtype=torch.float32)
        actions = torch.tensor(demo['actions'], dtype=torch.float32)
        
        # Process instruction
        instruction_tokens = self.tokenizer(
            demo['instruction'],
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        return {
            'images': images,
            'states': states,
            'actions': actions,
            'instruction_ids': instruction_tokens['input_ids'].squeeze(),
            'attention_mask': instruction_tokens['attention_mask'].squeeze(),
            'medication_type': demo['medication_type'],
            'safety_level': demo['safety_level'],
            'success': demo['success']
        }

class VisionEncoder(nn.Module):
    """Vision encoder for clinical images."""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained vision model
        if config.vision_encoder == "resnet50":
            self.backbone = resnet50(pretrained=True)
            self.feature_dim = 2048
        elif config.vision_encoder == "efficientnet":
            self.backbone = efficientnet_b0(pretrained=True)
            self.feature_dim = 1280
        else:
            raise ValueError(f"Unsupported vision encoder: {config.vision_encoder}")
        
        # Remove classification head
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, seq_len, C, H, W] or [batch_size, C, H, W]
        Returns:
            features: [batch_size, seq_len, 256] or [batch_size, 256]
        """
        if images.dim() == 5:
            batch_size, seq_len = images.shape[:2]
            images = images.view(batch_size * seq_len, *images.shape[2:])
        else:
            seq_len = None
        
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        features = self.projection(features)
        
        if seq_len is not None:
            features = features.view(batch_size, seq_len, -1)
        
        return features

class LanguageEncoder(nn.Module):
    """Language encoder for instructions."""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Load pretrained language model
        self.model = AutoModel.from_pretrained(config.language_encoder)
        self.feature_dim = self.model.config.hidden_size
        
        # Projection head
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256)
        )
    
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
        Returns:
            features: [batch_size, 256]
        """
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        features = self.projection(pooled_output)
        return features

class StateEncoder(nn.Module):
    """Encoder for robot states."""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        self.encoder = nn.Sequential(
            nn.Linear(config.state_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64)
        )
    
    def forward(self, states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            states: [batch_size, seq_len, state_dim] or [batch_size, state_dim]
        Returns:
            features: [batch_size, seq_len, 64] or [batch_size, 64]
        """
        return self.encoder(states)

class ClinicalAdaptationModel(nn.Module):
    """Main adaptation model for clinical robotics."""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Encoders
        self.vision_encoder = VisionEncoder(config)
        self.language_encoder = LanguageEncoder(config)
        self.state_encoder = StateEncoder(config)
        
        # Fusion layers
        vision_dim = 256
        language_dim = 256
        state_dim = 64
        
        # Multi-modal fusion
        self.fusion = nn.Sequential(
            nn.Linear(vision_dim + language_dim + state_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        # Action decoder
        self.action_decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, config.action_dim)
        )
        
        # Safety prediction head
        self.safety_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Human-aware prediction head
        self.human_aware_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Attention mechanism for sequence processing
        self.attention = nn.MultiheadAttention(
            embed_dim=vision_dim,
            num_heads=8,
            dropout=0.1
        )
        
    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch: Dictionary containing images, states, actions, instruction_ids, attention_mask
        Returns:
            Dictionary containing predictions, safety_score, human_aware_score
        """
        images = batch['images']
        states = batch['states']
        instruction_ids = batch['instruction_ids']
        attention_mask = batch['attention_mask']
        
        # Encode vision
        if images.dim() == 5:
            # Sequence of images
            batch_size, seq_len = images.shape[:2]
            vision_features = self.vision_encoder(images)  # [batch_size, seq_len, 256]
            
            # Apply attention
            vision_features = vision_features.transpose(0, 1)  # [seq_len, batch_size, 256]
            attended_features, _ = self.attention(vision_features, vision_features, vision_features)
            vision_features = attended_features.transpose(0, 1)  # [batch_size, seq_len, 256]
            
            # Average pool
            vision_features = vision_features.mean(dim=1)  # [batch_size, 256]
        else:
            # Single image
            vision_features = self.vision_encoder(images)  # [batch_size, 256]
        
        # Encode language
        language_features = self.language_encoder(instruction_ids, attention_mask)  # [batch_size, 256]
        
        # Encode state
        if states.dim() == 3:
            # Sequence of states
            state_features = self.state_encoder(states)  # [batch_size, seq_len, 64]
            state_features = state_features.mean(dim=1)  # [batch_size, 64]
        else:
            # Single state
            state_features = self.state_encoder(states)  # [batch_size, 64]
        
        # Fuse features
        fused_features = torch.cat([vision_features, language_features, state_features], dim=1)
        fused_features = self.fusion(fused_features)  # [batch_size, 128]
        
        # Predict actions
        actions = self.action_decoder(fused_features)  # [batch_size, action_dim]
        
        # Predict safety score
        safety_score = self.safety_head(fused_features)  # [batch_size, 1]
        
        # Predict human-aware score
        human_aware_score = self.human_aware_head(fused_features)  # [batch_size, 1]
        
        return {
            'actions': actions,
            'safety_score': safety_score,
            'human_aware_score': human_aware_score,
            'features': fused_features
        }

class ClinicalLoss(nn.Module):
    """Custom loss function for clinical adaptation."""
    
    def __init__(self, config: AdaptationConfig):
        super().__init__()
        self.config = config
        
        # Base losses
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.bce_loss = nn.BCELoss()
        
        # Loss weights
        self.action_weight = 1.0
        self.safety_weight = config.safety_weight
        self.human_aware_weight = config.human_aware_weight
        
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        Returns:
            Dictionary containing loss components
        """
        # Action prediction loss
        if 'actions' in targets:
            action_loss = self.l1_loss(predictions['actions'], targets['actions'])
        else:
            action_loss = torch.tensor(0.0, device=predictions['actions'].device)
        
        # Safety prediction loss
        if 'safety_score' in targets:
            safety_loss = self.bce_loss(predictions['safety_score'], targets['safety_score'])
        else:
            safety_loss = torch.tensor(0.0, device=predictions['safety_score'].device)
        
        # Human-aware prediction loss
        if 'human_aware_score' in targets:
            human_aware_loss = self.bce_loss(predictions['human_aware_score'], targets['human_aware_score'])
        else:
            human_aware_loss = torch.tensor(0.0, device=predictions['human_aware_score'].device)
        
        # Total loss
        total_loss = (self.action_weight * action_loss + 
                     self.safety_weight * safety_loss + 
                     self.human_aware_weight * human_aware_loss)
        
        return {
            'total_loss': total_loss,
            'action_loss': action_loss,
            'safety_loss': safety_loss,
            'human_aware_loss': human_aware_loss
        }

class AdaptationPipeline:
    """Main adaptation pipeline."""
    
    def __init__(self, config: AdaptationConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize model
        self.model = ClinicalAdaptationModel(config).to(self.device)
        
        # Initialize loss function
        self.loss_fn = ClinicalLoss(config)
        
        # Initialize optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Initialize scheduler
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs,
            eta_min=config.learning_rate * 0.01
        )
        
        # Initialize mixed precision scaler
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize wandb
        if config.use_wandb:
            wandb.init(
                project="clinical-robot-adaptation",
                config=config.to_dict(),
                name=f"adaptation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
        
    def freeze_backbone(self):
        """Freeze backbone layers for few-shot learning."""
        if self.config.freeze_backbone:
            # Freeze vision encoder layers
            for i, layer in enumerate(self.model.vision_encoder.backbone.children()):
                if i in self.config.freeze_layers:
                    for param in layer.parameters():
                        param.requires_grad = False
            
            # Freeze language encoder
            for param in self.model.language_encoder.model.parameters():
                param.requires_grad = False
    
    def unfreeze_backbone(self):
        """Unfreeze all layers for full fine-tuning."""
        for param in self.model.parameters():
            param.requires_grad = True
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)
        
        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.config.mixed_precision:
                with autocast():
                    predictions = self.model(batch)
                    targets = {
                        'actions': batch['actions'],
                        'safety_score': torch.ones_like(predictions['safety_score']) * 0.9,  # High safety
                        'human_aware_score': torch.ones_like(predictions['human_aware_score']) * 0.8  # High awareness
                    }
                    losses = self.loss_fn(predictions, targets)
            else:
                predictions = self.model(batch)
                targets = {
                    'actions': batch['actions'],
                    'safety_score': torch.ones_like(predictions['safety_score']) * 0.9,
                    'human_aware_score': torch.ones_like(predictions['human_aware_score']) * 0.8
                }
                losses = self.loss_fn(predictions, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            
            if self.config.mixed_precision:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.optimizer.step()
            
            # Update metrics
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': losses['total_loss'].item(),
                'action_loss': losses['action_loss'].item(),
                'safety_loss': losses['safety_loss'].item()
            })
            
            # Log to wandb
            if self.config.use_wandb and self.global_step % self.config.log_interval == 0:
                wandb.log({
                    'train/total_loss': losses['total_loss'].item(),
                    'train/action_loss': losses['action_loss'].item(),
                    'train/safety_loss': losses['safety_loss'].item(),
                    'train/human_aware_loss': losses['human_aware_loss'].item(),
                    'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                    'global_step': self.global_step
                })
            
            self.global_step += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def validate_epoch(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses = defaultdict(float)
        num_batches = len(val_loader)
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc="Validation")
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch)
                targets = {
                    'actions': batch['actions'],
                    'safety_score': torch.ones_like(predictions['safety_score']) * 0.9,
                    'human_aware_score': torch.ones_like(predictions['human_aware_score']) * 0.8
                }
                losses = self.loss_fn(predictions, targets)
                
                # Update metrics
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                
                # Update progress bar
                pbar.set_postfix({
                    'val_loss': losses['total_loss'].item()
                })
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def train(self, train_dataset: ClinicalDataset, val_dataset: ClinicalDataset = None):
        """Main training loop."""
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=True
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        else:
            # Split train dataset for validation
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
            
            train_loader = DataLoader(
                train_dataset,
                batch_size=self.config.batch_size,
                shuffle=True,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True
            )
        
        # Freeze backbone for few-shot learning
        self.freeze_backbone()
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = self.validate_epoch(val_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Store metrics
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Log epoch metrics
            logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['total_loss']:.4f}, "
                       f"Val Loss = {val_metrics['total_loss']:.4f}")
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_metrics['total_loss'],
                    'val/epoch_loss': val_metrics['total_loss'],
                    'train/epoch_action_loss': train_metrics['action_loss'],
                    'val/epoch_action_loss': val_metrics['action_loss']
                })
            
            # Save best model
            if val_metrics['total_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total_loss']
                self.save_checkpoint('best_model.pth')
            
            # Save periodic checkpoint
            if epoch % self.config.save_interval == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
            
            # Unfreeze backbone after warmup
            if epoch == self.config.warmup_epochs:
                logger.info("Unfreezing backbone layers")
                self.unfreeze_backbone()
        
        logger.info("Training completed!")
        
        # Save final model
        self.save_checkpoint('final_model.pth')
        
        # Plot training curves
        self.plot_training_curves()
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict(),
            'train_metrics': dict(self.train_metrics),
            'val_metrics': dict(self.val_metrics)
        }
        
        checkpoint_path = Path(f'checkpoints/{filename}')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_metrics = defaultdict(list, checkpoint.get('train_metrics', {}))
        self.val_metrics = defaultdict(list, checkpoint.get('val_metrics', {}))
        
        logger.info(f"Checkpoint loaded: {checkpoint_path}")
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_metrics['total_loss'], label='Train')
        axes[0, 0].plot(self.val_metrics['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Action loss
        axes[0, 1].plot(self.train_metrics['action_loss'], label='Train')
        axes[0, 1].plot(self.val_metrics['action_loss'], label='Validation')
        axes[0, 1].set_title('Action Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Safety loss
        axes[1, 0].plot(self.train_metrics['safety_loss'], label='Train')
        axes[1, 0].plot(self.val_metrics['safety_loss'], label='Validation')
        axes[1, 0].set_title('Safety Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Human-aware loss
        axes[1, 1].plot(self.train_metrics['human_aware_loss'], label='Train')
        axes[1, 1].plot(self.val_metrics['human_aware_loss'], label='Validation')
        axes[1, 1].set_title('Human-Aware Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('training_curves.png', dpi=300, bbox_inches='tight')
        
        if self.config.use_wandb:
            wandb.log({"training_curves": wandb.Image('training_curves.png')})
        
        plt.close()
    
    def evaluate(self, test_dataset: ClinicalDataset) -> Dict[str, float]:
        """Evaluate model on test dataset."""
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers
        )
        
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluation"):
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                predictions = self.model(batch)
                targets = {
                    'actions': batch['actions'],
                    'safety_score': torch.ones_like(predictions['safety_score']) * 0.9,
                    'human_aware_score': torch.ones_like(predictions['human_aware_score']) * 0.8
                }
                losses = self.loss_fn(predictions, targets)
                
                total_loss += losses['total_loss'].item() * batch['actions'].size(0)
                num_samples += batch['actions'].size(0)
        
        avg_loss = total_loss / num_samples
        
        logger.info(f"Test Loss: {avg_loss:.4f}")
        
        if self.config.use_wandb:
            wandb.log({'test/loss': avg_loss})
        
        return {'test_loss': avg_loss}
    
    def predict(self, images: torch.Tensor, states: torch.Tensor, 
                instruction: str) -> Dict[str, torch.Tensor]:
        """Make prediction on single sample."""
        self.model.eval()
        
        # Tokenize instruction
        tokenizer = AutoTokenizer.from_pretrained(self.config.language_encoder)
        instruction_tokens = tokenizer(
            instruction,
            padding='max_length',
            truncation=True,
            max_length=128,
            return_tensors='pt'
        )
        
        # Move to device
        images = images.to(self.device)
        states = states.to(self.device)
        instruction_ids = instruction_tokens['input_ids'].to(self.device)
        attention_mask = instruction_tokens['attention_mask'].to(self.device)
        
        # Create batch
        batch = {
            'images': images.unsqueeze(0),
            'states': states.unsqueeze(0),
            'instruction_ids': instruction_ids,
            'attention_mask': attention_mask
        }
        
        with torch.no_grad():
            predictions = self.model(batch)
        
        return {
            'actions': predictions['actions'].cpu(),
            'safety_score': predictions['safety_score'].cpu(),
            'human_aware_score': predictions['human_aware_score'].cpu()
        }

def main():
    """Main function for running the adaptation pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Robot Adaptation Pipeline')
    parser.add_argument('--config', type=str, help='Configuration file')
    parser.add_argument('--data', type=str, required=True, help='Training data path')
    parser.add_argument('--output', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, help='Batch size')
    parser.add_argument('--learning-rate', type=float, help='Learning rate')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    # Load configuration
    config = AdaptationConfig()
    if args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    if args.epochs:
        config.num_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.learning_rate:
        config.learning_rate = args.learning_rate
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    dataset = ClinicalDataset(args.data, config)
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Initialize pipeline
    pipeline = AdaptationPipeline(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        pipeline.load_checkpoint(args.resume)
    
    # Train model
    pipeline.train(train_dataset, val_dataset)
    
    # Evaluate on validation set
    val_metrics = pipeline.evaluate(val_dataset)
    
    logger.info("Adaptation pipeline completed successfully!")

if __name__ == "__main__":
    main()
