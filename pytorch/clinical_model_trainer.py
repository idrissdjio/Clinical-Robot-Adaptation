#!/usr/bin/env python3
"""
PyTorch Clinical Model Trainer
Advanced PyTorch-based training system for clinical robot adaptation models.

This module implements:
- Multi-GPU distributed training
- Mixed precision training with AMP
- Advanced loss functions for clinical constraints
- Learning rate scheduling and optimization
- Model checkpointing and versioning
- Training monitoring and visualization
- Hyperparameter tuning support
- Clinical data augmentation

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import warnings

# PyTorch and distributed training
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

# Optimization and scheduling
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau, OneCycleLR

# Monitoring and visualization
import wandb
import tensorboard
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

# Scientific computing
import numpy as np
import pandas as pd
from scipy import stats

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.adaptation_pipeline import ClinicalAdaptationModel, ClinicalDataset, AdaptationConfig
from models.gan_synthetic_data import ClinicalSyntheticDataPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_model_trainer.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

@dataclass
class TrainingConfig:
    """Configuration for model training."""
    
    # Model configuration
    model_type: str = "clinical_adaptation"
    foundation_model: str = "google/octo-base"
    pretrained_path: str = ""
    
    # Training configuration
    batch_size: int = 32
    num_epochs: int = 100
    learning_rate: float = 1e-4
    weight_decay: float = 1e-5
    warmup_epochs: int = 10
    max_grad_norm: float = 1.0
    
    # Optimization configuration
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    momentum: float = 0.9
    beta1: float = 0.9
    beta2: float = 0.999
    
    # Loss configuration
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'action': 1.0,
        'safety': 0.3,
        'human_aware': 0.2,
        'temporal_consistency': 0.1
    })
    
    # Data configuration
    data_path: str = ""
    val_split: float = 0.2
    num_workers: int = 4
    pin_memory: bool = True
    drop_last: bool = True
    
    # Augmentation configuration
    use_augmentation: bool = True
    augmentation_prob: float = 0.5
    synthetic_data_ratio: float = 0.0
    
    # Hardware configuration
    device: str = "cuda"
    mixed_precision: bool = True
    compile_model: bool = True
    find_unused_parameters: bool = False
    
    # Distributed training
    distributed: bool = False
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    backend: str = "nccl"
    
    # Monitoring and checkpointing
    log_interval: int = 10
    save_interval: int = 25
    eval_interval: int = 10
    keep_best_k: int = 3
    early_stopping_patience: int = 20
    
    # Experiment tracking
    use_wandb: bool = True
    use_tensorboard: bool = True
    project_name: str = "clinical-robot-adaptation"
    experiment_name: str = ""
    
    # Advanced features
    gradient_accumulation_steps: int = 1
    ema_decay: float = 0.999
    use_ema: bool = False
    label_smoothing: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'model_type': self.model_type,
            'foundation_model': self.foundation_model,
            'pretrained_path': self.pretrained_path,
            'batch_size': self.batch_size,
            'num_epochs': self.num_epochs,
            'learning_rate': self.learning_rate,
            'weight_decay': self.weight_decay,
            'warmup_epochs': self.warmup_epochs,
            'max_grad_norm': self.max_grad_norm,
            'optimizer': self.optimizer,
            'scheduler': self.scheduler,
            'momentum': self.momentum,
            'beta1': self.beta1,
            'beta2': self.beta2,
            'loss_weights': self.loss_weights,
            'data_path': self.data_path,
            'val_split': self.val_split,
            'num_workers': self.num_workers,
            'pin_memory': self.pin_memory,
            'drop_last': self.drop_last,
            'use_augmentation': self.use_augmentation,
            'augmentation_prob': self.augmentation_prob,
            'synthetic_data_ratio': self.synthetic_data_ratio,
            'device': self.device,
            'mixed_precision': self.mixed_precision,
            'compile_model': self.compile_model,
            'find_unused_parameters': self.find_unused_parameters,
            'distributed': self.distributed,
            'world_size': self.world_size,
            'rank': self.rank,
            'local_rank': self.local_rank,
            'backend': self.backend,
            'log_interval': self.log_interval,
            'save_interval': self.save_interval,
            'eval_interval': self.eval_interval,
            'keep_best_k': self.keep_best_k,
            'early_stopping_patience': self.early_stopping_patience,
            'use_wandb': self.use_wandb,
            'use_tensorboard': self.use_tensorboard,
            'project_name': self.project_name,
            'experiment_name': self.experiment_name,
            'gradient_accumulation_steps': self.gradient_accumulation_steps,
            'ema_decay': self.ema_decay,
            'use_ema': self.use_ema,
            'label_smoothing': self.label_smoothing
        }

class ClinicalLossFunction(nn.Module):
    """Advanced loss function for clinical robot adaptation."""
    
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.loss_weights = config.loss_weights
        
        # Base loss functions
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()
        self.huber_loss = nn.HuberLoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        
        # Advanced losses
        self.temporal_consistency_loss = TemporalConsistencyLoss()
        self.safety_margin_loss = SafetyMarginLoss()
        self.human_aware_loss = HumanAwareLoss()
        
        # Label smoothing
        self.label_smoothing = config.label_smoothing
    
    def forward(self, predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for clinical adaptation.
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dictionary containing loss components
        """
        device = predictions['actions'].device
        total_loss = torch.tensor(0.0, device=device)
        loss_components = {}
        
        # Action prediction loss
        if 'actions' in targets:
            action_loss = self._compute_action_loss(predictions['actions'], targets['actions'])
            total_loss += self.loss_weights['action'] * action_loss
            loss_components['action_loss'] = action_loss
        
        # Safety prediction loss
        if 'safety_score' in targets:
            safety_loss = self._compute_safety_loss(predictions['safety_score'], targets['safety_score'])
            total_loss += self.loss_weights['safety'] * safety_loss
            loss_components['safety_loss'] = safety_loss
        
        # Human-aware prediction loss
        if 'human_aware_score' in targets:
            human_loss = self._compute_human_aware_loss(
                predictions['human_aware_score'], targets['human_aware_score']
            )
            total_loss += self.loss_weights['human_aware'] * human_loss
            loss_components['human_aware_loss'] = human_loss
        
        # Temporal consistency loss
        if 'sequence_predictions' in predictions and 'sequence_targets' in targets:
            temporal_loss = self.temporal_consistency_loss(
                predictions['sequence_predictions'], targets['sequence_targets']
            )
            total_loss += self.loss_weights['temporal_consistency'] * temporal_loss
            loss_components['temporal_consistency_loss'] = temporal_loss
        
        # Safety margin loss
        if 'safety_margin_loss' in predictions:
            margin_loss = self.safety_margin_loss(predictions['safety_margin_loss'])
            total_loss += 0.1 * margin_loss
            loss_components['safety_margin_loss'] = margin_loss
        
        loss_components['total_loss'] = total_loss
        
        return loss_components
    
    def _compute_action_loss(self, pred_actions: torch.Tensor, target_actions: torch.Tensor) -> torch.Tensor:
        """Compute action prediction loss."""
        # Use Huber loss for robustness to outliers
        return self.huber_loss(pred_actions, target_actions)
    
    def _compute_safety_loss(self, pred_safety: torch.Tensor, target_safety: torch.Tensor) -> torch.Tensor:
        """Compute safety prediction loss."""
        # Use focal loss for imbalanced safety predictions
        return self.focal_loss(pred_safety.squeeze(), target_safety.squeeze())
    
    def _compute_human_aware_loss(self, pred_human: torch.Tensor, target_human: torch.Tensor) -> torch.Tensor:
        """Compute human-aware prediction loss."""
        # Use BCE with logits for binary classification
        return self.bce_loss(pred_human.squeeze(), target_human.squeeze())

class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""
    
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = 'mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class TemporalConsistencyLoss(nn.Module):
    """Loss for enforcing temporal consistency in predictions."""
    
    def __init__(self, consistency_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Enforce temporal consistency between consecutive predictions.
        
        Args:
            predictions: [batch_size, seq_len, feature_dim]
            targets: [batch_size, seq_len, feature_dim]
        
        Returns:
            Temporal consistency loss
        """
        # Compute temporal differences
        pred_diff = predictions[:, 1:] - predictions[:, :-1]
        target_diff = targets[:, 1:] - targets[:, :-1]
        
        # Consistency loss
        consistency_loss = F.mse_loss(pred_diff, target_diff)
        
        return self.consistency_weight * consistency_loss

class SafetyMarginLoss(nn.Module):
    """Loss for maintaining safety margins in predictions."""
    
    def __init__(self, margin: float = 0.1):
        super().__init__()
        self.margin = margin
    
    def forward(self, safety_predictions: torch.Tensor) -> torch.Tensor:
        """
        Penalize predictions that are too close to safety boundaries.
        
        Args:
            safety_predictions: Safety prediction scores
        
        Returns:
            Safety margin loss
        """
        # Penalize predictions close to 0.5 (decision boundary)
        margin_loss = torch.abs(safety_predictions - 0.5)
        margin_loss = torch.where(margin_loss < self.margin, 
                                 (self.margin - margin_loss) ** 2, 
                                 torch.zeros_like(margin_loss))
        
        return margin_loss.mean()

class HumanAwareLoss(nn.Module):
    """Loss for human-aware robot behavior."""
    
    def __init__(self, awareness_weight: float = 1.0):
        super().__init__()
        self.awareness_weight = awareness_weight
    
    def forward(self, human_predictions: torch.Tensor, human_targets: torch.Tensor) -> torch.Tensor:
        """
        Compute human-aware loss with emphasis on safety-critical situations.
        
        Args:
            human_predictions: Human-aware predictions
            human_targets: Human-aware targets
        
        Returns:
            Human-aware loss
        """
        # Weight loss more heavily when humans are present
        weights = human_targets + 0.1  # Ensure minimum weight
        weighted_loss = F.binary_cross_entropy_with_logits(
            human_predictions.squeeze(), 
            human_targets.squeeze(), 
            weight=weights
        )
        
        return self.awareness_weight * weighted_loss

class ClinicalDataAugmentation:
    """Data augmentation for clinical robot data."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.augmentation_prob = config.augmentation_prob
    
    def augment_batch(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Apply data augmentation to a batch."""
        if not self.config.use_augmentation:
            return batch
        
        augmented_batch = batch.copy()
        
        # Image augmentation
        if 'images' in batch:
            augmented_batch['images'] = self._augment_images(batch['images'])
        
        # State augmentation
        if 'states' in batch:
            augmented_batch['states'] = self._augment_states(batch['states'])
        
        # Action augmentation
        if 'actions' in batch:
            augmented_batch['actions'] = self._augment_actions(batch['actions'])
        
        return augmented_batch
    
    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        """Apply image augmentation."""
        # Random brightness adjustment
        if torch.rand(1) < self.augmentation_prob:
            brightness_factor = 0.8 + 0.4 * torch.rand(1)
            images = images * brightness_factor
            images = torch.clamp(images, 0, 1)
        
        # Random Gaussian noise
        if torch.rand(1) < self.augmentation_prob:
            noise = torch.randn_like(images) * 0.01
            images = images + noise
            images = torch.clamp(images, 0, 1)
        
        return images
    
    def _augment_states(self, states: torch.Tensor) -> torch.Tensor:
        """Apply state augmentation."""
        # Add small Gaussian noise to joint states
        if torch.rand(1) < self.augmentation_prob:
            noise = torch.randn_like(states) * 0.01
            states = states + noise
        
        return states
    
    def _augment_actions(self, actions: torch.Tensor) -> torch.Tensor:
        """Apply action augmentation."""
        # Add small Gaussian noise to actions
        if torch.rand(1) < self.augmentation_prob:
            noise = torch.randn_like(actions) * 0.005
            actions = actions + noise
        
        return actions

class EMA:
    """Exponential Moving Average for model weights."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # Initialize shadow weights
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        """Update EMA weights."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = self.decay * self.shadow[name] + (1 - self.decay) * param.data
    
    def apply_shadow(self, model: nn.Module):
        """Apply EMA weights to model."""
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        """Restore original weights."""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}

class ClinicalModelTrainer:
    """Advanced PyTorch trainer for clinical robot adaptation."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # Initialize distributed training
        if config.distributed:
            self._setup_distributed()
        
        # Initialize model
        self.model = self._create_model()
        self.ema = EMA(self.model, config.ema_decay) if config.use_ema else None
        
        # Initialize loss function
        self.loss_fn = ClinicalLossFunction(config)
        
        # Initialize optimizer
        self.optimizer = self._create_optimizer()
        
        # Initialize scheduler
        self.scheduler = self._create_scheduler()
        
        # Initialize mixed precision
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Initialize data augmentation
        self.augmentation = ClinicalDataAugmentation(config)
        
        # Initialize monitoring
        self._initialize_monitoring()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        
        # Checkpoints
        self.checkpoints = []
        
        logger.info("Clinical Model Trainer initialized")
    
    def _setup_distributed(self):
        """Setup distributed training."""
        if self.config.distributed:
            dist.init_process_group(
                backend=self.config.backend,
                init_method='env://',
                world_size=self.config.world_size,
                rank=self.config.rank
            )
            
            # Set device for this process
            torch.cuda.set_device(self.config.local_rank)
            self.device = torch.device(f'cuda:{self.config.local_rank}')
    
    def _create_model(self) -> nn.Module:
        """Create the model."""
        # Create adaptation config
        adaptation_config = AdaptationConfig(
            foundation_model=self.config.foundation_model,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            device=str(self.device),
            mixed_precision=self.config.mixed_precision
        )
        
        # Create model
        model = ClinicalAdaptationModel(adaptation_config)
        
        # Load pretrained weights if specified
        if self.config.pretrained_path and Path(self.config.pretrained_path).exists():
            checkpoint = torch.load(self.config.pretrained_path, map_location=self.device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Loaded pretrained weights from {self.config.pretrained_path}")
        
        # Move to device
        model = model.to(self.device)
        
        # Wrap with DDP for distributed training
        if self.config.distributed:
            model = DDP(model, device_ids=[self.config.local_rank], 
                       find_unused_parameters=self.config.find_unused_parameters)
        
        # Compile model for better performance (PyTorch 2.0+)
        if self.config.compile_model and hasattr(torch, 'compile'):
            try:
                model = torch.compile(model)
                logger.info("Model compiled for better performance")
            except Exception as e:
                logger.warning(f"Model compilation failed: {e}")
        
        return model
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer."""
        if self.config.optimizer.lower() == "adamw":
            return AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2)
            )
        elif self.config.optimizer.lower() == "adam":
            return Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(self.config.beta1, self.config.beta2)
            )
        elif self.config.optimizer.lower() == "sgd":
            return SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=self.config.momentum
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """Create learning rate scheduler."""
        if self.config.scheduler.lower() == "cosine":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs,
                eta_min=self.config.learning_rate * 0.01
            )
        elif self.config.scheduler.lower() == "step":
            return StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler.lower() == "plateau":
            return ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        elif self.config.scheduler.lower() == "onecycle":
            return OneCycleLR(
                self.optimizer,
                max_lr=self.config.learning_rate,
                epochs=self.config.num_epochs,
                steps_per_epoch=100  # This will be updated later
            )
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def _initialize_monitoring(self):
        """Initialize monitoring and logging."""
        # Initialize wandb
        if self.config.use_wandb and (not self.config.distributed or self.config.rank == 0):
            wandb.init(
                project=self.config.project_name,
                name=self.config.experiment_name or f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                config=self.config.to_dict()
            )
        
        # Initialize tensorboard
        if self.config.use_tensorboard and (not self.config.distributed or self.config.rank == 0):
            log_dir = f"./logs/tensorboard/{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            self.writer = SummaryWriter(log_dir)
        
        # Metrics tracking
        self.train_metrics = defaultdict(list)
        self.val_metrics = defaultdict(list)
    
    def train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses = defaultdict(float)
        num_batches = len(train_loader)
        
        # Progress bar
        if not self.config.distributed or self.config.rank == 0:
            pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        else:
            pbar = train_loader
        
        for batch_idx, batch in enumerate(pbar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Apply data augmentation
            batch = self.augmentation.augment_batch(batch)
            
            # Forward pass with mixed precision
            if self.config.mixed_precision:
                with autocast():
                    predictions = self.model(batch)
                    targets = self._create_targets(batch)
                    losses = self.loss_fn(predictions, targets)
            else:
                predictions = self.model(batch)
                targets = self._create_targets(batch)
                losses = self.loss_fn(predictions, targets)
            
            # Backward pass with gradient accumulation
            total_loss = losses['total_loss'] / self.config.gradient_accumulation_steps
            
            if self.config.mixed_precision:
                self.scaler.scale(total_loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                total_loss.backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Update EMA
            if self.config.use_ema and self.ema:
                self.ema.update(self.model)
            
            # Update metrics
            for key, value in losses.items():
                epoch_losses[key] += value.item()
            
            # Update global step
            self.global_step += 1
            
            # Log to wandb
            if self.config.use_wandb and (not self.config.distributed or self.config.rank == 0):
                if self.global_step % self.config.log_interval == 0:
                    wandb.log({
                        'train/total_loss': losses['total_loss'].item(),
                        'train/action_loss': losses['action_loss'].item(),
                        'train/safety_loss': losses['safety_loss'].item(),
                        'train/human_aware_loss': losses['human_aware_loss'].item(),
                        'train/learning_rate': self.optimizer.param_groups[0]['lr'],
                        'global_step': self.global_step
                    })
            
            # Update progress bar
            if not self.config.distributed or self.config.rank == 0:
                pbar.set_postfix({
                    'loss': losses['total_loss'].item(),
                    'action': losses['action_loss'].item(),
                    'safety': losses['safety_loss'].item()
                })
        
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
            # Progress bar
            if not self.config.distributed or self.config.rank == 0:
                pbar = tqdm(val_loader, desc="Validation")
            else:
                pbar = val_loader
            
            for batch in pbar:
                # Move batch to device
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                if self.config.mixed_precision:
                    with autocast():
                        predictions = self.model(batch)
                        targets = self._create_targets(batch)
                        losses = self.loss_fn(predictions, targets)
                else:
                    predictions = self.model(batch)
                    targets = self._create_targets(batch)
                    losses = self.loss_fn(predictions, targets)
                
                # Update metrics
                for key, value in losses.items():
                    epoch_losses[key] += value.item()
                
                # Update progress bar
                if not self.config.distributed or self.config.rank == 0:
                    pbar.set_postfix({'val_loss': losses['total_loss'].item()})
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return dict(epoch_losses)
    
    def _create_targets(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Create target tensors from batch."""
        targets = {}
        
        # Action targets
        if 'actions' in batch:
            targets['actions'] = batch['actions']
        
        # Safety targets (high safety for clinical tasks)
        targets['safety_score'] = torch.ones(batch['actions'].size(0), 1, device=self.device) * 0.9
        
        # Human-aware targets
        targets['human_aware_score'] = torch.ones(batch['actions'].size(0), 1, device=self.device) * 0.8
        
        return targets
    
    def train(self, train_dataset: ClinicalDataset, val_dataset: ClinicalDataset = None):
        """Main training loop."""
        # Create data loaders
        train_sampler = DistributedSampler(train_dataset) if self.config.distributed else None
        val_sampler = DistributedSampler(val_dataset) if self.config.distributed and val_dataset else None
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            drop_last=self.config.drop_last
        )
        
        if val_dataset:
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.batch_size,
                shuffle=False,
                sampler=val_sampler,
                num_workers=self.config.num_workers,
                pin_memory=self.config.pin_memory
            )
        else:
            val_loader = None
        
        # Add synthetic data if configured
        if self.config.synthetic_data_ratio > 0:
            train_dataset = self._add_synthetic_data(train_dataset)
        
        logger.info(f"Starting training for {self.config.num_epochs} epochs")
        logger.info(f"Training samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            
            # Update sampler for distributed training
            if train_sampler:
                train_sampler.set_epoch(epoch)
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Validate epoch
            val_metrics = {}
            if val_loader:
                val_metrics = self.validate_epoch(val_loader)
                
                # Update scheduler (for plateau scheduler)
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['total_loss'])
                else:
                    self.scheduler.step()
            
            # Store metrics
            for key, value in train_metrics.items():
                self.train_metrics[key].append(value)
            
            for key, value in val_metrics.items():
                self.val_metrics[key].append(value)
            
            # Log epoch metrics
            if not self.config.distributed or self.config.rank == 0:
                logger.info(f"Epoch {epoch}: Train Loss = {train_metrics['total_loss']:.4f}")
                if val_metrics:
                    logger.info(f"Epoch {epoch}: Val Loss = {val_metrics['total_loss']:.4f}")
                
                # Log to wandb
                if self.config.use_wandb:
                    wandb_log = {
                        'epoch': epoch,
                        'train/epoch_loss': train_metrics['total_loss'],
                        'train/epoch_action_loss': train_metrics['action_loss'],
                        'train/epoch_safety_loss': train_metrics['safety_loss'],
                        'train/epoch_human_aware_loss': train_metrics['human_aware_loss'],
                        'train/learning_rate': self.optimizer.param_groups[0]['lr']
                    }
                    
                    for key, value in val_metrics.items():
                        wandb_log[f'val/epoch_{key}'] = value
                    
                    wandb.log(wandb_log)
                
                # Log to tensorboard
                if self.config.use_tensorboard:
                    for key, value in train_metrics.items():
                        self.writer.add_scalar(f'train/{key}', value, epoch)
                    
                    for key, value in val_metrics.items():
                        self.writer.add_scalar(f'val/{key}', value, epoch)
                
                # Save checkpoint
                if epoch % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')
                
                # Check for improvement
                if val_metrics:
                    current_val_loss = val_metrics['total_loss']
                    if current_val_loss < self.best_val_loss:
                        self.best_val_loss = current_val_loss
                        self.epochs_without_improvement = 0
                        self.save_checkpoint('best_model.pth')
                    else:
                        self.epochs_without_improvement += 1
                    
                    # Early stopping
                    if self.epochs_without_improvement >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping triggered after {epoch} epochs")
                        break
        
        logger.info("Training completed!")
        
        # Save final model
        if not self.config.distributed or self.config.rank == 0:
            self.save_checkpoint('final_model.pth')
            
            # Plot training curves
            self.plot_training_curves()
    
    def _add_synthetic_data(self, dataset: ClinicalDataset) -> ClinicalDataset:
        """Add synthetic data to training dataset."""
        logger.info("Adding synthetic data to training dataset...")
        
        # Generate synthetic data
        synthetic_pipeline = ClinicalSyntheticDataPipeline(
            AdaptationConfig(
                batch_size=self.config.batch_size,
                device=str(self.device)
            )
        )
        
        # Generate synthetic samples
        num_synthetic = int(len(dataset) * self.config.synthetic_data_ratio)
        synthetic_data_path = synthetic_pipeline.generate_synthetic_dataset(
            num_synthetic, 
            "./synthetic_data"
        )
        
        # Load synthetic dataset
        synthetic_dataset = ClinicalDataset(synthetic_data_path, AdaptationConfig())
        
        # Combine datasets
        combined_dataset = torch.utils.data.ConcatDataset([dataset, synthetic_dataset])
        
        logger.info(f"Added {num_synthetic} synthetic samples to training dataset")
        
        return combined_dataset
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        if self.config.distributed and self.config.rank != 0:
            return
        
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
        
        # Save EMA weights if used
        if self.config.use_ema and self.ema:
            checkpoint['ema_state_dict'] = self.ema.shadow
        
        checkpoint_path = Path(f'checkpoints/{filename}')
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Keep only best K checkpoints
        self._manage_checkpoints()
    
    def _manage_checkpoints(self):
        """Manage checkpoint storage, keeping only best K."""
        if not self.config.distributed or self.config.rank == 0:
            checkpoint_dir = Path('checkpoints')
            checkpoints = list(checkpoint_dir.glob('checkpoint_epoch_*.pth'))
            
            # Sort by validation loss
            checkpoint_scores = []
            for checkpoint_path in checkpoints:
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    val_loss = checkpoint.get('best_val_loss', float('inf'))
                    checkpoint_scores.append((checkpoint_path, val_loss))
                except:
                    continue
            
            # Sort by loss (lower is better)
            checkpoint_scores.sort(key=lambda x: x[1])
            
            # Remove excess checkpoints
            for checkpoint_path, _ in checkpoint_scores[self.config.keep_best_k:]:
                try:
                    checkpoint_path.unlink()
                    logger.info(f"Removed old checkpoint: {checkpoint_path}")
                except:
                    pass
    
    def plot_training_curves(self):
        """Plot training and validation curves."""
        if self.config.distributed and self.config.rank != 0:
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Total loss
        axes[0, 0].plot(self.train_metrics['total_loss'], label='Train')
        if self.val_metrics['total_loss']:
            axes[0, 0].plot(self.val_metrics['total_loss'], label='Validation')
        axes[0, 0].set_title('Total Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Action loss
        axes[0, 1].plot(self.train_metrics['action_loss'], label='Train')
        if self.val_metrics.get('action_loss'):
            axes[0, 1].plot(self.val_metrics['action_loss'], label='Validation')
        axes[0, 1].set_title('Action Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Safety loss
        axes[1, 0].plot(self.train_metrics['safety_loss'], label='Train')
        if self.val_metrics.get('safety_loss'):
            axes[1, 0].plot(self.val_metrics['safety_loss'], label='Validation')
        axes[1, 0].set_title('Safety Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Human-aware loss
        axes[1, 1].plot(self.train_metrics['human_aware_loss'], label='Train')
        if self.val_metrics.get('human_aware_loss'):
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

def setup_distributed_training(rank: int, world_size: int, config: TrainingConfig):
    """Setup distributed training."""
    config.distributed = True
    config.world_size = world_size
    config.rank = rank
    config.local_rank = rank
    
    # Initialize distributed training
    dist.init_process_group(
        backend=config.backend,
        init_method='env://',
        world_size=world_size,
        rank=rank
    )
    
    # Set device
    torch.cuda.set_device(rank)
    config.device = f'cuda:{rank}'

def main():
    """Main function for training."""
    parser = argparse.ArgumentParser(description='Clinical Model Trainer')
    parser.add_argument('--config', type=str, required=True, help='Configuration file path')
    parser.add_argument('--data', type=str, required=True, help='Training data path')
    parser.add_argument('--resume', type=str, help='Resume from checkpoint')
    parser.add_argument('--distributed', action='store_true', help='Use distributed training')
    parser.add_argument('--world-size', type=int, default=1, help='World size for distributed training')
    
    args = parser.parse_args()
    
    # Load configuration
    with open(args.config, 'r') as f:
        config_dict = json.load(f)
    
    config = TrainingConfig(**config_dict)
    config.data_path = args.data
    
    if args.distributed and args.world_size > 1:
        # Distributed training
        mp.spawn(
            train_worker,
            args=(args.world_size, config, args.resume),
            nprocs=args.world_size,
            join=True
        )
    else:
        # Single GPU training
        train_worker(0, 1, config, args.resume)

def train_worker(rank: int, world_size: int, config: TrainingConfig, resume_path: str = None):
    """Training worker for distributed training."""
    if world_size > 1:
        setup_distributed_training(rank, world_size, config)
    
    # Create trainer
    trainer = ClinicalModelTrainer(config)
    
    # Load dataset
    dataset = ClinicalDataset(config.data_path, AdaptationConfig())
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Resume from checkpoint if specified
    if resume_path and Path(resume_path).exists():
        trainer.load_checkpoint(resume_path)
    
    # Train model
    trainer.train(train_dataset, val_dataset)
    
    # Cleanup distributed training
    if world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
