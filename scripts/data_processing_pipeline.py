#!/usr/bin/env python3
"""
ClinAdapt Data Processing Pipeline
Comprehensive data processing and analysis for clinical robot demonstrations.

This script implements the complete data processing pipeline for clinical medication
delivery robot demonstrations, including:
- Multi-modal data ingestion (images, robot states, actions, metadata)
- Quality assessment and filtering
- Clinical data augmentation
- Statistical analysis and visualization
- Dataset preparation for few-shot learning

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

# Memory optimization imports
import psutil
import gc
from memory_profiler import profile
import h5py
import warnings

# Data processing imports
import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import distance
from scipy.signal import savgol_filter
import cv2
from PIL import Image, ImageEnhance, ImageFilter

# Machine learning imports
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import StratifiedShuffleSplit
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Medical imaging imports (for medication object analysis)
import pydicom
from skimage import measure, filters, morphology, segmentation
from skimage.feature import hog, local_binary_pattern
from skimage.exposure import equalize_adapthist

# Clinical robotics specific imports
from scipy.spatial.transform import Rotation
from scipy.interpolate import interp1d
import trimesh

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_processing.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ClinicalDataProcessor:
    """
    Comprehensive data processor for clinical robot demonstrations.
    
    This class handles the complete pipeline from raw demonstration data
    to processed datasets ready for few-shot learning, including quality
    assessment, augmentation, and statistical analysis.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the clinical data processor.
        
        Args:
            config: Configuration dictionary containing processing parameters
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './processed_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Processing parameters
        self.image_size = tuple(config.get('image_size', (224, 224)))
        self.sequence_length = config.get('sequence_length', 32)
        self.quality_threshold = config.get('quality_threshold', 0.85)
        self.augmentation_factor = config.get('augmentation_factor', 2.0)
        
        # Clinical-specific parameters
        self.medication_types = config.get('medication_types', [
            'vial', 'blister_pack', 'syringe', 'bottle', 'pouch'
        ])
        self.grasp_types = config.get('grasp_types', [
            'precision', 'power', 'pinch', 'lateral', 'cylindrical'
        ])
        
        # Data storage
        self.raw_demonstrations = []
        self.processed_demonstrations = []
        self.quality_scores = []
        self.statistics = {}
        
        # Initialize processors
        self._initialize_processors()
        
        logger.info(f"Clinical Data Processor initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Target image size: {self.image_size}")
        logger.info(f"Quality threshold: {self.quality_threshold}")
    
    def _initialize_processors(self):
        """Initialize data processing components."""
        # Image processor for clinical environments
        self.image_processor = ClinicalImageProcessor(self.image_size)
        
        # Robot state processor
        self.robot_state_processor = RobotStateProcessor()
        
        # Action sequence processor
        self.action_processor = ActionSequenceProcessor()
        
        # Quality assessor
        self.quality_assessor = DemonstrationQualityAssessor(self.config)
        
        # Data augmenter
        self.augmenter = ClinicalDataAugmenter(self.config)
        
        # Statistical analyzer
        self.analyzer = ClinicalDataAnalyzer(self.config)
        
        logger.info("Data processing components initialized")
    
    def load_raw_data(self, data_path: str) -> bool:
        """
        Load raw demonstration data from various formats.
        
        Args:
            data_path: Path to raw demonstration data
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Loading raw data from {data_path}")
        
        data_path = Path(data_path)
        
        try:
            if data_path.is_file() and data_path.suffix == '.h5':
                self.raw_demonstrations = self._load_hdf5_data(data_path)
            elif data_path.is_file() and data_path.suffix == '.pkl':
                self.raw_demonstrations = self._load_pickle_data(data_path)
            elif data_path.is_dir():
                self.raw_demonstrations = self._load_directory_data(data_path)
            else:
                raise ValueError(f"Unsupported data format: {data_path}")
            
            logger.info(f"Loaded {len(self.raw_demonstrations)} raw demonstrations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load raw data: {str(e)}")
            return False
    
    def _load_hdf5_data(self, file_path: Path) -> List[Dict]:
        """Load demonstrations from HDF5 file."""
        demonstrations = []
        
        with h5py.File(file_path, 'r') as f:
            demo_groups = list(f.keys())
            
            for demo_name in demo_groups:
                demo_group = f[demo_name]
                
                # Load basic demonstration data
                demo = {
                    'id': demo_name,
                    'images': demo_group['images'][:],
                    'robot_states': demo_group['robot_states'][:],
                    'actions': demo_group['actions'][:],
                    'timestamps': demo_group['timestamps'][:] if 'timestamps' in demo_group else None,
                    'metadata': dict(demo_group.attrs)
                }
                
                # Load additional data if available
                if 'depth_images' in demo_group:
                    demo['depth_images'] = demo_group['depth_images'][:]
                if 'force_torque' in demo_group:
                    demo['force_torque'] = demo_group['force_torque'][:]
                if 'medication_info' in demo_group:
                    demo['medication_info'] = json.loads(demo_group['medication_info'][()])
                
                demonstrations.append(demo)
        
        return demonstrations
    
    def _load_pickle_data(self, file_path: Path) -> List[Dict]:
        """Load demonstrations from pickle file."""
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    def _load_directory_data(self, dir_path: Path) -> List[Dict]:
        """Load demonstrations from directory structure."""
        demonstrations = []
        
        # Look for JSON files with associated images
        for demo_file in dir_path.glob('*.json'):
            try:
                with open(demo_file, 'r') as f:
                    demo = json.load(f)
                    demo['id'] = demo_file.stem
                
                # Load associated image sequence
                image_dir = demo_file.with_suffix('')
                if image_dir.exists():
                    images = []
                    for img_file in sorted(image_dir.glob('*.jpg')):
                        img = cv2.imread(str(img_file))
                        if img is not None:
                            images.append(img)
                    demo['images'] = images
                
                # Load additional data files
                depth_file = demo_file.with_suffix('.npy')
                if depth_file.exists():
                    demo['depth_images'] = np.load(depth_file)
                
                actions_file = demo_file.with_suffix('.csv')
                if actions_file.exists():
                    demo['actions'] = pd.read_csv(actions_file).values
                
                demonstrations.append(demo)
                
            except Exception as e:
                logger.warning(f"Failed to load demonstration {demo_file}: {str(e)}")
                continue
        
        return demonstrations
    
    def assess_quality(self) -> Dict[str, Any]:
        """
        Assess quality of all demonstrations.
        
        Returns:
            Quality assessment results
        """
        logger.info("Assessing demonstration quality")
        
        quality_results = {
            'demonstration_scores': [],
            'overall_statistics': {},
            'quality_distribution': {},
            'recommendations': []
        }
        
        for i, demo in enumerate(self.raw_demonstrations):
            # Assess individual demonstration quality
            quality_score = self.quality_assessor.assess_demonstration(demo)
            quality_results['demonstration_scores'].append({
                'id': demo.get('id', f'demo_{i}'),
                'score': quality_score['overall_score'],
                'components': quality_score['components'],
                'passes_threshold': quality_score['overall_score'] >= self.quality_threshold
            })
            
            self.quality_scores.append(quality_score['overall_score'])
        
        # Compute overall statistics
        scores = [q['score'] for q in quality_results['demonstration_scores']]
        quality_results['overall_statistics'] = {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'min_score': np.min(scores),
            'max_score': np.max(scores),
            'pass_rate': np.mean([s >= self.quality_threshold for s in scores])
        }
        
        # Quality distribution
        quality_results['quality_distribution'] = {
            'excellent': np.sum(np.array(scores) >= 0.9),
            'good': np.sum((np.array(scores) >= 0.8) & (np.array(scores) < 0.9)),
            'acceptable': np.sum((np.array(scores) >= 0.7) & (np.array(scores) < 0.8)),
            'poor': np.sum(np.array(scores) < 0.7)
        }
        
        # Generate recommendations
        quality_results['recommendations'] = self._generate_quality_recommendations(
            quality_results['overall_statistics']
        )
        
        self.statistics['quality_assessment'] = quality_results
        
        logger.info(f"Quality assessment completed")
        logger.info(f"Mean quality score: {quality_results['overall_statistics']['mean_score']:.3f}")
        logger.info(f"Pass rate: {quality_results['overall_statistics']['pass_rate']:.3f}")
        
        return quality_results
    
    def _generate_quality_recommendations(self, stats: Dict) -> List[str]:
        """Generate recommendations based on quality statistics."""
        recommendations = []
        
        if stats['mean_score'] < 0.8:
            recommendations.append(
                "Overall demonstration quality is below target. Consider collecting more demonstrations "
                "with better lighting conditions and clearer medication object visibility."
            )
        
        if stats['pass_rate'] < 0.7:
            recommendations.append(
                "Low pass rate detected. Review data collection protocol and ensure "
                "consistent grasp quality and trajectory smoothness."
            )
        
        if stats['std_score'] > 0.2:
            recommendations.append(
                "High quality variance detected. Standardize data collection procedures "
                "and provide better training to data collectors."
            )
        
        return recommendations
    
    def process_demonstrations(self) -> bool:
        """
        Process all demonstrations with quality filtering and augmentation.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Processing demonstrations")
        
        try:
            # Filter high-quality demonstrations
            high_quality_indices = [
                i for i, score in enumerate(self.quality_scores)
                if score >= self.quality_threshold
            ]
            
            filtered_demos = [
                self.raw_demonstrations[i] for i in high_quality_indices
            ]
            
            logger.info(f"Filtered to {len(filtered_demos)} high-quality demonstrations")
            
            # Process each demonstration
            for demo in filtered_demos:
                processed_demo = self._process_single_demonstration(demo)
                self.processed_demonstrations.append(processed_demo)
            
            # Apply data augmentation
            augmented_demos = self.augmenter.augment_demonstrations(
                self.processed_demonstrations
            )
            
            # Combine original and augmented
            self.processed_demonstrations.extend(augmented_demos)
            
            logger.info(f"Processing completed: {len(self.processed_demonstrations)} total demonstrations")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process demonstrations: {str(e)}")
            return False
    
    def _process_single_demonstration(self, demo: Dict) -> Dict:
        """Process a single demonstration."""
        processed = demo.copy()
        
        # Process images
        if 'images' in demo:
            processed['processed_images'] = self.image_processor.process_image_sequence(
                demo['images']
            )
            processed['image_features'] = self.image_processor.extract_features(
                demo['images']
            )
        
        # Process robot states
        if 'robot_states' in demo:
            processed['processed_robot_states'] = self.robot_state_processor.process_states(
                demo['robot_states']
            )
            processed['state_features'] = self.robot_state_processor.extract_features(
                demo['robot_states']
            )
        
        # Process actions
        if 'actions' in demo:
            processed['processed_actions'] = self.action_processor.process_actions(
                demo['actions']
            )
            processed['action_features'] = self.action_processor.extract_features(
                demo['actions']
            )
        
        # Extract temporal features
        if all(k in demo for k in ['images', 'robot_states', 'actions']):
            processed['temporal_features'] = self._extract_temporal_features(demo)
        
        # Add processing metadata
        processed['processing_metadata'] = {
            'processed_at': datetime.now().isoformat(),
            'image_size': self.image_size,
            'quality_score': self.quality_scores[self.raw_demonstrations.index(demo)]
        }
        
        return processed
    
    def _extract_temporal_features(self, demo: Dict) -> Dict:
        """Extract temporal features from demonstration."""
        features = {}
        
        # Trajectory smoothness
        if 'actions' in demo:
            actions = np.array(demo['actions'])
            if len(actions) > 2:
                velocities = np.diff(actions, axis=0)
                accelerations = np.diff(velocities, axis=0)
                features['trajectory_smoothness'] = 1.0 / (1.0 + np.var(accelerations))
                features['max_velocity'] = np.max(np.linalg.norm(velocities, axis=1))
                features['max_acceleration'] = np.max(np.linalg.norm(accelerations, axis=1))
        
        # State consistency
        if 'robot_states' in demo:
            states = np.array(demo['robot_states'])
            if len(states) > 1:
                state_changes = np.diff(states, axis=0)
                features['state_consistency'] = 1.0 / (1.0 + np.var(state_changes))
        
        # Temporal correlation
        if 'images' in demo and 'actions' in demo:
            # Simple correlation between visual changes and actions
            image_changes = []
            for i in range(len(demo['images']) - 1):
                img1 = cv2.cvtColor(demo['images'][i], cv2.COLOR_BGR2GRAY)
                img2 = cv2.cvtColor(demo['images'][i+1], cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(img1, img2)
                image_changes.append(np.mean(diff))
            
            action_magnitudes = np.linalg.norm(np.diff(demo['actions'], axis=0), axis=1)
            
            if len(image_changes) > 0 and len(action_magnitudes) > 0:
                correlation = np.corrcoef(image_changes[:len(action_magnitudes)], action_magnitudes)[0, 1]
                features['visual_action_correlation'] = correlation if not np.isnan(correlation) else 0.0
        
        return features
    
    def analyze_data(self) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis of processed data.
        
        Returns:
            Analysis results
        """
        logger.info("Performing comprehensive data analysis")
        
        analysis_results = self.analyzer.analyze_demonstrations(
            self.processed_demonstrations
        )
        
        self.statistics['data_analysis'] = analysis_results
        
        logger.info("Data analysis completed")
        return analysis_results
    
    def save_processed_data(self, format: str = 'hdf5') -> bool:
        """
        Save processed demonstrations to file.
        
        Args:
            format: Output format ('hdf5', 'pickle', 'directory')
            
        Returns:
            True if successful, False otherwise
        """
        logger.info(f"Saving processed data in {format} format")
        
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format == 'hdf5':
                output_path = self.output_dir / f'processed_demonstrations_{timestamp}.h5'
                self._save_hdf5(output_path)
            elif format == 'pickle':
                output_path = self.output_dir / f'processed_demonstrations_{timestamp}.pkl'
                self._save_pickle(output_path)
            elif format == 'directory':
                output_path = self.output_dir / f'processed_demonstrations_{timestamp}'
                self._save_directory(output_path)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            # Save statistics
            stats_path = self.output_dir / f'processing_statistics_{timestamp}.json'
            with open(stats_path, 'w') as f:
                json.dump(self.statistics, f, indent=2, default=str)
            
            logger.info(f"Processed data saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save processed data: {str(e)}")
            return False
    
    def _save_hdf5(self, output_path: Path):
        """Save processed demonstrations to HDF5 format."""
        with h5py.File(output_path, 'w') as f:
            for i, demo in enumerate(self.processed_demonstrations):
                demo_group = f.create_group(f'demo_{i:04d}')
                
                # Save processed data
                if 'processed_images' in demo:
                    demo_group.create_dataset('images', data=demo['processed_images'])
                if 'processed_robot_states' in demo:
                    demo_group.create_dataset('robot_states', data=demo['processed_robot_states'])
                if 'processed_actions' in demo:
                    demo_group.create_dataset('actions', data=demo['processed_actions'])
                
                # Save features
                if 'image_features' in demo:
                    demo_group.create_dataset('image_features', data=demo['image_features'])
                if 'state_features' in demo:
                    demo_group.create_dataset('state_features', data=demo['state_features'])
                if 'action_features' in demo:
                    demo_group.create_dataset('action_features', data=demo['action_features'])
                if 'temporal_features' in demo:
                    demo_group.create_dataset('temporal_features', data=demo['temporal_features'])
                
                # Save metadata
                for key, value in demo.get('metadata', {}).items():
                    demo_group.attrs[key] = value
                for key, value in demo.get('processing_metadata', {}).items():
                    demo_group.attrs[f'processed_{key}'] = value
    
    def _save_pickle(self, output_path: Path):
        """Save processed demonstrations to pickle format."""
        with open(output_path, 'wb') as f:
            pickle.dump(self.processed_demonstrations, f)
    
    def _save_directory(self, output_path: Path):
        """Save processed demonstrations to directory structure."""
        output_path.mkdir(parents=True, exist_ok=True)
        
        for i, demo in enumerate(self.processed_demonstrations):
            demo_dir = output_path / f'demo_{i:04d}'
            demo_dir.mkdir(exist_ok=True)
            
            # Save images
            if 'processed_images' in demo:
                img_dir = demo_dir / 'images'
                img_dir.mkdir(exist_ok=True)
                for j, img in enumerate(demo['processed_images']):
                    cv2.imwrite(str(img_dir / f'img_{j:04d}.jpg'), img)
            
            # Save numpy arrays
            for key in ['processed_robot_states', 'processed_actions', 'image_features', 
                        'state_features', 'action_features', 'temporal_features']:
                if key in demo:
                    np.save(demo_dir / f'{key}.npy', demo[key])
            
            # Save metadata
            metadata = {**demo.get('metadata', {}), **demo.get('processing_metadata', {})}
            with open(demo_dir / 'metadata.json', 'w') as f:
                json.dump(metadata, f, indent=2)
    
    def generate_report(self) -> str:
        """
        Generate comprehensive data processing report.
        
        Returns:
            Path to generated report
        """
        logger.info("Generating data processing report")
        
        report_path = self.output_dir / f'processing_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.html'
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Report generated: {report_path}")
        return str(report_path)
    
    def _generate_html_report(self) -> str:
        """Generate HTML content for the report."""
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ClinAdapt Data Processing Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e9f5ff; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>ClinAdapt Data Processing Report</h1>
                <p>Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
                <p>Processed {len(self.processed_demonstrations)} demonstrations from {len(self.raw_demonstrations)} raw demonstrations</p>
            </div>
            
            <div class="section">
                <h2>Quality Assessment Results</h2>
                <div class="metric">Mean Quality Score: {np.mean(self.quality_scores):.3f}</div>
                <div class="metric">Pass Rate: {np.mean([s >= self.quality_threshold for s in self.quality_scores]):.3f}</div>
                <div class="metric">Std Deviation: {np.std(self.quality_scores):.3f}</div>
            </div>
            
            <div class="section">
                <h2>Processing Statistics</h2>
                <table>
                    <tr><th>Metric</th><th>Value</th></tr>
                    <tr><td>Raw Demonstrations</td><td>{len(self.raw_demonstrations)}</td></tr>
                    <tr><td>High-Quality Demonstrations</td><td>{len([s for s in self.quality_scores if s >= self.quality_threshold])}</td></tr>
                    <tr><td>Final Processed Demonstrations</td><td>{len(self.processed_demonstrations)}</td></tr>
                    <tr><td>Augmentation Factor</td><td>{self.augmentation_factor}</td></tr>
                </table>
            </div>
            
            <div class="section">
                <h2>Configuration</h2>
                <table>
                    <tr><th>Parameter</th><th>Value</th></tr>
                    <tr><td>Image Size</td><td>{self.image_size}</td></tr>
                    <tr><td>Quality Threshold</td><td>{self.quality_threshold}</td></tr>
                    <tr><td>Sequence Length</td><td>{self.sequence_length}</td></tr>
                </table>
            </div>
        </body>
        </html>
        """
        return html


class ClinicalImageProcessor:
    """Specialized image processor for clinical environments."""
    
    def __init__(self, target_size: Tuple[int, int]):
        self.target_size = target_size
        self.clinical_transforms = self._create_clinical_transforms()
    
    def _create_clinical_transforms(self):
        """Create transforms optimized for clinical environments."""
        return transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def process_image_sequence(self, images: List[np.ndarray]) -> np.ndarray:
        """Process a sequence of images for clinical environments."""
        processed = []
        
        for img in images:
            # Clinical-specific preprocessing
            img = self._enhance_clinical_image(img)
            img = self._normalize_lighting(img)
            img = self._remove_artifacts(img)
            
            # Resize and convert to RGB
            img = cv2.resize(img, self.target_size)
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            elif len(img.shape) == 2:
                img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            
            processed.append(img)
        
        return np.array(processed)
    
    def _enhance_clinical_image(self, img: np.ndarray) -> np.ndarray:
        """Enhance image for clinical object visibility."""
        # Convert to PIL for enhancement
        if len(img.shape) == 3:
            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(img)
        
        # Enhance contrast and sharpness for medication visibility
        enhancer = ImageEnhance.Contrast(pil_img)
        pil_img = enhancer.enhance(1.2)
        
        enhancer = ImageEnhance.Sharpness(pil_img)
        pil_img = enhancer.enhance(1.1)
        
        # Convert back to numpy
        img = np.array(pil_img)
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        return img
    
    def _normalize_lighting(self, img: np.ndarray) -> np.ndarray:
        """Normalize lighting conditions for clinical environments."""
        # Apply adaptive histogram equalization
        if len(img.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            
            # Apply CLAHE to L channel
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            
            # Merge channels and convert back
            lab = cv2.merge((l, a, b))
            img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            img = clahe.apply(img)
        
        return img
    
    def _remove_artifacts(self, img: np.ndarray) -> np.ndarray:
        """Remove common artifacts in clinical images."""
        # Apply median filter to remove salt-and-pepper noise
        img = cv2.medianBlur(img, 3)
        
        # Remove small artifacts using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        
        return img
    
    def extract_features(self, images: List[np.ndarray]) -> Dict[str, np.ndarray]:
        """Extract clinical-relevant features from images."""
        features = {}
        
        # Process first image for demonstration-level features
        if len(images) > 0:
            img = images[0]
            
            # Edge features for medication object detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
            edges = cv2.Canny(gray, 50, 150)
            features['edge_density'] = np.mean(edges) / 255.0
            
            # Texture features for surface classification
            lbp = local_binary_pattern(gray, P=8, R=1)
            features['texture_histogram'] = np.histogram(lbp.ravel(), bins=256)[0]
            
            # HOG features for shape recognition
            hog_features = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2))
            features['hog_features'] = hog_features
            
            # Color histogram for medication type classification
            if len(img.shape) == 3:
                hist_b = cv2.calcHist([img], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([img], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([img], [2], None, [256], [0, 256])
                features['color_histogram'] = np.concatenate([hist_b, hist_g, hist_r]).flatten()
        
        return features


class RobotStateProcessor:
    """Processor for robot joint states and sensor data."""
    
    def process_states(self, states: np.ndarray) -> np.ndarray:
        """Process robot states for clinical applications."""
        # Smooth joint trajectories
        if len(states) > 3:
            smoothed = np.zeros_like(states)
            for i in range(states.shape[1]):
                smoothed[:, i] = savgol_filter(states[:, i], window_length=5, polyorder=2)
            return smoothed
        return states
    
    def extract_features(self, states: np.ndarray) -> Dict[str, float]:
        """Extract features from robot states."""
        features = {}
        
        if len(states) > 1:
            # Joint velocity statistics
            velocities = np.diff(states, axis=0)
            features['mean_velocity'] = np.mean(np.linalg.norm(velocities, axis=1))
            features['max_velocity'] = np.max(np.linalg.norm(velocities, axis=1))
            features['velocity_variance'] = np.var(np.linalg.norm(velocities, axis=1))
            
            # Joint acceleration statistics
            if len(velocities) > 1:
                accelerations = np.diff(velocities, axis=0)
                features['mean_acceleration'] = np.mean(np.linalg.norm(accelerations, axis=1))
                features['max_acceleration'] = np.max(np.linalg.norm(accelerations, axis=1))
            
            # Range of motion
            features['joint_ranges'] = np.max(states, axis=0) - np.min(states, axis=0)
            features['total_range'] = np.sum(features['joint_ranges'])
            
            # Smoothness metrics
            features['trajectory_smoothness'] = 1.0 / (1.0 + features.get('velocity_variance', 0))
        
        return features


class ActionSequenceProcessor:
    """Processor for robot action sequences."""
    
    def process_actions(self, actions: np.ndarray) -> np.ndarray:
        """Process action sequences for clinical applications."""
        # Remove outliers and smooth actions
        if len(actions) > 3:
            processed = np.zeros_like(actions)
            for i in range(actions.shape[1]):
                # Remove outliers using median filter
                filtered = actions[:, i]
                for j in range(1, len(actions) - 1):
                    window = filtered[max(0, j-2):min(len(actions), j+3)]
                    filtered[j] = np.median(window)
                
                # Apply smoothing
                processed[:, i] = savgol_filter(filtered, window_length=5, polyorder=2)
            return processed
        return actions
    
    def extract_features(self, actions: np.ndarray) -> Dict[str, float]:
        """Extract features from action sequences."""
        features = {}
        
        if len(actions) > 1:
            # Action magnitude statistics
            magnitudes = np.linalg.norm(actions, axis=1)
            features['mean_magnitude'] = np.mean(magnitudes)
            features['max_magnitude'] = np.max(magnitudes)
            features['magnitude_variance'] = np.var(magnitudes)
            
            # Action smoothness
            if len(actions) > 2:
                diffs = np.diff(actions, axis=0)
                features['action_smoothness'] = 1.0 / (1.0 + np.var(diffs))
            
            # End-effector path analysis
            positions = actions[:, :3] if actions.shape[1] >= 3 else actions
            if len(positions) > 1:
                path_length = np.sum(np.linalg.norm(np.diff(positions, axis=0), axis=1))
                direct_distance = np.linalg.norm(positions[-1] - positions[0])
                features['path_efficiency'] = direct_distance / path_length if path_length > 0 else 1.0
        
        return features


class DemonstrationQualityAssessor:
    """Assess demonstration quality for clinical applications."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.weights = config.get('quality_weights', {
            'trajectory_smoothness': 0.3,
            'grasp_stability': 0.25,
            'visual_clarity': 0.2,
            'task_completion': 0.15,
            'safety_compliance': 0.1
        })
    
    def assess_demonstration(self, demo: Dict) -> Dict[str, float]:
        """Assess quality of a single demonstration."""
        scores = {}
        
        # Trajectory smoothness
        scores['trajectory_smoothness'] = self._assess_trajectory_smoothness(demo)
        
        # Grasp stability
        scores['grasp_stability'] = self._assess_grasp_stability(demo)
        
        # Visual clarity
        scores['visual_clarity'] = self._assess_visual_clarity(demo)
        
        # Task completion
        scores['task_completion'] = self._assess_task_completion(demo)
        
        # Safety compliance
        scores['safety_compliance'] = self._assess_safety_compliance(demo)
        
        # Compute weighted overall score
        overall_score = sum(
            scores[component] * self.weights.get(component, 0.2)
            for component in scores
        )
        
        return {
            'overall_score': overall_score,
            'components': scores
        }
    
    def _assess_trajectory_smoothness(self, demo: Dict) -> float:
        """Assess trajectory smoothness."""
        if 'actions' not in demo:
            return 0.5
        
        actions = np.array(demo['actions'])
        if len(actions) < 3:
            return 0.5
        
        # Compute smoothness based on acceleration variance
        velocities = np.diff(actions, axis=0)
        accelerations = np.diff(velocities, axis=0)
        
        # Lower acceleration variance = smoother trajectory
        smoothness = 1.0 / (1.0 + np.var(accelerations))
        return np.clip(smoothness, 0.0, 1.0)
    
    def _assess_grasp_stability(self, demo: Dict) -> float:
        """Assess grasp stability."""
        # This would analyze force/torque data if available
        # For now, use heuristic based on action consistency
        if 'actions' not in demo:
            return 0.5
        
        actions = np.array(demo['actions'])
        if len(actions) < 2:
            return 0.5
        
        # Check for sudden changes in end-effector position
        if actions.shape[1] >= 3:
            positions = actions[:, :3]
            position_changes = np.linalg.norm(np.diff(positions, axis=0), axis=1)
            stability = 1.0 / (1.0 + np.var(position_changes))
            return np.clip(stability, 0.0, 1.0)
        
        return 0.5
    
    def _assess_visual_clarity(self, demo: Dict) -> float:
        """Assess visual clarity of demonstration."""
        if 'images' not in demo or len(demo['images']) == 0:
            return 0.5
        
        images = demo['images']
        
        # Assess image quality metrics
        clarity_scores = []
        for img in images:
            # Contrast assessment
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            
            contrast = gray.std() / 255.0
            clarity_scores.append(min(contrast * 2, 1.0))  # Normalize and cap
        
        return np.mean(clarity_scores)
    
    def _assess_task_completion(self, demo: Dict) -> float:
        """Assess task completion based on demonstration metadata."""
        # Check if demonstration has completion metadata
        metadata = demo.get('metadata', {})
        
        if 'success' in metadata:
            return 1.0 if metadata['success'] else 0.0
        
        if 'completion_rate' in metadata:
            return metadata['completion_rate']
        
        # Default assessment based on action sequence
        if 'actions' in demo and len(demo['actions']) > 0:
            # Assume completion if action sequence is reasonable length
            return 0.8
        
        return 0.5
    
    def _assess_safety_compliance(self, demo: Dict) -> float:
        """Assess safety compliance of demonstration."""
        # Check for safety violations in metadata
        metadata = demo.get('metadata', {})
        
        if 'safety_violations' in metadata:
            violations = metadata['safety_violations']
            return max(0.0, 1.0 - violations * 0.2)
        
        # Assess based on velocity limits
        if 'actions' in demo:
            actions = np.array(demo['actions'])
            if len(actions) > 1:
                velocities = np.diff(actions, axis=0)
                max_velocity = np.max(np.linalg.norm(velocities, axis=1))
                
                # Assume safe velocity limit
                safe_velocity = 0.5  # m/s
                if max_velocity > safe_velocity * 2:
                    return 0.3
                elif max_velocity > safe_velocity:
                    return 0.7
                else:
                    return 1.0
        
        return 0.8


class ClinicalDataAugmenter:
    """Augment clinical demonstration data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.augmentation_probability = config.get('augmentation_probability', 0.5)
    
    def augment_demonstrations(self, demonstrations: List[Dict]) -> List[Dict]:
        """Augment demonstration data for clinical environments."""
        augmented = []
        
        for demo in demonstrations:
            # Always keep original
            augmented.append(demo)
            
            # Apply augmentations with probability
            if np.random.random() < self.augmentation_probability:
                # Lighting augmentation
                aug_demo = self._augment_lighting(demo)
                if aug_demo:
                    augmented.append(aug_demo)
                
                # Medication type variation
                aug_demo = self._augment_medication_type(demo)
                if aug_demo:
                    augmented.append(aug_demo)
                
                # Human presence simulation
                aug_demo = self._augment_human_presence(demo)
                if aug_demo:
                    augmented.append(aug_demo)
        
        return augmented
    
    def _augment_lighting(self, demo: Dict) -> Optional[Dict]:
        """Augment demonstration with lighting variations."""
        if 'images' not in demo:
            return None
        
        aug_demo = demo.copy()
        augmented_images = []
        
        for img in demo['images']:
            # Apply random lighting changes
            brightness_factor = np.random.uniform(0.7, 1.3)
            contrast_factor = np.random.uniform(0.8, 1.2)
            
            # Convert to PIL for augmentation
            if len(img.shape) == 3:
                pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            else:
                pil_img = Image.fromarray(img)
            
            # Apply transforms
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(brightness_factor)
            
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(contrast_factor)
            
            # Convert back to numpy
            aug_img = np.array(pil_img)
            if len(aug_img.shape) == 3:
                aug_img = cv2.cvtColor(aug_img, cv2.COLOR_RGB2BGR)
            
            augmented_images.append(aug_img)
        
        aug_demo['images'] = augmented_images
        aug_demo['metadata'] = aug_demo.get('metadata', {})
        aug_demo['metadata']['augmentation'] = 'lighting'
        
        return aug_demo
    
    def _augment_medication_type(self, demo: Dict) -> Optional[Dict]:
        """Augment demonstration with medication type variations."""
        medication_types = ['vial', 'blister_pack', 'syringe', 'bottle', 'pouch']
        
        aug_demo = demo.copy()
        current_type = demo.get('metadata', {}).get('medication_type', 'vial')
        
        # Select different medication type
        new_type = np.random.choice([t for t in medication_types if t != current_type])
        
        aug_demo['metadata'] = aug_demo.get('metadata', {})
        aug_demo['metadata']['medication_type'] = new_type
        aug_demo['metadata']['original_medication_type'] = current_type
        aug_demo['metadata']['augmentation'] = 'medication_type'
        
        return aug_demo
    
    def _augment_human_presence(self, demo: Dict) -> Optional[Dict]:
        """Augment demonstration with human presence simulation."""
        aug_demo = demo.copy()
        
        # Simulate human presence
        human_present = True
        human_position = np.random.uniform(-0.5, 0.5, 2)  # x, y position
        human_intent = np.random.choice([
            'approaching_robot', 'departing_robot', 'crossing_path',
            'working_nearby', 'observing'
        ])
        
        # Adjust actions for human awareness
        if 'actions' in demo:
            actions = np.array(demo['actions'])
            adjusted_actions = self._adjust_actions_for_human(actions, human_position, human_intent)
            aug_demo['actions'] = adjusted_actions
        
        aug_demo['metadata'] = aug_demo.get('metadata', {})
        aug_demo['metadata']['human_present'] = human_present
        aug_demo['metadata']['human_position'] = human_position.tolist()
        aug_demo['metadata']['human_intent'] = human_intent
        aug_demo['metadata']['augmentation'] = 'human_presence'
        
        return aug_demo
    
    def _adjust_actions_for_human(self, actions: np.ndarray, human_position: np.ndarray, 
                                human_intent: str) -> np.ndarray:
        """Adjust robot actions based on human presence."""
        adjusted_actions = actions.copy()
        
        # Safety distances based on intent
        safety_distances = {
            'approaching_robot': 0.8,
            'departing_robot': 0.5,
            'crossing_path': 1.0,
            'working_nearby': 0.6,
            'observing': 0.4
        }
        
        min_distance = safety_distances.get(human_intent, 0.5)
        
        # Adjust actions that come too close to human
        for i, action in enumerate(actions):
            if len(action) >= 3:
                ee_pos = action[:3]
                human_pos_3d = np.array([human_position[0], human_position[1], 0.8])
                
                distance = np.linalg.norm(ee_pos - human_pos_3d)
                
                if distance < min_distance:
                    # Adjust position away from human
                    direction = ee_pos - human_pos_3d
                    if np.linalg.norm(direction) > 0:
                        direction = direction / np.linalg.norm(direction)
                        new_pos = human_pos_3d + direction * min_distance
                        adjusted_actions[i, :3] = new_pos
        
        return adjusted_actions


class ClinicalDataAnalyzer:
    """Analyze clinical demonstration data and generate insights."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def analyze_demonstrations(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Perform comprehensive analysis of demonstrations."""
        analysis = {
            'overview': self._analyze_overview(demonstrations),
            'quality_analysis': self._analyze_quality_distribution(demonstrations),
            'temporal_analysis': self._analyze_temporal_patterns(demonstrations),
            'spatial_analysis': self._analyze_spatial_patterns(demonstrations),
            'medication_analysis': self._analyze_medication_types(demonstrations),
            'grasp_analysis': self._analyze_grasp_patterns(demonstrations),
            'correlation_analysis': self._analyze_correlations(demonstrations)
        }
        
        return analysis
    
    def _analyze_overview(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze basic overview statistics."""
        return {
            'total_demonstrations': len(demonstrations),
            'avg_sequence_length': np.mean([len(demo.get('actions', [])) for demo in demonstrations]),
            'total_actions': sum([len(demo.get('actions', [])) for demo in demonstrations]),
            'unique_medication_types': len(set([demo.get('metadata', {}).get('medication_type', 'unknown') for demo in demonstrations])),
            'unique_grasp_types': len(set([demo.get('metadata', {}).get('grasp_type', 'unknown') for demo in demonstrations]))
        }
    
    def _analyze_quality_distribution(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze quality score distribution."""
        quality_scores = []
        
        for demo in demonstrations:
            processing_meta = demo.get('processing_metadata', {})
            quality_scores.append(processing_meta.get('quality_score', 0.5))
        
        return {
            'mean_quality': np.mean(quality_scores),
            'std_quality': np.std(quality_scores),
            'min_quality': np.min(quality_scores),
            'max_quality': np.max(quality_scores),
            'quality_histogram': np.histogram(quality_scores, bins=10, range=(0, 1))[0].tolist(),
            'quality_bins': np.histogram(quality_scores, bins=10, range=(0, 1))[1].tolist()
        }
    
    def _analyze_temporal_patterns(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal patterns in demonstrations."""
        durations = []
        velocities = []
        
        for demo in demonstrations:
            if 'actions' in demo:
                actions = np.array(demo['actions'])
                if len(actions) > 1:
                    durations.append(len(actions))
                    vel = np.linalg.norm(np.diff(actions, axis=0), axis=1)
                    velocities.extend(vel.tolist())
        
        return {
            'avg_duration': np.mean(durations) if durations else 0,
            'duration_std': np.std(durations) if durations else 0,
            'avg_velocity': np.mean(velocities) if velocities else 0,
            'velocity_std': np.std(velocities) if velocities else 0,
            'duration_distribution': np.histogram(durations, bins=20)[0].tolist() if durations else []
        }
    
    def _analyze_spatial_patterns(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze spatial patterns in demonstrations."""
        positions = []
        
        for demo in demonstrations:
            if 'actions' in demo:
                actions = np.array(demo['actions'])
                if actions.shape[1] >= 3:
                    positions.extend(actions[:, :3].tolist())
        
        if positions:
            positions = np.array(positions)
            return {
                'workspace_bounds': {
                    'x_min': np.min(positions[:, 0]),
                    'x_max': np.max(positions[:, 0]),
                    'y_min': np.min(positions[:, 1]),
                    'y_max': np.max(positions[:, 1]),
                    'z_min': np.min(positions[:, 2]),
                    'z_max': np.max(positions[:, 2])
                },
                'workspace_center': np.mean(positions, axis=0).tolist(),
                'workspace_volume': np.prod(np.max(positions, axis=0) - np.min(positions, axis=0))
            }
        
        return {}
    
    def _analyze_medication_types(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze medication type distribution."""
        medication_counts = {}
        quality_by_type = {}
        
        for demo in demonstrations:
            med_type = demo.get('metadata', {}).get('medication_type', 'unknown')
            medication_counts[med_type] = medication_counts.get(med_type, 0) + 1
            
            quality = demo.get('processing_metadata', {}).get('quality_score', 0.5)
            if med_type not in quality_by_type:
                quality_by_type[med_type] = []
            quality_by_type[med_type].append(quality)
        
        # Compute statistics by type
        type_stats = {}
        for med_type, qualities in quality_by_type.items():
            type_stats[med_type] = {
                'count': medication_counts[med_type],
                'mean_quality': np.mean(qualities),
                'std_quality': np.std(qualities)
            }
        
        return {
            'type_distribution': medication_counts,
            'type_statistics': type_stats
        }
    
    def _analyze_grasp_patterns(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze grasp type distribution and success rates."""
        grasp_counts = {}
        success_by_grasp = {}
        
        for demo in demonstrations:
            grasp_type = demo.get('metadata', {}).get('grasp_type', 'unknown')
            grasp_counts[grasp_type] = grasp_counts.get(grasp_type, 0) + 1
            
            success = demo.get('metadata', {}).get('success', False)
            if grasp_type not in success_by_grasp:
                success_by_grasp[grasp_type] = []
            success_by_grasp[grasp_type].append(success)
        
        # Compute statistics by grasp type
        grasp_stats = {}
        for grasp_type, successes in success_by_grasp.items():
            grasp_stats[grasp_type] = {
                'count': grasp_counts[grasp_type],
                'success_rate': np.mean(successes),
                'total_successes': sum(successes)
            }
        
        return {
            'grasp_distribution': grasp_counts,
            'grasp_statistics': grasp_stats
        }
    
    def _analyze_correlations(self, demonstrations: List[Dict]) -> Dict[str, Any]:
        """Analyze correlations between different metrics."""
        # Extract metrics for correlation analysis
        quality_scores = []
        sequence_lengths = []
        medication_success_rates = []
        
        for demo in demonstrations:
            quality_scores.append(demo.get('processing_metadata', {}).get('quality_score', 0.5))
            sequence_lengths.append(len(demo.get('actions', [])))
            
            # Medication type success (binary for common types)
            med_type = demo.get('metadata', {}).get('medication_type', 'unknown')
            medication_success_rates.append(1.0 if med_type in ['vial', 'bottle'] else 0.0)
        
        # Compute correlations
        correlations = {}
        if len(quality_scores) > 1:
            correlations['quality_sequence_length'] = np.corrcoef(quality_scores, sequence_lengths)[0, 1]
            correlations['quality_medication_type'] = np.corrcoef(quality_scores, medication_success_rates)[0, 1]
        
        return correlations


def main():
    """Main function for data processing pipeline."""
    parser = argparse.ArgumentParser(description='Clinical Robot Data Processing Pipeline')
    parser.add_argument('--input', type=str, required=True, help='Input data path')
    parser.add_argument('--output', type=str, default='./processed_data', help='Output directory')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--format', type=str, default='hdf5', choices=['hdf5', 'pickle', 'directory'], help='Output format')
    parser.add_argument('--quality-threshold', type=float, default=0.85, help='Quality threshold')
    parser.add_argument('--augmentation-factor', type=float, default=2.0, help='Data augmentation factor')
    parser.add_argument('--generate-report', action='store_true', help='Generate HTML report')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'output_dir': args.output,
        'quality_threshold': args.quality_threshold,
        'augmentation_factor': args.augmentation_factor,
        'image_size': (224, 224),
        'sequence_length': 32,
        'quality_weights': {
            'trajectory_smoothness': 0.3,
            'grasp_stability': 0.25,
            'visual_clarity': 0.2,
            'task_completion': 0.15,
            'safety_compliance': 0.1
        }
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = json.load(f)
            config.update(user_config)
    
    # Initialize processor
    processor = ClinicalDataProcessor(config)
    
    # Process data
    if not processor.load_raw_data(args.input):
        logger.error("Failed to load raw data")
        return 1
    
    # Assess quality
    quality_results = processor.assess_quality()
    logger.info(f"Quality assessment: {quality_results['overall_statistics']}")
    
    # Process demonstrations
    if not processor.process_demonstrations():
        logger.error("Failed to process demonstrations")
        return 1
    
    # Analyze data
    analysis_results = processor.analyze_data()
    logger.info(f"Data analysis completed: {len(analysis_results)} analysis categories")
    
    # Save processed data
    if not processor.save_processed_data(args.format):
        logger.error("Failed to save processed data")
        return 1
    
    # Generate report
    if args.generate_report:
        report_path = processor.generate_report()
        logger.info(f"Report generated: {report_path}")
    
    logger.info("Data processing pipeline completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
