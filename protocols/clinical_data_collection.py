#!/usr/bin/env python3
"""
Clinical Data Collection Protocol Implementation
Standardized procedures for collecting robot demonstration data in clinical environments.

This module implements the Clinical Robot Adaptation data collection protocol,
providing standardized procedures for collecting high-quality demonstration
data in hospital pharmacy environments. It includes:
- Clinical environment setup and calibration
- Human-robot interaction protocols
- Data quality assurance and validation
- Safety monitoring and compliance
- Multi-modal data synchronization
- Clinical workflow integration

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import logging
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import warnings

# Core imports
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import yaml

# Robotics and hardware
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation
import trimesh

# Medical and clinical
from dicom2nifti import convert_dicom
import pydicom
from pydicom.dataset import Dataset

# Communication and networking
import socket
import requests
from websocket import create_connection
import serial

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_collection.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')


class ClinicalEnvironmentType(Enum):
    """Types of clinical environments for data collection."""
    HOSPITAL_PHARMACY = "hospital_pharmacy"
    CLINICAL_LABORATORY = "clinical_laboratory"
    OUTPATIENT_PHARMACY = "outpatient_pharmacy"
    EMERGENCY_DEPARTMENT = "emergency_department"
    INTENSIVE_CARE_UNIT = "intensive_care_unit"


class DataCollectionStatus(Enum):
    """Status of data collection session."""
    IDLE = "idle"
    CALIBRATING = "calibrating"
    COLLECTING = "collecting"
    PAUSED = "paused"
    VALIDATING = "validating"
    COMPLETED = "completed"
    ERROR = "error"


class SafetyLevel(Enum):
    """Safety levels for clinical data collection."""
    LOW_RISK = "low_risk"
    MEDIUM_RISK = "medium_risk"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class ClinicalEnvironmentConfig:
    """Configuration for clinical environment setup."""
    environment_type: ClinicalEnvironmentType
    layout_description: str
    workspace_dimensions: Dict[str, float]
    medication_types: List[str]
    equipment_list: List[str]
    safety_requirements: Dict[str, Any]
    personnel_requirements: Dict[str, Any]
    sterilization_procedures: List[str]
    workflow_constraints: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'environment_type': self.environment_type.value,
            'layout_description': self.layout_description,
            'workspace_dimensions': self.workspace_dimensions,
            'medication_types': self.medication_types,
            'equipment_list': self.equipment_list,
            'safety_requirements': self.safety_requirements,
            'personnel_requirements': self.personnel_requirements,
            'sterilization_procedures': self.sterilization_procedures,
            'workflow_constraints': self.workflow_constraints
        }


@dataclass
class DataCollectionSession:
    """Data collection session metadata."""
    session_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    environment_config: Optional[ClinicalEnvironmentConfig] = None
    operator_name: str = ""
    patient_id: Optional[str] = None
    medication_type: str = ""
    protocol_version: str = "1.0"
    safety_level: SafetyLevel = SafetyLevel.MEDIUM_RISK
    status: DataCollectionStatus = DataCollectionStatus.IDLE
    collected_demonstrations: int = 0
    quality_score: float = 0.0
    annotations: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format."""
        return {
            'session_id': self.session_id,
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'environment_config': self.environment_config.to_dict() if self.environment_config else None,
            'operator_name': self.operator_name,
            'patient_id': self.patient_id,
            'medication_type': self.medication_type,
            'protocol_version': self.protocol_version,
            'safety_level': self.safety_level.value,
            'status': self.status.value,
            'collected_demonstrations': self.collected_demonstrations,
            'quality_score': self.quality_score,
            'annotations': self.annotations
        }


class ClinicalDataCollectionProtocol:
    """
    Clinical Data Collection Protocol implementation.
    
    This class implements standardized procedures for collecting robot demonstration
    data in clinical environments, ensuring safety, quality, and compliance with
    clinical regulations and best practices.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize clinical data collection protocol.
        
        Args:
            config: Protocol configuration
        """
        self.config = config
        self.output_dir = Path(config.get('output_dir', './clinical_data'))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Protocol parameters
        self.max_demonstrations_per_session = config.get('max_demonstrations', 100)
        self.min_quality_threshold = config.get('min_quality_threshold', 0.7)
        self.safety_checks_enabled = config.get('safety_checks', True)
        self.real_time_validation = config.get('real_time_validation', True)
        
        # Data collection components
        self.environment_manager = ClinicalEnvironmentManager(config)
        self.safety_monitor = ClinicalSafetyMonitor(config)
        self.quality_assessor = DataQualityAssessor(config)
        self.data_synchronizer = MultiModalDataSynchronizer(config)
        self.workflow_integrator = ClinicalWorkflowIntegrator(config)
        
        # Session management
        self.current_session = None
        self.session_history = []
        self.collection_status = DataCollectionStatus.IDLE
        
        # Data storage
        self.collected_data = {
            'demonstrations': [],
            'metadata': [],
            'quality_metrics': [],
            'safety_logs': [],
            'workflow_events': []
        }
        
        # Thread management
        self.collection_thread = None
        self.monitoring_thread = None
        self.stop_collection = threading.Event()
        
        logger.info("Clinical Data Collection Protocol initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Safety checks: {'enabled' if self.safety_checks_enabled else 'disabled'}")
        logger.info(f"Real-time validation: {'enabled' if self.real_time_validation else 'disabled'}")
    
    def start_collection_session(self, 
                                environment_config: ClinicalEnvironmentConfig,
                                operator_name: str,
                                patient_id: Optional[str] = None,
                                medication_type: str = "",
                                safety_level: SafetyLevel = SafetyLevel.MEDIUM_RISK) -> str:
        """
        Start a new data collection session.
        
        Args:
            environment_config: Clinical environment configuration
            operator_name: Name of the operator
            patient_id: Patient identifier (if applicable)
            medication_type: Type of medication being handled
            safety_level: Safety level for the session
            
        Returns:
            Session ID
        """
        # Generate session ID
        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}"
        
        # Create session
        self.current_session = DataCollectionSession(
            session_id=session_id,
            start_time=datetime.now(),
            environment_config=environment_config,
            operator_name=operator_name,
            patient_id=patient_id,
            medication_type=medication_type,
            safety_level=safety_level,
            status=DataCollectionStatus.CALIBRATING
        )
        
        logger.info(f"Starting collection session: {session_id}")
        logger.info(f"Environment: {environment_config.environment_type.value}")
        logger.info(f"Operator: {operator_name}")
        logger.info(f"Safety level: {safety_level.value}")
        
        try:
            # Setup environment
            self._setup_environment(environment_config)
            
            # Calibrate systems
            self._calibrate_systems()
            
            # Start monitoring
            self._start_monitoring()
            
            # Update status
            self.current_session.status = DataCollectionStatus.COLLECTING
            self.collection_status = DataCollectionStatus.COLLECTING
            
            logger.info("Collection session started successfully")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to start collection session: {e}")
            self.current_session.status = DataCollectionStatus.ERROR
            raise
    
    def collect_demonstration(self, 
                            instruction: str,
                            target_medication: str,
                            grasp_type: str,
                            clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Collect a single robot demonstration.
        
        Args:
            instruction: Natural language instruction
            target_medication: Target medication type
            grasp_type: Type of grasp to perform
            clinical_context: Clinical context information
            
        Returns:
            Demonstration data and metadata
        """
        if self.collection_status != DataCollectionStatus.COLLECTING:
            raise RuntimeError("Collection session not active")
        
        if self.current_session.collected_demonstrations >= self.max_demonstrations_per_session:
            raise RuntimeError("Maximum demonstrations per session reached")
        
        logger.info(f"Collecting demonstration: {instruction}")
        logger.info(f"Target medication: {target_medication}")
        logger.info(f"Grasp type: {grasp_type}")
        
        # Create demonstration record
        demonstration_id = f"demo_{self.current_session.session_id}_{self.current_session.collected_demonstrations + 1}"
        
        demonstration_data = {
            'demonstration_id': demonstration_id,
            'session_id': self.current_session.session_id,
            'timestamp': datetime.now().isoformat(),
            'instruction': instruction,
            'target_medication': target_medication,
            'grasp_type': grasp_type,
            'clinical_context': clinical_context,
            'data': {},
            'metadata': {},
            'quality_metrics': {},
            'safety_events': []
        }
        
        try:
            # Start data collection
            start_time = time.time()
            
            # Collect multi-modal data
            multimodal_data = self._collect_multimodal_data(demonstration_id)
            demonstration_data['data'] = multimodal_data
            
            # Collect metadata
            metadata = self._collect_metadata(demonstration_id, clinical_context)
            demonstration_data['metadata'] = metadata
            
            # Real-time quality assessment
            if self.real_time_validation:
                quality_metrics = self.quality_assessor.assess_demonstration(multimodal_data, metadata)
                demonstration_data['quality_metrics'] = quality_metrics
                
                # Check quality threshold
                if quality_metrics['overall_quality'] < self.min_quality_threshold:
                    logger.warning(f"Demonstration quality below threshold: {quality_metrics['overall_quality']:.3f}")
                    demonstration_data['status'] = 'rejected'
                else:
                    demonstration_data['status'] = 'accepted'
            else:
                demonstration_data['status'] = 'pending_validation'
            
            # Safety monitoring
            safety_events = self.safety_monitor.get_session_events(demonstration_id)
            demonstration_data['safety_events'] = safety_events
            
            # Store demonstration
            self._store_demonstration(demonstration_data)
            
            # Update session
            self.current_session.collected_demonstrations += 1
            self.current_session.quality_score = np.mean([
                demo['quality_metrics'].get('overall_quality', 0) 
                for demo in self.collected_data['demonstrations']
                if 'quality_metrics' in demo
            ])
            
            collection_time = time.time() - start_time
            logger.info(f"Demonstration collected in {collection_time:.2f}s")
            logger.info(f"Quality score: {demonstration_data['quality_metrics'].get('overall_quality', 0):.3f}")
            
            return demonstration_data
            
        except Exception as e:
            logger.error(f"Failed to collect demonstration: {e}")
            demonstration_data['status'] = 'error'
            demonstration_data['error'] = str(e)
            return demonstration_data
    
    def pause_collection(self):
        """Pause data collection session."""
        if self.collection_status == DataCollectionStatus.COLLECTING:
            self.collection_status = DataCollectionStatus.PAUSED
            self.current_session.status = DataCollectionStatus.PAUSED
            logger.info("Data collection paused")
        else:
            logger.warning("Cannot pause - collection not active")
    
    def resume_collection(self):
        """Resume data collection session."""
        if self.collection_status == DataCollectionStatus.PAUSED:
            self.collection_status = DataCollectionStatus.COLLECTING
            self.current_session.status = DataCollectionStatus.COLLECTING
            logger.info("Data collection resumed")
        else:
            logger.warning("Cannot resume - collection not paused")
    
    def end_collection_session(self) -> Dict[str, Any]:
        """
        End the current data collection session.
        
        Returns:
            Session summary and statistics
        """
        if not self.current_session:
            raise RuntimeError("No active collection session")
        
        logger.info("Ending collection session")
        
        try:
            # Update session
            self.current_session.end_time = datetime.now()
            self.current_session.status = DataCollectionStatus.COMPLETED
            self.collection_status = DataCollectionStatus.COMPLETED
            
            # Stop monitoring
            self._stop_monitoring()
            
            # Final validation
            self._validate_session_data()
            
            # Generate session report
            session_summary = self._generate_session_summary()
            
            # Save session data
            self._save_session_data()
            
            # Add to history
            self.session_history.append(self.current_session)
            self.current_session = None
            
            logger.info("Collection session ended successfully")
            return session_summary
            
        except Exception as e:
            logger.error(f"Error ending collection session: {e}")
            self.current_session.status = DataCollectionStatus.ERROR
            raise
    
    def _setup_environment(self, environment_config: ClinicalEnvironmentConfig):
        """Setup clinical environment for data collection."""
        logger.info("Setting up clinical environment")
        
        # Initialize environment manager
        self.environment_manager.setup_environment(environment_config)
        
        # Verify safety requirements
        if self.safety_checks_enabled:
            safety_check = self.safety_monitor.verify_environment_setup(environment_config)
            if not safety_check['passed']:
                raise RuntimeError(f"Safety check failed: {safety_check['issues']}")
        
        logger.info("Environment setup completed")
    
    def _calibrate_systems(self):
        """Calibrate all systems for data collection."""
        logger.info("Calibrating systems")
        
        # Calibrate cameras
        self.environment_manager.calibrate_cameras()
        
        # Calibrate robot
        self.environment_manager.calibrate_robot()
        
        # Calibrate sensors
        self.environment_manager.calibrate_sensors()
        
        # Synchronize clocks
        self.data_synchronizer.synchronize_clocks()
        
        logger.info("System calibration completed")
    
    def _start_monitoring(self):
        """Start safety and quality monitoring."""
        logger.info("Starting monitoring systems")
        
        # Start safety monitoring
        self.safety_monitor.start_monitoring()
        
        # Start quality monitoring
        if self.real_time_validation:
            self.quality_assessor.start_monitoring()
        
        # Start data synchronization
        self.data_synchronizer.start_synchronization()
        
        logger.info("Monitoring systems started")
    
    def _stop_monitoring(self):
        """Stop all monitoring systems."""
        logger.info("Stopping monitoring systems")
        
        # Stop safety monitoring
        self.safety_monitor.stop_monitoring()
        
        # Stop quality monitoring
        if self.real_time_validation:
            self.quality_assessor.stop_monitoring()
        
        # Stop data synchronization
        self.data_synchronizer.stop_synchronization()
        
        logger.info("Monitoring systems stopped")
    
    def _collect_multimodal_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect multi-modal data for demonstration."""
        logger.info(f"Collecting multi-modal data for {demonstration_id}")
        
        multimodal_data = {}
        
        # Vision data
        vision_data = self.environment_manager.collect_vision_data(demonstration_id)
        multimodal_data['vision'] = vision_data
        
        # Robot state data
        robot_data = self.environment_manager.collect_robot_data(demonstration_id)
        multimodal_data['robot'] = robot_data
        
        # Sensor data
        sensor_data = self.environment_manager.collect_sensor_data(demonstration_id)
        multimodal_data['sensors'] = sensor_data
        
        # Audio data
        audio_data = self.environment_manager.collect_audio_data(demonstration_id)
        multimodal_data['audio'] = audio_data
        
        # Clinical data
        clinical_data = self.environment_manager.collect_clinical_data(demonstration_id)
        multimodal_data['clinical'] = clinical_data
        
        # Synchronize data
        synchronized_data = self.data_synchronizer.synchronize_data(multimodal_data)
        
        return synchronized_data
    
    def _collect_metadata(self, demonstration_id: str, clinical_context: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metadata for demonstration."""
        metadata = {
            'demonstration_id': demonstration_id,
            'collection_timestamp': datetime.now().isoformat(),
            'environment_config': self.current_session.environment_config.to_dict(),
            'operator_name': self.current_session.operator_name,
            'patient_id': self.current_session.patient_id,
            'medication_type': self.current_session.medication_type,
            'safety_level': self.current_session.safety_level.value,
            'clinical_context': clinical_context,
            'system_status': self.environment_manager.get_system_status(),
            'calibration_data': self.environment_manager.get_calibration_data()
        }
        
        return metadata
    
    def _store_demonstration(self, demonstration_data: Dict[str, Any]):
        """Store demonstration data."""
        # Add to collected data
        self.collected_data['demonstrations'].append(demonstration_data)
        
        # Save to file
        demo_dir = self.output_dir / self.current_session.session_id / 'demonstrations'
        demo_dir.mkdir(parents=True, exist_ok=True)
        
        demo_file = demo_dir / f"{demonstration_data['demonstration_id']}.json"
        with open(demo_file, 'w') as f:
            json.dump(demonstration_data, f, indent=2, default=str)
        
        # Save raw data files
        self._save_raw_data_files(demonstration_data, demo_dir)
    
    def _save_raw_data_files(self, demonstration_data: Dict[str, Any], demo_dir: Path):
        """Save raw data files for demonstration."""
        demo_id = demonstration_data['demonstration_id']
        
        # Save vision data
        if 'vision' in demonstration_data['data']:
            vision_dir = demo_dir / 'vision'
            vision_dir.mkdir(exist_ok=True)
            
            vision_data = demonstration_data['data']['vision']
            
            # Save RGB images
            if 'rgb_images' in vision_data:
                rgb_dir = vision_dir / 'rgb'
                rgb_dir.mkdir(exist_ok=True)
                
                for i, image in enumerate(vision_data['rgb_images']):
                    image_file = rgb_dir / f"frame_{i:04d}.jpg"
                    if isinstance(image, np.ndarray):
                        cv2.imwrite(str(image_file), image)
                    elif isinstance(image, str):
                        # Assume base64 encoded
                        import base64
                        image_data = base64.b64decode(image)
                        with open(image_file, 'wb') as f:
                            f.write(image_data)
            
            # Save depth images
            if 'depth_images' in vision_data:
                depth_dir = vision_dir / 'depth'
                depth_dir.mkdir(exist_ok=True)
                
                for i, depth_image in enumerate(vision_data['depth_images']):
                    depth_file = depth_dir / f"depth_{i:04d}.npy"
                    np.save(depth_file, depth_image)
        
        # Save robot data
        if 'robot' in demonstration_data['data']:
            robot_dir = demo_dir / 'robot'
            robot_dir.mkdir(exist_ok=True)
            
            robot_data = demonstration_data['data']['robot']
            
            # Save joint trajectories
            if 'joint_trajectories' in robot_data:
                trajectory_file = robot_dir / 'joint_trajectories.npy'
                np.save(trajectory_file, robot_data['joint_trajectories'])
            
            # Save end-effector poses
            if 'end_effector_poses' in robot_data:
                poses_file = robot_dir / 'end_effector_poses.npy'
                np.save(poses_file, robot_data['end_effector_poses'])
        
        # Save sensor data
        if 'sensors' in demonstration_data['data']:
            sensor_dir = demo_dir / 'sensors'
            sensor_dir.mkdir(exist_ok=True)
            
            sensor_data = demonstration_data['data']['sensors']
            
            # Save force/torque data
            if 'force_torque' in sensor_data:
                ft_file = sensor_dir / 'force_torque.npy'
                np.save(ft_file, sensor_data['force_torque'])
            
            # Save proximity data
            if 'proximity' in sensor_data:
                prox_file = sensor_dir / 'proximity.npy'
                np.save(prox_file, sensor_data['proximity'])
        
        # Save audio data
        if 'audio' in demonstration_data['data']:
            audio_dir = demo_dir / 'audio'
            audio_dir.mkdir(exist_ok=True)
            
            audio_data = demonstration_data['data']['audio']
            
            if 'audio_data' in audio_data:
                audio_file = audio_dir / 'audio.wav'
                import wave
                with wave.open(str(audio_file), 'wb') as wav_file:
                    wav_file.setnchannels(audio_data.get('channels', 1))
                    wav_file.setsampwidth(audio_data.get('sample_width', 2))
                    wav_file.setframerate(audio_data.get('sample_rate', 44100))
                    wav_file.writeframes(audio_data['audio_data'].tobytes())
    
    def _validate_session_data(self):
        """Validate all data collected in the session."""
        logger.info("Validating session data")
        
        self.current_session.status = DataCollectionStatus.VALIDATING
        
        # Validate demonstrations
        valid_demonstrations = []
        for demo in self.collected_data['demonstrations']:
            if demo.get('status') == 'accepted':
                valid_demonstrations.append(demo)
        
        # Update session statistics
        self.current_session.annotations['total_demonstrations'] = len(self.collected_data['demonstrations'])
        self.current_session.annotations['valid_demonstrations'] = len(valid_demonstrations)
        self.current_session.annotations['validation_rate'] = len(valid_demonstrations) / len(self.collected_data['demonstrations'])
        
        logger.info(f"Validation complete: {len(valid_demonstrations)}/{len(self.collected_data['demonstrations'])} demonstrations valid")
    
    def _generate_session_summary(self) -> Dict[str, Any]:
        """Generate session summary and statistics."""
        summary = {
            'session_info': self.current_session.to_dict(),
            'statistics': {
                'total_demonstrations': self.current_session.collected_demonstrations,
                'average_quality': self.current_session.quality_score,
                'collection_duration': (self.current_session.end_time - self.current_session.start_time).total_seconds(),
                'safety_events': len(self.safety_monitor.get_session_events()),
                'data_volume': self._calculate_data_volume()
            },
            'quality_metrics': self._aggregate_quality_metrics(),
            'safety_summary': self.safety_monitor.get_session_summary(),
            'recommendations': self._generate_recommendations()
        }
        
        return summary
    
    def _calculate_data_volume(self) -> Dict[str, float]:
        """Calculate total data volume collected."""
        volume = {
            'total_size_mb': 0.0,
            'vision_data_mb': 0.0,
            'robot_data_mb': 0.0,
            'sensor_data_mb': 0.0,
            'audio_data_mb': 0.0
        }
        
        session_dir = self.output_dir / self.current_session.session_id
        if session_dir.exists():
            for file_path in session_dir.rglob('*'):
                if file_path.is_file():
                    size_mb = file_path.stat().st_size / (1024 * 1024)
                    volume['total_size_mb'] += size_mb
                    
                    if 'vision' in str(file_path):
                        volume['vision_data_mb'] += size_mb
                    elif 'robot' in str(file_path):
                        volume['robot_data_mb'] += size_mb
                    elif 'sensors' in str(file_path):
                        volume['sensor_data_mb'] += size_mb
                    elif 'audio' in str(file_path):
                        volume['audio_data_mb'] += size_mb
        
        return volume
    
    def _aggregate_quality_metrics(self) -> Dict[str, float]:
        """Aggregate quality metrics across all demonstrations."""
        if not self.collected_data['demonstrations']:
            return {}
        
        metrics = {}
        
        # Collect all quality metrics
        all_metrics = []
        for demo in self.collected_data['demonstrations']:
            if 'quality_metrics' in demo:
                all_metrics.append(demo['quality_metrics'])
        
        if not all_metrics:
            return metrics
        
        # Aggregate each metric
        metric_names = set()
        for metric_dict in all_metrics:
            metric_names.update(metric_dict.keys())
        
        for metric_name in metric_names:
            values = [m.get(metric_name, 0) for m in all_metrics if metric_name in m]
            if values:
                metrics[f'mean_{metric_name}'] = np.mean(values)
                metrics[f'std_{metric_name}'] = np.std(values)
                metrics[f'min_{metric_name}'] = np.min(values)
                metrics[f'max_{metric_name}'] = np.max(values)
        
        return metrics
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on session data."""
        recommendations = []
        
        # Quality recommendations
        if self.current_session.quality_score < 0.8:
            recommendations.append("Consider improving demonstration quality - current score below optimal")
        
        # Safety recommendations
        safety_summary = self.safety_monitor.get_session_summary()
        if safety_summary['critical_events'] > 0:
            recommendations.append("Review safety protocols - critical events detected")
        
        # Volume recommendations
        volume = self._calculate_data_volume()
        if volume['total_size_mb'] > 1000:  # > 1GB
            recommendations.append("Consider data compression - large volume collected")
        
        # Duration recommendations
        duration = (self.current_session.end_time - self.current_session.start_time).total_seconds()
        if duration > 3600:  # > 1 hour
            recommendations.append("Consider shorter sessions - extended duration may affect quality")
        
        return recommendations
    
    def _save_session_data(self):
        """Save all session data to files."""
        session_dir = self.output_dir / self.current_session.session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save session summary
        summary = self._generate_session_summary()
        summary_file = session_dir / 'session_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Save collected data index
        index_file = session_dir / 'data_index.json'
        with open(index_file, 'w') as f:
            json.dump(self.collected_data, f, indent=2, default=str)
        
        # Save session metadata
        metadata_file = session_dir / 'session_metadata.json'
        with open(metadata_file, 'w') as f:
            json.dump(self.current_session.to_dict(), f, indent=2, default=str)
        
        logger.info(f"Session data saved to: {session_dir}")


class ClinicalEnvironmentManager:
    """Manages clinical environment setup and data collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.environment_config = None
        self.camera_systems = []
        self.robot_interface = None
        self.sensor_systems = []
        self.calibration_data = {}
        
    def setup_environment(self, environment_config: ClinicalEnvironmentConfig):
        """Setup clinical environment."""
        self.environment_config = environment_config
        
        # Initialize camera systems
        self._initialize_cameras()
        
        # Initialize robot interface
        self._initialize_robot()
        
        # Initialize sensor systems
        self._initialize_sensors()
        
        # Verify setup
        self._verify_setup()
    
    def _initialize_cameras(self):
        """Initialize camera systems."""
        camera_configs = self.config.get('cameras', [])
        
        for cam_config in camera_configs:
            camera = CameraSystem(cam_config)
            camera.initialize()
            self.camera_systems.append(camera)
    
    def _initialize_robot(self):
        """Initialize robot interface."""
        robot_config = self.config.get('robot', {})
        self.robot_interface = RobotInterface(robot_config)
        self.robot_interface.initialize()
    
    def _initialize_sensors(self):
        """Initialize sensor systems."""
        sensor_configs = self.config.get('sensors', [])
        
        for sensor_config in sensor_configs:
            sensor = SensorSystem(sensor_config)
            sensor.initialize()
            self.sensor_systems.append(sensor)
    
    def _verify_setup(self):
        """Verify environment setup."""
        # Check cameras
        for camera in self.camera_systems:
            if not camera.is_operational():
                raise RuntimeError(f"Camera {camera.id} not operational")
        
        # Check robot
        if not self.robot_interface.is_operational():
            raise RuntimeError("Robot interface not operational")
        
        # Check sensors
        for sensor in self.sensor_systems:
            if not sensor.is_operational():
                raise RuntimeError(f"Sensor {sensor.id} not operational")
    
    def calibrate_cameras(self):
        """Calibrate camera systems."""
        for camera in self.camera_systems:
            calibration_result = camera.calibrate()
            self.calibration_data[f'camera_{camera.id}'] = calibration_result
    
    def calibrate_robot(self):
        """Calibrate robot system."""
        calibration_result = self.robot_interface.calibrate()
        self.calibration_data['robot'] = calibration_result
    
    def calibrate_sensors(self):
        """Calibrate sensor systems."""
        for sensor in self.sensor_systems:
            calibration_result = sensor.calibrate()
            self.calibration_data[f'sensor_{sensor.id}'] = calibration_result
    
    def collect_vision_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect vision data."""
        vision_data = {
            'rgb_images': [],
            'depth_images': [],
            'camera_poses': [],
            'timestamps': []
        }
        
        for camera in self.camera_systems:
            camera_data = camera.capture_sequence()
            vision_data['rgb_images'].extend(camera_data['rgb_images'])
            vision_data['depth_images'].extend(camera_data['depth_images'])
            vision_data['camera_poses'].extend(camera_data['poses'])
            vision_data['timestamps'].extend(camera_data['timestamps'])
        
        return vision_data
    
    def collect_robot_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect robot state data."""
        return self.robot_interface.get_state_sequence()
    
    def collect_sensor_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect sensor data."""
        sensor_data = {}
        
        for sensor in self.sensor_systems:
            data = sensor.get_data_sequence()
            sensor_data[sensor.type] = data
        
        return sensor_data
    
    def collect_audio_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect audio data."""
        # Placeholder for audio collection
        return {
            'audio_data': np.random.randint(-32768, 32767, 44100, dtype=np.int16),
            'sample_rate': 44100,
            'channels': 1,
            'sample_width': 2
        }
    
    def collect_clinical_data(self, demonstration_id: str) -> Dict[str, Any]:
        """Collect clinical context data."""
        return {
            'patient_vitals': {
                'heart_rate': np.random.randint(60, 100),
                'blood_pressure': [120 + np.random.randint(-10, 10), 80 + np.random.randint(-5, 5)],
                'temperature': 36.5 + np.random.uniform(-0.5, 0.5)
            },
            'medication_info': {
                'name': self.environment_config.medication_types[0] if self.environment_config.medication_types else 'unknown',
                'dosage': '10mg',
                'administration_route': 'oral'
            },
            'environmental_conditions': {
                'temperature': 22.0 + np.random.uniform(-2, 2),
                'humidity': 45.0 + np.random.uniform(-10, 10),
                'lighting_level': np.random.uniform(300, 800)
            }
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'cameras': [camera.get_status() for camera in self.camera_systems],
            'robot': self.robot_interface.get_status(),
            'sensors': [sensor.get_status() for sensor in self.sensor_systems]
        }
    
    def get_calibration_data(self) -> Dict[str, Any]:
        """Get calibration data."""
        return self.calibration_data


class ClinicalSafetyMonitor:
    """Monitors safety during clinical data collection."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.safety_events = []
        self.monitoring_active = False
        self.safety_thresholds = config.get('safety_thresholds', {})
        
    def start_monitoring(self):
        """Start safety monitoring."""
        self.monitoring_active = True
        self.safety_events = []
    
    def stop_monitoring(self):
        """Stop safety monitoring."""
        self.monitoring_active = False
    
    def verify_environment_setup(self, environment_config: ClinicalEnvironmentConfig) -> Dict[str, Any]:
        """Verify environment setup meets safety requirements."""
        check_result = {
            'passed': True,
            'issues': [],
            'warnings': []
        }
        
        # Check workspace dimensions
        required_space = environment_config.safety_requirements.get('min_workspace_area', 10.0)
        actual_space = (environment_config.workspace_dimensions.get('length', 0) * 
                       environment_config.workspace_dimensions.get('width', 0))
        
        if actual_space < required_space:
            check_result['passed'] = False
            check_result['issues'].append(f"Insufficient workspace: {actual_space}m² < {required_space}m²")
        
        # Check emergency equipment
        required_equipment = environment_config.safety_requirements.get('emergency_equipment', [])
        available_equipment = environment_config.equipment_list
        
        for equipment in required_equipment:
            if equipment not in available_equipment:
                check_result['passed'] = False
                check_result['issues'].append(f"Missing emergency equipment: {equipment}")
        
        return check_result
    
    def get_session_events(self, demonstration_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get safety events for session or specific demonstration."""
        if demonstration_id:
            return [event for event in self.safety_events if event.get('demonstration_id') == demonstration_id]
        return self.safety_events
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get safety summary for current session."""
        critical_events = [e for e in self.safety_events if e.get('severity') == 'critical']
        warning_events = [e for e in self.safety_events if e.get('severity') == 'warning']
        
        return {
            'total_events': len(self.safety_events),
            'critical_events': len(critical_events),
            'warning_events': len(warning_events),
            'event_types': list(set(e.get('type', 'unknown') for e in self.safety_events)),
            'safety_score': 1.0 - (len(critical_events) * 0.5 + len(warning_events) * 0.1) / max(len(self.safety_events), 1)
        }


class DataQualityAssessor:
    """Assesses quality of collected demonstration data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.monitoring_active = False
        self.quality_metrics = []
        
    def start_monitoring(self):
        """Start quality monitoring."""
        self.monitoring_active = True
    
    def stop_monitoring(self):
        """Stop quality monitoring."""
        self.monitoring_active = False
    
    def assess_demonstration(self, multimodal_data: Dict[str, Any], metadata: Dict[str, Any]) -> Dict[str, float]:
        """Assess quality of a single demonstration."""
        quality_metrics = {}
        
        # Vision quality
        if 'vision' in multimodal_data:
            vision_quality = self._assess_vision_quality(multimodal_data['vision'])
            quality_metrics.update(vision_quality)
        
        # Robot data quality
        if 'robot' in multimodal_data:
            robot_quality = self._assess_robot_quality(multimodal_data['robot'])
            quality_metrics.update(robot_quality)
        
        # Sensor data quality
        if 'sensors' in multimodal_data:
            sensor_quality = self._assess_sensor_quality(multimodal_data['sensors'])
            quality_metrics.update(sensor_quality)
        
        # Overall quality
        quality_metrics['overall_quality'] = np.mean(list(quality_metrics.values()))
        
        return quality_metrics
    
    def _assess_vision_quality(self, vision_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess vision data quality."""
        quality = {}
        
        # RGB image quality
        if 'rgb_images' in vision_data:
            rgb_images = vision_data['rgb_images']
            if rgb_images:
                # Check image clarity
                clarity_scores = []
                for image in rgb_images:
                    if isinstance(image, np.ndarray):
                        # Compute Laplacian variance as clarity measure
                        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                        clarity = cv2.Laplacian(gray, cv2.CV_64F).var()
                        clarity_scores.append(clarity)
                
                if clarity_scores:
                    quality['rgb_clarity'] = np.mean(clarity_scores) / 1000.0  # Normalize
                    quality['rgb_consistency'] = 1.0 - (np.std(clarity_scores) / np.mean(clarity_scores))
        
        # Depth data quality
        if 'depth_images' in vision_data:
            depth_images = vision_data['depth_images']
            if depth_images:
                # Check depth validity
                valid_ratios = []
                for depth in depth_images:
                    if isinstance(depth, np.ndarray):
                        valid_pixels = np.sum(depth > 0)
                        total_pixels = depth.size
                        valid_ratios.append(valid_pixels / total_pixels)
                
                if valid_ratios:
                    quality['depth_validity'] = np.mean(valid_ratios)
        
        return quality
    
    def _assess_robot_quality(self, robot_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess robot data quality."""
        quality = {}
        
        # Joint trajectory quality
        if 'joint_trajectories' in robot_data:
            trajectories = robot_data['joint_trajectories']
            if isinstance(trajectories, np.ndarray) and trajectories.size > 0:
                # Check trajectory smoothness
                if trajectories.shape[0] > 2:
                    velocities = np.diff(trajectories, axis=0)
                    accelerations = np.diff(velocities, axis=0)
                    smoothness = 1.0 / (1.0 + np.var(accelerations))
                    quality['trajectory_smoothness'] = smoothness
                
                # Check data completeness
                expected_length = 100  # Expected number of timesteps
                completeness = trajectories.shape[0] / expected_length
                quality['data_completeness'] = min(completeness, 1.0)
        
        return quality
    
    def _assess_sensor_quality(self, sensor_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess sensor data quality."""
        quality = {}
        
        for sensor_type, data in sensor_data.items():
            if isinstance(data, np.ndarray) and data.size > 0:
                # Check for missing values
                missing_ratio = np.sum(np.isnan(data)) / data.size
                quality[f'{sensor_type}_completeness'] = 1.0 - missing_ratio
                
                # Check data range
                if sensor_type == 'force_torque':
                    # Force/torque should be within reasonable bounds
                    max_force = 50.0  # Newtons
                    max_torque = 5.0  # Newton-meters
                    valid_forces = np.abs(data[:, :3]) <= max_force
                    valid_torques = np.abs(data[:, 3:]) <= max_torque
                    validity_ratio = np.mean(np.all(valid_forces, axis=1) & np.all(valid_torques, axis=1))
                    quality[f'{sensor_type}_validity'] = validity_ratio
        
        return quality


class MultiModalDataSynchronizer:
    """Synchronizes multi-modal data streams."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.synchronization_active = False
        self.clock_offset = 0.0
        
    def start_synchronization(self):
        """Start data synchronization."""
        self.synchronization_active = True
        self._synchronize_clocks()
    
    def stop_synchronization(self):
        """Stop data synchronization."""
        self.synchronization_active = False
    
    def synchronize_clocks(self):
        """Synchronize all system clocks."""
        # Placeholder for clock synchronization
        self.clock_offset = 0.0
    
    def synchronize_data(self, multimodal_data: Dict[str, Any]) -> Dict[str, Any]:
        """Synchronize multi-modal data streams."""
        # Placeholder for data synchronization
        # In practice, this would align timestamps and interpolate data
        return multimodal_data


class ClinicalWorkflowIntegrator:
    """Integrates data collection with clinical workflows."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.workflow_state = {}
        
    def get_workflow_state(self) -> Dict[str, Any]:
        """Get current workflow state."""
        return self.workflow_state
    
    def update_workflow_state(self, state: Dict[str, Any]):
        """Update workflow state."""
        self.workflow_state.update(state)


# Placeholder classes for hardware interfaces
class CameraSystem:
    def __init__(self, config):
        self.config = config
        self.id = config.get('id', 'unknown')
    
    def initialize(self):
        pass
    
    def is_operational(self):
        return True
    
    def calibrate(self):
        return {'status': 'calibrated'}
    
    def capture_sequence(self):
        return {
            'rgb_images': [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)],
            'depth_images': [np.random.rand(480, 640).astype(np.float32) for _ in range(10)],
            'poses': [np.eye(4) for _ in range(10)],
            'timestamps': [time.time() + i * 0.1 for i in range(10)]
        }
    
    def get_status(self):
        return {'id': self.id, 'status': 'operational'}


class RobotInterface:
    def __init__(self, config):
        self.config = config
    
    def initialize(self):
        pass
    
    def is_operational(self):
        return True
    
    def calibrate(self):
        return {'status': 'calibrated'}
    
    def get_state_sequence(self):
        return {
            'joint_trajectories': np.random.rand(100, 7),
            'end_effector_poses': np.random.rand(100, 7),
            'gripper_states': np.random.rand(100, 1),
            'timestamps': [time.time() + i * 0.01 for i in range(100)]
        }
    
    def get_status(self):
        return {'status': 'operational'}


class SensorSystem:
    def __init__(self, config):
        self.config = config
        self.id = config.get('id', 'unknown')
        self.type = config.get('type', 'unknown')
    
    def initialize(self):
        pass
    
    def is_operational(self):
        return True
    
    def calibrate(self):
        return {'status': 'calibrated'}
    
    def get_data_sequence(self):
        if self.type == 'force_torque':
            return np.random.rand(100, 6)  # 6D force/torque
        elif self.type == 'proximity':
            return np.random.rand(100, 8)   # 8 proximity sensors
        else:
            return np.random.rand(100, 10)
    
    def get_status(self):
        return {'id': self.id, 'type': self.type, 'status': 'operational'}


def main():
    """Main function for clinical data collection protocol."""
    parser = argparse.ArgumentParser(description='Clinical Data Collection Protocol')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--output', type=str, default='./clinical_data', help='Output directory')
    parser.add_argument('--environment', type=str, default='hospital_pharmacy', help='Environment type')
    parser.add_argument('--operator', type=str, required=True, help='Operator name')
    parser.add_argument('--max-demos', type=int, default=50, help='Maximum demonstrations per session')
    
    args = parser.parse_args()
    
    # Load configuration
    config = {
        'output_dir': args.output,
        'max_demonstrations': args.max_demos,
        'min_quality_threshold': 0.7,
        'safety_checks': True,
        'real_time_validation': True,
        'cameras': [
            {'id': 'rgb_1', 'type': 'rgb', 'resolution': [640, 480]},
            {'id': 'depth_1', 'type': 'depth', 'resolution': [640, 480]}
        ],
        'robot': {'type': 'franka_panda'},
        'sensors': [
            {'id': 'ft_1', 'type': 'force_torque'},
            {'id': 'prox_1', 'type': 'proximity'}
        ]
    }
    
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            user_config = yaml.safe_load(f)
            config.update(user_config)
    
    # Initialize protocol
    protocol = ClinicalDataCollectionProtocol(config)
    
    # Create environment configuration
    environment_config = ClinicalEnvironmentConfig(
        environment_type=ClinicalEnvironmentType(args.environment),
        layout_description="Standard hospital pharmacy layout",
        workspace_dimensions={'length': 3.0, 'width': 2.0, 'height': 2.5},
        medication_types=['vial', 'blister_pack', 'syringe', 'bottle'],
        equipment_list=['robot_arm', 'medication_carousel', 'safety_barrier'],
        safety_requirements={'min_workspace_area': 6.0, 'emergency_equipment': ['emergency_stop']},
        personnel_requirements={'operator': True, 'supervisor': False},
        sterilization_procedures=['hand_disinfection', 'equipment_wiping'],
        workflow_constraints=['no_interruptions', 'sterile_field']
    )
    
    try:
        # Start collection session
        session_id = protocol.start_collection_session(
            environment_config=environment_config,
            operator_name=args.operator,
            safety_level=SafetyLevel.MEDIUM_RISK
        )
        
        logger.info(f"Started collection session: {session_id}")
        
        # Collect demonstrations (example)
        for i in range(5):
            demonstration = protocol.collect_demonstration(
                instruction=f"Pick up the medication from shelf {i+1}",
                target_medication="vial",
                grasp_type="precision",
                clinical_context={
                    'urgency': 'routine',
                    'patient_condition': 'stable',
                    'medication_priority': 'normal'
                }
            )
            
            logger.info(f"Collected demonstration {i+1}: {demonstration['status']}")
        
        # End session
        summary = protocol.end_collection_session()
        
        logger.info("Collection session completed successfully")
        logger.info(f"Total demonstrations: {summary['statistics']['total_demonstrations']}")
        logger.info(f"Average quality: {summary['statistics']['average_quality']:.3f}")
        
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
