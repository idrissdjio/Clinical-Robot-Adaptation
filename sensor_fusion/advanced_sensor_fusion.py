#!/usr/bin/env python3
"""
Advanced Sensor Fusion System for Clinical Robotics
Multi-sensor data fusion for enhanced perception and understanding.

This module implements:
- Multi-modal sensor data fusion (vision, depth, IMU, force/torque)
- Kalman filtering and state estimation
- Sensor calibration and synchronization
- Real-time fusion pipeline
- Uncertainty quantification
- Fault detection and sensor health monitoring
- Adaptive fusion weights
- Clinical environment perception

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
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import warnings

# Scientific computing
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import savgol_filter
from scipy.stats import multivariate_normal
import pandas as pd

# Computer vision
import cv2
import open3d as o3d

# Machine learning
import torch
import torch.nn as nn
import torch.nn.functional as F

# Kalman filtering
from filterpy.kalman import KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter
from filterpy.common import Q_discrete_white_noise

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sensor_fusion.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class SensorType(Enum):
    """Types of sensors."""
    RGB_CAMERA = "rgb_camera"
    DEPTH_CAMERA = "depth_camera"
    LIDAR = "lidar"
    IMU = "imu"
    FORCE_TORQUE = "force_torque"
    TACTILE = "tactile"
    MICROPHONE = "microphone"
    ULTRASOUND = "ultrasound"

class FusionMethod(Enum):
    """Sensor fusion methods."""
    KALMAN_FILTER = "kalman_filter"
    EXTENDED_KALMAN_FILTER = "extended_kalman_filter"
    PARTICLE_FILTER = "particle_filter"
    NEURAL_FUSION = "neural_fusion"
    BAYESIAN_FUSION = "bayesian_fusion"
    WEIGHTED_AVERAGE = "weighted_average"

class CalibrationStatus(Enum):
    """Calibration status."""
    UNCALIBRATED = "uncalibrated"
    CALIBRATING = "calibrating"
    CALIBRATED = "calibrated"
    ERROR = "error"

@dataclass
class SensorData:
    """Data from a single sensor."""
    sensor_id: str
    sensor_type: SensorType
    timestamp: datetime
    data: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0
    covariance: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sensor_id': self.sensor_id,
            'sensor_type': self.sensor_type.value,
            'timestamp': self.timestamp.isoformat(),
            'data_shape': self.data.shape,
            'confidence': self.confidence,
            'metadata': self.metadata
        }

@dataclass
class FusionConfig:
    """Configuration for sensor fusion."""
    
    # Fusion settings
    fusion_method: FusionMethod = FusionMethod.KALMAN_FILTER
    fusion_frequency: float = 100.0  # Hz
    synchronization_tolerance: float = 0.01  # seconds
    
    # Sensor settings
    enabled_sensors: List[SensorType] = field(default_factory=lambda: [
        SensorType.RGB_CAMERA,
        SensorType.DEPTH_CAMERA,
        SensorType.IMU,
        SensorType.FORCE_TORQUE
    ])
    
    # Kalman filter settings
    process_noise: float = 0.1
    measurement_noise: float = 0.1
    initial_covariance: float = 1.0
    
    # Neural fusion settings
    neural_fusion_model: str = ""
    neural_fusion_device: str = "cuda"
    
    # Calibration settings
    auto_calibrate: bool = True
    calibration_interval: int = 3600  # seconds
    
    # Fault detection
    enable_fault_detection: bool = True
    fault_threshold: float = 3.0  # standard deviations
    
    # Output settings
    output_state_dim: int = 15  # position (3), orientation (4), velocity (3), angular_velocity (3), force (3)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'fusion_method': self.fusion_method.value,
            'fusion_frequency': self.fusion_frequency,
            'synchronization_tolerance': self.synchronization_tolerance,
            'enabled_sensors': [s.value for s in self.enabled_sensors],
            'process_noise': self.process_noise,
            'measurement_noise': self.measurement_noise,
            'initial_covariance': self.initial_covariance,
            'neural_fusion_model': self.neural_fusion_model,
            'neural_fusion_device': self.neural_fusion_device,
            'auto_calibrate': self.auto_calibrate,
            'calibration_interval': self.calibration_interval,
            'enable_fault_detection': self.enable_fault_detection,
            'fault_threshold': self.fault_threshold,
            'output_state_dim': self.output_state_dim
        }

class SensorCalibrator:
    """Sensor calibration and synchronization."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        self.calibration_status = {}
        self.calibration_matrices = {}
    
    def calibrate_camera(self, camera_id: str, images: List[np.ndarray], 
                        pattern_size: Tuple[int, int] = (9, 6)) -> bool:
        """Calibrate camera using chessboard pattern."""
        logger.info(f"Calibrating camera {camera_id}")
        
        # Prepare object points
        objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)
        
        objpoints = []
        imgpoints = []
        
        # Find chessboard corners
        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
            
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
        
        if len(objpoints) == 0:
            logger.error("No chessboard corners found")
            self.calibration_status[camera_id] = CalibrationStatus.ERROR
            return False
        
        # Calibrate camera
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        
        if ret:
            self.calibration_matrices[camera_id] = {
                'camera_matrix': mtx,
                'distortion_coefficients': dist,
                'rotation_vectors': rvecs,
                'translation_vectors': tvecs
            }
            self.calibration_status[camera_id] = CalibrationStatus.CALIBRATED
            logger.info(f"Camera {camera_id} calibrated successfully")
            return True
        else:
            self.calibration_status[camera_id] = CalibrationStatus.ERROR
            logger.error(f"Camera {camera_id} calibration failed")
            return False
    
    def calibrate_imu(self, imu_id: str, data: List[Dict[str, np.ndarray]]) -> bool:
        """Calibrate IMU (gyroscope and accelerometer)."""
        logger.info(f"Calibrating IMU {imu_id}")
        
        # Calculate bias and scale factors
        accelerometer_data = np.array([d['accelerometer'] for d in data])
        gyroscope_data = np.array([d['gyroscope'] for d in data])
        
        # Calculate bias (mean of stationary data)
        accelerometer_bias = np.mean(accelerometer_data, axis=0)
        gyroscope_bias = np.mean(gyroscope_data, axis=0)
        
        # Calculate scale factors
        accelerometer_std = np.std(accelerometer_data, axis=0)
        gyroscope_std = np.std(gyroscope_data, axis=0)
        
        self.calibration_matrices[imu_id] = {
            'accelerometer_bias': accelerometer_bias,
            'gyroscope_bias': gyroscope_bias,
            'accelerometer_scale': 9.81 / accelerometer_std,  # Assuming 1g
            'gyroscope_scale': 1.0 / gyroscope_std
        }
        
        self.calibration_status[imu_id] = CalibrationStatus.CALIBRATED
        logger.info(f"IMU {imu_id} calibrated successfully")
        return True
    
    def synchronize_sensors(self, sensor_data: Dict[str, List[SensorData]]) -> Dict[str, List[SensorData]]:
        """Synchronize sensor data timestamps."""
        # Find reference timestamp (e.g., most recent camera frame)
        reference_timestamp = None
        
        for sensor_id, data_list in sensor_data.items():
            if data_list:
                latest = max(data_list, key=lambda x: x.timestamp)
                if reference_timestamp is None or latest.timestamp > reference_timestamp:
                    reference_timestamp = latest.timestamp
        
        if reference_timestamp is None:
            return sensor_data
        
        # Synchronize data to reference timestamp
        synchronized_data = {}
        for sensor_id, data_list in sensor_data.items():
            # Find closest data point
            synchronized = []
            for data in data_list:
                time_diff = abs((data.timestamp - reference_timestamp).total_seconds())
                if time_diff <= self.config.synchronization_tolerance:
                    synchronized.append(data)
            
            synchronized_data[sensor_id] = synchronized
        
        return synchronized_data

class NeuralFusionNetwork(nn.Module):
    """Neural network for sensor fusion."""
    
    def __init__(self, input_dims: Dict[str, int], output_dim: int):
        super().__init__()
        
        # Encoders for each sensor type
        self.vision_encoder = nn.Sequential(
            nn.Linear(input_dims.get('vision', 1000), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.depth_encoder = nn.Sequential(
            nn.Linear(input_dims.get('depth', 1000), 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128)
        )
        
        self.imu_encoder = nn.Sequential(
            nn.Linear(input_dims.get('imu', 6), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        self.force_encoder = nn.Sequential(
            nn.Linear(input_dims.get('force', 6), 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32)
        )
        
        # Fusion layer
        fusion_dim = 128 + 128 + 32 + 32
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, output_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=4)
    
    def forward(self, sensor_inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Forward pass through neural fusion network."""
        features = []
        
        # Encode each sensor type
        if 'vision' in sensor_inputs:
            vision_features = self.vision_encoder(sensor_inputs['vision'])
            features.append(vision_features)
        
        if 'depth' in sensor_inputs:
            depth_features = self.depth_encoder(sensor_inputs['depth'])
            features.append(depth_features)
        
        if 'imu' in sensor_inputs:
            imu_features = self.imu_encoder(sensor_inputs['imu'])
            features.append(imu_features)
        
        if 'force' in sensor_inputs:
            force_features = self.force_encoder(sensor_inputs['force'])
            features.append(force_features)
        
        # Concatenate features
        if features:
            fused = torch.cat(features, dim=1)
        else:
            fused = torch.zeros(sensor_inputs[list(sensor_inputs.keys())[0]].size(0), 128)
        
        # Apply attention
        fused = fused.unsqueeze(1)  # Add sequence dimension
        attended, _ = self.attention(fused, fused, fused)
        attended = attended.squeeze(1)
        
        # Final fusion
        output = self.fusion(attended)
        
        return output

class SensorFusionSystem:
    """Main sensor fusion system."""
    
    def __init__(self, config: FusionConfig):
        self.config = config
        
        # Initialize components
        self.calibrator = SensorCalibrator(config)
        self.neural_fusion_model = None
        
        # Initialize fusion filter
        if config.fusion_method == FusionMethod.KALMAN_FILTER:
            self.fusion_filter = self._init_kalman_filter()
        elif config.fusion_method == FusionMethod.EXTENDED_KALMAN_FILTER:
            self.fusion_filter = self._init_extended_kalman_filter()
        elif config.fusion_method == FusionMethod.NEURAL_FUSION:
            self.neural_fusion_model = self._init_neural_fusion()
        
        # Sensor data buffers
        self.sensor_data = defaultdict(lambda: deque(maxlen=100))
        self.fused_state = deque(maxlen=100)
        
        # Fault detection
        self.sensor_health = defaultdict(lambda: 1.0)
        
        # Monitoring
        self.is_running = False
        self.fusion_thread = None
        
        logger.info("Sensor Fusion System initialized")
    
    def _init_kalman_filter(self) -> KalmanFilter:
        """Initialize Kalman filter."""
        dim_x = self.config.output_state_dim
        dim_z = dim_x  # Assuming same dimension for simplicity
        
        kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initialize state covariance
        kf.P *= self.config.initial_covariance
        
        # Initialize process noise
        kf.Q = Q_discrete_white_noise(dim=dim_x, dt=1.0/self.config.fusion_frequency, var=self.config.process_noise)
        
        # Initialize measurement noise
        kf.R = Q_discrete_white_noise(dim=dim_z, dt=1.0/self.config.fusion_frequency, var=self.config.measurement_noise)
        
        return kf
    
    def _init_extended_kalman_filter(self) -> ExtendedKalmanFilter:
        """Initialize Extended Kalman Filter."""
        dim_x = self.config.output_state_dim
        dim_z = dim_x
        
        ekf = ExtendedKalmanFilter(dim_x=dim_x, dim_z=dim_z)
        
        # Initialize state covariance
        ekf.P *= self.config.initial_covariance
        
        return ekf
    
    def _init_neural_fusion(self) -> NeuralFusionNetwork:
        """Initialize neural fusion network."""
        input_dims = {
            'vision': 1000,  # Placeholder for vision features
            'depth': 1000,
            'imu': 6,
            'force': 6
        }
        
        model = NeuralFusionNetwork(input_dims, self.config.output_state_dim)
        
        if torch.cuda.is_available() and self.config.neural_fusion_device == "cuda":
            model = model.cuda()
        
        return model
    
    def add_sensor_data(self, sensor_data: SensorData):
        """Add sensor data to fusion system."""
        self.sensor_data[sensor_data.sensor_id].append(sensor_data)
        
        # Perform fault detection
        if self.config.enable_fault_detection:
            self._detect_sensor_fault(sensor_data)
    
    def _detect_sensor_fault(self, sensor_data: SensorData):
        """Detect sensor faults using statistical methods."""
        sensor_id = sensor_data.sensor_id
        data_buffer = self.sensor_data[sensor_id]
        
        if len(data_buffer) < 10:
            return
        
        # Calculate statistics
        recent_data = [d.data for d in list(data_buffer)[-10:]]
        mean = np.mean(recent_data, axis=0)
        std = np.std(recent_data, axis=0)
        
        # Check for anomalies
        z_scores = np.abs((sensor_data.data - mean) / (std + 1e-6))
        
        if np.any(z_scores > self.config.fault_threshold):
            self.sensor_health[sensor_id] *= 0.9  # Decrease health score
            logger.warning(f"Potential fault detected in sensor {sensor_id}")
        else:
            self.sensor_health[sensor_id] = min(1.0, self.sensor_health[sensor_id] + 0.01)
    
    def fuse_sensor_data(self, sensor_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse sensor data using configured method."""
        if self.config.fusion_method == FusionMethod.KALMAN_FILTER:
            return self._kalman_fusion(sensor_inputs)
        elif self.config.fusion_method == FusionMethod.EXTENDED_KALMAN_FILTER:
            return self._extended_kalman_fusion(sensor_inputs)
        elif self.config.fusion_method == FusionMethod.NEURAL_FUSION:
            return self._neural_fusion(sensor_inputs)
        elif self.config.fusion_method == FusionMethod.WEIGHTED_AVERAGE:
            return self._weighted_average_fusion(sensor_inputs)
        else:
            return self._weighted_average_fusion(sensor_inputs)
    
    def _kalman_fusion(self, sensor_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse sensor data using Kalman filter."""
        # Combine sensor inputs into measurement vector
        measurement = np.concatenate(list(sensor_inputs.values()))
        
        # Ensure dimensions match
        if measurement.shape[0] != self.fusion_filter.dim_z:
            # Pad or truncate
            if measurement.shape[0] < self.fusion_filter.dim_z:
                measurement = np.pad(measurement, (0, self.fusion_filter.dim_z - measurement.shape[0]))
            else:
                measurement = measurement[:self.fusion_filter.dim_z]
        
        # Predict and update
        self.fusion_filter.predict()
        self.fusion_filter.update(measurement)
        
        # Get state estimate
        state = self.fusion_filter.x
        
        return state
    
    def _extended_kalman_fusion(self, sensor_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse sensor data using Extended Kalman Filter."""
        # Placeholder for EKF implementation
        # In practice, this would implement non-linear state estimation
        return self._weighted_average_fusion(sensor_inputs)
    
    def _neural_fusion(self, sensor_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse sensor data using neural network."""
        if self.neural_fusion_model is None:
            return self._weighted_average_fusion(sensor_inputs)
        
        # Convert inputs to tensors
        tensor_inputs = {}
        for key, value in sensor_inputs.items():
            tensor_inputs[key] = torch.FloatTensor(value)
            if torch.cuda.is_available():
                tensor_inputs[key] = tensor_inputs[key].cuda()
        
        # Forward pass
        with torch.no_grad():
            output = self.neural_fusion_model(tensor_inputs)
        
        # Convert back to numpy
        if torch.cuda.is_available():
            output = output.cpu()
        
        return output.numpy()
    
    def _weighted_average_fusion(self, sensor_inputs: Dict[str, np.ndarray]) -> np.ndarray:
        """Fuse sensor data using weighted average."""
        # Calculate weights based on sensor health
        weights = []
        for sensor_id in sensor_inputs.keys():
            weight = self.sensor_health.get(sensor_id, 1.0)
            weights.append(weight)
        
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        # Combine sensor inputs
        combined = np.concatenate(list(sensor_inputs.values()))
        
        # Calculate weighted average
        # This is simplified - in practice would handle different dimensions properly
        fused_state = np.zeros(combined.shape)
        idx = 0
        for i, (sensor_id, data) in enumerate(sensor_inputs.items()):
            data_flat = data.flatten()
            fused_state[idx:idx+len(data_flat)] = weights[i] * data_flat
            idx += len(data_flat)
        
        return fused_state
    
    def start_fusion(self):
        """Start continuous sensor fusion."""
        if self.is_running:
            return
        
        self.is_running = True
        self.fusion_thread = threading.Thread(target=self._fusion_loop, daemon=True)
        self.fusion_thread.start()
        
        logger.info("Sensor fusion started")
    
    def stop_fusion(self):
        """Stop continuous sensor fusion."""
        self.is_running = False
        
        if self.fusion_thread:
            self.fusion_thread.join(timeout=5)
        
        logger.info("Sensor fusion stopped")
    
    def _fusion_loop(self):
        """Main fusion loop."""
        while self.is_running:
            try:
                # Collect latest sensor data
                latest_data = {}
                for sensor_id, data_buffer in self.sensor_data.items():
                    if data_buffer:
                        latest_data[sensor_id] = data_buffer[-1].data
                
                # Fuse data if multiple sensors available
                if len(latest_data) > 1:
                    fused_state = self.fuse_sensor_data(latest_data)
                    self.fused_state.append({
                        'timestamp': datetime.now(),
                        'state': fused_state
                    })
                
                time.sleep(1.0 / self.config.fusion_frequency)
                
            except Exception as e:
                logger.error(f"Fusion loop error: {e}")
    
    def get_fused_state(self) -> Optional[np.ndarray]:
        """Get current fused state estimate."""
        if self.fused_state:
            return self.fused_state[-1]['state']
        return None
    
    def get_sensor_health(self) -> Dict[str, float]:
        """Get health status of all sensors."""
        return dict(self.sensor_health)
    
    def generate_fusion_report(self, output_path: str = None) -> str:
        """Generate sensor fusion report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"./sensor_fusion_reports/fusion_report_{timestamp}.md"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Generate report
        report_content = f"""# Sensor Fusion Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Configuration

- **Fusion Method**: {self.config.fusion_method.value}
- **Fusion Frequency**: {self.config.fusion_frequency} Hz
- **Enabled Sensors**: {', '.join([s.value for s in self.config.enabled_sensors])}
- **Fault Detection**: {'Enabled' if self.config.enable_fault_detection else 'Disabled'}

## Sensor Health

"""
        
        for sensor_id, health in self.sensor_health.items():
            status = "Healthy" if health > 0.8 else "Degraded" if health > 0.5 else "Faulty"
            report_content += f"- **{sensor_id}**: {health:.2f} ({status})\n"
        
        report_content += f"""
## Calibration Status

"""
        
        for sensor_id, status in self.calibrator.calibration_status.items():
            report_content += f"- **{sensor_id}**: {status.value}\n"
        
        report_content += """
## Recommendations

"""
        
        # Add recommendations based on sensor health
        unhealthy_sensors = [sid for sid, health in self.sensor_health.items() if health < 0.8]
        if unhealthy_sensors:
            report_content += "### High Priority\n\n"
            for sensor_id in unhealthy_sensors:
                report_content += f"- Check sensor {sensor_id} for faults\n"
                report_content += f"- Consider recalibration or replacement\n"
        
        report_content += "\n---\n"
        report_content += "*Report generated by Sensor Fusion System*\n"
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(report_content)
        
        logger.info(f"Fusion report saved to: {output_path}")
        
        return str(output_path)

def main():
    """Main function for sensor fusion."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Sensor Fusion System')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--method', type=str, default='kalman',
                       choices=['kalman', 'ekf', 'neural', 'weighted'],
                       help='Fusion method')
    parser.add_argument('--action', type=str, default='fusion',
                       choices=['fusion', 'report', 'calibrate'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    # Load configuration
    config = FusionConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    config.fusion_method = FusionMethod(args.method)
    
    # Create fusion system
    fusion_system = SensorFusionSystem(config)
    
    # Perform action
    if args.action == "fusion":
        fusion_system.start_fusion()
        print("Sensor fusion running. Press Ctrl+C to stop.")
        try:
            while True:
                state = fusion_system.get_fused_state()
                if state is not None:
                    print(f"Fused state: {state[:5]}...")  # Print first 5 elements
                health = fusion_system.get_sensor_health()
                print(f"Sensor health: {health}")
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping fusion...")
            fusion_system.stop_fusion()
    
    elif args.action == "report":
        report_path = fusion_system.generate_fusion_report()
        print(f"Fusion report generated: {report_path}")
    
    elif args.action == "calibrate":
        print("Calibration requires sensor data - implement specific calibration procedure")

if __name__ == "__main__":
    main()
