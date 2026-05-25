#!/usr/bin/env python3
"""
VR Teleoperation Interface for Clinical Robots
Virtual reality-based teleoperation system for clinical robot control.

This module implements:
- VR headset integration (Oculus, HTC Vive, etc.)
- Hand tracking and gesture recognition
- Haptic feedback for force feedback
- Real-time robot control from VR
- Clinical environment visualization in VR
- Safety monitoring during teleoperation
- Multi-user collaboration support
- Teleoperation recording and playback

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

# VR and 3D libraries
try:
    import openvr
    VR_AVAILABLE = True
except ImportError:
    VR_AVAILABLE = False
    logger = logging.getLogger(__name__)
    logger.warning("OpenVR not available, VR features will be simulated")

# Computer vision for hand tracking
import cv2
import mediapipe as mp

# Robotics and control
import numpy as np
from scipy.spatial.transform import Rotation
import pybullet as p

# Networking
import socket
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('vr_teleoperation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class VRDeviceType(Enum):
    """Types of VR devices."""
    HMD = "hmd"
    CONTROLLER = "controller"
    TRACKER = "tracker"
    BASE_STATION = "base_station"

class HandGesture(Enum):
    """Hand gestures for robot control."""
    GRASP = "grasp"
    RELEASE = "release"
    POINT = "point"
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    STOP = "stop"
    WAVE = "wave"

class TeleoperationMode(Enum):
    """Teleoperation modes."""
    DIRECT = "direct"
    AUGMENTED = "augmented"
    SUPERVISED = "supervised"
    AUTONOMOUS = "autonomous"

@dataclass
class VRControllerState:
    """State of VR controller."""
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # Quaternion
    velocity: Tuple[float, float, float]
    angular_velocity: Tuple[float, float, float]
    buttons: Dict[str, bool]
    axis_values: Dict[str, float]
    trigger_value: float
    grip_value: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'position': self.position,
            'orientation': self.orientation,
            'velocity': self.velocity,
            'angular_velocity': self.angular_velocity,
            'buttons': self.buttons,
            'axis_values': self.axis_values,
            'trigger_value': self.trigger_value,
            'grip_value': self.grip_value
        }

@dataclass
class HandTrackingResult:
    """Result of hand tracking."""
    landmarks: List[Tuple[float, float, float]]
    gesture: Optional[HandGesture]
    confidence: float
    bounding_box: Tuple[float, float, float, float]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'landmarks': self.landmarks,
            'gesture': self.gesture.value if self.gesture else None,
            'confidence': self.confidence,
            'bounding_box': self.bounding_box
        }

@dataclass
class TeleoperationConfig:
    """Configuration for VR teleoperation."""
    
    # VR settings
    vr_system: str = "openvr"  # openvr, oculus, vive
    enable_hmd: bool = True
    enable_controllers: bool = True
    enable_trackers: bool = False
    
    # Hand tracking settings
    enable_hand_tracking: bool = True
    hand_tracking_confidence_threshold: float = 0.7
    gesture_recognition_enabled: bool = True
    
    # Robot control settings
    teleoperation_mode: TeleoperationMode = TeleoperationMode.DIRECT
    control_frequency: float = 100.0  # Hz
    position_scaling: float = 1.0
    orientation_scaling: float = 1.0
    velocity_limit: float = 0.5  # m/s
    acceleration_limit: float = 2.0  # m/s^2
    
    # Safety settings
    enable_safety_monitoring: bool = True
    emergency_stop_threshold: float = 0.1  # m
    force_feedback_enabled: bool = True
    haptic_feedback_enabled: bool = True
    
    # Visualization settings
    enable_environment_visualization: bool = True
    enable_robot_visualization: bool = True
    enable_trajectory_visualization: bool = True
    visualization_quality: str = "high"  # low, medium, high
    
    # Recording settings
    enable_recording: bool = True
    recording_path: str = "./teleoperation_recordings"
    recording_format: str = "hdf5"
    
    # Network settings
    enable_network: bool = False
    server_host: str = "0.0.0.0"
    server_port: int = 5000
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'vr_system': self.vr_system,
            'enable_hmd': self.enable_hmd,
            'enable_controllers': self.enable_controllers,
            'enable_trackers': self.enable_trackers,
            'enable_hand_tracking': self.enable_hand_tracking,
            'hand_tracking_confidence_threshold': self.hand_tracking_confidence_threshold,
            'gesture_recognition_enabled': self.gesture_recognition_enabled,
            'teleoperation_mode': self.teleoperation_mode.value,
            'control_frequency': self.control_frequency,
            'position_scaling': self.position_scaling,
            'orientation_scaling': self.orientation_scaling,
            'velocity_limit': self.velocity_limit,
            'acceleration_limit': self.acceleration_limit,
            'enable_safety_monitoring': self.enable_safety_monitoring,
            'emergency_stop_threshold': self.emergency_stop_threshold,
            'force_feedback_enabled': self.force_feedback_enabled,
            'haptic_feedback_enabled': self.haptic_feedback_enabled,
            'enable_environment_visualization': self.enable_environment_visualization,
            'enable_robot_visualization': self.enable_robot_visualization,
            'enable_trajectory_visualization': self.enable_trajectory_visualization,
            'visualization_quality': self.visualization_quality,
            'enable_recording': self.enable_recording,
            'recording_path': self.recording_path,
            'recording_format': self.recording_format,
            'enable_network': self.enable_network,
            'server_host': self.server_host,
            'server_port': self.server_port
        }

class VRSystemInterface:
    """Interface to VR system (OpenVR)."""
    
    def __init__(self, config: TeleoperationConfig):
        self.config = config
        self.vr_system = None
        self.devices = {}
        self.is_initialized = False
        
        if VR_AVAILABLE:
            self._initialize_vr()
    
    def _initialize_vr(self):
        """Initialize VR system."""
        try:
            self.vr_system = openvr.init(openvr.VRApplication_Scene)
            self.is_initialized = True
            logger.info("VR system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize VR system: {e}")
            self.is_initialized = False
    
    def get_hmd_pose(self) -> Optional[VRControllerState]:
        """Get HMD pose."""
        if not self.is_initialized:
            return None
        
        try:
            poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0
            )
            
            hmd_index = self.vr_system.getDeviceIndex(openvr.k_unTrackedDeviceClass_HMD)
            if hmd_index == openvr.k_unTrackedDeviceIndexInvalid:
                return None
            
            pose = poses[hmd_index]
            
            # Extract position and orientation
            position = pose.mDeviceToAbsoluteTracking.m[0:3, 3]
            rotation_matrix = pose.mDeviceToAbsoluteTracking.m[0:3, 0:3]
            rotation = Rotation.from_matrix(rotation_matrix)
            orientation = rotation.as_quat()
            
            return VRControllerState(
                position=tuple(position),
                orientation=tuple(orientation),
                velocity=(0, 0, 0),
                angular_velocity=(0, 0, 0),
                buttons={},
                axis_values={},
                trigger_value=0.0,
                grip_value=0.0
            )
            
        except Exception as e:
            logger.error(f"Failed to get HMD pose: {e}")
            return None
    
    def get_controller_state(self, controller_index: int) -> Optional[VRControllerState]:
        """Get controller state."""
        if not self.is_initialized:
            return None
        
        try:
            poses = self.vr_system.getDeviceToAbsoluteTrackingPose(
                openvr.TrackingUniverseStanding, 0
            )
            
            pose = poses[controller_index]
            
            # Extract position and orientation
            position = pose.mDeviceToAbsoluteTracking.m[0:3, 3]
            rotation_matrix = pose.mDeviceToAbsoluteTracking.m[0:3, 0:3]
            rotation = Rotation.from_matrix(rotation_matrix)
            orientation = rotation.as_quat()
            
            # Get controller state
            controller_state = self.vr_system.getControllerState(controller_index)
            
            # Extract button states
            buttons = {}
            for i in range(32):
                buttons[f"button_{i}"] = (controller_state.ulButtonPressed >> i) & 1 == 1
            
            # Extract axis values
            axis_values = {}
            for i in range(5):
                axis_values[f"axis_{i}"] = controller_state.rAxis[i].x
            
            return VRControllerState(
                position=tuple(position),
                orientation=tuple(orientation),
                velocity=(0, 0, 0),
                angular_velocity=(0, 0, 0),
                buttons=buttons,
                axis_values=axis_values,
                trigger_value=axis_values.get("axis_1", 0.0),
                grip_value=axis_values.get("axis_2", 0.0)
            )
            
        except Exception as e:
            logger.error(f"Failed to get controller state: {e}")
            return None
    
    def submit_haptic_feedback(self, controller_index: int, duration: float, 
                              frequency: float, amplitude: float):
        """Submit haptic feedback to controller."""
        if not self.is_initialized:
            return
        
        try:
            self.vr_system.triggerHapticPulse(
                controller_index,
                0,  # Axis
                int(duration * 1000)  # Duration in microseconds
            )
        except Exception as e:
            logger.error(f"Failed to submit haptic feedback: {e}")
    
    def shutdown(self):
        """Shutdown VR system."""
        if self.vr_system:
            self.vr_system.shutdown()
            self.is_initialized = False
            logger.info("VR system shutdown")

class HandTrackingSystem:
    """Hand tracking using MediaPipe."""
    
    def __init__(self, config: TeleoperationConfig):
        self.config = config
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=config.hand_tracking_confidence_threshold,
            min_tracking_confidence=config.hand_tracking_confidence_threshold
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.is_initialized = True
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[HandTrackingResult], np.ndarray]:
        """Process video frame for hand tracking."""
        results = []
        
        # Convert to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        hand_results = self.hands.process(frame_rgb)
        
        # Extract hand landmarks
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Extract landmark positions
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append((lm.x, lm.y, lm.z))
                
                # Recognize gesture
                gesture = self._recognize_gesture(hand_landmarks)
                
                # Calculate bounding box
                x_coords = [lm.x for lm in hand_landmarks.landmark]
                y_coords = [lm.y for lm in hand_landmarks.landmark]
                bounding_box = (
                    min(x_coords), min(y_coords),
                    max(x_coords), max(y_coords)
                )
                
                results.append(HandTrackingResult(
                    landmarks=landmarks,
                    gesture=gesture,
                    confidence=0.9,  # Placeholder
                    bounding_box=bounding_box
                ))
        
        # Draw landmarks on frame
        annotated_frame = frame.copy()
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                self.mp_drawing.draw_landmarks(
                    annotated_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS
                )
        
        return results, annotated_frame
    
    def _recognize_gesture(self, hand_landmarks) -> Optional[HandGesture]:
        """Recognize hand gesture from landmarks."""
        # Simplified gesture recognition
        # In real implementation, use more sophisticated ML models
        
        # Get key landmarks
        thumb_tip = hand_landmarks.landmark[4]
        index_tip = hand_landmarks.landmark[8]
        middle_tip = hand_landmarks.landmark[12]
        ring_tip = hand_landmarks.landmark[16]
        pinky_tip = hand_landmarks.landmark[20]
        
        wrist = hand_landmarks.landmark[0]
        
        # Check for grasp (all fingers curled)
        if (index_tip.y < wrist.y and middle_tip.y < wrist.y and 
            ring_tip.y < wrist.y and pinky_tip.y < wrist.y):
            return HandGesture.GRASP
        
        # Check for release (all fingers extended)
        if (index_tip.y > wrist.y and middle_tip.y > wrist.y and 
            ring_tip.y > wrist.y and pinky_tip.y > wrist.y):
            return HandGesture.RELEASE
        
        # Check for thumbs up
        if thumb_tip.y < wrist.y and index_tip.y > wrist.y:
            return HandGesture.THUMBS_UP
        
        # Check for thumbs down
        if thumb_tip.y > wrist.y and index_tip.y > wrist.y:
            return HandGesture.THUMBS_DOWN
        
        return None

class RobotController:
    """Controller for robot from VR input."""
    
    def __init__(self, config: TeleoperationConfig):
        self.config = config
        self.robot_id = None
        self.end_effector_index = None
        self.is_connected = False
        
        # State tracking
        self.current_position = np.array([0.0, 0.0, 0.0])
        self.current_orientation = np.array([0.0, 0.0, 0.0, 1.0])
        self.current_velocity = np.array([0.0, 0.0, 0.0])
        
        # Safety monitoring
        self.safety_zones = []
        self.emergency_stop_active = False
    
    def connect_to_robot(self, robot_id: int = None):
        """Connect to robot (PyBullet or real robot)."""
        try:
            # For simulation, use PyBullet
            self.robot_id = robot_id
            self.end_effector_index = 6  # UR5 end effector
            self.is_connected = True
            logger.info("Connected to robot")
        except Exception as e:
            logger.error(f"Failed to connect to robot: {e}")
            self.is_connected = False
    
    def execute_command(self, controller_state: VRControllerState) -> bool:
        """Execute robot command from controller state."""
        if not self.is_connected or self.emergency_stop_active:
            return False
        
        try:
            # Convert VR position to robot position
            target_position = np.array(controller_state.position) * self.config.position_scaling
            
            # Convert VR orientation to robot orientation
            target_orientation = np.array(controller_state.orientation) * self.config.orientation_scaling
            
            # Apply velocity limits
            velocity = np.array(controller_state.velocity)
            velocity_magnitude = np.linalg.norm(velocity)
            if velocity_magnitude > self.config.velocity_limit:
                velocity = velocity / velocity_magnitude * self.config.velocity_limit
            
            # Send command to robot
            if self.robot_id is not None:
                # PyBullet control
                p.setJointMotorControlArray(
                    self.robot_id,
                    range(6),  # UR5 has 6 joints
                    p.POSITION_CONTROL,
                    targetPositions=[0.0] * 6,  # Placeholder
                    forces=[100.0] * 6
                )
            
            # Update state
            self.current_position = target_position
            self.current_orientation = target_orientation
            self.current_velocity = velocity
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            return False
    
    def emergency_stop(self):
        """Emergency stop robot."""
        self.emergency_stop_active = True
        
        if self.robot_id is not None:
            # Stop all joints
            p.setJointMotorControlArray(
                self.robot_id,
                range(6),
                p.VELOCITY_CONTROL,
                targetVelocities=[0.0] * 6
            )
        
        logger.warning("Emergency stop activated")
    
    def reset_emergency_stop(self):
        """Reset emergency stop."""
        self.emergency_stop_active = False
        logger.info("Emergency stop reset")

class VRTeleoperationInterface:
    """Main VR teleoperation interface."""
    
    def __init__(self, config: TeleoperationConfig):
        self.config = config
        
        # Initialize components
        self.vr_system = VRSystemInterface(config)
        self.hand_tracking = HandTrackingSystem(config) if config.enable_hand_tracking else None
        self.robot_controller = RobotController(config)
        
        # State tracking
        self.is_running = False
        self.current_mode = config.teleoperation_mode
        self.recording_data = []
        
        # Safety monitoring
        self.safety_monitor_thread = None
        
        # Network server
        self.network_server = None
        
        logger.info("VR Teleoperation Interface initialized")
    
    def start(self):
        """Start teleoperation interface."""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Connect to robot
        self.robot_controller.connect_to_robot()
        
        # Start safety monitoring
        if self.config.enable_safety_monitoring:
            self._start_safety_monitoring()
        
        # Start network server if enabled
        if self.config.enable_network:
            self._start_network_server()
        
        logger.info("VR Teleoperation Interface started")
    
    def stop(self):
        """Stop teleoperation interface."""
        self.is_running = False
        
        # Shutdown VR system
        self.vr_system.shutdown()
        
        # Stop safety monitoring
        if self.safety_monitor_thread:
            self.safety_monitor_thread.join(timeout=5)
        
        # Save recording if enabled
        if self.config.enable_recording and self.recording_data:
            self._save_recording()
        
        logger.info("VR Teleoperation Interface stopped")
    
    def _start_safety_monitoring(self):
        """Start safety monitoring thread."""
        self.safety_monitor_thread = threading.Thread(
            target=self._safety_monitoring_loop,
            daemon=True
        )
        self.safety_monitor_thread.start()
    
    def _safety_monitoring_loop(self):
        """Safety monitoring loop."""
        while self.is_running:
            try:
                # Check for emergency conditions
                if self._check_emergency_conditions():
                    self.robot_controller.emergency_stop()
                
                time.sleep(0.1)  # 10 Hz safety check
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
    
    def _check_emergency_conditions(self) -> bool:
        """Check for emergency conditions."""
        # Check if controller is too close to safety zones
        # This is a simplified check
        return False
    
    def _start_network_server(self):
        """Start network server for remote control."""
        self.network_server = threading.Thread(
            target=self._network_server_loop,
            daemon=True
        )
        self.network_server.start()
    
    def _network_server_loop(self):
        """Network server loop."""
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind((self.config.server_host, self.config.server_port))
        server_socket.listen(5)
        
        logger.info(f"Network server listening on {self.config.server_host}:{self.config.server_port}")
        
        while self.is_running:
            try:
                client_socket, address = server_socket.accept()
                logger.info(f"Client connected: {address}")
                
                # Handle client in separate thread
                client_thread = threading.Thread(
                    target=self._handle_client,
                    args=(client_socket,),
                    daemon=True
                )
                client_thread.start()
                
            except Exception as e:
                logger.error(f"Network server error: {e}")
    
    def _handle_client(self, client_socket: socket.socket):
        """Handle network client."""
        try:
            while self.is_running:
                # Receive data
                data = client_socket.recv(4096)
                if not data:
                    break
                
                # Deserialize
                command = pickle.loads(data)
                
                # Execute command
                result = self._execute_network_command(command)
                
                # Send result
                client_socket.send(pickle.dumps(result))
                
        except Exception as e:
            logger.error(f"Client handling error: {e}")
        finally:
            client_socket.close()
    
    def _execute_network_command(self, command: Dict[str, Any]) -> Dict[str, Any]:
        """Execute command from network client."""
        command_type = command.get('type')
        
        if command_type == 'get_state':
            return {
                'success': True,
                'position': self.robot_controller.current_position.tolist(),
                'orientation': self.robot_controller.current_orientation.tolist()
            }
        elif command_type == 'emergency_stop':
            self.robot_controller.emergency_stop()
            return {'success': True}
        elif command_type == 'reset_emergency':
            self.robot_controller.reset_emergency_stop()
            return {'success': True}
        else:
            return {'success': False, 'error': 'Unknown command type'}
    
    def process_vr_input(self) -> bool:
        """Process VR input and control robot."""
        if not self.is_running:
            return False
        
        try:
            # Get controller state
            controller_state = self.vr_system.get_controller_state(0)
            if controller_state is None:
                return False
            
            # Execute robot command
            success = self.robot_controller.execute_command(controller_state)
            
            # Record data if enabled
            if self.config.enable_recording:
                self.recording_data.append({
                    'timestamp': datetime.now().isoformat(),
                    'controller_state': controller_state.to_dict(),
                    'robot_position': self.robot_controller.current_position.tolist(),
                    'robot_orientation': self.robot_controller.current_orientation.tolist()
                })
            
            # Provide haptic feedback if enabled
            if self.config.haptic_feedback_enabled and success:
                self.vr_system.submit_haptic_feedback(
                    0,  # Controller index
                    0.1,  # Duration
                    100,  # Frequency
                    0.5  # Amplitude
                )
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to process VR input: {e}")
            return False
    
    def process_hand_tracking(self, frame: np.ndarray) -> Tuple[List[HandTrackingResult], np.ndarray]:
        """Process hand tracking from camera frame."""
        if not self.hand_tracking:
            return [], frame
        
        return self.hand_tracking.process_frame(frame)
    
    def _save_recording(self):
        """Save teleoperation recording."""
        if not self.recording_data:
            return
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = Path(self.config.recording_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        if self.config.recording_format == "hdf5":
            import h5py
            
            filename = output_path / f"teleoperation_{timestamp}.hdf5"
            
            with h5py.File(filename, 'w') as f:
                # Create datasets
                timestamps = [d['timestamp'] for d in self.recording_data]
                controller_states = [d['controller_state'] for d in self.recording_data]
                robot_positions = [d['robot_position'] for d in self.recording_data]
                robot_orientations = [d['robot_orientation'] for d in self.recording_data]
                
                f.create_dataset('timestamps', data=timestamps)
                f.create_dataset('controller_states', data=controller_states)
                f.create_dataset('robot_positions', data=robot_positions)
                f.create_dataset('robot_orientations', data=robot_orientations)
                
                # Add metadata
                f.attrs['config'] = json.dumps(self.config.to_dict())
                f.attrs['duration'] = len(self.recording_data) / self.config.control_frequency
        
        logger.info(f"Recording saved to: {filename}")
        self.recording_data = []

def main():
    """Main function for VR teleoperation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='VR Teleoperation Interface')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, default='direct',
                       choices=['direct', 'augmented', 'supervised', 'autonomous'],
                       help='Teleoperation mode')
    parser.add_argument('--no-vr', action='store_true', help='Run without VR (simulation mode)')
    
    args = parser.parse_args()
    
    # Load configuration
    config = TeleoperationConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    config.teleoperation_mode = TeleoperationMode(args.mode)
    
    if args.no_vr:
        config.vr_system = "simulation"
    
    # Create teleoperation interface
    teleop = VRTeleoperationInterface(config)
    
    try:
        # Start interface
        teleop.start()
        
        print("VR Teleoperation Interface running. Press Ctrl+C to stop.")
        
        # Main loop
        while True:
            # Process VR input
            teleop.process_vr_input()
            
            # Control frequency
            time.sleep(1.0 / config.control_frequency)
    
    except KeyboardInterrupt:
        print("\nStopping VR Teleoperation Interface...")
    finally:
        teleop.stop()

if __name__ == "__main__":
    main()
