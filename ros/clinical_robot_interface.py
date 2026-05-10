#!/usr/bin/env python3
"""
ROS Interface for Clinical Robot Adaptation
Comprehensive ROS integration for clinical robot control and adaptation.

This module implements:
- ROS node for clinical robot control
- Real-time inference integration with PyTorch models
- Safety monitoring and emergency stop
- Human-robot interaction monitoring
- Multi-camera perception integration
- Clinical workflow management
- Real-time performance monitoring

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import warnings

# ROS imports
import rospy
import rospkg
import tf
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import Pose, PoseStamped, Twist, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo, JointState, PointCloud2
from std_msgs.msg import Header, String, Bool, Float32
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from control_msgs.msg import FollowJointTrajectoryAction, FollowJointTrajectoryGoal
from actionlib_msgs.msg import GoalStatus
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from nav_msgs.msg import Odometry, Path as NavPath

# OpenCV and computer vision
import cv2
import cv_bridge
from cv_bridge import CvBridge, CvBridgeError

# PyTorch and ML
import torch
import numpy as np

# Project imports
sys.path.append(str(Path(__file__).parent.parent))
from pipelines.adaptation_pipeline import AdaptationPipeline, AdaptationConfig
from models.model_interpretability import ModelInterpretabilityPipeline

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class ClinicalRobotNode:
    """Main ROS node for clinical robot control."""
    
    def __init__(self, config_path: str = None):
        # Initialize ROS node
        rospy.init_node('clinical_robot_node', anonymous=True)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.bridge = CvBridge()
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        # Initialize ML models
        self.adaptation_pipeline = None
        self.interpretability_pipeline = None
        self._load_models()
        
        # Robot state
        self.current_pose = Pose()
        self.current_joint_states = JointState()
        self.robot_ready = False
        
        # Safety monitoring
        self.safety_monitor = SafetyMonitor(self.config)
        self.emergency_stop_active = False
        
        # Human monitoring
        self.human_monitor = HumanMonitor(self.config)
        
        # Perception
        self.perception_manager = PerceptionManager(self.config)
        
        # Data buffers
        self.image_buffer = deque(maxlen=10)
        self.joint_state_buffer = deque(maxlen=50)
        self.instruction_buffer = deque(maxlen=5)
        
        # Performance monitoring
        self.performance_monitor = PerformanceMonitor()
        
        # Initialize publishers and subscribers
        self._init_publishers()
        self._init_subscribers()
        self._init_action_clients()
        
        # Initialize services
        self._init_services()
        
        # Start monitoring threads
        self._start_monitoring_threads()
        
        logger.info("Clinical Robot Node initialized successfully")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from file."""
        default_config = {
            'robot_name': 'clinical_robot',
            'model_path': './models/best_model.pth',
            'config_path': './configs/adaptation_config.json',
            'camera_topics': [
                '/camera_front/image_raw',
                '/camera_side/image_raw',
                '/camera_overhead/image_raw'
            ],
            'joint_state_topic': '/joint_states',
            'pose_topic': '/robot_pose',
            'trajectory_topic': '/joint_trajectory_controller/command',
            'safety_topics': {
                'emergency_stop': '/emergency_stop',
                'safety_status': '/safety_status'
            },
            'inference_frequency': 10.0,  # Hz
            'control_frequency': 100.0,   # Hz
            'monitoring_frequency': 1.0,  # Hz
            'safety_thresholds': {
                'human_distance': 0.5,      # meters
                'max_velocity': 0.3,        # m/s
                'max_force': 50.0,          # N
                'collision_risk': 0.7
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
            default_config.update(user_config)
        
        return default_config
    
    def _load_models(self):
        """Load ML models."""
        try:
            # Load adaptation pipeline config
            config_path = self.config.get('config_path')
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config_dict = json.load(f)
                config = AdaptationConfig(**config_dict)
                
                # Initialize adaptation pipeline
                self.adaptation_pipeline = AdaptationPipeline(config)
                
                # Load trained model
                model_path = self.config.get('model_path')
                if Path(model_path).exists():
                    self.adaptation_pipeline.load_checkpoint(model_path)
                    logger.info("Adaptation model loaded successfully")
                
                # Initialize interpretability pipeline
                self.interpretability_pipeline = ModelInterpretabilityPipeline(
                    self.adaptation_pipeline.model, config
                )
                
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.adaptation_pipeline = None
            self.interpretability_pipeline = None
    
    def _init_publishers(self):
        """Initialize ROS publishers."""
        # Trajectory publisher
        self.trajectory_pub = rospy.Publisher(
            self.config['trajectory_topic'],
            JointTrajectory,
            queue_size=1
        )
        
        # Status publisher
        self.status_pub = rospy.Publisher(
            '/robot_status',
            String,
            queue_size=1
        )
        
        # Safety publisher
        self.safety_pub = rospy.Publisher(
            self.config['safety_topics']['safety_status'],
            String,
            queue_size=1
        )
        
        # Performance publisher
        self.performance_pub = rospy.Publisher(
            '/performance_metrics',
            String,
            queue_size=1
        )
    
    def _init_subscribers(self):
        """Initialize ROS subscribers."""
        # Camera subscribers
        self.camera_subs = []
        for camera_topic in self.config['camera_topics']:
            sub = rospy.Subscriber(
                camera_topic,
                Image,
                self._camera_callback,
                callback_args=camera_topic
            )
            self.camera_subs.append(sub)
        
        # Joint state subscriber
        self.joint_state_sub = rospy.Subscriber(
            self.config['joint_state_topic'],
            JointState,
            self._joint_state_callback
        )
        
        # Pose subscriber
        self.pose_sub = rospy.Subscriber(
            self.config['pose_topic'],
            PoseStamped,
            self._pose_callback
        )
        
        # Instruction subscriber
        self.instruction_sub = rospy.Subscriber(
            '/clinical_instruction',
            String,
            self._instruction_callback
        )
        
        # Emergency stop subscriber
        self.emergency_stop_sub = rospy.Subscriber(
            self.config['safety_topics']['emergency_stop'],
            Bool,
            self._emergency_stop_callback
        )
    
    def _init_action_clients(self):
        """Initialize action clients."""
        # Trajectory following client
        self.trajectory_client = actionlib.SimpleActionClient(
            '/joint_trajectory_controller/follow_joint_trajectory',
            FollowJointTrajectoryAction
        )
        
        # Navigation client (if mobile robot)
        self.nav_client = actionlib.SimpleActionClient(
            '/move_base',
            MoveBaseAction
        )
    
    def _init_services(self):
        """Initialize ROS services."""
        # Inference service
        self.inference_service = rospy.Service(
            '/clinical_inference',
            ClinicalInference,
            self._inference_service_handler
        )
        
        # Safety check service
        self.safety_service = rospy.Service(
            '/safety_check',
            SafetyCheck,
            self._safety_service_handler
        )
        
        # Model explanation service
        self.explanation_service = rospy.Service(
            '/model_explanation',
            ModelExplanation,
            self._explanation_service_handler
        )
    
    def _start_monitoring_threads(self):
        """Start monitoring threads."""
        # Inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            daemon=True
        )
        self.inference_thread.start()
        
        # Safety monitoring thread
        self.safety_thread = threading.Thread(
            target=self._safety_monitoring_loop,
            daemon=True
        )
        self.safety_thread.start()
        
        # Performance monitoring thread
        self.performance_thread = threading.Thread(
            target=self._performance_monitoring_loop,
            daemon=True
        )
        self.performance_thread.start()
    
    def _camera_callback(self, msg: Image, camera_topic: str):
        """Handle camera image callback."""
        try:
            # Convert ROS image to OpenCV
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Store in buffer with timestamp
            self.image_buffer.append({
                'image': cv_image,
                'timestamp': msg.header.stamp,
                'camera_topic': camera_topic
            })
            
        except CvBridgeError as e:
            logger.error(f"CV Bridge error: {e}")
    
    def _joint_state_callback(self, msg: JointState):
        """Handle joint state callback."""
        self.current_joint_states = msg
        self.joint_state_buffer.append({
            'states': msg,
            'timestamp': rospy.Time.now()
        })
    
    def _pose_callback(self, msg: PoseStamped):
        """Handle pose callback."""
        self.current_pose = msg.pose
    
    def _instruction_callback(self, msg: String):
        """Handle clinical instruction callback."""
        self.instruction_buffer.append({
            'instruction': msg.data,
            'timestamp': rospy.Time.now()
        })
        
        logger.info(f"Received instruction: {msg.data}")
    
    def _emergency_stop_callback(self, msg: Bool):
        """Handle emergency stop callback."""
        self.emergency_stop_active = msg.data
        
        if msg.data:
            logger.warning("Emergency stop activated!")
            self._execute_emergency_stop()
        else:
            logger.info("Emergency stop deactivated")
    
    def _inference_loop(self):
        """Main inference loop."""
        rate = rospy.Rate(self.config['inference_frequency'])
        
        while not rospy.is_shutdown():
            try:
                # Check if we have data for inference
                if (len(self.image_buffer) > 0 and 
                    len(self.joint_state_buffer) > 0 and 
                    len(self.instruction_buffer) > 0):
                    
                    # Get latest data
                    latest_image = self.image_buffer[-1]
                    latest_joints = self.joint_state_buffer[-1]
                    latest_instruction = self.instruction_buffer[-1]
                    
                    # Run inference
                    if self.adaptation_pipeline:
                        prediction = self._run_inference(
                            latest_image['image'],
                            latest_joints['states'],
                            latest_instruction['instruction']
                        )
                        
                        # Process prediction
                        self._process_prediction(prediction)
                
                rate.sleep()
                
            except Exception as e:
                logger.error(f"Inference loop error: {e}")
    
    def _run_inference(self, image: np.ndarray, joint_states: JointState, 
                     instruction: str) -> Dict[str, Any]:
        """Run model inference."""
        try:
            # Preprocess image
            image_tensor = self._preprocess_image(image)
            
            # Preprocess joint states
            state_tensor = self._preprocess_joint_states(joint_states)
            
            # Run prediction
            with torch.no_grad():
                prediction = self.adaptation_pipeline.predict(
                    image_tensor, state_tensor, instruction
                )
            
            return {
                'actions': prediction['actions'].numpy(),
                'safety_score': prediction['safety_score'].item(),
                'human_aware_score': prediction['human_aware_score'].item(),
                'timestamp': rospy.Time.now()
            }
            
        except Exception as e:
            logger.error(f"Inference error: {e}")
            return None
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model."""
        # Resize and normalize
        image_resized = cv2.resize(image, (224, 224))
        image_normalized = image_resized.astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_normalized).permute(2, 0, 1)
        
        # Apply ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406])
        std = torch.tensor([0.229, 0.224, 0.225])
        image_tensor = (image_tensor - mean[:, None, None]) / std[:, None, None]
        
        return image_tensor
    
    def _preprocess_joint_states(self, joint_states: JointState) -> torch.Tensor:
        """Preprocess joint states for model."""
        # Convert to numpy and ensure correct format
        if len(joint_states.position) > 0:
            positions = np.array(joint_states.position)
        else:
            positions = np.zeros(7)  # Default for 7-DOF arm
        
        if len(joint_states.velocity) > 0:
            velocities = np.array(joint_states.velocity)
        else:
            velocities = np.zeros(7)
        
        if len(joint_states.effort) > 0:
            efforts = np.array(joint_states.effort)
        else:
            efforts = np.zeros(7)
        
        # Concatenate all state information
        state_vector = np.concatenate([positions, velocities, efforts])
        
        # Ensure correct size
        if len(state_vector) < 14:
            state_vector = np.pad(state_vector, (0, 14 - len(state_vector)))
        elif len(state_vector) > 14:
            state_vector = state_vector[:14]
        
        return torch.from_numpy(state_vector).float()
    
    def _process_prediction(self, prediction: Dict[str, Any]):
        """Process model prediction."""
        if prediction is None:
            return
        
        # Check safety
        safety_ok = self._check_prediction_safety(prediction)
        
        if not safety_ok:
            logger.warning("Prediction failed safety check")
            return
        
        # Convert to joint trajectory
        trajectory = self._prediction_to_trajectory(prediction['actions'])
        
        # Execute trajectory
        if not self.emergency_stop_active:
            self._execute_trajectory(trajectory)
    
    def _check_prediction_safety(self, prediction: Dict[str, Any]) -> bool:
        """Check if prediction is safe to execute."""
        # Check safety score
        if prediction['safety_score'] < self.config['safety_thresholds']['collision_risk']:
            return False
        
        # Check human-aware score
        if (prediction['human_aware_score'] < 0.5 and 
            self.human_monitor.get_human_distance() < self.config['safety_thresholds']['human_distance']):
            return False
        
        # Check velocity limits
        actions = prediction['actions']
        if np.any(np.abs(actions) > self.config['safety_thresholds']['max_velocity']):
            return False
        
        return True
    
    def _prediction_to_trajectory(self, actions: np.ndarray) -> JointTrajectory:
        """Convert prediction to joint trajectory."""
        trajectory = JointTrajectory()
        trajectory.header.stamp = rospy.Time.now()
        trajectory.header.frame_id = "base_link"
        
        # Joint names (example for 7-DOF arm)
        trajectory.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", 
            "joint_5", "joint_6", "joint_7"
        ]
        
        # Create trajectory point
        point = JointTrajectoryPoint()
        point.positions = actions.tolist()
        point.velocities = [0.0] * len(actions)
        point.accelerations = [0.0] * len(actions)
        point.time_from_start = rospy.Duration(1.0)  # 1 second to complete
        
        trajectory.points.append(point)
        
        return trajectory
    
    def _execute_trajectory(self, trajectory: JointTrajectory):
        """Execute joint trajectory."""
        try:
            # Send trajectory directly
            self.trajectory_pub.publish(trajectory)
            
            # Or use action client for better feedback
            if self.trajectory_client.wait_for_server(timeout=rospy.Duration(1.0)):
                goal = FollowJointTrajectoryGoal()
                goal.trajectory = trajectory
                self.trajectory_client.send_goal(goal)
            
        except Exception as e:
            logger.error(f"Trajectory execution error: {e}")
    
    def _execute_emergency_stop(self):
        """Execute emergency stop."""
        # Send zero velocity command
        stop_trajectory = JointTrajectory()
        stop_trajectory.header.stamp = rospy.Time.now()
        stop_trajectory.joint_names = [
            "joint_1", "joint_2", "joint_3", "joint_4", 
            "joint_5", "joint_6", "joint_7"
        ]
        
        stop_point = JointTrajectoryPoint()
        stop_point.positions = self.current_joint_states.position
        stop_point.velocities = [0.0] * len(self.current_joint_states.position)
        stop_point.time_from_start = rospy.Duration(0.1)
        
        stop_trajectory.points.append(stop_point)
        
        self.trajectory_pub.publish(stop_trajectory)
    
    def _safety_monitoring_loop(self):
        """Safety monitoring loop."""
        rate = rospy.Rate(self.config['monitoring_frequency'])
        
        while not rospy.is_shutdown():
            try:
                # Update safety status
                safety_status = self.safety_monitor.update_safety_status(
                    self.current_pose,
                    self.current_joint_states
                )
                
                # Publish safety status
                status_msg = String()
                status_msg.data = json.dumps(safety_status)
                self.safety_pub.publish(status_msg)
                
                # Check for safety violations
                if safety_status.get('violation', False):
                    logger.warning(f"Safety violation: {safety_status['violation_type']}")
                    self.emergency_stop_active = True
                    self._execute_emergency_stop()
                
                rate.sleep()
                
            except Exception as e:
                logger.error(f"Safety monitoring error: {e}")
    
    def _performance_monitoring_loop(self):
        """Performance monitoring loop."""
        rate = rospy.Rate(self.config['monitoring_frequency'])
        
        while not rospy.is_shutdown():
            try:
                # Update performance metrics
                metrics = self.performance_monitor.update_metrics()
                
                # Publish metrics
                metrics_msg = String()
                metrics_msg.data = json.dumps(metrics)
                self.performance_pub.publish(metrics_msg)
                
                rate.sleep()
                
            except Exception as e:
                logger.error(f"Performance monitoring error: {e}")
    
    def _inference_service_handler(self, req):
        """Handle inference service requests."""
        try:
            # Get latest data
            if len(self.image_buffer) == 0 or len(self.joint_state_buffer) == 0:
                return ClinicalInferenceResponse(
                    success=False,
                    message="No data available for inference"
                )
            
            latest_image = self.image_buffer[-1]
            latest_joints = self.joint_state_buffer[-1]
            
            # Run inference
            prediction = self._run_inference(
                latest_image['image'],
                latest_joints['states'],
                req.instruction
            )
            
            if prediction is None:
                return ClinicalInferenceResponse(
                    success=False,
                    message="Inference failed"
                )
            
            # Return response
            return ClinicalInferenceResponse(
                success=True,
                actions=prediction['actions'].tolist(),
                safety_score=prediction['safety_score'],
                human_aware_score=prediction['human_aware_score'],
                message="Inference successful"
            )
            
        except Exception as e:
            logger.error(f"Inference service error: {e}")
            return ClinicalInferenceResponse(
                success=False,
                message=f"Service error: {str(e)}"
            )
    
    def _safety_service_handler(self, req):
        """Handle safety check service requests."""
        try:
            # Get current safety status
            safety_status = self.safety_monitor.get_current_status()
            
            return SafetyCheckResponse(
                safe=safety_status['safe'],
                human_distance=safety_status['human_distance'],
                collision_risk=safety_status['collision_risk'],
                violations=safety_status['violations']
            )
            
        except Exception as e:
            logger.error(f"Safety service error: {e}")
            return SafetyCheckResponse(
                safe=False,
                human_distance=0.0,
                collision_risk=1.0,
                violations=[f"Service error: {str(e)}"]
            )
    
    def _explanation_service_handler(self, req):
        """Handle model explanation service requests."""
        try:
            if self.interpretability_pipeline is None:
                return ModelExplanationResponse(
                    success=False,
                    message="Interpretability pipeline not available"
                )
            
            # Get latest data
            if len(self.image_buffer) == 0 or len(self.joint_state_buffer) == 0:
                return ModelExplanationResponse(
                    success=False,
                    message="No data available for explanation"
                )
            
            latest_image = self.image_buffer[-1]
            latest_joints = self.joint_state_buffer[-1]
            
            # Create input data
            image_tensor = self._preprocess_image(latest_image['image'])
            state_tensor = self._preprocess_joint_states(latest_joints['states'])
            
            input_data = {
                'image': image_tensor.unsqueeze(0),
                'robot_state': state_tensor.unsqueeze(0),
                'medication_type': 0,  # Default
                'human_distance': torch.tensor([self.human_monitor.get_human_distance()])
            }
            
            # Generate explanation
            explanation = self.interpretability_pipeline.explain_single_instance(input_data)
            
            return ModelExplanationResponse(
                success=True,
                explanation=json.dumps(explanation['clinical_explanation']),
                attention_visualization=json.dumps({}),
                message="Explanation generated successfully"
            )
            
        except Exception as e:
            logger.error(f"Explanation service error: {e}")
            return ModelExplanationResponse(
                success=False,
                message=f"Service error: {str(e)}"
            )
    
    def spin(self):
        """Main spin loop."""
        try:
            rospy.spin()
        except KeyboardInterrupt:
            logger.info("Shutting down clinical robot node")
        finally:
            self._cleanup()
    
    def _cleanup(self):
        """Cleanup resources."""
        logger.info("Cleaning up resources")

# Supporting classes

class SafetyMonitor:
    """Safety monitoring for clinical robot."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.current_status = {
            'safe': True,
            'human_distance': 10.0,
            'collision_risk': 0.0,
            'violations': []
        }
    
    def update_safety_status(self, pose: Pose, joint_states: JointState) -> Dict[str, Any]:
        """Update safety status."""
        violations = []
        
        # Check human distance (simplified)
        human_distance = self._estimate_human_distance()
        if human_distance < self.config['safety_thresholds']['human_distance']:
            violations.append("Human too close")
        
        # Check joint velocities
        if len(joint_states.velocity) > 0:
            max_velocity = max(abs(v) for v in joint_states.velocity)
            if max_velocity > self.config['safety_thresholds']['max_velocity']:
                violations.append("Joint velocity too high")
        
        # Check collision risk (simplified)
        collision_risk = self._estimate_collision_risk(pose)
        if collision_risk > self.config['safety_thresholds']['collision_risk']:
            violations.append("High collision risk")
        
        self.current_status = {
            'safe': len(violations) == 0,
            'human_distance': human_distance,
            'collision_risk': collision_risk,
            'violations': violations
        }
        
        return self.current_status
    
    def get_current_status(self) -> Dict[str, Any]:
        """Get current safety status."""
        return self.current_status
    
    def _estimate_human_distance(self) -> float:
        """Estimate human distance (simplified)."""
        # In real implementation, this would use human detection
        return 2.0  # Default safe distance
    
    def _estimate_collision_risk(self, pose: Pose) -> float:
        """Estimate collision risk (simplified)."""
        # In real implementation, this would use environment model
        return 0.1  # Default low risk

class HumanMonitor:
    """Human monitoring for clinical robot."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.human_positions = deque(maxlen=10)
    
    def get_human_distance(self) -> float:
        """Get minimum human distance."""
        if len(self.human_positions) == 0:
            return 10.0  # Default far distance
        
        # Return minimum distance from recent detections
        return min(pos['distance'] for pos in self.human_positions)

class PerceptionManager:
    """Perception management for clinical robot."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
    
    def process_images(self, images: List[np.ndarray]) -> Dict[str, Any]:
        """Process camera images."""
        # In real implementation, this would include object detection, etc.
        return {
            'medications_detected': [],
            'humans_detected': [],
            'obstacles_detected': []
        }

class PerformanceMonitor:
    """Performance monitoring for clinical robot."""
    
    def __init__(self):
        self.start_time = time.time()
        self.inference_times = deque(maxlen=100)
        self.cpu_usage = deque(maxlen=100)
        self.memory_usage = deque(maxlen=100)
    
    def update_metrics(self) -> Dict[str, Any]:
        """Update performance metrics."""
        # System metrics
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent
        
        self.cpu_usage.append(cpu_percent)
        self.memory_usage.append(memory_percent)
        
        # Calculate averages
        avg_cpu = np.mean(self.cpu_usage) if self.cpu_usage else 0
        avg_memory = np.mean(self.memory_usage) if self.memory_usage else 0
        
        return {
            'uptime': time.time() - self.start_time,
            'cpu_usage': avg_cpu,
            'memory_usage': avg_memory,
            'inference_times': list(self.inference_times)
        }

# ROS Service definitions
from clinical_robot_msgs.srv import ClinicalInference, ClinicalInferenceResponse
from clinical_robot_msgs.srv import SafetyCheck, SafetyCheckResponse
from clinical_robot_msgs.srv import ModelExplanation, ModelExplanationResponse

def main():
    """Main function."""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='Clinical Robot ROS Node')
        parser.add_argument('--config', type=str, help='Configuration file path')
        args = parser.parse_args()
        
        # Create and run node
        node = ClinicalRobotNode(args.config)
        node.spin()
        
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        logger.error(f"Node error: {e}")

if __name__ == "__main__":
    main()
