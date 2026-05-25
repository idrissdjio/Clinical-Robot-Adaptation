#!/usr/bin/env python3
"""
Real-time Clinical Robot Simulation Environment with PyBullet
Comprehensive physics simulation for clinical robot testing and development.

This module implements:
- Real-time physics simulation with PyBullet
- Clinical environment modeling (pharmacy, hospital rooms)
- Medication object simulation and grasping
- Human simulation and interaction
- Multi-camera vision simulation
- Force/torque sensing and feedback
- Real-time control and monitoring

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

# PyBullet for physics simulation
import pybullet as p
import pybullet_data

# Scientific computing
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

# Computer vision
import cv2
from PIL import Image

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_simulator.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class SimulationMode(Enum):
    """Simulation modes."""
    GUI = "gui"
    DIRECT = "direct"
    SHARED_MEMORY = "shared_memory"

class ClinicalEnvironment(Enum):
    """Clinical environment types."""
    PHARMACY = "pharmacy"
    HOSPITAL_ROOM = "hospital_room"
    EMERGENCY_DEPARTMENT = "emergency_department"
    OPERATING_ROOM = "operating_room"
    INTENSIVE_CARE_UNIT = "intensive_care_unit"

@dataclass
class SimulationConfig:
    """Configuration for clinical simulation."""
    
    # Simulation settings
    mode: SimulationMode = SimulationMode.GUI
    time_step: float = 1.0 / 240.0  # 240 Hz
    gravity: float = -9.81
    solver_iterations: int = 10
    
    # Environment settings
    environment_type: ClinicalEnvironment = ClinicalEnvironment.PHARMACY
    workspace_size: Tuple[float, float, float] = (2.0, 2.0, 1.5)
    
    # Robot settings
    robot_model: str = "ur5"
    robot_position: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    robot_orientation: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    
    # Camera settings
    camera_positions: List[Dict[str, Any]] = field(default_factory=list)
    camera_resolution: Tuple[int, int] = (640, 480)
    camera_fov: float = 60.0
    
    # Medication object settings
    medication_types: List[str] = field(default_factory=lambda: ["vial", "bottle", "syringe", "blister_pack"])
    medication_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    
    # Human simulation
    enable_human: bool = True
    human_positions: List[Tuple[float, float, float]] = field(default_factory=list)
    human_behavior: str = "static"  # static, random, scripted
    
    # Physics settings
    enable_force_torque: bool = True
    enable_contact_detection: bool = True
    collision_margin: float = 0.001
    
    # Rendering
    enable_rendering: bool = True
    render_shadows: bool = True
    render_antialiasing: int = 4
    
    # Data recording
    record_data: bool = True
    record_interval: int = 10  # frames
    output_dir: str = "./simulation_data"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'time_step': self.time_step,
            'gravity': self.gravity,
            'solver_iterations': self.solver_iterations,
            'environment_type': self.environment_type.value,
            'workspace_size': self.workspace_size,
            'robot_model': self.robot_model,
            'robot_position': self.robot_position,
            'robot_orientation': self.robot_orientation,
            'camera_positions': self.camera_positions,
            'camera_resolution': self.camera_resolution,
            'camera_fov': self.camera_fov,
            'medication_types': self.medication_types,
            'medication_positions': self.medication_positions,
            'enable_human': self.enable_human,
            'human_positions': self.human_positions,
            'human_behavior': self.human_behavior,
            'enable_force_torque': self.enable_force_torque,
            'enable_contact_detection': self.enable_contact_detection,
            'collision_margin': self.collision_margin,
            'enable_rendering': self.enable_rendering,
            'render_shadows': self.render_shadows,
            'render_antialiasing': self.render_antialiasing,
            'record_data': self.record_data,
            'record_interval': self.record_interval,
            'output_dir': self.output_dir
        }

class MedicationObject:
    """Medication object in simulation."""
    
    def __init__(self, med_type: str, position: Tuple[float, float, float], 
                 orientation: Tuple[float, float, float, float] = (0, 0, 0, 1)):
        self.med_type = med_type
        self.position = position
        self.orientation = orientation
        self.body_id = None
        self.mass = self._get_mass(med_type)
        self.dimensions = self._get_dimensions(med_type)
    
    def _get_mass(self, med_type: str) -> float:
        """Get mass based on medication type."""
        masses = {
            'vial': 0.05,
            'bottle': 0.2,
            'syringe': 0.03,
            'blister_pack': 0.1
        }
        return masses.get(med_type, 0.1)
    
    def _get_dimensions(self, med_type: str) -> Tuple[float, float, float]:
        """Get dimensions based on medication type."""
        dimensions = {
            'vial': (0.02, 0.02, 0.08),
            'bottle': (0.06, 0.06, 0.15),
            'syringe': (0.015, 0.015, 0.12),
            'blister_pack': (0.1, 0.05, 0.01)
        }
        return dimensions.get(med_type, (0.05, 0.05, 0.1))

class HumanAgent:
    """Simulated human agent in clinical environment."""
    
    def __init__(self, position: Tuple[float, float, float], 
                 behavior: str = "static"):
        self.position = position
        self.behavior = behavior
        self.body_id = None
        self.velocity = np.array([0.0, 0.0, 0.0])
        self.target_position = np.array(position)
    
    def update(self, dt: float):
        """Update human position based on behavior."""
        if self.behavior == "random":
            # Random walk behavior
            if np.random.random() < 0.01:  # Change direction occasionally
                self.velocity = np.random.uniform(-0.1, 0.1, 3)
                self.velocity[2] = 0  # Keep on ground
            self.position += self.velocity * dt
        elif self.behavior == "scripted":
            # Move toward target
            direction = self.target_position - np.array(self.position)
            distance = np.linalg.norm(direction)
            if distance > 0.01:
                direction = direction / distance
                self.position += direction * 0.1 * dt

class ClinicalSimulator:
    """Main clinical simulation environment."""
    
    def __init__(self, config: SimulationConfig):
        self.config = config
        self.physics_client = None
        self.robot_id = None
        self.end_effector_index = None
        self.medications = []
        self.humans = []
        self.cameras = {}
        self.is_running = False
        
        # Data recording
        self.recorded_data = defaultdict(list)
        self.frame_count = 0
        
        # Initialize simulation
        self._initialize_simulation()
        
        logger.info("Clinical Simulator initialized")
    
    def _initialize_simulation(self):
        """Initialize PyBullet simulation."""
        try:
            # Connect to PyBullet
            if self.config.mode == SimulationMode.GUI:
                self.physics_client = p.connect(p.GUI)
            elif self.config.mode == SimulationMode.DIRECT:
                self.physics_client = p.connect(p.DIRECT)
            elif self.config.mode == SimulationMode.SHARED_MEMORY:
                self.physics_client = p.connect(p.SHARED_MEMORY)
            
            # Set simulation parameters
            p.setGravity(0, 0, self.config.gravity)
            p.setTimeStep(self.config.time_step)
            p.setPhysicsEngineParameter(
                numSolverIterations=self.config.solver_iterations,
                solverResidualThreshold=1e-7
            )
            
            # Add PyBullet data path
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
            
            # Load environment
            self._load_environment()
            
            # Load robot
            self._load_robot()
            
            # Load medications
            self._load_medications()
            
            # Load humans
            if self.config.enable_human:
                self._load_humans()
            
            # Setup cameras
            self._setup_cameras()
            
            # Setup rendering
            self._setup_rendering()
            
            logger.info("Simulation initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize simulation: {e}")
            raise
    
    def _load_environment(self):
        """Load clinical environment."""
        # Load plane
        plane_id = p.loadURDF("plane.urdf")
        
        # Load environment-specific objects
        if self.config.environment_type == ClinicalEnvironment.PHARMACY:
            self._load_pharmacy_environment()
        elif self.config.environment_type == ClinicalEnvironment.HOSPITAL_ROOM:
            self._load_hospital_room_environment()
        elif self.config.environment_type == ClinicalEnvironment.EMERGENCY_DEPARTMENT:
            self._load_emergency_department_environment()
        
        # Add workspace boundaries
        self._create_workspace_boundaries()
    
    def _load_pharmacy_environment(self):
        """Load pharmacy-specific environment."""
        # Add shelves
        shelf_positions = [
            (0.5, 0.3, 0.5),
            (0.5, -0.3, 0.5),
            (-0.5, 0.3, 0.5),
            (-0.5, -0.3, 0.5)
        ]
        
        for i, pos in enumerate(shelf_positions):
            self._create_shelf(pos, f"shelf_{i}")
        
        # Add counter
        self._create_counter((0.0, 0.8, 0.4))
    
    def _load_hospital_room_environment(self):
        """Load hospital room environment."""
        # Add bed
        self._create_bed((0.0, 0.0, 0.3))
        
        # Add bedside table
        self._create_table((0.5, 0.0, 0.4))
        
        # Add IV stand
        self._create_iv_stand((-0.5, 0.0, 0.0))
    
    def _load_emergency_department_environment(self):
        """Load emergency department environment."""
        # Add gurney
        self._create_gurney((0.0, 0.0, 0.4))
        
        # Add medical equipment cart
        self._create_medical_cart((0.5, 0.5, 0.3))
        
        # Add monitor stand
        self._create_monitor_stand((-0.5, 0.5, 0.0))
    
    def _create_shelf(self, position: Tuple[float, float, float], name: str):
        """Create a shelf object."""
        # Create shelf as a box
        shelf_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.05, 0.4],
            rgbaColor=[0.6, 0.4, 0.2, 1.0]
        )
        shelf_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.05, 0.4]
        )
        
        shelf_id = p.createMultiBody(
            baseMass=0.0,  # Static
            baseCollisionShapeIndex=shelf_collision,
            baseVisualShapeIndex=shelf_visual,
            basePosition=position
        )
        
        p.changeDynamics(shelf_id, -1, lateralFriction=0.5)
        return shelf_id
    
    def _create_counter(self, position: Tuple[float, float, float]):
        """Create a counter object."""
        counter_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.8, 0.1, 0.4],
            rgbaColor=[0.8, 0.7, 0.6, 1.0]
        )
        counter_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.8, 0.1, 0.4]
        )
        
        counter_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=counter_collision,
            baseVisualShapeIndex=counter_visual,
            basePosition=position
        )
        
        return counter_id
    
    def _create_bed(self, position: Tuple[float, float, float]):
        """Create a hospital bed."""
        bed_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.9, 0.2],
            rgbaColor=[0.9, 0.9, 0.9, 1.0]
        )
        bed_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.5, 0.9, 0.2]
        )
        
        bed_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=bed_collision,
            baseVisualShapeIndex=bed_visual,
            basePosition=position
        )
        
        return bed_id
    
    def _create_table(self, position: Tuple[float, float, float]):
        """Create a bedside table."""
        table_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3],
            rgbaColor=[0.7, 0.5, 0.3, 1.0]
        )
        table_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.3]
        )
        
        table_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=position
        )
        
        return table_id
    
    def _create_iv_stand(self, position: Tuple[float, float, float]):
        """Create an IV stand."""
        # Pole
        pole_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.02,
            length=1.5,
            rgbaColor=[0.8, 0.8, 0.8, 1.0]
        )
        pole_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.02,
            height=1.5
        )
        
        pole_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=pole_collision,
            baseVisualShapeIndex=pole_visual,
            basePosition=position
        )
        
        return pole_id
    
    def _create_gurney(self, position: Tuple[float, float, float]):
        """Create a gurney."""
        gurney_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.6, 1.0, 0.25],
            rgbaColor=[0.7, 0.7, 0.9, 1.0]
        )
        gurney_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.6, 1.0, 0.25]
        )
        
        gurney_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=gurney_collision,
            baseVisualShapeIndex=gurney_visual,
            basePosition=position
        )
        
        return gurney_id
    
    def _create_medical_cart(self, position: Tuple[float, float, float]):
        """Create a medical equipment cart."""
        cart_visual = p.createVisualShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.4],
            rgbaColor=[0.6, 0.6, 0.8, 1.0]
        )
        cart_collision = p.createCollisionShape(
            shapeType=p.GEOM_BOX,
            halfExtents=[0.3, 0.3, 0.4]
        )
        
        cart_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=cart_collision,
            baseVisualShapeIndex=cart_visual,
            basePosition=position
        )
        
        return cart_id
    
    def _create_monitor_stand(self, position: Tuple[float, float, float]):
        """Create a monitor stand."""
        stand_visual = p.createVisualShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.03,
            length=1.2,
            rgbaColor=[0.3, 0.3, 0.3, 1.0]
        )
        stand_collision = p.createCollisionShape(
            shapeType=p.GEOM_CYLINDER,
            radius=0.03,
            height=1.2
        )
        
        stand_id = p.createMultiBody(
            baseMass=0.0,
            baseCollisionShapeIndex=stand_collision,
            baseVisualShapeIndex=stand_visual,
            basePosition=position
        )
        
        return stand_id
    
    def _create_workspace_boundaries(self):
        """Create invisible workspace boundaries."""
        # Create walls around workspace
        ws_x, ws_y, ws_z = self.config.workspace_size
        
        wall_positions = [
            (ws_x/2, 0, ws_z/2),  # Right wall
            (-ws_x/2, 0, ws_z/2),  # Left wall
            (0, ws_y/2, ws_z/2),  # Front wall
            (0, -ws_y/2, ws_z/2)   # Back wall
        ]
        
        wall_orientations = [
            (0, 0, 0, 1),
            (0, 0, 0, 1),
            (0, 0, np.sqrt(2)/2, np.sqrt(2)/2),
            (0, 0, np.sqrt(2)/2, np.sqrt(2)/2)
        ]
        
        wall_sizes = [
            (0.05, ws_y, ws_z),
            (0.05, ws_y, ws_z),
            (ws_x, 0.05, ws_z),
            (ws_x, 0.05, ws_z)
        ]
        
        for i, (pos, orn, size) in enumerate(zip(wall_positions, wall_orientations, wall_sizes)):
            wall_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=size,
                rgbaColor=[0.8, 0.8, 0.8, 0.3]  # Semi-transparent
            )
            wall_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=size
            )
            
            wall_id = p.createMultiBody(
                baseMass=0.0,
                baseCollisionShapeIndex=wall_collision,
                baseVisualShapeIndex=wall_visual,
                basePosition=pos,
                baseOrientation=orn
            )
    
    def _load_robot(self):
        """Load robot into simulation."""
        try:
            # Load UR5 robot
            if self.config.robot_model == "ur5":
                self.robot_id = p.loadURDF(
                    "ur5.urdf",
                    basePosition=self.config.robot_position,
                    baseOrientation=self.config.robot_orientation,
                    useFixedBase=True
                )
            else:
                # Load generic robot
                self.robot_id = p.loadURDF(
                    "kuka_iiwa/model.urdf",
                    basePosition=self.config.robot_position,
                    baseOrientation=self.config.robot_orientation,
                    useFixedBase=True
                )
            
            # Get end effector index
            num_joints = p.getNumJoints(self.robot_id)
            self.end_effector_index = num_joints - 1
            
            # Enable force/torque sensing if configured
            if self.config.enable_force_torque:
                p.enableJointForceTorqueSensor(
                    self.robot_id,
                    self.end_effector_index,
                    enableSensor=1
                )
            
            # Set joint damping
            for i in range(num_joints):
                p.changeDynamics(
                    self.robot_id,
                    i,
                    linearDamping=0.1,
                    angularDamping=0.1
                )
            
            logger.info(f"Robot loaded: {self.config.robot_model}")
            
        except Exception as e:
            logger.error(f"Failed to load robot: {e}")
            raise
    
    def _load_medications(self):
        """Load medication objects into simulation."""
        for i, med_type in enumerate(self.config.medication_types):
            # Get position
            if i < len(self.config.medication_positions):
                position = self.config.medication_positions[i]
            else:
                # Generate random position within workspace
                ws_x, ws_y, ws_z = self.config.workspace_size
                position = (
                    np.random.uniform(-ws_x/4, ws_x/4),
                    np.random.uniform(-ws_y/4, ws_y/4),
                    0.8  # Table height
                )
            
            # Create medication object
            medication = MedicationObject(med_type, position)
            
            # Create visual shape
            dimensions = medication.dimensions
            medication_visual = p.createVisualShape(
                shapeType=p.GEOM_BOX,
                halfExtents=dimensions,
                rgbaColor=self._get_medication_color(med_type)
            )
            
            # Create collision shape
            medication_collision = p.createCollisionShape(
                shapeType=p.GEOM_BOX,
                halfExtents=dimensions
            )
            
            # Create body
            medication.body_id = p.createMultiBody(
                baseMass=medication.mass,
                baseCollisionShapeIndex=medication_collision,
                baseVisualShapeIndex=medication_visual,
                basePosition=position,
                baseOrientation=medication.orientation
            )
            
            # Set dynamics
            p.changeDynamics(
                medication.body_id,
                -1,
                lateralFriction=0.5,
                rollingFriction=0.01,
                spinningFriction=0.01,
                restitution=0.1
            )
            
            self.medications.append(medication)
            logger.info(f"Medication loaded: {med_type} at {position}")
    
    def _get_medication_color(self, med_type: str) -> List[float]:
        """Get color for medication type."""
        colors = {
            'vial': [0.8, 0.2, 0.2, 1.0],      # Red
            'bottle': [0.2, 0.8, 0.2, 1.0],    # Green
            'syringe': [0.2, 0.2, 0.8, 1.0],   # Blue
            'blister_pack': [0.8, 0.8, 0.2, 1.0] # Yellow
        }
        return colors.get(med_type, [0.5, 0.5, 0.5, 1.0])
    
    def _load_humans(self):
        """Load human agents into simulation."""
        for i, position in enumerate(self.config.human_positions):
            human = HumanAgent(position, self.config.human_behavior)
            
            # Create simple human representation (capsule)
            human_visual = p.createVisualShape(
                shapeType=p.GEOM_CAPSULE,
                radius=0.15,
                length=1.7,
                rgbaColor=[0.8, 0.6, 0.4, 1.0]
            )
            human_collision = p.createCollisionShape(
                shapeType=p.GEOM_CAPSULE,
                radius=0.15,
                height=1.7
            )
            
            human.body_id = p.createMultiBody(
                baseMass=70.0,  # Average human mass
                baseCollisionShapeIndex=human_collision,
                baseVisualShapeIndex=human_visual,
                basePosition=position
            )
            
            self.humans.append(human)
            logger.info(f"Human agent loaded at {position}")
    
    def _setup_cameras(self):
        """Setup simulation cameras."""
        # Default camera positions if not specified
        if not self.config.camera_positions:
            ws_x, ws_y, ws_z = self.config.workspace_size
            self.config.camera_positions = [
                {
                    'id': 'front',
                    'position': (0, -ws_y, ws_z),
                    'target': (0, 0, 0.5),
                    'up': (0, 0, 1)
                },
                {
                    'id': 'side',
                    'position': (-ws_x, 0, ws_z),
                    'target': (0, 0, 0.5),
                    'up': (0, 0, 1)
                },
                {
                    'id': 'overhead',
                    'position': (0, 0, ws_z + 1.0),
                    'target': (0, 0, 0),
                    'up': (0, 1, 0)
                }
            ]
        
        # Create cameras
        for camera_config in self.config.camera_positions:
            camera_id = camera_config['id']
            self.cameras[camera_id] = camera_config
    
    def _setup_rendering(self):
        """Setup rendering options."""
        if not self.config.enable_rendering:
            return
        
        # Configure renderer
        p.configureDebugVisualizer(
            p.COV_ENABLE_SHADOWS,
            self.config.render_shadows
        )
        
        p.configureDebugVisualizer(
            p.COV_ENABLE_ANTIALIASING,
            self.config.render_antialiasing
        )
        
        p.configureDebugVisualizer(
            p.COV_ENABLE_GUI,
            self.config.mode == SimulationMode.GUI
        )
    
    def get_camera_image(self, camera_id: str) -> np.ndarray:
        """Get image from camera."""
        if camera_id not in self.cameras:
            logger.error(f"Camera {camera_id} not found")
            return None
        
        camera_config = self.cameras[camera_id]
        
        # Get camera view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=camera_config['position'],
            cameraTargetPosition=camera_config['target'],
            cameraUpVector=camera_config['up']
        )
        
        # Get projection matrix
        aspect_ratio = self.config.camera_resolution[0] / self.config.camera_resolution[1]
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config.camera_fov,
            aspect=aspect_ratio,
            nearVal=0.1,
            farVal=10.0
        )
        
        # Get image
        width, height = self.config.camera_resolution
        img_arr = p.getCameraImage(
            width,
            height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL
        )
        
        # Convert to numpy array
        img = np.reshape(img_arr[2], (height, width, 4))[:, :, :3]
        
        return img
    
    def get_robot_state(self) -> Dict[str, Any]:
        """Get current robot state."""
        num_joints = p.getNumJoints(self.robot_id)
        
        joint_states = []
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_state = p.getJointState(self.robot_id, i)
            
            joint_states.append({
                'index': i,
                'name': joint_info[1].decode('utf-8'),
                'position': joint_state[0],
                'velocity': joint_state[1],
                'torque': joint_state[3],
                'force': joint_state[2]
            })
        
        # Get end effector pose
        end_effector_state = p.getLinkState(self.robot_id, self.end_effector_index)
        
        # Get force/torque if enabled
        force_torque = None
        if self.config.enable_force_torque:
            force_torque = p.getJointState(self.robot_id, self.end_effector_index)[2]
        
        return {
            'joint_states': joint_states,
            'end_effector_position': end_effector_state[0],
            'end_effector_orientation': end_effector_state[1],
            'force_torque': force_torque
        }
    
    def set_robot_joint_positions(self, joint_positions: List[float]):
        """Set robot joint positions."""
        for i, position in enumerate(joint_positions):
            p.resetJointState(self.robot_id, i, position)
    
    def get_medication_states(self) -> List[Dict[str, Any]]:
        """Get states of all medication objects."""
        states = []
        
        for medication in self.medications:
            pos, orn = p.getBasePositionAndOrientation(medication.body_id)
            lin_vel, ang_vel = p.getBaseVelocity(medication.body_id)
            
            states.append({
                'type': medication.med_type,
                'position': pos,
                'orientation': orn,
                'linear_velocity': lin_vel,
                'angular_velocity': ang_vel
            })
        
        return states
    
    def get_human_states(self) -> List[Dict[str, Any]]:
        """Get states of all human agents."""
        states = []
        
        for human in self.humans:
            pos, orn = p.getBasePositionAndOrientation(human.body_id)
            lin_vel, ang_vel = p.getBaseVelocity(human.body_id)
            
            states.append({
                'position': pos,
                'orientation': orn,
                'linear_velocity': lin_vel,
                'angular_velocity': ang_vel,
                'behavior': human.behavior
            })
        
        return states
    
    def check_collisions(self) -> List[Dict[str, Any]]:
        """Check for collisions in the simulation."""
        collisions = []
        
        if not self.config.enable_contact_detection:
            return collisions
        
        # Check robot-medication collisions
        for medication in self.medications:
            contact_points = p.getContactPoints(
                self.robot_id,
                medication.body_id
            )
            
            if contact_points:
                collisions.append({
                    'type': 'robot_medication',
                    'medication_type': medication.med_type,
                    'contact_points': len(contact_points)
                })
        
        # Check robot-human collisions
        for human in self.humans:
            contact_points = p.getContactPoints(
                self.robot_id,
                human.body_id
            )
            
            if contact_points:
                collisions.append({
                    'type': 'robot_human',
                    'contact_points': len(contact_points)
                })
        
        return collisions
    
    def step(self):
        """Step simulation forward by one time step."""
        p.stepSimulation()
        
        # Update humans
        for human in self.humans:
            human.update(self.config.time_step)
            # Update position in simulation
            p.resetBasePositionAndOrientation(
                human.body_id,
                human.position,
                [0, 0, 0, 1]
            )
        
        # Record data if enabled
        if self.config.record_data:
            self.frame_count += 1
            if self.frame_count % self.config.record_interval == 0:
                self._record_data()
    
    def _record_data(self):
        """Record simulation data."""
        # Record robot state
        robot_state = self.get_robot_state()
        self.recorded_data['robot_states'].append(robot_state)
        
        # Record medication states
        medication_states = self.get_medication_states()
        self.recorded_data['medication_states'].append(medication_states)
        
        # Record human states
        human_states = self.get_human_states()
        self.recorded_data['human_states'].append(human_states)
        
        # Record collisions
        collisions = self.check_collisions()
        self.recorded_data['collisions'].append(collisions)
        
        # Record camera images
        for camera_id in self.cameras:
            img = self.get_camera_image(camera_id)
            self.recorded_data[f'camera_{camera_id}'].append(img)
    
    def run(self, num_steps: int = 1000):
        """Run simulation for specified number of steps."""
        self.is_running = True
        logger.info(f"Running simulation for {num_steps} steps")
        
        try:
            for step in range(num_steps):
                if not self.is_running:
                    break
                
                self.step()
                
                # Print progress
                if step % 100 == 0:
                    logger.info(f"Simulation step: {step}/{num_steps}")
                
                # Small delay for real-time visualization
                if self.config.mode == SimulationMode.GUI:
                    time.sleep(0.01)
            
            logger.info("Simulation completed")
            
        except KeyboardInterrupt:
            logger.info("Simulation interrupted by user")
        finally:
            self.is_running = False
    
    def stop(self):
        """Stop simulation."""
        self.is_running = False
    
    def save_recorded_data(self, output_path: str = None):
        """Save recorded simulation data."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"{self.config.output_dir}/simulation_data_{timestamp}.npz"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert to numpy arrays and save
        save_dict = {}
        for key, value in self.recorded_data.items():
            if key.startswith('camera_'):
                # Save images separately
                continue
            save_dict[key] = value
        
        np.savez_compressed(output_path, **save_dict)
        
        logger.info(f"Simulation data saved to: {output_path}")
    
    def close(self):
        """Close simulation and cleanup."""
        p.disconnect()
        logger.info("Simulation closed")

def main():
    """Main function for running simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Robot Simulator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--mode', type=str, default='gui', choices=['gui', 'direct'], help='Simulation mode')
    parser.add_argument('--environment', type=str, default='pharmacy', 
                       choices=['pharmacy', 'hospital_room', 'emergency_department'],
                       help='Environment type')
    parser.add_argument('--steps', type=int, default=1000, help='Number of simulation steps')
    parser.add_argument('--output', type=str, help='Output directory for recorded data')
    
    args = parser.parse_args()
    
    # Load configuration
    config = SimulationConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Override with command line arguments
    config.mode = SimulationMode(args.mode)
    config.environment_type = ClinicalEnvironment(args.environment)
    if args.output:
        config.output_dir = args.output
    
    # Create simulator
    simulator = ClinicalSimulator(config)
    
    try:
        # Run simulation
        simulator.run(args.steps)
        
        # Save recorded data
        if config.record_data:
            simulator.save_recorded_data()
    
    except KeyboardInterrupt:
        logger.info("Simulation interrupted by user")
    finally:
        simulator.close()

if __name__ == "__main__":
    main()
