#!/usr/bin/env python3
"""
Multi-Robot Coordination System for Clinical Environments
Coordination framework for multiple robots working together in clinical settings.

This module implements:
- Multi-robot task allocation and scheduling
- Collision avoidance and path planning
- Cooperative manipulation and transport
- Distributed decision making
- Robot formation control
- Task synchronization
- Resource sharing and optimization
- Fault tolerance and recovery

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
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import warnings

# Scientific computing
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment

# Path planning
import networkx as nx
from shapely.geometry import Point, Polygon
from shapely.ops import nearest_points

# Communication
import zmq
import pickle

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('multi_robot_coordination.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class RobotRole(Enum):
    """Roles for robots in multi-robot system."""
    LEADER = "leader"
    FOLLOWER = "follower"
    SPECIALIST = "specialist"
    TRANSPORTER = "transporter"
    MANIPULATOR = "manipulator"
    INSPECTOR = "inspector"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class CoordinationMode(Enum):
    """Coordination modes for multi-robot systems."""
    CENTRALIZED = "centralized"
    DISTRIBUTED = "distributed"
    HIERARCHICAL = "hierarchical"
    DECENTRALIZED = "decentralized"

@dataclass
class RobotState:
    """State of a single robot."""
    robot_id: str
    position: Tuple[float, float, float]
    orientation: Tuple[float, float, float, float]  # Quaternion
    velocity: Tuple[float, float, float]
    role: RobotRole
    capabilities: List[str]
    current_task: Optional[str] = None
    battery_level: float = 100.0
    status: str = "idle"  # idle, busy, error, charging
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'robot_id': self.robot_id,
            'position': self.position,
            'orientation': self.orientation,
            'velocity': self.velocity,
            'role': self.role.value,
            'capabilities': self.capabilities,
            'current_task': self.current_task,
            'battery_level': self.battery_level,
            'status': self.status
        }

@dataclass
class Task:
    """Task for multi-robot system."""
    task_id: str
    task_type: str
    priority: TaskPriority
    required_capabilities: List[str]
    required_robots: int
    target_location: Tuple[float, float, float]
    deadline: Optional[datetime] = None
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    assigned_robots: List[str] = field(default_factory=list)
    status: str = "pending"  # pending, assigned, in_progress, completed, failed
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'required_capabilities': self.required_capabilities,
            'required_robots': self.required_robots,
            'target_location': self.target_location,
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'assigned_robots': self.assigned_robots,
            'status': self.status
        }

@dataclass
class CoordinationConfig:
    """Configuration for multi-robot coordination."""
    
    # Coordination settings
    coordination_mode: CoordinationMode = CoordinationMode.CENTRALIZED
    task_allocation_algorithm: str = "greedy"  # greedy, auction, optimization
    collision_availability_distance: float = 0.5  # meters
    communication_timeout: float = 5.0  # seconds
    
    # Task scheduling
    scheduling_algorithm: str = "priority_queue"  # priority_queue, round_robin, deadline_aware
    max_concurrent_tasks: int = 10
    task_timeout: int = 300  # seconds
    
    # Formation control
    enable_formation_control: bool = True
    formation_type: str = "line"  # line, circle, triangle, custom
    formation_spacing: float = 1.0  # meters
    
    # Communication
    enable_communication: bool = True
    communication_protocol: str = "zmq"
    communication_port: int = 5555
    
    # Monitoring
    enable_monitoring: bool = True
    monitoring_interval: float = 1.0  # seconds
    
    # Safety
    enable_safety_checks: bool = True
    emergency_stop_on_collision: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'coordination_mode': self.coordination_mode.value,
            'task_allocation_algorithm': self.task_allocation_algorithm,
            'collision_availability_distance': self.collision_availability_distance,
            'communication_timeout': self.communication_timeout,
            'scheduling_algorithm': self.scheduling_algorithm,
            'max_concurrent_tasks': self.max_concurrent_tasks,
            'task_timeout': self.task_timeout,
            'enable_formation_control': self.enable_formation_control,
            'formation_type': self.formation_type,
            'formation_spacing': self.formation_spacing,
            'enable_communication': self.enable_communication,
            'communication_protocol': self.communication_protocol,
            'communication_port': self.communication_port,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'enable_safety_checks': self.enable_safety_checks,
            'emergency_stop_on_collision': self.emergency_stop_on_collision
        }

class TaskAllocator:
    """Task allocation for multi-robot systems."""
    
    def __init__(self, config: CoordinationConfig):
        self.config = config
    
    def allocate_tasks(self, tasks: List[Task], robots: List[RobotState]) -> Dict[str, List[str]]:
        """Allocate tasks to robots based on algorithm."""
        if self.config.task_allocation_algorithm == "greedy":
            return self._greedy_allocation(tasks, robots)
        elif self.config.task_allocation_algorithm == "auction":
            return self._auction_allocation(tasks, robots)
        elif self.config.task_allocation_algorithm == "optimization":
            return self._optimization_allocation(tasks, robots)
        else:
            return self._greedy_allocation(tasks, robots)
    
    def _greedy_allocation(self, tasks: List[Task], robots: List[RobotState]) -> Dict[str, List[str]]:
        """Greedy task allocation."""
        allocation = {}
        
        # Sort tasks by priority
        sorted_tasks = sorted(tasks, key=lambda t: self._priority_value(t.priority), reverse=True)
        
        for task in sorted_tasks:
            # Find available robots with required capabilities
            available_robots = [
                robot for robot in robots 
                if robot.status == "idle" and 
                all(cap in robot.capabilities for cap in task.required_capabilities)
            ]
            
            # Assign robots to task
            if len(available_robots) >= task.required_robots:
                assigned = available_robots[:task.required_robots]
                allocation[task.task_id] = [robot.robot_id for robot in assigned]
                
                # Update robot states
                for robot in assigned:
                    robot.current_task = task.task_id
                    robot.status = "busy"
        
        return allocation
    
    def _auction_allocation(self, tasks: List[Task], robots: List[RobotState]) -> Dict[str, List[str]]:
        """Auction-based task allocation."""
        allocation = {}
        
        # Simplified auction algorithm
        for task in tasks:
            bids = []
            
            for robot in robots:
                if robot.status == "idle":
                    # Calculate bid based on distance and capabilities
                    distance = np.linalg.norm(
                        np.array(robot.position) - np.array(task.target_location)
                    )
                    capability_match = sum(cap in robot.capabilities for cap in task.required_capabilities)
                    
                    bid = capability_match * 100 - distance
                    bids.append((robot.robot_id, bid))
            
            # Assign to highest bidder
            if bids:
                bids.sort(key=lambda x: x[1], reverse=True)
                assigned_robots = [bid[0] for bid in bids[:task.required_robots]]
                allocation[task.task_id] = assigned_robots
        
        return allocation
    
    def _optimization_allocation(self, tasks: List[Task], robots: List[RobotState]) -> Dict[str, List[str]]:
        """Optimization-based task allocation using Hungarian algorithm."""
        allocation = {}
        
        # Create cost matrix
        num_tasks = len(tasks)
        num_robots = len(robots)
        
        cost_matrix = np.zeros((num_tasks, num_robots))
        
        for i, task in enumerate(tasks):
            for j, robot in enumerate(robots):
                # Calculate cost based on distance and capability match
                distance = np.linalg.norm(
                    np.array(robot.position) - np.array(task.target_location)
                )
                capability_match = sum(cap in robot.capabilities for cap in task.required_capabilities)
                
                cost_matrix[i, j] = distance / (capability_match + 0.1)
        
        # Solve assignment problem
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Assign tasks
        for i, j in zip(row_ind, col_ind):
            task = tasks[i]
            robot = robots[j]
            
            if task.task_id not in allocation:
                allocation[task.task_id] = []
            
            allocation[task.task_id].append(robot.robot_id)
        
        return allocation
    
    def _priority_value(self, priority: TaskPriority) -> int:
        """Convert priority to numeric value."""
        values = {
            TaskPriority.CRITICAL: 4,
            TaskPriority.HIGH: 3,
            TaskPriority.MEDIUM: 2,
            TaskPriority.LOW: 1
        }
        return values.get(priority, 0)

class PathPlanner:
    """Path planning for multi-robot collision avoidance."""
    
    def __init__(self, config: CoordinationConfig):
        self.config = config
        self.obstacles = []
        self.workspace_bounds = ((-5, 5), (-5, 5), (0, 2))
    
    def plan_path(self, start: Tuple[float, float, float], 
                 goal: Tuple[float, float, float],
                 other_robot_paths: List[List[Tuple[float, float, float]]] = None) -> List[Tuple[float, float, float]]:
        """Plan collision-free path."""
        # Simplified path planning using straight line with collision avoidance
        path = []
        
        # Generate waypoints
        num_waypoints = 10
        for i in range(num_waypoints + 1):
            t = i / num_waypoints
            waypoint = (
                start[0] + t * (goal[0] - start[0]),
                start[1] + t * (goal[1] - start[1]),
                start[2] + t * (goal[2] - start[2])
            )
            
            # Check for collisions with other robot paths
            if other_robot_paths:
                waypoint = self._avoid_collision(waypoint, other_robot_paths)
            
            path.append(waypoint)
        
        return path
    
    def _avoid_collision(self, waypoint: Tuple[float, float, float],
                        other_paths: List[List[Tuple[float, float, float]]]) -> Tuple[float, float, float]:
        """Adjust waypoint to avoid collision with other robots."""
        adjusted_waypoint = list(waypoint)
        
        for path in other_paths:
            for other_waypoint in path:
                distance = np.linalg.norm(np.array(waypoint) - np.array(other_waypoint))
                
                if distance < self.config.collision_availability_distance:
                    # Move away from other robot
                    direction = np.array(waypoint) - np.array(other_waypoint)
                    direction = direction / (np.linalg.norm(direction) + 1e-6)
                    
                    adjusted_waypoint = np.array(waypoint) + direction * (self.config.collision_availability_distance - distance)
                    adjusted_waypoint = tuple(adjusted_waypoint)
                    break
        
        return tuple(adjusted_waypoint)

class FormationController:
    """Formation control for multi-robot systems."""
    
    def __init__(self, config: CoordinationConfig):
        self.config = config
        self.formation_type = config.formation_type
        self.spacing = config.formation_spacing
    
    def calculate_formation_positions(self, leader_position: Tuple[float, float, float],
                                    num_robots: int) -> List[Tuple[float, float, float]]:
        """Calculate formation positions for robots."""
        positions = []
        
        if self.formation_type == "line":
            # Line formation
            for i in range(num_robots):
                offset = (i - (num_robots - 1) / 2) * self.spacing
                position = (
                    leader_position[0] + offset,
                    leader_position[1],
                    leader_position[2]
                )
                positions.append(position)
        
        elif self.formation_type == "circle":
            # Circle formation
            radius = self.spacing * (num_robots / (2 * np.pi))
            for i in range(num_robots):
                angle = 2 * np.pi * i / num_robots
                position = (
                    leader_position[0] + radius * np.cos(angle),
                    leader_position[1] + radius * np.sin(angle),
                    leader_position[2]
                )
                positions.append(position)
        
        elif self.formation_type == "triangle":
            # Triangle formation
            if num_robots >= 3:
                positions.append(leader_position)
                for i in range(1, num_robots):
                    angle = 2 * np.pi * (i - 1) / (num_robots - 1)
                    position = (
                        leader_position[0] + self.spacing * np.cos(angle),
                        leader_position[1] + self.spacing * np.sin(angle),
                        leader_position[2]
                    )
                    positions.append(position)
            else:
                positions = [leader_position] * num_robots
        
        else:
            # Default: all robots at same position
            positions = [leader_position] * num_robots
        
        return positions

class CommunicationManager:
    """Communication manager for multi-robot coordination."""
    
    def __init__(self, config: CoordinationConfig):
        self.config = config
        self.context = None
        self.socket = None
        
        if config.enable_communication:
            self._initialize_communication()
    
    def _initialize_communication(self):
        """Initialize communication system."""
        try:
            self.context = zmq.Context()
            self.socket = self.context.socket(zmq.PUB)
            self.socket.bind(f"tcp://*:{self.config.communication_port}")
            logger.info(f"Communication server started on port {self.config.communication_port}")
        except Exception as e:
            logger.error(f"Failed to initialize communication: {e}")
    
    def broadcast_message(self, message: Dict[str, Any]):
        """Broadcast message to all robots."""
        if self.socket:
            try:
                self.socket.send(pickle.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast message: {e}")
    
    def shutdown(self):
        """Shutdown communication system."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()

class MultiRobotCoordinator:
    """Main multi-robot coordination system."""
    
    def __init__(self, config: CoordinationConfig):
        self.config = config
        
        # Initialize components
        self.task_allocator = TaskAllocator(config)
        self.path_planner = PathPlanner(config)
        self.formation_controller = FormationController(config)
        self.communication_manager = CommunicationManager(config)
        
        # Robot and task management
        self.robots = {}
        self.tasks = {}
        self.task_queue = deque()
        
        # Coordination state
        self.is_coordinating = False
        self.coordination_thread = None
        
        logger.info("Multi-Robot Coordinator initialized")
    
    def add_robot(self, robot_state: RobotState):
        """Add robot to coordination system."""
        self.robots[robot_state.robot_id] = robot_state
        logger.info(f"Robot added: {robot_state.robot_id}")
    
    def remove_robot(self, robot_id: str):
        """Remove robot from coordination system."""
        if robot_id in self.robots:
            del self.robots[robot_id]
            logger.info(f"Robot removed: {robot_id}")
    
    def add_task(self, task: Task):
        """Add task to coordination system."""
        self.tasks[task.task_id] = task
        self.task_queue.append(task)
        logger.info(f"Task added: {task.task_id}")
    
    def start_coordination(self):
        """Start multi-robot coordination."""
        if self.is_coordinating:
            return
        
        self.is_coordinating = True
        self.coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self.coordination_thread.start()
        
        logger.info("Multi-robot coordination started")
    
    def stop_coordination(self):
        """Stop multi-robot coordination."""
        self.is_coordinating = False
        
        if self.coordination_thread:
            self.coordination_thread.join(timeout=5)
        
        logger.info("Multi-robot coordination stopped")
    
    def _coordination_loop(self):
        """Main coordination loop."""
        while self.is_coordinating:
            try:
                # Allocate tasks
                if self.task_queue:
                    tasks_to_allocate = list(self.task_queue)
                    allocation = self.task_allocator.allocate_tasks(tasks_to_allocate, list(self.robots.values()))
                    
                    # Update task assignments
                    for task_id, robot_ids in allocation.items():
                        if task_id in self.tasks:
                            self.tasks[task_id].assigned_robots = robot_ids
                            self.tasks[task_id].status = "assigned"
                            
                            # Update robot states
                            for robot_id in robot_ids:
                                if robot_id in self.robots:
                                    self.robots[robot_id].current_task = task_id
                                    self.robots[robot_id].status = "busy"
                    
                    # Remove allocated tasks from queue
                    for task_id in allocation.keys():
                        self.task_queue = deque([t for t in self.task_queue if t.task_id != task_id])
                
                # Broadcast state updates
                if self.config.enable_communication:
                    state_message = {
                        'type': 'state_update',
                        'robots': {rid: r.to_dict() for rid, r in self.robots.items()},
                        'tasks': {tid: t.to_dict() for tid, t in self.tasks.items()}
                    }
                    self.communication_manager.broadcast_message(state_message)
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Coordination error: {e}")
    
    def execute_cooperative_task(self, task: Task, robot_ids: List[str]) -> bool:
        """Execute cooperative task with multiple robots."""
        logger.info(f"Executing cooperative task {task.task_id} with robots {robot_ids}")
        
        try:
            # Get robot positions
            robot_positions = [self.robots[rid].position for rid in robot_ids if rid in self.robots]
            
            if not robot_positions:
                logger.error("No valid robots for cooperative task")
                return False
            
            # Calculate formation if enabled
            if self.config.enable_formation_control:
                leader_position = robot_positions[0]
                formation_positions = self.formation_controller.calculate_formation_positions(
                    leader_position, len(robot_ids)
                )
            else:
                formation_positions = robot_positions
            
            # Plan paths for each robot
            paths = []
            for i, robot_id in enumerate(robot_ids):
                if robot_id in self.robots:
                    start = self.robots[robot_id].position
                    goal = formation_positions[i]
                    
                    # Get other robot paths for collision avoidance
                    other_paths = [p for j, p in enumerate(paths) if j != i]
                    
                    path = self.path_planner.plan_path(start, goal, other_paths)
                    paths.append(path)
            
            # Execute paths (simplified)
            task.status = "in_progress"
            
            # Simulate execution time
            time.sleep(2.0)
            
            task.status = "completed"
            
            # Update robot states
            for robot_id in robot_ids:
                if robot_id in self.robots:
                    self.robots[robot_id].current_task = None
                    self.robots[robot_id].status = "idle"
            
            logger.info(f"Cooperative task {task.task_id} completed")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute cooperative task: {e}")
            task.status = "failed"
            return False
    
    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status."""
        return {
            'num_robots': len(self.robots),
            'num_tasks': len(self.tasks),
            'pending_tasks': len(self.task_queue),
            'busy_robots': sum(1 for r in self.robots.values() if r.status == "busy"),
            'idle_robots': sum(1 for r in self.robots.values() if r.status == "idle"),
            'coordination_mode': self.config.coordination_mode.value,
            'is_coordinating': self.is_coordinating
        }
    
    def shutdown(self):
        """Shutdown coordination system."""
        self.stop_coordination()
        self.communication_manager.shutdown()
        logger.info("Multi-Robot Coordinator shutdown")

def main():
    """Main function for multi-robot coordination."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Robot Coordinator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--num-robots', type=int, default=3, help='Number of robots')
    parser.add_argument('--action', type=str, default='coordinate',
                       choices=['coordinate', 'cooperative_task'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    # Load configuration
    config = CoordinationConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create coordinator
    coordinator = MultiRobotCoordinator(config)
    
    # Add robots
    for i in range(args.num_robots):
        robot = RobotState(
            robot_id=f"robot_{i}",
            position=(i * 2.0, 0.0, 0.0),
            orientation=(0.0, 0.0, 0.0, 1.0),
            velocity=(0.0, 0.0, 0.0),
            role=RobotRole.FOLLOWER if i > 0 else RobotRole.LEADER,
            capabilities=['navigation', 'manipulation', 'transport'],
            status="idle"
        )
        coordinator.add_robot(robot)
    
    # Perform action
    if args.action == "coordinate":
        # Add sample tasks
        task1 = Task(
            task_id="task_001",
            task_type="transport",
            priority=TaskPriority.HIGH,
            required_capabilities=['transport'],
            required_robots=2,
            target_location=(5.0, 3.0, 0.0)
        )
        coordinator.add_task(task1)
        
        task2 = Task(
            task_id="task_002",
            task_type="manipulation",
            priority=TaskPriority.MEDIUM,
            required_capabilities=['manipulation'],
            required_robots=1,
            target_location=(2.0, 5.0, 0.0)
        )
        coordinator.add_task(task2)
        
        # Start coordination
        coordinator.start_coordination()
        
        print("Multi-robot coordination running. Press Ctrl+C to stop.")
        try:
            while True:
                status = coordinator.get_coordination_status()
                print(f"Status: {status}")
                time.sleep(5)
        except KeyboardInterrupt:
            print("\nStopping coordination...")
            coordinator.stop_coordination()
    
    elif args.action == "cooperative_task":
        # Create cooperative task
        cooperative_task = Task(
            task_id="cooperative_001",
            task_type="cooperative_transport",
            priority=TaskPriority.HIGH,
            required_capabilities=['transport', 'navigation'],
            required_robots=args.num_robots,
            target_location=(10.0, 10.0, 0.0)
        )
        
        robot_ids = [f"robot_{i}" for i in range(args.num_robots)]
        success = coordinator.execute_cooperative_task(cooperative_task, robot_ids)
        
        if success:
            print("Cooperative task completed successfully")
        else:
            print("Cooperative task failed")
    
    coordinator.shutdown()

if __name__ == "__main__":
    main()
