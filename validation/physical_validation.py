#!/usr/bin/env python3
"""
Physical Validation Scripts for Clinical Robot Adaptation
Comprehensive validation framework for testing clinical robot performance in real environments.

This module implements:
- Physical safety validation tests
- Clinical task performance validation
- Human-robot interaction safety tests
- Environmental adaptation validation
- Real-time performance monitoring
- Statistical analysis and reporting
- Compliance checking for clinical standards

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
import pandas as pd
from scipy import stats
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt
import seaborn as sns

# Computer vision
import cv2
from PIL import Image
import open3d as o3d

# Robotics and simulation
import pybullet as p
import trimesh

# Data analysis
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('physical_validation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class ValidationStatus(Enum):
    """Validation test status."""
    PENDING = "pending"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"
    ERROR = "error"

class SafetyLevel(Enum):
    """Safety levels for validation."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ValidationConfig:
    """Configuration for physical validation."""
    
    # Test configuration
    test_duration: float = 60.0  # seconds
    sampling_rate: float = 30.0  # Hz
    max_test_attempts: int = 3
    
    # Safety thresholds
    max_velocity: float = 0.3  # m/s
    max_acceleration: float = 2.0  # m/s^2
    max_force: float = 50.0  # N
    min_human_distance: float = 0.5  # m
    max_collision_risk: float = 0.1
    
    # Performance thresholds
    min_success_rate: float = 0.8
    max_task_time: float = 30.0  # seconds
    min_grasp_success: float = 0.9
    
    # Clinical requirements
    medication_handling_accuracy: float = 0.95
    sterility_maintenance: float = 0.99
    emergency_response_time: float = 0.5  # seconds
    
    # Reporting
    generate_plots: bool = True
    save_raw_data: bool = True
    export_format: List[str] = field(default_factory=lambda: ["json", "csv", "pdf"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_duration': self.test_duration,
            'sampling_rate': self.sampling_rate,
            'max_test_attempts': self.max_test_attempts,
            'max_velocity': self.max_velocity,
            'max_acceleration': self.max_acceleration,
            'max_force': self.max_force,
            'min_human_distance': self.min_human_distance,
            'max_collision_risk': self.max_collision_risk,
            'min_success_rate': self.min_success_rate,
            'max_task_time': self.max_task_time,
            'min_grasp_success': self.min_grasp_success,
            'medication_handling_accuracy': self.medication_handling_accuracy,
            'sterility_maintenance': self.sterility_maintenance,
            'emergency_response_time': self.emergency_response_time,
            'generate_plots': self.generate_plots,
            'save_raw_data': self.save_raw_data,
            'export_format': self.export_format
        }

@dataclass
class ValidationResult:
    """Result of a validation test."""
    test_name: str
    status: ValidationStatus
    score: float
    duration: float
    metrics: Dict[str, Any]
    safety_violations: List[str]
    errors: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'test_name': self.test_name,
            'status': self.status.value,
            'score': self.score,
            'duration': self.duration,
            'metrics': self.metrics,
            'safety_violations': self.safety_violations,
            'errors': self.errors,
            'timestamp': self.timestamp.isoformat()
        }

class SafetyValidator:
    """Safety validation for clinical robot operations."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.safety_data = defaultdict(list)
        self.violations = []
    
    def validate_velocity_limits(self, trajectory: np.ndarray) -> ValidationResult:
        """Validate velocity limits."""
        test_name = "velocity_limits_validation"
        start_time = time.time()
        
        try:
            # Compute velocities
            velocities = np.diff(trajectory, axis=0) / (1.0 / self.config.sampling_rate)
            max_velocities = np.max(np.abs(velocities), axis=0)
            
            # Check violations
            violations = []
            for i, vel in enumerate(max_velocities):
                if vel > self.config.max_velocity:
                    violations.append(f"Joint {i}: velocity {vel:.3f} > limit {self.config.max_velocity}")
            
            # Calculate score
            max_violation = max(max_velocities) if max_velocities.any() else 0
            score = max(0.0, 1.0 - (max_violation / self.config.max_velocity - 1.0))
            
            status = ValidationStatus.PASSED if len(violations) == 0 else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'max_velocities': max_velocities.tolist(),
                    'velocity_violations': len(violations)
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )
    
    def validate_human_proximity(self, robot_trajectory: np.ndarray, 
                               human_positions: np.ndarray) -> ValidationResult:
        """Validate human proximity safety."""
        test_name = "human_proximity_validation"
        start_time = time.time()
        
        try:
            violations = []
            min_distances = []
            
            for i, robot_pos in enumerate(robot_trajectory):
                # Calculate distances to all humans
                distances = np.linalg.norm(human_positions - robot_pos, axis=1)
                min_dist = np.min(distances)
                min_distances.append(min_dist)
                
                if min_dist < self.config.min_human_distance:
                    violations.append(f"Time {i}: distance {min_dist:.3f} < limit {self.config.min_human_distance}")
            
            # Calculate score
            avg_min_distance = np.mean(min_distances)
            score = min(1.0, avg_min_distance / self.config.min_human_distance)
            
            status = ValidationStatus.PASSED if len(violations) == 0 else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'avg_min_distance': avg_min_distance,
                    'min_distance_overall': np.min(min_distances),
                    'proximity_violations': len(violations)
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )
    
    def validate_collision_risk(self, robot_trajectory: np.ndarray, 
                              obstacles: List[Dict[str, Any]]) -> ValidationResult:
        """Validate collision risk with obstacles."""
        test_name = "collision_risk_validation"
        start_time = time.time()
        
        try:
            violations = []
            collision_risks = []
            
            for i, robot_pos in enumerate(robot_trajectory):
                max_risk = 0.0
                
                for obstacle in obstacles:
                    obs_pos = np.array(obstacle['position'])
                    obs_size = np.array(obstacle['size'])
                    
                    # Calculate distance to obstacle
                    distance = np.linalg.norm(robot_pos - obs_pos)
                    
                    # Calculate risk based on distance and size
                    risk = 1.0 / (1.0 + distance / np.linalg.norm(obs_size))
                    max_risk = max(max_risk, risk)
                    
                    if risk > self.config.max_collision_risk:
                        violations.append(f"Time {i}: collision risk {risk:.3f} > limit {self.config.max_collision_risk}")
                
                collision_risks.append(max_risk)
            
            # Calculate score
            avg_risk = np.mean(collision_risks)
            score = max(0.0, 1.0 - avg_risk / self.config.max_collision_risk)
            
            status = ValidationStatus.PASSED if len(violations) == 0 else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'avg_collision_risk': avg_risk,
                    'max_collision_risk': np.max(collision_risks),
                    'collision_violations': len(violations)
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )

class ClinicalTaskValidator:
    """Clinical task performance validation."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.task_data = defaultdict(list)
    
    def validate_medication_handling(self, demonstrations: List[Dict[str, Any]]) -> ValidationResult:
        """Validate medication handling performance."""
        test_name = "medication_handling_validation"
        start_time = time.time()
        
        try:
            success_count = 0
            handling_times = []
            accuracy_scores = []
            violations = []
            
            for demo in demonstrations:
                # Check success
                if demo.get('success', False):
                    success_count += 1
                
                # Record handling time
                handling_time = demo.get('handling_time', 0.0)
                handling_times.append(handling_time)
                
                # Record accuracy
                accuracy = demo.get('accuracy', 0.0)
                accuracy_scores.append(accuracy)
                
                # Check for violations
                if handling_time > self.config.max_task_time:
                    violations.append(f"Handling time {handling_time:.2f}s > limit {self.config.max_task_time}s")
                
                if accuracy < self.config.medication_handling_accuracy:
                    violations.append(f"Accuracy {accuracy:.3f} < requirement {self.config.medication_handling_accuracy}")
            
            # Calculate metrics
            success_rate = success_count / len(demonstrations)
            avg_handling_time = np.mean(handling_times)
            avg_accuracy = np.mean(accuracy_scores)
            
            # Calculate overall score
            score = (success_rate * 0.4 + 
                   (1.0 - avg_handling_time / self.config.max_task_time) * 0.3 + 
                   avg_accuracy * 0.3)
            
            status = ValidationStatus.PASSED if (success_rate >= self.config.min_success_rate and 
                                              len(violations) == 0) else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'success_rate': success_rate,
                    'avg_handling_time': avg_handling_time,
                    'avg_accuracy': avg_accuracy,
                    'total_demonstrations': len(demonstrations)
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )
    
    def validate_grasp_success(self, grasp_attempts: List[Dict[str, Any]]) -> ValidationResult:
        """Validate grasp success rate."""
        test_name = "grasp_success_validation"
        start_time = time.time()
        
        try:
            successful_grasps = 0
            grasp_types = defaultdict(int)
            violations = []
            
            for attempt in grasp_attempts:
                # Check success
                if attempt.get('success', False):
                    successful_grasps += 1
                
                # Record grasp type
                grasp_type = attempt.get('grasp_type', 'unknown')
                grasp_types[grasp_type] += 1
            
            # Calculate success rate
            success_rate = successful_grasps / len(grasp_attempts)
            
            # Check for violations
            if success_rate < self.config.min_grasp_success:
                violations.append(f"Grasp success rate {success_rate:.3f} < requirement {self.config.min_grasp_success}")
            
            score = success_rate
            status = ValidationStatus.PASSED if success_rate >= self.config.min_grasp_success else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'success_rate': success_rate,
                    'total_attempts': len(grasp_attempts),
                    'successful_grasps': successful_grasps,
                    'grasp_types': dict(grasp_types)
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )
    
    def validate_sterility_maintenance(self, sterility_checks: List[Dict[str, Any]]) -> ValidationResult:
        """Validate sterility maintenance during procedures."""
        test_name = "sterility_maintenance_validation"
        start_time = time.time()
        
        try:
            violations = []
            sterile_procedures = 0
            
            for check in sterility_checks:
                # Check if procedure maintained sterility
                if check.get('sterile', False):
                    sterile_procedures += 1
                else:
                    violations.append(f"Sterility breach detected: {check.get('breach_type', 'unknown')}")
            
            # Calculate sterility rate
            sterility_rate = sterile_procedures / len(sterility_checks)
            
            # Check for violations
            if sterility_rate < self.config.sterility_maintenance:
                violations.append(f"Sterility rate {sterility_rate:.3f} < requirement {self.config.sterility_maintenance}")
            
            score = sterility_rate
            status = ValidationStatus.PASSED if sterility_rate >= self.config.sterility_maintenance else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'sterility_rate': sterility_rate,
                    'total_checks': len(sterility_checks),
                    'sterile_procedures': sterile_procedures
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )

class PerformanceValidator:
    """Performance validation for clinical robot systems."""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.performance_data = defaultdict(list)
    
    def validate_response_time(self, response_times: List[float]) -> ValidationResult:
        """Validate system response times."""
        test_name = "response_time_validation"
        start_time = time.time()
        
        try:
            violations = []
            
            # Calculate statistics
            avg_response_time = np.mean(response_times)
            p95_response_time = np.percentile(response_times, 95)
            p99_response_time = np.percentile(response_times, 99)
            max_response_time = np.max(response_times)
            
            # Check emergency response time
            emergency_times = [t for t in response_times if t < 1.0]  # Assume emergency responses are < 1s
            if emergency_times:
                avg_emergency_time = np.mean(emergency_times)
                if avg_emergency_time > self.config.emergency_response_time:
                    violations.append(f"Emergency response time {avg_emergency_time:.3f}s > limit {self.config.emergency_response_time}s")
            
            # Calculate score based on average response time
            target_time = 0.1  # 100ms target
            score = max(0.0, 1.0 - (avg_response_time - target_time) / target_time)
            
            status = ValidationStatus.PASSED if len(violations) == 0 else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'avg_response_time': avg_response_time,
                    'p95_response_time': p95_response_time,
                    'p99_response_time': p99_response_time,
                    'max_response_time': max_response_time,
                    'emergency_response_time': avg_emergency_time if emergency_times else 0.0
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )
    
    def validate_system_stability(self, system_metrics: List[Dict[str, Any]]) -> ValidationResult:
        """Validate system stability over time."""
        test_name = "system_stability_validation"
        start_time = time.time()
        
        try:
            violations = []
            
            # Extract metrics
            cpu_usage = [m['cpu_usage'] for m in system_metrics]
            memory_usage = [m['memory_usage'] for m in system_metrics]
            temperatures = [m.get('temperature', 0.0) for m in system_metrics]
            
            # Calculate stability metrics
            cpu_std = np.std(cpu_usage)
            memory_std = np.std(memory_usage)
            temp_std = np.std(temperatures)
            
            avg_cpu = np.mean(cpu_usage)
            avg_memory = np.mean(memory_usage)
            avg_temp = np.mean(temperatures)
            
            # Check for stability violations
            if cpu_std > 20.0:  # High CPU variability
                violations.append(f"CPU usage unstable: std={cpu_std:.2f}%")
            
            if memory_std > 15.0:  # High memory variability
                violations.append(f"Memory usage unstable: std={memory_std:.2f}%")
            
            if avg_cpu > 90.0:  # High average CPU
                violations.append(f"High average CPU usage: {avg_cpu:.2f}%")
            
            if avg_memory > 90.0:  # High average memory
                violations.append(f"High average memory usage: {avg_memory:.2f}%")
            
            # Calculate stability score
            stability_score = 1.0 - (cpu_std / 100.0 + memory_std / 100.0) / 2.0
            score = max(0.0, stability_score)
            
            status = ValidationStatus.PASSED if len(violations) == 0 else ValidationStatus.FAILED
            
            result = ValidationResult(
                test_name=test_name,
                status=status,
                score=score,
                duration=time.time() - start_time,
                metrics={
                    'cpu_stability': 1.0 - cpu_std / 100.0,
                    'memory_stability': 1.0 - memory_std / 100.0,
                    'avg_cpu_usage': avg_cpu,
                    'avg_memory_usage': avg_memory,
                    'avg_temperature': avg_temp
                },
                safety_violations=violations,
                errors=[]
            )
            
            return result
            
        except Exception as e:
            return ValidationResult(
                test_name=test_name,
                status=ValidationStatus.ERROR,
                score=0.0,
                duration=time.time() - start_time,
                metrics={},
                safety_violations=[],
                errors=[str(e)]
            )

class PhysicalValidationFramework:
    """Main physical validation framework."""
    
    def __init__(self, config: ValidationConfig, output_dir: str = "./validation_results"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize validators
        self.safety_validator = SafetyValidator(config)
        self.clinical_validator = ClinicalTaskValidator(config)
        self.performance_validator = PerformanceValidator(config)
        
        # Results storage
        self.validation_results = []
        self.test_data = defaultdict(list)
        
        # Monitoring
        self.is_running = False
        self.monitoring_thread = None
        
        logger.info("Physical Validation Framework initialized")
    
    def run_safety_validation(self, robot_trajectory: np.ndarray, 
                            human_positions: np.ndarray = None,
                            obstacles: List[Dict[str, Any]] = None) -> List[ValidationResult]:
        """Run comprehensive safety validation."""
        logger.info("Starting safety validation...")
        
        results = []
        
        # Velocity limits validation
        result = self.safety_validator.validate_velocity_limits(robot_trajectory)
        results.append(result)
        
        # Human proximity validation
        if human_positions is not None:
            result = self.safety_validator.validate_human_proximity(robot_trajectory, human_positions)
            results.append(result)
        
        # Collision risk validation
        if obstacles is not None:
            result = self.safety_validator.validate_collision_risk(robot_trajectory, obstacles)
            results.append(result)
        
        # Store results
        self.validation_results.extend(results)
        
        logger.info(f"Safety validation completed: {sum(1 for r in results if r.status == ValidationStatus.PASSED)}/{len(results)} tests passed")
        
        return results
    
    def run_clinical_validation(self, demonstrations: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Run clinical task validation."""
        logger.info("Starting clinical validation...")
        
        results = []
        
        # Medication handling validation
        result = self.clinical_validator.validate_medication_handling(demonstrations)
        results.append(result)
        
        # Extract grasp attempts from demonstrations
        grasp_attempts = []
        for demo in demonstrations:
            if 'grasp_attempts' in demo:
                grasp_attempts.extend(demo['grasp_attempts'])
        
        if grasp_attempts:
            result = self.clinical_validator.validate_grasp_success(grasp_attempts)
            results.append(result)
        
        # Extract sterility checks
        sterility_checks = []
        for demo in demonstrations:
            if 'sterility_checks' in demo:
                sterility_checks.extend(demo['sterility_checks'])
        
        if sterility_checks:
            result = self.clinical_validator.validate_sterility_maintenance(sterility_checks)
            results.append(result)
        
        # Store results
        self.validation_results.extend(results)
        
        logger.info(f"Clinical validation completed: {sum(1 for r in results if r.status == ValidationStatus.PASSED)}/{len(results)} tests passed")
        
        return results
    
    def run_performance_validation(self, response_times: List[float],
                                 system_metrics: List[Dict[str, Any]]) -> List[ValidationResult]:
        """Run performance validation."""
        logger.info("Starting performance validation...")
        
        results = []
        
        # Response time validation
        if response_times:
            result = self.performance_validator.validate_response_time(response_times)
            results.append(result)
        
        # System stability validation
        if system_metrics:
            result = self.performance_validator.validate_system_stability(system_metrics)
            results.append(result)
        
        # Store results
        self.validation_results.extend(results)
        
        logger.info(f"Performance validation completed: {sum(1 for r in results if r.status == ValidationStatus.PASSED)}/{len(results)} tests passed")
        
        return results
    
    def run_comprehensive_validation(self, test_data: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
        """Run comprehensive validation suite."""
        logger.info("Starting comprehensive validation...")
        
        all_results = {}
        
        # Safety validation
        if 'robot_trajectory' in test_data:
            safety_results = self.run_safety_validation(
                test_data['robot_trajectory'],
                test_data.get('human_positions'),
                test_data.get('obstacles')
            )
            all_results['safety'] = safety_results
        
        # Clinical validation
        if 'demonstrations' in test_data:
            clinical_results = self.run_clinical_validation(test_data['demonstrations'])
            all_results['clinical'] = clinical_results
        
        # Performance validation
        if 'response_times' in test_data or 'system_metrics' in test_data:
            performance_results = self.run_performance_validation(
                test_data.get('response_times', []),
                test_data.get('system_metrics', [])
            )
            all_results['performance'] = performance_results
        
        # Generate report
        self.generate_validation_report(all_results)
        
        logger.info("Comprehensive validation completed")
        
        return all_results
    
    def generate_validation_report(self, results: Dict[str, List[ValidationResult]]) -> str:
        """Generate comprehensive validation report."""
        report_path = self.output_dir / f"validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        
        # Calculate summary statistics
        total_tests = sum(len(test_results) for test_results in results.values())
        passed_tests = sum(1 for test_results in results.values() 
                          for result in test_results if result.status == ValidationStatus.PASSED)
        failed_tests = sum(1 for test_results in results.values() 
                          for result in test_results if result.status == ValidationStatus.FAILED)
        error_tests = sum(1 for test_results in results.values() 
                         for result in test_results if result.status == ValidationStatus.ERROR)
        
        overall_score = np.mean([result.score for test_results in results.values() 
                                 for result in test_results if result.status != ValidationStatus.ERROR])
        
        # Generate markdown report
        report_content = f"""# Clinical Robot Physical Validation Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Tests**: {total_tests}
- **Passed**: {passed_tests} ({passed_tests/total_tests*100:.1f}%)
- **Failed**: {failed_tests} ({failed_tests/total_tests*100:.1f}%)
- **Errors**: {error_tests} ({error_tests/total_tests*100:.1f}%)
- **Overall Score**: {overall_score:.3f}

## Test Results by Category

"""
        
        for category, test_results in results.items():
            category_passed = sum(1 for r in test_results if r.status == ValidationStatus.PASSED)
            category_total = len(test_results)
            category_score = np.mean([r.score for r in test_results if r.status != ValidationStatus.ERROR])
            
            report_content += f"""
### {category.title()} Validation

- **Tests Passed**: {category_passed}/{category_total} ({category_passed/category_total*100:.1f}%)
- **Category Score**: {category_score:.3f}

| Test Name | Status | Score | Duration (s) | Violations |
|-----------|--------|-------|-------------|------------|
"""
            
            for result in test_results:
                violations_count = len(result.safety_violations)
                report_content += f"| {result.test_name} | {result.status.value} | {result.score:.3f} | {result.duration:.2f} | {violations_count} |\n"
            
            # Add detailed metrics for failed tests
            failed_results = [r for r in test_results if r.status == ValidationStatus.FAILED]
            if failed_results:
                report_content += f"\n#### Failed Tests Details\n\n"
                for result in failed_results:
                    report_content += f"**{result.test_name}**\n"
                    report_content += f"- Score: {result.score:.3f}\n"
                    report_content += f"- Duration: {result.duration:.2f}s\n"
                    
                    if result.safety_violations:
                        report_content += "- Violations:\n"
                        for violation in result.safety_violations:
                            report_content += f"  - {violation}\n"
                    
                    if result.errors:
                        report_content += "- Errors:\n"
                        for error in result.errors:
                            report_content += f"  - {error}\n"
                    
                    report_content += "\n"
        
        # Add recommendations
        report_content += """
## Recommendations

"""
        
        if failed_tests > 0:
            report_content += """
### High Priority

1. **Address Safety Violations**: Review and fix all safety-related test failures
2. **Improve Clinical Performance**: Focus on failed clinical validation tests
3. **System Optimization**: Address performance bottlenecks identified

"""
        
        if overall_score < 0.8:
            report_content += """
### Medium Priority

1. **Model Retraining**: Consider retraining models with better clinical data
2. **Safety Thresholds**: Review and adjust safety thresholds if too restrictive
3. **System Calibration**: Recalibrate sensors and actuators

"""
        
        report_content += f"""
## Technical Details

### Configuration
```json
{json.dumps(self.config.to_dict(), indent=2)}
```

### Test Environment
- **Sampling Rate**: {self.config.sampling_rate} Hz
- **Test Duration**: {self.config.test_duration} seconds
- **Safety Thresholds**: Applied according to clinical standards

---
*Report generated by Clinical Robot Physical Validation Framework*
"""
        
        # Save report
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        # Generate plots if requested
        if self.config.generate_plots:
            self.generate_validation_plots(results)
        
        logger.info(f"Validation report saved to: {report_path}")
        
        return str(report_path)
    
    def generate_validation_plots(self, results: Dict[str, List[ValidationResult]]):
        """Generate validation plots."""
        # Create summary plot
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Test Status Distribution', 'Scores by Category', 
                          'Test Durations', 'Safety Violations'],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Collect data for plotting
        all_results = [result for test_results in results.values() for result in test_results]
        
        # Test status distribution
        status_counts = {}
        for result in all_results:
            status = result.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(status_counts.keys()),
                values=list(status_counts.values()),
                name="Test Status"
            ),
            row=1, col=1
        )
        
        # Scores by category
        category_scores = {}
        for category, test_results in results.items():
            scores = [r.score for r in test_results if r.status != ValidationStatus.ERROR]
            if scores:
                category_scores[category] = np.mean(scores)
        
        fig.add_trace(
            go.Bar(
                x=list(category_scores.keys()),
                y=list(category_scores.values()),
                name="Average Score"
            ),
            row=1, col=2
        )
        
        # Test durations
        test_names = [r.test_name for r in all_results if r.status != ValidationStatus.ERROR]
        durations = [r.duration for r in all_results if r.status != ValidationStatus.ERROR]
        
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=durations,
                name="Duration (s)"
            ),
            row=2, col=1
        )
        
        # Safety violations
        violation_counts = [len(r.safety_violations) for r in all_results if r.status != ValidationStatus.ERROR]
        
        fig.add_trace(
            go.Bar(
                x=test_names,
                y=violation_counts,
                name="Violations"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            title="Clinical Robot Validation Summary",
            height=800,
            showlegend=False
        )
        
        # Save plot
        plot_path = self.output_dir / f"validation_plots_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        fig.write_html(str(plot_path))
        
        logger.info(f"Validation plots saved to: {plot_path}")
    
    def export_results(self, format: str = "json"):
        """Export validation results."""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if format == "json":
            # Export as JSON
            results_data = {
                'timestamp': timestamp,
                'config': self.config.to_dict(),
                'results': [r.to_dict() for r in self.validation_results]
            }
            
            output_path = self.output_dir / f"validation_results_{timestamp}.json"
            with open(output_path, 'w') as f:
                json.dump(results_data, f, indent=2)
        
        elif format == "csv":
            # Export as CSV
            df_data = []
            for result in self.validation_results:
                row = result.to_dict()
                # Flatten metrics
                for key, value in row['metrics'].items():
                    row[f'metric_{key}'] = value
                del row['metrics']
                df_data.append(row)
            
            df = pd.DataFrame(df_data)
            output_path = self.output_dir / f"validation_results_{timestamp}.csv"
            df.to_csv(output_path, index=False)
        
        logger.info(f"Results exported to: {output_path}")

def main():
    """Main function for running validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Robot Physical Validation')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--data', type=str, required=True, help='Test data file')
    parser.add_argument('--output', type=str, default='./validation_results', help='Output directory')
    parser.add_argument('--format', nargs='+', default=['json', 'csv'], help='Export formats')
    
    args = parser.parse_args()
    
    # Load configuration
    config = ValidationConfig()
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Load test data
    with open(args.data, 'r') as f:
        test_data = json.load(f)
    
    # Convert to numpy arrays where needed
    if 'robot_trajectory' in test_data:
        test_data['robot_trajectory'] = np.array(test_data['robot_trajectory'])
    
    if 'human_positions' in test_data:
        test_data['human_positions'] = np.array(test_data['human_positions'])
    
    # Initialize validation framework
    validator = PhysicalValidationFramework(config, args.output)
    
    # Run validation
    results = validator.run_comprehensive_validation(test_data)
    
    # Export results
    for fmt in args.format:
        validator.export_results(fmt)
    
    logger.info("Physical validation completed successfully!")

if __name__ == "__main__":
    main()
