#!/usr/bin/env python3
"""
Clinical Workflow Orchestration System
Comprehensive workflow management for clinical robot operations.

This module implements:
- Clinical workflow definition and execution
- Task scheduling and dependency management
- Human-robot collaboration workflows
- Emergency response protocols
- Medication dispensing workflows
- Patient care workflows
- Workflow monitoring and analytics
- Compliance checking and validation

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
from collections import defaultdict, deque
from enum import Enum
import warnings

# Workflow and scheduling
import schedule
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.jobstores.sqlalchemy import SQLAlchemyJobStore

# Data handling
import pandas as pd
import numpy as np

# Monitoring
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('clinical_workflow.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"

class TaskPriority(Enum):
    """Task priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"

class WorkflowType(Enum):
    """Types of clinical workflows."""
    MEDICATION_DISPENSING = "medication_dispensing"
    PATIENT_CARE = "patient_care"
    EMERGENCY_RESPONSE = "emergency_response"
    STERILIZATION = "sterilization"
    INVENTORY_MANAGEMENT = "inventory_management"
    EQUIPMENT_MAINTENANCE = "equipment_maintenance"

@dataclass
class WorkflowConfig:
    """Configuration for workflow orchestration."""
    
    # Scheduler configuration
    scheduler_backend: str = "sqlalchemy"
    scheduler_db_url: str = "sqlite:///workflow_scheduler.db"
    max_concurrent_workflows: int = 5
    workflow_timeout: int = 3600  # seconds
    
    # Task configuration
    default_task_timeout: int = 300  # seconds
    max_retries: int = 3
    retry_delay: int = 60  # seconds
    
    # Monitoring configuration
    enable_monitoring: bool = True
    monitoring_interval: int = 30  # seconds
    log_retention_days: int = 30
    
    # Compliance configuration
    enable_compliance_checking: bool = True
    compliance_rules_path: str = "./compliance_rules.json"
    
    # Notification configuration
    enable_notifications: bool = True
    notification_channels: List[str] = field(default_factory=lambda: ["email", "slack"])
    
    # Analytics configuration
    enable_analytics: bool = True
    analytics_retention_days: int = 90
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'scheduler_backend': self.scheduler_backend,
            'scheduler_db_url': self.scheduler_db_url,
            'max_concurrent_workflows': self.max_concurrent_workflows,
            'workflow_timeout': self.workflow_timeout,
            'default_task_timeout': self.default_task_timeout,
            'max_retries': self.max_retries,
            'retry_delay': self.retry_delay,
            'enable_monitoring': self.enable_monitoring,
            'monitoring_interval': self.monitoring_interval,
            'log_retention_days': self.log_retention_days,
            'enable_compliance_checking': self.compliance_rules_path,
            'compliance_rules_path': self.compliance_rules_path,
            'enable_notifications': self.enable_notifications,
            'notification_channels': self.notification_channels,
            'enable_analytics': self.enable_analytics,
            'analytics_retention_days': self.analytics_retention_days
        }

@dataclass
class Task:
    """Individual task within a workflow."""
    id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    timeout: int
    dependencies: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    status: WorkflowStatus = WorkflowStatus.PENDING
    result: Any = None
    error: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'task_type': self.task_type,
            'priority': self.priority.value,
            'timeout': self.timeout,
            'dependencies': self.dependencies,
            'parameters': self.parameters,
            'retry_count': self.retry_count,
            'status': self.status.value,
            'result': str(self.result) if self.result is not None else None,
            'error': self.error,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None
        }

@dataclass
class Workflow:
    """Clinical workflow definition."""
    id: str
    name: str
    description: str
    workflow_type: WorkflowType
    priority: TaskPriority
    tasks: List[Task]
    status: WorkflowStatus = WorkflowStatus.PENDING
    current_task_index: int = 0
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'workflow_type': self.workflow_type.value,
            'priority': self.priority.value,
            'tasks': [task.to_dict() for task in self.tasks],
            'status': self.status.value,
            'current_task_index': self.current_task_index,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'metadata': self.metadata
        }

class ComplianceChecker:
    """Compliance checking for clinical workflows."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.rules = self._load_compliance_rules()
    
    def _load_compliance_rules(self) -> Dict[str, Any]:
        """Load compliance rules from file."""
        if Path(self.config.compliance_rules_path).exists():
            with open(self.config.compliance_rules_path, 'r') as f:
                return json.load(f)
        else:
            # Default compliance rules
            return {
                'medication_dispensing': {
                    'require_verification': True,
                    'require_double_check': True,
                    'max_dispensing_time': 300,
                    'require_patient_identification': True
                },
                'patient_care': {
                    'require_supervision': True,
                    'max_task_duration': 600,
                    'require_safety_check': True
                },
                'emergency_response': {
                    'max_response_time': 30,
                    'require_immediate_action': True,
                    'override_normal_procedures': True
                }
            }
    
    def check_workflow_compliance(self, workflow: Workflow) -> Tuple[bool, List[str]]:
        """Check if workflow complies with rules."""
        violations = []
        
        # Get rules for workflow type
        rules = self.rules.get(workflow.workflow_type.value, {})
        
        # Check each rule
        for rule_name, rule_value in rules.items():
            if not self._check_rule(workflow, rule_name, rule_value):
                violations.append(f"Rule violation: {rule_name}")
        
        return len(violations) == 0, violations
    
    def _check_rule(self, workflow: Workflow, rule_name: str, rule_value: Any) -> bool:
        """Check individual compliance rule."""
        # Simplified rule checking
        if rule_name == 'require_verification':
            return 'verification' in workflow.metadata
        elif rule_name == 'require_double_check':
            return workflow.metadata.get('double_check', False)
        elif rule_name == 'max_dispensing_time':
            return workflow.metadata.get('estimated_time', 0) <= rule_value
        elif rule_name == 'require_patient_identification':
            return 'patient_id' in workflow.metadata
        elif rule_name == 'require_supervision':
            return workflow.metadata.get('supervised', False)
        elif rule_name == 'max_task_duration':
            return workflow.metadata.get('estimated_time', 0) <= rule_value
        elif rule_name == 'require_safety_check':
            return workflow.metadata.get('safety_check', False)
        elif rule_name == 'max_response_time':
            return workflow.metadata.get('response_time', 0) <= rule_value
        elif rule_name == 'require_immediate_action':
            return workflow.priority == TaskPriority.CRITICAL
        elif rule_name == 'override_normal_procedures':
            return workflow.workflow_type == WorkflowType.EMERGENCY_RESPONSE
        
        return True

class WorkflowExecutor:
    """Executes individual tasks within a workflow."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        self.robot_interface = None
        self.compliance_checker = ComplianceChecker(config)
    
    def execute_task(self, task: Task) -> Tuple[bool, Any]:
        """Execute a single task."""
        logger.info(f"Executing task: {task.name}")
        
        task.start_time = datetime.now()
        task.status = WorkflowStatus.RUNNING
        
        try:
            # Execute based on task type
            if task.task_type == "robot_action":
                result = self._execute_robot_action(task)
            elif task.task_type == "human_verification":
                result = self._execute_human_verification(task)
            elif task.task_type == "data_collection":
                result = self._execute_data_collection(task)
            elif task.task_type == "safety_check":
                result = self._execute_safety_check(task)
            elif task.task_type == "notification":
                result = self._execute_notification(task)
            else:
                result = self._execute_generic_task(task)
            
            task.result = result
            task.status = WorkflowStatus.COMPLETED
            task.end_time = datetime.now()
            
            return True, result
            
        except Exception as e:
            task.error = str(e)
            task.status = WorkflowStatus.FAILED
            task.end_time = datetime.now()
            
            logger.error(f"Task execution failed: {e}")
            return False, None
    
    def _execute_robot_action(self, task: Task) -> Any:
        """Execute robot action task."""
        # Placeholder for robot action execution
        # In real implementation, this would interface with the robot
        action_type = task.parameters.get('action_type', 'move')
        target = task.parameters.get('target', [0, 0, 0])
        
        logger.info(f"Executing robot action: {action_type} to {target}")
        
        # Simulate execution time
        time.sleep(1.0)
        
        return {"action": action_type, "target": target, "success": True}
    
    def _execute_human_verification(self, task: Task) -> Any:
        """Execute human verification task."""
        verification_type = task.parameters.get('verification_type', 'visual')
        
        logger.info(f"Executing human verification: {verification_type}")
        
        # Simulate verification
        time.sleep(2.0)
        
        return {"verified": True, "verification_type": verification_type}
    
    def _execute_data_collection(self, task: Task) -> Any:
        """Execute data collection task."""
        data_type = task.parameters.get('data_type', 'sensor')
        
        logger.info(f"Collecting data: {data_type}")
        
        # Simulate data collection
        time.sleep(0.5)
        
        return {"data_collected": True, "data_type": data_type}
    
    def _execute_safety_check(self, task: Task) -> Any:
        """Execute safety check task."""
        check_type = task.parameters.get('check_type', 'workspace')
        
        logger.info(f"Performing safety check: {check_type}")
        
        # Simulate safety check
        time.sleep(1.0)
        
        return {"safe": True, "check_type": check_type}
    
    def _execute_notification(self, task: Task) -> Any:
        """Execute notification task."""
        message = task.parameters.get('message', '')
        recipients = task.parameters.get('recipients', [])
        
        logger.info(f"Sending notification to {recipients}: {message}")
        
        # Simulate notification
        time.sleep(0.5)
        
        return {"sent": True, "recipients": recipients}
    
    def _execute_generic_task(self, task: Task) -> Any:
        """Execute generic task."""
        logger.info(f"Executing generic task: {task.name}")
        
        # Simulate execution
        time.sleep(1.0)
        
        return {"completed": True}

class ClinicalWorkflowOrchestrator:
    """Main workflow orchestration system."""
    
    def __init__(self, config: WorkflowConfig):
        self.config = config
        
        # Initialize scheduler
        self.scheduler = self._initialize_scheduler()
        
        # Initialize executor
        self.executor = WorkflowExecutor(config)
        
        # Workflow storage
        self.workflows = {}
        self.workflow_history = []
        
        # Monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        # Analytics
        self.workflow_analytics = defaultdict(list)
        
        logger.info("Clinical Workflow Orchestrator initialized")
    
    def _initialize_scheduler(self) -> BackgroundScheduler:
        """Initialize the workflow scheduler."""
        jobstores = {
            'default': SQLAlchemyJobStore(url=self.config.scheduler_db_url)
        }
        
        scheduler = BackgroundScheduler(
            jobstores=jobstores,
            timezone='UTC'
        )
        
        scheduler.start()
        return scheduler
    
    def create_workflow(self, workflow: Workflow) -> bool:
        """Create and register a new workflow."""
        try:
            # Check compliance
            if self.config.enable_compliance_checking:
                is_compliant, violations = self.executor.compliance_checker.check_workflow_compliance(workflow)
                if not is_compliant:
                    logger.error(f"Workflow compliance check failed: {violations}")
                    return False
            
            # Store workflow
            self.workflows[workflow.id] = workflow
            
            # Schedule workflow execution
            self.scheduler.add_job(
                self._execute_workflow,
                'date',
                run_date=datetime.now() + timedelta(seconds=1),
                args=[workflow.id],
                id=workflow.id,
                replace_existing=True
            )
            
            logger.info(f"Workflow created: {workflow.id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create workflow: {e}")
            return False
    
    def _execute_workflow(self, workflow_id: str):
        """Execute a workflow."""
        if workflow_id not in self.workflows:
            logger.error(f"Workflow not found: {workflow_id}")
            return
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.RUNNING
        workflow.start_time = datetime.now()
        
        logger.info(f"Starting workflow execution: {workflow_id}")
        
        try:
            # Execute tasks in order
            for i, task in enumerate(workflow.tasks):
                workflow.current_task_index = i
                
                # Check dependencies
                if not self._check_task_dependencies(task, workflow):
                    logger.error(f"Task dependencies not met: {task.id}")
                    workflow.status = WorkflowStatus.FAILED
                    return
                
                # Execute task
                success, result = self.executor.execute_task(task)
                
                if not success:
                    # Retry if configured
                    if task.retry_count < self.config.max_retries:
                        task.retry_count += 1
                        logger.warning(f"Retrying task {task.id} (attempt {task.retry_count})")
                        time.sleep(self.config.retry_delay)
                        
                        # Retry task
                        success, result = self.executor.execute_task(task)
                    
                    if not success:
                        logger.error(f"Task execution failed: {task.id}")
                        workflow.status = WorkflowStatus.FAILED
                        return
            
            workflow.status = WorkflowStatus.COMPLETED
            workflow.end_time = datetime.now()
            
            # Add to history
            self.workflow_history.append(workflow)
            
            # Update analytics
            self._update_analytics(workflow)
            
            logger.info(f"Workflow completed: {workflow_id}")
            
        except Exception as e:
            logger.error(f"Workflow execution error: {e}")
            workflow.status = WorkflowStatus.FAILED
            workflow.end_time = datetime.now()
    
    def _check_task_dependencies(self, task: Task, workflow: Workflow) -> bool:
        """Check if task dependencies are satisfied."""
        for dep_id in task.dependencies:
            # Find dependency task
            dep_task = next((t for t in workflow.tasks if t.id == dep_id), None)
            
            if dep_task is None:
                logger.error(f"Dependency not found: {dep_id}")
                return False
            
            # Check if dependency is completed
            if dep_task.status != WorkflowStatus.COMPLETED:
                return False
        
        return True
    
    def pause_workflow(self, workflow_id: str) -> bool:
        """Pause a running workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            logger.info(f"Workflow paused: {workflow_id}")
            return True
        
        return False
    
    def resume_workflow(self, workflow_id: str) -> bool:
        """Resume a paused workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status == WorkflowStatus.PAUSED:
            # Reschedule workflow
            self.scheduler.add_job(
                self._execute_workflow,
                'date',
                run_date=datetime.now() + timedelta(seconds=1),
                args=[workflow_id],
                id=workflow_id,
                replace_existing=True
            )
            
            logger.info(f"Workflow resumed: {workflow_id}")
            return True
        
        return False
    
    def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a workflow."""
        if workflow_id not in self.workflows:
            return False
        
        workflow = self.workflows[workflow_id]
        
        # Remove from scheduler
        try:
            self.scheduler.remove_job(workflow_id)
        except:
            pass
        
        workflow.status = WorkflowStatus.CANCELLED
        workflow.end_time = datetime.now()
        
        logger.info(f"Workflow cancelled: {workflow_id}")
        return True
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow."""
        if workflow_id not in self.workflows:
            return None
        
        workflow = self.workflows[workflow_id]
        return workflow.to_dict()
    
    def list_workflows(self, status: WorkflowStatus = None) -> List[Dict[str, Any]]:
        """List all workflows, optionally filtered by status."""
        workflows = []
        
        for workflow in self.workflows.values():
            if status is None or workflow.status == status:
                workflows.append(workflow.to_dict())
        
        return workflows
    
    def _update_analytics(self, workflow: Workflow):
        """Update workflow analytics."""
        duration = (workflow.end_time - workflow.start_time).total_seconds()
        
        self.workflow_analytics['workflow_durations'].append(duration)
        self.workflow_analytics['workflow_types'].append(workflow.workflow_type.value)
        self.workflow_analytics['workflow_statuses'].append(workflow.status.value)
        
        # Task analytics
        for task in workflow.tasks:
            if task.start_time and task.end_time:
                task_duration = (task.end_time - task.start_time).total_seconds()
                self.workflow_analytics['task_durations'].append(task_duration)
                self.workflow_analytics['task_types'].append(task.task_type)
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get workflow analytics."""
        return {
            'total_workflows': len(self.workflow_history),
            'workflow_durations': self.workflow_analytics['workflow_durations'],
            'workflow_types': self.workflow_analytics['workflow_types'],
            'workflow_statuses': self.workflow_analytics['workflow_statuses'],
            'task_durations': self.workflow_analytics['task_durations'],
            'task_types': self.workflow_analytics['task_types']
        }
    
    def start_monitoring(self):
        """Start workflow monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Workflow monitoring started")
    
    def stop_monitoring(self):
        """Stop workflow monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Workflow monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop for workflows."""
        while self.is_monitoring:
            try:
                # Check for stuck workflows
                current_time = datetime.now()
                
                for workflow in self.workflows.values():
                    if workflow.status == WorkflowStatus.RUNNING:
                        # Check timeout
                        if (current_time - workflow.start_time).total_seconds() > self.config.workflow_timeout:
                            logger.warning(f"Workflow timeout: {workflow.id}")
                            self.cancel_workflow(workflow.id)
                
                # Check system resources
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                if cpu_usage > 90 or memory_usage > 90:
                    logger.warning(f"High resource usage: CPU {cpu_usage}%, Memory {memory_usage}%")
                
                time.sleep(self.config.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def shutdown(self):
        """Shutdown the orchestrator."""
        # Stop monitoring
        self.stop_monitoring()
        
        # Shutdown scheduler
        self.scheduler.shutdown()
        
        logger.info("Workflow Orchestrator shutdown")

# Predefined workflow templates

def create_medication_dispensing_workflow(patient_id: str, medications: List[str]) -> Workflow:
    """Create a medication dispensing workflow."""
    tasks = [
        Task(
            id="verify_prescription",
            name="Verify Prescription",
            description="Verify patient prescription",
            task_type="human_verification",
            priority=TaskPriority.HIGH,
            timeout=120,
            parameters={
                'verification_type': 'prescription',
                'patient_id': patient_id
            }
        ),
        Task(
            id="identify_patient",
            name="Identify Patient",
            description="Identify patient before dispensing",
            task_type="human_verification",
            priority=TaskPriority.CRITICAL,
            timeout=60,
            parameters={
                'verification_type': 'patient_id',
                'patient_id': patient_id
            },
            dependencies=["verify_prescription"]
        ),
        Task(
            id="safety_check",
            name="Safety Check",
            description="Perform safety check before dispensing",
            task_type="safety_check",
            priority=TaskPriority.HIGH,
            timeout=30,
            parameters={
                'check_type': 'workspace'
            },
            dependencies=["identify_patient"]
        ),
        Task(
            id="retrieve_medication",
            name="Retrieve Medication",
            description="Retrieve medication from storage",
            task_type="robot_action",
            priority=TaskPriority.HIGH,
            timeout=180,
            parameters={
                'action_type': 'retrieve',
                'medications': medications
            },
            dependencies=["safety_check"]
        ),
        Task(
            id="verify_medication",
            name="Verify Medication",
            description="Double-check medication before delivery",
            task_type="human_verification",
            priority=TaskPriority.CRITICAL,
            timeout=60,
            parameters={
                'verification_type': 'medication',
                'medications': medications
            },
            dependencies=["retrieve_medication"]
        ),
        Task(
            id="deliver_medication",
            name="Deliver Medication",
            description="Deliver medication to patient",
            task_type="robot_action",
            priority=TaskPriority.HIGH,
            timeout=120,
            parameters={
                'action_type': 'deliver',
                'patient_id': patient_id
            },
            dependencies=["verify_medication"]
        ),
        Task(
            id="notify_completion",
            name="Notify Completion",
            description="Notify staff of completion",
            task_type="notification",
            priority=TaskPriority.MEDIUM,
            timeout=30,
            parameters={
                'message': f'Medication dispensing completed for patient {patient_id}',
                'recipients': ['pharmacy_staff', 'nursing_station']
            },
            dependencies=["deliver_medication"]
        )
    ]
    
    workflow = Workflow(
        id=f"med_dispensing_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name="Medication Dispensing",
        description="Dispense medication to patient",
        workflow_type=WorkflowType.MEDICATION_DISPENSING,
        priority=TaskPriority.HIGH,
        tasks=tasks,
        metadata={
            'patient_id': patient_id,
            'medications': medications,
            'verification': True,
            'double_check': True,
            'estimated_time': 600
        }
    )
    
    return workflow

def create_emergency_response_workflow(emergency_type: str, location: Tuple[float, float, float]) -> Workflow:
    """Create an emergency response workflow."""
    tasks = [
        Task(
            id="assess_situation",
            name="Assess Situation",
            description="Assess emergency situation",
            task_type="safety_check",
            priority=TaskPriority.CRITICAL,
            timeout=30,
            parameters={
                'check_type': 'emergency',
                'emergency_type': emergency_type
            }
        ),
        Task(
            id="notify_emergency",
            name="Notify Emergency Response",
            description="Notify emergency response team",
            task_type="notification",
            priority=TaskPriority.CRITICAL,
            timeout=15,
            parameters={
                'message': f'Emergency: {emergency_type} at {location}',
                'recipients': ['emergency_team', 'supervisor']
            },
            dependencies=["assess_situation"]
        ),
        Task(
            id="execute_emergency_action",
            name="Execute Emergency Action",
            description="Execute emergency response action",
            task_type="robot_action",
            priority=TaskPriority.CRITICAL,
            timeout=60,
            parameters={
                'action_type': 'emergency_response',
                'emergency_type': emergency_type,
                'location': location
            },
            dependencies=["notify_emergency"]
        )
    ]
    
    workflow = Workflow(
        id=f"emergency_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        name="Emergency Response",
        description=f"Respond to {emergency_type} emergency",
        workflow_type=WorkflowType.EMERGENCY_RESPONSE,
        priority=TaskPriority.CRITICAL,
        tasks=tasks,
        metadata={
            'emergency_type': emergency_type,
            'location': location,
            'response_time': 30,
            'immediate_action': True
        }
    )
    
    return workflow

def main():
    """Main function for workflow orchestration."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clinical Workflow Orchestrator')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--workflow', type=str, help='Workflow type to execute')
    parser.add_argument('--patient-id', type=str, help='Patient ID for medication dispensing')
    parser.add_argument('--medications', nargs='+', help='Medications to dispense')
    
    args = parser.parse_args()
    
    # Load configuration
    config = WorkflowConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create orchestrator
    orchestrator = ClinicalWorkflowOrchestrator(config)
    
    # Start monitoring
    orchestrator.start_monitoring()
    
    try:
        if args.workflow == "medication_dispensing":
            if args.patient_id and args.medications:
                workflow = create_medication_dispensing_workflow(args.patient_id, args.medications)
                orchestrator.create_workflow(workflow)
                print(f"Medication dispensing workflow created for patient {args.patient_id}")
            else:
                print("Patient ID and medications required for medication dispensing")
        
        elif args.workflow == "emergency_response":
            workflow = create_emergency_response_workflow("fall", (0.5, 0.3, 0.0))
            orchestrator.create_workflow(workflow)
            print("Emergency response workflow created")
        
        else:
            print(f"Unknown workflow type: {args.workflow}")
            print("Available workflows: medication_dispensing, emergency_response")
        
        # Keep running
        print("Orchestrator running. Press Ctrl+C to stop.")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\nShutting down orchestrator")
    finally:
        orchestrator.shutdown()

if __name__ == "__main__":
    main()
