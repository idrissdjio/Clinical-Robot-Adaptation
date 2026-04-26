#!/usr/bin/env python3
"""
Clinical Robot Adaptation API Server
Real-time inference and management API for clinical robot deployment.

This FastAPI server provides:
- Real-time model inference endpoints
- Clinical data management APIs
- Model training and evaluation endpoints
- Safety monitoring and alerts
- Performance analytics and visualization
- Multi-modal data processing pipelines

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import warnings

# FastAPI and web framework
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Core ML and data processing
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import cv2

# Database and caching
import redis
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import pickle

# Monitoring and analytics
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
import wandb

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_server.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Initialize FastAPI app
app = FastAPI(
    title="Clinical Robot Adaptation API",
    description="Real-time inference and management API for clinical robot deployment",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Prometheus metrics
inference_counter = Counter('inference_requests_total', 'Total inference requests', ['model', 'status'])
inference_duration = Histogram('inference_duration_seconds', 'Inference duration')
active_connections = Gauge('active_connections', 'Active WebSocket connections')
system_memory = Gauge('system_memory_bytes', 'System memory usage')
cpu_usage = Gauge('cpu_usage_percent', 'CPU usage percentage')

# Database setup
Base = declarative_base()
engine = create_engine('sqlite:///clinical_robot.db')
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

class InferenceRecord(Base):
    __tablename__ = "inference_records"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)
    input_data_hash = Column(String)
    prediction = Column(Text)
    confidence = Column(Float)
    processing_time = Column(Float)
    safety_score = Column(Float)
    success = Column(Boolean)

class ModelMetrics(Base):
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_version = Column(String)
    accuracy = Column(Float)
    precision = Column(Float)
    recall = Column(Float)
    f1_score = Column(Float)
    safety_score = Column(Float)
    throughput = Column(Float)

# Create tables
Base.metadata.create_all(bind=engine)

# Pydantic models
class InferenceRequest(BaseModel):
    """Request model for inference endpoints."""
    model_version: str = Field(default="latest", description="Model version to use")
    image_data: Optional[str] = Field(None, description="Base64 encoded image data")
    robot_state: Optional[List[float]] = Field(None, description="Robot joint states and pose")
    instruction: Optional[str] = Field(None, description="Natural language instruction")
    safety_level: Optional[str] = Field(default="medium", description="Safety level (low/medium/high)")
    context: Optional[Dict[str, Any]] = Field(default={}, description="Additional context")

class InferenceResponse(BaseModel):
    """Response model for inference endpoints."""
    success: bool
    prediction: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None
    safety_score: Optional[float] = None
    processing_time: Optional[float] = None
    model_version: str
    timestamp: str
    warnings: Optional[List[str]] = None

class TrainingRequest(BaseModel):
    """Request model for training endpoints."""
    model_version: str = Field(description="Base model version")
    dataset_path: str = Field(description="Path to training dataset")
    num_epochs: int = Field(default=100, description="Number of training epochs")
    learning_rate: float = Field(default=1e-4, description="Learning rate")
    batch_size: int = Field(default=16, description="Batch size")
    safety_weight: float = Field(default=0.3, description="Safety loss weight")
    human_weight: float = Field(default=0.2, description="Human awareness loss weight")

class TrainingStatus(BaseModel):
    """Training status response model."""
    status: str  # running, completed, failed, paused
    progress: float  # 0.0 to 1.0
    current_epoch: int
    total_epochs: int
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    safety_score: Optional[float] = None
    eta_seconds: Optional[int] = None
    logs: Optional[List[str]] = None

class SafetyAlert(BaseModel):
    """Safety alert model."""
    alert_type: str
    severity: str  # low, medium, high, critical
    message: str
    timestamp: str
    robot_id: Optional[str] = None
    location: Optional[Dict[str, float]] = None
    resolved: bool = False

class PerformanceMetrics(BaseModel):
    """Performance metrics model."""
    timestamp: str
    throughput: float  # requests per second
    latency_p50: float  # milliseconds
    latency_p95: float  # milliseconds
    latency_p99: float  # milliseconds
    error_rate: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None

# Global state
class GlobalState:
    def __init__(self):
        self.model_cache = {}
        self.active_connections = set()
        self.training_jobs = {}
        self.safety_alerts = []
        self.performance_history = []
        
    def add_connection(self, websocket):
        self.active_connections.add(websocket)
        active_connections.set(len(self.active_connections))
        
    def remove_connection(self, websocket):
        self.active_connections.discard(websocket)
        active_connections.set(len(self.active_connections))

state = GlobalState()

# Clinical model wrapper
class ClinicalModelWrapper:
    """Wrapper for clinical robot adaptation model."""
    
    def __init__(self, model_path: str, config: Dict[str, Any]):
        self.model_path = model_path
        self.config = config
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.load_model()
        
    def load_model(self):
        """Load the clinical adaptation model."""
        try:
            # This would load the actual ClinicalOctoAdapter model
            # For now, we'll create a placeholder
            logger.info(f"Loading model from {self.model_path}")
            
            # Placeholder model loading
            self.model = torch.nn.Module()  # Placeholder
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def predict(self, request: InferenceRequest) -> Dict[str, Any]:
        """Run inference on the request."""
        start_time = time.time()
        
        try:
            # Preprocess inputs
            processed_inputs = self._preprocess_inputs(request)
            
            # Run model inference
            with torch.no_grad():
                # Placeholder inference
                predictions = {
                    'action': np.random.randn(7).tolist(),
                    'grasp_type': np.random.choice(['precision', 'power', 'pinch']),
                    'medication_type': np.random.choice(['vial', 'bottle', 'syringe']),
                    'confidence': np.random.uniform(0.8, 0.95)
                }
                
                safety_score = np.random.uniform(0.85, 0.98)
            
            processing_time = time.time() - start_time
            
            return {
                'predictions': predictions,
                'safety_score': safety_score,
                'processing_time': processing_time
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            raise
    
    def _preprocess_inputs(self, request: InferenceRequest) -> Dict[str, torch.Tensor]:
        """Preprocess inputs for model inference."""
        processed = {}
        
        # Process image data
        if request.image_data:
            # Decode base64 and preprocess
            image = self._decode_image(request.image_data)
            processed['image'] = self._preprocess_image(image)
        
        # Process robot state
        if request.robot_state:
            processed['robot_state'] = torch.tensor(request.robot_state, dtype=torch.float32)
        
        # Process instruction
        if request.instruction:
            processed['instruction'] = request.instruction
        
        return processed
    
    def _decode_image(self, base64_data: str) -> np.ndarray:
        """Decode base64 image data."""
        import base64
        image_data = base64.b64decode(base64_data)
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image
    
    def _preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """Preprocess image for model input."""
        # Resize, normalize, convert to tensor
        image = cv2.resize(image, (224, 224))
        image = image.astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image.unsqueeze(0)

# Load models on startup
def load_models():
    """Load available models on server startup."""
    models_dir = Path("models/trained")
    if models_dir.exists():
        for model_file in models_dir.glob("*.pt"):
            model_name = model_file.stem
            try:
                config = {"model_type": "clinical_octo"}
                wrapper = ClinicalModelWrapper(str(model_file), config)
                state.model_cache[model_name] = wrapper
                logger.info(f"Loaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")

# API Endpoints
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Clinical Robot Adaptation API",
        "version": "2.0.0",
        "description": "Real-time inference and management API for clinical robot deployment",
        "endpoints": {
            "inference": "/predict",
            "training": "/train",
            "safety": "/safety",
            "metrics": "/metrics",
            "monitoring": "/monitoring"
        },
        "models_loaded": list(state.model_cache.keys()),
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Run inference on clinical robot adaptation model."""
    start_time = time.time()
    
    try:
        # Get model
        model_name = request.model_version if request.model_version != "latest" else list(state.model_cache.keys())[0]
        
        if model_name not in state.model_cache:
            raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
        
        model = state.model_cache[model_name]
        
        # Run inference
        with inference_duration.time():
            result = model.predict(request)
        
        # Record metrics
        inference_counter.labels(model=model_name, status="success").inc()
        
        # Store in database
        db = SessionLocal()
        record = InferenceRecord(
            model_version=model_name,
            prediction=json.dumps(result['predictions']),
            confidence=result['predictions'].get('confidence', 0.0),
            processing_time=result['processing_time'],
            safety_score=result['safety_score'],
            success=True
        )
        db.add(record)
        db.commit()
        db.close()
        
        return InferenceResponse(
            success=True,
            prediction=result['predictions'],
            confidence=result['predictions'].get('confidence'),
            safety_score=result['safety_score'],
            processing_time=result['processing_time'],
            model_version=model_name,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        inference_counter.labels(model=model_name, status="error").inc()
        logger.error(f"Prediction failed: {e}")
        
        return InferenceResponse(
            success=False,
            model_version=request.model_version,
            timestamp=datetime.now().isoformat(),
            warnings=[str(e)]
        )

@app.post("/train", response_model=Dict[str, str])
async def start_training(request: TrainingRequest, background_tasks: BackgroundTasks):
    """Start model training in background."""
    job_id = f"train_{int(time.time())}"
    
    # Initialize training job
    state.training_jobs[job_id] = {
        "status": "starting",
        "progress": 0.0,
        "current_epoch": 0,
        "total_epochs": request.num_epochs,
        "start_time": time.time(),
        "request": request
    }
    
    # Start training in background
    background_tasks.add_task(run_training_job, job_id, request)
    
    return {"job_id": job_id, "status": "started"}

@app.get("/train/{job_id}", response_model=TrainingStatus)
async def get_training_status(job_id: str):
    """Get training job status."""
    if job_id not in state.training_jobs:
        raise HTTPException(status_code=404, detail="Training job not found")
    
    job = state.training_jobs[job_id]
    
    return TrainingStatus(
        status=job["status"],
        progress=job["progress"],
        current_epoch=job["current_epoch"],
        total_epochs=job["total_epochs"],
        loss=job.get("loss"),
        accuracy=job.get("accuracy"),
        safety_score=job.get("safety_score"),
        eta_seconds=job.get("eta_seconds"),
        logs=job.get("logs", [])
    )

@app.get("/safety/alerts", response_model=List[SafetyAlert])
async def get_safety_alerts():
    """Get current safety alerts."""
    return state.safety_alerts

@app.post("/safety/alerts", response_model=Dict[str, str])
async def create_safety_alert(alert: SafetyAlert):
    """Create new safety alert."""
    state.safety_alerts.append(alert)
    
    # Broadcast to WebSocket connections
    await broadcast_safety_alert(alert)
    
    return {"status": "alert_created", "alert_id": len(state.safety_alerts)}

@app.get("/metrics", response_class=JSONResponse)
async def get_metrics():
    """Get system performance metrics."""
    # Update system metrics
    system_memory.set(psutil.virtual_memory().used)
    cpu_usage.set(psutil.cpu_percent())
    
    # Calculate performance metrics
    recent_records = get_recent_inference_records(1000)
    
    if recent_records:
        processing_times = [r.processing_time for r in recent_records]
        latency_p50 = np.percentile(processing_times, 50) * 1000
        latency_p95 = np.percentile(processing_times, 95) * 1000
        latency_p99 = np.percentile(processing_times, 99) * 1000
        
        success_rate = sum(1 for r in recent_records if r.success) / len(recent_records)
        error_rate = 1.0 - success_rate
        
        throughput = len(recent_records) / 3600  # requests per hour
    else:
        latency_p50 = latency_p95 = latency_p99 = 0.0
        error_rate = 0.0
        throughput = 0.0
    
    metrics = PerformanceMetrics(
        timestamp=datetime.now().isoformat(),
        throughput=throughput,
        latency_p50=latency_p50,
        latency_p95=latency_p95,
        latency_p99=latency_p99,
        error_rate=error_rate,
        cpu_usage=psutil.cpu_percent(),
        memory_usage=psutil.virtual_memory().percent
    )
    
    return metrics.dict()

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(state.model_cache),
        "active_connections": len(state.active_connections),
        "system_memory": psutil.virtual_memory().percent,
        "cpu_usage": psutil.cpu_percent()
    }

@app.get("/prometheus/metrics")
async def prometheus_metrics():
    """Prometheus metrics endpoint."""
    return generate_latest()

# WebSocket endpoints
@app.websocket("/ws/monitoring")
async def websocket_monitoring(websocket: WebSocket):
    """WebSocket for real-time monitoring."""
    await websocket.accept()
    state.add_connection(websocket)
    
    try:
        while True:
            # Send real-time metrics
            metrics = await get_metrics()
            await websocket.send_json(metrics)
            await asyncio.sleep(1)  # Send updates every second
            
    except WebSocketDisconnect:
        state.remove_connection(websocket)

# Background tasks
async def run_training_job(job_id: str, request: TrainingRequest):
    """Run training job in background."""
    job = state.training_jobs[job_id]
    job["status"] = "running"
    
    try:
        # Simulate training process
        for epoch in range(request.num_epochs):
            if job_id not in state.training_jobs:
                break  # Job was cancelled
            
            # Update progress
            job["current_epoch"] = epoch + 1
            job["progress"] = (epoch + 1) / request.num_epochs
            
            # Simulate training metrics
            job["loss"] = 1.0 - (epoch + 1) / request.num_epochs + np.random.normal(0, 0.05)
            job["accuracy"] = (epoch + 1) / request.num_epochs + np.random.normal(0, 0.02)
            job["safety_score"] = 0.9 + np.random.normal(0, 0.02)
            
            # Calculate ETA
            elapsed_time = time.time() - job["start_time"]
            eta_seconds = int((elapsed_time / (epoch + 1)) * (request.num_epochs - epoch - 1))
            job["eta_seconds"] = eta_seconds
            
            # Add log entry
            log_entry = f"Epoch {epoch + 1}/{request.num_epochs} - Loss: {job['loss']:.4f}, Accuracy: {job['accuracy']:.4f}"
            if "logs" not in job:
                job["logs"] = []
            job["logs"].append(log_entry)
            
            # Simulate training time
            await asyncio.sleep(0.1)
        
        job["status"] = "completed"
        
    except Exception as e:
        job["status"] = "failed"
        job["error"] = str(e)
        logger.error(f"Training job {job_id} failed: {e}")

async def broadcast_safety_alert(alert: SafetyAlert):
    """Broadcast safety alert to all connected clients."""
    if state.active_connections:
        message = json.dumps({"type": "safety_alert", "alert": alert.dict()})
        for connection in state.active_connections.copy():
            try:
                await connection.send_text(message)
            except:
                state.remove_connection(connection)

# Utility functions
def get_recent_inference_records(limit: int = 1000) -> List[InferenceRecord]:
    """Get recent inference records from database."""
    db = SessionLocal()
    try:
        records = db.query(InferenceRecord).order_by(InferenceRecord.timestamp.desc()).limit(limit).all()
        return records
    finally:
        db.close()

# Startup and shutdown events
@app.on_event("startup")
async def startup_event():
    """Initialize server on startup."""
    logger.info("Starting Clinical Robot Adaptation API Server")
    
    # Load models
    load_models()
    
    # Initialize monitoring
    logger.info(f"Loaded {len(state.model_cache)} models")
    logger.info("API Server ready")

@app.on_event("shutdown")
async def shutdown_event():
    """Clean up on shutdown."""
    logger.info("Shutting down Clinical Robot Adaptation API Server")
    
    # Close database connections
    engine.dispose()
    
    # Close WebSocket connections
    for connection in state.active_connections.copy():
        try:
            await connection.close()
        except:
            pass

# Main execution
if __name__ == "__main__":
    uvicorn.run(
        "fastapi_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
