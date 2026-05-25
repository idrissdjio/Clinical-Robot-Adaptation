#!/usr/bin/env python3
"""
Advanced Analytics Dashboard for Clinical Robotics
Comprehensive analytics and visualization system for clinical robot operations.

This module implements:
- Real-time performance monitoring
- Clinical workflow analytics
- Safety incident tracking and analysis
- Predictive maintenance analytics
- Resource utilization analytics
- Patient outcome analytics
- Custom dashboard creation
- Automated report generation

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import warnings

# Data processing and analysis
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import savgol_filter

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Dashboard framework
import streamlit as st
from streamlit_option_menu import option_menu

# Database
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Monitoring
import psutil

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('advanced_analytics.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

class MetricType(Enum):
    """Types of metrics."""
    PERFORMANCE = "performance"
    SAFETY = "safety"
    CLINICAL = "clinical"
    OPERATIONAL = "operational"
    FINANCIAL = "financial"

class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class Metric:
    """Metric definition."""
    id: str
    name: str
    description: str
    metric_type: MetricType
    unit: str
    threshold_warning: Optional[float] = None
    threshold_critical: Optional[float] = None
    aggregation: str = "mean"  # mean, sum, count, max, min
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'metric_type': self.metric_type.value,
            'unit': self.unit,
            'threshold_warning': self.threshold_warning,
            'threshold_critical': self.threshold_critical,
            'aggregation': self.aggregation,
            'enabled': self.enabled
        }

@dataclass
class Alert:
    """Alert definition."""
    id: str
    metric_id: str
    severity: AlertSeverity
    condition: str  # >, <, ==, !=
    threshold: float
    message: str
    enabled: bool = True
    notification_channels: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.id,
            'metric_id': self.metric_id,
            'severity': self.severity.value,
            'condition': self.condition,
            'threshold': self.threshold,
            'message': self.message,
            'enabled': self.enabled,
            'notification_channels': self.notification_channels
        }

@dataclass
class DashboardConfig:
    """Configuration for analytics dashboard."""
    
    # General settings
    dashboard_title: str = "Clinical Robot Analytics Dashboard"
    refresh_interval: int = 5  # seconds
    data_retention_days: int = 90
    
    # Database settings
    database_url: str = "sqlite:///analytics.db"
    
    # Visualization settings
    default_theme: str = "plotly"  # plotly, matplotlib, seaborn
    color_palette: str = "viridis"
    figure_size: Tuple[int, int] = (12, 8)
    
    # Alert settings
    enable_alerts: bool = True
    alert_cooldown_minutes: int = 30
    
    # Export settings
    enable_export: bool = True
    export_formats: List[str] = field(default_factory=lambda: ["png", "pdf", "csv"])
    
    # Real-time settings
    enable_real_time: bool = True
    real_time_window_minutes: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'dashboard_title': self.dashboard_title,
            'refresh_interval': self.refresh_interval,
            'data_retention_days': self.data_retention_days,
            'database_url': self.database_url,
            'default_theme': self.default_theme,
            'color_palette': self.color_palette,
            'figure_size': self.figure_size,
            'enable_alerts': self.enable_alerts,
            'alert_cooldown_minutes': self.alert_cooldown_minutes,
            'enable_export': self.enable_export,
            'export_formats': self.export_formats,
            'enable_real_time': self.enable_real_time,
            'real_time_window_minutes': self.real_time_window_minutes
        }

# Database models
Base = declarative_base()

class MetricData(Base):
    """Metric data database model."""
    __tablename__ = "metric_data"
    
    id = Column(Integer, primary_key=True, index=True)
    metric_id = Column(String(100), nullable=False, index=True)
    timestamp = Column(DateTime, nullable=False, index=True)
    value = Column(Float, nullable=False)
    metadata = Column(Text, nullable=True)  # JSON string

class AlertHistory(Base):
    """Alert history database model."""
    __tablename__ = "alert_history"
    
    id = Column(Integer, primary_key=True, index=True)
    alert_id = Column(String(100), nullable=False, index=True)
    metric_id = Column(String(100), nullable=False)
    severity = Column(String(20), nullable=False)
    triggered_at = Column(DateTime, nullable=False, index=True)
    resolved_at = Column(DateTime, nullable=True)
    value = Column(Float, nullable=False)
    message = Column(Text, nullable=True)

class AnalyticsEngine:
    """Analytics engine for data processing and analysis."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
        # Initialize database
        self.engine = create_engine(config.database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Metrics and alerts
        self.metrics = self._initialize_metrics()
        self.alerts = self._initialize_alerts()
        
        # Data buffers
        self.metric_buffers = defaultdict(lambda: deque(maxlen=1000))
        
        # Alert cooldown tracking
        self.alert_cooldowns = defaultdict(lambda: datetime.min)
        
        logger.info("Analytics Engine initialized")
    
    def _initialize_metrics(self) -> Dict[str, Metric]:
        """Initialize default metrics."""
        metrics = {}
        
        # Performance metrics
        metrics['task_completion_rate'] = Metric(
            id='task_completion_rate',
            name='Task Completion Rate',
            description='Percentage of tasks completed successfully',
            metric_type=MetricType.PERFORMANCE,
            unit='%',
            threshold_warning=80.0,
            threshold_critical=70.0,
            aggregation='mean'
        )
        
        metrics['average_task_duration'] = Metric(
            id='average_task_duration',
            name='Average Task Duration',
            description='Average time to complete tasks',
            metric_type=MetricType.PERFORMANCE,
            unit='seconds',
            threshold_warning=300.0,
            threshold_critical=600.0,
            aggregation='mean'
        )
        
        metrics['robot_utilization'] = Metric(
            id='robot_utilization',
            name='Robot Utilization',
            description='Percentage of time robot is actively working',
            metric_type=MetricType.OPERATIONAL,
            unit='%',
            threshold_warning=50.0,
            threshold_critical=30.0,
            aggregation='mean'
        )
        
        # Safety metrics
        metrics['safety_incidents'] = Metric(
            id='safety_incidents',
            name='Safety Incidents',
            description='Number of safety incidents per hour',
            metric_type=MetricType.SAFETY,
            unit='count',
            threshold_warning=1.0,
            threshold_critical=5.0,
            aggregation='sum'
        )
        
        metrics['human_proximity_violations'] = Metric(
            id='human_proximity_violations',
            name='Human Proximity Violations',
            description='Number of times robot violated safe human distance',
            metric_type=MetricType.SAFETY,
            unit='count',
            threshold_warning=2.0,
            threshold_critical=10.0,
            aggregation='sum'
        )
        
        # Clinical metrics
        metrics['medication_errors'] = Metric(
            id='medication_errors',
            name='Medication Errors',
            description='Number of medication dispensing errors',
            metric_type=MetricType.CLINICAL,
            unit='count',
            threshold_warning=0.0,
            threshold_critical=1.0,
            aggregation='sum'
        )
        
        metrics['patient_satisfaction'] = Metric(
            id='patient_satisfaction',
            name='Patient Satisfaction Score',
            description='Average patient satisfaction rating',
            metric_type=MetricType.CLINICAL,
            unit='score',
            threshold_warning=4.0,
            threshold_critical=3.0,
            aggregation='mean'
        )
        
        # Operational metrics
        metrics['system_uptime'] = Metric(
            id='system_uptime',
            name='System Uptime',
            description='Percentage of time system is operational',
            metric_type=MetricType.OPERATIONAL,
            unit='%',
            threshold_warning=95.0,
            threshold_critical=90.0,
            aggregation='mean'
        )
        
        metrics['cpu_usage'] = Metric(
            id='cpu_usage',
            name='CPU Usage',
            description='Average CPU utilization',
            metric_type=MetricType.OPERATIONAL,
            unit='%',
            threshold_warning=80.0,
            threshold_critical=95.0,
            aggregation='mean'
        )
        
        metrics['memory_usage'] = Metric(
            id='memory_usage',
            name='Memory Usage',
            description='Average memory utilization',
            metric_type=MetricType.OPERATIONAL,
            unit='%',
            threshold_warning=80.0,
            threshold_critical=95.0,
            aggregation='mean'
        )
        
        return metrics
    
    def _initialize_alerts(self) -> Dict[str, Alert]:
        """Initialize default alerts."""
        alerts = {}
        
        # Performance alerts
        alerts['task_completion_low'] = Alert(
            id='task_completion_low',
            metric_id='task_completion_rate',
            severity=AlertSeverity.WARNING,
            condition='<',
            threshold=80.0,
            message='Task completion rate below 80%'
        )
        
        alerts['task_duration_high'] = Alert(
            id='task_duration_high',
            metric_id='average_task_duration',
            severity=AlertSeverity.WARNING,
            condition='>',
            threshold=300.0,
            message='Average task duration exceeds 5 minutes'
        )
        
        # Safety alerts
        alerts['safety_incident'] = Alert(
            id='safety_incident',
            metric_id='safety_incidents',
            severity=AlertSeverity.CRITICAL,
            condition='>',
            threshold=0.0,
            message='Safety incident detected'
        )
        
        alerts['human_proximity'] = Alert(
            id='human_proximity',
            metric_id='human_proximity_violations',
            severity=AlertSeverity.ERROR,
            condition='>',
            threshold=2.0,
            message='Multiple human proximity violations detected'
        )
        
        # Clinical alerts
        alerts['medication_error'] = Alert(
            id='medication_error',
            metric_id='medication_errors',
            severity=AlertSeverity.CRITICAL,
            condition='>',
            threshold=0.0,
            message='Medication dispensing error detected'
        )
        
        # Operational alerts
        alerts['system_down'] = Alert(
            id='system_down',
            metric_id='system_uptime',
            severity=AlertSeverity.CRITICAL,
            condition='<',
            threshold=90.0,
            message='System uptime below 90%'
        )
        
        alerts['high_cpu'] = Alert(
            id='high_cpu',
            metric_id='cpu_usage',
            severity=AlertSeverity.WARNING,
            condition='>',
            threshold=80.0,
            message='High CPU usage detected'
        )
        
        alerts['high_memory'] = Alert(
            id='high_memory',
            metric_id='memory_usage',
            severity=AlertSeverity.WARNING,
            condition='>',
            threshold=80.0,
            message='High memory usage detected'
        )
        
        return alerts
    
    def record_metric(self, metric_id: str, value: float, 
                     metadata: Dict[str, Any] = None):
        """Record metric value."""
        if metric_id not in self.metrics:
            logger.warning(f"Unknown metric: {metric_id}")
            return
        
        metric = self.metrics[metric_id]
        
        # Store in buffer
        self.metric_buffers[metric_id].append({
            'timestamp': datetime.now(),
            'value': value,
            'metadata': metadata or {}
        })
        
        # Store in database
        db = self.SessionLocal()
        try:
            record = MetricData(
                metric_id=metric_id,
                timestamp=datetime.now(),
                value=value,
                metadata=json.dumps(metadata) if metadata else None
            )
            db.add(record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record metric: {e}")
        finally:
            db.close()
        
        # Check alerts
        if self.config.enable_alerts:
            self._check_alerts(metric_id, value)
    
    def _check_alerts(self, metric_id: str, value: float):
        """Check if any alerts should be triggered."""
        # Find alerts for this metric
        metric_alerts = [a for a in self.alerts.values() if a.metric_id == metric_id and a.enabled]
        
        for alert in metric_alerts:
            # Check cooldown
            if datetime.now() - self.alert_cooldowns[alert.id] < timedelta(minutes=self.config.alert_cooldown_minutes):
                continue
            
            # Check condition
            should_trigger = False
            if alert.condition == '>' and value > alert.threshold:
                should_trigger = True
            elif alert.condition == '<' and value < alert.threshold:
                should_trigger = True
            elif alert.condition == '==' and value == alert.threshold:
                should_trigger = True
            elif alert.condition == '!=' and value != alert.threshold:
                should_trigger = True
            
            if should_trigger:
                self._trigger_alert(alert, value)
                self.alert_cooldowns[alert.id] = datetime.now()
    
    def _trigger_alert(self, alert: Alert, value: float):
        """Trigger an alert."""
        logger.warning(f"ALERT: {alert.message} (value: {value})")
        
        # Store in database
        db = self.SessionLocal()
        try:
            alert_record = AlertHistory(
                alert_id=alert.id,
                metric_id=alert.metric_id,
                severity=alert.severity.value,
                triggered_at=datetime.now(),
                value=value,
                message=alert.message
            )
            db.add(alert_record)
            db.commit()
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to record alert: {e}")
        finally:
            db.close()
    
    def get_metric_data(self, metric_id: str, start_time: datetime = None, 
                       end_time: datetime = None) -> pd.DataFrame:
        """Get metric data for time range."""
        db = self.SessionLocal()
        
        try:
            query = db.query(MetricData).filter(MetricData.metric_id == metric_id)
            
            if start_time:
                query = query.filter(MetricData.timestamp >= start_time)
            
            if end_time:
                query = query.filter(MetricData.timestamp <= end_time)
            
            query = query.order_by(MetricData.timestamp)
            
            results = query.all()
            
            data = []
            for result in results:
                data.append({
                    'timestamp': result.timestamp,
                    'value': result.value,
                    'metadata': json.loads(result.metadata) if result.metadata else {}
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get metric data: {e}")
            return pd.DataFrame()
        finally:
            db.close()
    
    def get_aggregated_metrics(self, metric_ids: List[str] = None, 
                              aggregation_period: str = "hour") -> Dict[str, float]:
        """Get aggregated metrics."""
        if metric_ids is None:
            metric_ids = list(self.metrics.keys())
        
        aggregated = {}
        
        for metric_id in metric_ids:
            if metric_id not in self.metrics:
                continue
            
            metric = self.metrics[metric_id]
            buffer = self.metric_buffers[metric_id]
            
            if not buffer:
                continue
            
            values = [item['value'] for item in buffer]
            
            if metric.aggregation == 'mean':
                aggregated[metric_id] = np.mean(values)
            elif metric.aggregation == 'sum':
                aggregated[metric_id] = np.sum(values)
            elif metric.aggregation == 'count':
                aggregated[metric_id] = len(values)
            elif metric.aggregation == 'max':
                aggregated[metric_id] = np.max(values)
            elif metric.aggregation == 'min':
                aggregated[metric_id] = np.min(values)
        
        return aggregated
    
    def get_alert_history(self, severity: AlertSeverity = None, 
                         start_time: datetime = None, 
                         end_time: datetime = None) -> pd.DataFrame:
        """Get alert history."""
        db = self.SessionLocal()
        
        try:
            query = db.query(AlertHistory)
            
            if severity:
                query = query.filter(AlertHistory.severity == severity.value)
            
            if start_time:
                query = query.filter(AlertHistory.triggered_at >= start_time)
            
            if end_time:
                query = query.filter(AlertHistory.triggered_at <= end_time)
            
            query = query.order_by(AlertHistory.triggered_at.desc())
            
            results = query.all()
            
            data = []
            for result in results:
                data.append({
                    'alert_id': result.alert_id,
                    'metric_id': result.metric_id,
                    'severity': result.severity,
                    'triggered_at': result.triggered_at,
                    'resolved_at': result.resolved_at,
                    'value': result.value,
                    'message': result.message
                })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            logger.error(f"Failed to get alert history: {e}")
            return pd.DataFrame()
        finally:
            db.close()

class DashboardVisualizer:
    """Visualization component for analytics dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
        # Set style
        sns.set_palette(config.color_palette)
        plt.style.use('seaborn')
    
    def create_time_series_plot(self, data: pd.DataFrame, metric_id: str) -> go.Figure:
        """Create time series plot for metric."""
        if data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data['timestamp'],
            y=data['value'],
            mode='lines+markers',
            name=metric_id,
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=4)
        ))
        
        # Add trend line
        if len(data) > 2:
            z = np.polyfit(range(len(data)), data['value'], 1)
            p = np.poly1d(z)
            fig.add_trace(go.Scatter(
                x=data['timestamp'],
                y=p(range(len(data))),
                mode='lines',
                name='Trend',
                line=dict(color='red', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title=f"{metric_id} Over Time",
            xaxis_title="Time",
            yaxis_title="Value",
            hovermode='x unified',
            template='plotly_white'
        )
        
        return fig
    
    def create_histogram(self, data: pd.DataFrame, metric_id: str) -> go.Figure:
        """Create histogram for metric distribution."""
        if data.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=data['value'],
            nbinsx=30,
            name=metric_id,
            marker_color='#1f77b4'
        ))
        
        # Add mean line
        mean_value = data['value'].mean()
        fig.add_vline(
            x=mean_value,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean: {mean_value:.2f}"
        )
        
        fig.update_layout(
            title=f"{metric_id} Distribution",
            xaxis_title="Value",
            yaxis_title="Count",
            template='plotly_white'
        )
        
        return fig
    
    def create_box_plot(self, data: pd.DataFrame, metric_ids: List[str]) -> go.Figure:
        """Create box plot comparing multiple metrics."""
        fig = go.Figure()
        
        for metric_id in metric_ids:
            metric_data = data[data['metric_id'] == metric_id]['value'] if 'metric_id' in data.columns else []
            if len(metric_data) > 0:
                fig.add_trace(go.Box(
                    y=metric_data,
                    name=metric_id,
                    boxpoints='outliers'
                ))
        
        fig.update_layout(
            title="Metric Comparison",
            yaxis_title="Value",
            template='plotly_white'
        )
        
        return fig
    
    def create_heatmap(self, correlation_matrix: pd.DataFrame) -> go.Figure:
        """Create heatmap for correlation matrix."""
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=np.round(correlation_matrix.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title="Metric Correlation Matrix",
            template='plotly_white'
        )
        
        return fig
    
    def create_gauge_chart(self, value: float, title: str, 
                          min_value: float = 0, max_value: float = 100) -> go.Figure:
        """Create gauge chart for single metric."""
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title},
            gauge={
                'axis': {'range': [min_value, max_value]},
                'bar': {'color': "#1f77b4"},
                'steps': [
                    {'range': [min_value, max_value * 0.5], 'color': "#ff6b6b"},
                    {'range': [max_value * 0.5, max_value * 0.8], 'color': "#ffd93d"},
                    {'range': [max_value * 0.8, max_value], 'color': "#6bcb77"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': max_value * 0.9
                }
            }
        ))
        
        fig.update_layout(template='plotly_white')
        
        return fig
    
    def create_scatter_plot(self, data: pd.DataFrame, x_metric: str, 
                           y_metric: str) -> go.Figure:
        """Create scatter plot comparing two metrics."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=data[x_metric],
            y=data[y_metric],
            mode='markers',
            marker=dict(
                size=8,
                color=data[y_metric],
                colorscale='Viridis',
                showscale=True
            )
        ))
        
        fig.update_layout(
            title=f"{x_metric} vs {y_metric}",
            xaxis_title=x_metric,
            yaxis_title=y_metric,
            template='plotly_white'
        )
        
        return fig

class AdvancedAnalyticsDashboard:
    """Main advanced analytics dashboard."""
    
    def __init__(self, config: DashboardConfig):
        self.config = config
        
        # Initialize components
        self.analytics_engine = AnalyticsEngine(config)
        self.visualizer = DashboardVisualizer(config)
        
        # Monitoring
        self.is_monitoring = False
        self.monitoring_thread = None
        
        logger.info("Advanced Analytics Dashboard initialized")
    
    def start_monitoring(self):
        """Start real-time monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
        
        logger.info("Real-time monitoring started")
    
    def stop_monitoring(self):
        """Stop real-time monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
        
        logger.info("Real-time monitoring stopped")
    
    def _monitoring_loop(self):
        """Monitoring loop for real-time data collection."""
        while self.is_monitoring:
            try:
                # Collect system metrics
                cpu_usage = psutil.cpu_percent()
                memory_usage = psutil.virtual_memory().percent
                
                self.analytics_engine.record_metric('cpu_usage', cpu_usage)
                self.analytics_engine.record_metric('memory_usage', memory_usage)
                
                # Simulate other metrics
                self.analytics_engine.record_metric('task_completion_rate', np.random.uniform(85, 95))
                self.analytics_engine.record_metric('average_task_duration', np.random.uniform(120, 180))
                self.analytics_engine.record_metric('robot_utilization', np.random.uniform(60, 80))
                self.analytics_engine.record_metric('system_uptime', 99.5)
                
                time.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def generate_dashboard_report(self, output_path: str = None) -> str:
        """Generate comprehensive dashboard report."""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"./analytics_reports/dashboard_report_{timestamp}.html"
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Get aggregated metrics
        metrics = self.analytics_engine.get_aggregated_metrics()
        
        # Get alert history
        alert_history = self.analytics_engine.get_alert_history()
        
        # Create HTML report
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>{self.config.dashboard_title}</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ text-align: center; margin-bottom: 30px; }}
        .metric-card {{ border: 1px solid #ddd; padding: 20px; margin: 10px; border-radius: 5px; }}
        .alert {{ padding: 10px; margin: 10px; border-radius: 5px; }}
        .alert-critical {{ background-color: #ff6b6b; }}
        .alert-warning {{ background-color: #ffd93d; }}
        .alert-error {{ background-color: #ff8c42; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{self.config.dashboard_title}</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <h2>Key Metrics</h2>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px;">
"""
        
        # Add metric cards
        for metric_id, value in metrics.items():
            metric = self.analytics_engine.metrics.get(metric_id)
            if metric:
                html_content += f"""
        <div class="metric-card">
            <h3>{metric.name}</h3>
            <p style="font-size: 24px; font-weight: bold;">{value:.2f} {metric.unit}</p>
            <p>{metric.description}</p>
        </div>
"""
        
        html_content += """
    </div>
    
    <h2>Recent Alerts</h2>
"""
        
        # Add alerts
        if not alert_history.empty:
            for _, alert in alert_history.head(10).iterrows():
                severity_class = f"alert-{alert['severity']}"
                html_content += f"""
        <div class="alert {severity_class}">
            <strong>{alert['severity'].upper()}</strong>: {alert['message']} ({alert['triggered_at']})
        </div>
"""
        else:
            html_content += "<p>No recent alerts</p>"
        
        html_content += """
</body>
</html>
"""
        
        # Save report
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Dashboard report saved to: {output_path}")
        
        return str(output_path)

def main():
    """Main function for analytics dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Advanced Analytics Dashboard')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--action', type=str, 
                       choices=['monitor', 'report', 'metrics'],
                       help='Action to perform')
    parser.add_argument('--output', type=str, help='Output path for reports')
    
    args = parser.parse_args()
    
    # Load configuration
    config = DashboardConfig()
    
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
            for key, value in config_dict.items():
                if hasattr(config, key):
                    setattr(config, key, value)
    
    # Create dashboard
    dashboard = AdvancedAnalyticsDashboard(config)
    
    # Perform action
    if args.action == "monitor":
        dashboard.start_monitoring()
        print("Analytics monitoring started. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping monitoring...")
            dashboard.stop_monitoring()
    
    elif args.action == "report":
        report_path = dashboard.generate_dashboard_report(args.output)
        print(f"Dashboard report generated: {report_path}")
    
    elif args.action == "metrics":
        metrics = dashboard.analytics_engine.get_aggregated_metrics()
        print("Current Metrics:")
        for metric_id, value in metrics.items():
            metric = dashboard.analytics_engine.metrics.get(metric_id)
            if metric:
                print(f"  {metric.name}: {value:.2f} {metric.unit}")

if __name__ == "__main__":
    main()
