#!/usr/bin/env python3
"""
Clinical Robot Adaptation Visualization Dashboard
Real-time monitoring and analytics dashboard for clinical robot deployment.

This dashboard provides:
- Real-time performance monitoring
- Safety alert visualization
- Model training progress tracking
- Clinical workflow analytics
- Multi-modal data visualization
- Interactive 3D robot visualization

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
import warnings

# Web framework and dashboard
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.figure_factory as ff

# Data processing and visualization
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import cv2

# Real-time data
import websockets
import requests
from asyncstdlib import aio
import threading
import queue

# 3D visualization
import trimesh
import open3d as o3d
from scipy.spatial.transform import Rotation

# Machine learning metrics
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Configuration
DASHBOARD_CONFIG = {
    "title": "Clinical Robot Adaptation Dashboard",
    "refresh_interval": 1000,  # milliseconds
    "api_base_url": "http://localhost:8000",
    "max_data_points": 1000,
    "theme": "streamlit"
}

class ClinicalDashboard:
    """Main dashboard class for clinical robot monitoring."""
    
    def __init__(self):
        self.config = DASHBOARD_CONFIG
        self.data_cache = {}
        self.alert_queue = queue.Queue()
        self.real_time_data = {}
        
        # Initialize session state
        if 'page' not in st.session_state:
            st.session_state.page = 'overview'
        if 'alerts' not in st.session_state:
            st.session_state.alerts = []
        if 'selected_model' not in st.session_state:
            st.session_state.selected_model = 'latest'
    
    def run(self):
        """Run the dashboard application."""
        st.set_page_config(
            page_title=self.config["title"],
            page_icon="🏥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS
        self._apply_custom_css()
        
        # Sidebar navigation
        self._render_sidebar()
        
        # Main content based on selected page
        if st.session_state.page == 'overview':
            self._render_overview_page()
        elif st.session_state.page == 'real_time':
            self._render_real_time_page()
        elif st.session_state.page == 'safety':
            self._render_safety_page()
        elif st.session_state.page == 'models':
            self._render_models_page()
        elif st.session_state.page == 'analytics':
            self._render_analytics_page()
        elif st.session_state.page == 'clinical':
            self._render_clinical_page()
        
        # Footer
        self._render_footer()
    
    def _apply_custom_css(self):
        """Apply custom CSS styling."""
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            font-weight: bold;
            color: #1f77b4;
            text-align: center;
            margin-bottom: 2rem;
        }
        
        .metric-card {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 1rem;
            border-radius: 10px;
            margin: 0.5rem 0;
        }
        
        .alert-critical {
            background-color: #ff4b4b;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        
        .alert-warning {
            background-color: #ffa500;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        
        .alert-info {
            background-color: #1f77b4;
            color: white;
            padding: 1rem;
            border-radius: 5px;
            margin: 0.5rem 0;
        }
        
        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        
        .status-online { background-color: #4caf50; }
        .status-offline { background-color: #f44336; }
        .status-warning { background-color: #ff9800; }
        </style>
        """, unsafe_allow_html=True)
    
    def _render_sidebar(self):
        """Render sidebar navigation."""
        with st.sidebar:
            st.markdown("# 🏥 Clinical Robot Dashboard")
            st.markdown("---")
            
            # Page navigation
            pages = {
                'overview': '📊 Overview',
                'real_time': '📡 Real-time Monitoring',
                'safety': '⚠️ Safety Monitoring',
                'models': '🤖 Model Management',
                'analytics': '📈 Analytics',
                'clinical': '🏥 Clinical Workflow'
            }
            
            for page_key, page_title in pages.items():
                if st.button(page_title, key=f"nav_{page_key}"):
                    st.session_state.page = page_key
            
            st.markdown("---")
            
            # Model selection
            st.markdown("### Model Selection")
            models = self._get_available_models()
            selected_model = st.selectbox(
                "Select Model",
                models,
                index=models.index(st.session_state.selected_model) if st.session_state.selected_model in models else 0
            )
            st.session_state.selected_model = selected_model
            
            # System status
            st.markdown("### System Status")
            status = self._get_system_status()
            
            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f'<span class="status-indicator status-{"online" if status["api_online"] else "offline"}"></span>API', unsafe_allow_html=True)
            with col2:
                st.markdown(f'<span class="status-indicator status-{"online" if status["models_loaded"] > 0 else "offline"}"></span>Models', unsafe_allow_html=True)
            
            st.metric("CPU Usage", f"{status['cpu_usage']:.1f}%")
            st.metric("Memory Usage", f"{status['memory_usage']:.1f}%")
            st.metric("Active Connections", status['active_connections'])
            
            # Recent alerts
            st.markdown("### Recent Alerts")
            alerts = self._get_recent_alerts(limit=3)
            for alert in alerts:
                alert_class = f"alert-{alert['severity']}"
                st.markdown(f'<div class="{alert_class}">{alert["message"]}</div>', unsafe_allow_html=True)
    
    def _render_overview_page(self):
        """Render overview dashboard page."""
        st.markdown('<div class="main-header">Clinical Robot Adaptation Overview</div>', unsafe_allow_html=True)
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            success_rate = self._get_success_rate()
            st.metric("Success Rate", f"{success_rate:.1f}%", delta=f"{success_rate - 85:.1f}%")
        
        with col2:
            safety_score = self._get_safety_score()
            st.metric("Safety Score", f"{safety_score:.3f}", delta=f"{safety_score - 0.9:.3f}")
        
        with col3:
            throughput = self._get_throughput()
            st.metric("Throughput", f"{throughput:.1f} req/hr", delta=f"{throughput - 50:.1f}")
        
        with col4:
            avg_latency = self._get_avg_latency()
            st.metric("Avg Latency", f"{avg_latency:.1f}ms", delta=f"{45 - avg_latency:.1f}ms")
        
        # Performance charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Trends")
            fig = self._create_performance_trend_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Safety Metrics")
            fig = self._create_safety_metrics_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Recent activity
        st.markdown("### Recent Activity")
        activity_data = self._get_recent_activity()
        
        if activity_data:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=activity_data['timestamp'],
                y=activity_data['success_rate'],
                mode='lines+markers',
                name='Success Rate',
                line=dict(color='#1f77b4')
            ))
            
            fig.update_layout(
                title="Recent Success Rate",
                xaxis_title="Time",
                yaxis_title="Success Rate (%)",
                hovermode='x unified'
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Model performance comparison
        st.markdown("### Model Performance Comparison")
        model_comparison = self._get_model_comparison()
        
        if model_comparison:
            fig = px.bar(
                model_comparison,
                x='model',
                y='accuracy',
                color='safety_score',
                title="Model Performance Comparison",
                labels={'accuracy': 'Accuracy', 'safety_score': 'Safety Score'}
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_real_time_page(self):
        """Render real-time monitoring page."""
        st.markdown('<div class="main-header">Real-time Monitoring</div>', unsafe_allow_html=True)
        
        # Auto-refresh toggle
        auto_refresh = st.checkbox("Auto-refresh", value=True)
        refresh_interval = st.slider("Refresh Interval (ms)", 500, 5000, 1000, 100)
        
        # Real-time metrics
        metrics_placeholder = st.empty()
        
        if auto_refresh:
            while auto_refresh:
                metrics = self._get_real_time_metrics()
                
            with metrics_placeholder.container():
                # Real-time metrics grid
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Current Requests/sec", f"{metrics['requests_per_second']:.1f}")
                
                with col2:
                    st.metric("Current Latency", f"{metrics['current_latency']:.1f}ms")
                
                with col3:
                    st.metric("Active Models", metrics['active_models'])
                
                with col4:
                    st.metric("System Load", f"{metrics['system_load']:.1f}%")
                
                # Real-time charts
                col1, col2 = st.columns(2)
                
                with col1:
                    fig = self._create_real_time_performance_chart()
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    fig = self._create_real_time_safety_chart()
                    st.plotly_chart(fig, use_container_width=True)
                
                # Log stream
                st.markdown("### Real-time Log Stream")
                log_data = self._get_real_time_logs()
                
                if log_data:
                    log_df = pd.DataFrame(log_data)
                    st.dataframe(log_df.tail(10), use_container_width=True)
                
                time.sleep(refresh_interval / 1000)
                if not auto_refresh:
                    break
                metrics_placeholder.empty()
    
    def _render_safety_page(self):
        """Render safety monitoring page."""
        st.markdown('<div class="main-header">Safety Monitoring</div>', unsafe_allow_html=True)
        
        # Safety overview
        col1, col2, col3 = st.columns(3)
        
        with col1:
            critical_alerts = len([a for a in st.session_state.alerts if a['severity'] == 'critical'])
            st.metric("Critical Alerts", critical_alerts, delta=f"-{critical_alerts}")
        
        with col2:
            safety_score = self._get_safety_score()
            st.metric("Safety Score", f"{safety_score:.3f}")
        
        with col3:
            uptime = self._get_system_uptime()
            st.metric("System Uptime", f"{uptime:.1f}h")
        
        # Safety alerts
        st.markdown("### Active Safety Alerts")
        
        alert_filters = st.multiselect(
            "Filter by Severity",
            ['critical', 'warning', 'info'],
            default=['critical', 'warning']
        )
        
        filtered_alerts = [a for a in st.session_state.alerts if a['severity'] in alert_filters]
        
        if filtered_alerts:
            for alert in filtered_alerts:
                alert_class = f"alert-{alert['severity']}"
                st.markdown(f'''
                <div class="{alert_class}">
                    <strong>{alert['type'].title()}</strong> - {alert['message']}
                    <br><small>{alert['timestamp']}</small>
                </div>
                ''', unsafe_allow_html=True)
        else:
            st.info("No active alerts matching selected filters")
        
        # Safety analytics
        st.markdown("### Safety Analytics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self._create_safety_trend_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self._create_safety_distribution_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Safety zones visualization
        st.markdown("### Safety Zones Visualization")
        fig = self._create_safety_zones_chart()
        st.plotly_chart(fig, use_container_width=True)
    
    def _render_models_page(self):
        """Render model management page."""
        st.markdown('<div class="main-header">Model Management</div>', unsafe_allow_html=True)
        
        # Model selection and info
        col1, col2 = st.columns([2, 1])
        
        with col1:
            models = self._get_available_models()
            selected_model = st.selectbox("Select Model", models)
            
            model_info = self._get_model_info(selected_model)
            
            st.markdown("### Model Information")
            st.json(model_info)
        
        with col2:
            st.markdown("### Model Actions")
            
            if st.button("Load Model", key="load_model"):
                self._load_model(selected_model)
                st.success(f"Model {selected_model} loaded successfully")
            
            if st.button("Unload Model", key="unload_model"):
                self._unload_model(selected_model)
                st.success(f"Model {selected_model} unloaded")
            
            if st.button("Evaluate Model", key="evaluate_model"):
                results = self._evaluate_model(selected_model)
                st.json(results)
        
        # Training progress
        st.markdown("### Training Progress")
        training_jobs = self._get_training_jobs()
        
        if training_jobs:
            for job_id, job_info in training_jobs.items():
                with st.expander(f"Training Job: {job_id}"):
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.progress(job_info['progress'])
                        st.text(f"Status: {job_info['status']}")
                        st.text(f"Epoch: {job_info['current_epoch']}/{job_info['total_epochs']}")
                    
                    with col2:
                        if job_info.get('loss'):
                            st.metric("Loss", f"{job_info['loss']:.4f}")
                        if job_info.get('accuracy'):
                            st.metric("Accuracy", f"{job_info['accuracy']:.4f}")
                        if job_info.get('eta_seconds'):
                            st.metric("ETA", f"{job_info['eta_seconds']}s")
        else:
            st.info("No active training jobs")
        
        # Model comparison
        st.markdown("### Model Performance Comparison")
        comparison_data = self._get_model_comparison()
        
        if comparison_data:
            fig = px.scatter(
                comparison_data,
                x='accuracy',
                y='safety_score',
                size='throughput',
                color='model',
                title="Model Performance Comparison",
                labels={
                    'accuracy': 'Accuracy',
                    'safety_score': 'Safety Score',
                    'throughput': 'Throughput (req/hr)'
                }
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_analytics_page(self):
        """Render analytics page."""
        st.markdown('<div class="main-header">Analytics & Insights</div>', unsafe_allow_html=True)
        
        # Analytics options
        col1, col2 = st.columns(2)
        
        with col1:
            date_range = st.date_input("Date Range", [datetime.now() - timedelta(days=7), datetime.now()])
        
        with col2:
            metric_type = st.selectbox("Metric Type", ["Performance", "Safety", "Usage", "Errors"])
        
        # Analytics charts
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Performance Analytics")
            fig = self._create_performance_analytics_chart(date_range, metric_type)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Usage Analytics")
            fig = self._create_usage_analytics_chart(date_range)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analytics table
        st.markdown("### Detailed Analytics")
        analytics_data = self._get_analytics_data(date_range, metric_type)
        
        if analytics_data:
            df = pd.DataFrame(analytics_data)
            st.dataframe(df, use_container_width=True)
            
            # Export options
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("Export CSV"):
                    csv = df.to_csv(index=False)
                    st.download_button("Download CSV", csv, "analytics.csv", "text/csv")
            
            with col2:
                if st.button("Export JSON"):
                    json_data = df.to_json(orient='records')
                    st.download_button("Download JSON", json_data, "analytics.json", "application/json")
            
            with col3:
                if st.button("Export Plot"):
                    fig = self._create_export_analytics_plot(df)
                    st.plotly_chart(fig)
        else:
            st.info("No data available for selected date range and metric type")
    
    def _render_clinical_page(self):
        """Render clinical workflow page."""
        st.markdown('<div class="main-header">Clinical Workflow Integration</div>', unsafe_allow_html=True)
        
        # Clinical metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Tasks Completed", "127", delta="12")
        
        with col2:
            st.metric("Medication Accuracy", "98.7%", delta="0.3%")
        
        with col3:
            st.metric("Patient Safety", "99.9%", delta="0.1%")
        
        with col4:
            st.metric("Workflow Efficiency", "94.2%", delta="2.1%")
        
        # Clinical workflow visualization
        st.markdown("### Clinical Workflow Visualization")
        
        # 3D robot visualization (placeholder)
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Robot Workspace")
            fig = self._create_robot_workspace_visualization()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("#### Task Distribution")
            fig = self._create_task_distribution_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Clinical data analysis
        st.markdown("### Clinical Data Analysis")
        
        # Medication handling statistics
        col1, col2 = st.columns(2)
        
        with col1:
            fig = self._create_medication_stats_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = self._create_clinical_efficiency_chart()
            st.plotly_chart(fig, use_container_width=True)
        
        # Patient safety metrics
        st.markdown("### Patient Safety Metrics")
        
        safety_data = self._get_patient_safety_data()
        
        if safety_data:
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=safety_data['date'],
                y=safety_data['safety_score'],
                mode='lines+markers',
                name='Safety Score',
                line=dict(color='green', width=3)
            ))
            
            fig.add_trace(go.Scatter(
                x=safety_data['date'],
                y=safety_data['error_rate'],
                mode='lines+markers',
                name='Error Rate',
                yaxis='y2',
                line=dict(color='red', width=2)
            ))
            
            fig.update_layout(
                title="Patient Safety Trends",
                xaxis_title="Date",
                yaxis_title="Safety Score",
                yaxis2=dict(
                    title="Error Rate (%)",
                    overlaying='y',
                    side='right'
                ),
                legend=dict(x=0.01, y=0.99)
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    def _render_footer(self):
        """Render dashboard footer."""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; font-size: 0.8em;'>
        Clinical Robot Adaptation Dashboard v2.0 | HIRO Laboratory, University of Colorado Boulder
        </div>
        """, unsafe_allow_html=True)
    
    # Chart creation methods
    def _create_performance_trend_chart(self):
        """Create performance trend chart."""
        # Generate sample data
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        success_rate = np.random.normal(0.9, 0.05, 30)
        safety_score = np.random.normal(0.95, 0.02, 30)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Success Rate', 'Safety Score'),
            vertical_spacing=0.1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=success_rate, name='Success Rate', line=dict(color='blue')),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(x=dates, y=safety_score, name='Safety Score', line=dict(color='green')),
            row=2, col=1
        )
        
        fig.update_layout(height=400, showlegend=False)
        return fig
    
    def _create_safety_metrics_chart(self):
        """Create safety metrics chart."""
        categories = ['Human Proximity', 'Velocity Limits', 'Workspace Bounds', 'Force Limits']
        values = [0.98, 0.95, 0.99, 0.97]
        
        fig = go.Figure(data=[
            go.Bar(name='Safety Compliance', x=categories, y=values, marker_color='lightblue')
        ])
        
        fig.update_layout(
            title='Safety Compliance Metrics',
            yaxis_title='Compliance Rate',
            yaxis=dict(range=[0.8, 1.0])
        )
        
        return fig
    
    def _create_real_time_performance_chart(self):
        """Create real-time performance chart."""
        # Generate real-time data
        times = pd.date_range(start=datetime.now(), periods=50, freq='S')
        performance = np.random.normal(0.9, 0.1, 50)
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=times, y=performance,
            mode='lines',
            name='Performance',
            line=dict(color='blue', width=2)
        ))
        
        fig.update_layout(
            title='Real-time Performance',
            xaxis_title='Time',
            yaxis_title='Performance Score',
            yaxis=dict(range=[0, 1])
        )
        
        return fig
    
    def _create_safety_zones_chart(self):
        """Create safety zones visualization."""
        # Create 2D safety zones
        x = np.linspace(-2, 2, 50)
        y = np.linspace(-2, 2, 50)
        X, Y = np.meshgrid(x, y)
        
        # Safety score based on distance from center
        Z = np.exp(-(X**2 + Y**2) / 2)
        
        fig = go.Figure(data=go.Contour(
            x=x, y=y, z=Z,
            colorscale='RdYlGn',
            contours=dict(
                showlabels=True,
                labelfont=dict(size=12, color='white')
            )
        ))
        
        fig.update_layout(
            title='Safety Zones Visualization',
            xaxis_title='X Position (m)',
            yaxis_title='Y Position (m)'
        )
        
        return fig
    
    # Data fetching methods (placeholder implementations)
    def _get_available_models(self) -> List[str]:
        """Get list of available models."""
        return ['latest', 'v1.0', 'v1.1', 'v2.0-beta']
    
    def _get_system_status(self) -> Dict[str, Any]:
        """Get current system status."""
        return {
            'api_online': True,
            'models_loaded': 2,
            'cpu_usage': 45.2,
            'memory_usage': 67.8,
            'active_connections': 5
        }
    
    def _get_success_rate(self) -> float:
        """Get current success rate."""
        return 87.3
    
    def _get_safety_score(self) -> float:
        """Get current safety score."""
        return 0.934
    
    def _get_throughput(self) -> float:
        """Get current throughput."""
        return 73.5
    
    def _get_avg_latency(self) -> float:
        """Get average latency."""
        return 42.7
    
    def _get_recent_alerts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent safety alerts."""
        return [
            {'type': 'human_proximity', 'severity': 'warning', 'message': 'Human detected in workspace', 'timestamp': '2024-01-15 10:30:00'},
            {'type': 'velocity_limit', 'severity': 'info', 'message': 'Velocity limit approached', 'timestamp': '2024-01-15 10:25:00'},
            {'type': 'system_error', 'severity': 'critical', 'message': 'Sensor malfunction detected', 'timestamp': '2024-01-15 10:20:00'}
        ]
    
    def _get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time metrics."""
        return {
            'requests_per_second': 12.5,
            'current_latency': 38.2,
            'active_models': 2,
            'system_load': 52.7
        }
    
    def _get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get model information."""
        return {
            'name': model_name,
            'version': '2.0.0',
            'trained_on': '2024-01-10',
            'accuracy': 0.923,
            'safety_score': 0.967,
            'parameters': '1.5B',
            'training_data': '50k demonstrations'
        }
    
    def _get_training_jobs(self) -> Dict[str, Dict[str, Any]]:
        """Get active training jobs."""
        return {
            'train_1642248000': {
                'status': 'running',
                'progress': 0.67,
                'current_epoch': 67,
                'total_epochs': 100,
                'loss': 0.234,
                'accuracy': 0.891,
                'eta_seconds': 1200
            }
        }
    
    def _create_robot_workspace_visualization(self):
        """Create robot workspace visualization."""
        # Placeholder 3D workspace visualization
        fig = go.Figure()
        
        # Add robot position
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[0.5],
            mode='markers',
            marker=dict(size=10, color='red'),
            name='Robot'
        ))
        
        # Add workspace boundaries
        fig.add_trace(go.Scatter3d(
            x=[-1, 1, 1, -1, -1],
            y=[-1, -1, 1, 1, -1],
            z=[0, 0, 0, 0, 0],
            mode='lines',
            line=dict(color='blue', width=2),
            name='Workspace'
        ))
        
        fig.update_layout(
            title='Robot Workspace',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)'
            )
        )
        
        return fig
    
    def _create_task_distribution_chart(self):
        """Create task distribution chart."""
        tasks = ['Pick', 'Place', 'Sort', 'Inspect', 'Package']
        counts = [45, 38, 22, 15, 7]
        
        fig = go.Figure(data=[
            go.Pie(labels=tasks, values=counts, hole=0.3)
        ])
        
        fig.update_layout(title='Task Distribution')
        return fig
    
    def _create_medication_stats_chart(self):
        """Create medication statistics chart."""
        medications = ['Vials', 'Bottles', 'Syringes', 'Blister Packs', 'Pouches']
        success_rates = [98.5, 97.2, 96.8, 94.3, 95.7]
        
        fig = go.Figure(data=[
            go.Bar(x=medications, y=success_rates, marker_color='lightgreen')
        ])
        
        fig.update_layout(
            title='Medication Handling Success Rates',
            xaxis_title='Medication Type',
            yaxis_title='Success Rate (%)',
            yaxis=dict(range=[90, 100])
        )
        
        return fig
    
    def _create_clinical_efficiency_chart(self):
        """Create clinical efficiency chart."""
        hours = list(range(24))
        efficiency = [0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.7, 0.9, 0.95, 0.9, 0.85, 0.8,
                     0.85, 0.8, 0.75, 0.8, 0.85, 0.9, 0.85, 0.7, 0.5, 0.4, 0.3, 0.2]
        
        fig = go.Figure(data=[
            go.Scatter(x=hours, y=efficiency, mode='lines+markers', line=dict(color='blue', width=2))
        ])
        
        fig.update_layout(
            title='Clinical Workflow Efficiency by Hour',
            xaxis_title='Hour of Day',
            yaxis_title='Efficiency Score'
        )
        
        return fig
    
    def _get_patient_safety_data(self) -> Dict[str, List]:
        """Get patient safety data."""
        dates = pd.date_range(start='2024-01-01', periods=30, freq='D')
        safety_scores = np.random.normal(0.98, 0.02, 30)
        error_rates = np.random.normal(0.5, 0.2, 30)
        
        return {
            'date': dates,
            'safety_score': safety_scores,
            'error_rate': error_rates
        }

# Main execution
def main():
    """Main function to run the dashboard."""
    dashboard = ClinicalDashboard()
    dashboard.run()

if __name__ == "__main__":
    main()
