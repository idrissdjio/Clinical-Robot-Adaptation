#!/usr/bin/env python3
"""
Performance Monitoring Tools
Performance tracking and monitoring utilities for clinical robotics applications.
"""

import time
import psutil
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict, deque
import threading
import json
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    """Performance metric data."""
    name: str
    value: float
    unit: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceReport:
    """Performance report summary."""
    start_time: datetime
    end_time: datetime
    metrics: List[PerformanceMetric]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'start_time': self.start_time.isoformat(),
            'end_time': self.end_time.isoformat(),
            'duration_seconds': (self.end_time - self.start_time).total_seconds(),
            'metrics': [
                {
                    'name': m.name,
                    'value': m.value,
                    'unit': m.unit,
                    'timestamp': m.timestamp.isoformat(),
                    'metadata': m.metadata
                }
                for m in self.metrics
            ],
            'summary': self.summary
        }

class PerformanceMonitor:
    """Performance monitoring system."""
    
    def __init__(self, max_metrics: int = 10000):
        self.metrics = defaultdict(lambda: deque(maxlen=max_metrics))
        self.start_time = datetime.now()
        self.is_monitoring = False
        self.monitoring_thread = None
        self.monitoring_interval = 1.0
    
    def record_metric(self, name: str, value: float, unit: str = "",
                     metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=datetime.now(),
            metadata=metadata or {}
        )
        self.metrics[name].append(metric)
    
    def get_metrics(self, name: str, 
                    since: Optional[datetime] = None) -> List[PerformanceMetric]:
        """Get metrics for a specific name."""
        metrics = list(self.metrics[name])
        
        if since:
            metrics = [m for m in metrics if m.timestamp >= since]
        
        return metrics
    
    def get_all_metrics(self) -> Dict[str, List[PerformanceMetric]]:
        """Get all recorded metrics."""
        return dict(self.metrics)
    
    def get_metric_summary(self, name: str) -> Dict[str, float]:
        """Get summary statistics for a metric."""
        metrics = list(self.metrics[name])
        
        if not metrics:
            return {}
        
        values = [m.value for m in metrics]
        
        return {
            'count': len(values),
            'mean': sum(values) / len(values),
            'min': min(values),
            'max': max(values),
            'sum': sum(values)
        }
    
    def start_monitoring(self):
        """Start continuous monitoring."""
        if self.is_monitoring:
            return
        
        self.is_monitoring = True
        self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitoring_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous monitoring."""
        self.is_monitoring = False
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5)
    
    def _monitoring_loop(self):
        """Monitoring loop for system metrics."""
        while self.is_monitoring:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=0.1)
                self.record_metric('cpu_usage', cpu_percent, '%')
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.record_metric('memory_usage', memory.percent, '%')
                self.record_metric('memory_available', memory.available / (1024**3), 'GB')
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.record_metric('disk_usage', disk.percent, '%')
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.record_metric('network_bytes_sent', net_io.bytes_sent, 'bytes')
                self.record_metric('network_bytes_recv', net_io.bytes_recv, 'bytes')
                
                time.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
    
    def generate_report(self, output_path: Optional[str] = None) -> PerformanceReport:
        """Generate performance report."""
        end_time = datetime.now()
        
        # Collect all metrics
        all_metrics = []
        summary = {}
        
        for name, metrics in self.metrics.items():
            all_metrics.extend(metrics)
            summary[name] = self.get_metric_summary(name)
        
        report = PerformanceReport(
            start_time=self.start_time,
            end_time=end_time,
            metrics=all_metrics,
            summary=summary
        )
        
        # Save to file if path provided
        if output_path:
            self._save_report(report, output_path)
        
        return report
    
    def _save_report(self, report: PerformanceReport, output_path: str):
        """Save report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"Performance report saved to: {output_path}")

class Timer:
    """Context manager for timing code execution."""
    
    def __init__(self, name: str, monitor: Optional[PerformanceMonitor] = None):
        self.name = name
        self.monitor = monitor
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.monitor:
            self.monitor.record_metric(f"timer_{self.name}", duration, 's')
        
        logger.debug(f"Timer '{self.name}': {duration:.4f}s")

def performance_timer(name: str, monitor: Optional[PerformanceMonitor] = None):
    """Decorator for timing function execution."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs) -> Any:
            with Timer(name, monitor):
                return func(*args, **kwargs)
        return wrapper
    return decorator

class ResourceProfiler:
    """Profile resource usage of code execution."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.baseline = None
    
    def profile(self, func: Callable, *args, **kwargs) -> Any:
        """Profile a function call."""
        # Record baseline
        self.baseline = {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'time': time.time()
        }
        
        # Start monitoring
        self.monitor.start_monitoring()
        
        # Execute function
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            # Stop monitoring
            self.monitor.stop_monitoring()
    
    def get_profile_summary(self) -> Dict[str, Any]:
        """Get profile summary."""
        if not self.baseline:
            return {}
        
        current = {
            'cpu': psutil.cpu_percent(),
            'memory': psutil.virtual_memory().percent,
            'time': time.time()
        }
        
        return {
            'duration': current['time'] - self.baseline['time'],
            'cpu_delta': current['cpu'] - self.baseline['cpu'],
            'memory_delta': current['memory'] - self.baseline['memory'],
            'metrics': self.monitor.get_all_metrics()
        }

if __name__ == '__main__':
    # Example usage
    monitor = PerformanceMonitor()
    
    # Record some metrics
    monitor.record_metric('test_metric', 42.0, 'units')
    monitor.record_metric('another_metric', 100.0, '%')
    
    # Get summary
    summary = monitor.get_metric_summary('test_metric')
    print(f"Summary: {summary}")
    
    # Use timer context manager
    with Timer('example', monitor):
        time.sleep(0.1)
    
    # Generate report
    report = monitor.generate_report('performance_report.json')
    print(f"Report generated with {len(report.metrics)} metrics")
