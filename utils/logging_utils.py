#!/usr/bin/env python3
"""
Logging Utilities
Centralized logging configuration and utilities for clinical robotics applications.
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
import json

class ClinicalLogger:
    """Custom logger for clinical robotics applications."""
    
    _loggers: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, name: str, log_file: Optional[str] = None,
                   level: str = "INFO", console: bool = True,
                   json_format: bool = False) -> logging.Logger:
        """Get or create a logger with specified configuration."""
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level.upper()))
        logger.handlers.clear()
        
        formatter = cls._get_formatter(json_format)
        
        # Console handler
        if console:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(getattr(logging, level.upper()))
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        
        # File handler
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, level.upper()))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._loggers[name] = logger
        return logger
    
    @classmethod
    def _get_formatter(cls, json_format: bool) -> logging.Formatter:
        """Get log formatter."""
        if json_format:
            return JsonFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
    
    @classmethod
    def setup_global_logging(cls, log_dir: str = "./logs",
                            level: str = "INFO",
                            json_format: bool = False):
        """Setup global logging configuration."""
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        
        # Root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, level.upper()))
        
        formatter = cls._get_formatter(json_format)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
        
        # File handler
        file_handler = TimedRotatingFileHandler(
            log_path / "clinical_robot.log",
            when='midnight',
            interval=1,
            backupCount=30
        )
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        file_handler.suffix = "%Y-%m-%d"
        root_logger.addHandler(file_handler)

class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        log_data = {
            'timestamp': datetime.utcnow().isoformat(),
            'logger': record.name,
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

class PerformanceLogger:
    """Logger for performance metrics."""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.metrics = {}
    
    def log_metric(self, name: str, value: float, unit: str = ""):
        """Log a performance metric."""
        self.metrics[name] = value
        self.logger.info(f"METRIC: {name}={value}{unit}")
    
    def log_timing(self, name: str, duration: float):
        """Log timing information."""
        self.log_metric(name, duration, "s")
    
    def get_metrics(self) -> Dict[str, float]:
        """Get all logged metrics."""
        return self.metrics.copy()

def setup_logger(name: str, **kwargs) -> logging.Logger:
    """Convenience function to setup a logger."""
    return ClinicalLogger.get_logger(name, **kwargs)

if __name__ == '__main__':
    # Example usage
    logger = setup_logger("test", level="DEBUG", console=True)
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
