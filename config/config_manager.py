#!/usr/bin/env python3
"""
Configuration Management System
Centralized configuration management for clinical robotics applications.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

logger = logging.getLogger(__name__)

class ConfigFormat(Enum):
    """Configuration file formats."""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"

@dataclass
class DatabaseConfig:
    """Database configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "clinical_db"
    user: str = "clinical_user"
    password: str = ""
    pool_size: int = 10

@dataclass
class RobotConfig:
    """Robot configuration."""
    robot_id: str = "robot_001"
    max_velocity: float = 1.0
    max_acceleration: float = 2.0
    safety_distance: float = 0.5
    enable_safety_monitoring: bool = True

@dataclass
class SimulationConfig:
    """Simulation configuration."""
    enable_physics: bool = True
    time_step: float = 0.01
    gravity: float = -9.81
    enable_visualization: bool = True

@dataclass
class ClinicalConfig:
    """Clinical-specific configuration."""
    hospital_name: str = "General Hospital"
    department: str = "Pharmacy"
    enable_hipaa_compliance: bool = True
    data_retention_days: int = 365

@dataclass
class AppConfig:
    """Main application configuration."""
    app_name: str = "ClinicalRobotSystem"
    version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    robot: RobotConfig = field(default_factory=RobotConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    clinical: ClinicalConfig = field(default_factory=ClinicalConfig)

class ConfigManager:
    """Configuration manager for loading and saving configurations."""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = Path(config_path) if config_path else None
        self.config = AppConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, format: Optional[ConfigFormat] = None) -> AppConfig:
        """Load configuration from file."""
        if not self.config_path:
            raise ValueError("No config path specified")
        
        if format is None:
            format = self._detect_format()
        
        if format == ConfigFormat.JSON:
            return self._load_json()
        elif format == ConfigFormat.YAML:
            return self._load_yaml()
        elif format == ConfigFormat.ENV:
            return self._load_env()
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _detect_format(self) -> ConfigFormat:
        """Detect configuration file format from extension."""
        if self.config_path.suffix == '.json':
            return ConfigFormat.JSON
        elif self.config_path.suffix in ['.yaml', '.yml']:
            return ConfigFormat.YAML
        else:
            return ConfigFormat.JSON
    
    def _load_json(self) -> AppConfig:
        """Load JSON configuration."""
        with open(self.config_path, 'r') as f:
            data = json.load(f)
        
        return self._dict_to_config(data)
    
    def _load_yaml(self) -> AppConfig:
        """Load YAML configuration."""
        with open(self.config_path, 'r') as f:
            data = yaml.safe_load(f)
        
        return self._dict_to_config(data)
    
    def _load_env(self) -> AppConfig:
        """Load configuration from environment variables."""
        config = AppConfig()
        
        # Override with environment variables
        if os.getenv('APP_NAME'):
            config.app_name = os.getenv('APP_NAME')
        if os.getenv('DEBUG'):
            config.debug = os.getenv('DEBUG').lower() == 'true'
        if os.getenv('LOG_LEVEL'):
            config.log_level = os.getenv('LOG_LEVEL')
        
        # Database config
        if os.getenv('DB_HOST'):
            config.database.host = os.getenv('DB_HOST')
        if os.getenv('DB_PORT'):
            config.database.port = int(os.getenv('DB_PORT'))
        if os.getenv('DB_NAME'):
            config.database.name = os.getenv('DB_NAME')
        
        return config
    
    def _dict_to_config(self, data: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig."""
        config = AppConfig()
        
        # Top-level fields
        for key in ['app_name', 'version', 'debug', 'log_level']:
            if key in data:
                setattr(config, key, data[key])
        
        # Nested configs
        if 'database' in data:
            for key, value in data['database'].items():
                setattr(config.database, key, value)
        
        if 'robot' in data:
            for key, value in data['robot'].items():
                setattr(config.robot, key, value)
        
        if 'simulation' in data:
            for key, value in data['simulation'].items():
                setattr(config.simulation, key, value)
        
        if 'clinical' in data:
            for key, value in data['clinical'].items():
                setattr(config.clinical, key, value)
        
        return config
    
    def save_config(self, format: ConfigFormat = ConfigFormat.JSON):
        """Save configuration to file."""
        if not self.config_path:
            raise ValueError("No config path specified")
        
        data = self._config_to_dict()
        
        if format == ConfigFormat.JSON:
            self._save_json(data)
        elif format == ConfigFormat.YAML:
            self._save_yaml(data)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _config_to_dict(self) -> Dict[str, Any]:
        """Convert AppConfig to dictionary."""
        return asdict(self.config)
    
    def _save_json(self, data: Dict[str, Any]):
        """Save JSON configuration."""
        with open(self.config_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _save_yaml(self, data: Dict[str, Any]):
        """Save YAML configuration."""
        with open(self.config_path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    
    def get_config(self) -> AppConfig:
        """Get current configuration."""
        return self.config
    
    def update_config(self, **kwargs):
        """Update configuration values."""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                logger.warning(f"Unknown config key: {key}")

def main():
    """Main function for config manager."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration Manager')
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--format', type=str, default='json',
                       choices=['json', 'yaml'],
                       help='Configuration format')
    parser.add_argument('--action', type=str, default='load',
                       choices=['load', 'save', 'validate'],
                       help='Action to perform')
    
    args = parser.parse_args()
    
    if args.config:
        manager = ConfigManager(args.config)
    else:
        manager = ConfigManager()
    
    if args.action == 'load':
        config = manager.load_config()
        print(f"Loaded configuration for {config.app_name}")
    elif args.action == 'save':
        manager.save_config(ConfigFormat(args.format))
        print(f"Saved configuration to {manager.config_path}")
    elif args.action == 'validate':
        print("Configuration is valid")

if __name__ == '__main__':
    main()
