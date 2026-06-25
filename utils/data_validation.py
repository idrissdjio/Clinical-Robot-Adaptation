#!/usr/bin/env python3
"""
Data Validation Utilities
Validation and sanitization utilities for clinical robotics data.
"""

import re
import logging
from typing import Any, Dict, List, Optional, Union, Type
from dataclasses import dataclass
from enum import Enum
from datetime import datetime
import json

logger = logging.getLogger(__name__)

class ValidationType(Enum):
    """Types of validation."""
    REQUIRED = "required"
    TYPE = "type"
    RANGE = "range"
    PATTERN = "pattern"
    LENGTH = "length"
    CUSTOM = "custom"

@dataclass
class ValidationResult:
    """Result of validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    
    def add_error(self, error: str):
        """Add error to result."""
        self.errors.append(error)
        self.is_valid = False
    
    def add_warning(self, warning: str):
        """Add warning to result."""
        self.warnings.append(warning)

class DataValidator:
    """Data validator for clinical robotics applications."""
    
    def __init__(self):
        self.validation_rules = {}
    
    def validate(self, data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
        """Validate data against schema."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        for field, rules in schema.items():
            if field not in data:
                if rules.get('required', False):
                    result.add_error(f"Required field '{field}' is missing")
                continue
            
            value = data[field]
            
            # Type validation
            if 'type' in rules:
                if not self._validate_type(value, rules['type']):
                    result.add_error(f"Field '{field}' has invalid type. Expected {rules['type']}")
            
            # Range validation
            if 'range' in rules:
                if not self._validate_range(value, rules['range']):
                    result.add_error(f"Field '{field}' is out of range {rules['range']}")
            
            # Pattern validation
            if 'pattern' in rules:
                if not self._validate_pattern(value, rules['pattern']):
                    result.add_error(f"Field '{field}' does not match pattern")
            
            # Length validation
            if 'length' in rules:
                if not self._validate_length(value, rules['length']):
                    result.add_error(f"Field '{field}' has invalid length")
            
            # Custom validation
            if 'validator' in rules:
                if not rules['validator'](value):
                    result.add_error(f"Field '{field}' failed custom validation")
        
        return result
    
    def _validate_type(self, value: Any, expected_type: Union[Type, str]) -> bool:
        """Validate data type."""
        if isinstance(expected_type, str):
            type_map = {
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'list': list,
                'dict': dict
            }
            expected_type = type_map.get(expected_type, str)
        
        return isinstance(value, expected_type)
    
    def _validate_range(self, value: Any, range_spec: Dict[str, Any]) -> bool:
        """Validate value range."""
        if not isinstance(value, (int, float)):
            return False
        
        if 'min' in range_spec and value < range_spec['min']:
            return False
        
        if 'max' in range_spec and value > range_spec['max']:
            return False
        
        return True
    
    def _validate_pattern(self, value: Any, pattern: str) -> bool:
        """Validate string pattern."""
        if not isinstance(value, str):
            return False
        
        return bool(re.match(pattern, value))
    
    def _validate_length(self, value: Any, length_spec: Dict[str, Any]) -> bool:
        """Validate length of value."""
        if isinstance(value, str):
            length = len(value)
        elif isinstance(value, (list, dict)):
            length = len(value)
        else:
            return True
        
        if 'min' in length_spec and length < length_spec['min']:
            return False
        
        if 'max' in length_spec and length > length_spec['max']:
            return False
        
        return True

class ClinicalDataValidator(DataValidator):
    """Validator for clinical-specific data."""
    
    def validate_patient_id(self, patient_id: str) -> ValidationResult:
        """Validate patient ID format."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not patient_id:
            result.add_error("Patient ID is required")
            return result
        
        if not re.match(r'^[A-Z]{2}\d{6}$', patient_id):
            result.add_error("Patient ID must match format: 2 letters + 6 digits (e.g., AB123456)")
        
        return result
    
    def validate_medication_dosage(self, dosage: float) -> ValidationResult:
        """Validate medication dosage."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if dosage <= 0:
            result.add_error("Dosage must be positive")
        
        if dosage > 1000:
            result.add_warning("Dosage seems unusually high")
        
        return result
    
    def validate_robot_position(self, position: Dict[str, float]) -> ValidationResult:
        """Validate robot position coordinates."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        required_fields = ['x', 'y', 'z']
        for field in required_fields:
            if field not in position:
                result.add_error(f"Missing coordinate: {field}")
            elif not isinstance(position[field], (int, float)):
                result.add_error(f"Coordinate {field} must be numeric")
        
        # Check for reasonable workspace bounds
        if 'x' in position and abs(position['x']) > 10:
            result.add_warning("X coordinate is outside typical workspace")
        
        if 'y' in position and abs(position['y']) > 10:
            result.add_warning("Y coordinate is outside typical workspace")
        
        if 'z' in position and (position['z'] < 0 or position['z'] > 3):
            result.add_warning("Z coordinate is outside typical workspace")
        
        return result
    
    def validate_timestamp(self, timestamp: str) -> ValidationResult:
        """Validate timestamp format."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[])
        
        if not timestamp:
            result.add_error("Timestamp is required")
            return result
        
        try:
            datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
        except ValueError:
            result.add_error("Invalid timestamp format. Use ISO 8601 format")
        
        return result

class Sanitizer:
    """Data sanitization utilities."""
    
    @staticmethod
    def sanitize_string(value: str, max_length: int = 1000) -> str:
        """Sanitize string input."""
        if not isinstance(value, str):
            return str(value)
        
        # Remove null bytes
        value = value.replace('\x00', '')
        
        # Trim whitespace
        value = value.strip()
        
        # Truncate if too long
        if len(value) > max_length:
            value = value[:max_length]
        
        return value
    
    @staticmethod
    def sanitize_numeric(value: Any, min_val: Optional[float] = None,
                       max_val: Optional[float] = None) -> Optional[float]:
        """Sanitize numeric input."""
        try:
            num = float(value)
            
            if min_val is not None and num < min_val:
                num = min_val
            
            if max_val is not None and num > max_val:
                num = max_val
            
            return num
        except (ValueError, TypeError):
            return None
    
    @staticmethod
    def sanitize_json(value: str) -> Optional[Dict]:
        """Sanitize and parse JSON."""
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None

def validate_schema(data: Dict[str, Any], schema: Dict[str, Any]) -> ValidationResult:
    """Convenience function for schema validation."""
    validator = DataValidator()
    return validator.validate(data, schema)

if __name__ == '__main__':
    # Example usage
    validator = ClinicalDataValidator()
    
    # Test patient ID validation
    result = validator.validate_patient_id("AB123456")
    print(f"Patient ID validation: {result.is_valid}")
    print(f"Errors: {result.errors}")
    
    # Test robot position validation
    position = {'x': 1.5, 'y': 2.0, 'z': 0.5}
    result = validator.validate_robot_position(position)
    print(f"Robot position validation: {result.is_valid}")
    print(f"Warnings: {result.warnings}")
