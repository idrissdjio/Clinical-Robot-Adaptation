#!/usr/bin/env python3
"""
OpenAPI Documentation Generator for Clinical Robot Adaptation API
Comprehensive API documentation with interactive examples and schemas.

This module generates:
- Complete OpenAPI 3.0 specification
- Interactive API documentation
- Request/response schemas
- Authentication documentation
- Error handling documentation
- Usage examples and tutorials

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import yaml

# FastAPI imports for OpenAPI generation
from fastapi import FastAPI, HTTPException, status
from fastapi.openapi.utils import get_openapi
from pydantic import BaseModel, Field
from pydantic.types import StrictBool, StrictInt, StrictFloat, StrictStr

# Project imports
sys.path.append(str(Path(__file__).parent.parent))

class OpenAPIGenerator:
    """Generate comprehensive OpenAPI documentation."""
    
    def __init__(self, app: FastAPI):
        self.app = app
        self.base_spec = self._get_base_spec()
        self.custom_schemas = self._define_custom_schemas()
        self.examples = self._define_examples()
        self.tags = self._define_tags()
    
    def _get_base_spec(self) -> Dict[str, Any]:
        """Get base OpenAPI specification."""
        return {
            "openapi": "3.0.0",
            "info": {
                "title": "Clinical Robot Adaptation API",
                "description": "Comprehensive API for clinical robot adaptation, including model inference, training, data processing, and monitoring.",
                "version": "2.0.0",
                "contact": {
                    "name": "Idriss Djiofack Teledjieu",
                    "email": "idriss.djiofack@colorado.edu",
                    "url": "https://github.com/idrissdjio/Clinical-Robot-Adaptation"
                },
                "license": {
                    "name": "MIT License",
                    "url": "https://opensource.org/licenses/MIT"
                }
            },
            "servers": [
                {
                    "url": "http://localhost:8000",
                    "description": "Development server"
                },
                {
                    "url": "https://api.clinical-robot.com",
                    "description": "Production server"
                }
            ],
            "paths": {},
            "components": {
                "schemas": {},
                "securitySchemes": {},
                "responses": {},
                "examples": {}
            },
            "tags": []
        }
    
    def _define_custom_schemas(self) -> Dict[str, Dict[str, Any]]:
        """Define custom schemas for API documentation."""
        return {
            "InferenceRequest": {
                "type": "object",
                "required": ["model_version"],
                "properties": {
                    "model_version": {
                        "type": "string",
                        "description": "Model version to use for inference",
                        "example": "latest"
                    },
                    "image_data": {
                        "type": "string",
                        "description": "Base64 encoded image data",
                        "format": "base64",
                        "example": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg=="
                    },
                    "robot_state": {
                        "type": "array",
                        "items": {"type": "number"},
                        "description": "Robot joint states and pose",
                        "example": [0.1, -0.5, 0.2, 1.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                    },
                    "instruction": {
                        "type": "string",
                        "description": "Natural language instruction",
                        "example": "Pick up the medication vial from shelf A2"
                    },
                    "safety_level": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Safety level for the operation",
                        "example": "medium"
                    },
                    "context": {
                        "type": "object",
                        "description": "Additional context information",
                        "example": {
                            "urgency": "routine",
                            "patient_condition": "stable",
                            "medication_priority": "normal"
                        }
                    }
                }
            },
            
            "InferenceResponse": {
                "type": "object",
                "properties": {
                    "success": {
                        "type": "boolean",
                        "description": "Whether the inference was successful"
                    },
                    "prediction": {
                        "type": "object",
                        "description": "Model prediction results",
                        "example": {
                            "action": [0.1, -0.2, 0.3, 0.5, -0.1, 0.2, 0.0],
                            "grasp_type": "precision",
                            "medication_type": "vial",
                            "confidence": 0.92
                        }
                    },
                    "confidence": {
                        "type": "number",
                        "format": "float",
                        "description": "Prediction confidence score",
                        "example": 0.92
                    },
                    "safety_score": {
                        "type": "number",
                        "format": "float",
                        "description": "Safety assessment score",
                        "example": 0.95
                    },
                    "processing_time": {
                        "type": "number",
                        "format": "float",
                        "description": "Processing time in seconds",
                        "example": 0.045
                    },
                    "model_version": {
                        "type": "string",
                        "description": "Model version used",
                        "example": "latest"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Timestamp of the inference",
                        "example": "2024-01-15T10:30:00Z"
                    },
                    "warnings": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Warning messages",
                        "example": ["Low confidence in grasp prediction"]
                    }
                }
            },
            
            "TrainingRequest": {
                "type": "object",
                "required": ["model_version", "dataset_path"],
                "properties": {
                    "model_version": {
                        "type": "string",
                        "description": "Base model version to fine-tune",
                        "example": "v1.0"
                    },
                    "dataset_path": {
                        "type": "string",
                        "description": "Path to training dataset",
                        "example": "/data/clinical_demonstrations.hdf5"
                    },
                    "num_epochs": {
                        "type": "integer",
                        "description": "Number of training epochs",
                        "example": 100,
                        "default": 100
                    },
                    "learning_rate": {
                        "type": "number",
                        "format": "float",
                        "description": "Learning rate for training",
                        "example": 0.0001,
                        "default": 0.0001
                    },
                    "batch_size": {
                        "type": "integer",
                        "description": "Batch size for training",
                        "example": 16,
                        "default": 16
                    },
                    "safety_weight": {
                        "type": "number",
                        "format": "float",
                        "description": "Weight for safety loss component",
                        "example": 0.3,
                        "default": 0.3
                    },
                    "human_weight": {
                        "type": "number",
                        "format": "float",
                        "description": "Weight for human awareness loss component",
                        "example": 0.2,
                        "default": 0.2
                    }
                }
            },
            
            "TrainingStatus": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["running", "completed", "failed", "paused"],
                        "description": "Current training status"
                    },
                    "progress": {
                        "type": "number",
                        "format": "float",
                        "description": "Training progress (0.0 to 1.0)",
                        "example": 0.67
                    },
                    "current_epoch": {
                        "type": "integer",
                        "description": "Current epoch number",
                        "example": 67
                    },
                    "total_epochs": {
                        "type": "integer",
                        "description": "Total number of epochs",
                        "example": 100
                    },
                    "loss": {
                        "type": "number",
                        "format": "float",
                        "description": "Current loss value",
                        "example": 0.234
                    },
                    "accuracy": {
                        "type": "number",
                        "format": "float",
                        "description": "Current accuracy",
                        "example": 0.891
                    },
                    "safety_score": {
                        "type": "number",
                        "format": "float",
                        "description": "Current safety score",
                        "example": 0.967
                    },
                    "eta_seconds": {
                        "type": "integer",
                        "description": "Estimated time remaining in seconds",
                        "example": 1200
                    },
                    "logs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Recent log entries",
                        "example": ["Epoch 67/100 - Loss: 0.234, Accuracy: 0.891"]
                    }
                }
            },
            
            "SafetyAlert": {
                "type": "object",
                "properties": {
                    "alert_type": {
                        "type": "string",
                        "description": "Type of safety alert",
                        "example": "human_proximity"
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["low", "medium", "high", "critical"],
                        "description": "Alert severity level"
                    },
                    "message": {
                        "type": "string",
                        "description": "Alert message",
                        "example": "Human detected in workspace - maintaining safe distance"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Alert timestamp",
                        "example": "2024-01-15T10:30:00Z"
                    },
                    "robot_id": {
                        "type": "string",
                        "description": "Robot identifier",
                        "example": "robot_001"
                    },
                    "location": {
                        "type": "object",
                        "description": "Alert location",
                        "example": {
                            "x": 0.5,
                            "y": -0.3,
                            "z": 0.8
                        }
                    },
                    "resolved": {
                        "type": "boolean",
                        "description": "Whether the alert has been resolved",
                        "example": false
                    }
                }
            },
            
            "PerformanceMetrics": {
                "type": "object",
                "properties": {
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Metrics timestamp",
                        "example": "2024-01-15T10:30:00Z"
                    },
                    "throughput": {
                        "type": "number",
                        "format": "float",
                        "description": "Requests per second",
                        "example": 12.5
                    },
                    "latency_p50": {
                        "type": "number",
                        "format": "float",
                        "description": "50th percentile latency in milliseconds",
                        "example": 38.2
                    },
                    "latency_p95": {
                        "type": "number",
                        "format": "float",
                        "description": "95th percentile latency in milliseconds",
                        "example": 75.6
                    },
                    "latency_p99": {
                        "type": "number",
                        "format": "float",
                        "description": "99th percentile latency in milliseconds",
                        "example": 120.4
                    },
                    "error_rate": {
                        "type": "number",
                        "format": "float",
                        "description": "Error rate (0.0 to 1.0)",
                        "example": 0.02
                    },
                    "cpu_usage": {
                        "type": "number",
                        "format": "float",
                        "description": "CPU usage percentage",
                        "example": 45.2
                    },
                    "memory_usage": {
                        "type": "number",
                        "format": "float",
                        "description": "Memory usage percentage",
                        "example": 67.8
                    },
                    "gpu_usage": {
                        "type": "number",
                        "format": "float",
                        "description": "GPU usage percentage",
                        "example": 78.5
                    }
                }
            },
            
            "ErrorResponse": {
                "type": "object",
                "properties": {
                    "error": {
                        "type": "string",
                        "description": "Error message",
                        "example": "Model not found"
                    },
                    "error_code": {
                        "type": "string",
                        "description": "Error code",
                        "example": "MODEL_NOT_FOUND"
                    },
                    "timestamp": {
                        "type": "string",
                        "format": "date-time",
                        "description": "Error timestamp",
                        "example": "2024-01-15T10:30:00Z"
                    },
                    "details": {
                        "type": "object",
                        "description": "Additional error details",
                        "example": {
                            "model_id": "invalid_model",
                            "available_models": ["latest", "v1.0", "v1.1"]
                        }
                    }
                }
            }
        }
    
    def _define_examples(self) -> Dict[str, Dict[str, Any]]:
        """Define API examples."""
        return {
            "inference_example": {
                "summary": "Basic inference request",
                "description": "Example of a basic inference request with image and robot state",
                "value": {
                    "model_version": "latest",
                    "image_data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==",
                    "robot_state": [0.1, -0.5, 0.2, 1.5, -0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                    "instruction": "Pick up the medication vial from shelf A2",
                    "safety_level": "medium",
                    "context": {
                        "urgency": "routine",
                        "patient_condition": "stable"
                    }
                }
            },
            
            "training_example": {
                "summary": "Model training request",
                "description": "Example of starting a model training job",
                "value": {
                    "model_version": "v1.0",
                    "dataset_path": "/data/clinical_demonstrations.hdf5",
                    "num_epochs": 100,
                    "learning_rate": 0.0001,
                    "batch_size": 16,
                    "safety_weight": 0.3,
                    "human_weight": 0.2
                }
            },
            
            "safety_alert_example": {
                "summary": "Safety alert example",
                "description": "Example of a safety alert notification",
                "value": {
                    "alert_type": "human_proximity",
                    "severity": "warning",
                    "message": "Human detected in workspace - maintaining safe distance",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "robot_id": "robot_001",
                    "location": {
                        "x": 0.5,
                        "y": -0.3,
                        "z": 0.8
                    },
                    "resolved": false
                }
            }
        }
    
    def _define_tags(self) -> List[Dict[str, Any]]:
        """Define API tags for organization."""
        return [
            {
                "name": "Inference",
                "description": "Model inference and prediction endpoints"
            },
            {
                "name": "Training",
                "description": "Model training and fine-tuning endpoints"
            },
            {
                "name": "Data",
                "description": "Data processing and management endpoints"
            },
            {
                "name": "Safety",
                "description": "Safety monitoring and alert endpoints"
            },
            {
                "name": "Monitoring",
                "description": "System monitoring and metrics endpoints"
            },
            {
                "name": "Authentication",
                "description": "User authentication and authorization"
            }
        ]
    
    def _define_security_schemes(self) -> Dict[str, Any]:
        """Define security schemes."""
        return {
            "BearerAuth": {
                "type": "http",
                "scheme": "bearer",
                "bearerFormat": "JWT",
                "description": "JWT authentication token"
            },
            "ApiKeyAuth": {
                "type": "apiKey",
                "in": "header",
                "name": "X-API-Key",
                "description": "API key for service-to-service authentication"
            },
            "OAuth2": {
                "type": "oauth2",
                "flows": {
                    "authorizationCode": {
                        "authorizationUrl": "https://api.clinical-robot.com/oauth/authorize",
                        "tokenUrl": "https://api.clinical-robot.com/oauth/token",
                        "scopes": {
                            "read": "Read access to resources",
                            "write": "Write access to resources",
                            "admin": "Administrative access"
                        }
                    }
                }
            }
        }
    
    def _define_responses(self) -> Dict[str, Any]:
        """Define common response schemas."""
        return {
            "SuccessResponse": {
                "description": "Successful operation",
                "content": {
                    "application/json": {
                        "schema": {"type": "object"},
                        "example": {"success": True, "message": "Operation completed successfully"}
                    }
                }
            },
            
            "ErrorResponse": {
                "description": "Error response",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            
            "ValidationError": {
                "description": "Validation error",
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {
                                "detail": {
                                    "type": "array",
                                    "items": {
                                        "type": "object",
                                        "properties": {
                                            "loc": {"type": "array", "items": {"type": "string"}},
                                            "msg": {"type": "string"},
                                            "type": {"type": "string"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            
            "UnauthorizedError": {
                "description": "Unauthorized access",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            
            "ForbiddenError": {
                "description": "Access forbidden",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            },
            
            "NotFoundError": {
                "description": "Resource not found",
                "content": {
                    "application/json": {
                        "schema": {"$ref": "#/components/schemas/ErrorResponse"}
                    }
                }
            }
        }
    
    def generate_openapi_spec(self) -> Dict[str, Any]:
        """Generate complete OpenAPI specification."""
        # Start with base spec
        spec = self.base_spec.copy()
        
        # Add custom schemas
        spec["components"]["schemas"].update(self.custom_schemas)
        
        # Add security schemes
        spec["components"]["securitySchemes"] = self._define_security_schemes()
        
        # Add common responses
        spec["components"]["responses"] = self._define_responses()
        
        # Add examples
        spec["components"]["examples"] = self.examples
        
        # Add tags
        spec["tags"] = self.tags
        
        # Add paths from FastAPI app
        if self.app:
            fastapi_spec = get_openapi(
                title=spec["info"]["title"],
                version=spec["info"]["version"],
                description=spec["info"]["description"],
                routes=self.app.routes,
                openapi_version=spec["openapi"]
            )
            
            # Merge paths
            spec["paths"].update(fastapi_spec.get("paths", {}))
            
            # Merge additional schemas from FastAPI
            spec["components"]["schemas"].update(fastapi_spec.get("components", {}).get("schemas", {}))
        
        return spec
    
    def save_openapi_spec(self, output_path: str, format: str = "yaml") -> str:
        """Save OpenAPI specification to file."""
        spec = self.generate_openapi_spec()
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == "yaml":
            with open(output_file, 'w') as f:
                yaml.dump(spec, f, default_flow_style=False, sort_keys=False)
        else:  # JSON
            with open(output_file, 'w') as f:
                json.dump(spec, f, indent=2, sort_keys=False)
        
        return str(output_file)
    
    def generate_postman_collection(self, output_path: str) -> str:
        """Generate Postman collection from OpenAPI spec."""
        spec = self.generate_openapi_spec()
        
        collection = {
            "info": {
                "name": spec["info"]["title"],
                "description": spec["info"]["description"],
                "version": spec["info"]["version"],
                "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
            },
            "item": [],
            "variable": [
                {
                    "key": "baseUrl",
                    "value": spec["servers"][0]["url"] if spec["servers"] else "http://localhost:8000",
                    "type": "string"
                },
                {
                    "key": "apiKey",
                    "value": "",
                    "type": "string"
                },
                {
                    "key": "bearerToken",
                    "value": "",
                    "type": "string"
                }
            ]
        }
        
        # Convert OpenAPI paths to Postman items
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    postman_item = {
                        "name": operation.get("summary", f"{method.upper()} {path}"),
                        "request": {
                            "method": method.upper(),
                            "header": [],
                            "url": {
                                "raw": "{{baseUrl}}" + path,
                                "host": ["{{baseUrl}}"],
                                "path": path.strip("/").split("/") if path != "/" else []
                            }
                        }
                    }
                    
                    # Add headers
                    if "security" in operation:
                        for security_scheme in operation["security"]:
                            for scheme_name in security_scheme:
                                if scheme_name == "BearerAuth":
                                    postman_item["request"]["header"].append({
                                        "key": "Authorization",
                                        "value": "Bearer {{bearerToken}}",
                                        "type": "text"
                                    })
                                elif scheme_name == "ApiKeyAuth":
                                    postman_item["request"]["header"].append({
                                        "key": "X-API-Key",
                                        "value": "{{apiKey}}",
                                        "type": "text"
                                    })
                    
                    # Add body if present
                    if "requestBody" in operation:
                        content = operation["requestBody"].get("content", {})
                        if "application/json" in content:
                            schema = content["application/json"].get("schema", {})
                            if "example" in content["application/json"]:
                                postman_item["request"]["body"] = {
                                    "mode": "raw",
                                    "raw": json.dumps(content["application/json"]["example"], indent=2),
                                    "options": {
                                        "raw": {
                                            "language": "json"
                                        }
                                    }
                                }
                    
                    # Add description
                    if "description" in operation:
                        postman_item["request"]["description"] = operation["description"]
                    
                    collection["item"].append(postman_item)
        
        # Save Postman collection
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            json.dump(collection, f, indent=2)
        
        return str(output_file)
    
    def generate_markdown_docs(self, output_path: str) -> str:
        """Generate Markdown documentation from OpenAPI spec."""
        spec = self.generate_openapi_spec()
        
        markdown_content = f"""# {spec['info']['title']} API Documentation

{spec['info']['description']}

**Version:** {spec['info']['version']}  
**Contact:** {spec['info']['contact']['name']} ({spec['info']['contact']['email']})  
**License:** {spec['info']['license']['name']}

## Table of Contents

- [Authentication](#authentication)
- [Endpoints](#endpoints)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [SDKs and Libraries](#sdks-and-libraries)

## Authentication

This API uses multiple authentication methods:

### JWT Bearer Token

```bash
curl -H "Authorization: Bearer YOUR_JWT_TOKEN" \\
     https://api.example.com/endpoint
```

### API Key

```bash
curl -H "X-API-Key: YOUR_API_KEY" \\
     https://api.example.com/endpoint
```

## Base URL

{spec['servers'][0]['url'] if spec['servers'] else 'http://localhost:8000'}

## Endpoints

"""
        
        # Add endpoints documentation
        for path, path_item in spec.get("paths", {}).items():
            for method, operation in path_item.items():
                if method.upper() in ["GET", "POST", "PUT", "DELETE", "PATCH"]:
                    markdown_content += f"""
### {method.upper()} {path}

{operation.get('description', 'No description available')}

**Tags:** {', '.join(tag.get('name', '') for tag in operation.get('tags', []))}

**Parameters:**
"""
                    
                    # Add parameters
                    parameters = operation.get('parameters', [])
                    if parameters:
                        for param in parameters:
                            param_name = param.get('name', '')
                            param_type = param.get('in', '')
                            param_required = param.get('required', False)
                            param_desc = param.get('description', '')
                            
                            markdown_content += f"- `{param_name}` ({param_type}): {param_desc}{' (required)' if param_required else ''}\n"
                    else:
                        markdown_content += "None\n"
                    
                    # Add request body
                    if 'requestBody' in operation:
                        markdown_content += "\n**Request Body:**\n"
                        content = operation['requestBody'].get('content', {})
                        if 'application/json' in content:
                            schema_ref = content['application/json'].get('schema', {}).get('$ref', '')
                            if schema_ref:
                                schema_name = schema_ref.split('/')[-1]
                                markdown_content += f"See [{schema_name}](#schema-{schema_name.lower()})\n"
                    
                    # Add responses
                    markdown_content += "\n**Responses:**\n"
                    responses = operation.get('responses', {})
                    for status_code, response in responses.items():
                        description = response.get('description', '')
                        markdown_content += f"- `{status_code}`: {description}\n"
                    
                    markdown_content += "\n---\n"
        
        # Add schema documentation
        if spec.get("components", {}).get("schemas"):
            markdown_content += "\n## Schemas\n\n"
            
            for schema_name, schema in spec["components"]["schemas"].items():
                markdown_content += f"### {schema_name}\n\n"
                
                if "description" in schema:
                    markdown_content += f"{schema['description']}\n\n"
                
                if "properties" in schema:
                    markdown_content += "**Properties:**\n"
                    for prop_name, prop_schema in schema["properties"].items():
                        prop_type = prop_schema.get("type", "unknown")
                        prop_desc = prop_schema.get("description", "")
                        prop_required = prop_name in schema.get("required", [])
                        
                        markdown_content += f"- `{prop_name}` ({prop_type}): {prop_desc}{' (required)' if prop_required else ''}\n"
                
                markdown_content += "\n---\n"
        
        # Save Markdown documentation
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)
        
        return str(output_file)

def create_openapi_documentation(app: FastAPI, output_dir: str = "./docs/api") -> Dict[str, str]:
    """Create comprehensive API documentation."""
    generator = OpenAPIGenerator(app)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Generate different formats
    files_created = {}
    
    # OpenAPI YAML
    yaml_file = generator.save_openapi_spec(
        output_path / "openapi.yaml", 
        format="yaml"
    )
    files_created["yaml"] = yaml_file
    
    # OpenAPI JSON
    json_file = generator.save_openapi_spec(
        output_path / "openapi.json", 
        format="json"
    )
    files_created["json"] = json_file
    
    # Postman collection
    postman_file = generator.generate_postman_collection(
        output_path / "postman_collection.json"
    )
    files_created["postman"] = postman_file
    
    # Markdown documentation
    markdown_file = generator.generate_markdown_docs(
        output_path / "README.md"
    )
    files_created["markdown"] = markdown_file
    
    return files_created

if __name__ == "__main__":
    # Example usage
    from fastapi import FastAPI
    
    app = FastAPI(title="Clinical Robot Adaptation API")
    
    # Create documentation
    files = create_openapi_documentation(app)
    
    print("API documentation created:")
    for format_type, file_path in files.items():
        print(f"  {format_type}: {file_path}")
