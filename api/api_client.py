#!/usr/bin/env python3
"""
API Client for External Services
HTTP client for interacting with external clinical and robotic services.
"""

import requests
import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from enum import Enum
import json
import time

logger = logging.getLogger(__name__)

class HttpMethod(Enum):
    """HTTP methods."""
    GET = "GET"
    POST = "POST"
    PUT = "PUT"
    DELETE = "DELETE"
    PATCH = "PATCH"

@dataclass
class APIResponse:
    """API response wrapper."""
    status_code: int
    data: Any
    headers: Dict[str, str]
    success: bool
    error: Optional[str] = None

class APIClient:
    """Generic API client for external services."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None,
                 timeout: int = 30, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        
        if api_key:
            self.session.headers.update({'Authorization': f'Bearer {api_key}'})
    
    def request(self, method: HttpMethod, endpoint: str,
                data: Optional[Dict[str, Any]] = None,
                params: Optional[Dict[str, Any]] = None,
                headers: Optional[Dict[str, str]] = None) -> APIResponse:
        """Make HTTP request with retry logic."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        request_headers = self.session.headers.copy()
        if headers:
            request_headers.update(headers)
        
        for attempt in range(self.max_retries):
            try:
                response = self.session.request(
                    method.value,
                    url,
                    json=data,
                    params=params,
                    headers=request_headers,
                    timeout=self.timeout
                )
                
                return APIResponse(
                    status_code=response.status_code,
                    data=response.json() if response.content else None,
                    headers=dict(response.headers),
                    success=response.status_code < 400,
                    error=response.text if response.status_code >= 400 else None
                )
                
            except requests.exceptions.Timeout:
                logger.warning(f"Request timeout (attempt {attempt + 1}/{self.max_retries})")
                if attempt < self.max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return APIResponse(
                        status_code=0,
                        data=None,
                        headers={},
                        success=False,
                        error="Request timeout"
                    )
            
            except requests.exceptions.RequestException as e:
                logger.error(f"Request failed: {e}")
                return APIResponse(
                    status_code=0,
                    data=None,
                    headers={},
                    success=False,
                    error=str(e)
                )
        
        return APIResponse(
            status_code=0,
            data=None,
            headers={},
            success=False,
            error="Max retries exceeded"
        )
    
    def get(self, endpoint: str, params: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make GET request."""
        return self.request(HttpMethod.GET, endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make POST request."""
        return self.request(HttpMethod.POST, endpoint, data=data)
    
    def put(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make PUT request."""
        return self.request(HttpMethod.PUT, endpoint, data=data)
    
    def delete(self, endpoint: str) -> APIResponse:
        """Make DELETE request."""
        return self.request(HttpMethod.DELETE, endpoint)
    
    def patch(self, endpoint: str, data: Optional[Dict[str, Any]] = None) -> APIResponse:
        """Make PATCH request."""
        return self.request(HttpMethod.PATCH, endpoint, data=data)

class ClinicalAPIClient(APIClient):
    """API client for clinical services."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        super().__init__(base_url, api_key)
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def get_patient_data(self, patient_id: str) -> APIResponse:
        """Get patient data."""
        return self.get(f"/patients/{patient_id}")
    
    def get_medication_info(self, medication_id: str) -> APIResponse:
        """Get medication information."""
        return self.get(f"/medications/{medication_id}")
    
    def submit_prescription(self, prescription: Dict[str, Any]) -> APIResponse:
        """Submit prescription."""
        return self.post("/prescriptions", data=prescription)
    
    def get_lab_results(self, patient_id: str) -> APIResponse:
        """Get lab results for patient."""
        return self.get(f"/patients/{patient_id}/lab-results")

class RobotAPIClient(APIClient):
    """API client for robot control services."""
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        super().__init__(base_url, api_key)
        self.session.headers.update({'Content-Type': 'application/json'})
    
    def get_robot_status(self, robot_id: str) -> APIResponse:
        """Get robot status."""
        return self.get(f"/robots/{robot_id}/status")
    
    def send_command(self, robot_id: str, command: Dict[str, Any]) -> APIResponse:
        """Send command to robot."""
        return self.post(f"/robots/{robot_id}/commands", data=command)
    
    def get_robot_logs(self, robot_id: str) -> APIResponse:
        """Get robot logs."""
        return self.get(f"/robots/{robot_id}/logs")
    
    def emergency_stop(self, robot_id: str) -> APIResponse:
        """Emergency stop robot."""
        return self.post(f"/robots/{robot_id}/emergency-stop")

def main():
    """Main function for API client."""
    import argparse
    
    parser = argparse.ArgumentParser(description='API Client')
    parser.add_argument('--base-url', type=str, required=True, help='Base URL')
    parser.add_argument('--api-key', type=str, help='API key')
    parser.add_argument('--type', type=str, default='generic',
                       choices=['generic', 'clinical', 'robot'],
                       help='Client type')
    
    args = parser.parse_args()
    
    if args.type == 'clinical':
        client = ClinicalAPIClient(args.base_url, args.api_key)
    elif args.type == 'robot':
        client = RobotAPIClient(args.base_url, args.api_key)
    else:
        client = APIClient(args.base_url, args.api_key)
    
    # Test connection
    response = client.get("/")
    print(f"Status: {response.status_code}")
    print(f"Success: {response.success}")
    if response.data:
        print(f"Data: {json.dumps(response.data, indent=2)}")

if __name__ == '__main__':
    main()
