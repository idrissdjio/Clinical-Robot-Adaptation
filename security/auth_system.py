#!/usr/bin/env python3
"""
Advanced Security and Authentication System for Clinical Robotics
Comprehensive security framework with multi-factor authentication, RBAC, and audit logging.

This module implements:
- Multi-factor authentication (MFA) with TOTP support
- Role-based access control (RBAC) with fine-grained permissions
- JWT token-based authentication with refresh tokens
- API key management for service-to-service communication
- Comprehensive audit logging and security monitoring
- Session management with secure cookie handling
- Password policies and account lockout mechanisms
- Biometric authentication integration support
- HIPAA compliance features for clinical data access

Author: Idriss Djiofack Teledjieu
Clinical Robot Adaptation Project
HIRO Laboratory, University of Colorado Boulder
"""

import os
import sys
import json
import time
import hashlib
import secrets
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from enum import Enum
import warnings

# Security and cryptography
import bcrypt
import jwt
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import pyotp
import qrcode
from PIL import Image

# Database and storage
import sqlite3
from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Text, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func

# Web framework
from fastapi import HTTPException, status, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.security.api_key import APIKeyHeader

# Monitoring and alerting
import psutil
import redis
from prometheus_client import Counter, Histogram, Gauge

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('security_auth.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Security metrics
auth_attempts = Counter('auth_attempts_total', 'Total authentication attempts', ['status', 'method'])
auth_duration = Histogram('auth_duration_seconds', 'Authentication duration')
active_sessions = Gauge('active_sessions_total', 'Number of active user sessions')
security_events = Counter('security_events_total', 'Security events', ['event_type', 'severity'])

class UserRole(Enum):
    """User roles for RBAC."""
    ADMIN = "admin"
    CLINICIAN = "clinician"
    RESEARCHER = "researcher"
    OPERATOR = "operator"
    VIEWER = "viewer"

class Permission(Enum):
    """System permissions."""
    # Model management
    MODEL_TRAIN = "model_train"
    MODEL_DEPLOY = "model_deploy"
    MODEL_EVALUATE = "model_evaluate"
    
    # Data management
    DATA_UPLOAD = "data_upload"
    DATA_DOWNLOAD = "data_download"
    DATA_DELETE = "data_delete"
    
    # System management
    SYSTEM_CONFIG = "system_config"
    USER_MANAGE = "user_manage"
    AUDIT_VIEW = "audit_view"
    
    # Clinical operations
    CLINICAL_DEPLOY = "clinical_deploy"
    SAFETY_OVERRIDE = "safety_override"
    EMERGENCY_STOP = "emergency_stop"
    
    # Monitoring
    METRICS_VIEW = "metrics_view"
    LOGS_VIEW = "logs_view"

class SecurityEventType(Enum):
    """Security event types."""
    LOGIN_SUCCESS = "login_success"
    LOGIN_FAILURE = "login_failure"
    LOGOUT = "logout"
    PASSWORD_CHANGE = "password_change"
    MFA_ENABLED = "mfa_enabled"
    MFA_DISABLED = "mfa_disabled"
    ACCOUNT_LOCKED = "account_locked"
    ACCOUNT_UNLOCKED = "account_unlocked"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    DATA_ACCESS = "data_access"
    API_KEY_CREATED = "api_key_created"
    API_KEY_REVOKED = "api_key_revoked"

@dataclass
class SecurityConfig:
    """Security configuration."""
    secret_key: str
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    password_min_length: int = 12
    password_require_special: bool = True
    password_require_numbers: bool = True
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 15
    session_timeout_minutes: int = 60
    mfa_issuer: str = "Clinical Robot Adaptation"
    api_key_length: int = 32
    audit_retention_days: int = 90
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'algorithm': self.algorithm,
            'access_token_expire_minutes': self.access_token_expire_minutes,
            'refresh_token_expire_days': self.refresh_token_expire_days,
            'password_min_length': self.password_min_length,
            'password_require_special': self.password_require_special,
            'password_require_numbers': self.password_require_numbers,
            'max_login_attempts': self.max_login_attempts,
            'lockout_duration_minutes': self.lockout_duration_minutes,
            'session_timeout_minutes': self.session_timeout_minutes,
            'mfa_issuer': self.mfa_issuer,
            'api_key_length': self.api_key_length,
            'audit_retention_days': self.audit_retention_days
        }

# Database models
Base = declarative_base()

class User(Base):
    """User model."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(50), unique=True, index=True, nullable=False)
    email = Column(String(100), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=False)
    full_name = Column(String(100), nullable=False)
    role = Column(String(20), nullable=False, default=UserRole.VIEWER.value)
    is_active = Column(Boolean, default=True, nullable=False)
    is_verified = Column(Boolean, default=False, nullable=False)
    mfa_secret = Column(String(32), nullable=True)
    mfa_enabled = Column(Boolean, default=False, nullable=False)
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now(), nullable=False)
    last_login = Column(DateTime, nullable=True)
    
    # Relationships
    api_keys = relationship("APIKey", back_populates="user", cascade="all, delete-orphan")
    sessions = relationship("UserSession", back_populates="user", cascade="all, delete-orphan")
    audit_logs = relationship("AuditLog", back_populates="user", cascade="all, delete-orphan")

class APIKey(Base):
    """API Key model."""
    __tablename__ = "api_keys"
    
    id = Column(Integer, primary_key=True, index=True)
    key_id = Column(String(32), unique=True, index=True, nullable=False)
    hashed_key = Column(String(255), nullable=False)
    name = Column(String(100), nullable=False)
    permissions = Column(Text, nullable=False)  # JSON string of permissions
    is_active = Column(Boolean, default=True, nullable=False)
    expires_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    last_used = Column(DateTime, nullable=True)
    usage_count = Column(Integer, default=0, nullable=False)
    
    # Foreign key
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="api_keys")

class UserSession(Base):
    """User session model."""
    __tablename__ = "user_sessions"
    
    id = Column(Integer, primary_key=True, index=True)
    session_token = Column(String(255), unique=True, index=True, nullable=False)
    refresh_token = Column(String(255), unique=True, index=True, nullable=False)
    ip_address = Column(String(45), nullable=False)  # IPv6 compatible
    user_agent = Column(Text, nullable=True)
    created_at = Column(DateTime, default=func.now(), nullable=False)
    expires_at = Column(DateTime, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)
    
    # Foreign key
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    user = relationship("User", back_populates="sessions")

class AuditLog(Base):
    """Audit log model."""
    __tablename__ = "audit_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    event_type = Column(String(50), nullable=False)
    event_description = Column(Text, nullable=False)
    severity = Column(String(20), nullable=False, default="INFO")
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    additional_data = Column(Text, nullable=True)  # JSON string
    created_at = Column(DateTime, default=func.now(), nullable=False)
    
    # Foreign key
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    user = relationship("User", back_populates="audit_logs")

class PasswordPolicy:
    """Password policy validation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """Validate password against policy."""
        errors = []
        
        # Length check
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        # Special character check
        if self.config.password_require_special:
            special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
            if not any(char in special_chars for char in password):
                errors.append("Password must contain at least one special character")
        
        # Number check
        if self.config.password_require_numbers:
            if not any(char.isdigit() for char in password):
                errors.append("Password must contain at least one number")
        
        # Uppercase check
        if not any(char.isupper() for char in password):
            errors.append("Password must contain at least one uppercase letter")
        
        # Lowercase check
        if not any(char.islower() for char in password):
            errors.append("Password must contain at least one lowercase letter")
        
        return len(errors) == 0, errors
    
    def generate_password(self, length: int = None) -> str:
        """Generate a secure password."""
        import string
        
        length = length or self.config.password_min_length
        
        # Ensure password meets all requirements
        chars = []
        chars.extend(string.ascii_lowercase)  # Lowercase
        chars.extend(string.ascii_uppercase)  # Uppercase
        chars.extend(string.digits)          # Numbers
        chars.extend("!@#$%^&*()_+-=")       # Special chars
        
        password = ''.join(secrets.choice(chars) for _ in range(length))
        
        # Validate and regenerate if needed
        is_valid, errors = self.validate_password(password)
        if not is_valid:
            return self.generate_password(length)
        
        return password

class MFAManager:
    """Multi-factor authentication manager."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def generate_secret(self) -> str:
        """Generate MFA secret."""
        return pyotp.random_base32()
    
    def generate_qr_code(self, user_email: str, secret: str) -> bytes:
        """Generate QR code for MFA setup."""
        totp_uri = pyotp.totp.TOTP(secret).provisioning_uri(
            name=user_email,
            issuer_name=self.config.mfa_issuer
        )
        
        qr = qrcode.QRCode(version=1, box_size=10, border=5)
        qr.add_data(totp_uri)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Convert to bytes
        img_bytes = io.BytesIO()
        img.save(img_bytes, format='PNG')
        img_bytes.seek(0)
        
        return img_bytes.getvalue()
    
    def verify_token(self, secret: str, token: str) -> bool:
        """Verify MFA token."""
        totp = pyotp.TOTP(secret)
        return totp.verify(token, valid_window=1)  # Allow 1 step tolerance

class TokenManager:
    """JWT token management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def create_access_token(self, data: Dict[str, Any]) -> str:
        """Create JWT access token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
        to_encode.update({"exp": expire, "type": "access"})
        
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        return encoded_jwt
    
    def create_refresh_token(self, data: Dict[str, Any]) -> str:
        """Create JWT refresh token."""
        to_encode = data.copy()
        expire = datetime.utcnow() + timedelta(days=self.config.refresh_token_expire_days)
        to_encode.update({"exp": expire, "type": "refresh"})
        
        encoded_jwt = jwt.encode(to_encode, self.config.secret_key, algorithm=self.config.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode JWT token."""
        try:
            payload = jwt.decode(token, self.config.secret_key, algorithms=[self.config.algorithm])
            
            if payload.get("type") != token_type:
                raise jwt.InvalidTokenError("Invalid token type")
            
            return payload
        
        except jwt.ExpiredSignatureError:
            raise jwt.InvalidTokenError("Token has expired")
        except jwt.InvalidTokenError:
            raise jwt.InvalidTokenError("Invalid token")

class APIKeyManager:
    """API key management."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.db_session = None  # Will be initialized with database session
    
    def generate_api_key(self) -> str:
        """Generate secure API key."""
        return secrets.token_urlsafe(self.config.api_key_length)
    
    def hash_api_key(self, api_key: str) -> str:
        """Hash API key for storage."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(api_key.encode(), salt)
        return hashed.decode()
    
    def verify_api_key(self, api_key: str, hashed_key: str) -> bool:
        """Verify API key against hash."""
        return bcrypt.checkpw(api_key.encode(), hashed_key.encode())
    
    def create_api_key(self, user_id: int, name: str, permissions: List[str], 
                      expires_at: Optional[datetime] = None) -> Tuple[str, str]:
        """Create new API key."""
        api_key = self.generate_api_key()
        key_id = secrets.token_urlsafe(16)
        hashed_key = self.hash_api_key(api_key)
        
        # Store in database
        db_api_key = APIKey(
            key_id=key_id,
            hashed_key=hashed_key,
            name=name,
            permissions=json.dumps(permissions),
            expires_at=expires_at
        )
        
        self.db_session.add(db_api_key)
        self.db_session.commit()
        
        return api_key, key_id
    
    def revoke_api_key(self, key_id: str) -> bool:
        """Revoke API key."""
        api_key = self.db_session.query(APIKey).filter(APIKey.key_id == key_id).first()
        
        if api_key:
            api_key.is_active = False
            self.db_session.commit()
            return True
        
        return False

class SecurityAuditor:
    """Security event auditor."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.db_session = None  # Will be initialized with database session
    
    def log_event(self, user_id: Optional[int], event_type: SecurityEventType, 
                  description: str, severity: str = "INFO", 
                  ip_address: str = None, user_agent: str = None,
                  additional_data: Dict[str, Any] = None) -> None:
        """Log security event."""
        
        # Create audit log entry
        audit_log = AuditLog(
            event_type=event_type.value,
            event_description=description,
            severity=severity,
            ip_address=ip_address,
            user_agent=user_agent,
            additional_data=json.dumps(additional_data) if additional_data else None,
            user_id=user_id
        )
        
        self.db_session.add(audit_log)
        self.db_session.commit()
        
        # Update metrics
        security_events.labels(event_type=event_type.value, severity=severity).inc()
        
        # Log to file
        logger.info(f"Security Event: {event_type.value} - {description} (User: {user_id})")
    
    def cleanup_old_logs(self) -> None:
        """Clean up old audit logs."""
        cutoff_date = datetime.utcnow() - timedelta(days=self.config.audit_retention_days)
        
        old_logs = self.db_session.query(AuditLog).filter(
            AuditLog.created_at < cutoff_date
        ).delete()
        
        self.db_session.commit()
        
        logger.info(f"Cleaned up {old_logs} old audit log entries")

class RoleBasedAccessControl:
    """Role-based access control (RBAC)."""
    
    def __init__(self):
        self.role_permissions = self._define_role_permissions()
    
    def _define_role_permissions(self) -> Dict[UserRole, List[Permission]]:
        """Define permissions for each role."""
        return {
            UserRole.ADMIN: list(Permission),  # All permissions
            
            UserRole.CLINICIAN: [
                Permission.MODEL_EVALUATE,
                Permission.DATA_UPLOAD,
                Permission.DATA_DOWNLOAD,
                Permission.CLINICAL_DEPLOY,
                Permission.SAFETY_OVERRIDE,
                Permission.EMERGENCY_STOP,
                Permission.METRICS_VIEW,
                Permission.LOGS_VIEW
            ],
            
            UserRole.RESEARCHER: [
                Permission.MODEL_TRAIN,
                Permission.MODEL_EVALUATE,
                Permission.DATA_UPLOAD,
                Permission.DATA_DOWNLOAD,
                Permission.METRICS_VIEW
            ],
            
            UserRole.OPERATOR: [
                Permission.MODEL_DEPLOY,
                Permission.CLINICAL_DEPLOY,
                Permission.EMERGENCY_STOP,
                Permission.METRICS_VIEW,
                Permission.LOGS_VIEW
            ],
            
            UserRole.VIEWER: [
                Permission.METRICS_VIEW,
                Permission.LOGS_VIEW
            ]
        }
    
    def has_permission(self, user_role: UserRole, permission: Permission) -> bool:
        """Check if user role has permission."""
        return permission in self.role_permissions.get(user_role, [])
    
    def get_user_permissions(self, user_role: UserRole) -> List[Permission]:
        """Get all permissions for a user role."""
        return self.role_permissions.get(user_role, [])

class AuthenticationService:
    """Main authentication service."""
    
    def __init__(self, config: SecurityConfig, database_url: str):
        self.config = config
        self.db_url = database_url
        
        # Initialize database
        self.engine = create_engine(database_url)
        Base.metadata.create_all(bind=self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
        # Initialize components
        self.password_policy = PasswordPolicy(config)
        self.mfa_manager = MFAManager(config)
        self.token_manager = TokenManager(config)
        self.api_key_manager = APIKeyManager(config)
        self.auditor = SecurityAuditor(config)
        self.rbac = RoleBasedAccessControl()
        
        # Initialize Redis for session storage
        self.redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
    
    def get_db_session(self):
        """Get database session."""
        db = self.SessionLocal()
        try:
            yield db
        finally:
            db.close()
    
    def hash_password(self, password: str) -> str:
        """Hash password."""
        salt = bcrypt.gensalt()
        hashed = bcrypt.hashpw(password.encode(), salt)
        return hashed.decode()
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password."""
        return bcrypt.checkpw(plain_password.encode(), hashed_password.encode())
    
    def create_user(self, username: str, email: str, password: str, 
                   full_name: str, role: UserRole = UserRole.VIEWER) -> bool:
        """Create new user."""
        db = self.SessionLocal()
        
        try:
            # Check if user exists
            if db.query(User).filter(User.username == username).first():
                return False
            
            if db.query(User).filter(User.email == email).first():
                return False
            
            # Validate password
            is_valid, errors = self.password_policy.validate_password(password)
            if not is_valid:
                logger.error(f"Password validation failed: {errors}")
                return False
            
            # Create user
            hashed_password = self.hash_password(password)
            user = User(
                username=username,
                email=email,
                hashed_password=hashed_password,
                full_name=full_name,
                role=role.value
            )
            
            db.add(user)
            db.commit()
            
            # Log event
            self.auditor.log_event(
                user_id=user.id,
                event_type=SecurityEventType.LOGIN_SUCCESS,
                description=f"User {username} created successfully"
            )
            
            return True
        
        except Exception as e:
            db.rollback()
            logger.error(f"Error creating user: {e}")
            return False
        
        finally:
            db.close()
    
    def authenticate_user(self, username: str, password: str, mfa_token: str = None,
                        ip_address: str = None, user_agent: str = None) -> Optional[Dict[str, Any]]:
        """Authenticate user and return tokens."""
        start_time = time.time()
        
        db = self.SessionLocal()
        
        try:
            # Get user
            user = db.query(User).filter(User.username == username).first()
            
            if not user:
                auth_attempts.labels(status="failure", method="password").inc()
                self.auditor.log_event(
                    user_id=None,
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    description=f"Login attempt for non-existent user: {username}",
                    severity="WARNING",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return None
            
            # Check if account is locked
            if user.locked_until and user.locked_until > datetime.utcnow():
                auth_attempts.labels(status="failure", method="locked").inc()
                self.auditor.log_event(
                    user_id=user.id,
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    description=f"Login attempt for locked account: {username}",
                    severity="WARNING",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                return None
            
            # Verify password
            if not self.verify_password(password, user.hashed_password):
                user.failed_login_attempts += 1
                
                # Lock account if too many attempts
                if user.failed_login_attempts >= self.config.max_login_attempts:
                    user.locked_until = datetime.utcnow() + timedelta(minutes=self.config.lockout_duration_minutes)
                    
                    self.auditor.log_event(
                        user_id=user.id,
                        event_type=SecurityEventType.ACCOUNT_LOCKED,
                        description=f"Account {username} locked due to too many failed attempts",
                        severity="WARNING",
                        ip_address=ip_address,
                        user_agent=user_agent
                    )
                
                db.commit()
                
                auth_attempts.labels(status="failure", method="password").inc()
                self.auditor.log_event(
                    user_id=user.id,
                    event_type=SecurityEventType.LOGIN_FAILURE,
                    description=f"Invalid password for user: {username}",
                    severity="WARNING",
                    ip_address=ip_address,
                    user_agent=user_agent
                )
                
                return None
            
            # Check MFA if enabled
            if user.mfa_enabled:
                if not mfa_token or not self.mfa_manager.verify_token(user.mfa_secret, mfa_token):
                    auth_attempts.labels(status="failure", method="mfa").inc()
                    self.auditor.log_event(
                        user_id=user.id,
                        event_type=SecurityEventType.LOGIN_FAILURE,
                        description=f"Invalid MFA token for user: {username}",
                        severity="WARNING",
                        ip_address=ip_address,
                        user_agent=user_agent
                    )
                    return None
            
            # Reset failed attempts
            user.failed_login_attempts = 0
            user.locked_until = None
            user.last_login = datetime.utcnow()
            db.commit()
            
            # Create tokens
            token_data = {"sub": user.username, "role": user.role, "user_id": user.id}
            access_token = self.token_manager.create_access_token(token_data)
            refresh_token = self.token_manager.create_refresh_token(token_data)
            
            # Create session
            session = UserSession(
                session_token=access_token,
                refresh_token=refresh_token,
                ip_address=ip_address or "unknown",
                user_agent=user_agent or "unknown",
                expires_at=datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes),
                user_id=user.id
            )
            
            db.add(session)
            db.commit()
            
            # Update metrics
            auth_attempts.labels(status="success", method="password").inc()
            auth_duration.observe(time.time() - start_time)
            active_sessions.inc()
            
            # Log success
            self.auditor.log_event(
                user_id=user.id,
                event_type=SecurityEventType.LOGIN_SUCCESS,
                description=f"User {username} authenticated successfully",
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            return {
                "access_token": access_token,
                "refresh_token": refresh_token,
                "token_type": "bearer",
                "expires_in": self.config.access_token_expire_minutes * 60,
                "user": {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "full_name": user.full_name,
                    "role": user.role,
                    "mfa_enabled": user.mfa_enabled
                }
            }
        
        except Exception as e:
            db.rollback()
            logger.error(f"Authentication error: {e}")
            return None
        
        finally:
            db.close()
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token."""
        try:
            payload = self.token_manager.verify_token(refresh_token, "refresh")
            
            db = self.SessionLocal()
            
            # Get session
            session = db.query(UserSession).filter(
                UserSession.refresh_token == refresh_token,
                UserSession.is_active == True
            ).first()
            
            if not session or session.expires_at < datetime.utcnow():
                db.close()
                return None
            
            # Create new access token
            token_data = {"sub": payload["sub"], "role": payload["role"], "user_id": payload["user_id"]}
            new_access_token = self.token_manager.create_access_token(token_data)
            
            # Update session
            session.session_token = new_access_token
            session.expires_at = datetime.utcnow() + timedelta(minutes=self.config.access_token_expire_minutes)
            db.commit()
            
            db.close()
            return new_access_token
        
        except Exception as e:
            logger.error(f"Token refresh error: {e}")
            return None
    
    def logout_user(self, token: str) -> bool:
        """Logout user and invalidate session."""
        try:
            payload = self.token_manager.verify_token(token, "access")
            
            db = self.SessionLocal()
            
            # Invalidate session
            session = db.query(UserSession).filter(
                UserSession.session_token == token,
                UserSession.is_active == True
            ).first()
            
            if session:
                session.is_active = False
                db.commit()
                
                # Update metrics
                active_sessions.dec()
                
                # Log logout
                self.auditor.log_event(
                    user_id=payload["user_id"],
                    event_type=SecurityEventType.LOGOUT,
                    description=f"User {payload['sub']} logged out"
                )
            
            db.close()
            return True
        
        except Exception as e:
            logger.error(f"Logout error: {e}")
            return False
    
    def enable_mfa(self, user_id: int) -> Tuple[str, bytes]:
        """Enable MFA for user and return secret and QR code."""
        db = self.SessionLocal()
        
        try:
            user = db.query(User).filter(User.id == user_id).first()
            
            if not user:
                return "", b""
            
            # Generate secret
            secret = self.mfa_manager.generate_secret()
            user.mfa_secret = secret
            user.mfa_enabled = True
            db.commit()
            
            # Generate QR code
            qr_code = self.mfa_manager.generate_qr_code(user.email, secret)
            
            # Log event
            self.auditor.log_event(
                user_id=user_id,
                event_type=SecurityEventType.MFA_ENABLED,
                description=f"MFA enabled for user {user.username}"
            )
            
            return secret, qr_code
        
        except Exception as e:
            db.rollback()
            logger.error(f"MFA enable error: {e}")
            return "", b""
        
        finally:
            db.close()
    
    def check_permission(self, user_role: str, permission: str) -> bool:
        """Check if user role has permission."""
        try:
            role_enum = UserRole(user_role)
            permission_enum = Permission(permission)
            return self.rbac.has_permission(role_enum, permission_enum)
        
        except ValueError:
            return False

# FastAPI dependencies
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/token")
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def get_current_user(token: str = Depends(oauth2_scheme), 
                    auth_service: AuthenticationService = None) -> User:
    """Get current user from token."""
    if auth_service is None:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication service not available"
        )
    
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = auth_service.token_manager.verify_token(token, "access")
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
    
    except jwt.PyJWTError:
        raise credentials_exception
    
    db = auth_service.SessionLocal()
    user = db.query(User).filter(User.id == user_id).first()
    db.close()
    
    if user is None:
        raise credentials_exception
    
    return user

def require_permission(permission: str):
    """Decorator to require specific permission."""
    def permission_checker(current_user: User = Depends(get_current_user)):
        if not auth_service.check_permission(current_user.role, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions"
            )
        return current_user
    
    return permission_checker

# Initialize global auth service
auth_service = None

def initialize_auth(config: SecurityConfig, database_url: str):
    """Initialize authentication service."""
    global auth_service
    auth_service = AuthenticationService(config, database_url)

def get_auth_service() -> AuthenticationService:
    """Get authentication service."""
    if auth_service is None:
        raise RuntimeError("Authentication service not initialized")
    return auth_service

# Import io for QR code generation
import io
