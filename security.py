"""
MAOF Security Module
Handles authentication, authorization, encryption, and security features
"""

import os
import secrets
import hashlib
import jwt
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
from cryptography.fernet import Fernet
from passlib.context import CryptContext

from maof_framework_enhanced import logger, Config


# ============================================================================
# Password Hashing
# ============================================================================

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


def hash_password(password: str) -> str:
    """Hash a password"""
    return pwd_context.hash(password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a hash"""
    return pwd_context.verify(plain_password, hashed_password)


# ============================================================================
# API Key Management
# ============================================================================

class APIKeyManager:
    """Manages API keys for authentication"""

    def __init__(self):
        self.keys: Dict[str, Dict] = {}

    def generate_key(self, user_id: str, description: str = "") -> str:
        """Generate a new API key"""
        api_key = f"maof_{secrets.token_urlsafe(32)}"
        self.keys[api_key] = {
            'user_id': user_id,
            'description': description,
            'created_at': datetime.utcnow(),
            'last_used': None,
            'active': True
        }
        logger.info("api_key_generated", user_id=user_id)
        return api_key

    def validate_key(self, api_key: str) -> Optional[Dict]:
        """Validate an API key"""
        if api_key in self.keys:
            key_info = self.keys[api_key]
            if key_info['active']:
                key_info['last_used'] = datetime.utcnow()
                return key_info
        return None

    def revoke_key(self, api_key: str) -> bool:
        """Revoke an API key"""
        if api_key in self.keys:
            self.keys[api_key]['active'] = False
            logger.info("api_key_revoked", api_key=api_key[:16] + "...")
            return True
        return False


# ============================================================================
# JWT Token Management
# ============================================================================

def create_jwt_token(
    user_id: str,
    secret_key: Optional[str] = None,
    expires_hours: int = 24,
    additional_claims: Optional[Dict] = None
) -> str:
    """Create a JWT token"""
    secret = secret_key or os.getenv('JWT_SECRET_KEY', 'default-secret-key')

    payload = {
        'user_id': user_id,
        'iat': datetime.utcnow(),
        'exp': datetime.utcnow() + timedelta(hours=expires_hours),
    }

    if additional_claims:
        payload.update(additional_claims)

    token = jwt.encode(payload, secret, algorithm=Config.API_KEY_ENCRYPTION_ALGO)
    return token


def verify_jwt_token(
    token: str,
    secret_key: Optional[str] = None
) -> Optional[Dict]:
    """Verify and decode a JWT token"""
    secret = secret_key or os.getenv('JWT_SECRET_KEY', 'default-secret-key')

    try:
        payload = jwt.decode(
            token,
            secret,
            algorithms=[Config.API_KEY_ENCRYPTION_ALGO]
        )
        return payload
    except jwt.ExpiredSignatureError:
        logger.warning("jwt_token_expired")
        return None
    except jwt.InvalidTokenError as e:
        logger.warning("jwt_token_invalid", error=str(e))
        return None


# ============================================================================
# Encryption
# ============================================================================

class EncryptionManager:
    """Manages encryption and decryption of sensitive data"""

    def __init__(self, key: Optional[bytes] = None):
        """
        Initialize with encryption key
        Args:
            key: Fernet key (32 url-safe base64-encoded bytes)
        """
        if key is None:
            # Generate new key
            self.key = Fernet.generate_key()
        else:
            self.key = key

        self.cipher = Fernet(self.key)

    def encrypt(self, data: str) -> str:
        """Encrypt data"""
        encrypted = self.cipher.encrypt(data.encode())
        return encrypted.decode()

    def decrypt(self, encrypted_data: str) -> str:
        """Decrypt data"""
        decrypted = self.cipher.decrypt(encrypted_data.encode())
        return decrypted.decode()

    def get_key(self) -> bytes:
        """Get the encryption key"""
        return self.key


# ============================================================================
# Content Filtering & PII Detection
# ============================================================================

class ContentFilter:
    """Filter and sanitize content"""

    # Common PII patterns (simplified)
    PII_PATTERNS = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        'phone': r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
        'ssn': r'\b\d{3}-\d{2}-\d{4}\b',
        'credit_card': r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b',
    }

    # Toxic content keywords (simplified - use ML model in production)
    TOXIC_KEYWORDS = [
        'hack', 'exploit', 'malware', 'virus', 'phishing',
        'ddos', 'attack', 'breach', 'ransomware'
    ]

    def detect_pii(self, text: str) -> Dict[str, List[str]]:
        """Detect PII in text"""
        import re
        detected = {}

        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            if matches:
                detected[pii_type] = matches

        return detected

    def redact_pii(self, text: str) -> str:
        """Redact PII from text"""
        import re
        redacted = text

        for pii_type, pattern in self.PII_PATTERNS.items():
            redacted = re.sub(pattern, f'[{pii_type.upper()}_REDACTED]', redacted)

        return redacted

    def check_toxicity(self, text: str) -> Dict[str, Any]:
        """Simple toxicity check"""
        text_lower = text.lower()
        found_keywords = []

        for keyword in self.TOXIC_KEYWORDS:
            if keyword in text_lower:
                found_keywords.append(keyword)

        is_toxic = len(found_keywords) > 0
        toxicity_score = min(len(found_keywords) * 0.2, 1.0)

        return {
            'is_toxic': is_toxic,
            'toxicity_score': toxicity_score,
            'keywords': found_keywords
        }


# ============================================================================
# Role-Based Access Control (RBAC)
# ============================================================================

class Role:
    """User role"""
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"
    SERVICE = "service"


class Permission:
    """Permissions"""
    SUBMIT_TASK = "submit_task"
    VIEW_TASK = "view_task"
    MANAGE_AGENTS = "manage_agents"
    VIEW_METRICS = "view_metrics"
    ADMIN = "admin"


class RBAC:
    """Role-Based Access Control"""

    # Role to permissions mapping
    ROLE_PERMISSIONS = {
        Role.ADMIN: [
            Permission.SUBMIT_TASK,
            Permission.VIEW_TASK,
            Permission.MANAGE_AGENTS,
            Permission.VIEW_METRICS,
            Permission.ADMIN
        ],
        Role.USER: [
            Permission.SUBMIT_TASK,
            Permission.VIEW_TASK,
            Permission.VIEW_METRICS
        ],
        Role.READONLY: [
            Permission.VIEW_TASK,
            Permission.VIEW_METRICS
        ],
        Role.SERVICE: [
            Permission.SUBMIT_TASK,
            Permission.VIEW_TASK
        ]
    }

    def __init__(self):
        self.user_roles: Dict[str, str] = {}

    def assign_role(self, user_id: str, role: str):
        """Assign a role to a user"""
        if role not in [Role.ADMIN, Role.USER, Role.READONLY, Role.SERVICE]:
            raise ValueError(f"Invalid role: {role}")
        self.user_roles[user_id] = role
        logger.info("role_assigned", user_id=user_id, role=role)

    def check_permission(self, user_id: str, permission: str) -> bool:
        """Check if user has permission"""
        role = self.user_roles.get(user_id)
        if not role:
            return False

        permissions = self.ROLE_PERMISSIONS.get(role, [])
        return permission in permissions

    def get_user_permissions(self, user_id: str) -> List[str]:
        """Get all permissions for a user"""
        role = self.user_roles.get(user_id)
        if not role:
            return []
        return self.ROLE_PERMISSIONS.get(role, [])


# ============================================================================
# Security Manager (Facade)
# ============================================================================

class SecurityManager:
    """Unified security management"""

    def __init__(self, encryption_key: Optional[bytes] = None):
        self.api_keys = APIKeyManager()
        self.encryption = EncryptionManager(encryption_key)
        self.content_filter = ContentFilter()
        self.rbac = RBAC()
        self.jwt_secret = os.getenv('JWT_SECRET_KEY', secrets.token_urlsafe(32))

    def create_user_token(self, user_id: str, role: str) -> str:
        """Create a user token with role"""
        return create_jwt_token(
            user_id,
            self.jwt_secret,
            expires_hours=Config.JWT_EXPIRY_HOURS,
            additional_claims={'role': role}
        )

    def verify_user_token(self, token: str) -> Optional[Dict]:
        """Verify a user token"""
        return verify_jwt_token(token, self.jwt_secret)

    def generate_api_key(self, user_id: str, description: str = "") -> str:
        """Generate API key for a user"""
        return self.api_keys.generate_key(user_id, description)

    def validate_api_key(self, api_key: str) -> Optional[Dict]:
        """Validate API key"""
        return self.api_keys.validate_key(api_key)

    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data"""
        return self.encryption.encrypt(data)

    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data"""
        return self.encryption.decrypt(encrypted_data)

    def sanitize_content(self, text: str, redact_pii: bool = True) -> str:
        """Sanitize content"""
        if redact_pii:
            return self.content_filter.redact_pii(text)
        return text

    def check_content_safety(self, text: str) -> Dict[str, Any]:
        """Check content for PII and toxicity"""
        pii_detected = self.content_filter.detect_pii(text)
        toxicity = self.content_filter.check_toxicity(text)

        return {
            'pii_detected': pii_detected,
            'has_pii': len(pii_detected) > 0,
            'toxicity': toxicity,
            'is_safe': len(pii_detected) == 0 and not toxicity['is_toxic']
        }

    def authorize(self, user_id: str, permission: str) -> bool:
        """Check if user is authorized for permission"""
        return self.rbac.check_permission(user_id, permission)
