"""
MAOF Configuration Management
Centralized configuration loading and validation
"""

import os
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from enum import Enum

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class Environment(str, Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Log levels"""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class DatabaseConfig:
    """Database configuration"""
    postgres_url: str = field(default_factory=lambda: os.getenv('DATABASE_URL', 'postgresql://localhost/maof'))
    redis_url: str = field(default_factory=lambda: os.getenv('REDIS_URL', 'redis://localhost:6379/0'))
    pool_size: int = field(default_factory=lambda: int(os.getenv('DB_POOL_SIZE', '10')))
    max_overflow: int = field(default_factory=lambda: int(os.getenv('DB_MAX_OVERFLOW', '20')))


@dataclass
class APIConfig:
    """API server configuration"""
    host: str = field(default_factory=lambda: os.getenv('API_HOST', '0.0.0.0'))
    port: int = field(default_factory=lambda: int(os.getenv('API_PORT', '8000')))
    cors_enabled: bool = field(default_factory=lambda: os.getenv('CORS_ENABLED', 'true').lower() == 'true')
    cors_origins: List[str] = field(default_factory=lambda: os.getenv('CORS_ORIGINS', '*').split(','))
    api_keys: List[str] = field(default_factory=lambda: os.getenv('MAOF_API_KEYS', '').split(','))


@dataclass
class SecurityConfig:
    """Security configuration"""
    jwt_secret: str = field(default_factory=lambda: os.getenv('JWT_SECRET_KEY', 'change-me-in-production'))
    jwt_expiry_hours: int = field(default_factory=lambda: int(os.getenv('JWT_EXPIRY_HOURS', '24')))
    encryption_key: Optional[str] = field(default_factory=lambda: os.getenv('ENCRYPTION_KEY'))
    enable_pii_detection: bool = field(default_factory=lambda: os.getenv('ENABLE_PII_DETECTION', 'true').lower() == 'true')
    enable_content_filtering: bool = field(default_factory=lambda: os.getenv('ENABLE_CONTENT_FILTERING', 'true').lower() == 'true')


@dataclass
class MonitoringConfig:
    """Monitoring configuration"""
    metrics_enabled: bool = field(default_factory=lambda: os.getenv('METRICS_ENABLED', 'true').lower() == 'true')
    tracing_enabled: bool = field(default_factory=lambda: os.getenv('TRACING_ENABLED', 'false').lower() == 'true')
    jaeger_endpoint: Optional[str] = field(default_factory=lambda: os.getenv('JAEGER_ENDPOINT'))


@dataclass
class AgentConfig:
    """Agent configuration"""
    routing_strategy: str = field(default_factory=lambda: os.getenv('ROUTING_STRATEGY', 'intelligent'))
    default_max_tokens: int = field(default_factory=lambda: int(os.getenv('DEFAULT_MAX_TOKENS', '4096')))
    default_rate_limit: int = field(default_factory=lambda: int(os.getenv('DEFAULT_RATE_LIMIT', '10')))
    token_budget: int = field(default_factory=lambda: int(os.getenv('TOKEN_BUDGET', '0')))
    budget_window: int = field(default_factory=lambda: int(os.getenv('BUDGET_WINDOW', '3600')))
    use_mock_agents: bool = field(default_factory=lambda: os.getenv('USE_MOCK_AGENTS', 'false').lower() == 'true')


@dataclass
class ProviderKeys:
    """AI Provider API keys"""
    openai: Optional[str] = field(default_factory=lambda: os.getenv('OPENAI_API_KEY'))
    anthropic: Optional[str] = field(default_factory=lambda: os.getenv('ANTHROPIC_API_KEY'))
    google: Optional[str] = field(default_factory=lambda: os.getenv('GOOGLE_API_KEY'))


@dataclass
class FeatureFlags:
    """Feature flags"""
    enable_caching: bool = field(default_factory=lambda: os.getenv('ENABLE_CACHING', 'true').lower() == 'true')
    cache_ttl: int = field(default_factory=lambda: int(os.getenv('CACHE_TTL', '3600')))
    enable_retries: bool = field(default_factory=lambda: os.getenv('ENABLE_RETRIES', 'true').lower() == 'true')
    max_retry_attempts: int = field(default_factory=lambda: int(os.getenv('MAX_RETRY_ATTEMPTS', '3')))


@dataclass
class MAOFConfig:
    """Main MAOF configuration"""
    environment: Environment = field(default_factory=lambda: Environment(os.getenv('ENVIRONMENT', 'development')))
    log_level: LogLevel = field(default_factory=lambda: LogLevel(os.getenv('LOG_LEVEL', 'INFO')))
    debug: bool = field(default_factory=lambda: os.getenv('DEBUG', 'false').lower() == 'true')

    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    api: APIConfig = field(default_factory=APIConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    provider_keys: ProviderKeys = field(default_factory=ProviderKeys)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Runtime settings
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.getenv('MAX_CONCURRENT_TASKS', '10')))
    default_task_timeout: int = field(default_factory=lambda: int(os.getenv('DEFAULT_TASK_TIMEOUT', '60')))

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        warnings = []

        # Check for production readiness
        if self.environment == Environment.PRODUCTION:
            if self.security.jwt_secret == 'change-me-in-production':
                warnings.append("JWT_SECRET_KEY should be changed in production")

            if not self.security.encryption_key:
                warnings.append("ENCRYPTION_KEY should be set in production")

            if self.debug:
                warnings.append("DEBUG mode should be disabled in production")

            if not self.provider_keys.openai and not self.provider_keys.anthropic and not self.provider_keys.google:
                warnings.append("No AI provider API keys configured")

        # Check database URLs
        if 'localhost' in self.database.postgres_url and self.environment == Environment.PRODUCTION:
            warnings.append("PostgreSQL URL points to localhost in production")

        if 'localhost' in self.database.redis_url and self.environment == Environment.PRODUCTION:
            warnings.append("Redis URL points to localhost in production")

        return warnings

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary (excluding sensitive data)"""
        return {
            'environment': self.environment.value,
            'log_level': self.log_level.value,
            'debug': self.debug,
            'api': {
                'host': self.api.host,
                'port': self.api.port,
                'cors_enabled': self.api.cors_enabled,
            },
            'monitoring': {
                'metrics_enabled': self.monitoring.metrics_enabled,
                'tracing_enabled': self.monitoring.tracing_enabled,
            },
            'agent': {
                'routing_strategy': self.agent.routing_strategy,
                'default_max_tokens': self.agent.default_max_tokens,
                'default_rate_limit': self.agent.default_rate_limit,
            },
            'features': {
                'caching': self.features.enable_caching,
                'retries': self.features.enable_retries,
            },
            'providers_configured': {
                'openai': bool(self.provider_keys.openai),
                'anthropic': bool(self.provider_keys.anthropic),
                'google': bool(self.provider_keys.google),
            }
        }


# Global configuration instance
config = MAOFConfig()

# Validate configuration and log warnings
warnings = config.validate()
if warnings:
    import structlog
    logger = structlog.get_logger()
    for warning in warnings:
        logger.warning("config_validation", message=warning)
