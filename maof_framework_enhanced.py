"""
Multi-Agent Orchestration Framework (MAOF) - Enhanced Version
Version: 2.0 - Production Ready with Real API Integration
Implements Phase 1 & Phase 2 Enhancements
"""

import asyncio
import json
import logging
import time
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Union, Callable
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import secrets
from functools import wraps

# Third-party imports
try:
    import tiktoken
except ImportError:
    tiktoken = None

# Configure structured logging
import structlog
logger = structlog.get_logger()


# ============================================================================
# Configuration & Constants
# ============================================================================

class Config:
    """Global configuration with safety limits"""
    # Token limits per provider (conservative to avoid overruns)
    TOKEN_LIMITS = {
        'openai-gpt4': {'max_input': 8000, 'max_output': 4000, 'safety_margin': 0.9},
        'openai-gpt35': {'max_input': 3500, 'max_output': 1000, 'safety_margin': 0.9},
        'claude-opus': {'max_input': 180000, 'max_output': 4000, 'safety_margin': 0.9},
        'claude-sonnet': {'max_input': 180000, 'max_output': 4000, 'safety_margin': 0.9},
        'gemini-pro': {'max_input': 30000, 'max_output': 2000, 'safety_margin': 0.9},
    }

    # Retry configuration
    MAX_RETRIES = 3
    BASE_RETRY_DELAY = 1.0
    MAX_RETRY_DELAY = 32.0
    RETRY_JITTER = 0.1

    # Rate limiting
    DEFAULT_RATE_LIMIT = 10  # requests per minute
    RATE_LIMIT_WINDOW = 60  # seconds

    # Security
    API_KEY_ENCRYPTION_ALGO = 'HS256'
    JWT_EXPIRY_HOURS = 24

    # Database
    REDIS_URL = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
    POSTGRES_URL = os.getenv('DATABASE_URL', 'postgresql://localhost/maof')

    # Monitoring
    METRICS_ENABLED = os.getenv('METRICS_ENABLED', 'true').lower() == 'true'
    TRACING_ENABLED = os.getenv('TRACING_ENABLED', 'true').lower() == 'true'


# ============================================================================
# Enums and Data Classes
# ============================================================================

class AgentType(Enum):
    """Types of AI agents supported by the framework"""
    LLM = "llm"
    VISION = "vision"
    AUDIO = "audio"
    CODE = "code"
    SPECIALIZED = "specialized"
    MULTIMODAL = "multimodal"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


class RoutingStrategy(Enum):
    """Agent routing strategies"""
    CAPABILITY_BASED = "capability"
    COST_OPTIMIZED = "cost"
    PERFORMANCE_PRIORITY = "performance"
    ROUND_ROBIN = "round_robin"
    INTELLIGENT = "intelligent"
    LOAD_BALANCED = "load_balanced"


class ErrorCode(Enum):
    """Standardized error codes"""
    RATE_LIMIT = "rate_limit_exceeded"
    TOKEN_LIMIT = "token_limit_exceeded"
    API_ERROR = "api_error"
    TIMEOUT = "timeout"
    INVALID_INPUT = "invalid_input"
    AUTHENTICATION = "authentication_failed"
    CIRCUIT_OPEN = "circuit_breaker_open"
    NO_AGENTS = "no_available_agents"


@dataclass
class TokenUsage:
    """Token usage tracking"""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated: bool = False

    def __post_init__(self):
        if self.total_tokens == 0:
            self.total_tokens = self.input_tokens + self.output_tokens


@dataclass
class AgentConfig:
    """Enhanced configuration for an AI agent"""
    agent_id: str
    name: str
    provider: str
    agent_type: AgentType
    endpoint: str
    api_key: str
    capabilities: List[str]
    model_name: str = ""
    max_tokens: int = 4096
    rate_limit: int = 10
    timeout: int = 30
    cost_per_1k_input: float = 0.01
    cost_per_1k_output: float = 0.03
    retry_attempts: int = 3
    supports_streaming: bool = False
    supports_functions: bool = False
    context_window: int = 4096

    # Token budget management
    token_budget: Optional[int] = None  # Max tokens per time window
    budget_window: int = 3600  # Budget window in seconds

    # Security
    encryption_key: Optional[str] = None

    def __post_init__(self):
        """Initialize encryption key if not provided"""
        if not self.encryption_key:
            self.encryption_key = secrets.token_hex(32)
        if not self.model_name:
            self.model_name = self.provider.lower()


@dataclass
class Task:
    """Enhanced task representation"""
    task_id: str
    task_type: str
    content: Any
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict = field(default_factory=dict)
    constraints: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    timeout: Optional[int] = None
    max_cost: Optional[float] = None
    max_tokens: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'content': self.content,
            'priority': self.priority.value,
            'context': self.context,
            'constraints': self.constraints,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class TaskResult:
    """Enhanced result from task execution"""
    task_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    error_code: Optional[ErrorCode] = None
    processing_time: float = 0
    tokens_used: Optional[TokenUsage] = None
    cost: float = 0
    metadata: Dict = field(default_factory=dict)
    retry_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'task_id': self.task_id,
            'agent_id': self.agent_id,
            'success': self.success,
            'result': self.result,
            'error': self.error,
            'error_code': self.error_code.value if self.error_code else None,
            'processing_time': self.processing_time,
            'tokens_used': {
                'input': self.tokens_used.input_tokens,
                'output': self.tokens_used.output_tokens,
                'total': self.tokens_used.total_tokens,
            } if self.tokens_used else None,
            'cost': self.cost,
            'retry_count': self.retry_count,
            'timestamp': self.timestamp.isoformat(),
        }


# ============================================================================
# Utility Functions
# ============================================================================

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """Count tokens in text using tiktoken"""
    if tiktoken is None:
        # Fallback: rough estimation (1 token ~= 4 chars)
        return len(text) // 4

    try:
        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback to rough estimation
        return len(text) // 4


def exponential_backoff(attempt: int, base_delay: float = 1.0, max_delay: float = 32.0) -> float:
    """Calculate exponential backoff with jitter"""
    delay = min(base_delay * (2 ** attempt), max_delay)
    jitter = delay * Config.RETRY_JITTER * (2 * secrets.SystemRandom().random() - 1)
    return delay + jitter


def retry_with_backoff(max_retries: int = 3, backoff_func: Callable = exponential_backoff):
    """Decorator for retry logic with exponential backoff"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(max_retries):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries - 1:
                        delay = backoff_func(attempt)
                        logger.warning(
                            "retry_attempt",
                            function=func.__name__,
                            attempt=attempt + 1,
                            max_retries=max_retries,
                            delay=delay,
                            error=str(e)
                        )
                        await asyncio.sleep(delay)
                    else:
                        logger.error(
                            "retry_exhausted",
                            function=func.__name__,
                            attempts=max_retries,
                            error=str(e)
                        )
            raise last_exception
        return wrapper
    return decorator


# ============================================================================
# Enhanced Circuit Breaker
# ============================================================================

class CircuitBreaker:
    """Enhanced circuit breaker with configurable thresholds and states"""

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 2,
        reset_timeout: int = 60,
        half_open_max_calls: int = 3
    ):
        self.failure_count = 0
        self.success_count = 0
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.reset_timeout = reset_timeout
        self.half_open_max_calls = half_open_max_calls
        self.half_open_calls = 0
        self.last_failure_time = 0
        self.last_state_change = time.time()
        self.state = 'closed'  # closed, open, half-open
        self._lock = asyncio.Lock()

    async def record_success(self):
        """Record successful call"""
        async with self._lock:
            if self.state == 'half-open':
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self._transition_to_closed()
            elif self.state == 'closed':
                # Reset failure count on success
                self.failure_count = max(0, self.failure_count - 1)

    async def record_failure(self):
        """Record failed call"""
        async with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.state == 'half-open':
                self._transition_to_open()
            elif self.state == 'closed' and self.failure_count >= self.failure_threshold:
                self._transition_to_open()

    async def can_execute(self) -> bool:
        """Check if execution is allowed"""
        async with self._lock:
            if self.state == 'closed':
                return True

            if self.state == 'open':
                if time.time() - self.last_failure_time > self.reset_timeout:
                    self._transition_to_half_open()
                    return True
                return False

            # half-open state
            if self.half_open_calls < self.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False

    def _transition_to_closed(self):
        """Transition to closed state"""
        logger.info("circuit_breaker_closed")
        self.state = 'closed'
        self.failure_count = 0
        self.success_count = 0
        self.half_open_calls = 0
        self.last_state_change = time.time()

    def _transition_to_open(self):
        """Transition to open state"""
        logger.warning("circuit_breaker_opened", failures=self.failure_count)
        self.state = 'open'
        self.success_count = 0
        self.half_open_calls = 0
        self.last_state_change = time.time()

    def _transition_to_half_open(self):
        """Transition to half-open state"""
        logger.info("circuit_breaker_half_open")
        self.state = 'half-open'
        self.success_count = 0
        self.failure_count = 0
        self.half_open_calls = 0
        self.last_state_change = time.time()

    def get_state(self) -> Dict:
        """Get current state information"""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time,
            'time_in_state': time.time() - self.last_state_change,
        }


# ============================================================================
# Rate Limiter
# ============================================================================

class RateLimiter:
    """Token bucket rate limiter"""

    def __init__(self, rate: int, window: int = 60):
        """
        Args:
            rate: Number of requests allowed per window
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.tokens = rate
        self.last_update = time.time()
        self._lock = asyncio.Lock()

    async def acquire(self) -> bool:
        """Try to acquire a token"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_update

            # Refill tokens based on elapsed time
            self.tokens = min(
                self.rate,
                self.tokens + (elapsed / self.window) * self.rate
            )
            self.last_update = now

            if self.tokens >= 1:
                self.tokens -= 1
                return True

            # Calculate wait time
            wait_time = (1 - self.tokens) * self.window / self.rate
            await asyncio.sleep(wait_time)
            self.tokens = 0
            self.last_update = time.time()
            return True


# ============================================================================
# Token Budget Manager
# ============================================================================

class TokenBudgetManager:
    """Manages token budgets for agents"""

    def __init__(self):
        self.budgets: Dict[str, Dict] = {}
        self._lock = asyncio.Lock()

    async def check_budget(
        self,
        agent_id: str,
        estimated_tokens: int,
        budget: int,
        window: int
    ) -> bool:
        """Check if request is within token budget"""
        async with self._lock:
            now = time.time()

            if agent_id not in self.budgets:
                self.budgets[agent_id] = {
                    'tokens_used': 0,
                    'window_start': now,
                    'budget': budget,
                    'window': window
                }

            budget_info = self.budgets[agent_id]

            # Reset if window has passed
            if now - budget_info['window_start'] > window:
                budget_info['tokens_used'] = 0
                budget_info['window_start'] = now

            # Check if adding tokens would exceed budget
            if budget_info['tokens_used'] + estimated_tokens > budget:
                return False

            return True

    async def consume_tokens(self, agent_id: str, tokens: int):
        """Record token consumption"""
        async with self._lock:
            if agent_id in self.budgets:
                self.budgets[agent_id]['tokens_used'] += tokens


# ============================================================================
# Base Agent Class (Enhanced)
# ============================================================================

class BaseAgent(ABC):
    """Enhanced base class for all agents with production features"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0
        self.error_count = 0
        self.success_count = 0
        self.last_request_time = 0
        self.circuit_breaker = CircuitBreaker()
        self.rate_limiter = RateLimiter(config.rate_limit, Config.RATE_LIMIT_WINDOW)
        self.token_budget_manager = TokenBudgetManager()

        # Metrics
        self.response_times: List[float] = []
        self.max_response_times = 1000  # Keep last 1000

    @abstractmethod
    async def _call_api(self, task: Task) -> Dict[str, Any]:
        """Call the actual API - must be implemented by subclasses"""
        pass

    async def process(self, task: Task) -> TaskResult:
        """Process a task with full error handling and retry logic"""
        start_time = time.time()
        retry_count = 0

        # Check circuit breaker
        if not await self.circuit_breaker.can_execute():
            logger.warning("circuit_breaker_blocked", agent_id=self.config.agent_id)
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=False,
                result=None,
                error="Circuit breaker is open",
                error_code=ErrorCode.CIRCUIT_OPEN,
                processing_time=time.time() - start_time
            )

        # Estimate token usage
        estimated_tokens = self._estimate_tokens(task)

        # Check token budget if configured
        if self.config.token_budget:
            if not await self.token_budget_manager.check_budget(
                self.config.agent_id,
                estimated_tokens,
                self.config.token_budget,
                self.config.budget_window
            ):
                logger.warning(
                    "token_budget_exceeded",
                    agent_id=self.config.agent_id,
                    estimated=estimated_tokens
                )
                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.config.agent_id,
                    success=False,
                    result=None,
                    error="Token budget exceeded",
                    error_code=ErrorCode.TOKEN_LIMIT,
                    processing_time=time.time() - start_time
                )

        # Check task constraints
        if task.max_tokens and estimated_tokens > task.max_tokens:
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=False,
                result=None,
                error=f"Estimated tokens ({estimated_tokens}) exceed task limit ({task.max_tokens})",
                error_code=ErrorCode.TOKEN_LIMIT,
                processing_time=time.time() - start_time
            )

        # Retry loop
        last_exception = None
        for attempt in range(self.config.retry_attempts):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()

                # Call API
                result_data = await self._call_api(task)

                # Extract token usage
                tokens_used = result_data.get('tokens_used', TokenUsage(
                    input_tokens=estimated_tokens // 2,
                    output_tokens=estimated_tokens // 2,
                    estimated=True
                ))

                # Update token budget
                if self.config.token_budget:
                    await self.token_budget_manager.consume_tokens(
                        self.config.agent_id,
                        tokens_used.total_tokens
                    )

                # Calculate cost
                cost = self.calculate_cost(
                    tokens_used.input_tokens,
                    tokens_used.output_tokens
                )

                # Update metrics
                processing_time = time.time() - start_time
                self.request_count += 1
                self.success_count += 1
                self.total_tokens += tokens_used.total_tokens
                self.total_cost += cost
                self.response_times.append(processing_time)
                if len(self.response_times) > self.max_response_times:
                    self.response_times = self.response_times[-self.max_response_times:]

                # Record success with circuit breaker
                await self.circuit_breaker.record_success()

                logger.info(
                    "task_processed",
                    agent_id=self.config.agent_id,
                    task_id=task.task_id,
                    tokens=tokens_used.total_tokens,
                    cost=cost,
                    time=processing_time
                )

                return TaskResult(
                    task_id=task.task_id,
                    agent_id=self.config.agent_id,
                    success=True,
                    result=result_data.get('result'),
                    processing_time=processing_time,
                    tokens_used=tokens_used,
                    cost=cost,
                    retry_count=retry_count,
                    metadata=result_data.get('metadata', {})
                )

            except Exception as e:
                last_exception = e
                retry_count += 1
                self.error_count += 1

                # Record failure with circuit breaker
                await self.circuit_breaker.record_failure()

                logger.error(
                    "task_failed",
                    agent_id=self.config.agent_id,
                    task_id=task.task_id,
                    attempt=attempt + 1,
                    error=str(e)
                )

                # Retry with backoff if not last attempt
                if attempt < self.config.retry_attempts - 1:
                    delay = exponential_backoff(attempt)
                    await asyncio.sleep(delay)

        # All retries exhausted
        return TaskResult(
            task_id=task.task_id,
            agent_id=self.config.agent_id,
            success=False,
            result=None,
            error=str(last_exception),
            error_code=ErrorCode.API_ERROR,
            processing_time=time.time() - start_time,
            retry_count=retry_count
        )

    def _estimate_tokens(self, task: Task) -> int:
        """Estimate token usage for a task"""
        content_str = str(task.content)
        context_str = json.dumps(task.context) if task.context else ""
        total_text = content_str + context_str

        # Use model-specific token counting
        tokens = count_tokens(total_text, self.config.model_name)

        # Apply safety margin
        model_key = f"{self.config.provider.lower()}-{self.config.model_name.lower()}"
        if model_key in Config.TOKEN_LIMITS:
            safety_margin = Config.TOKEN_LIMITS[model_key]['safety_margin']
            tokens = int(tokens / safety_margin)

        return tokens

    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost

    async def health_check(self) -> bool:
        """Check if agent is healthy and available"""
        try:
            circuit_state = await self.circuit_breaker.can_execute()
            return circuit_state and self.error_count < (self.request_count * 0.5)
        except Exception:
            return False

    def get_metrics(self) -> Dict:
        """Get agent performance metrics"""
        avg_response_time = (
            sum(self.response_times) / len(self.response_times)
            if self.response_times else 0
        )

        p95_response_time = 0
        if self.response_times:
            sorted_times = sorted(self.response_times)
            p95_idx = int(len(sorted_times) * 0.95)
            p95_response_time = sorted_times[p95_idx] if p95_idx < len(sorted_times) else 0

        return {
            'agent_id': self.config.agent_id,
            'requests': self.request_count,
            'successes': self.success_count,
            'errors': self.error_count,
            'success_rate': (
                self.success_count / self.request_count
                if self.request_count > 0 else 0
            ),
            'total_tokens': self.total_tokens,
            'total_cost': self.total_cost,
            'avg_response_time': avg_response_time,
            'p95_response_time': p95_response_time,
            'circuit_breaker': self.circuit_breaker.get_state(),
        }


# File continues in next message due to length...
