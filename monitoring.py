"""
MAOF Monitoring & Observability
Includes: Prometheus metrics, OpenTelemetry tracing, structured logging
"""

import time
import os
from typing import Dict, Any, Optional
from functools import wraps
from datetime import datetime

# Prometheus metrics (with graceful fallback)
try:
    from prometheus_client import Counter, Histogram, Gauge, Info, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# OpenTelemetry (with graceful fallback)
try:
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    OPENTELEMETRY_AVAILABLE = True
except ImportError:
    OPENTELEMETRY_AVAILABLE = False

from maof_framework_enhanced import logger


# ============================================================================
# Prometheus Metrics
# ============================================================================

class PrometheusMetrics:
    """Prometheus metrics collector"""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("prometheus_not_available", message="Install with: pip install prometheus-client")
            self.enabled = False
            return

        self.enabled = True

        # Request metrics
        self.task_counter = Counter(
            'maof_tasks_total',
            'Total number of tasks processed',
            ['agent_id', 'task_type', 'status']
        )

        self.task_duration = Histogram(
            'maof_task_duration_seconds',
            'Task processing duration in seconds',
            ['agent_id', 'task_type']
        )

        # Token metrics
        self.token_counter = Counter(
            'maof_tokens_total',
            'Total tokens consumed',
            ['agent_id', 'token_type']
        )

        # Cost metrics
        self.cost_counter = Counter(
            'maof_cost_total_usd',
            'Total cost in USD',
            ['agent_id']
        )

        # Error metrics
        self.error_counter = Counter(
            'maof_errors_total',
            'Total number of errors',
            ['agent_id', 'error_type']
        )

        # Agent health
        self.agent_health = Gauge(
            'maof_agent_health',
            'Agent health status (1=healthy, 0=unhealthy)',
            ['agent_id']
        )

        # Circuit breaker state
        self.circuit_breaker_state = Gauge(
            'maof_circuit_breaker_state',
            'Circuit breaker state (0=closed, 1=half-open, 2=open)',
            ['agent_id']
        )

        # Active requests
        self.active_requests = Gauge(
            'maof_active_requests',
            'Number of active requests',
            ['agent_id']
        )

        # System info
        self.system_info = Info(
            'maof_system',
            'MAOF system information'
        )
        self.system_info.info({
            'version': '2.0.0',
            'python_version': os.sys.version.split()[0]
        })

        logger.info("prometheus_metrics_initialized")

    def record_task(self, agent_id: str, task_type: str, status: str, duration: float):
        """Record a task execution"""
        if not self.enabled:
            return

        self.task_counter.labels(
            agent_id=agent_id,
            task_type=task_type,
            status=status
        ).inc()

        self.task_duration.labels(
            agent_id=agent_id,
            task_type=task_type
        ).observe(duration)

    def record_tokens(self, agent_id: str, input_tokens: int, output_tokens: int):
        """Record token usage"""
        if not self.enabled:
            return

        self.token_counter.labels(
            agent_id=agent_id,
            token_type='input'
        ).inc(input_tokens)

        self.token_counter.labels(
            agent_id=agent_id,
            token_type='output'
        ).inc(output_tokens)

    def record_cost(self, agent_id: str, cost: float):
        """Record cost"""
        if not self.enabled:
            return

        self.cost_counter.labels(agent_id=agent_id).inc(cost)

    def record_error(self, agent_id: str, error_type: str):
        """Record an error"""
        if not self.enabled:
            return

        self.error_counter.labels(
            agent_id=agent_id,
            error_type=error_type
        ).inc()

    def set_agent_health(self, agent_id: str, healthy: bool):
        """Set agent health status"""
        if not self.enabled:
            return

        self.agent_health.labels(agent_id=agent_id).set(1 if healthy else 0)

    def set_circuit_breaker_state(self, agent_id: str, state: str):
        """Set circuit breaker state"""
        if not self.enabled:
            return

        state_map = {'closed': 0, 'half-open': 1, 'open': 2}
        self.circuit_breaker_state.labels(agent_id=agent_id).set(
            state_map.get(state, 0)
        )

    def increment_active_requests(self, agent_id: str):
        """Increment active requests"""
        if not self.enabled:
            return
        self.active_requests.labels(agent_id=agent_id).inc()

    def decrement_active_requests(self, agent_id: str):
        """Decrement active requests"""
        if not self.enabled:
            return
        self.active_requests.labels(agent_id=agent_id).dec()

    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not self.enabled:
            return ""
        return generate_latest(REGISTRY).decode('utf-8')


# ============================================================================
# OpenTelemetry Tracing
# ============================================================================

class TracingManager:
    """OpenTelemetry distributed tracing manager"""

    def __init__(self, service_name: str = "maof", jaeger_endpoint: Optional[str] = None):
        if not OPENTELEMETRY_AVAILABLE:
            logger.warning("opentelemetry_not_available", message="Install with: pip install opentelemetry-*")
            self.enabled = False
            return

        self.enabled = True
        self.service_name = service_name

        # Create resource
        resource = Resource.create({"service.name": service_name})

        # Create tracer provider
        provider = TracerProvider(resource=resource)

        # Add exporters
        if jaeger_endpoint:
            # Jaeger exporter
            jaeger_exporter = JaegerExporter(
                agent_host_name=jaeger_endpoint.split(':')[0],
                agent_port=int(jaeger_endpoint.split(':')[1]) if ':' in jaeger_endpoint else 6831,
            )
            provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
        else:
            # Console exporter for development
            console_exporter = ConsoleSpanExporter()
            provider.add_span_processor(BatchSpanProcessor(console_exporter))

        # Set as global tracer provider
        trace.set_tracer_provider(provider)

        # Get tracer
        self.tracer = trace.get_tracer(__name__)

        logger.info("tracing_initialized", service=service_name)

    def start_span(self, name: str, attributes: Optional[Dict] = None):
        """Start a new span"""
        if not self.enabled:
            return None

        span = self.tracer.start_span(name)
        if attributes:
            for key, value in attributes.items():
                span.set_attribute(key, str(value))
        return span

    def trace_task(self, func):
        """Decorator to trace a function as a span"""
        if not self.enabled:
            return func

        @wraps(func)
        async def wrapper(*args, **kwargs):
            with self.tracer.start_as_current_span(func.__name__):
                return await func(*args, **kwargs)

        return wrapper


# ============================================================================
# Performance Monitor
# ============================================================================

class PerformanceMonitor:
    """Monitor and track performance metrics"""

    def __init__(self):
        self.metrics_history = []
        self.max_history = 10000

    def record_metric(
        self,
        metric_type: str,
        value: float,
        tags: Optional[Dict[str, str]] = None
    ):
        """Record a performance metric"""
        metric = {
            'timestamp': datetime.utcnow(),
            'type': metric_type,
            'value': value,
            'tags': tags or {}
        }

        self.metrics_history.append(metric)

        # Trim history
        if len(self.metrics_history) > self.max_history:
            self.metrics_history = self.metrics_history[-self.max_history:]

    def get_statistics(
        self,
        metric_type: str,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric type"""
        import statistics

        # Filter metrics
        filtered = [
            m for m in self.metrics_history
            if m['type'] == metric_type
        ]

        # Apply time window if specified
        if time_window_seconds:
            from datetime import timedelta
            cutoff = datetime.utcnow() - timedelta(seconds=time_window_seconds)
            filtered = [m for m in filtered if m['timestamp'] >= cutoff]

        if not filtered:
            return {}

        values = [m['value'] for m in filtered]

        return {
            'count': len(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'stdev': statistics.stdev(values) if len(values) > 1 else 0,
            'min': min(values),
            'max': max(values),
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
        }


# ============================================================================
# Health Check System
# ============================================================================

class HealthChecker:
    """Comprehensive health checking system"""

    def __init__(self):
        self.checks = {}
        self.last_check_results = {}

    def register_check(self, name: str, check_func):
        """Register a health check"""
        self.checks[name] = check_func
        logger.info("health_check_registered", name=name)

    async def run_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {
            'timestamp': datetime.utcnow().isoformat(),
            'overall_status': 'healthy',
            'checks': {}
        }

        for name, check_func in self.checks.items():
            try:
                start = time.time()
                result = await check_func()
                duration = time.time() - start

                results['checks'][name] = {
                    'status': 'healthy' if result else 'unhealthy',
                    'duration_ms': duration * 1000,
                    'timestamp': datetime.utcnow().isoformat()
                }

                if not result:
                    results['overall_status'] = 'degraded'

            except Exception as e:
                results['checks'][name] = {
                    'status': 'error',
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                }
                results['overall_status'] = 'unhealthy'

        self.last_check_results = results
        return results

    def get_last_results(self) -> Dict[str, Any]:
        """Get last health check results"""
        return self.last_check_results


# ============================================================================
# Monitoring Facade
# ============================================================================

class MonitoringSystem:
    """Unified monitoring system"""

    def __init__(
        self,
        enable_prometheus: bool = True,
        enable_tracing: bool = False,
        jaeger_endpoint: Optional[str] = None
    ):
        self.prometheus = PrometheusMetrics() if enable_prometheus else None
        self.tracing = TracingManager(jaeger_endpoint=jaeger_endpoint) if enable_tracing else None
        self.performance = PerformanceMonitor()
        self.health = HealthChecker()

        logger.info(
            "monitoring_system_initialized",
            prometheus=self.prometheus.enabled if self.prometheus else False,
            tracing=self.tracing.enabled if self.tracing else False
        )

    def record_task_execution(
        self,
        agent_id: str,
        task_type: str,
        duration: float,
        success: bool,
        tokens_input: int = 0,
        tokens_output: int = 0,
        cost: float = 0.0,
        error_type: Optional[str] = None
    ):
        """Record complete task execution"""
        status = 'success' if success else 'failure'

        # Prometheus
        if self.prometheus:
            self.prometheus.record_task(agent_id, task_type, status, duration)
            if tokens_input or tokens_output:
                self.prometheus.record_tokens(agent_id, tokens_input, tokens_output)
            if cost > 0:
                self.prometheus.record_cost(agent_id, cost)
            if error_type:
                self.prometheus.record_error(agent_id, error_type)

        # Performance monitoring
        self.performance.record_metric(
            f'task_duration_{task_type}',
            duration,
            {'agent_id': agent_id, 'status': status}
        )

        if cost > 0:
            self.performance.record_metric(
                'task_cost',
                cost,
                {'agent_id': agent_id, 'task_type': task_type}
            )

    def update_agent_status(
        self,
        agent_id: str,
        healthy: bool,
        circuit_breaker_state: str
    ):
        """Update agent status metrics"""
        if self.prometheus:
            self.prometheus.set_agent_health(agent_id, healthy)
            self.prometheus.set_circuit_breaker_state(agent_id, circuit_breaker_state)

    def get_prometheus_metrics(self) -> str:
        """Get Prometheus metrics"""
        if self.prometheus:
            return self.prometheus.get_metrics()
        return ""

    async def get_health_status(self) -> Dict[str, Any]:
        """Get health status"""
        return await self.health.run_checks()

    def get_performance_stats(
        self,
        metric_type: str,
        time_window_seconds: Optional[int] = None
    ) -> Dict[str, float]:
        """Get performance statistics"""
        return self.performance.get_statistics(metric_type, time_window_seconds)


# ============================================================================
# Global Monitoring Instance
# ============================================================================

# Initialize global monitoring system
monitoring = MonitoringSystem(
    enable_prometheus=os.getenv('METRICS_ENABLED', 'true').lower() == 'true',
    enable_tracing=os.getenv('TRACING_ENABLED', 'false').lower() == 'true',
    jaeger_endpoint=os.getenv('JAEGER_ENDPOINT')
)


# ============================================================================
# Monitoring Decorators
# ============================================================================

def monitor_task(agent_id: str = "unknown"):
    """Decorator to monitor task execution"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start = time.time()
            success = False
            error_type = None

            try:
                result = await func(*args, **kwargs)
                success = True
                return result
            except Exception as e:
                error_type = type(e).__name__
                raise
            finally:
                duration = time.time() - start
                monitoring.record_task_execution(
                    agent_id=agent_id,
                    task_type=func.__name__,
                    duration=duration,
                    success=success,
                    error_type=error_type
                )

        return wrapper
    return decorator
