"""
MAOF REST API
FastAPI-based REST API for the Multi-Agent Orchestration Framework
"""

import os
import secrets
from typing import Optional, List, Dict, Any
from datetime import datetime, timedelta
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, Security, status, BackgroundTasks
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import MAOF components
try:
    from maof_framework_enhanced import (
        Orchestrator, Task, TaskPriority, AgentConfig, AgentType,
        RoutingStrategy, logger, Config
    )
    from agents import create_agent
    from database import Database, hash_content
    from security import SecurityManager, verify_jwt_token, create_jwt_token
except ImportError as e:
    logger.error("import_error", error=str(e))
    raise


# ============================================================================
# Pydantic Models for API
# ============================================================================

class TaskRequest(BaseModel):
    """Task submission request"""
    task_type: str = Field(..., description="Type of task (text, code, analysis, etc.)")
    content: Any = Field(..., description="Task content")
    priority: str = Field(default="medium", description="Priority: low, medium, high, critical")
    context: Optional[Dict] = Field(default=None, description="Additional context")
    constraints: Optional[Dict] = Field(default=None, description="Task constraints")
    metadata: Optional[Dict] = Field(default=None, description="Task metadata")
    session_id: Optional[str] = Field(default=None, description="Session ID for context")
    max_cost: Optional[float] = Field(default=None, description="Maximum cost allowed")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens allowed")
    timeout: Optional[int] = Field(default=None, description="Timeout in seconds")


class TaskResponse(BaseModel):
    """Task submission response"""
    task_id: str
    status: str
    message: str


class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    agent_id: Optional[str] = None
    success: Optional[bool] = None
    result: Optional[Any] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None
    tokens_used: Optional[Dict] = None
    cost: Optional[float] = None
    created_at: Optional[str] = None
    completed_at: Optional[str] = None


class AgentRegistrationRequest(BaseModel):
    """Agent registration request"""
    agent_id: str
    name: str
    provider: str
    agent_type: str
    endpoint: str
    api_key: str
    capabilities: List[str]
    model_name: Optional[str] = None
    max_tokens: int = 4096
    rate_limit: int = 10
    timeout: int = 30
    cost_per_1k_input: float = 0.01
    cost_per_1k_output: float = 0.03


class MetricsResponse(BaseModel):
    """System metrics response"""
    timestamp: str
    total_requests: int
    total_errors: int
    total_cost: float
    total_tokens: int
    agents: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    version: str
    timestamp: str
    database_connected: bool
    cache_connected: bool
    agents_count: int
    agents_healthy: int


# ============================================================================
# Global State
# ============================================================================

class AppState:
    """Global application state"""
    def __init__(self):
        self.orchestrator: Optional[Orchestrator] = None
        self.database: Optional[Database] = None
        self.security: Optional[SecurityManager] = None
        self.task_results: Dict[str, Any] = {}  # In-memory task results
        self.initialized = False


app_state = AppState()


# ============================================================================
# Lifespan Management
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    logger.info("api_starting")

    # Initialize database
    app_state.database = Database(
        postgres_url=Config.POSTGRES_URL,
        redis_url=Config.REDIS_URL,
        use_vector_store=False  # Enable if needed
    )
    await app_state.database.connect()

    # Initialize security
    app_state.security = SecurityManager()

    # Initialize orchestrator (import after other components)
    from orchestrator import create_orchestrator
    app_state.orchestrator = await create_orchestrator(app_state.database)

    app_state.initialized = True
    logger.info("api_started", agents=len(app_state.orchestrator.agents))

    yield

    # Shutdown
    logger.info("api_shutting_down")
    if app_state.database:
        await app_state.database.disconnect()
    logger.info("api_shutdown_complete")


# ============================================================================
# FastAPI Application
# ============================================================================

app = FastAPI(
    title="MAOF API",
    description="Multi-Agent Orchestration Framework REST API",
    version="2.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ============================================================================
# Dependencies
# ============================================================================

async def verify_api_key(
    api_key: Optional[str] = Security(api_key_header)
) -> str:
    """Verify API key"""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required"
        )

    # Verify against configured keys
    valid_keys = os.getenv('MAOF_API_KEYS', '').split(',')
    if api_key not in valid_keys:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )

    return api_key


async def get_orchestrator() -> 'Orchestrator':
    """Get orchestrator instance"""
    if not app_state.initialized or not app_state.orchestrator:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Orchestrator not initialized"
        )
    return app_state.orchestrator


async def get_database() -> Database:
    """Get database instance"""
    if not app_state.database:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Database not initialized"
        )
    return app_state.database


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "name": "MAOF API",
        "version": "2.0.0",
        "status": "running" if app_state.initialized else "initializing"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agents_count = len(app_state.orchestrator.agents) if app_state.orchestrator else 0
    agents_healthy = 0

    if app_state.orchestrator:
        for agent in app_state.orchestrator.agents.values():
            if await agent.health_check():
                agents_healthy += 1

    return HealthResponse(
        status="healthy" if app_state.initialized else "initializing",
        version="2.0.0",
        timestamp=datetime.utcnow().isoformat(),
        database_connected=app_state.database.is_connected() if app_state.database else False,
        cache_connected=app_state.database.cache._connected if app_state.database else False,
        agents_count=agents_count,
        agents_healthy=agents_healthy
    )


@app.post("/tasks", response_model=TaskResponse)
async def submit_task(
    request: TaskRequest,
    background_tasks: BackgroundTasks,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    database: Database = Depends(get_database),
    api_key: str = Depends(verify_api_key)
):
    """Submit a new task"""
    try:
        # Generate task ID
        task_id = f"task-{secrets.token_hex(8)}"

        # Parse priority
        priority_map = {
            'low': TaskPriority.LOW,
            'medium': TaskPriority.MEDIUM,
            'high': TaskPriority.HIGH,
            'critical': TaskPriority.CRITICAL
        }
        priority = priority_map.get(request.priority.lower(), TaskPriority.MEDIUM)

        # Create task
        task = Task(
            task_id=task_id,
            task_type=request.task_type,
            content=request.content,
            priority=priority,
            context=request.context or {},
            constraints=request.constraints or {},
            metadata=request.metadata or {},
            max_cost=request.max_cost,
            max_tokens=request.max_tokens,
            timeout=request.timeout
        )

        # Check cache for duplicate content
        content_hash = hash_content(request.content)
        cached_result = await database.cache.get_cached_result(content_hash)

        if cached_result:
            logger.info("cache_hit", task_id=task_id, content_hash=content_hash)
            app_state.task_results[task_id] = cached_result
            return TaskResponse(
                task_id=task_id,
                status="completed",
                message="Result from cache"
            )

        # Save task to database
        await database.db.save_task({
            'task_id': task_id,
            'task_type': request.task_type,
            'content_hash': content_hash,
            'priority': priority.value,
            'status': 'pending',
            'session_id': request.session_id,
            'metadata': request.metadata or {}
        })

        # Process task in background
        background_tasks.add_task(
            process_task_background,
            task,
            request.session_id,
            content_hash,
            orchestrator,
            database
        )

        logger.info("task_submitted", task_id=task_id, type=request.task_type)

        return TaskResponse(
            task_id=task_id,
            status="processing",
            message="Task submitted successfully"
        )

    except Exception as e:
        logger.error("task_submission_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit task: {str(e)}"
        )


@app.get("/tasks/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(
    task_id: str,
    database: Database = Depends(get_database),
    api_key: str = Depends(verify_api_key)
):
    """Get task status and result"""
    try:
        # Check in-memory results first
        if task_id in app_state.task_results:
            result = app_state.task_results[task_id]
            return TaskStatusResponse(
                task_id=task_id,
                status="completed" if result.get('success') else "failed",
                agent_id=result.get('agent_id'),
                success=result.get('success'),
                result=result.get('result'),
                error=result.get('error'),
                processing_time=result.get('processing_time'),
                tokens_used=result.get('tokens_used'),
                cost=result.get('cost'),
                completed_at=result.get('timestamp')
            )

        # TODO: Query database for task status
        # This would involve querying TaskModel and TaskResultModel

        return TaskStatusResponse(
            task_id=task_id,
            status="processing",
            message="Task is being processed"
        )

    except Exception as e:
        logger.error("get_task_status_error", task_id=task_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get task status: {str(e)}"
        )


@app.get("/metrics", response_model=MetricsResponse)
async def get_metrics(
    orchestrator: Orchestrator = Depends(get_orchestrator),
    api_key: str = Depends(verify_api_key)
):
    """Get system metrics"""
    try:
        metrics = orchestrator.get_metrics()

        return MetricsResponse(
            timestamp=datetime.utcnow().isoformat(),
            total_requests=metrics['total_requests'],
            total_errors=metrics['total_errors'],
            total_cost=metrics['total_cost'],
            total_tokens=metrics['total_tokens'],
            agents=metrics['agents']
        )

    except Exception as e:
        logger.error("get_metrics_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )


@app.get("/agents")
async def list_agents(
    orchestrator: Orchestrator = Depends(get_orchestrator),
    api_key: str = Depends(verify_api_key)
):
    """List all registered agents"""
    try:
        agents = []
        for agent_id, agent in orchestrator.agents.items():
            agents.append({
                'agent_id': agent_id,
                'name': agent.config.name,
                'provider': agent.config.provider,
                'type': agent.config.agent_type.value,
                'capabilities': agent.config.capabilities,
                'metrics': agent.get_metrics()
            })

        return {'agents': agents, 'count': len(agents)}

    except Exception as e:
        logger.error("list_agents_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list agents: {str(e)}"
        )


@app.post("/agents")
async def register_agent(
    request: AgentRegistrationRequest,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    api_key: str = Depends(verify_api_key)
):
    """Register a new agent"""
    try:
        # Create agent config
        config = AgentConfig(
            agent_id=request.agent_id,
            name=request.name,
            provider=request.provider,
            agent_type=AgentType(request.agent_type),
            endpoint=request.endpoint,
            api_key=request.api_key,
            capabilities=request.capabilities,
            model_name=request.model_name or "",
            max_tokens=request.max_tokens,
            rate_limit=request.rate_limit,
            timeout=request.timeout,
            cost_per_1k_input=request.cost_per_1k_input,
            cost_per_1k_output=request.cost_per_1k_output
        )

        # Create and register agent
        agent = create_agent(config)
        orchestrator.register_agent(agent)

        logger.info("agent_registered", agent_id=request.agent_id)

        return {
            'status': 'success',
            'message': f'Agent {request.agent_id} registered successfully'
        }

    except Exception as e:
        logger.error("register_agent_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to register agent: {str(e)}"
        )


@app.delete("/agents/{agent_id}")
async def unregister_agent(
    agent_id: str,
    orchestrator: Orchestrator = Depends(get_orchestrator),
    api_key: str = Depends(verify_api_key)
):
    """Unregister an agent"""
    try:
        orchestrator.unregister_agent(agent_id)
        logger.info("agent_unregistered", agent_id=agent_id)

        return {
            'status': 'success',
            'message': f'Agent {agent_id} unregistered successfully'
        }

    except Exception as e:
        logger.error("unregister_agent_error", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to unregister agent: {str(e)}"
        )


# ============================================================================
# Background Task Processing
# ============================================================================

async def process_task_background(
    task: Task,
    session_id: Optional[str],
    content_hash: str,
    orchestrator: Orchestrator,
    database: Database
):
    """Process task in background"""
    try:
        # Update task status
        await database.db.update_task_status(task.task_id, 'processing')

        # Process task
        result = await orchestrator.process_task(task, session_id)

        # Store result
        app_state.task_results[task.task_id] = result.to_dict()

        # Cache result if successful
        if result.success:
            await database.cache.cache_result(
                content_hash,
                result.to_dict(),
                ttl=3600  # 1 hour
            )

        # Save to database
        await database.db.save_result({
            'task_id': task.task_id,
            'agent_id': result.agent_id,
            'success': result.success,
            'result_hash': content_hash,
            'error': result.error,
            'error_code': result.error_code.value if result.error_code else None,
            'processing_time': result.processing_time,
            'tokens_input': result.tokens_used.input_tokens if result.tokens_used else 0,
            'tokens_output': result.tokens_used.output_tokens if result.tokens_used else 0,
            'tokens_total': result.tokens_used.total_tokens if result.tokens_used else 0,
            'cost': result.cost,
            'retry_count': result.retry_count,
            'metadata': result.metadata
        })

        # Update task status
        status = 'completed' if result.success else 'failed'
        await database.db.update_task_status(
            task.task_id,
            status,
            result.agent_id
        )

        logger.info(
            "task_completed",
            task_id=task.task_id,
            success=result.success,
            agent=result.agent_id
        )

    except Exception as e:
        logger.error("background_task_error", task_id=task.task_id, error=str(e))
        app_state.task_results[task.task_id] = {
            'success': False,
            'error': str(e),
            'task_id': task.task_id
        }
        await database.db.update_task_status(task.task_id, 'failed')


# ============================================================================
# Run Server
# ============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
