"""
MAOF Database Integration
Supports: PostgreSQL (metadata), Redis (caching/rate limiting), Vector DBs (embeddings)
"""

import json
import pickle
import hashlib
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import asyncio

# Database client imports (with graceful fallbacks)
try:
    import redis.asyncio as redis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False

try:
    import asyncpg
    ASYNCPG_AVAILABLE = True
except ImportError:
    ASYNCPG_AVAILABLE = False

try:
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker, declarative_base
    from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Boolean, Text
    SQLALCHEMY_AVAILABLE = True
    Base = declarative_base()
except ImportError:
    SQLALCHEMY_AVAILABLE = False
    Base = None

from maof_framework_enhanced import logger, Config


# ============================================================================
# SQLAlchemy Models
# ============================================================================

if SQLALCHEMY_AVAILABLE:
    class TaskModel(Base):
        """Task metadata model"""
        __tablename__ = 'tasks'

        task_id = Column(String(255), primary_key=True)
        task_type = Column(String(100), index=True)
        content_hash = Column(String(64))  # Hash of content for caching
        priority = Column(Integer)
        status = Column(String(50), index=True)  # pending, processing, completed, failed
        agent_id = Column(String(255), index=True, nullable=True)
        session_id = Column(String(255), index=True, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        started_at = Column(DateTime, nullable=True)
        completed_at = Column(DateTime, nullable=True)
        metadata = Column(JSON, nullable=True)

    class TaskResultModel(Base):
        """Task result model"""
        __tablename__ = 'task_results'

        id = Column(Integer, primary_key=True, autoincrement=True)
        task_id = Column(String(255), index=True)
        agent_id = Column(String(255), index=True)
        success = Column(Boolean)
        result_hash = Column(String(64))  # Hash of result for deduplication
        error = Column(Text, nullable=True)
        error_code = Column(String(50), nullable=True)
        processing_time = Column(Float)
        tokens_input = Column(Integer)
        tokens_output = Column(Integer)
        tokens_total = Column(Integer)
        cost = Column(Float)
        retry_count = Column(Integer, default=0)
        created_at = Column(DateTime, default=datetime.utcnow, index=True)
        metadata = Column(JSON, nullable=True)

    class AgentMetricsModel(Base):
        """Agent metrics model"""
        __tablename__ = 'agent_metrics'

        id = Column(Integer, primary_key=True, autoincrement=True)
        agent_id = Column(String(255), index=True)
        timestamp = Column(DateTime, default=datetime.utcnow, index=True)
        requests_count = Column(Integer, default=0)
        success_count = Column(Integer, default=0)
        error_count = Column(Integer, default=0)
        total_tokens = Column(Integer, default=0)
        total_cost = Column(Float, default=0.0)
        avg_response_time = Column(Float, default=0.0)
        p95_response_time = Column(Float, default=0.0)
        circuit_breaker_state = Column(String(20))
        metadata = Column(JSON, nullable=True)

    class SessionModel(Base):
        """Session model for context management"""
        __tablename__ = 'sessions'

        session_id = Column(String(255), primary_key=True)
        user_id = Column(String(255), index=True, nullable=True)
        created_at = Column(DateTime, default=datetime.utcnow)
        updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
        expires_at = Column(DateTime, nullable=True)
        context = Column(JSON, nullable=True)
        metadata = Column(JSON, nullable=True)


# ============================================================================
# Redis Cache & Rate Limiter
# ============================================================================

class RedisCache:
    """Redis-based caching and rate limiting"""

    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or Config.REDIS_URL
        self.client: Optional[redis.Redis] = None
        self._connected = False

    async def connect(self):
        """Connect to Redis"""
        if not REDIS_AVAILABLE:
            logger.warning("redis_not_available", message="Install with: pip install redis")
            return

        try:
            self.client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True
            )
            await self.client.ping()
            self._connected = True
            logger.info("redis_connected", url=self.redis_url)
        except Exception as e:
            logger.error("redis_connection_failed", error=str(e))
            self._connected = False

    async def disconnect(self):
        """Disconnect from Redis"""
        if self.client:
            await self.client.close()
            self._connected = False

    async def get(self, key: str) -> Optional[str]:
        """Get value from cache"""
        if not self._connected:
            return None
        try:
            return await self.client.get(key)
        except Exception as e:
            logger.error("redis_get_error", key=key, error=str(e))
            return None

    async def set(
        self,
        key: str,
        value: str,
        ttl: Optional[int] = None
    ) -> bool:
        """Set value in cache with optional TTL (seconds)"""
        if not self._connected:
            return False
        try:
            if ttl:
                await self.client.setex(key, ttl, value)
            else:
                await self.client.set(key, value)
            return True
        except Exception as e:
            logger.error("redis_set_error", key=key, error=str(e))
            return False

    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._connected:
            return False
        try:
            await self.client.delete(key)
            return True
        except Exception as e:
            logger.error("redis_delete_error", key=key, error=str(e))
            return False

    async def check_rate_limit(
        self,
        key: str,
        limit: int,
        window: int
    ) -> bool:
        """
        Check if rate limit is exceeded
        Args:
            key: Rate limit key (e.g., user_id, ip_address)
            limit: Maximum requests allowed
            window: Time window in seconds
        Returns:
            True if under limit, False if exceeded
        """
        if not self._connected:
            return True  # Allow if Redis unavailable

        try:
            current = await self.client.get(key)
            if current is None:
                # First request in window
                await self.client.setex(key, window, "1")
                return True

            count = int(current)
            if count >= limit:
                return False

            # Increment counter
            await self.client.incr(key)
            return True

        except Exception as e:
            logger.error("redis_rate_limit_error", key=key, error=str(e))
            return True  # Allow on error to avoid blocking

    async def cache_result(
        self,
        content_hash: str,
        result: Any,
        ttl: int = 3600
    ) -> bool:
        """Cache a result with content hash as key"""
        if not self._connected:
            return False

        try:
            cache_key = f"result:{content_hash}"
            serialized = json.dumps(result)
            return await self.set(cache_key, serialized, ttl)
        except Exception as e:
            logger.error("cache_result_error", error=str(e))
            return False

    async def get_cached_result(self, content_hash: str) -> Optional[Any]:
        """Get cached result by content hash"""
        if not self._connected:
            return None

        try:
            cache_key = f"result:{content_hash}"
            cached = await self.get(cache_key)
            if cached:
                return json.loads(cached)
            return None
        except Exception as e:
            logger.error("get_cached_result_error", error=str(e))
            return None


# ============================================================================
# PostgreSQL Database Manager
# ============================================================================

class DatabaseManager:
    """PostgreSQL database manager with connection pooling"""

    def __init__(self, database_url: str = None):
        self.database_url = database_url or Config.POSTGRES_URL
        self.engine = None
        self.session_maker = None
        self._connected = False

    async def connect(self):
        """Connect to database and create tables"""
        if not SQLALCHEMY_AVAILABLE:
            logger.warning(
                "sqlalchemy_not_available",
                message="Install with: pip install asyncpg sqlalchemy"
            )
            return

        try:
            # Convert postgresql:// to postgresql+asyncpg://
            db_url = self.database_url.replace('postgresql://', 'postgresql+asyncpg://')

            self.engine = create_async_engine(
                db_url,
                echo=False,
                pool_size=10,
                max_overflow=20,
                pool_pre_ping=True,
            )

            self.session_maker = sessionmaker(
                self.engine,
                class_=AsyncSession,
                expire_on_commit=False
            )

            # Create tables
            async with self.engine.begin() as conn:
                await conn.run_sync(Base.metadata.create_all)

            self._connected = True
            logger.info("database_connected", url=db_url)

        except Exception as e:
            logger.error("database_connection_failed", error=str(e))
            self._connected = False

    async def disconnect(self):
        """Disconnect from database"""
        if self.engine:
            await self.engine.dispose()
            self._connected = False

    async def save_task(self, task_data: Dict) -> bool:
        """Save task to database"""
        if not self._connected:
            return False

        try:
            async with self.session_maker() as session:
                task_model = TaskModel(**task_data)
                session.add(task_model)
                await session.commit()
                return True
        except Exception as e:
            logger.error("save_task_error", error=str(e))
            return False

    async def save_result(self, result_data: Dict) -> bool:
        """Save task result to database"""
        if not self._connected:
            return False

        try:
            async with self.session_maker() as session:
                result_model = TaskResultModel(**result_data)
                session.add(result_model)
                await session.commit()
                return True
        except Exception as e:
            logger.error("save_result_error", error=str(e))
            return False

    async def save_agent_metrics(self, metrics_data: Dict) -> bool:
        """Save agent metrics to database"""
        if not self._connected:
            return False

        try:
            async with self.session_maker() as session:
                metrics_model = AgentMetricsModel(**metrics_data)
                session.add(metrics_model)
                await session.commit()
                return True
        except Exception as e:
            logger.error("save_metrics_error", error=str(e))
            return False

    async def update_task_status(
        self,
        task_id: str,
        status: str,
        agent_id: Optional[str] = None
    ) -> bool:
        """Update task status"""
        if not self._connected:
            return False

        try:
            async with self.session_maker() as session:
                from sqlalchemy import update
                stmt = update(TaskModel).where(
                    TaskModel.task_id == task_id
                ).values(status=status)

                if agent_id:
                    stmt = stmt.values(agent_id=agent_id)

                if status == 'processing':
                    stmt = stmt.values(started_at=datetime.utcnow())
                elif status in ['completed', 'failed']:
                    stmt = stmt.values(completed_at=datetime.utcnow())

                await session.execute(stmt)
                await session.commit()
                return True
        except Exception as e:
            logger.error("update_task_status_error", error=str(e))
            return False


# ============================================================================
# Vector Database Interface (Abstract)
# ============================================================================

class VectorStore(ABC):
    """Abstract interface for vector databases"""

    @abstractmethod
    async def connect(self):
        """Connect to vector store"""
        pass

    @abstractmethod
    async def upsert(self, vectors: List[Dict]) -> bool:
        """Insert or update vectors"""
        pass

    @abstractmethod
    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Search for similar vectors"""
        pass

    @abstractmethod
    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors by ID"""
        pass


# ============================================================================
# In-Memory Vector Store (Fallback)
# ============================================================================

class InMemoryVectorStore(VectorStore):
    """Simple in-memory vector store for development/testing"""

    def __init__(self):
        self.vectors: Dict[str, Dict] = {}

    async def connect(self):
        """No-op for in-memory store"""
        logger.info("using_in_memory_vector_store")

    async def upsert(self, vectors: List[Dict]) -> bool:
        """Store vectors in memory"""
        try:
            for vec in vectors:
                self.vectors[vec['id']] = vec
            return True
        except Exception as e:
            logger.error("vector_upsert_error", error=str(e))
            return False

    async def search(
        self,
        query_vector: List[float],
        top_k: int = 10,
        filter_dict: Optional[Dict] = None
    ) -> List[Dict]:
        """Simple cosine similarity search"""
        try:
            import numpy as np

            results = []
            query_vec = np.array(query_vector)

            for vec_id, vec_data in self.vectors.items():
                # Apply filters
                if filter_dict:
                    skip = False
                    for key, value in filter_dict.items():
                        if vec_data.get('metadata', {}).get(key) != value:
                            skip = True
                            break
                    if skip:
                        continue

                # Calculate cosine similarity
                stored_vec = np.array(vec_data['values'])
                similarity = np.dot(query_vec, stored_vec) / (
                    np.linalg.norm(query_vec) * np.linalg.norm(stored_vec)
                )

                results.append({
                    'id': vec_id,
                    'score': float(similarity),
                    'metadata': vec_data.get('metadata', {})
                })

            # Sort by similarity and return top_k
            results.sort(key=lambda x: x['score'], reverse=True)
            return results[:top_k]

        except Exception as e:
            logger.error("vector_search_error", error=str(e))
            return []

    async def delete(self, ids: List[str]) -> bool:
        """Delete vectors from memory"""
        try:
            for vec_id in ids:
                self.vectors.pop(vec_id, None)
            return True
        except Exception as e:
            logger.error("vector_delete_error", error=str(e))
            return False


# ============================================================================
# Database Facade
# ============================================================================

class Database:
    """Unified database interface"""

    def __init__(
        self,
        postgres_url: Optional[str] = None,
        redis_url: Optional[str] = None,
        use_vector_store: bool = False
    ):
        self.db = DatabaseManager(postgres_url)
        self.cache = RedisCache(redis_url)
        self.vector_store = InMemoryVectorStore() if use_vector_store else None

    async def connect(self):
        """Connect to all databases"""
        await self.db.connect()
        await self.cache.connect()
        if self.vector_store:
            await self.vector_store.connect()

    async def disconnect(self):
        """Disconnect from all databases"""
        await self.db.disconnect()
        await self.cache.disconnect()

    def is_connected(self) -> bool:
        """Check if databases are connected"""
        return self.db._connected or self.cache._connected


def hash_content(content: Any) -> str:
    """Generate hash of content for caching/deduplication"""
    content_str = json.dumps(content, sort_keys=True) if isinstance(content, dict) else str(content)
    return hashlib.sha256(content_str.encode()).hexdigest()
