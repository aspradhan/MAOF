"""
Multi-Agent Orchestration Framework (MAOF)
Main Implementation Module
Version: 1.0
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from collections import defaultdict
from datetime import datetime
import hashlib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('MAOF')


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


@dataclass
class AgentConfig:
    """Configuration for an AI agent"""
    agent_id: str
    name: str
    provider: str
    agent_type: AgentType
    endpoint: str
    api_key: str
    capabilities: List[str]
    max_tokens: int = 4096
    rate_limit: int = 100
    timeout: int = 30
    cost_per_1k_input: float = 0.01
    cost_per_1k_output: float = 0.03
    retry_attempts: int = 3
    
    def __post_init__(self):
        """Encrypt API key after initialization"""
        self.api_key = self._encrypt_key(self.api_key)
    
    def _encrypt_key(self, key: str) -> str:
        """Simple encryption for API keys (use proper encryption in production)"""
        return hashlib.sha256(key.encode()).hexdigest()[:16] + "..." 


@dataclass
class Task:
    """Represents a task to be processed by agents"""
    task_id: str
    task_type: str
    content: Any
    priority: TaskPriority = TaskPriority.MEDIUM
    context: Dict = None
    constraints: Dict = None
    metadata: Dict = None
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.context is None:
            self.context = {}
        if self.constraints is None:
            self.constraints = {}
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TaskResult:
    """Result from task execution"""
    task_id: str
    agent_id: str
    success: bool
    result: Any
    error: Optional[str] = None
    processing_time: float = 0
    tokens_used: Dict = None
    cost: float = 0
    metadata: Dict = None


# ============================================================================
# Base Agent Class
# ============================================================================

class BaseAgent(ABC):
    """Abstract base class for all agents"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.request_count = 0
        self.total_tokens = 0
        self.total_cost = 0
        self.error_count = 0
        self.last_request_time = 0
        self.circuit_breaker = CircuitBreaker()
    
    @abstractmethod
    async def process(self, task: Task) -> TaskResult:
        """Process a task and return results"""
        pass
    
    async def health_check(self) -> bool:
        """Check if agent is healthy and available"""
        try:
            # Simple health check - can be overridden
            return self.circuit_breaker.state != 'open'
        except Exception:
            return False
    
    def calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate cost for token usage"""
        input_cost = (input_tokens / 1000) * self.config.cost_per_1k_input
        output_cost = (output_tokens / 1000) * self.config.cost_per_1k_output
        return input_cost + output_cost
    
    async def rate_limit_check(self) -> bool:
        """Check if we're within rate limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < (1 / self.config.rate_limit):
            wait_time = (1 / self.config.rate_limit) - time_since_last
            await asyncio.sleep(wait_time)
        
        self.last_request_time = time.time()
        return True


# ============================================================================
# Concrete Agent Implementations
# ============================================================================

class OpenAIAgent(BaseAgent):
    """OpenAI GPT models agent"""
    
    async def process(self, task: Task) -> TaskResult:
        """Process task using OpenAI API"""
        start_time = time.time()
        
        try:
            # Rate limiting
            await self.rate_limit_check()
            
            # Simulate API call (replace with actual implementation)
            await asyncio.sleep(0.5)  # Simulated delay
            
            result = {
                "response": f"OpenAI processed: {task.content}",
                "model": "gpt-4",
                "tokens": {"input": 100, "output": 150}
            }
            
            # Update metrics
            self.request_count += 1
            self.total_tokens += 250
            cost = self.calculate_cost(100, 150)
            self.total_cost += cost
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=True,
                result=result,
                processing_time=time.time() - start_time,
                tokens_used=result["tokens"],
                cost=cost
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"OpenAI agent error: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=False,
                result=None,
                error=str(e),
                processing_time=time.time() - start_time
            )


class ClaudeAgent(BaseAgent):
    """Anthropic Claude models agent"""
    
    async def process(self, task: Task) -> TaskResult:
        """Process task using Claude API"""
        start_time = time.time()
        
        try:
            await self.rate_limit_check()
            
            # Simulate API call
            await asyncio.sleep(0.4)
            
            result = {
                "response": f"Claude analyzed: {task.content}",
                "model": "claude-3",
                "tokens": {"input": 90, "output": 140}
            }
            
            self.request_count += 1
            self.total_tokens += 230
            cost = self.calculate_cost(90, 140)
            self.total_cost += cost
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=True,
                result=result,
                processing_time=time.time() - start_time,
                tokens_used=result["tokens"],
                cost=cost
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Claude agent error: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=False,
                result=None,
                error=str(e),
                processing_time=time.time() - start_time
            )


class GeminiAgent(BaseAgent):
    """Google Gemini models agent"""
    
    async def process(self, task: Task) -> TaskResult:
        """Process task using Gemini API"""
        start_time = time.time()
        
        try:
            await self.rate_limit_check()
            
            # Simulate API call
            await asyncio.sleep(0.6)
            
            result = {
                "response": f"Gemini evaluated: {task.content}",
                "model": "gemini-pro",
                "tokens": {"input": 110, "output": 160}
            }
            
            self.request_count += 1
            self.total_tokens += 270
            cost = self.calculate_cost(110, 160)
            self.total_cost += cost
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=True,
                result=result,
                processing_time=time.time() - start_time,
                tokens_used=result["tokens"],
                cost=cost
            )
            
        except Exception as e:
            self.error_count += 1
            logger.error(f"Gemini agent error: {e}")
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=self.config.agent_id,
                success=False,
                result=None,
                error=str(e),
                processing_time=time.time() - start_time
            )


# ============================================================================
# Circuit Breaker Pattern
# ============================================================================

class CircuitBreaker:
    """Implements circuit breaker pattern for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, reset_timeout: int = 60):
        self.failure_count = 0
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half-open
    
    def record_success(self):
        """Record successful call"""
        if self.state == 'half-open':
            self.state = 'closed'
            self.failure_count = 0
    
    def record_failure(self):
        """Record failed call"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = 'open'
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")
    
    def can_execute(self) -> bool:
        """Check if execution is allowed"""
        if self.state == 'closed':
            return True
        
        if self.state == 'open':
            if time.time() - self.last_failure_time > self.reset_timeout:
                self.state = 'half-open'
                logger.info("Circuit breaker moved to half-open state")
                return True
            return False
        
        # half-open state
        return True


# ============================================================================
# Context Manager
# ============================================================================

class ContextManager:
    """Manages context and state across agent interactions"""
    
    def __init__(self, max_history: int = 100):
        self.conversation_history = []
        self.shared_memory = {}
        self.max_history = max_history
        self.sessions = {}
    
    def create_session(self, session_id: str) -> Dict:
        """Create a new session"""
        self.sessions[session_id] = {
            'created_at': datetime.now(),
            'history': [],
            'context': {}
        }
        return self.sessions[session_id]
    
    def add_interaction(self, session_id: str, agent_id: str, 
                       task: Task, result: TaskResult):
        """Add an interaction to the history"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        interaction = {
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'task': task.content,
            'result': result.result,
            'success': result.success
        }
        
        self.sessions[session_id]['history'].append(interaction)
        
        # Trim history if needed
        if len(self.sessions[session_id]['history']) > self.max_history:
            self.sessions[session_id]['history'] = \
                self.sessions[session_id]['history'][-self.max_history:]
    
    def get_context(self, session_id: str) -> Dict:
        """Get context for a session"""
        if session_id not in self.sessions:
            return {}
        
        return {
            'history': self.sessions[session_id]['history'][-10:],
            'context': self.sessions[session_id].get('context', {}),
            'shared_memory': self.shared_memory
        }
    
    def update_context(self, session_id: str, key: str, value: Any):
        """Update session context"""
        if session_id not in self.sessions:
            self.create_session(session_id)
        
        self.sessions[session_id]['context'][key] = value
    
    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]


# ============================================================================
# Router
# ============================================================================

class Router:
    """Routes tasks to appropriate agents"""
    
    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.agent_scores = defaultdict(lambda: {'success': 0, 'total': 0})
    
    def select_agent(self, task: Task, available_agents: List[BaseAgent]) -> BaseAgent:
        """Select the best agent for a task"""
        
        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(task, available_agents)
        elif self.strategy == RoutingStrategy.PERFORMANCE_PRIORITY:
            return self._performance_select(task, available_agents)
        elif self.strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._capability_select(task, available_agents)
        else:  # INTELLIGENT
            return self._intelligent_select(task, available_agents)
    
    def _capability_select(self, task: Task, agents: List[BaseAgent]) -> BaseAgent:
        """Select based on agent capabilities"""
        task_type = task.task_type
        
        # Find agents with matching capabilities
        capable_agents = [
            agent for agent in agents 
            if task_type in agent.config.capabilities
        ]
        
        if not capable_agents:
            # Fallback to first available agent
            return agents[0] if agents else None
        
        # Return agent with lowest error rate among capable agents
        return min(capable_agents, key=lambda a: a.error_count)
    
    def _cost_optimized_select(self, task: Task, agents: List[BaseAgent]) -> BaseAgent:
        """Select the most cost-effective agent"""
        return min(agents, key=lambda a: a.config.cost_per_1k_output)
    
    def _performance_select(self, task: Task, agents: List[BaseAgent]) -> BaseAgent:
        """Select based on performance history"""
        best_agent = None
        best_score = -1
        
        for agent in agents:
            agent_id = agent.config.agent_id
            if self.agent_scores[agent_id]['total'] > 0:
                score = self.agent_scores[agent_id]['success'] / \
                        self.agent_scores[agent_id]['total']
                if score > best_score:
                    best_score = score
                    best_agent = agent
        
        return best_agent if best_agent else agents[0]
    
    def _round_robin_select(self, agents: List[BaseAgent]) -> BaseAgent:
        """Simple round-robin selection"""
        if not hasattr(self, '_rr_index'):
            self._rr_index = 0
        
        agent = agents[self._rr_index % len(agents)]
        self._rr_index += 1
        return agent
    
    def _intelligent_select(self, task: Task, agents: List[BaseAgent]) -> BaseAgent:
        """Intelligent selection based on multiple factors"""
        scores = {}
        
        for agent in agents:
            score = 0
            agent_id = agent.config.agent_id
            
            # Capability match (weight: 40%)
            if task.task_type in agent.config.capabilities:
                score += 40
            
            # Success rate (weight: 30%)
            if self.agent_scores[agent_id]['total'] > 0:
                success_rate = self.agent_scores[agent_id]['success'] / \
                              self.agent_scores[agent_id]['total']
                score += success_rate * 30
            
            # Cost efficiency (weight: 20%)
            # Lower cost = higher score
            max_cost = max(a.config.cost_per_1k_output for a in agents)
            if max_cost > 0:
                cost_score = (1 - agent.config.cost_per_1k_output / max_cost) * 20
                score += cost_score
            
            # Availability (weight: 10%)
            if agent.circuit_breaker.state == 'closed':
                score += 10
            elif agent.circuit_breaker.state == 'half-open':
                score += 5
            
            scores[agent] = score
        
        # Return agent with highest score
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def update_scores(self, agent_id: str, success: bool):
        """Update agent performance scores"""
        self.agent_scores[agent_id]['total'] += 1
        if success:
            self.agent_scores[agent_id]['success'] += 1


# ============================================================================
# Orchestrator
# ============================================================================

class Orchestrator:
    """Main orchestration engine"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.router = Router()
        self.context_manager = ContextManager()
        self.task_queue = asyncio.Queue()
        self.result_queue = asyncio.Queue()
        self.running = False
    
    def register_agent(self, agent: BaseAgent):
        """Register an agent with the orchestrator"""
        agent_id = agent.config.agent_id
        self.agents[agent_id] = agent
        logger.info(f"Registered agent: {agent_id}")
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Unregistered agent: {agent_id}")
    
    async def process_task(self, task: Task, session_id: Optional[str] = None) -> TaskResult:
        """Process a single task"""
        # Get available agents
        available_agents = await self._get_available_agents()
        
        if not available_agents:
            logger.error("No available agents to process task")
            return TaskResult(
                task_id=task.task_id,
                agent_id="none",
                success=False,
                result=None,
                error="No available agents"
            )
        
        # Select agent
        agent = self.router.select_agent(task, available_agents)
        
        # Add context if session exists
        if session_id:
            context = self.context_manager.get_context(session_id)
            task.context.update(context)
        
        # Process task
        try:
            result = await agent.process(task)
            
            # Update router scores
            self.router.update_scores(agent.config.agent_id, result.success)
            
            # Update context if session exists
            if session_id:
                self.context_manager.add_interaction(
                    session_id, agent.config.agent_id, task, result
                )
            
            # Update circuit breaker
            if result.success:
                agent.circuit_breaker.record_success()
            else:
                agent.circuit_breaker.record_failure()
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing task: {e}")
            agent.circuit_breaker.record_failure()
            
            return TaskResult(
                task_id=task.task_id,
                agent_id=agent.config.agent_id,
                success=False,
                result=None,
                error=str(e)
            )
    
    async def process_workflow(self, tasks: List[Task], parallel: bool = False) -> List[TaskResult]:
        """Process multiple tasks as a workflow"""
        if parallel:
            # Process tasks in parallel
            tasks_coroutines = [self.process_task(task) for task in tasks]
            results = await asyncio.gather(*tasks_coroutines, return_exceptions=True)
            
            # Convert exceptions to TaskResults
            final_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    final_results.append(TaskResult(
                        task_id=tasks[i].task_id,
                        agent_id="unknown",
                        success=False,
                        result=None,
                        error=str(result)
                    ))
                else:
                    final_results.append(result)
            
            return final_results
        else:
            # Process tasks sequentially
            results = []
            for task in tasks:
                result = await self.process_task(task)
                results.append(result)
                
                # Pass result to next task's context
                if len(tasks) > tasks.index(task) + 1:
                    next_task = tasks[tasks.index(task) + 1]
                    next_task.context['previous_result'] = result.result
            
            return results
    
    async def _get_available_agents(self) -> List[BaseAgent]:
        """Get list of available agents"""
        available = []
        
        for agent in self.agents.values():
            if await agent.health_check() and agent.circuit_breaker.can_execute():
                available.append(agent)
        
        return available
    
    def get_metrics(self) -> Dict:
        """Get system metrics"""
        metrics = {
            'agents': {},
            'total_requests': 0,
            'total_errors': 0,
            'total_cost': 0,
            'total_tokens': 0
        }
        
        for agent_id, agent in self.agents.items():
            metrics['agents'][agent_id] = {
                'requests': agent.request_count,
                'errors': agent.error_count,
                'tokens': agent.total_tokens,
                'cost': agent.total_cost,
                'circuit_breaker': agent.circuit_breaker.state,
                'success_rate': (agent.request_count - agent.error_count) / agent.request_count 
                                if agent.request_count > 0 else 0
            }
            
            metrics['total_requests'] += agent.request_count
            metrics['total_errors'] += agent.error_count
            metrics['total_cost'] += agent.total_cost
            metrics['total_tokens'] += agent.total_tokens
        
        return metrics


# ============================================================================
# Result Aggregator
# ============================================================================

class ResultAggregator:
    """Aggregates and synthesizes results from multiple agents"""
    
    def __init__(self):
        self.strategies = {
            'first': self._first_valid,
            'majority_vote': self._majority_vote,
            'weighted_average': self._weighted_average,
            'consensus': self._consensus,
            'merge': self._merge_results
        }
    
    def aggregate(self, results: List[TaskResult], strategy: str = 'merge') -> Any:
        """Aggregate multiple results using specified strategy"""
        if strategy not in self.strategies:
            strategy = 'merge'
        
        return self.strategies[strategy](results)
    
    def _first_valid(self, results: List[TaskResult]) -> Any:
        """Return first valid result"""
        for result in results:
            if result.success and result.result:
                return result.result
        return None
    
    def _majority_vote(self, results: List[TaskResult]) -> Any:
        """Return most common result"""
        from collections import Counter
        
        valid_results = [str(r.result) for r in results if r.success]
        if not valid_results:
            return None
        
        counter = Counter(valid_results)
        return counter.most_common(1)[0][0]
    
    def _weighted_average(self, results: List[TaskResult]) -> Any:
        """Calculate weighted average based on confidence scores"""
        # This is a simplified example
        valid_results = [r for r in results if r.success]
        if not valid_results:
            return None
        
        # For demonstration, using processing time as inverse weight
        total_weight = sum(1 / r.processing_time for r in valid_results)
        
        if total_weight == 0:
            return valid_results[0].result
        
        # In real implementation, would need numerical results to average
        return valid_results[0].result
    
    def _consensus(self, results: List[TaskResult]) -> Any:
        """Build consensus from results"""
        valid_results = [r.result for r in results if r.success]
        
        if not valid_results:
            return None
        
        # Simple consensus: all results must agree
        first_result = str(valid_results[0])
        if all(str(r) == first_result for r in valid_results):
            return valid_results[0]
        
        # If no consensus, return with confidence score
        return {
            'consensus': False,
            'results': valid_results,
            'confidence': len(set(str(r) for r in valid_results)) / len(valid_results)
        }
    
    def _merge_results(self, results: List[TaskResult]) -> Any:
        """Merge all results into a combined output"""
        merged = {
            'responses': [],
            'success_count': 0,
            'total_processing_time': 0,
            'total_cost': 0,
            'agents_used': []
        }
        
        for result in results:
            if result.success:
                merged['responses'].append({
                    'agent_id': result.agent_id,
                    'result': result.result
                })
                merged['success_count'] += 1
            
            merged['total_processing_time'] += result.processing_time
            merged['total_cost'] += result.cost
            merged['agents_used'].append(result.agent_id)
        
        return merged


# ============================================================================
# Example Usage
# ============================================================================

async def main_example():
    """Example usage of the MAOF framework"""
    
    # Create orchestrator
    orchestrator = Orchestrator()
    
    # Configure and register agents
    openai_config = AgentConfig(
        agent_id="openai-1",
        name="GPT-4",
        provider="OpenAI",
        agent_type=AgentType.LLM,
        endpoint="https://api.openai.com/v1",
        api_key=os.getenv("OPENAI_API_KEY", "demo-key"),
        capabilities=["text", "code", "analysis"],
        max_tokens=8192
    )
    
    claude_config = AgentConfig(
        agent_id="claude-1",
        name="Claude 3",
        provider="Anthropic",
        agent_type=AgentType.LLM,
        endpoint="https://api.anthropic.com/v1",
        api_key=os.getenv("ANTHROPIC_API_KEY", "demo-key"),
        capabilities=["text", "code", "safety"],
        max_tokens=100000
    )
    
    gemini_config = AgentConfig(
        agent_id="gemini-1",
        name="Gemini Pro",
        provider="Google",
        agent_type=AgentType.MULTIMODAL,
        endpoint="https://api.google.com/gemini",
        api_key=os.getenv("GOOGLE_API_KEY", "demo-key"),
        capabilities=["text", "vision", "search"],
        max_tokens=32768
    )
    
    # Register agents
    orchestrator.register_agent(OpenAIAgent(openai_config))
    orchestrator.register_agent(ClaudeAgent(claude_config))
    orchestrator.register_agent(GeminiAgent(gemini_config))
    
    # Create tasks
    task1 = Task(
        task_id="task-001",
        task_type="text",
        content="Analyze the sentiment of this text: 'I love this product!'",
        priority=TaskPriority.HIGH
    )
    
    task2 = Task(
        task_id="task-002",
        task_type="code",
        content="Write a Python function to calculate fibonacci numbers",
        priority=TaskPriority.MEDIUM
    )
    
    task3 = Task(
        task_id="task-003",
        task_type="analysis",
        content="Compare the pros and cons of microservices architecture",
        priority=TaskPriority.LOW
    )
    
    # Process single task
    print("Processing single task...")
    result1 = await orchestrator.process_task(task1)
    print(f"Result: {result1.success}, Agent: {result1.agent_id}")
    
    # Process workflow (sequential)
    print("\nProcessing sequential workflow...")
    workflow_results = await orchestrator.process_workflow(
        [task1, task2, task3], 
        parallel=False
    )
    for result in workflow_results:
        print(f"Task {result.task_id}: Success={result.success}, Agent={result.agent_id}")
    
    # Process workflow (parallel)
    print("\nProcessing parallel workflow...")
    parallel_results = await orchestrator.process_workflow(
        [task1, task2, task3], 
        parallel=True
    )
    for result in parallel_results:
        print(f"Task {result.task_id}: Success={result.success}, Agent={result.agent_id}")
    
    # Aggregate results
    aggregator = ResultAggregator()
    merged = aggregator.aggregate(parallel_results, strategy='merge')
    print(f"\nAggregated results: {merged['success_count']} successful out of {len(parallel_results)}")
    
    # Get metrics
    metrics = orchestrator.get_metrics()
    print(f"\nSystem metrics: {json.dumps(metrics, indent=2)}")


if __name__ == "__main__":
    # Run the example
    asyncio.run(main_example())
