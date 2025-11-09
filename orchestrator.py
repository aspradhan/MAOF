"""
MAOF Orchestrator - Enhanced Version
Integrates all components: agents, routing, context, database, monitoring
"""

import os
import asyncio
from typing import Dict, List, Optional, Any
from collections import defaultdict

from maof_framework_enhanced import (
    Task, TaskResult, AgentConfig, AgentType, RoutingStrategy,
    TokenUsage, logger
)
from agents import create_agent, OpenAIAgent, ClaudeAgent, GeminiAgent, MockAgent
from database import Database


# ============================================================================
# Enhanced Router with Load Balancing
# ============================================================================

class Router:
    """Enhanced router with multiple strategies"""

    def __init__(self, strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT):
        self.strategy = strategy
        self.agent_scores = defaultdict(lambda: {'success': 0, 'total': 0})
        self._rr_index = 0

    def select_agent(self, task: Task, available_agents: List) -> Optional[Any]:
        """Select the best agent for a task"""
        if not available_agents:
            return None

        if self.strategy == RoutingStrategy.ROUND_ROBIN:
            return self._round_robin_select(available_agents)
        elif self.strategy == RoutingStrategy.COST_OPTIMIZED:
            return self._cost_optimized_select(task, available_agents)
        elif self.strategy == RoutingStrategy.PERFORMANCE_PRIORITY:
            return self._performance_select(task, available_agents)
        elif self.strategy == RoutingStrategy.CAPABILITY_BASED:
            return self._capability_select(task, available_agents)
        elif self.strategy == RoutingStrategy.LOAD_BALANCED:
            return self._load_balanced_select(available_agents)
        else:  # INTELLIGENT
            return self._intelligent_select(task, available_agents)

    def _capability_select(self, task: Task, agents: List) -> Any:
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

    def _cost_optimized_select(self, task: Task, agents: List) -> Any:
        """Select the most cost-effective agent"""
        return min(agents, key=lambda a: a.config.cost_per_1k_output)

    def _performance_select(self, task: Task, agents: List) -> Any:
        """Select based on performance history"""
        best_agent = None
        best_score = -1

        for agent in agents:
            agent_id = agent.config.agent_id
            if self.agent_scores[agent_id]['total'] > 0:
                score = (
                    self.agent_scores[agent_id]['success'] /
                    self.agent_scores[agent_id]['total']
                )
                if score > best_score:
                    best_score = score
                    best_agent = agent

        return best_agent if best_agent else agents[0]

    def _round_robin_select(self, agents: List) -> Any:
        """Simple round-robin selection"""
        agent = agents[self._rr_index % len(agents)]
        self._rr_index += 1
        return agent

    def _load_balanced_select(self, agents: List) -> Any:
        """Select agent with lowest current load"""
        # Use request count as load indicator
        return min(agents, key=lambda a: a.request_count)

    def _intelligent_select(self, task: Task, agents: List) -> Any:
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
                success_rate = (
                    self.agent_scores[agent_id]['success'] /
                    self.agent_scores[agent_id]['total']
                )
                score += success_rate * 30

            # Cost efficiency (weight: 20%)
            max_cost = max(a.config.cost_per_1k_output for a in agents)
            if max_cost > 0:
                cost_score = (1 - agent.config.cost_per_1k_output / max_cost) * 20
                score += cost_score

            # Availability (weight: 10%)
            circuit_state = agent.circuit_breaker.state
            if circuit_state == 'closed':
                score += 10
            elif circuit_state == 'half-open':
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
# Enhanced Context Manager
# ============================================================================

class ContextManager:
    """Enhanced context management with database persistence"""

    def __init__(self, max_history: int = 100, database: Optional[Database] = None):
        self.max_history = max_history
        self.sessions: Dict[str, Dict] = {}
        self.database = database

    def create_session(self, session_id: str, user_id: Optional[str] = None) -> Dict:
        """Create a new session"""
        from datetime import datetime, timedelta

        session = {
            'created_at': datetime.now(),
            'updated_at': datetime.now(),
            'history': [],
            'context': {},
            'user_id': user_id,
            'expires_at': datetime.now() + timedelta(hours=24)
        }
        self.sessions[session_id] = session

        logger.info("session_created", session_id=session_id, user_id=user_id)
        return session

    def add_interaction(
        self,
        session_id: str,
        agent_id: str,
        task: Task,
        result: TaskResult
    ):
        """Add an interaction to the session history"""
        from datetime import datetime

        if session_id not in self.sessions:
            self.create_session(session_id)

        interaction = {
            'timestamp': datetime.now(),
            'agent_id': agent_id,
            'task': task.content,
            'task_type': task.task_type,
            'result': result.result if result.success else None,
            'success': result.success,
            'error': result.error,
            'tokens_used': (
                result.tokens_used.total_tokens
                if result.tokens_used else 0
            ),
            'cost': result.cost
        }

        self.sessions[session_id]['history'].append(interaction)
        self.sessions[session_id]['updated_at'] = datetime.now()

        # Trim history if needed
        if len(self.sessions[session_id]['history']) > self.max_history:
            self.sessions[session_id]['history'] = \
                self.sessions[session_id]['history'][-self.max_history:]

    def get_context(self, session_id: str) -> Dict:
        """Get context for a session"""
        if session_id not in self.sessions:
            return {}

        session = self.sessions[session_id]

        return {
            'history': session['history'][-10:],  # Last 10 interactions
            'context': session.get('context', {}),
            'session_info': {
                'created_at': session['created_at'],
                'updated_at': session['updated_at'],
                'user_id': session.get('user_id')
            }
        }

    def update_context(self, session_id: str, key: str, value: Any):
        """Update session context"""
        from datetime import datetime

        if session_id not in self.sessions:
            self.create_session(session_id)

        self.sessions[session_id]['context'][key] = value
        self.sessions[session_id]['updated_at'] = datetime.now()

    def clear_session(self, session_id: str):
        """Clear a session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info("session_cleared", session_id=session_id)


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
        valid_results = [r for r in results if r.success]
        if not valid_results:
            return None

        # Use processing time as inverse weight
        total_weight = sum(1 / r.processing_time for r in valid_results if r.processing_time > 0)

        if total_weight == 0:
            return valid_results[0].result

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
            'confidence': 1.0 - (len(set(str(r) for r in valid_results)) / len(valid_results))
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
# Main Orchestrator
# ============================================================================

class Orchestrator:
    """Enhanced orchestration engine"""

    def __init__(
        self,
        database: Optional[Database] = None,
        routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
    ):
        self.agents: Dict[str, Any] = {}
        self.router = Router(routing_strategy)
        self.context_manager = ContextManager(database=database)
        self.result_aggregator = ResultAggregator()
        self.database = database

    def register_agent(self, agent):
        """Register an agent with the orchestrator"""
        agent_id = agent.config.agent_id
        self.agents[agent_id] = agent
        logger.info("agent_registered", agent_id=agent_id, provider=agent.config.provider)

    def unregister_agent(self, agent_id: str):
        """Unregister an agent"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info("agent_unregistered", agent_id=agent_id)

    async def process_task(
        self,
        task: Task,
        session_id: Optional[str] = None
    ) -> TaskResult:
        """Process a single task"""
        # Get available agents
        available_agents = await self._get_available_agents()

        if not available_agents:
            logger.error("no_available_agents", task_id=task.task_id)
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

            return result

        except Exception as e:
            logger.error("orchestrator_error", task_id=task.task_id, error=str(e))

            return TaskResult(
                task_id=task.task_id,
                agent_id=agent.config.agent_id if agent else "unknown",
                success=False,
                result=None,
                error=str(e)
            )

    async def process_workflow(
        self,
        tasks: List[Task],
        parallel: bool = False,
        session_id: Optional[str] = None
    ) -> List[TaskResult]:
        """Process multiple tasks as a workflow"""
        if parallel:
            # Process tasks in parallel
            tasks_coroutines = [
                self.process_task(task, session_id) for task in tasks
            ]
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
                result = await self.process_task(task, session_id)
                results.append(result)

                # Pass result to next task's context
                task_idx = tasks.index(task)
                if task_idx < len(tasks) - 1:
                    next_task = tasks[task_idx + 1]
                    next_task.context['previous_result'] = result.result

            return results

    async def _get_available_agents(self) -> List:
        """Get list of available agents"""
        available = []

        for agent in self.agents.values():
            if await agent.health_check() and await agent.circuit_breaker.can_execute():
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
            agent_metrics = agent.get_metrics()
            metrics['agents'][agent_id] = agent_metrics

            metrics['total_requests'] += agent.request_count
            metrics['total_errors'] += agent.error_count
            metrics['total_cost'] += agent.total_cost
            metrics['total_tokens'] += agent.total_tokens

        return metrics


# ============================================================================
# Orchestrator Factory
# ============================================================================

async def create_orchestrator(
    database: Optional[Database] = None,
    routing_strategy: RoutingStrategy = RoutingStrategy.INTELLIGENT
) -> Orchestrator:
    """Create and initialize an orchestrator with default agents"""
    orchestrator = Orchestrator(database, routing_strategy)

    # Register default agents from environment variables
    agents_to_register = []

    # OpenAI
    if os.getenv('OPENAI_API_KEY'):
        openai_config = AgentConfig(
            agent_id="openai-gpt4",
            name="GPT-4",
            provider="openai",
            agent_type=AgentType.LLM,
            endpoint="https://api.openai.com/v1",
            api_key=os.getenv('OPENAI_API_KEY'),
            capabilities=["text", "code", "analysis", "general"],
            model_name="gpt-4",
            max_tokens=8192,
            cost_per_1k_input=0.03,
            cost_per_1k_output=0.06,
            context_window=8192
        )
        agents_to_register.append(('openai', openai_config))

    # Claude
    if os.getenv('ANTHROPIC_API_KEY'):
        claude_config = AgentConfig(
            agent_id="claude-sonnet",
            name="Claude 3 Sonnet",
            provider="anthropic",
            agent_type=AgentType.LLM,
            endpoint="https://api.anthropic.com/v1",
            api_key=os.getenv('ANTHROPIC_API_KEY'),
            capabilities=["text", "code", "analysis", "safety", "general"],
            model_name="claude-3-sonnet-20240229",
            max_tokens=4096,
            cost_per_1k_input=0.003,
            cost_per_1k_output=0.015,
            context_window=200000
        )
        agents_to_register.append(('claude', claude_config))

    # Gemini
    if os.getenv('GOOGLE_API_KEY'):
        gemini_config = AgentConfig(
            agent_id="gemini-pro",
            name="Gemini Pro",
            provider="google",
            agent_type=AgentType.MULTIMODAL,
            endpoint="https://generativelanguage.googleapis.com",
            api_key=os.getenv('GOOGLE_API_KEY'),
            capabilities=["text", "vision", "multimodal", "general"],
            model_name="gemini-pro",
            max_tokens=2048,
            cost_per_1k_input=0.00025,
            cost_per_1k_output=0.0005,
            context_window=32768
        )
        agents_to_register.append(('gemini', gemini_config))

    # Register agents
    for provider, config in agents_to_register:
        try:
            agent = create_agent(config)
            orchestrator.register_agent(agent)
        except Exception as e:
            logger.error("agent_registration_failed", provider=provider, error=str(e))

    # If no agents registered, add a mock agent for testing
    if len(orchestrator.agents) == 0:
        logger.warning("no_api_keys_found_using_mock")
        mock_config = AgentConfig(
            agent_id="mock-agent",
            name="Mock Agent",
            provider="mock",
            agent_type=AgentType.LLM,
            endpoint="http://localhost",
            api_key="mock-key",
            capabilities=["text", "code", "analysis", "general"],
            model_name="mock-model"
        )
        orchestrator.register_agent(MockAgent(mock_config))

    logger.info("orchestrator_created", agents=len(orchestrator.agents))
    return orchestrator
