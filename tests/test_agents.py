"""
Unit tests for MAOF agents
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from maof_framework_enhanced import AgentConfig, AgentType, Task, TaskPriority
from agents import MockAgent, create_agent


@pytest.fixture
def mock_agent_config():
    """Create a mock agent configuration"""
    return AgentConfig(
        agent_id="test-agent",
        name="Test Agent",
        provider="mock",
        agent_type=AgentType.LLM,
        endpoint="http://test",
        api_key="test-key",
        capabilities=["text", "code"],
        model_name="test-model"
    )


@pytest.fixture
def sample_task():
    """Create a sample task"""
    return Task(
        task_id="test-001",
        task_type="text",
        content="Test content",
        priority=TaskPriority.MEDIUM
    )


class TestMockAgent:
    """Test MockAgent implementation"""

    @pytest.mark.asyncio
    async def test_agent_initialization(self, mock_agent_config):
        """Test agent initialization"""
        agent = MockAgent(mock_agent_config)
        assert agent.config.agent_id == "test-agent"
        assert agent.request_count == 0
        assert agent.total_cost == 0

    @pytest.mark.asyncio
    async def test_agent_process_task(self, mock_agent_config, sample_task):
        """Test task processing"""
        agent = MockAgent(mock_agent_config)
        result = await agent.process(sample_task)

        assert result.task_id == sample_task.task_id
        assert result.success is True
        assert result.agent_id == "test-agent"
        assert agent.request_count == 1

    @pytest.mark.asyncio
    async def test_agent_health_check(self, mock_agent_config):
        """Test health check"""
        agent = MockAgent(mock_agent_config)
        health = await agent.health_check()
        assert health is True

    @pytest.mark.asyncio
    async def test_cost_calculation(self, mock_agent_config):
        """Test cost calculation"""
        agent = MockAgent(mock_agent_config)
        cost = agent.calculate_cost(1000, 500)
        expected = (1000/1000 * 0.01) + (500/1000 * 0.03)
        assert abs(cost - expected) < 0.001


class TestAgentFactory:
    """Test agent factory"""

    def test_create_mock_agent(self, mock_agent_config):
        """Test creating a mock agent"""
        agent = create_agent(mock_agent_config)
        assert isinstance(agent, MockAgent)
        assert agent.config.agent_id == "test-agent"


@pytest.mark.asyncio
async def test_circuit_breaker(mock_agent_config, sample_task):
    """Test circuit breaker functionality"""
    agent = MockAgent(mock_agent_config)

    # Simulate failures to trigger circuit breaker
    for i in range(6):
        await agent.circuit_breaker.record_failure()

    # Circuit should be open
    can_execute = await agent.circuit_breaker.can_execute()
    assert can_execute is False
    assert agent.circuit_breaker.state == 'open'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
