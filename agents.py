"""
MAOF Agent Implementations with Real API Integrations
Includes: OpenAI, Anthropic Claude, Google Gemini
"""

import os
import asyncio
from typing import Dict, Any
from maof_framework_enhanced import (
    BaseAgent, Task, TaskResult, TokenUsage, AgentConfig, ErrorCode, logger
)

# API Client imports (with graceful fallbacks)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai_sdk_not_available", message="Install with: pip install openai")

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic_sdk_not_available", message="Install with: pip install anthropic")

try:
    import google.generativeai as genai
    GOOGLE_AVAILABLE = True
except ImportError:
    GOOGLE_AVAILABLE = False
    logger.warning("google_sdk_not_available", message="Install with: pip install google-generativeai")


# ============================================================================
# OpenAI Agent Implementation
# ============================================================================

class OpenAIAgent(BaseAgent):
    """Production OpenAI agent with real API integration"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        if not OPENAI_AVAILABLE:
            raise ImportError("OpenAI SDK not installed. Run: pip install openai")

        # Initialize OpenAI client
        self.client = openai.AsyncOpenAI(
            api_key=os.getenv('OPENAI_API_KEY', config.api_key),
            timeout=config.timeout,
            max_retries=0  # We handle retries ourselves
        )

        # Default model mapping
        self.model = config.model_name or "gpt-3.5-turbo"

    async def _call_api(self, task: Task) -> Dict[str, Any]:
        """Call OpenAI API"""
        try:
            # Prepare messages
            messages = self._prepare_messages(task)

            # Calculate max tokens for response
            max_tokens = min(
                task.max_tokens or self.config.max_tokens,
                self.config.context_window
            )

            # Make API call
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=task.constraints.get('temperature', 0.7),
                top_p=task.constraints.get('top_p', 1.0),
                n=1,
                stream=False,
            )

            # Extract result
            choice = response.choices[0]
            result_text = choice.message.content

            # Extract token usage
            tokens_used = TokenUsage(
                input_tokens=response.usage.prompt_tokens,
                output_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                estimated=False
            )

            return {
                'result': result_text,
                'tokens_used': tokens_used,
                'metadata': {
                    'model': response.model,
                    'finish_reason': choice.finish_reason,
                    'response_id': response.id,
                }
            }

        except openai.RateLimitError as e:
            logger.error("openai_rate_limit", error=str(e))
            raise Exception(f"Rate limit exceeded: {e}")
        except openai.APITimeoutError as e:
            logger.error("openai_timeout", error=str(e))
            raise Exception(f"API timeout: {e}")
        except openai.AuthenticationError as e:
            logger.error("openai_auth_error", error=str(e))
            raise Exception(f"Authentication failed: {e}")
        except Exception as e:
            logger.error("openai_api_error", error=str(e))
            raise

    def _prepare_messages(self, task: Task) -> list:
        """Prepare messages for OpenAI API"""
        messages = []

        # System message from context if available
        if task.context.get('system_prompt'):
            messages.append({
                'role': 'system',
                'content': task.context['system_prompt']
            })

        # Conversation history from context
        if task.context.get('history'):
            for msg in task.context['history']:
                if isinstance(msg, dict) and 'role' in msg and 'content' in msg:
                    messages.append(msg)

        # Current user message
        if isinstance(task.content, str):
            messages.append({
                'role': 'user',
                'content': task.content
            })
        elif isinstance(task.content, dict) and 'messages' in task.content:
            messages.extend(task.content['messages'])
        else:
            messages.append({
                'role': 'user',
                'content': str(task.content)
            })

        return messages


# ============================================================================
# Anthropic Claude Agent Implementation
# ============================================================================

class ClaudeAgent(BaseAgent):
    """Production Anthropic Claude agent with real API integration"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("Anthropic SDK not installed. Run: pip install anthropic")

        # Initialize Anthropic client
        self.client = anthropic.AsyncAnthropic(
            api_key=os.getenv('ANTHROPIC_API_KEY', config.api_key),
            timeout=config.timeout,
            max_retries=0  # We handle retries ourselves
        )

        # Default model mapping
        self.model = config.model_name or "claude-3-sonnet-20240229"

    async def _call_api(self, task: Task) -> Dict[str, Any]:
        """Call Anthropic Claude API"""
        try:
            # Prepare messages
            messages = self._prepare_messages(task)
            system_prompt = task.context.get('system_prompt', '')

            # Calculate max tokens for response
            max_tokens = min(
                task.max_tokens or self.config.max_tokens,
                4096  # Claude's max output tokens
            )

            # Make API call
            response = await self.client.messages.create(
                model=self.model,
                max_tokens=max_tokens,
                temperature=task.constraints.get('temperature', 0.7),
                system=system_prompt,
                messages=messages,
            )

            # Extract result
            result_text = ""
            for block in response.content:
                if block.type == "text":
                    result_text += block.text

            # Extract token usage
            tokens_used = TokenUsage(
                input_tokens=response.usage.input_tokens,
                output_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
                estimated=False
            )

            return {
                'result': result_text,
                'tokens_used': tokens_used,
                'metadata': {
                    'model': response.model,
                    'stop_reason': response.stop_reason,
                    'response_id': response.id,
                }
            }

        except anthropic.RateLimitError as e:
            logger.error("claude_rate_limit", error=str(e))
            raise Exception(f"Rate limit exceeded: {e}")
        except anthropic.APITimeoutError as e:
            logger.error("claude_timeout", error=str(e))
            raise Exception(f"API timeout: {e}")
        except anthropic.AuthenticationError as e:
            logger.error("claude_auth_error", error=str(e))
            raise Exception(f"Authentication failed: {e}")
        except Exception as e:
            logger.error("claude_api_error", error=str(e))
            raise

    def _prepare_messages(self, task: Task) -> list:
        """Prepare messages for Claude API"""
        messages = []

        # Conversation history from context (skip system messages)
        if task.context.get('history'):
            for msg in task.context['history']:
                if isinstance(msg, dict) and msg.get('role') != 'system':
                    messages.append({
                        'role': msg['role'],
                        'content': msg['content']
                    })

        # Current user message
        if isinstance(task.content, str):
            messages.append({
                'role': 'user',
                'content': task.content
            })
        elif isinstance(task.content, dict) and 'messages' in task.content:
            # Filter out system messages
            for msg in task.content['messages']:
                if msg.get('role') != 'system':
                    messages.append(msg)
        else:
            messages.append({
                'role': 'user',
                'content': str(task.content)
            })

        return messages


# ============================================================================
# Google Gemini Agent Implementation
# ============================================================================

class GeminiAgent(BaseAgent):
    """Production Google Gemini agent with real API integration"""

    def __init__(self, config: AgentConfig):
        super().__init__(config)
        if not GOOGLE_AVAILABLE:
            raise ImportError("Google Generative AI SDK not installed. Run: pip install google-generativeai")

        # Initialize Gemini
        api_key = os.getenv('GOOGLE_API_KEY', config.api_key)
        genai.configure(api_key=api_key)

        # Default model mapping
        model_name = config.model_name or "gemini-pro"
        self.model = genai.GenerativeModel(model_name)

    async def _call_api(self, task: Task) -> Dict[str, Any]:
        """Call Google Gemini API"""
        try:
            # Prepare prompt
            prompt = self._prepare_prompt(task)

            # Generation config
            generation_config = {
                'temperature': task.constraints.get('temperature', 0.7),
                'top_p': task.constraints.get('top_p', 1.0),
                'top_k': task.constraints.get('top_k', 40),
                'max_output_tokens': min(
                    task.max_tokens or self.config.max_tokens,
                    2048  # Gemini's typical max
                ),
            }

            # Make API call (Gemini SDK doesn't have native async, so we wrap it)
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=generation_config
            )

            # Extract result
            result_text = response.text if hasattr(response, 'text') else str(response)

            # Estimate tokens (Gemini doesn't always provide exact counts)
            # We'll use our estimation
            estimated_input = len(prompt) // 4
            estimated_output = len(result_text) // 4

            tokens_used = TokenUsage(
                input_tokens=estimated_input,
                output_tokens=estimated_output,
                total_tokens=estimated_input + estimated_output,
                estimated=True
            )

            return {
                'result': result_text,
                'tokens_used': tokens_used,
                'metadata': {
                    'model': self.model.model_name,
                    'finish_reason': (
                        response.candidates[0].finish_reason.name
                        if hasattr(response, 'candidates') and response.candidates
                        else 'UNKNOWN'
                    ),
                }
            }

        except Exception as e:
            # Gemini SDK has different exception structure
            error_msg = str(e)
            if 'quota' in error_msg.lower() or 'rate' in error_msg.lower():
                logger.error("gemini_rate_limit", error=error_msg)
                raise Exception(f"Rate limit exceeded: {e}")
            elif 'timeout' in error_msg.lower():
                logger.error("gemini_timeout", error=error_msg)
                raise Exception(f"API timeout: {e}")
            elif 'auth' in error_msg.lower() or 'api key' in error_msg.lower():
                logger.error("gemini_auth_error", error=error_msg)
                raise Exception(f"Authentication failed: {e}")
            else:
                logger.error("gemini_api_error", error=error_msg)
                raise

    def _prepare_prompt(self, task: Task) -> str:
        """Prepare prompt for Gemini API"""
        parts = []

        # System prompt
        if task.context.get('system_prompt'):
            parts.append(f"Instructions: {task.context['system_prompt']}\n")

        # Conversation history
        if task.context.get('history'):
            parts.append("Context:\n")
            for msg in task.context['history']:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    parts.append(f"{role.capitalize()}: {content}\n")

        # Current message
        if isinstance(task.content, str):
            parts.append(f"\nUser: {task.content}")
        else:
            parts.append(f"\nUser: {str(task.content)}")

        return '\n'.join(parts)


# ============================================================================
# Mock Agent for Testing (when real APIs unavailable)
# ============================================================================

class MockAgent(BaseAgent):
    """Mock agent for testing without real API calls"""

    async def _call_api(self, task: Task) -> Dict[str, Any]:
        """Simulate API call"""
        await asyncio.sleep(0.1)  # Simulate network delay

        estimated_tokens = self._estimate_tokens(task)
        input_tokens = estimated_tokens // 2
        output_tokens = estimated_tokens // 2

        result_text = f"Mock response for: {str(task.content)[:100]}"

        tokens_used = TokenUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            estimated=True
        )

        return {
            'result': result_text,
            'tokens_used': tokens_used,
            'metadata': {
                'model': 'mock-model',
                'finish_reason': 'stop',
            }
        }


# ============================================================================
# Agent Factory
# ============================================================================

def create_agent(config: AgentConfig) -> BaseAgent:
    """Factory function to create appropriate agent based on provider"""
    provider = config.provider.lower()

    if provider == 'openai':
        if not OPENAI_AVAILABLE:
            logger.warning("openai_unavailable_using_mock", agent_id=config.agent_id)
            return MockAgent(config)
        return OpenAIAgent(config)

    elif provider == 'anthropic' or provider == 'claude':
        if not ANTHROPIC_AVAILABLE:
            logger.warning("claude_unavailable_using_mock", agent_id=config.agent_id)
            return MockAgent(config)
        return ClaudeAgent(config)

    elif provider == 'google' or provider == 'gemini':
        if not GOOGLE_AVAILABLE:
            logger.warning("gemini_unavailable_using_mock", agent_id=config.agent_id)
            return MockAgent(config)
        return GeminiAgent(config)

    else:
        logger.warning("unknown_provider_using_mock", provider=provider, agent_id=config.agent_id)
        return MockAgent(config)
