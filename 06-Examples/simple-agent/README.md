# Simple Agent Example

Complete example of creating and using an agent with OrchaMesh.

---

## Overview

This example demonstrates:
- Creating an agent using OrchaMesh Agent Studio
- Defining clear input/output contracts
- Implementing error handling
- Adding observability
- Testing the agent

---

## Prerequisites

- OrchaMesh account (Individual or Enterprise plan)
- Node.js 20+ or Python 3.9+
- API key for AI provider (OpenAI, Anthropic, or Google)

---

## Agent Specification

**Purpose**: Summarize articles and extract key points

**Input**: Article URL or text
**Output**: Summary with key points

---

## Implementation (TypeScript)

```typescript
// simple-agent.ts
import { OrchaMeshClient } from '@orchamesh/client';

// Initialize OrchaMesh client
const client = new OrchaMeshClient({
  apiKey: process.env.ORCHAMESH_API_KEY!,
  endpoint: 'https://api.orchamesh.com'
});

// Define agent
async function createAgent() {
  const agent = await client.agents.create({
    name: 'article-summarizer',
    version: '1.0.0',
    description: 'Summarizes articles and extracts key points',

    // Model configuration
    model: {
      provider: 'openai',
      model: 'gpt-4',
      temperature: 0.3,  // Lower for more consistent summaries
      max_tokens: 2000
    },

    // Capabilities
    capabilities: [
      'text_summarization',
      'key_point_extraction'
    ],

    // System instructions
    instructions: `
      You are an expert at summarizing articles. When given an article:

      1. Read the entire article carefully
      2. Identify the main topic and thesis
      3. Extract 3-5 key points
      4. Write a concise summary (2-3 paragraphs)

      Format your response as:
      {
        "summary": "...",
        "key_points": ["point 1", "point 2", ...],
        "main_topic": "...",
        "word_count": number
      }
    `,

    // Input schema
    input_schema: {
      type: 'object',
      required: ['content'],
      properties: {
        content: {
          type: 'string',
          description: 'Article text to summarize'
        },
        max_length: {
          type: 'integer',
          default: 200,
          description: 'Maximum words in summary'
        }
      }
    },

    // Output schema
    output_schema: {
      type: 'object',
      properties: {
        summary: { type: 'string' },
        key_points: {
          type: 'array',
          items: { type: 'string' }
        },
        main_topic: { type: 'string' },
        word_count: { type: 'integer' }
      }
    },

    // Constraints
    constraints: {
      max_cost_per_request: 0.10,  // $0.10
      timeout_seconds: 30
    },

    // Tags
    tags: ['summarization', 'content', 'production']
  });

  console.log(`Agent created: ${agent.id}`);
  return agent;
}

// Execute agent
async function executeAgent(agentId: string, article: string) {
  try {
    const result = await client.agents.execute({
      agentId,
      input: {
        content: article,
        max_length: 200
      },
      // Observability
      metadata: {
        source: 'api',
        user_id: 'user-123'
      }
    });

    if (result.success) {
      console.log('Summary:', result.output.summary);
      console.log('Key Points:', result.output.key_points);
      console.log('Cost:', `$${result.metadata.cost.toFixed(4)}`);
      console.log('Tokens:', result.metadata.tokens_used);
    } else {
      console.error('Error:', result.error);
    }

    return result;

  } catch (error) {
    if (error instanceof RateLimitError) {
      console.log('Rate limit hit, waiting...');
      await sleep(error.retry_after_seconds * 1000);
      return executeAgent(agentId, article);  // Retry
    }

    if (error instanceof TimeoutError) {
      console.log('Timeout, returning partial results');
      return error.partial_results;
    }

    throw error;  // Re-throw unknown errors
  }
}

// Test the agent
async function testAgent(agentId: string) {
  const test_article = `
    Artificial Intelligence has made significant progress in 2024.
    New models like GPT-4 and Claude 3 have demonstrated remarkable
    capabilities in reasoning, coding, and multimodal understanding.

    Key developments include improved context windows, better reasoning
    capabilities, and more reliable factual accuracy. However, challenges
    remain in areas like hallucination reduction and cost optimization.

    Enterprises are increasingly adopting AI agents for automation,
    customer service, and content creation. The market is expected to
    grow significantly over the next five years.
  `;

  console.log('Testing agent...');
  const result = await executeAgent(agentId, test_article);

  // Validate output
  assert(result.success, 'Agent should succeed');
  assert(result.output.key_points.length >= 3, 'Should have at least 3 key points');
  assert(result.output.summary.length > 50, 'Summary should be substantial');

  console.log('✅ All tests passed!');
}

// Main function
async function main() {
  // Create agent
  const agent = await createAgent();

  // Test agent
  await testAgent(agent.id);

  // Promote to production
  await client.agents.promote({
    agentId: agent.id,
    environment: 'production',
    approval: {
      approver: 'tech-lead@company.com',
      reason: 'Tests passed successfully'
    }
  });

  console.log('Agent deployed to production! 🚀');
}

// Run
main().catch(console.error);

// Helper function
function sleep(ms: number) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function assert(condition: boolean, message: string) {
  if (!condition) {
    throw new Error(`Assertion failed: ${message}`);
  }
}
```

---

## Implementation (Python)

```python
# simple_agent.py
import os
import asyncio
from orchamesh import OrchaMeshClient
from orchamesh.exceptions import RateLimitError, TimeoutError

# Initialize client
client = OrchaMeshClient(
    api_key=os.getenv('ORCHAMESH_API_KEY'),
    endpoint='https://api.orchamesh.com'
)

async def create_agent():
    """Create the summarization agent"""
    agent = await client.agents.create({
        'name': 'article-summarizer',
        'version': '1.0.0',
        'description': 'Summarizes articles and extracts key points',

        'model': {
            'provider': 'openai',
            'model': 'gpt-4',
            'temperature': 0.3,
            'max_tokens': 2000
        },

        'capabilities': [
            'text_summarization',
            'key_point_extraction'
        ],

        'instructions': """
            You are an expert at summarizing articles. When given an article:
            1. Read the entire article carefully
            2. Identify the main topic and thesis
            3. Extract 3-5 key points
            4. Write a concise summary (2-3 paragraphs)
        """,

        'constraints': {
            'max_cost_per_request': 0.10,
            'timeout_seconds': 30
        },

        'tags': ['summarization', 'content', 'production']
    })

    print(f"Agent created: {agent.id}")
    return agent

async def execute_agent(agent_id: str, article: str):
    """Execute the agent"""
    try:
        result = await client.agents.execute(
            agent_id=agent_id,
            input={
                'content': article,
                'max_length': 200
            }
        )

        if result.success:
            print('Summary:', result.output['summary'])
            print('Key Points:', result.output['key_points'])
            print(f"Cost: ${result.metadata.cost:.4f}")

        return result

    except RateLimitError as e:
        print(f'Rate limit hit, waiting {e.retry_after_seconds}s...')
        await asyncio.sleep(e.retry_after_seconds)
        return await execute_agent(agent_id, article)

    except TimeoutError as e:
        print('Timeout, returning partial results')
        return e.partial_results

async def main():
    # Create agent
    agent = await create_agent()

    # Test article
    article = """
        Artificial Intelligence has made significant progress in 2024...
    """

    # Execute
    result = await execute_agent(agent.id, article)

    # Validate
    assert result.success
    assert len(result.output['key_points']) >= 3

    print('✅ All tests passed!')

# Run
if __name__ == '__main__':
    asyncio.run(main())
```

---

## Running the Example

### TypeScript

```bash
# Install dependencies
npm install @orchamesh/client

# Set API key
export ORCHAMESH_API_KEY="your-key-here"

# Run
npm run ts-node simple-agent.ts
```

### Python

```bash
# Install dependencies
pip install orchamesh-client

# Set API key
export ORCHAMESH_API_KEY="your-key-here"

# Run
python simple_agent.py
```

---

## Expected Output

```
Agent created: agent-abc123
Testing agent...
Summary: Artificial Intelligence made significant advances in 2024...
Key Points: ['GPT-4 and Claude 3 show remarkable capabilities', 'Challenges remain in hallucination reduction', 'Enterprise adoption is increasing']
Cost: $0.0234
Tokens: 1567
✅ All tests passed!
Agent deployed to production! 🚀
```

---

## Key Takeaways

1. ✅ **Clear Purpose**: Agent has one clear job
2. ✅ **Defined Contracts**: Input/output schemas specified
3. ✅ **Error Handling**: Handles rate limits and timeouts
4. ✅ **Cost Control**: Budget constraints set
5. ✅ **Observability**: Logs cost and tokens
6. ✅ **Testing**: Validates output before production
7. ✅ **Versioning**: Semantic versioning used

---

## Next Steps

1. Try modifying the agent instructions
2. Add HITL approval workflow
3. Create a workflow with multiple agents
4. Implement custom connectors
5. Add comprehensive test suite

---

## Resources

- [Agent Design Principles](../../02-Best-Practices/Agent-Design-Principles.md)
- [OrchaMesh Integration](../../04-OrchaMesh-Integration/Getting-Started.md)
- [Testing Strategies](../../08-Testing-Quality/Testing-Strategies.md)

---

**This is the foundation for all agent development!** 🚀
