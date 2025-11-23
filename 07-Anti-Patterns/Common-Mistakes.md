# Common Anti-Patterns in Multi-Agent Systems

Learn from these mistakes so you don't have to make them yourself.

---

## 1. The "God Agent" Anti-Pattern

### ❌ What It Is

Creating one mega-agent that does everything.

```yaml
# BAD
agent:
  name: "super-agent"
  capabilities:
    - research
    - writing
    - coding
    - data_analysis
    - email_sending
    - database_management
    - customer_support
    - content_moderation
    # ... 20 more capabilities
```

### Why It's Bad

- Impossible to test thoroughly
- Hard to maintain
- Unclear responsibilities
- Can't optimize for specific tasks
- Failure in one area affects everything
- Expensive (uses most capable model for everything)

### ✅ Solution: Single Responsibility

```yaml
# GOOD
research_agent:
  capabilities: [research, summarization]

writing_agent:
  capabilities: [content_creation, editing]

code_agent:
  capabilities: [code_generation, code_review]
```

**Compose agents in workflows instead of creating mega-agents.**

---

## 2. Hard-Coded Credentials

### ❌ What It Is

Embedding API keys and secrets in agent definitions.

```python
# BAD
agent_config = {
    "model": "gpt-4",
    "api_key": "sk-abc123...",  # NEVER DO THIS
    "database_password": "mypassword123"  # NEVER DO THIS
}
```

### Why It's Bad

- Security risk (credentials in version control)
- Can't rotate credentials easily
- Different keys for dev/staging/prod
- Compliance violations

### ✅ Solution: Environment Variables + Secrets Management

```python
# GOOD
import os
from orchamesh import SecretsManager

secrets = SecretsManager()

agent_config = {
    "model": "gpt-4",
    "api_key": secrets.get("OPENAI_API_KEY"),
    "database_password": secrets.get("DB_PASSWORD")
}
```

**Use OrchaMesh's secrets management or environment variables.**

---

## 3. Ignoring Rate Limits

### ❌ What It Is

Making requests without respecting API rate limits.

```python
# BAD
for task in large_task_list:  # 10,000 tasks
    result = await agent.execute(task)  # Will hit rate limit!
```

### Why It's Bad

- API errors and failures
- Account suspension
- Wasted money on failed requests
- Unpredictable behavior

### ✅ Solution: Rate Limiting + Batching

```python
# GOOD
from orchamesh import RateLimiter

limiter = RateLimiter(
    requests_per_minute=50,
    burst_size=10
)

async with limiter:
    for task in large_task_list:
        await limiter.acquire()
        result = await agent.execute(task)
```

**Always implement rate limiting, especially for production.**

---

## 4. No Cost Tracking

### ❌ What It Is

Running agents without tracking costs.

```python
# BAD
while True:
    result = await expensive_agent.execute(user_input)
    # No idea how much this costs!
```

### Why It's Bad

- Surprise bills
- Can't optimize costs
- No budget control
- Can't justify ROI

### ✅ Solution: Cost Monitoring + Budgets

```python
# GOOD
agent_config = {
    "budget": {
        "max_cost_per_request": 0.50,
        "max_cost_per_hour": 100.00,
        "alert_threshold": 0.80
    }
}

result = await agent.execute(task)
print(f"Cost: ${result.metadata.cost:.4f}")
print(f"Budget remaining: ${agent.budget_remaining:.2f}")
```

**Track every dollar. Set budgets. Get alerts.**

---

## 5. Missing Error Handling

### ❌ What It Is

No error handling or generic catch-all errors.

```python
# BAD
try:
    result = await agent.execute(task)
except Exception as e:
    print("Something went wrong")  # Not helpful!
    return None
```

### Why It's Bad

- Can't diagnose problems
- No recovery strategy
- Poor user experience
- Lost context

### ✅ Solution: Specific Error Handling

```python
# GOOD
from orchamesh.exceptions import (
    RateLimitError,
    TimeoutError,
    ModelError,
    AuthenticationError
)

try:
    result = await agent.execute(task)
except RateLimitError as e:
    # Wait and retry
    await asyncio.sleep(e.retry_after)
    result = await agent.execute(task)
except TimeoutError:
    # Return partial results
    result = agent.get_partial_results()
except AuthenticationError:
    # Alert ops team
    alert_team("Agent authentication failed")
    raise
except ModelError as e:
    # Log and use fallback
    logger.error(f"Model error: {e}")
    result = await fallback_agent.execute(task)
```

**Handle specific errors with specific strategies.**

---

## 6. No Observability

### ❌ What It Is

Running agents in production without logs, metrics, or traces.

```python
# BAD
agent.execute(task)  # What happened? No idea!
```

### Why It's Bad

- Can't debug production issues
- No visibility into costs
- Can't optimize performance
- Compliance problems

### ✅ Solution: Comprehensive Observability

```python
# GOOD
from orchamesh import Observability

obs = Observability(
    logging=True,
    metrics=True,
    tracing=True
)

with obs.trace("agent-execution"):
    result = await agent.execute(task)

    # Automatic logging
    # Automatic metrics
    # Automatic distributed tracing
```

**Instrument everything from day one.**

---

## 7. Stateless Agents (When You Need State)

### ❌ What It Is

Not maintaining context across interactions.

```python
# BAD
# User: "Research quantum computing"
result1 = agent.execute("Research quantum computing")

# User: "Now summarize it" (Agent has no context!)
result2 = agent.execute("Now summarize it")  # Summarize what?
```

### Why It's Bad

- Poor user experience
- Wasted API calls
- Can't have conversations
- Lost context

### ✅ Solution: Session Management

```python
# GOOD
session = await orchamesh.sessions.create({
    "user_id": user.id,
    "context_window": 10  # Keep last 10 interactions
})

result1 = await agent.execute(
    task="Research quantum computing",
    session_id=session.id
)

result2 = await agent.execute(
    task="Now summarize it",  # Agent has context from result1
    session_id=session.id
)
```

**Use OrchaMesh session management for stateful interactions.**

---

## 8. Synchronous Orchestration

### ❌ What It Is

Running independent agents sequentially instead of in parallel.

```python
# BAD - Takes 30 seconds (10s + 10s + 10s)
result1 = await agent1.execute(task)  # 10 seconds
result2 = await agent2.execute(task)  # 10 seconds
result3 = await agent3.execute(task)  # 10 seconds
```

### Why It's Bad

- Slow
- Wastes time
- Poor user experience
- Underutilized resources

### ✅ Solution: Parallel Execution

```python
# GOOD - Takes 10 seconds (all in parallel)
import asyncio

results = await asyncio.gather(
    agent1.execute(task),
    agent2.execute(task),
    agent3.execute(task)
)
```

**Run independent operations in parallel.**

---

## 9. No Versioning

### ❌ What It Is

Modifying agents without version control.

```python
# BAD
agent.instructions = "New instructions..."  # Overwrites old version
await agent.update()  # Can't rollback!
```

### Why It's Bad

- Can't rollback
- No change history
- Can't A/B test
- Compliance issues

### ✅ Solution: Semantic Versioning

```python
# GOOD
new_version = await agent.create_version({
    "version": "2.0.0",
    "instructions": "New instructions...",
    "changelog": "Improved accuracy by 15%"
})

# Test new version
results = await test_agent(new_version)

if results.success_rate > 0.95:
    await new_version.promote_to_production()
else:
    # Keep old version
    pass
```

**Always version agents. Never modify production directly.**

---

## 10. Missing HITL for Critical Decisions

### ❌ What It Is

Letting agents make important decisions without human oversight.

```python
# BAD
# Agent automatically executes financial transactions
transaction = agent.execute("Transfer $10,000 to vendor")
# No human approval!
```

### Why It's Bad

- Risk of mistakes
- Compliance violations
- No accountability
- Trust issues

### ✅ Solution: Human-in-the-Loop

```python
# GOOD
workflow = orchamesh.workflow.create({
    "steps": [
        {"agent": "analyze_transaction"},
        {"hitl": "requires_approval", "reviewers": ["finance-team"]},
        {"agent": "execute_transaction", "condition": "approved"}
    ]
})

# Human must approve before execution
result = await workflow.execute(transaction)
```

**Critical decisions require human approval.**

---

## 11. No Testing Strategy

### ❌ What It Is

Deploying agents without testing.

```python
# BAD
agent = create_agent(config)
# Deploy to production immediately!
await deploy_to_production(agent)
```

### Why It's Bad

- Unknown behavior
- Will break in production
- No confidence
- Costly mistakes

### ✅ Solution: Comprehensive Testing

```python
# GOOD
agent = create_agent(config)

# Unit tests
unit_tests = await test_runner.run_unit_tests(agent)
assert unit_tests.passed

# Integration tests
integration_tests = await test_runner.run_integration_tests(agent)
assert integration_tests.passed

# Load tests
load_tests = await test_runner.run_load_tests(agent, requests_per_second=100)
assert load_tests.p95_latency < 2.0

# Only then deploy
await deploy_to_production(agent)
```

**Test thoroughly before deploying.**

---

## 12. Ignoring Token Limits

### ❌ What It Is

Not checking context window limits.

```python
# BAD
huge_document = load_document("10MB.txt")
result = await agent.execute(huge_document)  # Exceeds token limit!
```

### Why It's Bad

- API errors
- Wasted money
- Truncated inputs
- Incorrect results

### ✅ Solution: Token Management

```python
# GOOD
from orchamesh import TokenCounter

counter = TokenCounter(model="gpt-4")
tokens = counter.count(huge_document)

if tokens > agent.max_tokens:
    # Chunk the document
    chunks = chunk_document(huge_document, max_tokens=agent.max_tokens)
    results = [await agent.execute(chunk) for chunk in chunks]
    result = aggregate_results(results)
else:
    result = await agent.execute(huge_document)
```

**Always check token limits before calling APIs.**

---

## Summary: The 12 Commandments

1. ✅ **One agent, one responsibility**
2. ✅ **Never hard-code credentials**
3. ✅ **Respect rate limits**
4. ✅ **Track every dollar**
5. ✅ **Handle errors specifically**
6. ✅ **Observe everything**
7. ✅ **Maintain context when needed**
8. ✅ **Parallelize independent operations**
9. ✅ **Version all changes**
10. ✅ **Require approval for critical decisions**
11. ✅ **Test before deploying**
12. ✅ **Check token limits**

---

## Further Reading

- [Best Practices: Agent Design](../02-Best-Practices/Agent-Design-Principles.md)
- [Testing Strategies](../08-Testing-Quality/Testing-Strategies.md)
- [Security Patterns](../02-Best-Practices/Security-And-Governance.md)

---

**Avoid these anti-patterns and your agents will be reliable, maintainable, and cost-effective!** 🎯
