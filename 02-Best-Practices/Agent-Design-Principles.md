# Agent Design Principles

Best practices for designing AI agents in OrchaMesh and multi-agent systems.

---

## Core Principles

### 1. Single Responsibility Principle

**Each agent should have ONE clear, well-defined purpose.**

#### ✅ Good Example
```yaml
name: "research-agent"
purpose: "Search and summarize academic papers on specific topics"
capabilities:
  - search_papers
  - extract_key_findings
  - generate_summaries
```

#### ❌ Bad Example
```yaml
name: "super-agent"
purpose: "Do everything: research, code, write, analyze, manage tasks, send emails..."
# Too many responsibilities!
```

**Why?**
- Easier to test
- Easier to maintain
- Easier to compose
- Clearer failure modes

---

### 2. Composability

**Agents should work together seamlessly through well-defined interfaces.**

#### Design for Composition

```yaml
# Agent 1: Research Agent
inputs:
  - topic: string
outputs:
  - findings: []Finding

# Agent 2: Writer Agent
inputs:
  - findings: []Finding
outputs:
  - article: string

# Agent 3: Editor Agent
inputs:
  - article: string
outputs:
  - edited_article: string
```

**Workflow Composition:**
```
research-agent → writer-agent → editor-agent
```

**Benefits:**
- Reusable agents
- Flexible workflows
- Easy to swap implementations
- Clear data contracts

---

### 3. Explicit Contracts

**Define clear input/output contracts for every agent.**

#### Input Contract

```typescript
interface AgentInput {
  // Required fields
  task: string;
  context?: Record<string, any>;

  // Constraints
  max_tokens?: number;
  max_cost?: number;
  timeout?: number;

  // Policy
  policy?: {
    pii_handling: "mask" | "remove" | "allow";
    data_retention_days: number;
  };
}
```

#### Output Contract

```typescript
interface AgentOutput {
  // Result
  result: any;
  success: boolean;

  // Metadata
  tokens_used: number;
  cost: number;
  processing_time_ms: number;
  model_used: string;

  // Observability
  trace_id: string;
  logs: []LogEntry;

  // Compliance
  pii_detected: boolean;
  policy_violations: []string;
}
```

---

### 4. Observability First

**Every agent must emit comprehensive telemetry.**

#### Required Telemetry

```python
class AgentTelemetry:
    # Execution
    start_time: datetime
    end_time: datetime
    success: bool
    error: Optional[str]

    # Resources
    tokens_input: int
    tokens_output: int
    cost: float
    model: str

    # Context
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]

    # Business
    task_type: str
    priority: str
    user_id: str

    # Compliance
    pii_detected: bool
    policy_applied: str
```

**Why This Matters:**
- Debug production issues
- Track costs accurately
- Meet compliance requirements
- Optimize performance
- Understand user behavior

---

### 5. Graceful Failure

**Agents must handle failures elegantly and provide actionable information.**

#### Failure Handling Pattern

```python
class AgentFailure:
    error_code: str  # Machine-readable
    error_message: str  # Human-readable
    is_retryable: bool
    retry_after_seconds: Optional[int]
    partial_result: Optional[any]
    recommendations: []str

@dataclass
class FailureExample:
    error_code = "RATE_LIMIT_EXCEEDED"
    error_message = "OpenAI rate limit exceeded (60 requests/min)"
    is_retryable = True
    retry_after_seconds = 15
    partial_result = {"progress": "50%", "completed_items": 5}
    recommendations = [
        "Reduce request rate",
        "Upgrade API tier",
        "Use exponential backoff"
    ]
```

#### Failure Categories

| Category | Retryable | Action |
|----------|-----------|--------|
| **Rate Limit** | Yes | Wait and retry |
| **Authentication** | No | Fix credentials |
| **Invalid Input** | No | Correct input |
| **Timeout** | Yes | Retry with longer timeout |
| **Model Error** | Maybe | Log and investigate |
| **Network** | Yes | Exponential backoff |

---

### 6. Cost Awareness

**Every agent must track and respect cost constraints.**

#### Cost Tracking

```python
class CostTracker:
    # Pre-execution
    def estimate_cost(self, input: str) -> float:
        tokens = count_tokens(input)
        return tokens * self.model_cost_per_token

    # Validation
    def check_budget(self, estimated_cost: float) -> bool:
        return self.used_budget + estimated_cost <= self.max_budget

    # Post-execution
    def record_cost(self, actual_cost: float):
        self.used_budget += actual_cost
        self.emit_metric("agent.cost", actual_cost)
```

#### Budget Controls

```yaml
agent:
  name: "research-agent"
  budget:
    max_cost_per_request: 0.50  # $0.50
    max_cost_per_hour: 10.00     # $10/hour
    max_cost_per_day: 100.00     # $100/day
    alert_threshold: 0.80        # Alert at 80%
```

---

### 7. Determinism & Reproducibility

**Agent behavior should be reproducible when possible.**

#### Configuration for Reproducibility

```python
class AgentConfig:
    # Model settings
    model: str = "gpt-4"
    temperature: float = 0.7
    seed: Optional[int] = 42  # For reproducibility

    # Execution
    max_retries: int = 3
    timeout: int = 30

    # Version
    version: str = "1.0.0"
    config_hash: str  # Hash of all settings
```

**Why?**
- Debugging is easier
- Testing is reliable
- Results are explainable
- Compliance audits work

---

### 8. Version Control

**All agents must be versioned and changes tracked.**

#### Versioning Strategy

```yaml
agent:
  id: "research-agent"
  version: "2.1.0"
  changelog:
    - version: "2.1.0"
      date: "2025-01-15"
      changes:
        - "Added citation extraction"
        - "Improved error handling"
      breaking: false

    - version: "2.0.0"
      date: "2025-01-01"
      changes:
        - "Switched to Claude Sonnet"
        - "Changed output format"
      breaking: true
```

**Semantic Versioning:**
- **Major** (2.0.0): Breaking changes
- **Minor** (2.1.0): New features, backward compatible
- **Patch** (2.1.1): Bug fixes

---

### 9. Testing Strategy

**Every agent must have comprehensive tests.**

#### Test Pyramid

```
        /\
       /  \  E2E Tests (5%)
      /    \
     /------\  Integration Tests (15%)
    /        \
   /----------\  Unit Tests (80%)
  /            \
```

#### Test Types

```python
# Unit Tests
def test_agent_input_validation():
    assert agent.validate_input(invalid_input) == False

# Integration Tests
def test_agent_with_real_api():
    result = agent.process(test_input)
    assert result.success == True

# E2E Tests
def test_complete_workflow():
    result = workflow.execute(multi_agent_pipeline)
    assert result.all_agents_succeeded()
```

---

### 10. Security by Default

**Security must be built-in, not bolted on.**

#### Security Checklist

- [ ] **Authentication**: All API calls authenticated
- [ ] **Authorization**: RBAC/ABAC enforced
- [ ] **Encryption**: Secrets encrypted at rest
- [ ] **PII Protection**: PII detected and masked
- [ ] **Input Validation**: All inputs sanitized
- [ ] **Rate Limiting**: Abuse prevention
- [ ] **Audit Logging**: All actions logged
- [ ] **Kill Switch**: Emergency stop mechanism

#### PII Handling Example

```python
class PIIHandler:
    def process_input(self, text: str) -> ProcessedInput:
        pii_detected = self.detect_pii(text)

        if pii_detected:
            if self.policy == "mask":
                text = self.mask_pii(text)
            elif self.policy == "remove":
                text = self.remove_pii(text)
            elif self.policy == "reject":
                raise SecurityError("PII detected, request rejected")

        return ProcessedInput(
            text=text,
            pii_detected=len(pii_detected) > 0,
            pii_types=pii_detected
        )
```

---

## Design Checklist

Before deploying an agent to production, verify:

### Functional Requirements
- [ ] Single, clear responsibility
- [ ] Well-defined input/output contracts
- [ ] Comprehensive error handling
- [ ] Graceful degradation
- [ ] Version controlled

### Non-Functional Requirements
- [ ] Observability instrumented
- [ ] Cost tracking implemented
- [ ] Performance benchmarked
- [ ] Security hardened
- [ ] Documentation complete

### Operational Requirements
- [ ] Monitoring configured
- [ ] Alerts defined
- [ ] Runbook created
- [ ] Rollback plan documented
- [ ] On-call rotation assigned

---

## Common Design Patterns

### 1. Request-Response Pattern
Simple synchronous agent interaction.

### 2. Pub-Sub Pattern
Asynchronous event-driven communication.

### 3. Pipeline Pattern
Sequential processing through multiple agents.

### 4. Scatter-Gather Pattern
Parallel execution with result aggregation.

### 5. Circuit Breaker Pattern
Prevent cascade failures.

### 6. Retry with Backoff Pattern
Handle transient failures.

See [03-Architecture-Patterns/](../03-Architecture-Patterns/) for detailed examples.

---

## Further Reading

- [Multi-Agent Coordination](./Multi-Agent-Coordination.md)
- [Error Handling Strategies](./Error-Handling-Strategies.md)
- [Cost Optimization](./Cost-Optimization.md)
- [Security and Governance](./Security-And-Governance.md)

---

**Remember**: Great agents are simple, focused, observable, and resilient. 🎯
