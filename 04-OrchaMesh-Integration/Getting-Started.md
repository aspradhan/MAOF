# Getting Started with OrchaMesh

Complete guide to integrating with OrchaMesh - The Enterprise AI/ML Agent Control Fabric

---

## What is OrchaMesh?

OrchaMesh is an **Enterprise AI/ML Agent Control Fabric** that provides:

- **Agent Studio**: Full-lifecycle agent management (CRUD, versioning, publishing)
- **HITL Control Room**: Human-in-the-loop collaboration workflows
- **Orchestration Engine**: DAG-based workflow management
- **Multi-Agent Coordination**: Collaborative agent sessions
- **Model Abstraction**: Provider-agnostic AI model routing
- **Policy & Governance**: RBAC/ABAC, data policies, compliance
- **Observability**: Comprehensive dashboards and analytics
- **Marketplace**: Pre-built templates and integrations

**Repository**: https://github.com/aspradhan/AIAgentOS

---

## Prerequisites

Before integrating with OrchaMesh:

### 1. OrchaMesh Account

Choose a subscription plan:

| Plan | Price | Best For |
|------|-------|----------|
| **Free Trial** | 8 days free | Evaluation |
| **Individual** | $100/month | Small teams (1-10 users) |
| **Enterprise** | $250/user/month | Large teams (10+ users) |

**Sign up**: Visit OrchaMesh platform and create account

### 2. Environment Setup

```bash
# Required
Node.js 20.x+
PostgreSQL 16+
npm or yarn

# Optional (for local development)
Docker
Redis (for caching)
```

### 3. API Keys

Obtain API keys for your AI providers:

```bash
# OpenAI
export OPENAI_API_KEY="sk-..."

# Anthropic
export ANTHROPIC_API_KEY="sk-ant-..."

# Google
export GOOGLE_API_KEY="..."
```

---

## Installation

### Option 1: Using OrchaMesh Hosted

**Simplest option** - No infrastructure needed.

1. **Sign up** at OrchaMesh platform
2. **Configure** your organization
3. **Invite** team members
4. **Start building** agents

### Option 2: Self-Hosted Deployment

**For enterprises with specific requirements.**

```bash
# Clone OrchaMesh
git clone https://github.com/aspradhan/AIAgentOS.git
cd AIAgentOS

# Install dependencies
npm install
cd ai-agent-os/apps/ui && npm install
cd ../server && npm install

# Configure environment
cp .env.example .env
# Edit .env with your settings

# Initialize database
npm run db:push

# Run development
npm run dev

# Or production
npm run build
npm run start
```

**Access:**
- Frontend: http://localhost:5000
- API: http://localhost:3000

---

## Quick Start: Your First Agent

### Step 1: Create Agent via Agent Studio

```typescript
// Using OrchaMesh API
import { OrchaMeshClient } from '@orchamesh/client';

const client = new OrchaMeshClient({
  apiKey: process.env.ORCHAMESH_API_KEY,
  endpoint: 'https://api.orchamesh.com'
});

// Define agent
const agent = await client.agents.create({
  name: 'research-assistant',
  description: 'AI assistant that researches topics and generates summaries',
  version: '1.0.0',
  model: {
    provider: 'openai',
    model: 'gpt-4',
    temperature: 0.7,
    max_tokens: 4000
  },
  capabilities: [
    'web_search',
    'document_analysis',
    'summarization'
  ],
  instructions: `
    You are a research assistant. When given a topic:
    1. Search for relevant information
    2. Analyze credibility of sources
    3. Generate comprehensive summary
    4. Cite all sources
  `,
  tags: ['research', 'analysis', 'production']
});

console.log(`Agent created: ${agent.id}`);
```

### Step 2: Test Agent

```typescript
// Execute agent task
const result = await client.agents.execute({
  agentId: agent.id,
  input: {
    task: 'Research the impact of AI on healthcare in 2024',
    constraints: {
      max_sources: 10,
      max_cost: 0.50
    }
  }
});

console.log('Result:', result.output);
console.log('Cost:', result.metadata.cost);
console.log('Tokens:', result.metadata.tokens_used);
```

### Step 3: Create Workflow

```typescript
// Create multi-step workflow
const workflow = await client.workflows.create({
  name: 'research-to-article',
  description: 'Research topic and write article',
  dag: {
    nodes: [
      {
        id: 'research',
        type: 'agent',
        agentId: agent.id,
        inputs: {
          task: '${workflow.input.topic}'
        }
      },
      {
        id: 'write',
        type: 'agent',
        agentId: 'writer-agent-id',
        inputs: {
          findings: '${research.output.summary}'
        }
      },
      {
        id: 'review',
        type: 'hitl',
        reviewers: ['team-lead@company.com'],
        approval_required: true
      }
    ],
    edges: [
      { from: 'research', to: 'write' },
      { from: 'write', to: 'review' }
    ]
  }
});
```

---

## OrchaMesh Agent Studio

The Agent Studio is your primary interface for agent management.

### Creating Agents

**Via UI:**
1. Navigate to Agent Studio
2. Click "Create New Agent"
3. Fill in agent details
4. Configure model settings
5. Define capabilities
6. Add instructions
7. Test and publish

**Via API:**
```typescript
const agent = await client.agents.create({
  // Agent definition
});
```

### Agent Versioning

OrchaMesh supports semantic versioning:

```typescript
// Create new version
const v2 = await client.agents.version({
  agentId: agent.id,
  version: '2.0.0',
  changes: {
    model: 'gpt-4-turbo',
    instructions: 'Updated instructions...'
  },
  changelog: 'Upgraded to GPT-4 Turbo for better performance'
});

// Promote to production
await client.agents.promote({
  agentId: agent.id,
  version: '2.0.0',
  environment: 'production',
  approval: {
    approver: 'tech-lead@company.com',
    reason: 'Performance testing passed'
  }
});
```

### Agent Tags

Organize agents with tags:

```typescript
await client.agents.tag({
  agentId: agent.id,
  tags: [
    'production',
    'customer-facing',
    'high-priority',
    'pii-handling'
  ]
});

// Find agents by tag
const productionAgents = await client.agents.find({
  tags: ['production']
});
```

---

## HITL Control Room

Human-in-the-loop workflows for critical decisions.

### Creating Approval Workflow

```typescript
const workflow = await client.workflows.create({
  name: 'content-approval',
  steps: [
    {
      id: 'generate',
      type: 'agent',
      agent: 'content-writer'
    },
    {
      id: 'review',
      type: 'hitl',
      config: {
        reviewers: ['editor@company.com'],
        approval_threshold: 1,
        timeout_hours: 24,
        escalation: {
          after_hours: 48,
          escalate_to: ['manager@company.com']
        }
      }
    },
    {
      id: 'publish',
      type: 'action',
      condition: '${review.approved}'
    }
  ]
});
```

### Handling Feedback

```typescript
// Agent receives feedback from human reviewer
const feedback = await client.hitl.getFeedback({
  workflowId: workflow.id,
  stepId: 'review'
});

if (feedback.status === 'rejected') {
  // Reprocess with corrections
  await client.workflows.retry({
    workflowId: workflow.id,
    fromStep: 'generate',
    context: {
      feedback: feedback.comments,
      corrections: feedback.changes
    }
  });
}
```

---

## Policy as Code

Define governance rules programmatically.

### Data Governance Policy

```yaml
# policy.yaml
policy:
  name: "pii-protection"
  version: "1.0.0"
  rules:
    - name: "mask-pii"
      description: "Automatically mask PII in agent outputs"
      conditions:
        - agent_tags contains "customer-facing"
      actions:
        - type: "mask_pii"
          patterns:
            - email
            - phone
            - ssn
            - credit_card

    - name: "data-retention"
      description: "Delete logs after 90 days"
      conditions:
        - log_type == "agent_execution"
      actions:
        - type: "delete_after_days"
          days: 90

    - name: "cost-limit"
      description: "Alert when agent costs exceed budget"
      conditions:
        - agent_cost > 10.00
      actions:
        - type: "alert"
          channels: ["email", "slack"]
          recipients: ["ops@company.com"]
```

### Applying Policy

```typescript
// Upload policy
await client.policies.create({
  name: 'pii-protection',
  definition: policyYaml,
  scope: {
    agents: ['customer-support-*'],
    environments: ['production']
  }
});

// Apply to agent
await client.agents.applyPolicy({
  agentId: agent.id,
  policyId: policy.id
});
```

---

## Model Abstraction

Route requests to different AI providers.

### Configuring Providers

```typescript
// Configure multiple providers
await client.models.configure({
  providers: [
    {
      name: 'openai',
      apiKey: process.env.OPENAI_API_KEY,
      models: ['gpt-4', 'gpt-3.5-turbo'],
      default: true
    },
    {
      name: 'anthropic',
      apiKey: process.env.ANTHROPIC_API_KEY,
      models: ['claude-3-opus', 'claude-3-sonnet'],
      fallback: true
    },
    {
      name: 'google',
      apiKey: process.env.GOOGLE_API_KEY,
      models: ['gemini-pro']
    }
  ]
});
```

### Routing Policies

```typescript
// Intelligent routing
await client.models.setRoutingPolicy({
  strategy: 'cost-optimized',
  rules: [
    {
      condition: 'task_type == "simple"',
      route_to: 'gpt-3.5-turbo'
    },
    {
      condition: 'task_type == "complex"',
      route_to: 'gpt-4'
    },
    {
      condition: 'tokens > 50000',
      route_to: 'claude-3-opus'  // Large context window
    }
  ],
  fallback: {
    provider: 'anthropic',
    model: 'claude-3-sonnet'
  }
});
```

---

## Observability

Track everything that matters.

### Enabling Observability

```typescript
// Configure observability
await client.observability.configure({
  metrics: {
    enabled: true,
    providers: ['prometheus'],
    export_interval_seconds: 60
  },
  logging: {
    level: 'info',
    destinations: ['console', 'elasticsearch'],
    structured: true
  },
  tracing: {
    enabled: true,
    provider: 'jaeger',
    sample_rate: 0.1  // 10% sampling
  }
});
```

### Viewing Dashboards

OrchaMesh provides built-in dashboards:

- **Agent Performance**: Success rate, latency, cost
- **Cost Analytics**: Spend by agent, model, team
- **Usage Patterns**: Popular agents, peak hours
- **Compliance**: Policy violations, audit trail
- **ROI Metrics**: Value delivered vs cost

**Access**: OrchaMesh UI → Observability → Dashboards

---

## Custom Connectors

Extend OrchaMesh with custom integrations.

### Creating a Connector

```typescript
// Define connector
const connector = await client.connectors.create({
  name: 'salesforce-connector',
  version: '1.0.0',
  description: 'Integration with Salesforce CRM',
  authentication: {
    type: 'oauth2',
    config: {
      client_id: process.env.SALESFORCE_CLIENT_ID,
      client_secret: process.env.SALESFORCE_CLIENT_SECRET,
      token_url: 'https://login.salesforce.com/services/oauth2/token'
    }
  },
  actions: [
    {
      name: 'get_lead',
      description: 'Retrieve lead by ID',
      inputs: {
        lead_id: 'string'
      },
      implementation: `
        async function getLead(leadId) {
          const response = await fetch(
            \`\${baseUrl}/services/data/v58.0/sobjects/Lead/\${leadId}\`,
            { headers: { Authorization: \`Bearer \${token}\` } }
          );
          return response.json();
        }
      `
    }
  ]
});
```

### Using Connector in Agent

```typescript
const agent = await client.agents.create({
  name: 'sales-assistant',
  connectors: ['salesforce-connector'],
  instructions: `
    You can access Salesforce data using:
    - get_lead(lead_id) - Get lead details
    - update_lead(lead_id, data) - Update lead
    - create_task(data) - Create follow-up task
  `
});
```

---

## Best Practices

### 1. Use Agent Studio for All Agents
✅ Version control built-in
✅ Promotion gates
✅ Audit trail

### 2. Implement HITL for Critical Decisions
✅ Human approval for sensitive actions
✅ Feedback loops for improvement
✅ Compliance requirements met

### 3. Apply Policies from Day One
✅ Data governance automated
✅ Cost controls enforced
✅ Security baseline maintained

### 4. Monitor Everything
✅ Real-time dashboards
✅ Cost tracking
✅ Performance metrics

### 5. Use Connectors from Marketplace
✅ Pre-built integrations
✅ Battle-tested code
✅ Maintained by OrchaMesh

---

## Next Steps

1. **Read**: [Agent Studio Integration](./Agent-Studio-Integration.md)
2. **Explore**: [HITL Control Room Usage](./HITL-Control-Room-Usage.md)
3. **Learn**: [Policy as Code Guide](./Policy-As-Code-Guide.md)
4. **Build**: [Custom Connectors](./Custom-Connectors.md)

---

## Support

- **Documentation**: OrchaMesh docs in repository
- **Email**: support@orchamesh.com
- **Enterprise**: abhishekspradhan@gmail.com

---

**You're now ready to build production AI agents with OrchaMesh!** 🚀
