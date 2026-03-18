# Agent Harness

**A complete, extensible Python framework for building AI agents and multi-agent systems.**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen)

Agent Harness provides production-ready primitives for single-agent workflows, multi-agent orchestration, tool use, memory management, and LLM provider abstraction — all in a clean, async-first API.

---

## Features

### 🤖 Agent Types
- **ReActAgent** — Reason + Act loop with automatic tool invocation
- **PlanAgent** — Plan → Execute → Synthesize with dynamic replanning
- **ConversationalAgent** — Single-turn LLM calls for chat, summarization, and analysis

### 🔧 Tool System
- **`@tool` decorator** — Auto-generates JSON Schema from type hints and docstrings
- **ToolRegistry & ToolExecutor** — Concurrent execution with timeout and error handling
- **Built-in tools** — File I/O, HTTP requests, Python execution, directory listing

### 💭 Memory
- **ShortTermMemory** — Sliding-window or token-limited conversation buffer
- **WorkingMemory** — Key-value scratchpad for intermediate reasoning state
- **LongTermMemory** — Semantic vector search with pluggable embedding functions

### 🔀 Orchestration
- **Pipeline** — Sequential agent chains with conditional steps and input transforms
- **DAGOrchestrator** — Parallel execution graph with dependency resolution and cycle detection
- **AgentRouter** — Intent-based routing via callables or regex patterns
- **AgentTeam** — Multi-agent collaboration (supervisor, debate, round-robin modes)

### 📡 LLM Providers
- **OpenAIProvider** — GPT models with streaming support
- **AnthropicProvider** — Claude models with native tool-use format
- **RetryableLLM** — Exponential backoff with configurable retry policies
- **FallbackChain** — Automatic provider failover
- **RateLimiter** — Token-bucket rate limiting

### 🔍 Observability
- **EventBus** — Wildcard event subscriptions across all components
- **Tracer & Span** — Structured tracing with parent-child span relationships
- **Exporters** — Console (colored) and JSON Lines file export

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Orchestration    Pipeline │ DAG │ Router │ Team            │
├─────────────────────────────────────────────────────────────┤
│  Agent            ReActAgent │ PlanAgent │ ConversationalAgent
├─────────────────────────────────────────────────────────────┤
│  Context          AgentContext │ StateManager │ Variables    │
├─────────────────────────────────────────────────────────────┤
│  Memory           ShortTerm │ Working │ LongTerm            │
├──────────────────────────┬──────────────────────────────────┤
│  Tool                    │  LLM                             │
│  @tool │ Registry │      │  OpenAI │ Anthropic │            │
│  Executor │ Builtins     │  Retry │ Fallback │ RateLimiter  │
├──────────────────────────┴──────────────────────────────────┤
│  Core             Message │ Event │ Config │ Errors         │
├─────────────────────────────────────────────────────────────┤
│  Tracing          Tracer │ Span │ Console/JSON Exporters    │
└─────────────────────────────────────────────────────────────┘
```

| Layer | Purpose |
|---|---|
| **Orchestration** | Compose agents into pipelines, DAGs, routers, and collaborative teams. |
| **Agent** | Reasoning strategies — ReAct loops, plan-and-execute, or direct conversation. |
| **Context** | Execution state, scoped variables, and state machine transitions per agent. |
| **Memory** | Conversation history, working scratchpad, and semantic long-term retrieval. |
| **Tool** | Declarative tool definitions with concurrent, sandboxed execution. |
| **LLM** | Provider abstraction with retry, rate-limiting, and fallback chains. |
| **Core** | Shared primitives — messages, events, configuration, typed errors. |
| **Tracing** | Structured spans and exporters for debugging and monitoring agent runs. |

---

## Quick Start

### Installation

```bash
pip install -e ".[dev]"
```

### Minimal Example

```python
import asyncio
from agent_harness import ReActAgent, tool, HarnessConfig
from agent_harness.llm import LLM

@tool
async def search_web(query: str) -> str:
    """Search the web for information."""
    return f"Top result for '{query}': Python is a versatile programming language."

async def main():
    config = HarnessConfig.load("config.yaml")
    agent = ReActAgent(name="researcher", llm=LLM(config), tools=[search_web], config=config)

    result = await agent.run("What is Python?")
    print(result.output)

asyncio.run(main())
```

Set your API key and run:

```bash
export OPENAI_API_KEY="sk-..."
python example.py
```

---

## Core Concepts

### Agents

Three built-in agent types cover the most common reasoning strategies:

```python
from agent_harness import ReActAgent, PlanAgent, ConversationalAgent

# ReAct: Think → Act → Observe loop
react = ReActAgent(name="react", llm=llm, tools=[search_web])

# Plan-and-Execute: Plan → Execute each step → Synthesize
planner = PlanAgent(name="planner", llm=llm, tools=[search_web], allow_replan=True)

# Conversational: Single LLM call, no tools
chat = ConversationalAgent(name="chat", llm=llm)
```

All agents inherit from `BaseAgent` and share a common interface: `await agent.run(input) -> AgentResult`.

### Tools

Define tools with the `@tool` decorator. Type hints and docstrings are automatically converted to JSON Schema for LLM function calling:

```python
from agent_harness import tool

@tool(name="calculator", description="Evaluate a math expression")
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A valid Python math expression.
    """
    return str(eval(expression))
```

Both sync and async functions are supported. Parameters support `str`, `int`, `float`, `bool`, `list`, and `dict` types.

### LLM Providers

Swap providers without changing agent code:

```python
from agent_harness import HarnessConfig
from agent_harness.llm import LLM

config = HarnessConfig.load("config.yaml")
llm = LLM(config)  # provider auto-resolved from config.llm.provider
```

Add resilience with retry and fallback:

```python
from agent_harness.llm.retry import RetryableLLM, RetryPolicy, FallbackChain

resilient = RetryableLLM(openai_llm, policy=RetryPolicy(max_retries=3))
fallback = FallbackChain(providers=[openai_llm, anthropic_llm])
```

### Memory

Three memory types serve different purposes:

```python
from agent_harness.memory import ShortTermMemory, WorkingMemory

# Conversation buffer (sliding window or token-limited)
short_term = ShortTermMemory(max_messages=50, strategy="sliding_window")

# Key-value scratchpad for reasoning state
working = WorkingMemory()
working.set("plan", "Step 1: Search → Step 2: Analyze → Step 3: Report")
working.set("current_step", "Step 1")
print(working.to_prompt_string())
```

For semantic search, use `LongTermMemory` with a vector store and embedding function:

```python
from agent_harness.memory import LongTermMemory
from agent_harness.memory.storage.numpy import NumpyVectorStore

ltm = LongTermMemory(store=NumpyVectorStore(), embedding_fn=embed)
await ltm.add("Important finding", metadata={"source": "paper1"})
results = await ltm.query("related topic", top_k=5)
```

### Orchestration

Compose agents into complex workflows:

```python
from agent_harness.orchestration import Pipeline, PipelineStep

pipeline = Pipeline(steps=[
    PipelineStep(agent=researcher),
    PipelineStep(
        agent=writer,
        transform=lambda x: f"Write an article based on: {x}",
    ),
    PipelineStep(
        agent=editor,
        condition=lambda x: len(x) > 100,  # skip if output is short
    ),
])
result = await pipeline.run("Latest advances in AI safety")
```

Other orchestration patterns:

```python
from agent_harness.orchestration import DAGOrchestrator, DAGNode, AgentRouter, Route, AgentTeam

# Parallel DAG — nodes without dependencies run concurrently
dag = DAGOrchestrator(nodes=[
    DAGNode(id="search", agent=searcher),
    DAGNode(id="analyze", agent=analyzer, dependencies=["search"]),
    DAGNode(id="report", agent=reporter, dependencies=["analyze"]),
])

# Intent-based routing
router = AgentRouter(
    routes=[
        Route(agent=coder, condition=r"code|program|function"),
        Route(agent=researcher, condition=lambda x: "search" in x.lower()),
    ],
    fallback=general_agent,
)

# Multi-agent team (supervisor, debate, or round-robin)
team = AgentTeam(
    agents=[researcher, analyst, writer],
    mode="supervisor",
)
```

### Context

`AgentContext` bundles memory, state, variables, and events. Use `fork()` to create child contexts for sub-agents that share global state but maintain independent execution:

```python
from agent_harness.context import AgentContext

parent_ctx = AgentContext.create(config=config)
parent_ctx.variables.set("task", "research AI safety", scope=Scope.GLOBAL)

# Child shares long-term memory, global variables, and event bus
# but gets fresh short-term memory, working memory, and state
child_ctx = parent_ctx.fork(name="sub-agent")
```

---

## Examples

| File | Description |
|---|---|
| `react_agent.py` | Basic ReAct agent with custom tools and event logging |
| `plan_and_execute.py` | PlanAgent that breaks down complex research tasks |
| `multi_agent_pipeline.py` | Sequential pipeline: researcher → writer → editor |
| `agent_team.py` | Supervisor-mode team collaboration with shared context |
| `deep_research.py` | DAG orchestration for parallel research and synthesis |

Run any example:

```bash
export OPENAI_API_KEY="sk-..."
python examples/react_agent.py
```

---

## Configuration

### YAML Config

Create a `config.yaml` to configure all components declaratively:

```yaml
llm:
  provider: openai
  model: gpt-4o
  temperature: 0.7
  max_tokens: 4096
  timeout: 120.0

tool:
  max_concurrency: 5
  default_timeout: 30.0
  sandbox_enabled: false

memory:
  short_term_max_messages: 50
  short_term_max_tokens: 8000
  short_term_strategy: sliding_window

tracing:
  enabled: true
  exporter: json_file
  export_path: ./traces

verbose: false
```

Load it in code:

```python
config = HarnessConfig.load("config.yaml")
```

### Environment Variables

All settings can be overridden via environment variables:

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `HARNESS_LLM_PROVIDER` | LLM provider (`openai`, `anthropic`) | `openai` |
| `HARNESS_LLM_MODEL` | Model name | `gpt-4o` |
| `HARNESS_LLM_TEMPERATURE` | Sampling temperature | `0.7` |
| `HARNESS_LLM_MAX_TOKENS` | Max output tokens | `4096` |
| `HARNESS_VERBOSE` | Enable verbose logging | `false` |
| `HARNESS_TRACING_ENABLED` | Enable tracing | `true` |

Environment variables take precedence when using `HarnessConfig.from_env()`. Merge configs with `config.merge(other)`.

---

## Project Structure

```
agent_harness/
├── src/agent_harness/
│   ├── agent/            # BaseAgent, ReActAgent, PlanAgent, ConversationalAgent
│   ├── context/          # AgentContext, StateManager, ContextVariables
│   ├── core/             # Message, Event, Config, Errors, Registry
│   ├── llm/              # OpenAI/Anthropic providers, retry, fallback
│   ├── memory/           # ShortTerm, Working, LongTerm, vector storage
│   ├── orchestration/    # Pipeline, DAG, Router, Team
│   ├── prompt/           # PromptTemplate, PromptBuilder, PromptLibrary
│   ├── tool/             # @tool decorator, Registry, Executor, builtins
│   ├── tracing/          # Tracer, Span, Console/JSON exporters
│   └── utils/            # Shared utilities
├── tests/                # Test suite
├── examples/             # Usage examples
├── pyproject.toml        # Package configuration
└── environment.yml       # Conda environment
```

---

## Testing

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Lint and type-check:

```bash
ruff check src/
mypy src/
```

---

## License

MIT
