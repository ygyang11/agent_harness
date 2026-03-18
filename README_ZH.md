# Agent Harness

**一个完整、可扩展的 Python 框架，用于构建 AI Agent 与多 Agent 系统。**

![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
![License MIT](https://img.shields.io/badge/license-MIT-green)
![Tests Passing](https://img.shields.io/badge/tests-passing-brightgreen)

Agent Harness 提供了一组面向生产可用的基础能力，用于构建单 Agent 工作流、多 Agent 编排、工具调用、记忆管理以及 LLM Provider 抽象，并以干净的 async-first API 统一呈现。

---

## 特性

### 🤖 Agent 类型
- **ReActAgent** — 支持自动工具调用的 Reason + Act 循环
- **PlanAgent** — Plan → Execute → Synthesize，并支持动态重规划
- **ConversationalAgent** — 单轮 LLM 调用，适用于聊天、总结与分析

### 🔧 工具体系
- **`@tool` decorator** — 基于类型注解与 docstring 自动生成 JSON Schema
- **ToolRegistry & ToolExecutor** — 支持并发执行、超时控制与错误处理
- **Built-in tools** — 内置文件 I/O、HTTP 请求、Python 执行、目录遍历

### 💭 记忆系统
- **ShortTermMemory** — 滑动窗口或 token 限制的会话缓冲区
- **WorkingMemory** — 用于中间推理状态的键值 scratchpad
- **LongTermMemory** — 结合可插拔 embedding 函数的语义向量检索

### 🔀 编排能力
- **Pipeline** — 支持条件步骤与输入变换的顺序 Agent 链
- **DAGOrchestrator** — 支持依赖解析与环检测的并行执行图
- **AgentRouter** — 基于 callable 或 regex pattern 的意图路由
- **AgentTeam** — 多 Agent 协作模式（supervisor、debate、round-robin）

### 📡 LLM Provider
- **OpenAIProvider** — 支持 GPT系列 与 streaming
- **AnthropicProvider** — 支持 Claude 模型及原生 tool-use 格式
- **RetryableLLM** — 可配置重试策略的指数退避封装
- **FallbackChain** — 自动 Provider 故障切换
- **RateLimiter** — 基于令牌桶的请求限流

### 🔍 可观测性
- **EventBus** — 支持 wildcard 订阅的统一事件系统
- **Tracer & Span** — 带父子层级关系的结构化 tracing
- **Exporters** — 支持 Console（彩色）与 JSON Lines 文件导出

---

## 架构

```text
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

| 层级 | 作用 |
|---|---|
| **Orchestration** | 将 Agent 组织为 pipeline、DAG、router 与协作团队。 |
| **Agent** | 承载推理策略：ReAct 循环、plan-and-execute 或直接对话。 |
| **Context** | 管理每个 Agent 的执行状态、作用域变量与状态机迁移。 |
| **Memory** | 管理会话历史、工作 scratchpad 以及长期语义检索。 |
| **Tool** | 提供声明式工具定义与并发、可控的执行能力。 |
| **LLM** | 提供统一 Provider 抽象，并支持重试、限流与降级链。 |
| **Core** | 提供共享基础原语：message、event、config、typed errors。 |
| **Tracing** | 提供结构化 span 与导出器，用于调试与监控 Agent 运行。 |

---

## 快速开始

### 安装

```bash
pip install -e ".[dev]"
```

### 最小示例

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

设置 API Key 后运行：

```bash
export OPENAI_API_KEY="sk-..."
python example.py
```

---

## 核心概念

### Agents

框架内置了三种 Agent 类型，覆盖最常见的推理策略：

```python
from agent_harness import ReActAgent, PlanAgent, ConversationalAgent

# ReAct: Think → Act → Observe 循环
react = ReActAgent(name="react", llm=llm, tools=[search_web])

# Plan-and-Execute: Plan → Execute each step → Synthesize
planner = PlanAgent(name="planner", llm=llm, tools=[search_web], allow_replan=True)

# Conversational: 单次 LLM 调用，不使用工具
chat = ConversationalAgent(name="chat", llm=llm)
```

所有 Agent 都继承自 `BaseAgent`，共享统一接口：`await agent.run(input) -> AgentResult`。

### Tools

通过 `@tool` decorator 可以快速定义工具。类型注解与 docstring 会自动转换为 LLM function calling 所需的 JSON Schema：

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

框架同时支持同步与异步函数；参数支持 `str`、`int`、`float`、`bool`、`list` 与 `dict`。

### LLM Providers

无需修改 Agent 代码即可切换底层 Provider：

```python
from agent_harness import HarnessConfig
from agent_harness.llm import LLM

config = HarnessConfig.load("config.yaml")
llm = LLM(config)  # 根据 config.llm.provider 自动匹配 provider
```

还可以叠加重试与降级能力：

```python
from agent_harness.llm.retry import RetryableLLM, RetryPolicy, FallbackChain

resilient = RetryableLLM(openai_llm, policy=RetryPolicy(max_retries=3))
fallback = FallbackChain(providers=[openai_llm, anthropic_llm])
```

### Memory

三种记忆分别承担不同职责：

```python
from agent_harness.memory import ShortTermMemory, WorkingMemory

# 会话缓冲区（滑动窗口或 token 限制）
short_term = ShortTermMemory(max_messages=50, strategy="sliding_window")

# 面向推理状态的键值 scratchpad
working = WorkingMemory()
working.set("plan", "Step 1: Search → Step 2: Analyze → Step 3: Report")
working.set("current_step", "Step 1")
print(working.to_prompt_string())
```

对于语义检索，可使用 `LongTermMemory` 配合向量存储与 embedding 函数：

```python
from agent_harness.memory import LongTermMemory
from agent_harness.memory.storage.numpy import NumpyVectorStore

ltm = LongTermMemory(store=NumpyVectorStore(), embedding_fn=embed)
await ltm.add("Important finding", metadata={"source": "paper1"})
results = await ltm.query("related topic", top_k=5)
```

### Orchestration

可以将多个 Agent 组合为复杂工作流：

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
        condition=lambda x: len(x) > 100,  # 如果输出过短则跳过
    ),
])
result = await pipeline.run("Latest advances in AI safety")
```

其他编排模式：

```python
from agent_harness.orchestration import DAGOrchestrator, DAGNode, AgentRouter, Route, AgentTeam

# 并行 DAG —— 无依赖节点可并发执行
dag = DAGOrchestrator(nodes=[
    DAGNode(id="search", agent=searcher),
    DAGNode(id="analyze", agent=analyzer, dependencies=["search"]),
    DAGNode(id="report", agent=reporter, dependencies=["analyze"]),
])

# 基于意图的路由
router = AgentRouter(
    routes=[
        Route(agent=coder, condition=r"code|program|function"),
        Route(agent=researcher, condition=lambda x: "search" in x.lower()),
    ],
    fallback=general_agent,
)

# 多 Agent 团队（supervisor、debate 或 round-robin）
team = AgentTeam(
    agents=[researcher, analyst, writer],
    mode="supervisor",
)
```

### Context

`AgentContext` 将 memory、state、variables 与 events 统一封装。通过 `fork()` 可以为子 Agent 创建共享全局状态、但独立执行过程的上下文：

```python
from agent_harness.context import AgentContext

parent_ctx = AgentContext.create(config=config)
parent_ctx.variables.set("task", "research AI safety", scope=Scope.GLOBAL)

# 子上下文共享 long-term memory、global variables 与 event bus
# 但拥有独立的 short-term memory、working memory 与 state
child_ctx = parent_ctx.fork(name="sub-agent")
```

---

## 示例

| 文件 | 说明 |
|---|---|
| `react_agent.py` | 使用自定义工具与事件日志的基础 ReAct Agent |
| `plan_and_execute.py` | 使用 PlanAgent 对复杂研究任务进行拆解与执行 |
| `multi_agent_pipeline.py` | 顺序 Pipeline：researcher → writer → editor |
| `agent_team.py` | 共享上下文下的 supervisor 模式团队协作 |
| `deep_research.py` | 使用 DAG 并行执行研究与综合的完整示例 |

运行任意示例：

```bash
export OPENAI_API_KEY="sk-..."
python examples/react_agent.py
```

---

## 配置

### YAML Config

你可以通过 `config.yaml` 以声明式方式统一配置所有组件：

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

在代码中加载：

```python
config = HarnessConfig.load("config.yaml")
```

### Environment Variables

所有设置都可以通过环境变量覆盖：

| 变量 | 说明 | 默认值 |
|---|---|---|
| `OPENAI_API_KEY` | OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic API key | — |
| `HARNESS_LLM_PROVIDER` | LLM provider（`openai`、`anthropic`） | `openai` |
| `HARNESS_LLM_MODEL` | 模型名称 | `gpt-4o` |
| `HARNESS_LLM_TEMPERATURE` | 采样温度 | `0.7` |
| `HARNESS_LLM_MAX_TOKENS` | 最大输出 token 数 | `4096` |
| `HARNESS_VERBOSE` | 是否开启详细日志 | `false` |
| `HARNESS_TRACING_ENABLED` | 是否开启 tracing | `true` |

使用 `HarnessConfig.from_env()` 时，环境变量优先级更高。多个配置对象可通过 `config.merge(other)` 合并。

---

## 项目结构

```text
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

## 测试

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Lint 与类型检查：

```bash
ruff check src/
mypy src/
```

---

## License

MIT
