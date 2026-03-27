<p align="center">
  <img src="docs/images/banner.svg" alt="Agent Harness" width="100%">
</p>

<p align="center">
  <b>轻量 · 易上手 · 可扩展的 AI Agent 框架</b>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT">
  <a href="README.md"><img src="https://img.shields.io/badge/📄_English-click-lightgrey" alt="English"></a>
</p>

十行代码，跑通一个能调用工具的 Agent。
再组合几个组件，就是一套 Multi-Agent 并行工作流。

Agent-Harness 备好了构建 Agent 要用的东西 ——
工具调用、上下文管理、任务编排 ——
剩下的事情是你的：读懂它，改造它，搭出你自己的框架。

> **No magic, no lock-in. Clone it, hack it, make it yours.**

---

## 近期更新

- `2026.03.27` **v0.2.0** — 上线 Skill 系统，Agent 按需加载领域技能，自动适配行为。
  内置 `humanizer` 和 `comm-lit-review` 两个skill。
- `2026.03.26` 新增内置工具 `paper_search`、`paper_fetch`，接入 arXiv 和 Semantic Scholar。
  `pdf_parser` 可靠性增强。
- `2026.03.22` **v0.1.0** — 首次发布。三种 Agent、四种编排模式、三层记忆、双 Provider、
  完整 Tracing，以及 5 个内置工具（`web_search`、`web_fetch`、`terminal`、`pdf_parser`、`take_notes`）。

---

## 亮点

### 即刻上手

⚡ **极简 API** — `@tool` 定义工具，创建 Agent，调用 `run()`。十行代码，直接跑。

🔧 **零模板代码** — `@tool` 装饰器根据你的类型注解和 docstring，自动生成 Tool JSON Schema。同步异步都行。

🧩 **天生可改** — 没有藏起来的魔法。Agent、工具、记忆、LLM Provider，每个组件都能继承、替换、重写。

### 从简单到复杂

🤖 **多种 Agent 模式** — `ReActAgent` 工具调用循环，`PlanAndExecuteAgent` 多步任务拆解 + 动态重规划，`ConversationalAgent` 直接对话。

🔀 **四种编排方式** — `Pipeline` 顺序链式，`DAGOrchestrator` 并行依赖图，`AgentRouter` 意图路由，`AgentTeam` 多 Agent 协作（supervisor / debate / round-robin）。

🧱 **结构化上下文** — 对话缓冲、工作暂存、长期知识检索。Agent 每一步都在构建自己需要的上下文。

### 为生产环境打造

🔍 **内置 Tracing** — LLM 调用、工具执行、推理步骤，全部自动追踪，Span 层级清晰可查。支持控制台和 JSON 导出。

🌐 **不绑定供应商** — 开箱支持 OpenAI 和 Anthropic，内置重试、限流、降级链。换 Provider 不用改 Agent 代码。

⚙️ **灵活配置** — 一个 YAML 文件管理所有组件，环境变量随时覆盖，也可以按 Agent 单独定制。

---

## 快速开始

### 1. 环境准备

```bash
git clone https://github.com/yourname/Agent-Harness.git
cd Agent-Harness

# 选一种方式创建环境
conda env create -f environment.yml    # conda
# 或: python -m venv .venv && source .venv/bin/activate
# 或: uv venv && source .venv/bin/activate

pip install -e ".[dev]"
```

### 2. 配置

创建 `config.yaml`（完整选项见 [config_example.yaml](config_example.yaml)）：

```yaml
llm:
  provider: openai
  model: gpt-5.4
  api_key: sk-...
  base_url: https://api.openai.com/v1
  reasoning_effort: high
  # ...
```

> 所有配置项都能用 `HARNESS_` 前缀的环境变量覆盖，比如 `HARNESS_LLM_MODEL`。

### 3. 运行示例

```bash
python examples/react_agent.py           # ReAct Agent + 工具调用
python examples/react_agent.py --stream  # 流式输出
```

### 4. 写你自己的 Agent

```python
import asyncio
from agent_harness import ReActAgent, tool, HarnessConfig

@tool
def calculate(expression: str) -> str:
    """Evaluate a math expression.

    Args:
        expression: A valid Python math expression like '2 + 3 * 4'.
    """
    return str(eval(expression))

async def main():
    config = HarnessConfig.load("config.yaml")
    agent = ReActAgent(
        name="assistant",
        tools=[calculate],
        config=config,
    )
    result = await agent.run("What is (42 * 37 + 15) / 3?")
    print(result.output)
    print(f"Steps: {result.step_count}, Tokens: {result.usage.total_tokens}")

asyncio.run(main())
```

---

## 架构

<p align="center">
  <img src="docs/images/architecture.svg" alt="Architecture" width="100%">
</p>

### 内置 Tracing

LLM 调用、工具执行、推理过程，跑的时候自动追踪：

```
▶ [agent] agent.assistant (start)
  input: What's the weather in Paris and Tokyo? Also, what is the population of France divided by 4?
  ✓ [internal] step.1 (9073.0ms)
    agent: assistant
    • llm_call {agent=assistant, message_count=2}
    • tool_call {agent=assistant, tool=get_weather, args={'city': 'Paris'}}
    • tool_call {agent=assistant, tool=get_weather, args={'city': 'Tokyo'}}
    • tool_call {agent=assistant, tool=get_population, args={'country': 'France'}}
    • tool_result {content='Paris: 17°C, rainy'}
    • tool_result {content='Tokyo: 20°C, partly cloudy'}
    • tool_result {content='68 million'}
  ✓ [internal] step.2 (1464.4ms)
    agent: assistant
    • llm_call {agent=assistant, message_count=6}
    • tool_call {agent=assistant, tool=calculate, args={'expression': '68_000_000/4'}}
    • tool_result {content='17000000.0'}
  ✓ [internal] step.3 (2119.9ms)
    agent: assistant
    • llm_call {agent=assistant, message_count=8}
✓ [agent] agent.assistant (12659.1ms)
```

---

## 示例

`examples/` 包含覆盖核心能力的示例：

- **[react_agent.py](examples/react_agent.py)** — ReAct 推理循环 + 自定义工具
- **[plan_and_execute.py](examples/plan_and_execute.py)** — 自动拆解任务，逐步调用工具执行，支持动态重规划
- **[multi_agent_pipeline.py](examples/multi_agent_pipeline.py)** — 一个文件跑通三种编排：Pipeline 顺序、DAG 并行、Router 路由
- **[agent_team.py](examples/agent_team.py)** — 多 Agent 协作：supervisor、debate、round-robin
- **[deep_research.py](examples/deep_research.py)** — 完整流程：规划 → 并行研究（DAG）→ 交叉评审（Team）→ 综合报告
- **[skill_demo.py](examples/skill_demo.py)** — 展示如何使用 `skills` 来做特定任务，以文本润色为例

---

## 参与贡献

写了好用的 Tool、新的 Agent 模式、或者改进了某个模块？欢迎贡献。

保持现有风格就好，enjoy building.

本项目基于 [MIT License](LICENSE) 开源。
