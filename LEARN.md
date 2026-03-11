# LEARN.md

## 前言

这份文档面向**有编程基础，但对 `agent_harness` 完全陌生**的开发者。
它不是 README 的改写版，而是一份基于真实源码整理出的“学习地图”：一边解释框架里**有什么**，一边解释它**为什么这样设计**，以及这些组件在运行时**如何串起来**。

本文写作时直接参考了以下真实材料：

- `./framework_api_reference.txt`
- `README.md`
- `examples/*.py`
- `tests/**/*.py`
- `src/agent_harness/**/*.py`

阅读建议：

1. 先看第一章，建立整体分层感。
2. 再看第七章和第八章，把单 Agent 和多 Agent 的运行链路串起来。
3. 最后回头细看第二到第六章，补齐事件、工具、记忆、上下文等基础设施。

---

## 第一章：架构总览

### 1.1 框架分层

`agent_harness` 的源码主要集中在 `src/agent_harness/` 下，可以粗略分成九层：

1. **Core**：最底层协议与公共能力，例如 `Message`、`EventBus`、`HarnessConfig`、异常体系。
2. **LLM**：把 OpenAI / Anthropic 差异折叠到统一接口 `BaseLLM` 下。
3. **Tool**：把普通 Python 函数或类包装成可供 LLM 调用的工具。
4. **Memory**：短期对话记忆、工作记忆、长期语义记忆。
5. **Context**：把配置、状态、变量、记忆、事件、追踪聚合到 `AgentContext`。
6. **Agent**：真正执行任务的主体，包括 `ReActAgent`、`PlanAgent`、`ConversationalAgent`。
7. **Orchestration**：让多个 Agent 以 Pipeline、DAG、Router、Team 的方式协作。
8. **Tracing**：结构化追踪，记录 `Span`，导出到控制台或 JSONL。
9. **Prompt**：模板、库、构建器，用于组织提示词，但不是核心执行链路的唯一入口。

从目录上看，大致如下：

```python
src/agent_harness/
├── core/
├── llm/
├── tool/
├── memory/
├── context/
├── agent/
├── orchestration/
├── tracing/
├── prompt/
└── utils/
```

### 1.2 对外暴露了什么

根导出位于 `src/agent_harness/__init__.py`：

```python
from agent_harness.core.message import Message, Role, ToolCall, ToolResult
from agent_harness.core.config import HarnessConfig
from agent_harness.core.event import Event, EventBus
from agent_harness.core.errors import HarnessError
from agent_harness.tool.base import BaseTool, ToolSchema
from agent_harness.tool.decorator import tool
from agent_harness.agent.base import BaseAgent, AgentResult
from agent_harness.agent.react import ReActAgent
from agent_harness.agent.planner import PlanAgent
from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.context.context import AgentContext
```

这段导出本身就说明了作者的心智模型：

- **消息协议**、**配置**、**事件**是公共基础。
- Agent、Tool、Context 是最常被业务代码直接实例化的对象。
- 框架希望用户首先通过 `ReActAgent` / `PlanAgent` / `ConversationalAgent` 来上手，而不是从底层拼装所有细节。

### 1.3 一条最重要的运行链路

如果只记住一条链路，应该记住下面这条：

```python
用户输入
→ Message.user(...)
→ BaseAgent.run()
→ short_term_memory.add_message(...)
→ Agent.step()
→ BaseAgent.call_llm()
→ BaseLLM.generate_with_events()
→ Provider(OpenAI/Anthropic).generate()
→ assistant Message / tool_calls
→ BaseAgent.execute_tools()
→ ToolExecutor.execute_batch()
→ Message.tool(...)
→ 再次进入下一轮 step()
→ 直到 StepResult.response 不为 None
→ AgentResult 返回
```

这是整个框架的主干。后面的所有章节，本质上都是在解释这条链路中的某一个节点。

### 1.4 用例如何落到这套架构上

`examples/` 目录展示了几种典型组合方式：

- `examples/react_agent.py`：最小可用闭环，展示 Tool + ReAct。
- `examples/plan_and_execute.py`：展示 `PlanAgent` 的三阶段工作方式。
- `examples/multi_agent_pipeline.py`：展示 `Pipeline` 和 `DAGOrchestrator`。
- `examples/agent_team.py`：展示 `AgentTeam` 的三种协作模式。
- `examples/deep_research.py`：把 `PlanAgent + ReActAgent + ConversationalAgent + DAGOrchestrator` 组合成一个完整研究流水线。

其中 `examples/deep_research.py` 最能代表框架作者想支持的“复杂工作流”：先规划，再分支并行执行，再统一收敛输出。

---

## 第二章：核心基础设施（Core）

Core 层对应 `src/agent_harness/core/`，包含：

- `message.py`
- `event.py`
- `config.py`
- `errors.py`
- `registry.py`
- `lifecycle.py`

这层的角色不是“做业务”，而是提供**一套稳定协议**，让其他层可以互相解耦。

### 2.1 Message：框架的统一通信协议

文件：`src/agent_harness/core/message.py`

最关键的几个模型如下：

```python
class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:12]}")
    name: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    tool_call_id: str
    content: str
    is_error: bool = False


class Message(BaseModel):
    role: Role
    content: str | None = None
    name: str | None = None
    tool_calls: list[ToolCall] | None = None
    tool_result: ToolResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
```

这套模型解决了一个很关键的问题：

> 框架里 LLM、Agent、Tool、Memory、Provider 都需要交换信息，但它们不应该各自定义一套格式。

于是 `Message` 成了全框架统一协议：

- 用户输入是 `Message.user(...)`
- 系统提示是 `Message.system(...)`
- LLM 输出是 `Message.assistant(...)`
- 工具结果是 `Message.tool(...)`

这能带来两个直接好处：

1. **LLM Provider 更容易做适配**：OpenAI 和 Anthropic 的请求格式不同，但内部都先统一成 `Message`。
2. **Memory 更容易复用**：短期记忆、工作记忆都对外暴露 `get_context_messages()`，返回的都是 `list[Message]`。

`Message` 还提供了几个工厂方法：

```python
@classmethod
def system(cls, content: str, **kwargs: Any) -> Message:
    return cls(role=Role.SYSTEM, content=content, **kwargs)

@classmethod
def user(cls, content: str, **kwargs: Any) -> Message:
    return cls(role=Role.USER, content=content, **kwargs)

@classmethod
def assistant(
    cls,
    content: str | None = None,
    tool_calls: list[ToolCall] | None = None,
    **kwargs: Any,
) -> Message:
    return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls, **kwargs)
```

这说明作者希望外部代码以“语义化构造”而不是“手写 role 字符串”的方式使用消息对象。

### 2.2 EventBus：组件之间的低耦合协作层

文件：`src/agent_harness/core/event.py`

`EventBus` 的核心实现很短，但很关键：

```python
class EventBus:
    def __init__(self) -> None:
        self._handlers: dict[str, list[EventHandler]] = {}
        self._lock = asyncio.Lock()

    async def emit(self, event: Event) -> None:
        matching_handlers: list[EventHandler] = []
        async with self._lock:
            for pattern, handlers in self._handlers.items():
                if fnmatch.fnmatch(event.type, pattern):
                    matching_handlers.extend(handlers)

        if not matching_handlers:
            return

        results = await asyncio.gather(
            *(h(event) for h in matching_handlers),
            return_exceptions=True,
        )
```

#### 2.2.1 为什么需要 EventBus

如果 `BaseAgent`、`ToolExecutor`、`BaseLLM` 直接互相调用日志或监控模块，会造成：

- 组件之间硬依赖；
- 调试和生产逻辑耦合；
- 想接入新的观测器、统计器、UI 时必须改核心逻辑。

`EventBus` 的设计思路是：

- 核心组件只负责 `emit()`；
- 外部系统通过 `on(pattern, handler)` 订阅；
- 双方互不感知具体实现。

这是一种典型的**发布-订阅**结构。

#### 2.2.2 wildcard 机制为什么重要

用户特别要求展开 `wildcard` 机制，这里是框架里非常实用的一个点。

源码注释写得很明确：

```python
Wildcard matching:
    - "agent.*" matches "agent.step.start", "agent.run.end", etc.
    - "*" matches everything
```

关键实现是 `fnmatch.fnmatch(event.type, pattern)`。

这意味着订阅者可以按**命名空间前缀**监听事件，而不用把每个具体事件都列出来：

- `agent.*`：监听所有 Agent 相关事件。
- `tool.execute.*`：监听工具执行生命周期。
- `llm.generate.*`：监听模型调用。
- `*`：监听整个系统。

测试 `tests/unit/test_core/test_event.py` 也验证了这个语义：

```python
bus.on("agent.*", handler)

await bus.emit(Event(type="agent.step.start"))
await bus.emit(Event(type="agent.run.end"))
await bus.emit(Event(type="tool.execute.start"))

assert "agent.step.start" in received
assert "agent.run.end" in received
assert "tool.execute.start" not in received
```

这个设计的价值在于：**事件名本身就是一棵可扩展的命名树**。

当系统长大时，你不用改订阅方，只要遵循命名规范继续发事件即可。

#### 2.2.3 并发执行与容错策略

`emit()` 内部使用 `asyncio.gather(..., return_exceptions=True)`，说明作者做了两个决定：

1. **多个 handler 并发执行**，避免一个慢 handler 阻塞整个事件流。
2. **某个 handler 抛错不会阻止其他 handler 运行**。

这和测试行为一致：`test_handler_error_does_not_stop_others` 显式验证了坏 handler 不会中断好 handler。

这是一种很工程化的取舍：事件系统的职责是“尽量投递”，不是“强一致事务”。

### 2.3 EventEmitter：让任意组件拥有发事件能力

同文件里还有一个 `EventEmitter` mixin：

```python
class EventEmitter:
    _event_bus: EventBus | None = None

    def set_event_bus(self, bus: EventBus) -> None:
        self._event_bus = bus

    async def emit(self, event_type: str, *, source: str | None = None, **data: Any) -> None:
        if self._event_bus is None:
            return
        event = Event(
            type=event_type,
            data=data,
            source=source or getattr(self, "name", self.__class__.__name__),
        )
        await self._event_bus.emit(event)
```

这段设计很克制：

- 组件不持有完整监控系统，只持有一个可选的 `EventBus`。
- 没有总线时直接 no-op，不阻塞主流程。
- `source` 默认取 `name` 属性，没有就退回类名。

于是 `BaseAgent`、`ToolExecutor`、`BaseLLM` 都可以共享同一套事件发布方式。

### 2.4 HarnessConfig：配置分层与合并

文件：`src/agent_harness/core/config.py`

`HarnessConfig` 是一棵聚合配置树：

```python
class HarnessConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tool: ToolConfig = Field(default_factory=ToolConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tracing: TracingConfig = Field(default_factory=TracingConfig)
    verbose: bool = False
```

它支持三种来源：

1. 默认值；
2. `from_yaml()`；
3. `from_env()`；

以及 `merge()` 深度合并。

设计动机很明确：

- 示例代码可以直接用默认值；
- 本地开发可以用环境变量快速覆盖；
- 真正部署时可以从 YAML 加载一套结构化配置。

`LLMConfig.model_post_init()` 还有一个非常实用的小设计：

```python
def model_post_init(self, __context: Any) -> None:
    if self.api_key is None:
        env_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
        }
        env_var = env_map.get(self.provider)
        if env_var:
            self.api_key = os.environ.get(env_var)
```

也就是说，Provider 只需要知道你是谁（`openai` / `anthropic`），就能自动找环境变量，不要求业务层反复写样板代码。

### 2.5 异常层级：把失败显式分类

文件：`src/agent_harness/core/errors.py`

异常树的主干如下：

```python
class HarnessError(Exception):
    ...

class LLMError(HarnessError):
    ...

class ToolError(HarnessError):
    ...

class ContextError(HarnessError):
    ...

class OrchestrationError(HarnessError):
    ...

class AgentError(HarnessError):
    ...
```

每类下面继续细分，例如：

- `LLMRateLimitError`
- `LLMAuthenticationError`
- `LLMContextLengthError`
- `ToolNotFoundError`
- `ToolTimeoutError`
- `StateTransitionError`
- `CyclicDependencyError`
- `MaxStepsExceededError`

这种分层的意义不只是“好看”，而是为了让上层能做不同恢复策略：

- `LLMRateLimitError` 适合重试；
- `ToolNotFoundError` 多半是配置错误；
- `StateTransitionError` 是框架内部状态机不一致；
- `MaxStepsExceededError` 是 Agent 策略没有收敛。

### 2.6 Registry 与 Lifecycle：小而通用的底座

文件：

- `src/agent_harness/core/registry.py`
- `src/agent_harness/core/lifecycle.py`

`Registry[T]` 是一个非常朴素的命名注册表：

```python
class Registry(Generic[T]):
    def register(self, name: str, item: T) -> None:
        self._items[name] = item

    def get(self, name: str) -> T:
        if name not in self._items:
            raise KeyError(...)
        return self._items[name]
```

`ToolRegistry`、`PromptLibrary` 都建立在这个通用基类上。

`lifecycle.py` 里的 `Initializable` / `Disposable` 目前还只是协议：

```python
@runtime_checkable
class Initializable(Protocol):
    async def initialize(self) -> None: ...

@runtime_checkable
class Disposable(Protocol):
    async def dispose(self) -> None: ...
```

它们的价值在于给未来扩展留出一致接口，比如：

- 需要懒初始化的向量存储；
- 需要释放连接池的 Provider；
- 需要 shutdown 的后台组件。

也就是说，这一层不是复杂，而是刻意保持**协议先行、实现后补**。

---

## 第三章：LLM 抽象层

LLM 层位于 `src/agent_harness/llm/`，核心文件有：

- `types.py`
- `base.py`
- `openai_provider.py`
- `anthropic_provider.py`
- `retry.py`

这一层的目标不是“封装所有模型能力”，而是提供一个足够稳定的最小公共面：

- 统一输入：`list[Message]`
- 统一输出：`LLMResponse`
- 可选工具：`list[ToolSchema]`
- 可选流式：`stream()`

### 3.1 types.py：统一返回值

文件：`src/agent_harness/llm/types.py`

最重要的是 `LLMResponse`：

```python
class LLMResponse(BaseModel):
    message: Message
    usage: Usage = Field(default_factory=Usage)
    finish_reason: FinishReason = FinishReason.STOP
    model: str | None = None
    raw_response: Any | None = None

    @property
    def has_tool_calls(self) -> bool:
        return self.message.has_tool_calls
```

这里面有两个很好的设计点：

1. **把 provider-specific response 保留在 `raw_response`**，方便调试；
2. **业务代码主要只看 `message` 和 `has_tool_calls`**，减少和 OpenAI / Anthropic SDK 的耦合。

### 3.2 BaseLLM：统一抽象与事件包装

文件：`src/agent_harness/llm/base.py`

抽象接口是：

```python
class BaseLLM(ABC, EventEmitter):
    @abstractmethod
    async def generate(... ) -> LLMResponse:
        ...

    @abstractmethod
    async def stream(... ) -> AsyncIterator[StreamDelta]:
        ...
```

真正值得注意的是 `generate_with_events()`：

```python
async def generate_with_events(
    self,
    messages: list[Message],
    tools: list[ToolSchema] | None = None,
    **kwargs: Any,
) -> LLMResponse:
    await self.emit("llm.generate.start", model=self.model_name, message_count=len(messages))
    try:
        response = await self.generate(messages, tools=tools, **kwargs)
        await self.emit(
            "llm.generate.end",
            model=self.model_name,
            usage=response.usage.model_dump(),
            finish_reason=response.finish_reason.value,
        )
        return response
    except Exception as e:
        await self.emit("llm.generate.error", model=self.model_name, error=str(e))
        raise
```

也就是说：

- provider 只管实现 `generate()`；
- 公共事件埋点由 `BaseLLM` 统一包起来；
- `BaseAgent.call_llm()` 调的也是这个包装版本，而不是裸 `generate()`。

这样做的好处是：

- 所有 LLM Provider 自动获得一致的事件语义；
- 监控侧只要监听 `llm.generate.*` 即可，不必关心 provider 类型。

### 3.3 OpenAIProvider：把框架消息翻译成 OpenAI Chat Completions

文件：`src/agent_harness/llm/openai_provider.py`

OpenAI 适配器的核心任务有三个：

1. 把 `Message` 转换成 OpenAI API 所需 dict；
2. 把 `ToolSchema` 转成 OpenAI function calling format；
3. 把 OpenAI 返回结果再转回 `LLMResponse`。

消息格式化逻辑如下：

```python
@staticmethod
def _format_message(msg: Message) -> dict[str, Any]:
    result: dict[str, Any] = {"role": msg.role.value}

    if msg.role == Role.TOOL and msg.tool_result:
        result["tool_call_id"] = msg.tool_result.tool_call_id
        result["content"] = msg.tool_result.content
        return result

    if msg.content is not None:
        result["content"] = msg.content

    if msg.tool_calls:
        result["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.name,
                    "arguments": json.dumps(tc.arguments),
                },
            }
            for tc in msg.tool_calls
        ]
```

这个设计说明 `Message` 模型足够通用，既能表示普通对话，也能表示工具调用链。

工具请求构造逻辑在 `_build_request()`：

```python
if tools:
    request["tools"] = [t.to_openai_format() for t in tools]
    if tool_choice:
        if tool_choice in ("auto", "required", "none"):
            request["tool_choice"] = tool_choice
        else:
            request["tool_choice"] = {
                "type": "function",
                "function": {"name": tool_choice},
            }
```

这意味着框架层面已经支持：

- 自动决定是否用工具；
- 强制使用某个工具；
- 禁止工具调用；

只是当前 Agent 层默认没有大规模暴露这些策略参数。

### 3.4 AnthropicProvider：同一协议，不同翻译方式

文件：`src/agent_harness/llm/anthropic_provider.py`

Anthropic 的差异不在“能力”，而在协议细节：

- `system` 不是消息，而是顶层参数；
- 工具调用和工具结果都放在 content block 里；
- 工具结果消息不是 `tool` role，而是 `user` role 的 `tool_result` block。

最关键的转换逻辑在 `_split_system_message()`：

```python
for msg in messages:
    if msg.role == Role.SYSTEM:
        system_content = msg.content
        continue

    if msg.role == Role.TOOL and msg.tool_result:
        api_messages.append({
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_result.tool_call_id,
                    "content": msg.tool_result.content,
                    "is_error": msg.tool_result.is_error,
                }
            ],
        })
        continue

    if msg.role == Role.ASSISTANT and msg.tool_calls:
        content_blocks: list[dict[str, Any]] = []
        if msg.content:
            content_blocks.append({"type": "text", "text": msg.content})
        for tc in msg.tool_calls:
            content_blocks.append({
                "type": "tool_use",
                "id": tc.id,
                "name": tc.name,
                "input": tc.arguments,
            })
```

这段代码很好地体现了 LLM 抽象层的真正价值：

- 对 Agent 而言，不需要知道 Anthropic 的特殊协议；
- 只要喂 `Message` 和 `ToolSchema`，Provider 自己消化差异。

### 3.5 generate / stream 的职责分离

`generate()` 用于一次性响应。
`stream()` 用于增量输出，返回 `AsyncIterator[StreamDelta]`。

目前框架对 streaming 的支持已经有类型和 provider 适配，但在 Agent 主执行链里，默认走的还是 `generate_with_events()`，也就是**非流式主路径**。

这说明当前版本的优先级是：

1. 先把 Agent、Tool、Memory、Orchestration 的闭环跑通；
2. 再在需要时扩展更完整的流式交互体验。

### 3.6 Retry / Fallback / RateLimiter：把“脆弱外部依赖”包起来

文件：`src/agent_harness/llm/retry.py`

这里不是一个继承体系，而是一组“包裹器”与策略对象。

#### RetryPolicy

```python
class RetryPolicy(BaseModel):
    max_retries: int = 3
    backoff_factor: float = 2.0
    initial_delay: float = 1.0
    max_delay: float = 60.0
    retryable_exceptions: list[str] = [
        "LLMRateLimitError",
        "ConnectionError",
        "TimeoutError",
    ]
```

它把重试策略参数化，避免写死在 Provider 中。

#### RetryableLLM

```python
class RetryableLLM:
    async def generate(... ) -> LLMResponse:
        for attempt in range(self.policy.max_retries + 1):
            try:
                if self.rate_limiter:
                    await self.rate_limiter.acquire()
                return await self.llm.generate(messages, tools=tools, **kwargs)
            except Exception as e:
                ...
```

这里采用的是**组合而不是继承**。原因很直接：

- 不想让每个 Provider 都实现一遍重试；
- 也不想污染 `BaseLLM` 的核心语义；
- 想让用户按需包裹。

#### FallbackChain

```python
class FallbackChain:
    async def generate(... ) -> LLMResponse:
        for provider in self.providers:
            try:
                return await provider.generate(messages, tools=tools, **kwargs)
            except Exception as e:
                logger.warning("Provider %s failed: %s, trying next", provider, e)
```

这适合把“主模型 + 备选模型”串起来。

#### RateLimiter

```python
class RateLimiter:
    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            self._timestamps = [t for t in self._timestamps if now - t < self._window]
            if len(self._timestamps) >= self._max_requests:
                wait_time = self._timestamps[0] + self._window - now
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
            self._timestamps.append(time.monotonic())
```

它是典型的滑动窗口限流，并不关心 token 数，只关心请求数。

### 3.7 这一层的设计重点

LLM 层最重要的不是“支持了几个 Provider”，而是它实现了下面几个抽象边界：

- Agent 永远面向 `BaseLLM`，而不是具体 SDK；
- 消息统一为 `Message`；
- 工具统一为 `ToolSchema`；
- 使用量统一为 `Usage`；
- 可观测性统一挂在 `generate_with_events()` 上。

这使得上层策略代码可以专注于“如何推理”，而不是“怎么调用某家 API”。

---

## 第四章：工具系统

工具系统位于 `src/agent_harness/tool/`，核心文件有：

- `base.py`
- `decorator.py`
- `registry.py`
- `executor.py`
- `builtin/*.py`

这一层的目标是把 Python 可执行能力包装成 LLM 可以理解和调用的结构。

### 4.1 BaseTool 与 ToolSchema

文件：`src/agent_harness/tool/base.py`

最底层抽象分为两部分：

1. `BaseTool`：运行时可执行对象；
2. `ToolSchema`：给 LLM 看的静态接口描述。

```python
class ToolSchema(BaseModel):
    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=lambda: {
        "type": "object",
        "properties": {},
        "required": [],
    })

    def to_openai_format(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }
```

这段代码非常值得注意：**ToolSchema 是 provider-neutral 的中间层**。

也就是说，框架并不是直接生成 OpenAI 格式或 Anthropic 格式，而是先生成自己的统一 Schema，再由 Provider 转换。

### 4.2 `@tool` decorator：从函数签名自动构建 schema

文件：`src/agent_harness/tool/decorator.py`

这是工具系统最“像框架”的地方。用户特别要求展开这一点。

#### 4.2.1 从函数到工具对象的总流程

`@tool` 的入口如下：

```python
def tool(
    fn: Callable | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    def decorator(f: Callable) -> FunctionTool:
        schema = _build_schema_from_function(f, name, description)
        return FunctionTool(
            fn=f,
            name=schema.name,
            description=schema.description,
            schema=schema,
        )
```

也就是说，装饰器做了两件事：

1. 分析函数，构建 `ToolSchema`；
2. 把函数包成 `FunctionTool`。

#### 4.2.2 schema 是怎么从函数签名推导出来的

核心逻辑在 `_build_schema_from_function()`：

```python
def _build_schema_from_function(
    fn: Callable,
    name: str | None,
    description: str | None,
) -> ToolSchema:
    fn_name = name or fn.__name__
    fn_doc = description or (fn.__doc__ or "").split("\n")[0].strip() or fn_name

    hints = get_type_hints(fn)
    sig = inspect.signature(fn)
    param_docs = _parse_docstring_params(fn.__doc__)

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue

        py_type = hints.get(param_name, str)
        prop = _python_type_to_json_schema(py_type)
        prop["description"] = param_docs.get(param_name, "")

        if param.default is not inspect.Parameter.empty:
            prop["default"] = param.default
        else:
            required.append(param_name)

        properties[param_name] = prop
```

这个过程可以拆成 6 步：

1. **确定工具名**：优先用装饰器参数 `name`，否则用函数名；
2. **确定工具描述**：优先用装饰器参数 `description`，否则取 docstring 第一行；
3. **读取类型提示**：`get_type_hints(fn)`；
4. **读取函数签名**：`inspect.signature(fn)`；
5. **解析参数文档**：`_parse_docstring_params(fn.__doc__)`；
6. **构造 JSON Schema**：填充 `properties` 和 `required`。

这和 FastAPI 的思路非常像：**让 Python 函数本身既是实现，又是接口声明**。

#### 4.2.3 类型到 JSON Schema 的映射规则

```python
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}
```

对于 `list[str]` 这类泛型，`_python_type_to_json_schema()` 会进一步展开 `items`：

```python
if origin is list:
    args = getattr(py_type, "__args__", ())
    items = _python_type_to_json_schema(args[0]) if args else {}
    return {"type": "array", "items": items}
```

测试 `tests/unit/test_tool/test_decorator.py` 验证了这些行为：

- `str -> string`
- `int -> integer`
- `float -> number`
- `bool -> boolean`
- `list[str] -> {"type": "array", "items": {"type": "string"}}`
- 未知类型默认回退为 `string`

这是一种典型的**保守型映射**：不追求覆盖 Python 所有复杂类型，而是优先保证 LLM function calling 场景足够稳定。

#### 4.2.4 docstring 是怎么变成参数描述的

`_parse_docstring_params()` 支持 Google 风格 `Args:` 块：

```python
def _parse_docstring_params(docstring: str | None) -> dict[str, str]:
    ...
    if stripped.lower().startswith("args:"):
        in_args = True
        continue
    ...
    match = re.match(r"(\w+)(?:\s*\([^)]*\))?\s*:\s*(.+)", stripped)
    if match:
        params[match.group(1)] = match.group(2).strip()
```

所以像下面这种函数：

```python
@tool
async def calculate(expression: str) -> str:
    """Evaluate a mathematical expression.

    Args:
        expression: A Python math expression to evaluate, e.g. '2 + 3 * 4'.
    """
```

参数描述会自动进入 Schema。

这件事很重要，因为对 LLM 来说，参数名只是弱提示，**真正帮助它正确选参与组装参数的是 description**。

#### 4.2.5 FunctionTool：把 sync / async 函数统一执行

```python
class FunctionTool(BaseTool):
    def __init__(...):
        super().__init__(name=name, description=description)
        self._fn = ensure_async(fn)
        self._schema = schema

    async def execute(self, **kwargs: Any) -> str:
        result = await self._fn(**kwargs)
        return str(result) if result is not None else ""
```

这里有两个细节：

- `ensure_async(fn)` 让同步函数也能被异步统一调用；
- 返回值统一转成 `str`，`None` 变成空字符串。

这说明工具系统面向的是“文本结果回填给 LLM”的主路径，而不是复杂结构化对象回传。

### 4.3 ToolRegistry：把工具组织起来

文件：`src/agent_harness/tool/registry.py`

`ToolRegistry` 本身不复杂，本质上是 `Registry[BaseTool]` 的包装器：

```python
class ToolRegistry:
    def register(self, tool: BaseTool) -> None:
        self._registry.register(tool.name, tool)

    def get_schemas(self) -> list[ToolSchema]:
        return [tool.get_schema() for tool in self.list_tools()]
```

它的价值在于分离两种关注点：

- `BaseAgent` 不需要知道工具存在哪里；
- `ToolExecutor` 只关心根据名字查工具并执行；
- LLM 只需要 `get_schemas()` 提供 schema 列表。

### 4.4 ToolExecutor：真正负责执行与防护

文件：`src/agent_harness/tool/executor.py`

工具真正运行时，走的是 `ToolExecutor` 而不是 `BaseTool` 本身。

核心代码如下：

```python
class ToolExecutor(EventEmitter):
    def __init__(self, registry: ToolRegistry, config: ToolConfig | None = None) -> None:
        self.registry = registry
        self.config = config or ToolConfig()
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
```

以及：

```python
async with self._semaphore:
    result_str = await asyncio.wait_for(
        tool.execute(**tool_call.arguments),
        timeout=timeout,
    )
```

这里体现了三个工程目标：

1. **限并发**：避免工具无限并行把系统打爆；
2. **超时保护**：避免工具悬挂导致 Agent 卡死；
3. **错误折叠为 ToolResult**：尽量把失败回传给模型，而不是直接把主流程打断。

批量执行也非常直接：

```python
async def execute_batch(self, tool_calls: list[ToolCall], timeout: float | None = None) -> list[ToolResult]:
    if not tool_calls:
        return []

    tasks = [self.execute(tc, timeout=timeout) for tc in tool_calls]
    return list(await asyncio.gather(*tasks))
```

这正好对应 `ReActAgent` 中“可以一次调用多个独立工具”的设计。

### 4.5 Tool 系统在运行时怎么接入 Agent

`BaseAgent.__init__()` 中会完成这段 wiring：

```python
self.tool_registry = ToolRegistry()
for t in (tools or []):
    self.tool_registry.register(t)
self.tool_executor = ToolExecutor(
    self.tool_registry,
    config=self.context.config.tool,
)
```

所以 Agent 层不需要自己处理：

- 工具注册；
- schema 收集；
- 并发执行；
- timeout；
- error -> `ToolResult` 映射；

它只需要：

- 把 `ToolCall` 交给 `execute_tools()`；
- 从 `ToolResult` 继续往下推理。

### 4.6 内置工具：框架内置了一批“函数工具”示例

文件：

- `src/agent_harness/tool/builtin/file_ops.py`
- `src/agent_harness/tool/builtin/http_request.py`
- `src/agent_harness/tool/builtin/python_exec.py`
- `src/agent_harness/tool/builtin/web_search.py`

这些文件本身也能反映设计哲学：

- 内置工具并没有特殊基类；
- 仍然是普通函数 + `@tool`；
- 框架鼓励用户用最简单的方式扩展能力。

例如 `read_file`：

```python
@tool
def read_file(path: str, encoding: str = "utf-8") -> str:
    try:
        return Path(path).read_text(encoding=encoding)
    except FileNotFoundError:
        return f"Error: file not found – {path}"
```

例如 `python_exec`：

```python
@tool
async def python_exec(code: str, timeout: int = 30) -> str:
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".py")
    try:
        ...
        proc = await asyncio.create_subprocess_exec(
            "python",
            tmp_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
```

这也提醒你：框架对“工具是什么”没有强约束，它只要求工具最终能变成：

- 一段 schema；
- 一个 `async execute(**kwargs) -> str`。

### 4.7 工具系统为什么这样设计

总结一下，这一层的关键设计点是：

- 对开发者友好：函数签名 + docstring 即可定义工具；
- 对模型友好：自动生成 JSON Schema；
- 对执行友好：并发、超时、错误回传都内置；
- 对 Provider 友好：统一 `ToolSchema`，再分发到 OpenAI / Anthropic。

所以工具系统不是一个“插件市场”，而是一个**把 Python 能力翻译给 LLM 的桥接层**。

---

## 第五章：记忆系统

记忆系统位于 `src/agent_harness/memory/`，包括：

- `base.py`
- `short_term.py`
- `working.py`
- `long_term.py`
- `storage/base.py`
- `storage/numpy_store.py`

### 5.1 为什么要分三种 Memory

很多 Agent 框架把“记忆”说成一个概念，但这里作者明确拆成三层：

1. **ShortTermMemory**：对话上下文缓冲区；
2. **WorkingMemory**：当前执行过程中的结构化 scratchpad；
3. **LongTermMemory**：语义检索型知识库。

这三者对应的是三种完全不同的问题：

- 我最近说了什么？
- 我当前推理到哪一步？
- 我以前见过哪些相关知识？

如果把三种需求混在一个类里，接口会很快失控。拆开以后，每层职责都非常清晰。

### 5.2 BaseMemory：统一接口，允许不同后端

文件：`src/agent_harness/memory/base.py`

统一接口是：

```python
class BaseMemory(ABC):
    @abstractmethod
    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        ...

    @abstractmethod
    async def add_message(self, message: Message) -> None:
        ...

    @abstractmethod
    async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
        ...

    @abstractmethod
    async def get_context_messages(self) -> list[Message]:
        ...
```

这个接口故意同时保留了两种入口：

- `add(content)`：适合存普通文本；
- `add_message(message)`：适合直接消费框架消息流。

这保证了 Memory 既能脱离 Agent 单独使用，也能无缝嵌到 Agent 主循环中。

### 5.3 ShortTermMemory：对话缓冲区

文件：`src/agent_harness/memory/short_term.py`

`ShortTermMemory` 是最直接的上下文管理器，内部持有 `self._messages: list[Message]`。

它支持两种裁剪策略：

- `sliding_window`
- `token_limit`

#### 5.3.1 sliding window 的行为

```python
def _trim_sliding_window(self) -> None:
    if len(self._messages) <= self.max_messages:
        return

    system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
    non_system = [m for m in self._messages if m.role != Role.SYSTEM]

    keep_count = self.max_messages - len(system_msgs)
    self._messages = system_msgs + non_system[-keep_count:]
```

这里最关键的一句话在类注释里：

> The system message (if present) is always preserved regardless of strategy.

也就是说，系统提示永远不参与普通裁剪。

这是合理的，因为：

- system prompt 通常定义了 Agent 身份和行为边界；
- 如果它被挤掉，后面整轮推理语义都会飘。

#### 5.3.2 token limit 的行为

```python
def _trim_token_limit(self) -> None:
    system_msgs = [m for m in self._messages if m.role == Role.SYSTEM]
    non_system = [m for m in self._messages if m.role != Role.SYSTEM]

    result: list[Message] = []
    budget = self.max_tokens - count_messages_tokens(system_msgs, self.model)

    for msg in reversed(non_system):
        msg_tokens = count_messages_tokens([msg], self.model)
        if budget - msg_tokens < 0:
            break
        result.append(msg)
        budget -= msg_tokens
```

这说明它采用的是**从后往前保留最近消息**的策略，而不是做摘要压缩。

设计上很务实：

- 简单；
- 可预测；
- 对多数短会话足够；
- 不引入额外摘要误差。

### 5.4 WorkingMemory：结构化 scratchpad

文件：`src/agent_harness/memory/working.py`

这是本框架里一个非常关键、也非常容易被误解的组件。

它不是“聊天记录”，而是**当前执行态的结构化状态区**。

源码定义：

```python
class WorkingMemory(BaseMemory):
    def __init__(self) -> None:
        self._scratchpad: dict[str, Any] = {}
        self._history: list[str] = []
```

它有一套显式的 key-value 接口：

```python
def set(self, key: str, value: Any) -> None:
    self._scratchpad[key] = value


def get(self, key: str, default: Any = None) -> Any:
    return self._scratchpad.get(key, default)
```

典型用法在 `PlanAgent` 里非常明显：

```python
self.context.working_memory.set("plan", self._plan.progress_summary)
self.context.working_memory.set("current_step", current.description)
```

#### 5.4.1 为什么 WorkingMemory 以 system message 注入

用户特别要求讲清楚这一点。直接看源码：

```python
async def get_context_messages(self) -> list[Message]:
    prompt_str = self.to_prompt_string()
    if not prompt_str:
        return []
    return [Message.system(prompt_str)]
```

以及 `BaseAgent.call_llm()`：

```python
working_msgs = await self.context.working_memory.get_context_messages()
if working_msgs:
    inject_idx = 1 if messages and messages[0].role.value == "system" else 0
    messages = messages[:inject_idx] + working_msgs + messages[inject_idx:]
```

也就是说，WorkingMemory 被设计成：

- 序列化为一段文本；
- 包装成 `Message.system(...)`；
- 插入到原 system prompt 后面、普通对话前面。

为什么不是 user message？为什么不是 assistant message？

因为作者想让它扮演的不是“某一轮说过的话”，而是“当前运行时约束与状态说明”。

把 WorkingMemory 注入成 system message，有三个直接好处：

1. **优先级更高**：它和 system prompt 一样，属于模型在回答前就应考虑的背景条件。
2. **不污染对话语义**：如果放成 user/assistant，会让模型误以为这是历史对话内容，而不是执行状态。
3. **和短期记忆解耦**：短期记忆负责 conversation history，WorkingMemory 负责 execution state，两者角色清晰。

`to_prompt_string()` 也能看出这种意图：

```python
def to_prompt_string(self) -> str:
    if not self._scratchpad:
        return ""

    lines = ["[Working Memory]"]
    for key, value in self._scratchpad.items():
        val_str = str(value)
        if len(val_str) > 500:
            val_str = val_str[:500] + "..."
        lines.append(f"  {key}: {val_str}")
    return "\n".join(lines)
```

它不是为人类 UI 设计的，而是为 LLM 提供一种简单、稳定、代价低的 scratchpad 注入格式。

#### 5.4.2 为什么不直接让 Agent 持有若干字段

比如 `PlanAgent` 完全可以自己持有 `_plan`、`_current_step`，那为什么还要写入 `WorkingMemory`？

原因是：

- Agent 自己的 Python 字段，LLM 看不到；
- LLM 只能看到 prompt 上下文；
- `WorkingMemory` 的价值就是把**程序内部状态投影回模型上下文**。

这是一种“程序状态 → 提示词状态”的桥接设计。

### 5.5 LongTermMemory：语义检索型记忆

文件：`src/agent_harness/memory/long_term.py`

`LongTermMemory` 和前两者最不一样的地方在于：它不直接参与每轮上下文拼装。

```python
async def get_context_messages(self) -> list[Message]:
    return []
```

这说明作者把它定位成**按需查询**，而不是自动注入。

核心接口如下：

```python
class LongTermMemory(BaseMemory):
    def __init__(self, store: BaseVectorStore, embedding_fn: EmbeddingFn) -> None:
        self._store = store
        self._embed = embedding_fn

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> None:
        doc_id = uuid.uuid4().hex
        embedding = await self._embed(content)
        doc = VectorDocument(...)
        await self._store.upsert([doc])
```

查询则是：

```python
async def query(self, query: str, top_k: int = 5) -> list[MemoryItem]:
    query_embedding = await self._embed(query)
    results = await self._store.search(
        query_embedding=query_embedding,
        top_k=top_k,
    )
```

这说明长期记忆并不试图替你做 RAG orchestration，它只提供：

- embedding；
- 向量存储；
- 检索接口；

至于检索结果什么时候查、怎么注入 prompt，需要 Agent 或业务代码自行决定。

这种设计更底层，但也更灵活。

### 5.6 向量存储抽象与 NumpyVectorStore

文件：

- `src/agent_harness/memory/storage/base.py`
- `src/agent_harness/memory/storage/numpy_store.py`

抽象接口 `BaseVectorStore` 定义了：

```python
class BaseVectorStore(ABC):
    async def upsert(self, documents: list[VectorDocument]) -> None: ...
    async def search(self, query_embedding: list[float], top_k: int = 5, filter_metadata: dict[str, Any] | None = None) -> list[VectorSearchResult]: ...
    async def delete(self, ids: list[str]) -> None: ...
    async def count(self) -> int: ...
    async def clear(self) -> None: ...
```

`NumpyVectorStore` 则给了一个简单内存实现，便于示例和小规模场景使用。

它的关键点：

- `_documents` 存原始文档；
- `_embeddings` 存矩阵；
- `_dirty` 标记索引是否要重建；
- 默认用 cosine similarity；
- 支持 `filter_metadata`。

这部分说明作者的思路是：

- 先定义统一接口；
- 提供一个零门槛参考实现；
- 把大规模存储（FAISS、Milvus、PGVector 等）的接入留给扩展层。

### 5.7 记忆层的职责分工总结

| 类型 | 文件 | 核心职责 | 是否自动注入上下文 |
|---|---|---|---|
| `ShortTermMemory` | `memory/short_term.py` | 最近对话历史 | 是 |
| `WorkingMemory` | `memory/working.py` | 当前执行 scratchpad | 是 |
| `LongTermMemory` | `memory/long_term.py` | 语义检索知识库 | 否 |

这个拆分是整个框架非常重要的设计清晰点。

---

## 第六章：上下文与状态

上下文相关代码位于 `src/agent_harness/context/`：

- `context.py`
- `state.py`
- `variables.py`

这一层回答的问题是：

> 一个 Agent 在运行时，到底依赖哪些共享对象？这些对象哪些应共享，哪些必须隔离？

### 6.1 AgentContext：统一运行容器

文件：`src/agent_harness/context/context.py`

`AgentContext` 聚合了 Agent 运行所需的主要依赖：

```python
class AgentContext:
    def __init__(
        self,
        config: HarnessConfig | None = None,
        short_term_memory: ShortTermMemory | None = None,
        long_term_memory: LongTermMemory | None = None,
        working_memory: WorkingMemory | None = None,
        state: StateManager | None = None,
        variables: ContextVariables | None = None,
        event_bus: EventBus | None = None,
        tracer: Tracer | None = None,
    ) -> None:
```

如果没有显式传入，它会自动创建默认组件。

这有两个重要含义：

1. **新手可以低门槛上手**：大多数示例完全不用自己拼 Context；
2. **高级用户可以手动注入依赖**：比如共享某个 `EventBus`、自定义 Memory、挂入 `Tracer`。

### 6.2 StateManager：Agent 的有限状态机

文件：`src/agent_harness/context/state.py`

状态枚举：

```python
class AgentState(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    OBSERVING = "observing"
    PLANNING = "planning"
    FINISHED = "finished"
    ERROR = "error"
```

允许的迁移写在 `_TRANSITIONS`：

```python
_TRANSITIONS: dict[AgentState, set[AgentState]] = {
    AgentState.IDLE: {AgentState.THINKING, AgentState.PLANNING, AgentState.FINISHED},
    AgentState.THINKING: {AgentState.ACTING, AgentState.FINISHED, AgentState.ERROR, AgentState.PLANNING},
    AgentState.ACTING: {AgentState.OBSERVING, AgentState.ERROR},
    AgentState.OBSERVING: {AgentState.THINKING, AgentState.FINISHED, AgentState.PLANNING, AgentState.ERROR},
    AgentState.PLANNING: {AgentState.THINKING, AgentState.ACTING, AgentState.FINISHED, AgentState.ERROR},
    AgentState.FINISHED: {AgentState.IDLE},
    AgentState.ERROR: {AgentState.IDLE},
}
```

为什么要这么做？

因为 Agent 的运行不是“一个函数调到底”，而是一条可观测状态链：

- 开始时 idle；
- 调 LLM 时 thinking；
- 调工具时 acting；
- 收到工具结果时 observing；
- 收敛时 finished；
- 失败时 error。

有了这层状态机，很多事情都更容易做：

- 可观测性；
- UI 展示；
- 调试；
- 非法流程保护。

测试 `tests/unit/test_context/test_context.py` 也验证了非法迁移会抛 `StateTransitionError`。

### 6.3 ContextVariables：共享变量与作用域

文件：`src/agent_harness/context/variables.py`

这里实现的是一个两层命名空间：

```python
class Scope(str, Enum):
    AGENT = "agent"
    GLOBAL = "global"
```

底层存储有两个 dict：

```python
self._agent_store: dict[str, Any] = {}
self._global_store: dict[str, Any] = global_store if global_store is not None else {}
```

访问策略是：

- `get()` 先查 agent scope；
- 再查 global scope。

这意味着局部变量可以覆盖全局变量。

这样的设计适合多 Agent 场景：

- 团队共享事实、预算、会话级配置，放 `GLOBAL`；
- 某个 Agent 的临时私有状态，放 `AGENT`。

### 6.4 `AgentContext.fork()` 的共享 / 隔离语义

用户特别要求展开这一点。

源码注释已经把语义写得很清楚：

```python
Child context shares:
- long_term_memory (shared knowledge base)
- variables (GLOBAL scope variables)
- event_bus (unified event stream)
- tracer (unified tracing)
- config

Child gets fresh:
- short_term_memory (independent conversation)
- working_memory (independent scratchpad)
- state (independent state machine)
```

具体实现：

```python
def fork(self, name: str | None = None) -> AgentContext:
    child_vars = self.variables.fork()
    return AgentContext(
        config=self.config,
        short_term_memory=ShortTermMemory(...),
        long_term_memory=self.long_term_memory,
        working_memory=WorkingMemory(),
        state=StateManager(),
        variables=child_vars,
        event_bus=self.event_bus,
        tracer=self.tracer,
    )
```

#### 6.4.1 为什么要这样分

这是一个很典型的“子任务上下文”设计问题。

如果全部共享：

- 子 Agent 会污染父 Agent 的对话历史；
- scratchpad 会相互覆盖；
- 状态机会互相踩踏；

如果全部隔离：

- 子 Agent 无法共享长期知识；
- 事件流被切碎；
- 团队协作时无法共享全局变量；

所以 `fork()` 选择了中间方案：

- **conversation 与 execution state 隔离**；
- **knowledge、global vars、events、tracing 共享**。

这是多 Agent 系统里非常合理的一种折中。

#### 6.4.2 测试如何验证这个语义

`tests/unit/test_context/test_context.py` 有一组很好的行为测试：

- `test_fork_shares_event_bus`
- `test_fork_shares_config`
- `test_fork_shares_long_term_memory`
- `test_fork_independent_short_term_memory`
- `test_fork_independent_working_memory`
- `test_fork_independent_state`
- `test_fork_shared_global_variables`
- `test_fork_independent_agent_variables`

例如共享 global 变量：

```python
parent.variables.set("shared_key", "shared_val", scope=Scope.GLOBAL)
child = parent.fork()
assert child.variables.get("shared_key") == "shared_val"

child.variables.set("child_global", 42, scope=Scope.GLOBAL)
assert parent.variables.get("child_global") == 42
```

而 agent 作用域变量不会共享。

这正是“共享团队事实、隔离个人工作台”的语义。

### 6.5 Context 层为什么单独存在

你可以把 `AgentContext` 看成是 Agent 的“运行时容器”，它的重要性在于：

- 它把跨组件依赖聚合起来；
- 它定义了哪些东西应共享、哪些应隔离；
- 它让 Agent 实例化不至于有十几个构造参数散落在业务层。

如果没有这层，`BaseAgent` 会变成一个巨大的依赖注入中心，维护成本会明显上升。

---

## 第七章：Agent 实现

Agent 相关代码位于 `src/agent_harness/agent/`：

- `base.py`
- `react.py`
- `planner.py`
- `conversational.py`
- `hooks.py`

这是框架的执行核心。

### 7.1 BaseAgent：公共控制器，而不是具体策略

文件：`src/agent_harness/agent/base.py`

`BaseAgent` 的构造参数：

```python
class BaseAgent(ABC, EventEmitter):
    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        tools: list[BaseTool] | None = None,
        context: AgentContext | None = None,
        hooks: AgentHooks | None = None,
        max_steps: int = 20,
        system_prompt: str = "",
    ) -> None:
```

从职责上说，它负责：

- 初始化上下文；
- 注册工具；
- 连接事件总线；
- 执行主循环；
- 调 LLM；
- 调工具；
- 汇总结果。

但它**不负责具体推理策略**。这点非常重要。

### 7.2 `BaseAgent.run()` 与 `step()` 的职责边界

用户特别要求讲清楚这一点。

#### `run()` 负责什么

`run()` 是框架级控制流，负责：

1. 归一化输入；
2. 把 system prompt 和用户输入写入短期记忆；
3. 设置初始状态；
4. 触发生命周期 hooks；
5. 发运行事件；
6. 循环调用 `step()`；
7. 检查是否结束；
8. 处理 `max_steps`；
9. 异常转发与状态置错；
10. 打包 `AgentResult`。

核心结构如下：

```python
async def run(self, input: str | Message) -> AgentResult:
    if isinstance(input, str):
        input_msg = Message.user(input)
        input_text = input
    else:
        input_msg = input
        input_text = input.content or ""

    if self.system_prompt:
        await self.context.short_term_memory.add_message(
            Message.system(self.system_prompt)
        )
    await self.context.short_term_memory.add_message(input_msg)
    self.context.state.transition(AgentState.THINKING)

    for step_num in range(1, self.max_steps + 1):
        step_result = await self.step()
        steps.append(step_result)
        if step_result.response is not None:
            final_output = step_result.response
            self.context.state.transition(AgentState.FINISHED)
            break
```

#### `step()` 负责什么

`step()` 是**单步推理策略接口**。

它只需要解决一个问题：

> 在当前上下文下，Agent 下一步做什么？

返回值是 `StepResult`：

```python
class StepResult(BaseModel):
    thought: str | None = None
    action: list[ToolCall] | None = None
    observation: list[ToolResult] | None = None
    response: str | None = None
```

只要某次 `step()` 返回 `response != None`，`run()` 就会结束主循环。

#### 为什么要拆成 run / step

因为这两类逻辑变化频率不同：

- `run()` 是通用框架控制流，应该稳定；
- `step()` 是 Agent 策略层，应该易于扩展。

如果二者混在一起，每实现一种新 Agent 都要复制一遍主循环，重复代码会很多，行为也容易不一致。

所以这其实是一个经典模式：

- `run()` 是模板方法（Template Method）；
- `step()` 是子类覆盖点。

### 7.3 `call_llm()`：消息拼装的真正入口

`BaseAgent.call_llm()` 是整个运行链路的关键节点：

```python
async def call_llm(
    self,
    messages: list[Message] | None = None,
    tools: list[ToolSchema] | None = None,
    **kwargs: Any,
) -> LLMResponse:
    if messages is None:
        messages = await self.context.short_term_memory.get_context_messages()

    working_msgs = await self.context.working_memory.get_context_messages()
    if working_msgs:
        inject_idx = 1 if messages and messages[0].role.value == "system" else 0
        messages = messages[:inject_idx] + working_msgs + messages[inject_idx:]

    if tools is None and self.tool_schemas:
        tools = self.tool_schemas

    response = await self.llm.generate_with_events(messages, tools=tools, **kwargs)
    await self.context.short_term_memory.add_message(response.message)
    return response
```

这段代码说明：

- 默认上下文来自 `ShortTermMemory`；
- `WorkingMemory` 在这里被注入；
- 工具 schema 默认在这里挂入；
- LLM 的 assistant 响应会被写回短期记忆；

换句话说，`call_llm()` 就是“Agent 与模型对话时的上下文拼装器”。

### 7.4 `execute_tools()`：把模型决策变成外部动作

```python
async def execute_tools(self, tool_calls: list[ToolCall]) -> list[ToolResult]:
    self.context.state.transition(AgentState.ACTING)

    results = await self.tool_executor.execute_batch(tool_calls)

    self.context.state.transition(AgentState.OBSERVING)

    for result in results:
        await self.context.short_term_memory.add_message(
            Message.tool(
                tool_call_id=result.tool_call_id,
                content=result.content,
                is_error=result.is_error,
            )
        )
```

这里最重要的设计不是“调用了工具”，而是：

- 状态从 `THINKING -> ACTING -> OBSERVING`；
- 工具结果被重新包装成 `Message.tool(...)` 写回短期记忆；
- 下一次 LLM 调用就能像看普通历史消息一样看到工具结果。

这就是 ReAct 循环里“Observe”环节的落地点。

### 7.5 Hooks：开放观测与插桩点

文件：`src/agent_harness/agent/hooks.py`

`AgentHooks` 用 `Protocol` 定义了若干回调点：

```python
class AgentHooks(Protocol):
    async def on_run_start(self, agent_name: str, input_text: str) -> None: ...
    async def on_step_start(self, agent_name: str, step: int) -> None: ...
    async def on_llm_call(self, agent_name: str, messages: list[Message]) -> None: ...
    async def on_tool_call(self, agent_name: str, tool_call: ToolCall) -> None: ...
    async def on_tool_result(self, agent_name: str, result: ToolResult) -> None: ...
    async def on_step_end(self, agent_name: str, step: int) -> None: ...
    async def on_run_end(self, agent_name: str, output: str) -> None: ...
    async def on_error(self, agent_name: str, error: Exception) -> None: ...
```

这和 `EventBus` 是互补关系：

- Hook 更像“同进程内、强类型、面向 Agent 生命周期”的扩展点；
- EventBus 更像“跨组件、命名空间化、弱耦合”的广播层。

### 7.6 ReActAgent：Think → Act → Observe

文件：`src/agent_harness/agent/react.py`

`ReActAgent.step()` 很短：

```python
async def step(self) -> StepResult:
    response = await self.call_llm()

    if response.has_tool_calls:
        tool_calls = response.message.tool_calls or []
        results = await self.execute_tools(tool_calls)
        return StepResult(
            thought=response.message.content,
            action=tool_calls,
            observation=results,
        )

    return StepResult(
        thought=None,
        response=response.message.content or "",
    )
```

它的特征非常鲜明：

- 一轮 step 先问 LLM；
- 如果 LLM 决定调工具，就执行工具并返回 observation；
- 如果没有工具调用，就把 LLM 文本当成最终答案结束。

也就是说，**ReActAgent 是最贴近“LLM 自主决策 + 外部动作”原始闭环的 Agent**。

在 `examples/react_agent.py` 里，使用方式也最直白：

```python
agent = ReActAgent(
    name="assistant",
    llm=llm,
    tools=[calculate, get_weather, get_population],
)

result = await agent.run(query)
```

### 7.7 PlanAgent：Plan → Execute → Synthesize

文件：`src/agent_harness/agent/planner.py`

`PlanAgent` 比 `ReActAgent` 多了一层显式计划结构：

```python
class PlanStep(BaseModel):
    id: str
    description: str
    status: str = "pending"
    result: str | None = None


class Plan(BaseModel):
    goal: str = ""
    steps: list[PlanStep] = Field(default_factory=list)
```

内部用 `_phase` 驱动多阶段状态：

```python
self._phase: str = "planning"  # "planning" | "executing" | "synthesizing"
```

`step()` 根据 `_phase` 分发：

```python
async def step(self) -> StepResult:
    if self._phase == "planning":
        return await self._plan_step()
    elif self._phase == "executing":
        return await self._execute_step()
    else:
        return await self._synthesize_step()
```

#### 7.7.1 planning 阶段

```python
response = await self.call_llm(tools=None)
content = response.message.content or ""
plan_data = parse_json_lenient(content)
self._plan = Plan(...)
self.context.working_memory.set("plan", self._plan.progress_summary)
self._phase = "executing"
```

这里有两个关键点：

- planning 阶段禁用工具，要求先产出计划；
- 计划生成后写入 `WorkingMemory`，供后续步骤继续看到。

#### 7.7.2 execution 阶段

```python
current.status = "in_progress"
self.context.working_memory.set("plan", self._plan.progress_summary)
self.context.working_memory.set("current_step", current.description)

exec_msg = Message.user(
    f"Execute this step of the plan:\n"
    f"Step [{current.id}]: {current.description}\n\n"
    f"Current plan progress:\n{self._plan.progress_summary}\n\n"
    f"Use tools if needed. When done, provide the result for this step."
)
await self.context.short_term_memory.add_message(exec_msg)
response = await self.call_llm()
```

这里能看到 PlanAgent 的思路：

- 不是把“整份计划”完全交给内部 Python 逻辑执行；
- 而是每次把当前 step 和计划进度重新组织成 prompt，再交给 LLM + Tool 组合去完成。

所以 PlanAgent 本质上是：

> 在 ReAct 外面再套一层“显式计划状态机”。

#### 7.7.3 synthesis 阶段

```python
synthesis_msg = Message.user(
    f"All steps of the plan are now complete.\n\n"
    f"Plan summary:\n{self._plan.progress_summary}\n\n"
    f"Provide a comprehensive final answer that synthesizes all the results."
)
await self.context.short_term_memory.add_message(synthesis_msg)
response = await self.call_llm(tools=None)
```

这里再次禁用工具，目的是收束输出，而不是继续探索。

#### 7.7.4 关于 `allow_replan`

构造函数里有 `allow_replan: bool = True`，但当前源码中没有真正使用这个标志去触发重新规划逻辑。

这意味着：

- 从 API 设计上，作者显然想支持 replanning；
- 但当前实现仍是“先规划一次，再顺序执行，再综合”。

学习框架时要区分**设计意图**和**当前版本已落地行为**。

### 7.8 ConversationalAgent：单步对话 Agent

文件：`src/agent_harness/agent/conversational.py`

实现最简单：

```python
async def step(self) -> StepResult:
    response = await self.call_llm(tools=None)
    return StepResult(
        response=response.message.content or "",
    )
```

它的定位是：

- 不使用工具；
- 一次 LLM 调用结束；
- 适合 chat、总结、改写、翻译、综合。

### 7.9 ReAct / Plan / Conversational 三种 Agent 的差异

这是用户要求重点展开的另一个点。

| Agent | 文件 | 核心循环 | 是否显式规划 | 是否调工具 | 典型场景 |
|---|---|---|---|---|---|
| `ReActAgent` | `agent/react.py` | Think → Act → Observe 循环 | 否 | 是 | 检索、查询、带工具问答 |
| `PlanAgent` | `agent/planner.py` | Plan → Execute → Synthesize | 是 | 执行阶段可用 | 复杂多步骤任务 |
| `ConversationalAgent` | `agent/conversational.py` | 单次 generate | 否 | 否 | 纯文本生成/总结 |

更细一点地说：

- **ReActAgent** 把控制权更多交给 LLM：模型决定何时调工具、调几个工具、何时结束。
- **PlanAgent** 在外层加入显式结构：先形成计划，再让每个子步骤各自走一个小型 ReAct/文本执行过程。
- **ConversationalAgent** 则刻意减少控制流，只保留“给上下文 -> 拿答案”。

这三个 Agent 不是强弱关系，而是三种不同的执行假设：

- 任务是否需要外部动作？
- 任务是否需要显式规划？
- 任务是否需要分多步收敛？

### 7.10 Message 如何流经 BaseAgent / LLM / Tool / Memory / Context

把前面各层串起来，一轮典型的 ReAct 执行链路如下：

#### 第 1 步：用户输入进入 `run()`

```python
input_msg = Message.user(input)
await self.context.short_term_memory.add_message(input_msg)
```

此时：

- 输入从纯字符串变成统一 `Message`；
- 进入 `ShortTermMemory`；
- `AgentContext.state` 转为 `THINKING`。

#### 第 2 步：`step()` 调用 `call_llm()`

```python
messages = await self.context.short_term_memory.get_context_messages()
working_msgs = await self.context.working_memory.get_context_messages()
messages = messages[:inject_idx] + working_msgs + messages[inject_idx:]
response = await self.llm.generate_with_events(messages, tools=tools, **kwargs)
```

此时：

- 短期记忆提供对话历史；
- 工作记忆提供当前执行状态；
- 工具 schema 一并传给模型；
- `BaseLLM` 发出 `llm.generate.*` 事件。

#### 第 3 步：Provider 适配请求并调用外部 LLM

OpenAI / Anthropic Provider 把 `Message` 翻译成各自 API 协议，得到 `LLMResponse`。

#### 第 4 步：assistant message 写回短期记忆

```python
await self.context.short_term_memory.add_message(response.message)
```

如果是工具调用型 assistant message，它的 `tool_calls` 也会一起存进去。

#### 第 5 步：如果有工具调用，进入 `execute_tools()`

```python
results = await self.tool_executor.execute_batch(tool_calls)
await self.context.short_term_memory.add_message(Message.tool(...))
```

此时：

- `ToolExecutor` 并发调用工具；
- 工具结果重新包装成 `Message.tool(...)`；
- 写回短期记忆。

#### 第 6 步：下一轮 `step()` 继续

下一次 `call_llm()` 时，模型看到的上下文里已经有：

- 原 system prompt；
- WorkingMemory；
- 用户问题；
- assistant 的 tool call 决策；
- tool result 消息；

于是它就能基于观察结果继续判断：

- 还要不要调工具；
- 还是已经可以直接回答。

这就是整个框架最核心的闭环。

---

## 第八章：编排系统

编排层位于 `src/agent_harness/orchestration/`：

- `pipeline.py`
- `dag.py`
- `router.py`
- `team.py`

这一层不改变单个 Agent 的行为，只负责**如何把多个 Agent 组织起来**。

### 8.1 Pipeline：线性串联

文件：`src/agent_harness/orchestration/pipeline.py`

`Pipeline` 的语义最简单：前一个 Agent 的输出作为下一个 Agent 的输入。

```python
class PipelineStep(BaseModel, arbitrary_types_allowed=True):
    agent: Any
    name: str = ""
    condition: Any | None = None
    transform: Any | None = None
```

执行逻辑：

```python
for step in self.steps:
    if step.condition and not step.condition(current_input):
        skipped.append(step.name)
        continue

    step_input = step.transform(current_input) if step.transform else current_input
    result = await step.agent.run(step_input)
    step_results[step.name] = result
    current_input = result.output
```

这意味着 Pipeline 关注的是**有序变换**：

- 可以跳过某步；
- 可以在进入下一步前改写输入；
- 每一步都能保留自己的 `AgentResult`。

在 `examples/multi_agent_pipeline.py` 里，`researcher -> writer` 就是典型 Pipeline。

### 8.2 DAGOrchestrator：批次执行与最大并行度

文件：`src/agent_harness/orchestration/dag.py`

这是用户要求重点展开的部分。

`DAGNode` 定义了节点：

```python
class DAGNode(BaseModel, arbitrary_types_allowed=True):
    id: str
    agent: Any
    dependencies: list[str] = Field(default_factory=list)
    input_transform: Any | None = None
```

`DAGResult` 会记录两个关键结果：

```python
class DAGResult(BaseModel):
    outputs: dict[str, AgentResult] = Field(default_factory=dict)
    execution_order: list[list[str]] = Field(default_factory=list)
```

这里的 `execution_order` 不是简单列表，而是 `list[list[str]]`，也就是**按批次记录**。

#### 8.2.1 为什么是“批次执行”而不是逐点调度

看主循环：

```python
while len(completed) < len(self.nodes):
    ready = [
        nid for nid, node in self.nodes.items()
        if nid not in completed
        and all(dep in completed for dep in node.dependencies)
    ]

    execution_order.append(ready)
```

也就是说，每一轮都会找出**当前所有依赖已满足的节点**，组成一个 batch。

然后：

```python
batch_results = await asyncio.gather(
    *(run_node(nid) for nid in ready)
)
```

这就是批次并行设计。

它的价值有三点：

1. **逻辑清晰**：一眼能看出 DAG 在哪几轮完成了哪些节点；
2. **并发充分**：同一批没有依赖关系的节点可以同时跑；
3. **结果可追踪**：`execution_order` 能直接还原调度过程。

#### 8.2.2 默认输入拼装策略

节点输入的生成规则是：

```python
if node.input_transform:
    node_input = node.input_transform(results)
elif node.dependencies:
    dep_outputs = [results[d].output for d in node.dependencies if d in results]
    node_input = "\n\n".join(dep_outputs)
else:
    node_input = input
```

这是一种非常实用的默认策略：

- 有自定义 `input_transform`，就完全按你写的来；
- 否则如果有依赖，就把依赖输出拼接起来；
- 根节点则直接吃初始输入。

所以 DAGOrchestrator 没有引入复杂的中间数据流 DSL，而是选择“够用且显式”。

#### 8.2.3 DAG 的安全性校验

在构造时会 `_validate_dag()`，用 DFS 检测：

- 依赖节点是否存在；
- 是否有环。

一旦发现环，就抛 `CyclicDependencyError`。

这很关键，因为一旦允许带环图进入运行期，`ready` 可能永远为空，调度器就会死锁。

### 8.3 AgentRouter：基于规则的动态分发

文件：`src/agent_harness/orchestration/router.py`

`Route` 有两个关键字段：

- `agent`
- `condition`

`condition` 可以是：

- callable：`Callable[[str], bool]`
- regex string

匹配逻辑：

```python
@staticmethod
def _matches(route: Route, input: str) -> bool:
    if callable(route.condition):
        return route.condition(input)
    if isinstance(route.condition, str):
        return bool(re.search(route.condition, input, re.IGNORECASE))
    return False
```

这说明 Router 定位得很明确：它不是 LLM Router，而是**规则路由器**。

适合：

- 把 code 类问题路给 coder；
- 把检索类问题路给 researcher；
- 用 fallback 兜底。

### 8.4 AgentTeam：多 Agent 协作模式

文件：`src/agent_harness/orchestration/team.py`

当前实现支持三种模式：

- `supervisor`
- `debate`
- `round_robin`

#### supervisor

- 所有 worker 并行跑原始任务；
- supervisor 再综合所有 worker 结果。

#### debate

- 各 Agent 独立作答；
- 最后由 `supervisor` 或最后一个 agent 充当 judge 综合。

#### round_robin

- Agent 按轮次依次接力；
- 每个人都看到此前累计输出；
- 最终输出取最后一名 Agent 最后一轮结果。

这个实现很“教学型”：

- 它不是复杂调度系统；
- 而是把几种常见多 Agent 协作范式直白地编码出来；
- 非常适合学习和扩展。

### 8.5 编排层和 Context 的关系

一个容易忽略的点是：这些编排器内部基本都是直接调用 `agent.run(...)`。

也就是说，它们**不自动帮你 fork context**。

这带来的含义是：

- 如果多个 Agent 需要共享或隔离上下文，应该在创建 Agent 时就配置好各自 `AgentContext`；
- Orchestration 层负责流程组织，而不是运行态依赖注入。

这是一个很干净的边界：

- Agent 处理“怎么做”；
- Orchestrator 处理“谁先做、谁后做、谁并行做”。

---

## 第九章：可观测性

可观测性相关代码位于 `src/agent_harness/tracing/` 和 `src/agent_harness/core/event.py`。

这一章需要分开看两条线：

1. **EventBus 事件流**：当前已经深度接入 Agent / LLM / Tool；
2. **Tracer / Span / Exporter**：已经实现，但当前版本更多是基础设施，尚未在 Agent 主流程里自动大面积埋点。

### 9.1 EventBus：当前版本最完整的观测主线

事件发射点分布如下：

- `BaseAgent.run()`：`agent.run.start` / `agent.step.start` / `agent.step.end` / `agent.run.end` / `agent.run.error`
- `BaseLLM.generate_with_events()`：`llm.generate.start` / `llm.generate.end` / `llm.generate.error`
- `ToolExecutor.execute()`：`tool.execute.start` / `tool.execute.end` / `tool.execute.error`

这已经足以拼出一条相当完整的执行时间线。

例如你可以订阅：

```python
bus.on("agent.*", handler)
bus.on("llm.generate.*", handler)
bus.on("tool.execute.*", handler)
```

就能拿到 Agent 调度、LLM 调用、Tool 执行的主要生命周期。

### 9.2 Span：结构化追踪单元

文件：`src/agent_harness/tracing/tracer.py`

`Span` 的字段定义如下：

```python
class Span(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    name: str
    kind: str = "internal"
    start_time: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    end_time: datetime | None = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    events: list[SpanEvent] = Field(default_factory=list)
    status: str = "ok"
    error_message: str | None = None
```

这套模型基本已经具备 OpenTelemetry 风格的最小必要结构：

- trace id：一次完整调用链；
- span id：单个工作单元；
- parent span id：树结构；
- attributes：结构化上下文；
- events：span 内局部事件；
- status / error：错误标记。

### 9.3 Tracer：生成和管理 Span

同文件里的核心逻辑是 `span()`：

```python
@asynccontextmanager
async def span(self, name: str, kind: str = "internal", **attributes: Any) -> AsyncIterator[Span]:
    if not self._enabled:
        yield Span(name=name, kind=kind)
        return

    trace_id = self._current_trace_id or uuid.uuid4().hex
    parent_span_id = self._current_span_id

    s = Span(
        trace_id=trace_id,
        parent_span_id=parent_span_id,
        name=name,
        kind=kind,
        attributes=attributes,
    )
```

进入 span 时：

- 继承当前 trace_id；
- 把当前 span_id 作为 parent；
- 更新 tracer 的“当前 span 上下文”。

退出 span 时：

```python
finally:
    s.finish()
    self._collector.add_span(s)
    self._current_trace_id = prev_trace_id
    self._current_span_id = prev_span_id
```

这就形成了一棵父子 span 树。

此外还有 `trace()` 装饰器，可以给 async 函数套上自动 span。

### 9.4 TraceCollector：收集已完成 spans

`TraceCollector` 目前非常轻量：

```python
class TraceCollector:
    def __init__(self) -> None:
        self._spans: list[Span] = []

    def add_span(self, span: Span) -> None:
        self._spans.append(span)

    def get_trace(self, trace_id: str) -> list[Span]:
        return [s for s in self._spans if s.trace_id == trace_id]
```

它本质上是一个内存聚合器。

这说明当前实现更偏向：

- 先把 span 数据模型和采集路径建立起来；
- 再由 exporter 决定怎么输出；
- 暂未引入复杂查询索引、采样、远程上报。

### 9.5 Exporter：如何把 Span 送出去

#### ConsoleExporter

文件：`src/agent_harness/tracing/exporters/console.py`

```python
class ConsoleExporter:
    def export(self, spans: list[Span]) -> None:
        for span in spans:
            self._print_span(span)
```

它会按 span 的 `kind` 输出颜色，并打印：

- 状态；
- 名称；
- duration；
- attributes；
- error；
- events。

虽然注释里提到“build tree”，但当前实现实际上是**按输入顺序平铺打印**，没有真正根据 parent-child 重建树缩进结构。

#### JsonFileExporter

文件：`src/agent_harness/tracing/exporters/json_file.py`

```python
class JsonFileExporter:
    def export(self, spans: list[Span]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with open(self._path, "a") as f:
            for span in spans:
                line = span.model_dump_json()
                f.write(line + "\n")
```

这是典型 JSON Lines 格式，优点是：

- 追加写简单；
- 方便后处理；
- 适合日志系统或离线分析。

### 9.6 当前源码里 Tracer 的接入状态

这里要特别说明一个“设计存在、接入有限”的事实：

- `AgentContext` 有 `tracer: Tracer | None` 字段；
- `Tracer` / `Span` / `Exporter` 都已经实现；
- 但 `BaseAgent.run()`、`BaseAgent.call_llm()`、`ToolExecutor.execute()` 当前并没有自动创建 span。

所以就当前版本而言：

- **事件观测已经是主路径**；
- **Tracer 更像已经搭好的基础设施，等待更深集成**。

这不是坏事，反而体现出架构的层次感：

- 事件系统先把“发生了什么”广播出去；
- tracing 系统再逐步把“发生在什么 trace/span 树里”补齐。

### 9.7 可观测层为什么要分 Event 和 Trace 两套

因为两者解决的问题不一样：

- **Event** 适合广播、告警、计数、UI 刷新；
- **Trace/Span** 适合重建一次调用的完整执行树。

前者是“事件流”，后者是“因果结构”。

成熟系统往往两者都需要。

---

## 第十章：扩展指引

这一章回答的问题是：如果你要把 `agent_harness` 当成基础框架继续扩展，应该从哪里下手。

### 10.1 扩展一个新 Tool

优先推荐最简单的方式：函数 + `@tool`。

参考 `examples/react_agent.py`：

```python
@tool
async def get_population(country: str) -> str:
    """Look up the approximate population of a country.

    Args:
        country: Name of the country.
    """
    populations = {
        "france": "68 million",
        "germany": "84 million",
    }
    return populations.get(country.lower(), f"Population data not available for {country}")
```

什么时候需要继承 `BaseTool`？

- 你需要自定义 `get_schema()`；
- 你不想通过函数签名自动推导；
- 你的工具背后有复杂状态或连接池。

### 10.2 扩展一个新 LLM Provider

你需要实现 `BaseLLM`：

- `generate()`
- `stream()`

同时处理三件事：

1. `Message -> provider request` 的转换；
2. `ToolSchema -> provider tool format` 的转换；
3. `provider response -> LLMResponse` 的转换。

可直接参考：

- `src/agent_harness/llm/openai_provider.py`
- `src/agent_harness/llm/anthropic_provider.py`

如果你的 Provider 协议和 OpenAI 类似，通常可以优先仿照 `OpenAIProvider`。
如果 system/tool 协议差异更大，则更接近 `AnthropicProvider` 的适配方式。

### 10.3 扩展一个新 Memory

如果你只是想替换长期记忆后端，通常只要实现 `BaseVectorStore`。

如果你要新增一种记忆范式，则实现 `BaseMemory`，重点考虑：

- `get_context_messages()` 是否自动注入；
- `query()` 是 recency-based 还是 relevance-based；
- 是否需要 `Message` 和 `content` 双入口；
- 生命周期是否需要初始化和清理。

### 10.4 扩展一个新 Agent

最自然的路径是继承 `BaseAgent` 并实现 `step()`。

例如你可以实现：

- 反思型 Agent：先回答，再批判，再修正；
- 辩论型单 Agent：内部模拟多轮自问自答；
- RAG Agent：每轮 step 先查 `LongTermMemory.query()`，再决定是否调工具。

设计建议：

- 把公共控制流交给 `run()`；
- 把策略差异尽量放进 `step()`；
- 用 `WorkingMemory` 把内部状态暴露给 LLM；
- 用 `ShortTermMemory` 持续保留可供模型回看的执行痕迹。

### 10.5 扩展编排层

如果现有 `Pipeline` / `DAGOrchestrator` / `AgentRouter` / `AgentTeam` 不够，可以考虑新增编排器。

推荐遵循当前风格：

- 输入输出尽量简单；
- 复用 `agent.run()`；
- 不把上下文管理和调度硬绑死；
- 尽量显式记录执行顺序和中间结果。

`DAGResult.execution_order` 就是一个很好的例子：它不是必须，但非常利于调试和解释系统行为。

### 10.6 扩展 Prompt 子系统

虽然 Prompt 不是用户要求重点章节，但它对扩展很有帮助。

相关文件：

- `src/agent_harness/prompt/template.py`
- `src/agent_harness/prompt/builder.py`
- `src/agent_harness/prompt/library.py`

`PromptTemplate` 支持 Jinja2 模板：

```python
template = PromptTemplate(
    template="You are a {{ role }}. Answer about {{ topic }}.",
    input_variables=["role", "topic"],
)
rendered = template.render(role="researcher", topic="AI safety")
```

`PromptBuilder` 则能流式拼接消息：

```python
messages = (
    PromptBuilder()
    .system("You are a helpful assistant.")
    .user("What is AI?")
    .build()
)
```

如果你要把 prompt 资产模块化，这一层是现成入口。

### 10.7 一个实际扩展策略

如果你准备把这个框架用于真实项目，我建议按以下顺序扩展：

1. 先定义自己的工具集；
2. 再封装统一 Provider 与重试策略；
3. 再决定是否需要长期记忆；
4. 最后再做复杂编排。

原因是：

- 工具和模型接入决定了“能做什么”；
- Memory 和 Orchestration 决定的是“如何规模化和复杂化”。

---

## 附录

### 附录 A：关键文件速查表

| 主题 | 文件 |
|---|---|
| Message 协议 | `src/agent_harness/core/message.py` |
| EventBus | `src/agent_harness/core/event.py` |
| 配置 | `src/agent_harness/core/config.py` |
| 异常 | `src/agent_harness/core/errors.py` |
| Tool 抽象 | `src/agent_harness/tool/base.py` |
| `@tool` 装饰器 | `src/agent_harness/tool/decorator.py` |
| Tool 执行器 | `src/agent_harness/tool/executor.py` |
| 短期记忆 | `src/agent_harness/memory/short_term.py` |
| 工作记忆 | `src/agent_harness/memory/working.py` |
| 长期记忆 | `src/agent_harness/memory/long_term.py` |
| Context | `src/agent_harness/context/context.py` |
| 状态机 | `src/agent_harness/context/state.py` |
| 变量作用域 | `src/agent_harness/context/variables.py` |
| BaseAgent | `src/agent_harness/agent/base.py` |
| ReActAgent | `src/agent_harness/agent/react.py` |
| PlanAgent | `src/agent_harness/agent/planner.py` |
| ConversationalAgent | `src/agent_harness/agent/conversational.py` |
| Pipeline | `src/agent_harness/orchestration/pipeline.py` |
| DAG | `src/agent_harness/orchestration/dag.py` |
| Router | `src/agent_harness/orchestration/router.py` |
| Team | `src/agent_harness/orchestration/team.py` |
| Tracer / Span | `src/agent_harness/tracing/tracer.py` |
| Console Exporter | `src/agent_harness/tracing/exporters/console.py` |
| JSON Exporter | `src/agent_harness/tracing/exporters/json_file.py` |

### 附录 B：推荐阅读顺序

如果你要真正读源码，建议按下面顺序：

1. `core/message.py`
2. `agent/base.py`
3. `agent/react.py`
4. `tool/decorator.py`
5. `tool/executor.py`
6. `memory/short_term.py` 与 `memory/working.py`
7. `context/context.py` 与 `context/state.py`
8. `llm/base.py` + 一个具体 provider
9. `agent/planner.py`
10. `orchestration/dag.py`
11. `tracing/tracer.py`

### 附录 C：示例程序与学习路径

- `examples/react_agent.py`：先理解单 Agent + Tool 的最小闭环。
- `examples/plan_and_execute.py`：理解显式规划与工作记忆注入。
- `examples/multi_agent_pipeline.py`：理解串行和并行编排差异。
- `examples/agent_team.py`：理解协作模式差异。
- `examples/deep_research.py`：理解整个框架如何组合成复杂系统。

### 附录 D：一句话总结各模块职责

- **Core**：定义公共协议。
- **LLM**：屏蔽模型厂商差异。
- **Tool**：把 Python 能力翻译给模型。
- **Memory**：分别管理对话、工作态、长期知识。
- **Context**：决定运行期依赖如何聚合与共享。
- **Agent**：定义单体推理策略。
- **Orchestration**：定义多 Agent 协作流程。
- **Tracing / Event**：让系统可观察、可调试。

### 附录 E：最后再回看一次整条链路

如果把整个框架压缩成一句话，可以这样理解：

> `agent_harness` 用 `Message` 统一通信协议，用 `AgentContext` 聚合运行时依赖，用 `BaseAgent` 承担控制流，用 `BaseLLM` 抽象模型调用，用 `ToolSchema + ToolExecutor` 把外部能力接给模型，用 `ShortTermMemory / WorkingMemory / LongTermMemory` 管理不同时间尺度的信息，再用 `Pipeline / DAG / Router / Team` 把多个 Agent 编排成更复杂的系统。

这也是你从“知道这个框架有哪些类”，走向“真正理解它为什么这么设计”的关键。
