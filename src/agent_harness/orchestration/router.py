"""Agent router for intent-based dispatching."""
from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, TYPE_CHECKING

from pydantic import BaseModel

from agent_harness.agent.base import BaseAgent, AgentResult

if TYPE_CHECKING:
    from agent_harness.llm.base import BaseLLM

logger = logging.getLogger(__name__)


class Route(BaseModel, arbitrary_types_allowed=True):
    """A routing rule."""
    agent: Any  # BaseAgent
    name: str = ""
    condition: Any  # Callable[[str], bool] | str (regex pattern)
    description: str = ""

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = getattr(self.agent, 'name', 'unnamed')


class AgentRouter:
    """Routes requests to different agents based on rules and optional LLM routing.

    Routing strategies:
    - Callable condition: function(input) -> bool
    - String condition: regex pattern match
    - LLM-based: Let an LLM choose the best route(s) from descriptions

    Example:
        router = AgentRouter(
            routes=[
                Route(agent=coder, condition=lambda x: "code" in x.lower(), name="coder"),
                Route(agent=researcher, condition=r"search|find|look up", name="researcher"),
            ],
            fallback=general_agent,
            llm=openai_provider,       # Enable LLM routing
            llm_first=False,           # Try rules first, LLM as fallback
        )
        result = await router.run("Write a Python function")
    """

    def __init__(
        self,
        routes: list[Route],
        fallback: BaseAgent | None = None,
        llm: BaseLLM | None = None,
        llm_first: bool = False,
    ) -> None:
        self.routes = routes
        self.fallback = fallback
        self.llm = llm
        self.llm_first = llm_first

    async def run(self, input: str) -> AgentResult:
        """Route the input to the matching agent."""
        if self.llm_first and self.llm:
            # LLM routing first
            llm_result = await self._llm_route(input)
            if llm_result is not None:
                return llm_result

            # Fall back to rule matching
            for route in self.routes:
                if self._matches(route, input):
                    logger.info("Router: matched route '%s' (rule fallback)", route.name)
                    return await route.agent.run(input)
        else:
            # Rule matching first
            for route in self.routes:
                if self._matches(route, input):
                    logger.info("Router: matched route '%s'", route.name)
                    return await route.agent.run(input)

            # LLM routing fallback
            if self.llm:
                llm_result = await self._llm_route(input)
                if llm_result is not None:
                    return llm_result

        if self.fallback:
            logger.info("Router: no match, using fallback")
            return await self.fallback.run(input)

        from agent_harness.core.errors import OrchestrationError
        raise OrchestrationError(
            f"No route matched and no fallback configured. Input: {input[:100]}"
        )

    async def _llm_route(self, input: str) -> AgentResult | None:
        """Use LLM to select the best route(s)."""
        assert self.llm is not None

        from agent_harness.core.message import Message  # noqa: PLC0415
        from agent_harness.utils.json_utils import parse_json_lenient  # noqa: PLC0415

        route_descriptions = "\n".join(
            f"- {r.name}: {r.description}" if r.description else f"- {r.name}"
            for r in self.routes
        )

        prompt = (
            f"You are a routing assistant. Given a user request and a list of available agents, "
            f"select the most appropriate agent(s) to handle the request.\n\n"
            f"Available agents:\n{route_descriptions}\n\n"
            f"User request: {input}\n\n"
            f'Respond with a JSON object: {{"agents": ["agent_name1", "agent_name2"]}}\n'
            f"Select one or more agents. If none is suitable, respond with an empty list."
        )

        try:
            response = await self.llm.generate(
                [Message.system(prompt)],
                tools=None,
            )
            content = response.message.content or ""
            data = parse_json_lenient(content)

            if isinstance(data, dict):
                agent_names = data.get("agents", [])
            elif isinstance(data, list):
                agent_names = data
            else:
                return None

            if not agent_names:
                return None

            # Map names to routes
            route_map = {r.name: r for r in self.routes}
            matched_routes = [route_map[name] for name in agent_names if name in route_map]

            if not matched_routes:
                return None

            if len(matched_routes) == 1:
                logger.info("Router: LLM selected route '%s'", matched_routes[0].name)
                return await matched_routes[0].agent.run(input)

            # Multiple agents — run in parallel and synthesize
            logger.info("Router: LLM selected %d routes, running in parallel", len(matched_routes))
            tasks = [r.agent.run(input) for r in matched_routes]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect successful results
            outputs: list[str] = []
            first_result: AgentResult | None = None
            for route, result in zip(matched_routes, results):
                if isinstance(result, Exception):
                    logger.warning("Router: agent '%s' failed: %s", route.name, result)
                    continue
                outputs.append(f"[{route.name}]: {result.output}")
                if first_result is None:
                    first_result = result

            if not outputs:
                return None

            if len(outputs) == 1 and first_result is not None:
                return first_result

            # Synthesize multiple outputs
            synthesis_prompt = (
                f"Multiple agents provided answers to the request: {input}\n\n"
                + "\n\n".join(outputs) +
                "\n\nSynthesize these into a single comprehensive answer."
            )
            synthesis_response = await self.llm.generate(
                [Message.user(synthesis_prompt)], tools=None,
            )
            return AgentResult(
                output=synthesis_response.message.content or outputs[0],
                messages=[],
                steps=[],
                usage=synthesis_response.usage,
            )

        except Exception as e:
            logger.warning("Router: LLM routing failed: %s", e)
            return None

    @staticmethod
    def _matches(route: Route, input: str) -> bool:
        if callable(route.condition):
            return route.condition(input)
        if isinstance(route.condition, str):
            return bool(re.search(route.condition, input, re.IGNORECASE))
        return False
