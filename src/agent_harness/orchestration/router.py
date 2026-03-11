"""Agent router for intent-based dispatching."""
from __future__ import annotations

import logging
import re
from typing import Any, Callable

from pydantic import BaseModel

from agent_harness.agent.base import BaseAgent, AgentResult

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
    """Routes requests to different agents based on rules.

    Routing strategies:
    - Callable condition: function(input) -> bool
    - String condition: regex pattern match

    Example:
        router = AgentRouter(
            routes=[
                Route(agent=coder, condition=lambda x: "code" in x.lower(), name="coder"),
                Route(agent=researcher, condition=r"search|find|look up", name="researcher"),
            ],
            fallback=general_agent,
        )
        result = await router.run("Write a Python function")
    """

    def __init__(
        self,
        routes: list[Route],
        fallback: BaseAgent | None = None,
    ) -> None:
        self.routes = routes
        self.fallback = fallback

    async def run(self, input: str) -> AgentResult:
        """Route the input to the matching agent."""
        for route in self.routes:
            if self._matches(route, input):
                logger.info("Router: matched route '%s'", route.name)
                return await route.agent.run(input)

        if self.fallback:
            logger.info("Router: no match, using fallback")
            return await self.fallback.run(input)

        from agent_harness.core.errors import OrchestrationError
        raise OrchestrationError(
            f"No route matched and no fallback configured. Input: {input[:100]}"
        )

    @staticmethod
    def _matches(route: Route, input: str) -> bool:
        if callable(route.condition):
            return route.condition(input)
        if isinstance(route.condition, str):
            return bool(re.search(route.condition, input, re.IGNORECASE))
        return False
