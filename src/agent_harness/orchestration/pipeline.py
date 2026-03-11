"""Sequential pipeline orchestration."""
from __future__ import annotations

import logging
from typing import Any, Callable

from pydantic import BaseModel, Field

from agent_harness.agent.base import BaseAgent, AgentResult

logger = logging.getLogger(__name__)


class PipelineStep(BaseModel, arbitrary_types_allowed=True):
    """A single step in a pipeline."""
    agent: Any  # BaseAgent (pydantic can't validate ABC)
    name: str = ""
    condition: Any | None = None  # Callable[[str], bool]
    transform: Any | None = None  # Callable[[str], str] input transform

    def model_post_init(self, __context: Any) -> None:
        if not self.name:
            self.name = getattr(self.agent, 'name', 'unnamed')


class PipelineResult(BaseModel):
    """Result from a pipeline execution."""
    output: str
    step_results: dict[str, AgentResult] = Field(default_factory=dict)
    skipped_steps: list[str] = Field(default_factory=list)


class Pipeline:
    """Sequential agent pipeline.

    Each agent's output becomes the next agent's input.
    Supports conditional execution and input transformation.

    Example:
        pipeline = Pipeline(steps=[
            PipelineStep(agent=researcher),
            PipelineStep(agent=writer, transform=lambda x: f"Write article based on: {x}"),
        ])
        result = await pipeline.run("Latest AI news")
    """

    def __init__(self, steps: list[PipelineStep]) -> None:
        self.steps = steps

    async def run(self, input: str) -> PipelineResult:
        current_input = input
        step_results: dict[str, AgentResult] = {}
        skipped: list[str] = []

        for step in self.steps:
            # Check condition
            if step.condition and not step.condition(current_input):
                logger.info("Pipeline: skipping step '%s' (condition not met)", step.name)
                skipped.append(step.name)
                continue

            # Apply input transform
            step_input = step.transform(current_input) if step.transform else current_input

            logger.info("Pipeline: running step '%s'", step.name)
            result = await step.agent.run(step_input)
            step_results[step.name] = result
            current_input = result.output

        return PipelineResult(
            output=current_input,
            step_results=step_results,
            skipped_steps=skipped,
        )
