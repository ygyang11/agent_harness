"""Integration tests: orchestration patterns (Pipeline, AgentRouter)."""
from __future__ import annotations

import pytest

from agent_harness.agent.conversational import ConversationalAgent
from agent_harness.context.context import AgentContext
from agent_harness.orchestration import AgentRouter, Pipeline, PipelineStep, Route

from tests.conftest import MockLLM


def _make_agent(name: str, response: str, config) -> ConversationalAgent:
    """Helper: create a ConversationalAgent with a single canned response."""
    llm = MockLLM()
    llm.add_text_response(response)
    ctx = AgentContext.create(config)
    return ConversationalAgent(name=name, llm=llm, context=ctx)


# ---------------------------------------------------------------------------
# Pipeline tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_chains_two_agents(config):
    """Pipeline should run both agents sequentially, feeding output forward."""
    agent_a = _make_agent("summarizer", "Short summary of input.", config)
    agent_b = _make_agent("translator", "Resumen corto de la entrada.", config)

    pipeline = Pipeline(steps=[
        PipelineStep(agent=agent_a),
        PipelineStep(agent=agent_b),
    ])

    result = await pipeline.run("Some long article text")

    assert result.output == "Resumen corto de la entrada."
    assert "summarizer" in result.step_results
    assert "translator" in result.step_results
    assert result.step_results["summarizer"].output == "Short summary of input."
    assert len(result.skipped_steps) == 0


@pytest.mark.asyncio
async def test_pipeline_skips_step_when_condition_false(config):
    """Pipeline should skip a step whose condition returns False."""
    agent_a = _make_agent("always", "First output.", config)
    agent_b = _make_agent("conditional", "Should not run.", config)

    pipeline = Pipeline(steps=[
        PipelineStep(agent=agent_a),
        PipelineStep(agent=agent_b, name="conditional", condition=lambda x: "SKIP" not in x),
    ])

    # agent_a returns "First output." which does NOT contain "SKIP", so agent_b runs
    result = await pipeline.run("Go ahead")
    assert result.output == "Should not run."
    assert "conditional" not in result.skipped_steps

    # Now make agent_a return something containing "SKIP"
    agent_a2 = _make_agent("always2", "SKIP this step.", config)
    agent_b2 = _make_agent("conditional2", "Should not run.", config)
    pipeline2 = Pipeline(steps=[
        PipelineStep(agent=agent_a2),
        PipelineStep(agent=agent_b2, name="conditional2", condition=lambda x: "SKIP" not in x),
    ])
    result2 = await pipeline2.run("Go ahead")
    assert result2.output == "SKIP this step."
    assert "conditional2" in result2.skipped_steps


# ---------------------------------------------------------------------------
# AgentRouter tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_router_dispatches_by_regex(config):
    """AgentRouter should route to the agent whose regex matches the input."""
    math_agent = _make_agent("math", "42", config)
    weather_agent = _make_agent("weather", "Sunny and 25°C.", config)

    router = AgentRouter(routes=[
        Route(agent=math_agent, name="math", condition=r"calc|math|add|\d+\s*\+"),
        Route(agent=weather_agent, name="weather", condition=r"weather|forecast|temperature"),
    ])

    math_result = await router.run("Please calc 6 * 7")
    assert math_result.output == "42"

    weather_result = await router.run("What is the weather today?")
    assert weather_result.output == "Sunny and 25°C."


@pytest.mark.asyncio
async def test_router_uses_fallback_when_no_match(config):
    """AgentRouter should use the fallback agent when no route matches."""
    specific_agent = _make_agent("specific", "Specific answer.", config)
    fallback_agent = _make_agent("fallback", "I can help with that.", config)

    router = AgentRouter(
        routes=[Route(agent=specific_agent, name="specific", condition=r"^EXACT_MATCH$")],
        fallback=fallback_agent,
    )

    result = await router.run("Something completely different")
    assert result.output == "I can help with that."
