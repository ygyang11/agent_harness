# Agent Harness Examples

Run any example with `python examples/<filename>.py` (requires `OPENAI_API_KEY` in env).

| File | Description |
|------|-------------|
| `react_agent.py` | ReActAgent with `@tool` decorated functions — the fundamental agent pattern |
| `plan_and_execute.py` | PlanAgent for multi-step task decomposition and execution |
| `multi_agent_pipeline.py` | Pipeline sequential chaining + DAG parallel orchestration |
| `agent_team.py` | AgentTeam with supervisor, debate, and round-robin collaboration modes |
| `deep_research.py` | Full deep research scenario — PlanAgent + parallel ReActAgents + synthesis |
