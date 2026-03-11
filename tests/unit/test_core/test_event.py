"""Tests for agent_harness.core.event — EventBus, Event, EventEmitter."""
from __future__ import annotations

import pytest

from agent_harness.core.event import Event, EventBus, EventEmitter


class TestEvent:
    def test_creation(self) -> None:
        evt = Event(type="agent.run.start", data={"key": "val"}, source="test")
        assert evt.type == "agent.run.start"
        assert evt.data == {"key": "val"}
        assert evt.source == "test"
        assert evt.timestamp is not None

    def test_defaults(self) -> None:
        evt = Event(type="x")
        assert evt.data == {}
        assert evt.source is None


class TestEventBus:
    @pytest.mark.asyncio
    async def test_emit_receive(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.on("test.event", handler)
        await bus.emit(Event(type="test.event", data={"n": 1}))

        assert len(received) == 1
        assert received[0].data == {"n": 1}

    @pytest.mark.asyncio
    async def test_wildcard_matching(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.on("agent.*", handler)

        await bus.emit(Event(type="agent.step.start"))
        await bus.emit(Event(type="agent.run.end"))
        await bus.emit(Event(type="tool.execute.start"))

        assert "agent.step.start" in received
        assert "agent.run.end" in received
        assert "tool.execute.start" not in received

    @pytest.mark.asyncio
    async def test_star_matches_everything(self) -> None:
        bus = EventBus()
        received: list[str] = []

        async def handler(event: Event) -> None:
            received.append(event.type)

        bus.on("*", handler)
        await bus.emit(Event(type="anything.here"))
        assert len(received) == 1

    @pytest.mark.asyncio
    async def test_off_removes_handler(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.on("x.y", handler)
        await bus.emit(Event(type="x.y"))
        assert len(received) == 1

        bus.off("x.y", handler)
        await bus.emit(Event(type="x.y"))
        assert len(received) == 1  # no new events

    @pytest.mark.asyncio
    async def test_off_nonexistent_handler(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass  # pragma: no cover

        # Should not raise
        bus.off("no.such", handler)

    @pytest.mark.asyncio
    async def test_multiple_handlers(self) -> None:
        bus = EventBus()
        results: list[int] = []

        async def h1(event: Event) -> None:
            results.append(1)

        async def h2(event: Event) -> None:
            results.append(2)

        bus.on("m", h1)
        bus.on("m", h2)
        await bus.emit(Event(type="m"))
        assert sorted(results) == [1, 2]

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        bus = EventBus()

        async def handler(event: Event) -> None:
            pass  # pragma: no cover

        bus.on("a", handler)
        bus.on("b", handler)
        bus.clear()

        received: list[Event] = []

        async def capture(event: Event) -> None:
            received.append(event)  # pragma: no cover

        await bus.emit(Event(type="a"))
        assert received == []

    @pytest.mark.asyncio
    async def test_handler_error_does_not_stop_others(self) -> None:
        bus = EventBus()
        results: list[int] = []

        async def bad_handler(event: Event) -> None:
            raise RuntimeError("boom")

        async def good_handler(event: Event) -> None:
            results.append(1)

        bus.on("e", bad_handler)
        bus.on("e", good_handler)
        await bus.emit(Event(type="e"))

        assert results == [1]

    @pytest.mark.asyncio
    async def test_no_handlers_does_not_error(self) -> None:
        bus = EventBus()
        await bus.emit(Event(type="orphan"))


class TestEventEmitter:
    @pytest.mark.asyncio
    async def test_emit_with_bus(self) -> None:
        bus = EventBus()
        received: list[Event] = []

        async def handler(event: Event) -> None:
            received.append(event)

        bus.on("comp.*", handler)

        emitter = EventEmitter()
        emitter.set_event_bus(bus)
        await emitter.emit("comp.done", result="ok")

        assert len(received) == 1
        assert received[0].data["result"] == "ok"

    @pytest.mark.asyncio
    async def test_emit_without_bus(self) -> None:
        emitter = EventEmitter()
        # Should not raise when no bus is attached
        await emitter.emit("some.event", key="val")

    @pytest.mark.asyncio
    async def test_source_defaults_to_class_name(self) -> None:
        bus = EventBus()
        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.on("*", handler)

        emitter = EventEmitter()
        emitter.set_event_bus(bus)
        await emitter.emit("test.event")

        assert captured[0].source == "EventEmitter"

    @pytest.mark.asyncio
    async def test_source_from_name_attribute(self) -> None:
        bus = EventBus()
        captured: list[Event] = []

        async def handler(event: Event) -> None:
            captured.append(event)

        bus.on("*", handler)

        class Named(EventEmitter):
            name = "my_component"

        emitter = Named()
        emitter.set_event_bus(bus)
        await emitter.emit("x.y")

        assert captured[0].source == "my_component"
