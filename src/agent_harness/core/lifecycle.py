"""Component lifecycle protocols.

Define standard lifecycle interfaces that framework components can implement
for proper initialization and cleanup.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Initializable(Protocol):
    """Component that needs async initialization before use."""

    async def initialize(self) -> None:
        """Initialize the component. Called once before first use."""
        ...


@runtime_checkable
class Disposable(Protocol):
    """Component that holds resources requiring cleanup."""

    async def dispose(self) -> None:
        """Release resources. Called when the component is no longer needed."""
        ...
