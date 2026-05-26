"""Long-term memory for HomeMate.

Lightweight JSON-on-disk store of past episodes (one per user turn) and a
profile rollup. A short ``memory_brief`` is injected into the LLM system
prompt each turn so the agent can personalise its replies.

Vector-store backends can be slotted in later behind the same
``MemoryStore`` interface.
"""

from __future__ import annotations

from .store import (
    Episode,
    MemoryStore,
    Profile,
    build_episode_from_turn,
    default_memory_dir,
)

__all__ = [
    "Episode",
    "MemoryStore",
    "Profile",
    "build_episode_from_turn",
    "default_memory_dir",
]
