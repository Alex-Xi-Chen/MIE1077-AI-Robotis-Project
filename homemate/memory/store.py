"""JSON-backed memory store.

Two files under the memory directory:

- ``episodes.jsonl`` — one JSON object per line, one line per user turn
- ``profile.json``   — a small rollup updated after every episode

The store is deliberately dependency-free (stdlib only) and deterministic
for tests: the only non-deterministic input is the timestamp, which can be
overridden via the ``now`` callable on ``MemoryStore``.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Episode:
    """One user turn — what was asked, what the robot did, what was sensed."""
    timestamp: str
    user_message: str
    detected_emotion: str | None
    robot_room_start: str | None
    robot_room_end: str | None
    owner_room: str | None
    tools_used: list[str] = field(default_factory=list)
    iot_changes: list[dict[str, Any]] = field(default_factory=list)
    dialogue: list[str] = field(default_factory=list)
    final_text: str = ""

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Episode":
        return cls(
            timestamp=d.get("timestamp", ""),
            user_message=d.get("user_message", ""),
            detected_emotion=d.get("detected_emotion"),
            robot_room_start=d.get("robot_room_start"),
            robot_room_end=d.get("robot_room_end"),
            owner_room=d.get("owner_room"),
            tools_used=list(d.get("tools_used", [])),
            iot_changes=list(d.get("iot_changes", [])),
            dialogue=list(d.get("dialogue", [])),
            final_text=d.get("final_text", ""),
        )


@dataclass
class Profile:
    """Stable rollup of everything we've learned about this owner.

    Kept small so it fits easily into the system prompt every turn.
    """
    first_seen: str | None = None
    last_seen: str | None = None
    total_episodes: int = 0
    emotion_counts: dict[str, int] = field(default_factory=dict)
    device_action_counts: dict[str, int] = field(default_factory=dict)
    recent_requests: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    MAX_RECENT_REQUESTS = 20

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Profile":
        return cls(
            first_seen=d.get("first_seen"),
            last_seen=d.get("last_seen"),
            total_episodes=int(d.get("total_episodes", 0)),
            emotion_counts=dict(d.get("emotion_counts", {})),
            device_action_counts=dict(d.get("device_action_counts", {})),
            recent_requests=list(d.get("recent_requests", [])),
            notes=list(d.get("notes", [])),
        )

    def update_from_episode(self, ep: Episode) -> None:
        if self.first_seen is None:
            self.first_seen = ep.timestamp
        self.last_seen = ep.timestamp
        self.total_episodes += 1
        if ep.detected_emotion:
            self.emotion_counts[ep.detected_emotion] = (
                self.emotion_counts.get(ep.detected_emotion, 0) + 1
            )
        for change in ep.iot_changes:
            key = f"{change.get('device_id')}:{change.get('action')}"
            self.device_action_counts[key] = self.device_action_counts.get(key, 0) + 1
        if ep.user_message:
            self.recent_requests.append(ep.user_message)
            if len(self.recent_requests) > self.MAX_RECENT_REQUESTS:
                self.recent_requests = self.recent_requests[-self.MAX_RECENT_REQUESTS:]


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------


def default_memory_dir() -> Path:
    """Resolve the on-disk memory directory.

    Honours ``HOMEMATE_MEMORY_DIR`` if set; otherwise defaults to
    ``<project_root>/data/memory`` (the project root is the parent of the
    ``homemate`` package).
    """
    env = os.environ.get("HOMEMATE_MEMORY_DIR", "").strip()
    if env:
        return Path(env).expanduser().resolve()
    pkg_root = Path(__file__).resolve().parents[2]
    return pkg_root / "data" / "memory"


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class MemoryStore:
    """File-backed memory.

    Thread-safety: not required — the agent runs in a single worker thread and
    only one turn is in flight at a time.
    """

    EPISODES_FILE = "episodes.jsonl"
    PROFILE_FILE = "profile.json"

    def __init__(self, root: Path | str | None = None,
                 *, now: Callable[[], str] = _utcnow_iso) -> None:
        self.root = Path(root) if root is not None else default_memory_dir()
        self.now = now
        self.root.mkdir(parents=True, exist_ok=True)
        self._profile = self._load_profile()

    # --- paths ---

    @property
    def episodes_path(self) -> Path:
        return self.root / self.EPISODES_FILE

    @property
    def profile_path(self) -> Path:
        return self.root / self.PROFILE_FILE

    # --- profile ---

    def _load_profile(self) -> Profile:
        if not self.profile_path.exists():
            return Profile()
        try:
            return Profile.from_json(json.loads(self.profile_path.read_text("utf-8")))
        except (json.JSONDecodeError, OSError):
            return Profile()

    def _save_profile(self) -> None:
        self.profile_path.write_text(
            json.dumps(self._profile.to_json(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def profile(self) -> Profile:
        return self._profile

    # --- episodes ---

    def record_episode(self, ep: Episode) -> None:
        if not ep.timestamp:
            ep.timestamp = self.now()
        with self.episodes_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ep.to_json(), ensure_ascii=False) + "\n")
        self._profile.update_from_episode(ep)
        self._save_profile()

    def all_episodes(self) -> list[Episode]:
        if not self.episodes_path.exists():
            return []
        out: list[Episode] = []
        with self.episodes_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Episode.from_json(json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return out

    def recent_episodes(self, n: int) -> list[Episode]:
        return self.all_episodes()[-n:]

    # --- reset ---

    def reset(self) -> None:
        for p in (self.episodes_path, self.profile_path):
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        self._profile = Profile()

    # --- brief for the LLM system prompt ---

    def memory_brief(self, max_episodes: int = 3) -> str:
        prof = self._profile
        if prof.total_episodes == 0:
            return ""

        lines: list[str] = []
        lines.append(f"- You have assisted this owner across {prof.total_episodes} prior turn(s).")
        if prof.emotion_counts:
            top_emotions = sorted(prof.emotion_counts.items(),
                                  key=lambda kv: -kv[1])[:3]
            parts = [f"{k} ({v})" for k, v in top_emotions]
            lines.append(f"- Emotions you have seen most: {', '.join(parts)}.")
        if prof.device_action_counts:
            top_devs = sorted(prof.device_action_counts.items(),
                              key=lambda kv: -kv[1])[:3]
            parts = [f"{k} x{v}" for k, v in top_devs]
            lines.append(f"- Frequent device actions: {', '.join(parts)}.")
        recent = self.recent_episodes(max_episodes)
        if recent:
            lines.append("- Recent requests:")
            for ep in recent:
                msg = (ep.user_message or "").strip().replace("\n", " ")
                if len(msg) > 80:
                    msg = msg[:77] + "..."
                tag = f" [emotion: {ep.detected_emotion}]" if ep.detected_emotion else ""
                lines.append(f"  * \"{msg}\"{tag}")
        if prof.notes:
            lines.append("- Owner notes:")
            for note in prof.notes[-3:]:
                lines.append(f"  * {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helper: build an Episode from a turn's TurnResult + before-state
# ---------------------------------------------------------------------------


def build_episode_from_turn(
    *,
    user_message: str,
    robot_room_start: str | None,
    robot_room_end: str | None,
    owner_room: str | None,
    tool_trace: Iterable[dict[str, Any]],
    spoken: Iterable[str],
    final_text: str,
    timestamp: str | None = None,
) -> Episode:
    """Project a TurnResult + world before/after into an :class:`Episode`."""
    tools_used: list[str] = []
    iot_changes: list[dict[str, Any]] = []
    detected_emotion: str | None = None

    for step in tool_trace:
        name = step.get("name", "")
        if name and name not in tools_used:
            tools_used.append(name)
        out = step.get("output", {}) or {}
        if name == "read_emotion" and out.get("ok"):
            detected_emotion = out.get("emotion")
        if name == "set_device" and out.get("ok"):
            inp = step.get("input", {}) or {}
            iot_changes.append({
                "device_id": inp.get("device_id"),
                "action": inp.get("action"),
                "state": out.get("state"),
            })

    return Episode(
        timestamp=timestamp or "",
        user_message=user_message,
        detected_emotion=detected_emotion,
        robot_room_start=robot_room_start,
        robot_room_end=robot_room_end,
        owner_room=owner_room,
        tools_used=tools_used,
        iot_changes=iot_changes,
        dialogue=list(spoken),
        final_text=final_text,
    )
