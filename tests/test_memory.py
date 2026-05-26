"""Tests for the long-term memory module.

No webcam, no Pygame, no Anthropic — these run with stdlib only and create
a fresh temp directory for each test (no shared state).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from tempfile import TemporaryDirectory

from homemate.action.skills import Skills
from homemate.cognition.llm_agent import MockLLM
from homemate.memory import Episode, MemoryStore, Profile, build_episode_from_turn
from homemate.perception.emotion import MockEmotionDetector
from homemate.world.apartment import Apartment
from homemate.world.entities import Owner, Robot, place_in_room
from homemate.world.iot import IoTNetwork


def _make_skills(seed: int = 1, emotion: str = "sad") -> Skills:
    rng = random.Random(seed)
    apt = Apartment()
    robot = Robot(0, 0)
    owner = Owner(0, 0)
    place_in_room(robot, apt, "living_room", rng)
    place_in_room(owner, apt, "bedroom", rng)
    iot = IoTNetwork.default()
    emo = MockEmotionDetector()
    emo.start()
    emo.inject(emotion)
    return Skills(apt, robot, owner, iot, emo)


def test_episode_round_trip() -> None:
    ep = Episode(
        timestamp="2026-05-25T10:00:00Z",
        user_message="open bedroom curtains",
        detected_emotion="sad",
        robot_room_start="living_room",
        robot_room_end="bedroom",
        owner_room="bedroom",
        tools_used=["find_owner", "read_emotion", "set_device"],
        iot_changes=[{"device_id": "curtain.bedroom", "action": "open",
                      "state": {"open": True}}],
        dialogue=["Hi, I'm here."],
        final_text="Done.",
    )
    blob = ep.to_json()
    assert json.dumps(blob)  # is JSON-serialisable
    assert Episode.from_json(blob) == ep


def test_profile_rolls_up_emotion_and_device_counts() -> None:
    prof = Profile()
    ep1 = Episode(timestamp="t1", user_message="hi", detected_emotion="sad",
                  robot_room_start=None, robot_room_end=None, owner_room=None,
                  iot_changes=[{"device_id": "curtain.bedroom", "action": "open"}])
    ep2 = Episode(timestamp="t2", user_message="again", detected_emotion="sad",
                  robot_room_start=None, robot_room_end=None, owner_room=None,
                  iot_changes=[{"device_id": "curtain.bedroom", "action": "open"},
                               {"device_id": "coffee.kitchen", "action": "brew"}])
    prof.update_from_episode(ep1)
    prof.update_from_episode(ep2)
    assert prof.total_episodes == 2
    assert prof.emotion_counts == {"sad": 2}
    assert prof.device_action_counts == {
        "curtain.bedroom:open": 2,
        "coffee.kitchen:brew": 1,
    }
    assert prof.first_seen == "t1" and prof.last_seen == "t2"
    assert prof.recent_requests == ["hi", "again"]


def test_memory_store_persists_across_instances() -> None:
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        s1 = MemoryStore(root)
        ep = Episode(timestamp="2026-05-25T10:00:00Z",
                     user_message="open curtains", detected_emotion="happy",
                     robot_room_start="living_room", robot_room_end="bedroom",
                     owner_room="bedroom",
                     iot_changes=[{"device_id": "curtain.bedroom",
                                   "action": "open"}])
        s1.record_episode(ep)
        # Reopen — profile should persist
        s2 = MemoryStore(root)
        assert s2.profile().total_episodes == 1
        assert s2.profile().emotion_counts == {"happy": 1}
        assert s2.all_episodes()[0].user_message == "open curtains"


def test_memory_brief_empty_when_no_history() -> None:
    with TemporaryDirectory() as tmp:
        assert MemoryStore(Path(tmp)).memory_brief() == ""


def test_memory_brief_includes_counts_and_recent() -> None:
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp))
        for i, emo in enumerate(("sad", "sad", "happy")):
            store.record_episode(Episode(
                timestamp=f"t{i}",
                user_message=f"request {i}",
                detected_emotion=emo,
                robot_room_start="living_room",
                robot_room_end="bedroom",
                owner_room="bedroom",
                iot_changes=[{"device_id": "curtain.bedroom", "action": "open"}],
            ))
        brief = store.memory_brief()
        assert "3 prior turn" in brief
        assert "sad (2)" in brief
        assert "curtain.bedroom:open x3" in brief
        assert "request 2" in brief  # most recent included


def test_memory_store_reset() -> None:
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp))
        store.record_episode(Episode(
            timestamp="t", user_message="x", detected_emotion="sad",
            robot_room_start=None, robot_room_end=None, owner_room=None,
        ))
        assert store.profile().total_episodes == 1
        store.reset()
        assert store.profile().total_episodes == 0
        assert store.all_episodes() == []


def test_build_episode_from_turn_extracts_emotion_and_iot() -> None:
    tool_trace = [
        {"name": "find_owner", "input": {}, "output": {"ok": True, "owner_room": "bedroom"}},
        {"name": "read_emotion", "input": {}, "output": {"ok": True, "emotion": "tired"}},
        {"name": "speak", "input": {"text": "Hi."}, "output": {"ok": True, "spoken": "Hi."}},
        {"name": "set_device",
         "input": {"device_id": "coffee.kitchen", "action": "brew"},
         "output": {"ok": True, "state": {"brewing": True}}},
        # A failed set_device should NOT be recorded as an iot change
        {"name": "set_device",
         "input": {"device_id": "nope", "action": "brew"},
         "output": {"ok": False, "error": "unknown device"}},
    ]
    ep = build_episode_from_turn(
        user_message="brew coffee",
        robot_room_start="living_room",
        robot_room_end="kitchen",
        owner_room="bedroom",
        tool_trace=tool_trace,
        spoken=["Hi."],
        final_text="Done.",
    )
    assert ep.detected_emotion == "tired"
    assert ep.tools_used == ["find_owner", "read_emotion", "speak", "set_device"]
    assert len(ep.iot_changes) == 1
    assert ep.iot_changes[0]["device_id"] == "coffee.kitchen"
    assert ep.dialogue == ["Hi."]


def test_mockllm_records_episode() -> None:
    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp))
        skills = _make_skills(emotion="tired")
        agent = MockLLM(skills, memory=store)
        agent.run_turn("brew coffee for me")
        eps = store.all_episodes()
        assert len(eps) == 1
        ep = eps[0]
        assert ep.detected_emotion == "tired"
        assert ep.robot_room_start == "living_room"
        # MockLLM moves to owner (bedroom) then to coffee.kitchen (kitchen)
        assert ep.robot_room_end == "kitchen"
        assert any(c["device_id"] == "coffee.kitchen" for c in ep.iot_changes)
        # profile updated
        prof = store.profile()
        assert prof.total_episodes == 1
        assert prof.emotion_counts.get("tired") == 1


def test_llm_agent_build_system_includes_memory_brief() -> None:
    """We don't have an API key here; construct without going through __init__'s
    Anthropic() call by patching the client lazily.
    """
    from homemate.cognition.llm_agent import LLMAgent

    with TemporaryDirectory() as tmp:
        store = MemoryStore(Path(tmp))
        store.record_episode(Episode(
            timestamp="t", user_message="open curtains", detected_emotion="happy",
            robot_room_start="living_room", robot_room_end="bedroom",
            owner_room="bedroom",
            iot_changes=[{"device_id": "curtain.bedroom", "action": "open"}],
        ))
        skills = _make_skills()

        # Bypass __init__ (which requires the anthropic package + API key) by
        # manually constructing the agent fields we need for build_system.
        agent = LLMAgent.__new__(LLMAgent)
        agent.skills = skills
        agent.memory = store
        agent.model = "claude-sonnet-4-6"
        agent.max_iters = 1
        agent.history = []

        sys_text = agent.build_system()
        assert "What you remember about this owner" in sys_text
        assert "1 prior turn" in sys_text
        assert "happy (1)" in sys_text
