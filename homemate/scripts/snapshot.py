"""Render one Pygame frame headlessly and save it to disk.

Used to generate the screenshot embedded in the README without having to
manually capture the live window.

Run with::

    python -m homemate.scripts.snapshot

The script forces SDL's dummy video driver so no window pops up; it then
constructs a realistic-looking demo scene (robot next to the owner in the
bedroom, bedroom curtains open, living-room lamp on, toaster cooking, plus
a sample dialogue exchange) and saves the rendered surface as PNG.
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

# Force headless SDL so we never pop a window.
os.environ["SDL_VIDEODRIVER"] = "dummy"


def main(out_path: Path | None = None) -> int:
    # Patch config BEFORE App() runs so the emotion detector and agent paths
    # use the deterministic mocks. We can't rely on env vars alone because
    # main.py's load_dotenv(override=True) will reload values from .env.
    from homemate import config
    config.USE_MOCK_EMOTION = True
    config.USE_MOCK_LLM = True

    from homemate.main import App  # imports pygame lazily
    from homemate.world.entities import place_in_room

    app = App()

    # Seed a realistic, visually interesting scene.
    rng = random.Random(42)
    place_in_room(app.owner, app.apt, "bedroom", rng)
    # Place the robot one tile away from the owner so the relationship reads.
    app.robot.x = max(1, app.owner.x - 1)
    app.robot.y = app.owner.y

    # IoT actuated states: open bedroom curtain, living-room lamp on,
    # toaster cooking mid-cycle, coffee maker idle.
    app.iot.act("curtain.bedroom", "open")
    app.iot.act("lamp.living_room", "on")
    app.iot.act("toaster.kitchen", "start", level=3)
    app.iot.get("toaster.kitchen").state["progress"] = 0.55

    # Inject an empathetic-looking dialogue thread. Kept short so the
    # rendered text fits inside the sidebar without clipping.
    app.skills.dialogue.extend([
        ("you",   "I'm tired. Brew coffee?"),
        ("robot", "On it. Starting coffee now."),
        ("you",   "Open the bedroom curtains too."),
        ("robot", "Opening curtains for you."),
        ("you",   "Thanks."),
        ("robot", "Any time. Rest up."),
    ])
    app.last_agent_summary = "Done. Detected emotion: tired."
    app.status_msg = "Agent done. Detected emotion: tired."
    app.emotion.inject("tired")

    # Render one frame and save it.
    app._draw()

    out_path = out_path or (Path(__file__).resolve().parents[2]
                            / "docs" / "images" / "pygame_demo.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import pygame
    pygame.image.save(app.screen, str(out_path))
    print(f"Saved screenshot to {out_path}")

    pygame.quit()
    return 0


if __name__ == "__main__":
    sys.exit(main())
