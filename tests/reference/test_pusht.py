"""Reference parity tests comparing Rust PushT against Pymunk/Chipmunk2D."""

from __future__ import annotations

import ctypes
import os
from pathlib import Path

import numpy as np
import pymunk
import pytest
from pymunk.vec2d import Vec2d

pytestmark = pytest.mark.reference

_TARGET_DIR = Path(
    os.environ.get(
        "CARGO_TARGET_DIR", Path(__file__).resolve().parents[2] / "target"
    )
)
LIB_PATH = _TARGET_DIR / "release" / "libmatterstate_envs.so"


def _load_pusht() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(LIB_PATH))
    lib.ms_pusht_step.argtypes = [
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_float,
        ctypes.c_float,
    ]
    return lib


def _make_pymunk_pusht(
    agent_pos: tuple[float, float],
    block_pos: tuple[float, float],
    block_angle: float,
) -> tuple[pymunk.Space, pymunk.Body, pymunk.Body]:
    space = pymunk.Space()
    space.gravity = (0, 0)
    space.damping = 0.0
    walls = [
        pymunk.Segment(space.static_body, (5, 506), (5, 5), 2),
        pymunk.Segment(space.static_body, (5, 5), (506, 5), 2),
        pymunk.Segment(space.static_body, (506, 5), (506, 506), 2),
        pymunk.Segment(space.static_body, (5, 506), (506, 506), 2),
    ]
    space.add(*walls)

    agent = pymunk.Body(body_type=pymunk.Body.KINEMATIC)
    agent.position = agent_pos
    agent.friction = 1
    space.add(agent, pymunk.Circle(agent, 15))

    scale, length, mass = 30, 4, 1
    v1 = [
        (-length * scale / 2, scale),
        (length * scale / 2, scale),
        (length * scale / 2, 0),
        (-length * scale / 2, 0),
    ]
    v2 = [
        (-scale / 2, scale),
        (-scale / 2, length * scale),
        (scale / 2, length * scale),
        (scale / 2, scale),
    ]
    inertia = 2 * pymunk.moment_for_poly(mass, vertices=v1)
    block = pymunk.Body(mass, inertia)
    s1 = pymunk.Poly(block, v1)
    s2 = pymunk.Poly(block, v2)
    block.center_of_gravity = (s1.center_of_gravity + s2.center_of_gravity) / 2
    block.position = block_pos
    block.angle = block_angle
    block.friction = 1
    space.add(block, s1, s2)
    return space, agent, block


def _pymunk_step(
    space: pymunk.Space,
    agent: pymunk.Body,
    action: tuple[float, float],
) -> None:
    dt = 0.01
    act = Vec2d(*action)
    for _ in range(10):
        acc = 100.0 * (act - agent.position) + 20.0 * (
            Vec2d(0, 0) - agent.velocity
        )
        agent.velocity += acc * dt
        space.step(dt)


def test_pusht_free_motion_pd_parity() -> None:
    """When not contacting the block, Rust PD control matches Pymunk tightly."""
    lib = _load_pusht()
    space, agent, _ = _make_pymunk_pusht((80.0, 80.0), (350.0, 350.0), 0.0)
    state = np.array([80.0, 80.0, 0.0, 0.0, 350.0, 350.0, 0.0], np.float32)

    actions = [(150.0, 120.0), (200.0, 90.0), (100.0, 200.0)] * 5
    for ax, ay in actions:
        _pymunk_step(space, agent, (ax, ay))
        lib.ms_pusht_step(
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ax, ay
        )
        np.testing.assert_allclose(
            state[:2], [agent.position.x, agent.position.y], atol=1e-2
        )


def test_pusht_contact_direction_and_rotation_parity() -> None:
    """Off-center push moves the T-block in the same direction and sign."""
    lib = _load_pusht()
    space, agent, block = _make_pymunk_pusht(
        (290.0, 190.0), (256.0, 256.0), 0.0
    )
    # Pymunk block.position is at local (0, 0), CoM at (0, 45) -> world y = 301
    state = np.array([290.0, 190.0, 0.0, 0.0, 256.0, 301.0, 0.0], np.float32)

    for _ in range(15):
        _pymunk_step(space, agent, (290.0, 320.0))
        lib.ms_pusht_step(
            state.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), 290.0, 320.0
        )

    assert block.position.y > 256.0
    assert state[5] > 301.0
    # Both rotate in the same direction for a right-of-center upward push
    assert np.sign(np.sin(block.angle)) == np.sign(np.sin(state[6]))
