"""Variable-link-length 2-link Reacher backed by the Rust core."""

from matterstate_envs import _rs

IMG: int = _rs.IMG


def step(
    t1: list[float],
    t2: list[float],
    a1: list[float],
    a2: list[float],
) -> tuple[list[float], list[float]]:
    """Advance joint angles ``(t1, t2)`` without mutating inputs."""
    return _rs.reacher_step(t1, t2, a1, a2)


def render(
    t1: list[float],
    t2: list[float],
    l1: list[float],
    l2: list[float],
) -> list[float]:
    """Render ``(B, 48, 48)`` flat float32 frames."""
    return _rs.reacher_morph_render(t1, t2, l1, l2)
