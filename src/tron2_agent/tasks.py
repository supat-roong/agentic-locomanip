"""Task definitions with ground-truth success checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .sim import Sim


def in_bin(sim: Sim, obj: str, bin_name: str = "bin") -> bool:
    b = next(b for b in sim.cfg.bins if b.name == bin_name)
    q, c = sim.body_pos(obj), sim.body_pos(bin_name)
    return bool(abs(q[0] - c[0]) < b.half[0] and abs(q[1] - c[1]) < b.half[1] and q[2] < c[2] + b.wall_h + 0.03)


@dataclass
class Task:
    name: str
    instruction: str
    check: Callable[[Sim], bool]


TASKS = {
    "red_to_bin": Task(
        "red_to_bin",
        "Put the red cube into the bin.",
        lambda s: in_bin(s, "red_cube"),
    ),
    "blue_to_bin": Task(
        "blue_to_bin",
        "Find the blue cube and put it in the bin.",
        lambda s: in_bin(s, "blue_cube"),
    ),
    "can_to_bin": Task(
        "can_to_bin",
        "Bring the green can to the bin.",
        lambda s: in_bin(s, "green_can"),
    ),
    "two_cubes": Task(
        "two_cubes",
        "Put both cubes (red and blue) into the bin.",
        lambda s: in_bin(s, "red_cube") and in_bin(s, "blue_cube"),
    ),
}
