"""Scene construction: WFYG robot + two tables, graspable objects and a bin."""

from __future__ import annotations

from dataclasses import dataclass, field

import mujoco
import numpy as np

from .paths import ROBOT_XML

TABLE_H = 0.62
TABLE_HALF = (0.30, 0.45)  # x, y half-extents of the table top
HEAD_PITCH_DEG = 30.0


@dataclass
class Table:
    name: str
    pos: tuple[float, float]  # centre (x, y)


@dataclass
class Obj:
    name: str
    color: str
    shape: str  # "box" | "cylinder"
    size: tuple[float, ...]  # box half-extents / cylinder (radius, half-height)
    pos: tuple[float, float]  # (x, y) world, resting on a table
    rgba: tuple[float, float, float, float]


@dataclass
class Bin:
    name: str
    pos: tuple[float, float]
    half: tuple[float, float] = (0.10, 0.12)
    wall_h: float = 0.05


@dataclass
class SceneCfg:
    tables: list[Table] = field(
        default_factory=lambda: [Table("table_A", (2.0, 0.0)), Table("table_B", (0.0, 2.2))]
    )
    objects: list[Obj] = field(
        default_factory=lambda: [
            Obj("red_cube", "red", "box", (0.02, 0.02, 0.02), (1.82, 0.10), (0.85, 0.1, 0.1, 1)),
            Obj("blue_cube", "blue", "box", (0.02, 0.02, 0.02), (1.82, -0.15), (0.1, 0.25, 0.85, 1)),
            Obj("green_can", "green", "cylinder", (0.022, 0.035), (2.05, 0.25), (0.1, 0.7, 0.2, 1)),
        ]
    )
    bins: list[Bin] = field(default_factory=lambda: [Bin("bin", (0.0, 2.02))])


def _add_cameras(spec: mujoco.MjSpec) -> None:
    th = np.radians(HEAD_PITCH_DEG)
    base = spec.body("base_Link")
    # Chest/head camera near the D435 mount, looking forward and down.
    base.add_camera(
        name="head",
        pos=[0.12, 0.0, 0.30],
        xyaxes=[0, -1, 0, np.sin(th), 0, np.cos(th)],
        fovy=70,
    )
    # Wrist camera looking along the approach axis (gripper_base_Link +z).
    spec.body("gripper_base_Link").add_camera(
        name="wrist", pos=[-0.07, 0.0, 0.0], xyaxes=[0, -1, 0, -1, 0, 0], fovy=80
    )
    spec.worldbody.add_camera(
        name="overview", pos=[-1.6, -1.6, 3.0], xyaxes=[0.707, -0.707, 0, 0.45, 0.45, 0.77], fovy=60
    )


def build_model(cfg: SceneCfg | None = None) -> tuple[mujoco.MjModel, SceneCfg]:
    cfg = cfg or SceneCfg()
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.option.noslip_iterations = 3
    w = spec.worldbody
    _add_cameras(spec)
    w.add_light(pos=[1, 1, 4], dir=[0, 0, -1], diffuse=[0.6, 0.6, 0.6], castshadow=False)

    for t in cfg.tables:
        b = w.add_body(name=t.name, pos=[t.pos[0], t.pos[1], 0])
        b.add_geom(
            name=f"{t.name}_top",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[TABLE_HALF[0], TABLE_HALF[1], 0.02],
            pos=[0, 0, TABLE_H - 0.02],
            rgba=[0.55, 0.4, 0.25, 1],
        )
        for sx in (-1, 1):
            for sy in (-1, 1):
                b.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[0.025, 0.025, (TABLE_H - 0.04) / 2],
                    pos=[sx * (TABLE_HALF[0] - 0.03), sy * (TABLE_HALF[1] - 0.03), (TABLE_H - 0.04) / 2],
                    rgba=[0.3, 0.2, 0.1, 1],
                )

    for bn in cfg.bins:
        b = w.add_body(name=bn.name, pos=[bn.pos[0], bn.pos[1], TABLE_H])
        hx, hy, h = bn.half[0], bn.half[1], bn.wall_h
        rgba = [0.8, 0.8, 0.8, 1]
        b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, hy, 0.005], pos=[0, 0, 0.005], rgba=rgba)
        for s in (-1, 1):
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.005, hy, h / 2], pos=[s * hx, 0, h / 2], rgba=rgba)
            b.add_geom(type=mujoco.mjtGeom.mjGEOM_BOX, size=[hx, 0.005, h / 2], pos=[0, s * hy, h / 2], rgba=rgba)

    for o in cfg.objects:
        half_h = o.size[2] if o.shape == "box" else o.size[1]
        b = w.add_body(name=o.name, pos=[o.pos[0], o.pos[1], TABLE_H + half_h + 0.001])
        b.add_freejoint(name=f"{o.name}_free")
        gtype = mujoco.mjtGeom.mjGEOM_BOX if o.shape == "box" else mujoco.mjtGeom.mjGEOM_CYLINDER
        b.add_geom(
            name=f"{o.name}_geom",
            type=gtype,
            size=list(o.size) + [0] * (3 - len(o.size)),
            rgba=list(o.rgba),
            mass=0.05,
            friction=[1.5, 0.05, 0.01],
            condim=4,
        )
    return spec.compile(), cfg
