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
    bins: list[Bin] = field(default_factory=lambda: [Bin("bin", (0.0, 1.95))])


def randomized_cfg(seed: int) -> SceneCfg:
    """Default layout with object positions jittered (staying within arm reach of the table edge)."""
    import dataclasses

    rng = np.random.default_rng(seed)
    cfg = SceneCfg()
    cfg.objects = [
        dataclasses.replace(o, pos=(o.pos[0] + rng.uniform(-0.03, 0.03), o.pos[1] + rng.uniform(-0.05, 0.05)))
        for o in cfg.objects
    ]
    return cfg


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
    # Chase camera for videos: fixed world-frame offset from the robot, three-quarter view.
    # (The MJCF's own "track" camera follows the whole scene's COM, so tables pull it off the robot.)
    offset = np.array([-1.5, -1.5, 0.55])
    fwd = -offset / np.linalg.norm(offset)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    base.add_camera(
        name="follow",
        mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM,
        pos=offset.tolist(),
        xyaxes=[*right, *np.cross(right, fwd)],
        fovy=55,
    )
    spec.worldbody.add_camera(
        name="overview", pos=[-1.6, -1.6, 3.0], xyaxes=[0.707, -0.707, 0, 0.45, 0.45, 0.77], fovy=60
    )


# Livery matched to LimX's TRON 2 product renders: metallic gunmetal links,
# charcoal actuator housings, black tyres with a red hub ring, cyan LED strips.
_MATERIALS = {
    "tron_gunmetal": dict(rgba=[0.42, 0.44, 0.46, 1], specular=0.6, shininess=0.7, reflectance=0.05),
    "tron_charcoal": dict(rgba=[0.17, 0.18, 0.19, 1], specular=0.4, shininess=0.5),
    "tron_black": dict(rgba=[0.06, 0.06, 0.07, 1], specular=0.2, shininess=0.3),
    "tron_red": dict(rgba=[0.85, 0.12, 0.10, 1], specular=0.3, shininess=0.4),
    "tron_led": dict(rgba=[0.25, 0.80, 1.00, 1], emission=1.0),
}
_PART_MATERIAL = {
    "base_Link": "tron_gunmetal",
    "proximal_pitch": "tron_charcoal",
    "proximal_roll": "tron_gunmetal",
    "proximal_yaw": "tron_gunmetal",
    "knee": "tron_gunmetal",
    "wheel": "tron_black",
    "transition_Link": "tron_charcoal",
    "arm_base_Link": "tron_charcoal",
    "arm1_Link": "tron_charcoal",
    "arm2_Link": "tron_gunmetal",
    "arm3_Link": "tron_charcoal",
    "arm4_Link": "tron_gunmetal",
    "arm5_Link": "tron_charcoal",
    "arm6_Link": "tron_gunmetal",
    "gripper": "tron_charcoal",
    "camera_mount_Link": "tron_black",
    "radar_Link": "tron_black",
    "antenna": "tron_black",
}


def _visual_geom(body: mujoco.MjsBody, material: str, **kw) -> None:
    # geom rgba overrides material rgba in MuJoCo, so set both
    body.add_geom(contype=0, conaffinity=0, group=1, density=0, material=material, rgba=_MATERIALS[material]["rgba"], **kw)


def _paint_robot(spec: mujoco.MjSpec) -> None:
    """Colour the robot like the real TRON 2 (the official MJCF is plain grey)."""
    for name, attrs in _MATERIALS.items():
        spec.add_material(name=name, **attrs)
    for g in spec.geoms:
        if g.parent.name == "world":
            continue
        if g.contype == 0 and g.conaffinity == 0:
            mat = next((v for k, v in _PART_MATERIAL.items() if g.meshname.startswith(k)), "tron_gunmetal")
            g.material = mat
            g.rgba = _MATERIALS[mat]["rgba"]
        else:
            g.rgba = [0, 0, 0, 0]  # collision shapes stay in physics but are not drawn
    # Red hub ring on the outer face of each wheel.
    axle = [0.7071068, 0.7071068, 0, 0]
    for side, s in (("L", 1), ("R", -1)):
        wheel = spec.body(f"wheel_{side}_Link")
        cyl = mujoco.mjtGeom.mjGEOM_CYLINDER
        _visual_geom(wheel, "tron_red", type=cyl, size=[0.062, 0.001, 0], pos=[0, s * 0.0280, 0], quat=axle)
        _visual_geom(wheel, "tron_charcoal", type=cyl, size=[0.052, 0.001, 0], pos=[0, s * 0.0292, 0], quat=axle)
    # Cyan LED strips along the front edges of the torso.
    base = spec.body("base_Link")
    for s in (1, -1):
        _visual_geom(base, "tron_led", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.004, 0.005, 0.06], pos=[0.158, s * 0.085, 0.125])


def build_model(cfg: SceneCfg | None = None) -> tuple[mujoco.MjModel, SceneCfg]:
    cfg = cfg or SceneCfg()
    spec = mujoco.MjSpec.from_file(str(ROBOT_XML))
    spec.option.noslip_iterations = 3
    w = spec.worldbody
    _paint_robot(spec)
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
