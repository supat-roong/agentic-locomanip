"""Simulation runtime: 1 kHz physics, 50 Hz locomotion policy, arm/gripper PD."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

from .arm import ARM_JOINTS, GRIPPER_JOINTS, GRIPPER_OPEN, STOW_Q, ArmIK
from .locomotion import LEG_JOINTS, LegState, WheelLegPolicy
from .paths import check_assets
from .scene import SceneCfg, build_model

ARM_KP, ARM_KD, ARM_TAU = 60.0, 3.0, 30.0
GRIP_KP, GRIP_KD, GRIP_FORCE = 400.0, 10.0, 10.0


@dataclass
class BasePose:
    x: float
    y: float
    yaw: float
    z: float
    tilt_deg: float


class Sim:
    def __init__(self, cfg: SceneCfg | None = None, render_size=(480, 640), start_yaw: float = 0.0):
        self.start_yaw = start_yaw
        check_assets()
        self.model, self.cfg = build_model(cfg)
        self.m = self.model
        self.d = mujoco.MjData(self.m)
        self.dt = self.m.opt.timestep
        m = self.m
        self.leg_q = [m.joint(n + "_Joint").qposadr[0] for n in LEG_JOINTS]
        self.leg_v = [m.joint(n + "_Joint").dofadr[0] for n in LEG_JOINTS]
        self.leg_u = [m.actuator(n).id for n in LEG_JOINTS]
        self.arm_q = [m.joint(j).qposadr[0] for j in ARM_JOINTS]
        self.arm_v = [m.joint(j).dofadr[0] for j in ARM_JOINTS]
        self.arm_u = [m.actuator(j.replace("_Joint", "")).id for j in ARM_JOINTS]
        self.grip_q = [m.joint(j).qposadr[0] for j in GRIPPER_JOINTS]
        self.grip_v = [m.joint(j).dofadr[0] for j in GRIPPER_JOINTS]
        self.grip_u = [m.actuator(j.replace("_Joint", "")).id for j in GRIPPER_JOINTS]
        self.policy = WheelLegPolicy()
        self.decimation = int(round(1.0 / (self.dt * self.policy.policy_hz)))
        self.ik = ArmIK(m)
        self.render_size = render_size
        self._renderer = None
        self.reset()

    # ------------------------------------------------------------------ setup
    def reset(self) -> None:
        mujoco.mj_resetData(self.m, self.d)
        self.d.qpos[self.arm_q] = STOW_Q
        self.d.qpos[3:7] = [np.cos(self.start_yaw / 2), 0, 0, np.sin(self.start_yaw / 2)]
        mujoco.mj_forward(self.m, self.d)
        wheel = self.m.body("wheel_L_Link").id
        r = max(
            self.m.geom_size[g][0]
            for g in range(self.m.ngeom)
            if self.m.geom_bodyid[g] == wheel and self.m.geom_contype[g]
        )
        self.d.qpos[2] += -(self.d.xpos[wheel][2] - r) + 0.005
        mujoco.mj_forward(self.m, self.d)
        self.arm_target = STOW_Q.copy()
        self.grip_target = GRIPPER_OPEN
        self.policy.reset()
        self.n_steps = 0
        self.step_seconds(2.0)  # settle into balance

    # --------------------------------------------------------------- control
    def _leg_state(self) -> LegState:
        d = self.d
        return LegState(d.qpos[self.leg_q].copy(), d.qvel[self.leg_v].copy(), d.qpos[3:7].copy(), d.qvel[3:6].copy())

    def step(self, n: int = 1) -> None:
        d = self.d
        for _ in range(n):
            s = self._leg_state()
            if self.n_steps % self.decimation == 0:
                self.policy.step_policy(s)
            d.ctrl[self.leg_u] = self.policy.torques(s)
            e = self.arm_target - d.qpos[self.arm_q]
            tau = ARM_KP * e - ARM_KD * d.qvel[self.arm_v] + d.qfrc_bias[self.arm_v]  # + gravity comp
            d.ctrl[self.arm_u] = np.clip(tau, -ARM_TAU, ARM_TAU)
            g_des = np.array([self.grip_target, -self.grip_target])
            f = GRIP_KP * (g_des - d.qpos[self.grip_q]) - GRIP_KD * d.qvel[self.grip_v]
            d.ctrl[self.grip_u] = np.clip(f, -GRIP_FORCE, GRIP_FORCE)
            mujoco.mj_step(self.m, d)
            self.n_steps += 1

    def step_seconds(self, t: float) -> None:
        self.step(int(round(t / self.dt)))

    def set_base_command(self, vx: float, wz: float) -> None:
        self.policy.set_command(vx, wz)

    # ----------------------------------------------------------------- state
    @property
    def time(self) -> float:
        return self.d.time

    def base_pose(self) -> BasePose:
        R = self.d.xmat[self.m.body("base_Link").id].reshape(3, 3)
        return BasePose(
            float(self.d.qpos[0]),
            float(self.d.qpos[1]),
            float(np.arctan2(R[1, 0], R[0, 0])),
            float(self.d.qpos[2]),
            float(np.degrees(np.arccos(np.clip(R[2, 2], -1, 1)))),
        )

    def fallen(self) -> bool:
        p = self.base_pose()
        return p.tilt_deg > 35 or p.z < 0.45

    def body_pos(self, name: str) -> np.ndarray:
        return self.d.body(name).xpos.copy()

    def ee_pos(self) -> np.ndarray:
        return self.d.body("gripper_pick").xpos.copy()

    def arm_q_now(self) -> np.ndarray:
        return self.d.qpos[self.arm_q].copy()

    def gripper_opening(self) -> float:
        return float(self.d.qpos[self.grip_q[0]] - self.d.qpos[self.grip_q[1]])

    # ---------------------------------------------------------------- render
    def render(self, camera: str = "head") -> np.ndarray:
        if self._renderer is None:
            h, w = self.render_size
            self._renderer = mujoco.Renderer(self.m, height=h, width=w)
        self._renderer.update_scene(self.d, camera=camera)
        return self._renderer.render().copy()

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
