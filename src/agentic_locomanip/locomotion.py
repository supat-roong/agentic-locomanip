"""Wheel-leg locomotion using LimX's pretrained WF_TRON2A ONNX policy.

Mirrors controllers/WheelfootController.py from limxdynamics/tron2_rl_deploy_python:
policy at 50 Hz (decimation 10 on a 500 Hz loop), PD on legs, damping-only
velocity control on wheels. The policy was trained on WF (no arm), so on WFYG
it sees ~7 kg extra mass on top; keep the arm still while driving.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnxruntime as ort
import yaml

from .paths import POLICY_DIR

LEG_JOINTS = [
    "proximal_pitch_L",
    "proximal_roll_L",
    "proximal_yaw_L",
    "knee_L",
    "wheel_L",
    "proximal_pitch_R",
    "proximal_roll_R",
    "proximal_yaw_R",
    "knee_R",
    "wheel_R",
]
WHEEL_IDX = (4, 9)
YAW_IDX = (2, 7)
POS_OBS_IDX = [0, 1, 2, 3, 5, 6, 7, 8]  # wheel angles are not observed


def quat_rotate_inverse(q_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q_wxyz
    qv = np.array([x, y, z])
    return v * (2 * w * w - 1) - np.cross(qv, v) * 2 * w + qv * np.dot(qv, v) * 2


@dataclass
class LegState:
    q: np.ndarray  # (10,) joint positions in LEG_JOINTS order
    dq: np.ndarray  # (10,)
    quat_wxyz: np.ndarray  # base orientation
    gyro: np.ndarray  # base angular velocity in body frame


class WheelLegPolicy:
    def __init__(self, policy_dir=POLICY_DIR):
        cfg = yaml.safe_load(open(policy_dir / "params.yaml"))["PointfootCfg"]
        c = cfg["control"]
        self.kp, self.kd = c["leg_joint_stiffness"], c["leg_joint_damping"]
        self.kp_light, self.kd_light = c["light_joint_stiffness"], c["light_joint_damping"]
        self.kd_wheel = c["wheel_joint_damping"]
        self.tau_leg, self.tau_light = c["leg_joint_torque_limit"], c["light_joint_torque_limit"]
        self.tau_wheel = c["wheel_joint_torque_limit"]
        self.scale_pos, self.scale_vel = c["action_scale_pos"], c["action_scale_vel"]
        self.clip_act = cfg["normalization"]["clip_scales"]["clip_actions"]
        self.clip_obs = cfg["normalization"]["clip_scales"]["clip_observations"]
        self.hist_len = cfg["size"]["obs_history_length"]
        self.obs_size = cfg["size"]["policy_obs_size"]
        mx = cfg["commands"]["max"]
        self.cmd_max = np.array([mx["lin_vel_x"], mx["lin_vel_y"], mx["ang_vel_yaw"]])
        self.policy_hz = cfg["loop_frequency"] / c["decimation"]
        self.default_q = np.zeros(10)

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.enc = ort.InferenceSession(str(policy_dir / "encoder.onnx"), opts)
        self.pol = ort.InferenceSession(str(policy_dir / "policy.onnx"), opts)
        self.reset()

    def reset(self) -> None:
        self.actions = np.zeros(10)
        self.history = None
        self.cmd = np.zeros(3)

    def set_command(self, vx: float, wz: float) -> None:
        """Body-frame forward velocity (m/s) and yaw rate (rad/s)."""
        self.cmd = np.clip(np.array([vx, 0.0, wz]), -self.cmd_max, self.cmd_max)

    def _obs(self, s: LegState) -> np.ndarray:
        g = quat_rotate_inverse(s.quat_wxyz, np.array([0.0, 0.0, -1.0]))
        obs = np.concatenate(
            [
                s.gyro * 0.25,
                g,
                (s.q - self.default_q)[POS_OBS_IDX],
                s.dq * 0.05,
                self.actions,
            ]
        )
        return np.clip(obs, -self.clip_obs, self.clip_obs)

    def step_policy(self, s: LegState) -> None:
        """Run at policy_hz."""
        obs = self._obs(s)
        if self.history is None:
            self.history = np.tile(obs, self.hist_len)
        self.history = np.concatenate([self.history[self.obs_size :], obs])
        z = self.enc.run(None, {self.enc.get_inputs()[0].name: self.history.astype(np.float32)})
        z = np.asarray(z[0]).ravel()
        # deploy code scales the (already clipped) command by cmd_max again
        cmd_obs = self.cmd_max * self.cmd
        x = np.concatenate([z, obs, cmd_obs]).astype(np.float32)
        a = self.pol.run(None, {self.pol.get_inputs()[0].name: x})
        self.actions = np.clip(np.asarray(a[0]).ravel(), -self.clip_act, self.clip_act)

    def torques(self, s: LegState) -> np.ndarray:
        """Run at sim rate: PD toward the latest action."""
        tau = np.zeros(10)
        for i in range(10):
            if i in WHEEL_IDX:
                v = self.actions[i] * self.scale_vel
                tau[i] = np.clip(self.kd_wheel * (v - s.dq[i]), -self.tau_wheel, self.tau_wheel)
                continue
            kp, kd, lim = (
                (self.kp_light, self.kd_light, self.tau_light)
                if i in YAW_IDX
                else (self.kp, self.kd, self.tau_leg)
            )
            q_des = self.actions[i] * self.scale_pos + self.default_q[i]
            tau[i] = np.clip(kp * (q_des - s.q[i]) - kd * s.dq[i], -lim, lim)
        return tau
