"""6-DoF arm (AgileX Piper X on WFYG) kinematics: damped least-squares IK.

The end-effector frame is body `gripper_pick`; its local x-axis is the
approach direction (along the fingers) and the fingers close along local y.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

ARM_JOINTS = [f"arm{i}_Joint" for i in range(1, 7)]
GRIPPER_JOINTS = ["gripper1_Joint", "gripper2_Joint"]
EE_BODY = "gripper_pick"
STOW_Q = np.zeros(6)
GRIPPER_OPEN = 0.045  # per-finger opening (m), joint range 0..0.05
GRIPPER_CLOSED = 0.0


@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float
    ang_err: float
    ok: bool


class ArmIK:
    def __init__(self, model: mujoco.MjModel):
        self.m = model
        self.d = mujoco.MjData(model)
        self.qadr = np.array([model.joint(j).qposadr[0] for j in ARM_JOINTS])
        self.vadr = np.array([model.joint(j).dofadr[0] for j in ARM_JOINTS])
        self.lo = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 0]
        self.hi = model.jnt_range[[model.joint(j).id for j in ARM_JOINTS], 1]
        self.ee = model.body(EE_BODY).id

    def fk(self, qpos_full: np.ndarray, q_arm: np.ndarray | None = None):
        self.d.qpos[:] = qpos_full
        if q_arm is not None:
            self.d.qpos[self.qadr] = q_arm
        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        return self.d.xpos[self.ee].copy(), self.d.xmat[self.ee].reshape(3, 3).copy()

    def solve(
        self,
        qpos_full: np.ndarray,
        target_pos: np.ndarray,
        approach: np.ndarray | None = None,
        finger_axis: np.ndarray | None = None,
        q_init: np.ndarray | None = None,
        iters: int = 300,
        damping: float = 0.05,
        pos_tol: float = 0.005,
        ang_tol: float = 0.05,
    ) -> IKResult:
        """Solve arm joints so gripper_pick reaches target_pos (world frame).

        approach: desired world direction of the EE x-axis (e.g. [0,0,-1] = top-down).
        finger_axis: desired world direction of the EE y-axis (closing direction).
        The rest of the robot (base, legs) is frozen at qpos_full.
        """
        q = np.array(q_init if q_init is not None else qpos_full[self.qadr], dtype=float)
        jacp = np.zeros((3, self.m.nv))
        jacr = np.zeros((3, self.m.nv))
        target_pos = np.asarray(target_pos, float)
        for _ in range(iters):
            p, R = self.fk(qpos_full, q)
            e_pos = target_pos - p
            errs, rows = [e_pos], []
            mujoco.mj_jacBody(self.m, self.d, jacp, jacr, self.ee)
            Jp, Jr = jacp[:, self.vadr], jacr[:, self.vadr]
            rows.append(Jp)
            e_ang = np.zeros(3)
            if approach is not None:
                e_ang += np.cross(R[:, 0], approach / np.linalg.norm(approach))
            if finger_axis is not None:
                e_ang += 0.5 * np.cross(R[:, 1], finger_axis / np.linalg.norm(finger_axis))
            if approach is not None or finger_axis is not None:
                errs.append(0.5 * e_ang)
                rows.append(0.5 * Jr)
            e = np.concatenate(errs)
            J = np.vstack(rows)
            if np.linalg.norm(e_pos) < pos_tol and np.linalg.norm(e_ang) < ang_tol:
                break
            dq = J.T @ np.linalg.solve(J @ J.T + damping**2 * np.eye(J.shape[0]), e)
            q = np.clip(q + np.clip(dq, -0.2, 0.2), self.lo, self.hi)
        p, R = self.fk(qpos_full, q)
        pe = float(np.linalg.norm(target_pos - p))
        ae = 0.0
        if approach is not None:
            ae = float(np.arccos(np.clip(R[:, 0] @ approach / np.linalg.norm(approach), -1, 1)))
        return IKResult(q, pe, ae, pe < 0.01 and ae < 0.15)
