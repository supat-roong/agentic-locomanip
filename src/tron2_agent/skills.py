"""Robot skills exposed to the planner. Each returns a JSON-able dict.

Perception is privileged (simulator ground truth) in this first version;
`detect()` only reports items inside the head camera frustum so the planner
still has to look around / move to find things.
"""

from __future__ import annotations

from typing import Callable

import mujoco
import numpy as np

from .arm import GRIPPER_CLOSED, GRIPPER_OPEN, STOW_Q
from .scene import TABLE_H, TABLE_HALF
from .sim import Sim

CTRL_DT = 0.02  # skill control tick (s)
STANDOFF = 0.38  # base-to-item distance for manipulation
EDGE_CLEAR = 0.20  # min base-to-table-edge distance
ROBOT_RADIUS = 0.35
TOPDOWN = np.array([0.0, 0.0, -1.0])
IK_SEEDS = [
    np.array([0.0, 1.2, -1.0, 0.0, 0.8, 0.0]),
    np.array([0.0, 1.8, -1.5, 0.0, 1.2, 0.0]),
    np.array([0.0, 0.8, -0.6, 0.0, 0.6, 0.0]),
]


def wrap(a: float) -> float:
    return float((a + np.pi) % (2 * np.pi) - np.pi)


def ok(msg: str, **data) -> dict:
    return {"success": True, "message": msg, **data}


def fail(msg: str, **data) -> dict:
    return {"success": False, "message": msg, **data}


class Skills:
    def __init__(self, sim: Sim, on_tick: Callable[[Sim], None] | None = None):
        self.sim = sim
        self.on_tick = on_tick
        self.holding: str | None = None
        self._hold: tuple[float, float, float] | None = None  # base pose to keep during manipulation
        cfg = sim.cfg
        self.objects = {o.name: o for o in cfg.objects}
        self.bins = {b.name: b for b in cfg.bins}
        self.tables = {t.name: t for t in cfg.tables}

    # ---------------------------------------------------------------- helpers
    def _tick(self, n: int = 1) -> bool:
        """Advance n control ticks. Returns False if the robot fell."""
        steps = int(round(CTRL_DT / self.sim.dt))
        for _ in range(n):
            if self._hold is not None:
                self._station_keep()
            self.sim.step(steps)
            if self.on_tick:
                self.on_tick(self.sim)
            if self.sim.fallen():
                return False
        return True

    def _station_keep(self) -> None:
        hx, hy, hyaw = self._hold
        b = self.sim.base_pose()
        c, sn = np.cos(b.yaw), np.sin(b.yaw)
        fwd = c * (hx - b.x) + sn * (hy - b.y)
        self.sim.set_base_command(float(np.clip(1.5 * fwd, -0.2, 0.2)), float(np.clip(2.0 * wrap(hyaw - b.yaw), -0.4, 0.4)))

    def _hold_here(self) -> None:
        b = self.sim.base_pose()
        self._hold = (b.x, b.y, b.yaw)

    def _wait(self, seconds: float) -> bool:
        return self._tick(max(1, int(round(seconds / CTRL_DT))))

    def _to_robot_frame(self, p_world: np.ndarray) -> tuple[float, float]:
        b = self.sim.base_pose()
        dx, dy = p_world[0] - b.x, p_world[1] - b.y
        c, s = np.cos(b.yaw), np.sin(b.yaw)
        return float(c * dx + s * dy), float(-s * dx + c * dy)

    def _item_pos(self, name: str) -> np.ndarray | None:
        if name in self.objects or name in self.bins:
            return self.sim.body_pos(name)
        return None

    def _table_of(self, p: np.ndarray):
        for t in self.tables.values():
            if abs(p[0] - t.pos[0]) <= TABLE_HALF[0] + 0.02 and abs(p[1] - t.pos[1]) <= TABLE_HALF[1] + 0.02:
                return t
        return None

    def _in_head_view(self, p: np.ndarray) -> bool:
        m, d = self.sim.m, self.sim.d
        cid = m.camera("head").id
        R = d.cam_xmat[cid].reshape(3, 3)
        v = R.T @ (p - d.cam_xpos[cid])
        if v[2] >= -0.05:  # camera looks along -z
            return False
        fy = np.radians(m.cam_fovy[cid]) / 2
        h, w = self.sim.render_size
        fx = np.arctan(np.tan(fy) * w / h)
        return abs(v[0] / -v[2]) < np.tan(fx) and abs(v[1] / -v[2]) < np.tan(fy)

    def _contacts_with(self, body_name: str) -> set[str]:
        m, d = self.sim.m, self.sim.d
        bid = m.body(body_name).id
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = m.geom_bodyid[c.geom1], m.geom_bodyid[c.geom2]
            if b1 == bid:
                out.add(m.body(b2).name)
            elif b2 == bid:
                out.add(m.body(b1).name)
        return out

    def _is_grasped(self, name: str) -> bool:
        c = self._contacts_with(name)
        return "gripper1_Link" in c and "gripper2_Link" in c

    def _segment_hits_table(self, a: np.ndarray, b: np.ndarray) -> str | None:
        for t in self.tables.values():
            hx, hy = TABLE_HALF[0] + ROBOT_RADIUS - 0.05, TABLE_HALF[1] + ROBOT_RADIUS - 0.05
            for s in np.linspace(0, 1, 50):
                p = a + (b - a) * s
                if abs(p[0] - t.pos[0]) < hx and abs(p[1] - t.pos[1]) < hy:
                    return t.name
        return None

    # -------------------------------------------------------------- perception
    def get_state(self) -> dict:
        return ok("robot state", **self._state())

    def _state(self) -> dict:
        b = self.sim.base_pose()
        return dict(
            base={"x": round(b.x, 3), "y": round(b.y, 3), "yaw_deg": round(np.degrees(b.yaw), 1)},
            tilt_deg=round(b.tilt_deg, 1),
            holding=self.holding,
            gripper_opening_m=round(self.sim.gripper_opening(), 3),
            fallen=self.sim.fallen(),
            sim_time_s=round(self.sim.time, 2),
        )

    def detect(self) -> dict:
        """Items visible in the head camera, with world and robot-relative positions."""
        mujoco.mj_forward(self.sim.m, self.sim.d)
        items = []
        names = list(self.objects) + list(self.bins) + list(self.tables)
        for n in names:
            p = self.sim.body_pos(n)
            if n in self.tables:
                p = p + np.array([0, 0, TABLE_H])
            if not self._in_head_view(p):
                continue
            fwd, left = self._to_robot_frame(p)
            kind = "object" if n in self.objects else "bin" if n in self.bins else "table"
            entry = {
                "name": n,
                "kind": kind,
                "world_xyz": [round(float(v), 3) for v in p],
                "forward_m": round(fwd, 2),
                "left_m": round(left, 2),
                "distance_m": round(float(np.hypot(fwd, left)), 2),
            }
            if kind == "object":
                o = self.objects[n]
                entry.update(color=o.color, shape=o.shape)
                t = self._table_of(p)
                entry["on"] = t.name if t else ("gripper" if self.holding == n else "floor")
            if kind == "bin":
                t = self._table_of(p)
                entry["on"] = t.name if t else "floor"
            items.append(entry)
        return ok(f"{len(items)} items visible", items=items)

    # -------------------------------------------------------------- navigation
    def rotate(self, angle_deg: float) -> dict:
        b = self.sim.base_pose()
        return self.navigate_to(b.x, b.y, np.degrees(b.yaw) + angle_deg)

    def navigate_to(self, x: float, y: float, yaw_deg: float | None = None, tol: float = 0.05) -> dict:
        self._hold = None
        b = self.sim.base_pose()
        start, goal = np.array([b.x, b.y]), np.array([x, y], float)
        hit = self._segment_hits_table(start, goal) if np.linalg.norm(goal - start) > tol else None
        if hit:
            return fail(f"straight path to ({x:.2f},{y:.2f}) collides with {hit}; pick an intermediate waypoint")
        dist0 = float(np.linalg.norm(goal - start))
        max_ticks = int((8.0 + 5.0 * dist0) / CTRL_DT)
        phase = "drive" if dist0 > tol else "align"
        for _ in range(max_ticks):
            b = self.sim.base_pose()
            dvec = goal - np.array([b.x, b.y])
            dist = float(np.linalg.norm(dvec))
            if phase == "drive":
                if dist < tol:
                    phase = "align"
                    continue
                heading = np.arctan2(dvec[1], dvec[0])
                err = wrap(heading - b.yaw)
                reverse = dist < 0.3 and abs(err) > np.pi / 2  # small overshoot: back up
                if reverse:
                    err = wrap(err + np.pi)
                wz = np.clip(2.0 * err, -1.0, 1.0)
                vx = 0.0 if abs(err) > 0.35 else np.clip(1.2 * dist, 0.08, 0.7) * np.cos(err)
                self.sim.set_base_command(-vx if reverse else vx, wz)
            else:
                if yaw_deg is None:
                    break
                err = wrap(np.radians(yaw_deg) - b.yaw)
                if abs(err) < np.radians(3):
                    break
                self.sim.set_base_command(0.0, np.clip(2.0 * err, -1.0, 1.0) + 0.15 * np.sign(err))
            if not self._tick():
                self.sim.set_base_command(0, 0)
                return fail("robot fell over during navigation", state=self._state())
        self.sim.set_base_command(0.0, 0.0)
        self._wait(0.6)
        b = self.sim.base_pose()
        pos_err = float(np.hypot(b.x - x, b.y - y))
        yaw_err = abs(np.degrees(wrap(np.radians(yaw_deg) - b.yaw))) if yaw_deg is not None else 0.0
        st = {"x": round(b.x, 3), "y": round(b.y, 3), "yaw_deg": round(np.degrees(b.yaw), 1)}
        if pos_err < 0.12 and yaw_err < 8:
            return ok("arrived", base=st, pos_err_m=round(pos_err, 3))
        return fail("did not reach goal (timeout)", base=st, pos_err_m=round(pos_err, 3))

    def standoff_pose(self, name: str) -> tuple[np.ndarray, float] | str:
        p = self._item_pos(name)
        if p is None:
            return f"unknown item '{name}'"
        t = self._table_of(p)
        if t is None:
            return f"{name} is not on a table"
        b = self.sim.base_pose()
        tc = np.array(t.pos)
        # candidate edges: outward normal n, distance from item to edge
        cands = []
        for n, half, axis in (
            (np.array([1.0, 0]), TABLE_HALF[0], 0),
            (np.array([-1.0, 0]), TABLE_HALF[0], 0),
            (np.array([0, 1.0]), TABLE_HALF[1], 1),
            (np.array([0, -1.0]), TABLE_HALF[1], 1),
        ):
            d_edge = half - n[axis] * (p[axis] - tc[axis])
            robot_side = float(n @ (np.array([b.x, b.y]) - tc))
            cands.append((d_edge, -robot_side, n))
        cands.sort(key=lambda c: (round(c[0] / 0.1), c[1]))  # nearest edge, tie -> robot side
        d_edge, _, n = cands[0]
        dist = max(STANDOFF, d_edge + EDGE_CLEAR)
        if dist > 0.5:
            return f"{name} is {d_edge:.2f} m from the nearest table edge; too far to reach"
        goal = p[:2] + n * dist
        yaw = float(np.arctan2(-n[1], -n[0]))
        return goal, yaw

    def approach(self, name: str) -> dict:
        """Drive to a manipulation pose in front of an object or bin."""
        sp = self.standoff_pose(name)
        if isinstance(sp, str):
            return fail(sp)
        goal, yaw = sp
        n = -np.array([np.cos(yaw), np.sin(yaw)])
        pre = goal + n * 0.45
        b = self.sim.base_pose()
        here = np.array([b.x, b.y])
        if self._segment_hits_table(here, pre):
            wp = self._detour(here, pre)
            if wp is None:
                return fail(f"approach {name}: no collision-free route found")
            r = self.navigate_to(*wp)
            if not r["success"]:
                return fail(f"approach {name}: {r['message']}")
        r = self.navigate_to(pre[0], pre[1], np.degrees(yaw), tol=0.06)
        if not r["success"]:
            return fail(f"approach {name}: {r['message']}", **{k: v for k, v in r.items() if k != "success"})
        # final straight-in move, temporarily ignoring the table inflation
        r = self._drive_straight_to(goal, yaw)
        if not r["success"]:
            return r
        fwd, left = self._to_robot_frame(self.sim.body_pos(name))
        return ok(f"in front of {name}", item_forward_m=round(fwd, 3), item_left_m=round(left, 3), state=self._state())

    def _detour(self, a: np.ndarray, b: np.ndarray) -> np.ndarray | None:
        """Single waypoint around table corners that makes a->wp->b collision free."""
        best, best_len = None, np.inf
        m = ROBOT_RADIUS + 0.15
        for t in self.tables.values():
            for sx in (-1, 1):
                for sy in (-1, 1):
                    wp = np.array(t.pos) + np.array([sx * (TABLE_HALF[0] + m), sy * (TABLE_HALF[1] + m)])
                    if self._segment_hits_table(a, wp) or self._segment_hits_table(wp, b):
                        continue
                    length = np.linalg.norm(wp - a) + np.linalg.norm(b - wp)
                    if length < best_len:
                        best, best_len = wp, length
        return best

    def _drive_straight_to(self, goal: np.ndarray, yaw: float) -> dict:
        self._hold = None
        for _ in range(int(6.0 / CTRL_DT)):
            b = self.sim.base_pose()
            dvec = goal - np.array([b.x, b.y])
            c, s = np.cos(b.yaw), np.sin(b.yaw)
            fwd, lat = c * dvec[0] + s * dvec[1], -s * dvec[0] + c * dvec[1]
            if abs(fwd) < 0.02:
                break
            err = wrap(yaw - b.yaw) + 1.5 * np.clip(lat, -0.1, 0.1) * np.sign(fwd)
            self.sim.set_base_command(np.clip(1.0 * fwd, -0.25, 0.25), np.clip(2.0 * err, -0.6, 0.6))
            if not self._tick():
                return fail("robot fell over")
        self.sim.set_base_command(0, 0)
        self._wait(0.8)
        self._hold_here()
        return ok("aligned")

    # ------------------------------------------------------------ manipulation
    def _solve(self, target: np.ndarray, finger_axis: np.ndarray | None = None):
        best = None
        seeds = [self.sim.arm_target.copy(), *IK_SEEDS]
        for q0 in seeds:
            r = self.sim.ik.solve(self.sim.d.qpos.copy(), target, TOPDOWN, finger_axis, q_init=q0)
            if r.ok and (best is None or np.linalg.norm(r.q - self.sim.arm_target) < np.linalg.norm(best.q - self.sim.arm_target)):
                best = r
        return best

    def _move_joints(self, q_goal: np.ndarray, speed: float = 1.2) -> bool:
        q0 = self.sim.arm_target.copy()
        n = max(1, int(np.max(np.abs(q_goal - q0)) / (speed * CTRL_DT)))
        for a in np.linspace(0, 1, n + 1)[1:]:
            self.sim.arm_target = q0 + (q_goal - q0) * a
            if not self._tick():
                return False
        return self._wait(0.3)

    def _servo_ee(self, target: np.ndarray, finger_axis=None, speed=0.15, tol=0.008, timeout=6.0) -> dict:
        """Straight-line Cartesian move of gripper_pick to target, re-solving IK every tick
        against the live (balancing) base pose."""
        target = np.asarray(target, float)
        start = self.sim.ee_pos()
        L = float(np.linalg.norm(target - start))
        n_path = max(1, int(L / (speed * CTRL_DT)))
        for k in range(int(timeout / CTRL_DT)):
            wp = start + (target - start) * min(1.0, (k + 1) / n_path)
            # aim at the waypoint plus integral-ish correction of the tracking error
            r = self.sim.ik.solve(self.sim.d.qpos.copy(), wp, TOPDOWN, finger_axis, q_init=self.sim.arm_target, iters=30)
            dq = np.clip(r.q - self.sim.arm_target, -0.06, 0.06)
            self.sim.arm_target = self.sim.arm_target + dq
            if not self._tick():
                return fail("robot fell over")
            if k + 1 >= n_path and np.linalg.norm(self.sim.ee_pos() - target) < tol:
                break
        err = float(np.linalg.norm(self.sim.ee_pos() - target))
        if err < 0.025:
            return ok("reached", ee_err_m=round(err, 4))
        return fail("could not reach target with the gripper", ee_err_m=round(err, 3))

    def _finger_axis_for(self, name: str) -> np.ndarray:
        """Closing direction: the object's horizontal face axis closest to robot-lateral."""
        b = self.sim.base_pose()
        lateral = np.array([-np.sin(b.yaw), np.cos(b.yaw), 0.0])
        R = self.sim.d.body(name).xmat.reshape(3, 3)
        axes = [R[:, 0], R[:, 1]] if self.objects[name].shape == "box" else [lateral]
        best = max(axes, key=lambda a: abs(a @ lateral))
        best = best * np.sign(best @ lateral)
        best[2] = 0
        return best / np.linalg.norm(best)

    def _reach_check(self, p: np.ndarray, name: str) -> dict | None:
        fwd, left = self._to_robot_frame(p)
        if not (0.18 <= fwd <= 0.55 and abs(left) <= 0.25):
            return fail(
                f"{name} is out of arm reach (forward={fwd:.2f} m, left={left:.2f} m); "
                "reach needs forward 0.18-0.55 m and |left| <= 0.25 m. Call approach first."
            )
        return None

    def pick(self, name: str) -> dict:
        if name not in self.objects:
            return fail(f"unknown object '{name}'")
        if self.holding:
            return fail(f"already holding {self.holding}; place it first")
        p = self.sim.body_pos(name)
        if (e := self._reach_check(p, name)) is not None:
            return e
        if self._hold is None:
            self._hold_here()
        fa = self._finger_axis_for(name)
        o = self.objects[name]
        half_h = o.size[2] if o.shape == "box" else o.size[1]
        pre = p + np.array([0, 0, half_h + 0.12])
        grasp = p + np.array([0, 0, max(0.0, half_h - 0.025)])
        if self._solve(pre, fa) is None or self._solve(grasp, fa) is None:
            return fail(f"no IK solution to grasp {name} from here; re-approach")
        self.sim.grip_target = GRIPPER_OPEN
        q_pre = self._solve(pre, fa).q
        if not self._move_joints(q_pre):
            return fail("robot fell over")
        r = self._servo_ee(pre, fa, speed=0.2)
        if not r["success"]:
            return fail(f"pre-grasp: {r['message']}")
        r = self._servo_ee(grasp, fa, speed=0.08, tol=0.005)
        if not r["success"]:
            self._retreat()
            return fail(f"descend: {r['message']}")
        self.sim.grip_target = GRIPPER_CLOSED
        self._wait(0.8)
        z0 = self.sim.body_pos(name)[2]
        r = self._servo_ee(self.sim.ee_pos() + np.array([0, 0, 0.15]), fa, speed=0.12)
        lifted = self.sim.body_pos(name)[2] - z0
        if lifted > 0.08 and self._is_grasped(name):
            self.holding = name
            self._move_joints(STOW_Q, speed=0.8)
            self._wait(0.5)
            if not self._is_grasped(name):
                self.holding = None
                return fail(f"{name} slipped while stowing the arm")
            return ok(f"picked {name}", state=self._state())
        self.sim.grip_target = GRIPPER_OPEN
        self._retreat()
        return fail(f"grasp failed: {name} lifted only {lifted:.3f} m")

    def place(self, target: str) -> dict:
        """Place the held object into a bin (or onto the table at the bin's spot)."""
        if not self.holding:
            return fail("not holding anything")
        if target not in self.bins and target not in self.tables:
            return fail(f"unknown place target '{target}'")
        if target in self.bins:
            bn = self.bins[target]
            p = self.sim.body_pos(target) + np.array([0, 0, bn.wall_h + 0.10])
        else:
            b = self.sim.base_pose()
            p = np.array([b.x + 0.42 * np.cos(b.yaw), b.y + 0.42 * np.sin(b.yaw), TABLE_H + 0.08])
            if self._table_of(p) is None:
                return fail(f"no {target} surface in front of the robot; approach it first")
        if (e := self._reach_check(p, target)) is not None:
            return e
        if self._hold is None:
            self._hold_here()
        b = self.sim.base_pose()
        fa = np.array([-np.sin(b.yaw), np.cos(b.yaw), 0.0])
        sol = self._solve(p, fa)
        if sol is None:
            return fail(f"no IK solution to place into {target} from here; re-approach")
        if not self._move_joints(sol.q, speed=0.8):
            return fail("robot fell over")
        r = self._servo_ee(p, fa, speed=0.1)
        if not r["success"]:
            return fail(f"place: {r['message']}")
        obj = self.holding
        self.sim.grip_target = GRIPPER_OPEN
        self._wait(0.8)
        self.holding = None
        self._retreat()
        q = self.sim.body_pos(obj)
        inside = None
        if target in self.bins:
            c, bn = self.sim.body_pos(target), self.bins[target]
            inside = bool(abs(q[0] - c[0]) < bn.half[0] and abs(q[1] - c[1]) < bn.half[1] and q[2] < c[2] + bn.wall_h + 0.03)
        if inside is False:
            return fail(f"released {obj} but it is not inside {target}", object_xyz=[round(float(v), 3) for v in q])
        return ok(f"placed {obj} in/on {target}", state=self._state())

    def _retreat(self) -> None:
        self._servo_ee(self.sim.ee_pos() + np.array([0, 0, 0.10]), speed=0.15, timeout=2.0)
        self._move_joints(STOW_Q, speed=0.8)

    def stow_arm(self) -> dict:
        if not self._move_joints(STOW_Q, speed=0.8):
            return fail("robot fell over")
        return ok("arm stowed")

    def open_gripper(self) -> dict:
        self.sim.grip_target = GRIPPER_OPEN
        self._wait(0.6)
        self.holding = None
        return ok("gripper open")
