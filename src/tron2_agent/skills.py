"""Robot skills exposed to the planner. Each returns a JSON-able dict.

Perception is privileged (simulator ground truth) in this first version;
`detect()` only reports items inside the head camera frustum so the planner
still has to look around / move to find things.
"""

from __future__ import annotations

from collections.abc import Callable

import mujoco
import numpy as np

from .arm import GRIPPER_CLOSED, GRIPPER_OPEN, STOW_Q
from .scene import TABLE_H, TABLE_HALF
from .sim import Sim

CTRL_DT = 0.02  # skill control tick (s)
STANDOFF = 0.38  # base-to-item distance for manipulation
EDGE_CLEAR = 0.20  # min base-to-table-edge distance
ROBOT_RADIUS = 0.35
NEAR_EDGE_MIN = 0.15  # closest the base centre may stop to a table edge
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
        self.seen: dict[str, np.ndarray] = {}  # last-seen world position of each item

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

    @staticmethod
    def _box_dist(p: np.ndarray, centre, hx: float, hy: float) -> float:
        """Chebyshev-style distance to an axis-aligned box (negative inside)."""
        return max(abs(p[0] - centre[0]) - hx, abs(p[1] - centre[1]) - hy)

    def _segment_hits_table(self, a: np.ndarray, b: np.ndarray) -> str | None:
        """Table whose inflated footprint the straight segment a->b enters.

        If the robot already stands inside an inflated footprint (e.g. right
        after manipulating at a table) only motion that goes deeper is blocked.
        """
        hx, hy = TABLE_HALF[0] + ROBOT_RADIUS - 0.05, TABLE_HALF[1] + ROBOT_RADIUS - 0.05
        for t in self.tables.values():
            d0 = self._box_dist(a, t.pos, hx, hy)
            limit = min(0.0, d0 - 0.01)
            for s_ in np.linspace(0, 1, 50)[1:]:
                if self._box_dist(a + (b - a) * s_, t.pos, hx, hy) < limit:
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

    def _observe(self) -> list[str]:
        """Update item memory from the head camera; return names currently visible."""
        mujoco.mj_forward(self.sim.m, self.sim.d)
        visible = []
        for n in list(self.objects) + list(self.bins) + list(self.tables):
            p = self.sim.body_pos(n)
            if n in self.tables:
                p = p + np.array([0, 0, TABLE_H])
            if self._in_head_view(p):
                self.seen[n] = p
                visible.append(n)
        if self.holding:
            self.seen[self.holding] = self.sim.body_pos(self.holding)
        return visible

    def _describe(self, n: str, p: np.ndarray) -> dict:
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
        if kind == "table":
            entry["x_range"] = [round(float(p[0]) - TABLE_HALF[0], 2), round(float(p[0]) + TABLE_HALF[0], 2)]
            entry["y_range"] = [round(float(p[1]) - TABLE_HALF[1], 2), round(float(p[1]) + TABLE_HALF[1], 2)]
        else:
            t = self._table_of(p)
            if t is not None and self.holding != n:
                tx, ty = t.pos
                edges = {
                    "x_min": (p[0] - (tx - TABLE_HALF[0]), f"x = {tx - TABLE_HALF[0]:.2f}"),
                    "x_max": ((tx + TABLE_HALF[0]) - p[0], f"x = {tx + TABLE_HALF[0]:.2f}"),
                    "y_min": (p[1] - (ty - TABLE_HALF[1]), f"y = {ty - TABLE_HALF[1]:.2f}"),
                    "y_max": ((ty + TABLE_HALF[1]) - p[1], f"y = {ty + TABLE_HALF[1]:.2f}"),
                }
                side = min(edges, key=lambda k: edges[k][0])
                entry["closest_table_edge"] = {
                    "side": side,
                    "edge": edges[side][1],
                    "item_to_edge_m": round(float(edges[side][0]), 2),
                }
            entry["on"] = "gripper" if self.holding == n else (t.name if t else "floor")
        return entry

    def detect(self) -> dict:
        """Items visible in the head camera now, plus remembered last-seen items."""
        visible = self._observe()
        items = [self._describe(n, self.seen[n]) for n in visible]
        remembered = [self._describe(n, p) for n, p in self.seen.items() if n not in visible]
        msg = f"{len(items)} items visible"
        if remembered:
            msg += f", {len(remembered)} remembered from earlier views"
        return ok(msg, items=items, remembered=remembered)

    # -------------------------------------------------------------- navigation
    def rotate(self, angle_deg: float) -> dict:
        b = self.sim.base_pose()
        return self.navigate_to(b.x, b.y, np.degrees(b.yaw) + angle_deg)

    def _near_table(self, p: np.ndarray):
        """(table, outward edge normal, distance to that edge) if p is inside a table's inflated zone."""
        hx, hy = TABLE_HALF[0] + ROBOT_RADIUS - 0.05, TABLE_HALF[1] + ROBOT_RADIUS - 0.05
        for t in self.tables.values():
            if self._box_dist(p, t.pos, hx, hy) < 0:
                dx, dy = p[0] - t.pos[0], p[1] - t.pos[1]
                ex, ey = abs(dx) - TABLE_HALF[0], abs(dy) - TABLE_HALF[1]
                if ex >= ey:
                    return t, np.array([np.sign(dx) or 1.0, 0.0]), ex
                return t, np.array([0.0, np.sign(dy) or 1.0]), ey
        return None

    def navigate_to(self, x: float, y: float, yaw_deg: float | None = None, tol: float = 0.05) -> dict:
        """Drive to a floor point, routing around tables.

        Goals next to a table edge (manipulation poses) are allowed down to
        NEAR_EDGE_MIN from the edge: the robot routes to a point straight out
        from that edge, then drives in perpendicular to it.
        """
        b = self.sim.base_pose()
        start, goal = np.array([b.x, b.y]), np.array([x, y], float)
        in_place = np.linalg.norm(goal - start) <= tol
        for t in self.tables.values():
            if not in_place and self._box_dist(goal, t.pos, *TABLE_HALF) < NEAR_EDGE_MIN - 1e-6:
                return fail(
                    f"({x:.2f},{y:.2f}) is on {t.name} or closer than {NEAR_EDGE_MIN:.2f} m to its edge "
                    f"({t.name} spans x {t.pos[0] - TABLE_HALF[0]:.2f}..{t.pos[0] + TABLE_HALF[0]:.2f}, "
                    f"y {t.pos[1] - TABLE_HALF[1]:.2f}..{t.pos[1] + TABLE_HALF[1]:.2f}). "
                    "Stand on the floor 0.15-0.30 m outside an edge instead. " + self._edge_limits(t, goal)
                )
        if (e := self._tuck_arm()) is not None:
            return e
        if in_place:  # turn in place (rotate)
            return self._drive_to(goal, yaw_deg, tol)
        near = self._near_table(goal)
        final = goal
        if near is not None:  # manipulation pose: come in straight from outside the edge
            t, n, _ = near
            goal = goal + n * 0.45  # pre-pose 0.45 m straight out, room to correct drift on the way in
        if np.linalg.norm(goal - start) > tol and (hit := self._segment_hits_table(start, goal)):
            wp = self._detour(start, goal)
            if wp is None:
                return fail(f"no collision-free route to ({x:.2f},{y:.2f}) around {hit}")
            r = self._drive_to(wp, None, tol)
            if not r["success"]:
                return r
        if near is None:
            return self._drive_to(goal, yaw_deg, tol)
        r = self._drive_to(goal, float(np.degrees(np.arctan2(-n[1], -n[0]))), 0.06)
        if not r["success"]:
            return r
        r = self._drive_straight_to(final, float(np.arctan2(-n[1], -n[0])))
        if not r["success"]:
            return r
        if yaw_deg is not None and abs(wrap(np.radians(yaw_deg) - self.sim.base_pose().yaw)) > np.radians(4):
            r = self._drive_to(final, yaw_deg, 0.5)
            if not r["success"]:
                return r
        self._hold_here()
        b = self.sim.base_pose()
        st = {"x": round(b.x, 3), "y": round(b.y, 3), "yaw_deg": round(np.degrees(b.yaw), 1)}
        pos_err = float(np.hypot(b.x - x, b.y - y))
        if pos_err > 0.12:
            return fail("did not reach goal next to the table", base=st, pos_err_m=round(pos_err, 3))
        return ok(f"arrived next to {t.name}", base=st, pos_err_m=round(pos_err, 3), now_visible=self._observe())

    def _edge_limits(self, t, goal: np.ndarray) -> str:
        """The valid standing range on the side of table t nearest to goal."""
        x0, x1 = t.pos[0] - TABLE_HALF[0], t.pos[0] + TABLE_HALF[0]
        y0, y1 = t.pos[1] - TABLE_HALF[1], t.pos[1] + TABLE_HALF[1]
        m = NEAR_EDGE_MIN + 0.01  # quote a limit with margin so it is safely accepted
        sides = {
            "x_min": (abs(goal[0] - x0), f"x must be <= {x0 - m:.2f}"),
            "x_max": (abs(goal[0] - x1), f"x must be >= {x1 + m:.2f}"),
            "y_min": (abs(goal[1] - y0), f"y must be <= {y0 - m:.2f}"),
            "y_max": (abs(goal[1] - y1), f"y must be >= {y1 + m:.2f}"),
        }
        side = min(sides, key=lambda k: sides[k][0])
        return f"To stand on the {side} side, {sides[side][1]}."

    def _drive_to(self, goal: np.ndarray, yaw_deg: float | None, tol: float) -> dict:
        """Turn-drive-align to a floor point with no collision checks (callers route first)."""
        self._hold = None
        x, y = float(goal[0]), float(goal[1])
        b = self.sim.base_pose()
        start = np.array([b.x, b.y])
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
        newly = self._observe()
        if pos_err < 0.12 and yaw_err < 8:
            return ok("arrived", base=st, pos_err_m=round(pos_err, 3), now_visible=newly)
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

    def _finger_axis(self, yaw_deg: float) -> np.ndarray:
        """Closing direction for gripper yaw: fingers close side-to-side relative to that heading.
        The gripper is symmetric, so flip it toward robot-lateral to avoid needless wrist turns."""
        a = np.radians(yaw_deg)
        fa = np.array([-np.sin(a), np.cos(a), 0.0])
        b = self.sim.base_pose()
        return fa if fa @ np.array([-np.sin(b.yaw), np.cos(b.yaw), 0.0]) >= 0 else -fa

    def _grasped_object(self) -> str | None:
        return next((n for n in self.objects if self._is_grasped(n)), None)

    def _update_holding(self) -> str | None:
        """Re-check what is between the fingers; returns the name of an object that was dropped."""
        was, self.holding = self.holding, self._grasped_object()
        return was if was and was != self.holding else None

    def _where(self, name: str) -> str:
        q = self.sim.body_pos(name)
        for bn_name, bn in self.bins.items():
            c = self.sim.body_pos(bn_name)
            if abs(q[0] - c[0]) < bn.half[0] and abs(q[1] - c[1]) < bn.half[1] and q[2] < c[2] + bn.wall_h + 0.03:
                return bn_name
        if self.holding == name:
            return "gripper"
        t = self._table_of(q)
        return t.name if t is not None and q[2] > TABLE_H - 0.02 else "floor"

    def _ee_report(self) -> dict:
        return dict(ee_xyz=[round(float(v), 3) for v in self.sim.ee_pos()], holding=self.holding)

    def move_eef(self, x: float, y: float, z: float, yaw_deg: float) -> dict:
        """Move the gripper tip (pointing straight down) to a world point, gripper rotated to yaw_deg."""
        p = np.array([x, y, z], float)
        fwd, left = self._to_robot_frame(p)
        if not (0.18 <= fwd <= 0.60 and abs(left) <= 0.30):
            return fail(
                f"({x:.2f},{y:.2f}) is out of arm reach (forward={fwd:.2f} m, left={left:.2f} m from the base); "
                "the arm reaches forward 0.18-0.60 m and at most 0.30 m sideways. Use navigate_to to move the base first."
            )
        if not (TABLE_H - 0.05 <= z <= TABLE_H + 0.40):
            return fail(f"z={z:.2f} is outside the arm's working height {TABLE_H - 0.05:.2f}-{TABLE_H + 0.40:.2f} m")
        if self._hold is None:
            self._hold_here()
        fa = self._finger_axis(yaw_deg)
        if np.linalg.norm(self.sim.ee_pos() - p) > 0.20:  # long move: joint-space to near the goal first
            sol = self._solve(p, fa)
            if sol is None:
                return fail("no arm (IK) solution for that point and yaw from here; move the base or change yaw_deg")
            if not self._move_joints(sol.q, speed=0.8):
                return fail("robot fell over")
        r = self._servo_ee(p, fa, speed=0.10, tol=0.005)
        dropped = self._update_holding()
        extra = {"dropped": dropped, "dropped_onto": self._where(dropped)} if dropped else {}
        if not r["success"]:
            return fail(f"move_eef: {r['message']}", **self._ee_report(), **extra)
        return ok("gripper at target" + (f"; {dropped} slipped out" if dropped else ""), **self._ee_report(), **extra)

    def gripper(self, action: str) -> dict:
        if action not in ("open", "close"):
            return fail("gripper action must be 'open' or 'close'")
        if self._hold is None:
            self._hold_here()
        if action == "close":
            self.sim.grip_target = GRIPPER_CLOSED
            self._wait(0.8)
            self._update_holding()
            if self.holding:
                return ok(f"gripper closed on {self.holding}", **self._ee_report())
            return fail("gripper closed on nothing", **self._ee_report())
        obj = self.holding
        self.sim.grip_target = GRIPPER_OPEN
        self._wait(0.8)
        self.holding = None
        if obj is None:
            return ok("gripper open", **self._ee_report())
        return ok(f"released {obj}; it is now on/in {self._where(obj)}", object_now_on=self._where(obj), **self._ee_report())

    def _tuck_arm(self) -> dict | None:
        """Before driving: lift clear of the table and fold the arm, keeping any held object."""
        if np.max(np.abs(self.sim.arm_target - STOW_Q)) < 0.05:
            return None
        if self.sim.ee_pos()[2] < TABLE_H + 0.12:
            self._servo_ee(self.sim.ee_pos() + np.array([0, 0, 0.10]), speed=0.15, timeout=2.0)
        if not self._move_joints(STOW_Q, speed=0.8):
            return fail("robot fell over")
        self._update_holding()
        return None
