import numpy as np
import pytest

from tron2_agent.paths import ROBOT_XML

pytestmark = pytest.mark.skipif(not ROBOT_XML.exists(), reason="run scripts/fetch_assets.sh first")


@pytest.fixture(scope="module")
def sim():
    from tron2_agent.sim import Sim

    s = Sim()
    yield s
    s.close()


def test_stands_and_drives(sim):
    sim.reset()
    assert not sim.fallen()
    x0 = sim.base_pose().x
    sim.set_base_command(0.5, 0.0)
    sim.step_seconds(3.0)
    sim.set_base_command(0.0, 0.0)
    sim.step_seconds(1.0)
    p = sim.base_pose()
    assert not sim.fallen() and p.x - x0 > 0.5 and p.tilt_deg < 10


def test_topdown_ik_reaches_table_height(sim):
    sim.reset()
    p = sim.base_pose()
    tgt = np.array([p.x + 0.42, p.y, 0.66])
    r = sim.ik.solve(sim.d.qpos.copy(), tgt, np.array([0, 0, -1.0]), q_init=np.array([0, 1.2, -1.0, 0, 0.8, 0]))
    assert r.ok and r.pos_err < 0.01


def test_move_eef_requires_reach(sim):
    from tron2_agent.skills import Skills

    sim.reset()
    r = Skills(sim).move_eef(1.82, 0.10, 0.76, 0.0)  # red cube, robot still at the origin
    assert not r["success"] and "out of arm reach" in r["message"]


def test_moving_away_from_table_is_allowed(sim):
    from tron2_agent.skills import Skills

    k = Skills(sim)
    near = np.array([1.5, 0.0])  # inside table_A's inflated footprint
    assert k._segment_hits_table(near, np.array([0.5, 0.0])) is None
    assert k._segment_hits_table(near, np.array([2.0, 0.0])) == "table_A"


def test_navigate_to_rejects_points_on_a_table(sim):
    from tron2_agent.skills import Skills

    sim.reset()
    sk = Skills(sim)
    r = sk.navigate_to(1.82, 0.10)  # red cube's position, on table_A
    assert not r["success"] and "table_A" in r["message"] and "x 1.70..2.30" in r["message"]
    assert abs(sim.base_pose().x) < 0.1  # rejected before moving
    assert "x must be <= 1.54" in r["message"]
    assert "closer than" not in sk.navigate_to(1.55, 0.10, 0.0)["message"]  # exactly 0.15 m out is allowed


def test_navigate_to_routes_around_table(sim):
    from tron2_agent.skills import Skills

    sim.reset()
    sk = Skills(sim)
    r = sk.navigate_to(2.0, 0.9)  # beside table_A; the straight path clips its corner
    assert r["success"], r
    p = sim.base_pose()
    assert np.hypot(p.x - 2.0, p.y - 0.9) < 0.12 and not sim.fallen()


def test_navigate_then_move_eef_grasp_and_drop(sim):
    """The planner's job: stand 0.4 m out from the nearest edge, hover, lower, close, lift, drop in the bin."""
    from tron2_agent.skills import Skills

    sim.reset()
    sk = Skills(sim)
    assert sk.navigate_to(1.42, 0.10, 0.0)["success"]  # red cube at (1.82, 0.10), table edge x = 1.70
    p = sim.body_pos("red_cube")
    assert sk.gripper("open")["success"]
    assert sk.move_eef(p[0], p[1], p[2] + 0.12, 0.0)["success"]
    assert sk.move_eef(p[0], p[1], p[2], 0.0)["success"]
    r = sk.gripper("close")
    assert r["success"] and r["holding"] == "red_cube", r
    assert sk.move_eef(p[0], p[1], p[2] + 0.15, 0.0)["success"]
    assert sim.body_pos("red_cube")[2] > p[2] + 0.1
    held = next(i for i in sk.detect()["items"] + sk.detect()["remembered"] if i["name"] == "red_cube")
    assert held["on"] == "gripper" and "world_xyz" not in held
    r = sk.navigate_to(0.0, 1.55, 90.0)  # bin at (0, 1.95), table_B edge y = 1.75
    assert r["success"] and sk.holding == "red_cube", r
    c = sim.body_pos("bin")
    assert sk.move_eef(c[0], c[1], c[2] + 0.15, 90.0)["success"]
    r = sk.gripper("open")
    assert r["object_now_on"] == "bin", r
