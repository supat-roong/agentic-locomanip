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


def test_pick_requires_reach(sim):
    from tron2_agent.skills import Skills

    sim.reset()
    r = Skills(sim).pick("red_cube")
    assert not r["success"] and "out of arm reach" in r["message"]
