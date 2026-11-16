import pytest

from tron2_agent.paths import ROBOT_XML

pytestmark = pytest.mark.skipif(not ROBOT_XML.exists(), reason="run scripts/fetch_assets.sh first")


def test_scripted_red_to_bin_succeeds():
    from tron2_agent.agent import AgentRunner, scripted_pick_place
    from tron2_agent.sim import Sim
    from tron2_agent.skills import Skills
    from tron2_agent.tasks import TASKS

    sim = Sim()
    res = AgentRunner(Skills(sim), scripted_pick_place("red_cube"), use_images=False, verbose=False).run("x")
    assert res.failed_skills == 0
    assert TASKS["red_to_bin"].check(sim)
