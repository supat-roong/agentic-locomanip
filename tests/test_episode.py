import pytest

from agentic_locomanip.paths import ROBOT_XML

pytestmark = pytest.mark.skipif(not ROBOT_XML.exists(), reason="run scripts/fetch_assets.sh first")


def test_scripted_red_to_bin_succeeds():
    from agentic_locomanip.agent import AgentRunner, scripted_pick_place
    from agentic_locomanip.sim import Sim
    from agentic_locomanip.skills import Skills
    from agentic_locomanip.tasks import TASKS

    sim = Sim()
    sk = Skills(sim)
    res = AgentRunner(sk, scripted_pick_place(sk, "red_cube"), use_images=False, verbose=False).run("x")
    assert res.failed_skills == 0
    assert TASKS["red_to_bin"].check(sim)
