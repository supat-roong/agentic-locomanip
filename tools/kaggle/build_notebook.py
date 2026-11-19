"""Write tools/kaggle/notebook/agentic-locomanip-demo.ipynb (the Kaggle notebook).

The notebook expects the repo source as a private Kaggle dataset (see push.py),
fetches the public LimX assets itself, runs Ollama on the GPU and records both
use cases with annotated demo videos.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "gemma4:12b-it-qat"

CELLS: list[tuple[str, str]] = [
    ("markdown", f"""# Agentic Loco-Manipulation with a Local VLM

A wheel-legged robot with an arm finds a cube, drives to it, picks it up and drops it in a bin. Every decision is made
by a **vision-language model ({MODEL}) running locally in Ollama** on this notebook's T4 GPU, one tool call at a time.
No cloud API, no task-specific policy, no pick or place macro.

Source code: [github.com/supat-roong/agentic-locomanip](https://github.com/supat-roong/agentic-locomanip)

**What this notebook does (about 30 min on a T4):**

1. Installs the simulator (MuJoCo) and downloads the open LimX TRON2 robot model and its locomotion policy.
2. Starts Ollama on the GPU and pulls `{MODEL}` (7 GB).
3. Runs two tasks with the model in the loop: **red_to_bin** and **two_cubes**.
4. Renders an annotated demo video of each run and prints a summary table."""),
    ("markdown", """## The robot and the scene

The robot is a **LimX TRON2 WFYG**: two legs ending in wheels, a torso and one 6-DoF arm with a parallel gripper. The scene
is a room with two tables, a red cube, a blue cube, a green can and a bin. The robot starts away from the tables and
does not know where anything is until it looks.

## How the agent works

```
 instruction ──► Planner (local VLM via Ollama)
                   │  one tool call per turn               ▲ JSON result + head camera image
                   ▼                                        │
               Validator (schema, types, bounds) ──► Tools ─┘
                                                   │
                                                   ▼
               MuJoCo simulation (1 kHz)
                 ├─ legs + wheels: pretrained LimX RL policy (ONNX, 50 Hz), commanded with (vx, wz)
                 ├─ arm: damped least-squares IK + Cartesian servo with gravity compensation
                 └─ gripper: force-limited PD
```

The model never outputs motor commands, but it does all of the task reasoning. Each tool runs its own closed-loop
controller inside the simulator and returns a short JSON result, plus a camera image after moving or turning.

| Tool | What it does |
|---|---|
| `detect()` | Items in the head camera view plus items seen earlier, with world positions and the closest table edge |
| `look(camera)` | Attach a head or wrist camera image |
| `rotate(angle_deg)` | Turn in place (the model searches by turning 45 degrees and calling `detect`) |
| `navigate_to(x, y, yaw_deg)` | Drive to a floor point, routing around tables |
| `move_eef(x, y, z, yaw_deg)` | Move the gripper tip (pointing down) to a world point |
| `gripper(action)` | Open or close the gripper |
| `get_state()` | Base pose, held object, gripper opening |
| `done(success, summary)` | End the episode |

To pick up a cube, the model has to work out where to stand from the cube's position and table edge, then hover above
it, lower, close, lift, drive to the bin and open above it. Tool results are written to steer a small model: a rejected
`navigate_to` states the valid range, `gripper("close")` reminds it where the bin is, and a release that misses the bin
is reported as a failure. With `--explain` every call also carries a one-sentence `reason`, shown in the demo videos.

Success is checked from simulator ground truth (is the cube inside the bin?), not from what the model claims."""),
    ("markdown", """## Setup

Copy the source (attached as a Kaggle dataset), install the Python packages and fetch the robot assets
(Apache-2.0, LimX Dynamics)."""),
    ("code", """import glob, os, shutil, subprocess, tarfile

SRC = None
for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True):
    SRC = os.path.dirname(p)
    break
if SRC is None:  # archive not auto-extracted
    tgz = glob.glob("/kaggle/input/**/agentic-locomanip*.tar.gz", recursive=True)[0]
    tarfile.open(tgz).extractall("/kaggle/tmp_src")
    SRC = os.path.dirname(glob.glob("/kaggle/tmp_src/**/pyproject.toml", recursive=True)[0])
REPO = "/kaggle/working/agentic-locomanip"
shutil.copytree(SRC, REPO, dirs_exist_ok=True)
os.chdir(REPO)
print("repo at", REPO, os.listdir(REPO))"""),
    ("code", """%%bash
set -e
cd /kaggle/working/agentic-locomanip
pip install -q mujoco onnxruntime ollama imageio imageio-ffmpeg pyyaml scipy pillow 2>&1 | tail -2
bash scripts/fetch_assets.sh
python -c "import mujoco, numpy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)" """),
    ("markdown", """Start Ollama on the GPU and pull the model."""),
    ("code", f"""%%bash
# Ollama on the GPU
(command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null))
curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1 || (tail -20 /tmp/ollama_install.log; exit 1)
nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 1; done
ollama --version
ollama pull {MODEL} 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv"""),
    ("markdown", """MuJoCo needs an offscreen OpenGL context to render camera images and video. Try EGL on the GPU first."""),
    ("code", """# Headless rendering: try EGL (GPU), fall back to OSMesa (CPU)
import os, subprocess, sys

def render_ok(backend):
    code = ("import os; os.environ['MUJOCO_GL']='%s'; import sys; sys.path.insert(0,'src'); "
            "from agentic_locomanip.sim import Sim; s=Sim(); print(s.render('follow').shape)" % backend)
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    print(backend, "->", (r.stdout or r.stderr).strip().splitlines()[-1:])
    return r.returncode == 0

if render_ok("egl"):
    os.environ["MUJOCO_GL"] = "egl"
else:
    subprocess.run("apt-get install -y -qq libosmesa6 libosmesa6-dev >/dev/null", shell=True)
    assert render_ok("osmesa"), "no working MuJoCo renderer"
    os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYTHONPATH"] = os.path.abspath("src")
print("MUJOCO_GL =", os.environ["MUJOCO_GL"])"""),
    ("markdown", """## Use case 1: red_to_bin

Instruction: **"Put the red cube into the bin."** The log below lists every tool call the model made, whether it
succeeded, and the first part of the result the model read back."""),
    ("code", f"""import glob, json, subprocess, sys, time

def run_case(task, max_steps):
    t0 = time.time()
    cmd = [sys.executable, "-m", "agentic_locomanip.cli", "run", "--task", task, "--model", "{MODEL}",
           "--video", "--explain", "--max-steps", str(max_steps)]
    log = open(f"/kaggle/working/{{task}}.log", "w")
    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT)
    run_dir = sorted(glob.glob(f"runs/*_{{task}}_*"))[-1]
    res = json.load(open(f"{{run_dir}}/result.json"))
    print(f"{{task}}: task_success={{res['task_success']}} steps={{res['steps']}} "
          f"wall={{(time.time() - t0) / 60:.1f}} min  ({{run_dir}})")
    for line in open(f"{{run_dir}}/trace.jsonl"):
        r = json.loads(line)
        if "tool" in r:
            a = ", ".join(f"{{k}}={{v}}" for k, v in r["args"].items())
            mark = "ok " if r["result"].get("success") else "ERR"
            print(f"  {{r['step']:>2}} {{mark}} {{r['tool']}}({{a}})  -> {{r['result'].get('message', '')[:90]}}")
    return run_dir, res

red_dir, red_res = run_case("red_to_bin", 40)"""),
    ("markdown", """## Use case 2: two_cubes

Instruction: **"Put both cubes (red and blue) into the bin."** Two full grasp-and-drop cycles. The robot has to
remember where the bin is, go back for the second cube and repeat."""),
    ("code", """two_dir, two_res = run_case("two_cubes", 50)"""),
    ("markdown", """## Demo videos

Each video shows the robot on the left. On the right: the prompt, the head camera image the model saw, its reason for
the next call, and the tool call log. Model thinking time is shown as a short pause and robot motion plays at 1.5x."""),
    ("code", """# Annotated demo videos: robot view + prompt, camera image, model's reason, tool call and result
for task, d in (("red_to_bin", red_dir), ("two_cubes", two_dir)):
    out = f"/kaggle/working/demo_{task}.mp4"
    subprocess.run([sys.executable, "-m", "agentic_locomanip.cli", "demo", d, "--out", out], check=True)
    print(out)"""),
    ("code", """from IPython.display import Video, display
display(Video("/kaggle/working/demo_red_to_bin.mp4", embed=True, width=960))"""),
    ("code", """display(Video("/kaggle/working/demo_two_cubes.mp4", embed=True, width=960))"""),
    ("markdown", """## Summary

`tool_calls` counts every call the model made, `failed_calls` the ones a tool rejected (the model reads the error and
tries again), `sim_s` simulated robot time and `llm_s` total model latency."""),
    ("code", """import pandas as pd
rows = []
for task, res in (("red_to_bin", red_res), ("two_cubes", two_res)):
    rows.append(dict(task=task, success=res["task_success"], tool_calls=res["steps"], failed_calls=res["failed_skills"],
                     sim_s=round(res["sim_time_s"]), llm_s=round(res["llm_time_s"]), wall_min=round(res["wall_time_s"] / 60, 1)))
shutil.copytree("runs", "/kaggle/working/runs_out", dirs_exist_ok=True)
pd.DataFrame(rows)"""),
    ("markdown", """## Limitations and next steps

- **Perception is privileged.** `detect()` reads positions from the simulator, filtered to what the head camera can see.
  The next step is VLM pointing on the camera image plus depth back-projection.
- **The locomotion policy never saw arm motion**, so the arm is folded before every drive and the base holds position
  while the arm moves.
- **Navigation only avoids tables** (straight-line checks plus one detour waypoint).
- Runs are not always successful. On randomized scenes gemma4:12b solved 4 of 5 episodes on a Mac. The usual failure is
  looping back to the same spot after the first cube.

Full code, tests and more results: [github.com/supat-roong/agentic-locomanip](https://github.com/supat-roong/agentic-locomanip)"""),
]


def main() -> None:
    nb = {
        "cells": [
            {"cell_type": t, "metadata": {}, "source": src.splitlines(keepends=True),
             **({"outputs": [], "execution_count": None} if t == "code" else {})}
            for t, src in CELLS
        ],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python"}},
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).parent / "notebook" / "agentic-locomanip-demo.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
