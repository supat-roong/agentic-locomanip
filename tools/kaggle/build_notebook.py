"""Write tools/kaggle/notebook/tron2-agent-demo.ipynb (the Kaggle notebook).

The notebook expects the repo source as a private Kaggle dataset (see push.py),
fetches the public LimX assets itself, runs Ollama on the GPU and records both
use cases with annotated demo videos.
"""

from __future__ import annotations

import json
from pathlib import Path

MODEL = "gemma4:12b-it-qat"

CELLS: list[tuple[str, str]] = [
    ("markdown", f"""# TRON2 wheel-legged robot, driven by a local VLM

A LimX TRON2 WFYG (wheel-legged base, 6-DoF arm, parallel gripper) in MuJoCo, planned step by step by
**{MODEL}** running in Ollama on the Kaggle GPU. The model only calls tools:

| Tool | What it does |
|---|---|
| `detect()` | items in the head camera, with world positions and closest table edge |
| `look(camera)` | attach a head or wrist camera image |
| `rotate(angle_deg)` | turn in place |
| `navigate_to(x, y, yaw_deg)` | drive to a floor point, routing around tables |
| `move_eef(x, y, z, yaw_deg)` | move the gripper tip (pointing down) to a world point |
| `gripper(action)` | open or close |
| `get_state()`, `done(success, summary)` | |

There is no pick or place skill: the model works out where to stand, then hovers, lowers, closes, lifts and drops.

Two use cases: **red_to_bin** ("Put the red cube into the bin.") and **two_cubes** ("Put both cubes (red and blue) into the bin.")."""),
    ("code", """import glob, os, shutil, subprocess, tarfile

SRC = None
for p in glob.glob("/kaggle/input/**/pyproject.toml", recursive=True):
    SRC = os.path.dirname(p)
    break
if SRC is None:  # archive not auto-extracted
    tgz = glob.glob("/kaggle/input/**/tron2-agent*.tar.gz", recursive=True)[0]
    tarfile.open(tgz).extractall("/kaggle/tmp_src")
    SRC = os.path.dirname(glob.glob("/kaggle/tmp_src/**/pyproject.toml", recursive=True)[0])
REPO = "/kaggle/working/tron2-agent"
shutil.copytree(SRC, REPO, dirs_exist_ok=True)
os.chdir(REPO)
print("repo at", REPO, os.listdir(REPO))"""),
    ("code", """%%bash
set -e
cd /kaggle/working/tron2-agent
pip install -q mujoco onnxruntime ollama imageio imageio-ffmpeg pyyaml scipy pillow 2>&1 | tail -2
bash scripts/fetch_assets.sh
python -c "import mujoco, numpy; print('mujoco', mujoco.__version__, 'numpy', numpy.__version__)" """),
    ("code", f"""%%bash
# Ollama on the GPU
(command -v zstd >/dev/null || (apt-get update -qq && apt-get install -y -qq zstd >/dev/null))
curl -fsSL https://ollama.com/install.sh | sh > /tmp/ollama_install.log 2>&1 || (tail -20 /tmp/ollama_install.log; exit 1)
nohup ollama serve > /kaggle/working/ollama.log 2>&1 &
for i in $(seq 1 30); do curl -s localhost:11434 >/dev/null && break; sleep 1; done
ollama --version
ollama pull {MODEL} 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv"""),
    ("code", """# Headless rendering: try EGL (GPU), fall back to OSMesa (CPU)
import os, subprocess, sys

def render_ok(backend):
    code = ("import os; os.environ['MUJOCO_GL']='%s'; import sys; sys.path.insert(0,'src'); "
            "from tron2_agent.sim import Sim; s=Sim(); print(s.render('follow').shape)" % backend)
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
    ("code", f"""import glob, json, subprocess, sys, time

def run_case(task, max_steps):
    t0 = time.time()
    cmd = [sys.executable, "-m", "tron2_agent.cli", "run", "--task", task, "--model", "{MODEL}",
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
    ("code", """two_dir, two_res = run_case("two_cubes", 50)"""),
    ("code", """# Annotated demo videos: robot view + prompt, camera image, model's reason, tool call and result
for task, d in (("red_to_bin", red_dir), ("two_cubes", two_dir)):
    out = f"/kaggle/working/demo_{task}.mp4"
    subprocess.run([sys.executable, "-m", "tron2_agent.cli", "demo", d, "--out", out], check=True)
    print(out)"""),
    ("code", """from IPython.display import Video, display
display(Video("/kaggle/working/demo_red_to_bin.mp4", embed=True, width=960))"""),
    ("code", """display(Video("/kaggle/working/demo_two_cubes.mp4", embed=True, width=960))"""),
    ("code", """import pandas as pd
rows = []
for task, res in (("red_to_bin", red_res), ("two_cubes", two_res)):
    rows.append(dict(task=task, success=res["task_success"], tool_calls=res["steps"], failed_calls=res["failed_skills"],
                     sim_s=round(res["sim_time_s"]), llm_s=round(res["llm_time_s"]), wall_min=round(res["wall_time_s"] / 60, 1)))
shutil.copytree("runs", "/kaggle/working/runs_out", dirs_exist_ok=True)
pd.DataFrame(rows)"""),
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
    out = Path(__file__).parent / "notebook" / "tron2-agent-demo.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(out)


if __name__ == "__main__":
    main()
