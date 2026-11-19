# agentic-locomanip

Agentic loco-manipulation in MuJoCo, demonstrated on the LimX TRON2 WFYG (wheel-legged base with one 6-DoF arm). A local vision-language model running in Ollama plans the task by calling robot tools. Everything runs on an Apple Silicon Mac (developed on an M4 with 16 GB), or on a Kaggle T4 GPU.

![gemma4:12b puts the red cube into the bin: 15 tool calls, none failed](docs/demo_kaggle_red_to_bin.gif)

*"Put the red cube into the bin." gemma4:12b-it-qat on a Kaggle T4: 15 tool calls, none failed. Left: the robot. Right: what the model saw, its reason and the tool call log. Sped up 2x.*

**Run it yourself:** [Kaggle notebook: Agentic Loco-Manipulation with a Local VLM](https://www.kaggle.com/code/supatroongpraiwan/agentic-loco-manipulation-with-a-local-vlm) (free T4 GPU, about 30 min, no setup).

## Architecture

```
 instruction ──► Planner (local VLM via Ollama, default gemma4:12b-it-qat)
                   │  tool call (one per turn)            ▲ JSON result + head camera image
                   ▼                                       │
               Validator (schema, types, bounds) ──► Tools ┘
                                                   │  detect / rotate / navigate_to / move_eef / gripper ...
                                                   ▼
               Sim (MuJoCo, 1 kHz)
                 ├─ legs + wheels: LimX WF_TRON2A RL policy (ONNX, 50 Hz) taking (vx, wz) commands
                 ├─ base station keeping during manipulation
                 ├─ arm: damped least-squares IK + Cartesian servo, PD with gravity compensation
                 └─ gripper: force-limited PD
```

The planner never outputs motor commands, but it does all of the task reasoning. There is no pick, place or approach skill: the model reads positions from `detect`, works out where to stand, then hovers, lowers, closes, lifts and drops with `move_eef` and `gripper`. Each tool runs its own closed-loop control inside the simulator.

**Locomotion.** The legs run the pretrained `WF_TRON2A` policy from `limxdynamics/tron2_rl_deploy_python`. It was trained without an arm. It still balances the WFYG body (about 7 kg heavier), but the base drifts when the arm reaches out, so a position-hold loop runs during arm moves. `navigate_to` folds the arm before driving (a held object stays held).

**Perception.** `detect()` uses simulator ground truth, filtered to items inside the head camera's field of view. Items seen earlier are remembered. Objects and the bin come with their world position and closest table edge, tables with their x and y ranges. A held object is reported only as "in the gripper", so its coordinates cannot be mistaken for a place to go. Camera images go to the model for visual context.

**Manipulation.** The gripper always points straight down. The arm reaches 0.18 to 0.60 m in front of the base, at most 0.30 m to the side, from table height up to 0.40 m above it. Tool results are written to steer a small model: a rejected `navigate_to` names the valid range ("x must be <= 1.54"), `gripper("close")` reports what it grasped and where the bin is, and a release that misses the bin is reported as a failure.

## Tools

| Tool | Arguments | What it does |
|---|---|---|
| `detect` | | Items in view plus remembered items, with positions and closest table edge |
| `look` | `camera` (head, wrist) | Attach a camera image |
| `rotate` | `angle_deg` | Turn in place (positive is left). Search by turning 45 degrees the same way and calling `detect` |
| `navigate_to` | `x`, `y`, `yaw_deg` | Drive to a floor point, routing around tables. Must stay at least 0.15 m outside table edges |
| `move_eef` | `x`, `y`, `z`, `yaw_deg` | Move the gripper tip to a world point |
| `gripper` | `action` (open, close) | Open or close the gripper |
| `get_state` | | Base pose, held object, gripper opening |
| `done` | `success`, `summary` | End the episode |

With `--explain` every tool also takes a required `reason` string (one sentence), which the demo video shows.

## Setup

```bash
./scripts/fetch_assets.sh          # robot MJCF + meshes, WF policy (Apache-2.0, LimX Dynamics)
uv sync
ollama pull gemma4:12b-it-qat      # needs a recent Ollama (0.40+)
```

## Usage

```bash
uv run agentic-locomanip run --task red_to_bin --planner scripted --video   # no LLM, sanity check
uv run agentic-locomanip run --task red_to_bin --video --explain            # gemma4:12b-it-qat
uv run agentic-locomanip demo runs/<run dir>                                # annotated demo.mp4
uv run agentic-locomanip run --instruction "put the blue cube in the bin" --model qwen3-vl:8b-instruct
uv run agentic-locomanip eval --task two_cubes --episodes 2 --seed 10 --max-steps 50   # randomized scenes
uv run pytest
```

Each run writes `runs/<stamp>_<task>_<model>/` containing `trace.jsonl` (every tool call, result, reason and model latency), the camera images sent to the model, `result.json`, and with `--video` a chase-camera `video.mp4`, a head-camera `head.mp4` and `frames.json`.

`agentic-locomanip demo` turns a run recorded with `--video` into a 1280x720 video: the robot on the left, and on the right the prompt, the live head camera (with the last image the model received as an inset), its reason, the tool call and the result. The head camera is recorded as `head.mp4`. Model thinking time is shown as a 2.5 s pause per call and robot motion plays at 1.5x (`--think-s`, `--speed`).

## Kaggle

The public notebook [Agentic Loco-Manipulation with a Local VLM](https://www.kaggle.com/code/supatroongpraiwan/agentic-loco-manipulation-with-a-local-vlm) explains the setup, runs `red_to_bin` and `two_cubes` with gemma4:12b-it-qat on a T4 and renders both demo videos. To push your own copy:

```bash
uv run python tools/kaggle/push.py --user <kaggle-username>            # private
uv run python tools/kaggle/push.py --user <kaggle-username> --public   # public
```

The notebook (T4 GPU, internet on) clones this repo from GitHub, fetches the LimX assets, installs Ollama and pulls the model itself.

## Tasks

Each demo: the robot on the left; on the right the live head camera, the model's reason and the tool call log. All four runs used `gemma4:12b-it-qat` on the M4 (16 GB), default scene. GIFs play at 2x the demo video.

### `red_to_bin`

*"Put the red cube into the bin."* 17 tool calls, 1 failed, 5.3 min on the M4.

![red_to_bin demo](docs/demo_red_to_bin.gif)

### `blue_to_bin`

*"Find the blue cube and put it in the bin."* 19 tool calls, 1 failed, 9.9 min on the M4.

![blue_to_bin demo](docs/demo_blue_to_bin.gif)

### `can_to_bin`

*"Bring the green can to the bin."* 18 tool calls, 2 failed, 8.8 min on the M4.

![can_to_bin demo](docs/demo_can_to_bin.gif)

### `two_cubes`

*"Put both cubes (red and blue) into the bin."* 31 tool calls, 3 failed, 13.8 min on the M4.

![two_cubes demo](docs/demo_two_cubes.gif)

## Models

| Role | Model | Size |
|---|---|---|
| Planner (default) | `gemma4:12b-it-qat` | 7.2 GB |
| Planner alternative | `qwen3-vl:8b-instruct` | 6.2 GB |

Any Ollama model with tool calling works through `--model`. Use `--no-images` for text-only models.

## License

MIT for this repository (see `LICENSE`). The robot model and locomotion policy fetched by `scripts/fetch_assets.sh` come from [LimX Dynamics](https://github.com/limxdynamics) under Apache-2.0 and are not redistributed here.
