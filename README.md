# tron2-agent

Agentic navigation and manipulation for the LimX TRON2 WFYG (wheel-legged base with one 6-DoF arm) in MuJoCo. A local vision-language model running in Ollama plans the task by calling robot skills as tools. Everything runs on an Apple Silicon Mac (developed on an M4 with 16 GB).

## Architecture

```
 instruction ──► Planner (local VLM via Ollama, e.g. qwen3-vl:8b-instruct)
                   │  tool call (one per turn)            ▲ JSON result + head camera image
                   ▼                                       │
               Validator (schema, types, bounds) ──► Skills ┘
                                                   │  detect / search / navigate_to / approach / pick / place ...
                                                   ▼
               Sim (MuJoCo, 1 kHz)
                 ├─ legs + wheels: LimX WF_TRON2A RL policy (ONNX, 50 Hz) taking (vx, wz) commands
                 ├─ base station keeping during manipulation
                 ├─ arm: damped least-squares IK + Cartesian servo, PD with gravity compensation
                 └─ gripper: force-limited PD
```

The planner never outputs motor commands. It sequences coarse skills, and each skill runs its own closed-loop control inside the simulator. This follows the "LLM as high-level planner" pattern from the GPT-6 Astra evaluations, sized for a small local model.

**Locomotion.** The legs run the pretrained `WF_TRON2A` policy from `limxdynamics/tron2_rl_deploy_python`. It was trained without an arm. It still balances the WFYG body (about 7 kg heavier), but the base drifts when the arm reaches out. A position-hold loop counters this during manipulation.

**Perception.** `detect()` uses simulator ground truth, filtered to items inside the head camera's field of view. Items seen earlier are remembered, and `approach` refuses items that have never been seen, so the planner still has to search (`search(name)` turns in place until the item is in view). Camera images go to the model for visual context only.

**Manipulation.** Grasps are top-down. The arm reaches from 0.60 to 1.10 m high and 0.18 to 0.55 m in front of the base. `approach()` puts the base about 0.38 m from the target, square to the nearest table edge.

## Layout

| Path | Contents |
|---|---|
| `src/tron2_agent/locomotion.py` | ONNX wheel-leg policy wrapper (observation layout matches LimX deploy code) |
| `src/tron2_agent/arm.py` | Arm IK |
| `src/tron2_agent/scene.py` | Scene: robot, two tables, cubes, can, bin, head and wrist cameras |
| `src/tron2_agent/sim.py` | Physics loop and low-level control |
| `src/tron2_agent/skills.py` | Skills exposed as tools |
| `src/tron2_agent/agent.py` | Tool schemas, validator, Ollama and scripted planners, episode runner |
| `src/tron2_agent/tasks.py` | Tasks with ground-truth success checks |
| `src/tron2_agent/cli.py` | `tron2-agent run` and `tron2-agent eval` |
| `scripts/fetch_assets.sh` | Downloads the LimX robot model and policy into `assets/` |

## Setup

```bash
./scripts/fetch_assets.sh          # robot MJCF + meshes, WF policy (Apache-2.0, LimX Dynamics)
uv sync
ollama pull qwen3-vl:8b-instruct   # needs a recent Ollama (0.40+)
```

## Usage

```bash
uv run tron2-agent run --task red_to_bin --planner scripted --video   # no LLM, sanity check
uv run tron2-agent run --task red_to_bin --video                      # qwen3-vl:8b-instruct
uv run tron2-agent run --instruction "put the blue cube in the bin" --model gemma4:12b-it-qat
uv run tron2-agent eval --task two_cubes --episodes 3 --seed 10   # randomized start heading + object jitter
uv run pytest
```

Each run writes `runs/<stamp>_<task>_<model>/` containing `trace.jsonl` (every tool call, result, and model latency), the camera images sent to the model, `result.json`, and `video.mp4` when `--video` is set.

## Tasks

| Task | Instruction |
|---|---|
| `red_to_bin` | Put the red cube into the bin. |
| `blue_to_bin` | Find the blue cube and put it in the bin. |
| `can_to_bin` | Bring the green can to the bin. |
| `two_cubes` | Put both cubes (red and blue) into the bin. |

## Results

`qwen3-vl:8b-instruct` on an M4 (16 GB), randomized scenes (`--seed`):

| Task | Episodes | Success | Tool calls per episode | Wall time per episode |
|---|---|---|---|---|
| `two_cubes` | 3 (seeds 10-12) | 3/3 | 15-17 | 6-7 min |
| `can_to_bin` (scripted planner, no LLM) | 4 (seeds 1-4) | 4/4 | 7 | ~9 s |

Single default-scene runs of `red_to_bin`, `can_to_bin` and `two_cubes` also succeeded. Most of the wall time is model latency (about 18 s per call with an image attached).

## Models

| Role | Model | Size |
|---|---|---|
| Planner (default) | `qwen3-vl:8b-instruct` | 6.2 GB |
| Planner alternative | `gemma4:12b-it-qat` | 7.2 GB |
| Fast or second model | `qwen3-vl:4b`, `gemma4:e4b` | 3.3 / 6.1 GB |

Any Ollama model with tool calling works through `--model`. Use `--no-images` for text-only models.

## Known limitations

- Perception is privileged. Next step: replace `detect()` with VLM pointing on the head camera plus depth back-projection.
- The locomotion policy never saw arm motion. Driving with the arm extended is untested; skills stow the arm before moving.
- Navigation only avoids tables (straight-line checks plus one detour waypoint). It does not plan around arbitrary obstacles.
- Every model call takes about 5 to 20 s on an M4 with images attached, so one episode takes a few minutes of wall time.
