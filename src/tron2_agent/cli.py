"""Command-line entry point.

  tron2-agent run --task red_to_bin                       # qwen3-vl:8b-instruct via Ollama
  tron2-agent run --task red_to_bin --model gemma4:12b-it-qat
  tron2-agent run --instruction "put the blue cube in the bin"
  tron2-agent run --task red_to_bin --planner scripted    # no LLM baseline
  tron2-agent eval --task red_to_bin --episodes 5
  tron2-agent run --task two_cubes --video --explain && tron2-agent demo runs/<dir>
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict

import numpy as np

from .agent import AgentRunner, OllamaPlanner, scripted_pick_place
from .paths import RUNS
from .scene import randomized_cfg
from .sim import Sim
from .skills import Skills
from .tasks import TASKS

DEFAULT_MODEL = "qwen3-vl:8b-instruct"
SCRIPTED_OBJ = {"red_to_bin": "red_cube", "blue_to_bin": "blue_cube", "can_to_bin": "green_can"}


class VideoRecorder:
    def __init__(self, path, fps: int = 15, camera: str = "follow"):
        import imageio.v2 as imageio

        self.path = path
        self.w = imageio.get_writer(str(path), fps=fps, macro_block_size=8)
        self.period, self.camera, self.next_t = 1.0 / fps, camera, None
        self.times: list[float] = []  # sim time of each frame, used by `tron2-agent demo`

    def __call__(self, sim: Sim) -> None:
        if self.next_t is None:
            self.next_t = sim.time
        if sim.time >= self.next_t:
            self.w.append_data(sim.render(self.camera))
            self.times.append(round(sim.time, 3))
            self.next_t += self.period

    def close(self) -> None:
        self.w.close()
        self.path.with_name("frames.json").write_text(json.dumps(self.times))


def episode(args, instruction: str, task_name: str | None, out_dir, verbose=True, seed: int | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    if seed is None:
        sim = Sim()
    else:  # random start heading + jittered objects
        yaw = float(np.random.default_rng(seed + 1000).uniform(-np.pi, np.pi))
        sim = Sim(randomized_cfg(seed), start_yaw=yaw)
    rec = VideoRecorder(out_dir / "video.mp4") if args.video else None
    skills = Skills(sim, on_tick=rec)
    if args.planner == "scripted":
        if task_name not in SCRIPTED_OBJ:
            raise SystemExit(f"scripted planner supports tasks {list(SCRIPTED_OBJ)}")
        planner = scripted_pick_place(skills, SCRIPTED_OBJ[task_name])
    else:
        planner = OllamaPlanner(args.model, host=args.host, think=args.think)
    runner = AgentRunner(
        skills,
        planner,
        max_steps=args.max_steps,
        use_images=not args.no_images,
        log_dir=out_dir,
        verbose=verbose,
        explain=args.explain,
    )
    res = runner.run(instruction)
    if rec:
        rec.close()
    task = TASKS.get(task_name) if task_name else None
    out = {k: v for k, v in asdict(res).items() if k != "trace"}
    out.update(
        instruction=instruction,
        model=args.model if args.planner == "ollama" else "scripted",
        task=task_name,
        task_success=task.check(sim) if task else None,
        seed=seed,
    )
    (out_dir / "result.json").write_text(json.dumps(out, indent=2, default=str))
    sim.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser(prog="tron2-agent")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "eval"):
        p = sub.add_parser(name)
        p.add_argument("--task", choices=sorted(TASKS), default="red_to_bin")
        p.add_argument("--instruction", help="free-form instruction (overrides --task text)")
        p.add_argument("--planner", choices=["ollama", "scripted"], default="ollama")
        p.add_argument("--model", default=DEFAULT_MODEL)
        p.add_argument("--host", default=None, help="Ollama host, default http://localhost:11434")
        p.add_argument("--think", action=argparse.BooleanOptionalAction, default=None)
        p.add_argument("--max-steps", type=int, default=30)
        p.add_argument("--no-images", action="store_true", help="text-only observations")
        p.add_argument("--video", action="store_true", help="record robot-tracking video (video.mp4)")
        p.add_argument("--seed", type=int, default=None, help="randomize start heading and object positions")
        p.add_argument("--explain", action="store_true", help="model adds a one-line reason to every tool call")
        if name == "eval":
            p.add_argument("--episodes", type=int, default=5)
    d = sub.add_parser("demo", help="render an annotated demo video from a run recorded with --video")
    d.add_argument("run_dir")
    d.add_argument("--out", default=None, help="output path, default <run_dir>/demo.mp4")
    d.add_argument("--think-s", type=float, default=2.5, help="seconds shown per model decision")
    d.add_argument("--speed", type=float, default=1.5, help="playback speed of robot motion")
    args = ap.parse_args()
    if args.cmd == "demo":
        from pathlib import Path

        from .demo import make_demo

        run_dir = Path(args.run_dir)
        out = make_demo(run_dir, Path(args.out) if args.out else run_dir / "demo.mp4", args.think_s, args.speed)
        print(f"demo: {out}")
        return

    stamp = time.strftime("%Y%m%d-%H%M%S")
    instruction = args.instruction or TASKS[args.task].instruction
    task_name = args.task
    tag = args.model.replace(":", "_").replace("/", "_") if args.planner == "ollama" else "scripted"
    if args.cmd == "run":
        out_dir = RUNS / f"{stamp}_{task_name}_{tag}"
        out = episode(args, instruction, task_name, out_dir, seed=args.seed)
        print(json.dumps(out, indent=2, default=str))
        print(f"logs: {out_dir}")
    else:
        rows = []
        for i in range(args.episodes):
            out_dir = RUNS / f"{stamp}_eval_{task_name}_{tag}" / f"ep{i:02d}"
            seed = None if args.seed is None else args.seed + i
            out = episode(args, instruction, task_name, out_dir, verbose=False, seed=seed)
            rows.append(out)
            print(f"ep{i} seed={seed}: task_success={out['task_success']} claimed={out['success_claimed']} steps={out['steps']} "
                  f"llm={out['llm_time_s']:.0f}s wall={out['wall_time_s']:.0f}s invalid={out['invalid_calls']} :: {out['summary'][:80]}")
        sr = sum(bool(r["task_success"]) for r in rows) / len(rows)
        print(f"success rate {sr:.0%} over {len(rows)} episodes")


if __name__ == "__main__":
    main()
