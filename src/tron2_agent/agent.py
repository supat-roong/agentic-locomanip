"""Agentic harness: a (local) VLM plans by calling robot skills as tools.

Loop: observation (state + optional camera image) -> model -> tool calls ->
validate -> execute skill in sim -> result back to model. Same shape as the
GPT-6 Astra / ROBOTIS AI Worker harness, but with coarse skills so a small
local model only has to sequence them.
"""

from __future__ import annotations

import copy
import io
import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from PIL import Image

from .skills import Skills

SYSTEM_PROMPT = """You control a LimX TRON2 wheel-legged robot with one 6-DoF arm and a parallel gripper, in a simulated room.
You act ONLY by calling tools. Call one tool at a time and read its JSON result before deciding the next step.

World frame: x/y in metres on the floor, yaw in degrees (0 = +x, 90 = +y). Tables are 0.62 m high.
Skills:
- detect(): items visible in the head camera now, plus 'remembered' items seen earlier. Objects and the bin give world_xyz and the table they are 'on'; tables give their x_range and y_range.
- look(camera): get a camera image ("head" or "wrist") to check the scene visually.
- search(name): turn in place until the named item is in view. Use it when an item is not visible or not remembered.
- rotate(angle_deg): turn in place (positive = left).
- navigate_to(x, y, yaw_deg): drive to a floor point and face yaw_deg. It routes around tables by itself. The point must be on the floor, at least 0.15 m outside every table edge.
- pick(name): grasp an object. The robot must stand with the object about 0.4 m straight ahead (reach: 0.18-0.55 m forward, at most 0.25 m sideways).
- place(target): put the held object into the bin. Same reach rule as pick.
- get_state(), stow_arm(), open_gripper()
- done(success, summary): call when the task is finished or impossible.

The arm can only reach over a table edge, so to pick or place you must work out where to stand:
1. Get the item's world_xyz (ix, iy) and its table's x_range [x0, x1] and y_range [y0, y1] from detect.
2. Compute the item's distance to all four edges: ix - x0, x1 - ix, iy - y0, y1 - iy. The smallest one is the closest edge.
3. Stand 0.40 m from the item, straight out past that edge, facing the item:
   closest x0: (ix - 0.40, iy), yaw 0     closest x1: (ix + 0.40, iy), yaw 180
   closest y0: (ix, iy - 0.40), yaw 90    closest y1: (ix, iy + 0.40), yaw -90
Example A: item (3.15, -1.00), x_range [3.00, 3.60], y_range [-1.45, -0.55]. Distances 0.15, 0.45, 0.45, 0.45, so x0 is closest: navigate_to(x=2.75, y=-1.00, yaw_deg=0).
Example B: item (-1.00, 3.10), x_range [-1.30, -0.70], y_range [2.90, 3.80]. Distances 0.30, 0.30, 0.20, 0.70, so y0 is closest: navigate_to(x=-1.00, y=2.70, yaw_deg=90).
The robot holds one object at a time. Plan for "put X in the bin": detect -> navigate next to X -> pick X -> search bin (if not visible or remembered) -> navigate next to the bin -> place bin. For several objects, repeat this for each one, then call done.
If a skill fails, read the message and recover; positions in the message tell you how far off you are. Do not repeat a call that just failed with the same arguments. Never invent names: use names returned by detect or search."""


def _fn(name: str, desc: str, props: dict | None = None, required: list[str] | None = None) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": desc,
            "parameters": {"type": "object", "properties": props or {}, "required": required or []},
        },
    }


TOOLS = [
    _fn("detect", "List objects, bins and tables visible in the head camera with positions."),
    _fn("get_state", "Robot base pose, held object, gripper opening."),
    _fn(
        "look",
        "Capture a camera image and attach it to the conversation.",
        {"camera": {"type": "string", "enum": ["head", "wrist"]}},
        ["camera"],
    ),
    _fn("search", "Turn in place until the named item is in view.", {"name": {"type": "string"}}, ["name"]),
    _fn("rotate", "Turn in place by angle_deg (positive = counter-clockwise/left).", {"angle_deg": {"type": "number"}}, ["angle_deg"]),
    _fn(
        "navigate_to",
        "Drive to a floor point (x, y) and face yaw_deg; routes around tables. Must be at least 0.15 m outside table edges.",
        {"x": {"type": "number"}, "y": {"type": "number"}, "yaw_deg": {"type": "number"}},
        ["x", "y"],
    ),
    _fn("pick", "Grasp and lift an object about 0.4 m straight ahead of the robot.", {"name": {"type": "string"}}, ["name"]),
    _fn("place", "Place the held object into a bin about 0.4 m straight ahead of the robot.", {"target": {"type": "string"}}, ["target"]),
    _fn("stow_arm", "Move the arm to its stowed pose."),
    _fn("open_gripper", "Open the gripper (drops anything held)."),
    _fn(
        "done",
        "Finish the episode.",
        {"success": {"type": "boolean"}, "summary": {"type": "string"}},
        ["success", "summary"],
    ),
]
TOOL_SCHEMAS = {t["function"]["name"]: t["function"]["parameters"] for t in TOOLS}

EXPLAIN_PROMPT = (
    "\nEvery tool has a 'reason' argument: fill it with one short sentence explaining why this call is the right next step."
)


def with_reason(tools: list[dict]) -> list[dict]:
    """Copy of the tool list where every tool takes a required 'reason' string (for demos and debugging)."""
    out = copy.deepcopy(tools)
    for t in out:
        params = t["function"]["parameters"]
        params["properties"]["reason"] = {
            "type": "string",
            "description": "One short sentence: why this call is the right next step.",
        }
        params["required"] = [*params["required"], "reason"]
    return out


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


@dataclass
class PlannerReply:
    content: str
    tool_calls: list[ToolCall]
    latency_s: float = 0.0


class Planner(Protocol):
    def __call__(self, messages: list[dict], tools: list[dict]) -> PlannerReply: ...


# ------------------------------------------------------------------ planners
class OllamaPlanner:
    def __init__(self, model: str, host: str | None = None, temperature: float = 0.2, num_ctx: int = 16384, think: bool | None = None):
        import ollama

        self.client = ollama.Client(host=host) if host else ollama.Client()
        self.model, self.think = model, think
        self.options = {"temperature": temperature, "num_ctx": num_ctx}

    def __call__(self, messages, tools) -> PlannerReply:
        t0 = time.time()
        kw = {"think": self.think} if self.think is not None else {}
        r = self.client.chat(model=self.model, messages=messages, tools=tools, options=self.options, **kw)
        calls = [ToolCall(tc.function.name, dict(tc.function.arguments or {})) for tc in (r.message.tool_calls or [])]
        return PlannerReply(r.message.content or "", calls, time.time() - t0)


class ScriptedPlanner:
    """Deterministic baseline/test double: replays a fixed tool sequence."""

    def __init__(self, calls: list[tuple[str, dict | Callable[[], dict]]]):
        self.calls = list(calls)

    def __call__(self, messages, tools) -> PlannerReply:
        if not self.calls:
            return PlannerReply("", [ToolCall("done", {"success": True, "summary": "script finished"})])
        name, args = self.calls.pop(0)
        return PlannerReply("", [ToolCall(name, args() if callable(args) else dict(args))])


def scripted_pick_place(skills: Skills, obj: str, target: str = "bin") -> ScriptedPlanner:
    """Baseline that knows the geometry: stands at Skills.standoff_pose of each item."""

    def stand_at(name: str):
        def args() -> dict:
            sp = skills.standoff_pose(name)
            if isinstance(sp, str):
                raise RuntimeError(sp)
            (x, y), yaw = sp
            return {"x": float(x), "y": float(y), "yaw_deg": float(np.degrees(yaw))}

        return args

    return ScriptedPlanner(
        [
            ("search", {"name": obj}),
            ("navigate_to", stand_at(obj)),
            ("pick", {"name": obj}),
            ("search", {"name": target}),
            ("navigate_to", stand_at(target)),
            ("place", {"target": target}),
        ]
    )


# ---------------------------------------------------------------- validation
_JSON_TYPES = {"number": (int, float), "string": str, "boolean": bool}


def validate_call(call: ToolCall) -> str | None:
    """Return an error message if the call does not match its schema."""
    schema = TOOL_SCHEMAS.get(call.name)
    if schema is None:
        return f"unknown tool '{call.name}'. Available: {sorted(TOOL_SCHEMAS)}"
    props = schema["properties"]
    for k in schema["required"]:
        if k not in call.args:
            return f"{call.name}: missing required argument '{k}'"
    for k, v in call.args.items():
        if k not in props:
            return f"{call.name}: unexpected argument '{k}'"
        typ = props[k]["type"]
        if typ == "number" and isinstance(v, str):
            try:
                call.args[k] = float(v)
                continue
            except ValueError:
                pass
        if typ == "boolean" and isinstance(v, str) and v.lower() in ("true", "false"):
            call.args[k] = v.lower() == "true"
            continue
        if not isinstance(call.args[k], _JSON_TYPES[typ]) or (typ == "number" and isinstance(v, bool)):
            return f"{call.name}: argument '{k}' must be {typ}, got {v!r}"
        if "enum" in props[k] and v not in props[k]["enum"]:
            return f"{call.name}: argument '{k}' must be one of {props[k]['enum']}"
    if call.name == "rotate" and abs(call.args["angle_deg"]) > 360:
        return "rotate: |angle_deg| must be <= 360"
    if call.name == "navigate_to" and (abs(call.args["x"]) > 10 or abs(call.args["y"]) > 10):
        return "navigate_to: target outside the 10 m work area"
    return None


# -------------------------------------------------------------------- runner
@dataclass
class EpisodeResult:
    success_claimed: bool | None
    summary: str
    steps: int
    sim_time_s: float
    wall_time_s: float
    llm_time_s: float
    invalid_calls: int
    failed_skills: int
    trace: list[dict] = field(default_factory=list)


def _png(img: np.ndarray, max_side: int = 448) -> bytes:
    im = Image.fromarray(img)
    im.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


class AgentRunner:
    def __init__(
        self,
        skills: Skills,
        planner: Planner,
        max_steps: int = 30,
        max_sim_s: float = 600.0,
        use_images: bool = True,
        log_dir: Path | None = None,
        verbose: bool = True,
        explain: bool = False,
    ):
        self.skills, self.planner = skills, planner
        self.explain = explain
        self.tools = with_reason(TOOLS) if explain else TOOLS
        self.max_steps, self.max_sim_s = max_steps, max_sim_s
        self.use_images, self.log_dir, self.verbose = use_images, log_dir, verbose
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)

    def _log(self, rec: dict) -> None:
        if self.verbose:
            short = json.dumps(rec, default=str)
            print(short if len(short) < 600 else short[:600] + " ...")
        if self.log_dir:
            with open(self.log_dir / "trace.jsonl", "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")

    def _image_msg(self, camera: str, step: int) -> dict:
        img = self.skills.sim.render(camera)
        if self.log_dir:
            Image.fromarray(img).save(self.log_dir / f"step{step:02d}_{camera}.png")
        return {"role": "user", "content": f"[{camera} camera image]", "images": [_png(img)]}

    def _execute(self, call: ToolCall) -> dict:
        sk = self.skills
        a = call.args
        match call.name:
            case "detect":
                return sk.detect()
            case "get_state":
                return sk.get_state()
            case "search":
                return sk.search(str(a["name"]))
            case "rotate":
                return sk.rotate(float(a["angle_deg"]))
            case "navigate_to":
                return sk.navigate_to(float(a["x"]), float(a["y"]), a.get("yaw_deg"))
            case "pick":
                return sk.pick(str(a["name"]))
            case "place":
                return sk.place(str(a["target"]))
            case "stow_arm":
                return sk.stow_arm()
            case "open_gripper":
                return sk.open_gripper()
            case "look":
                return {"success": True, "message": f"{a['camera']} image attached below"}
        return {"success": False, "message": f"no executor for {call.name}"}

    @staticmethod
    def _drop_old_images(messages: list[dict], keep: int = 1) -> None:
        seen = 0
        for m in reversed(messages):
            if m.get("images"):
                seen += 1
                if seen > keep:
                    m.pop("images")
                    m["content"] += " (image removed to save context)"

    def run(self, instruction: str) -> EpisodeResult:
        t_wall = time.time()
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT + (EXPLAIN_PROMPT if self.explain else "")}]
        first = {"role": "user", "content": f"Task: {instruction}\nCurrent state: {json.dumps(self.skills.get_state())}"}
        if self.use_images:
            first["images"] = self._image_msg("head", 0)["images"]
        msgs.append(first)
        llm_time, invalid, failed, nudges = 0.0, 0, 0, 0
        fail_counts: dict[str, int] = {}
        tool_fail_streak: dict[str, int] = {}
        claimed, summary, trace = None, "step limit reached", []
        step = 0
        for step in range(1, self.max_steps + 1):
            reply = self.planner(msgs, self.tools)
            llm_time += reply.latency_s
            assistant = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                assistant["tool_calls"] = [
                    {"function": {"name": c.name, "arguments": dict(c.args)}} for c in reply.tool_calls
                ]
            msgs.append(assistant)
            if not reply.tool_calls:
                nudges += 1
                self._log({"step": step, "assistant_text": reply.content, "latency_s": round(reply.latency_s, 2)})
                if nudges > 3:
                    summary = "model stopped calling tools"
                    break
                msgs.append({"role": "user", "content": "Respond by calling exactly one tool. Call done(success, summary) when finished."})
                continue
            call = reply.tool_calls[0]  # one action per turn
            if len(reply.tool_calls) > 1:
                note = f" (only the first of {len(reply.tool_calls)} tool calls was executed; call one tool at a time)"
            else:
                note = ""
            reason = str(call.args.pop("reason", "") or "")
            err = validate_call(call)
            t0, sim_t0 = time.time(), self.skills.sim.time
            if err:
                invalid += 1
                result = {"success": False, "message": "invalid call: " + err}
            elif call.name == "done":
                claimed, summary = bool(call.args["success"]), str(call.args["summary"])
                result = {"success": True, "message": "episode finished"}
            else:
                result = self._execute(call)
                if not result.get("success", False):
                    failed += 1
            rec = {
                "step": step,
                "tool": call.name,
                "args": call.args,
                "result": result,
                "reason": reason,
                "thought": reply.content[:500],
                "llm_s": round(reply.latency_s, 2),
                "skill_s": round(time.time() - t0, 2),
                "sim_t0": round(sim_t0, 3),
                "sim_t": round(self.skills.sim.time, 3),
            }
            if not result.get("success", False):
                key = call.name + json.dumps(call.args, sort_keys=True)
                fail_counts[key] = fail_counts.get(key, 0) + 1
                tool_fail_streak[call.name] = tool_fail_streak.get(call.name, 0) + 1
                if fail_counts[key] >= 2 or tool_fail_streak[call.name] >= 2:
                    result["hint"] = (
                        f"{call.name} has failed {tool_fail_streak[call.name]} times in a row. Do what the message says, "
                        "or try a different step of the plan."
                    )
            elif call.name != "done":
                tool_fail_streak[call.name] = 0
            trace.append(rec)
            self._log(rec)
            msgs.append({"role": "tool", "tool_name": call.name, "content": json.dumps(result) + note})
            if call.name == "done":
                break
            if call.name == "look" and not err:
                msgs.append(self._image_msg(call.args["camera"], step))
            elif self.use_images and call.name in ("rotate", "search", "navigate_to", "pick", "place"):
                msgs.append(self._image_msg("head", step))
            self._drop_old_images(msgs)
            if self.skills.sim.fallen():
                summary = "robot fell over"
                break
            if self.skills.sim.time > self.max_sim_s:
                summary = "sim time limit reached"
                break
        return EpisodeResult(claimed, summary, step, self.skills.sim.time, time.time() - t_wall, llm_time, invalid, failed, trace)
