"""Agentic harness: a (local) VLM plans by calling robot skills as tools.

Loop: observation (state + optional camera image) -> model -> tool calls ->
validate -> execute skill in sim -> result back to model. Same shape as the
GPT-6 Astra / ROBOTIS AI Worker harness, but with coarse skills so a small
local model only has to sequence them.
"""

from __future__ import annotations

import io
import json
import time
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
- detect(): items visible in the head camera now, plus 'remembered' items seen earlier, with world and robot-relative positions.
- look(camera): get a camera image ("head" or "wrist") to check the scene visually.
- search(name): turn in place until the named item is in view. Use it when an item is not visible or not remembered.
- rotate(angle_deg): turn in place (positive = left).
- navigate_to(x, y, yaw_deg): drive to a floor position. Fails if the straight path hits a table.
- approach(name): drive to a manipulation pose in front of an object or bin (handles routing around tables).
- pick(name): grasp an object; you must have called approach(name) first.
- place(target): put the held object into a bin; call approach(target) first.
- get_state(), stow_arm(), open_gripper()
- done(success, summary): call when the task is finished or impossible.

Typical plan for "put X in the bin": detect -> approach X -> pick X -> search bin (if not visible/remembered) -> approach bin -> place bin -> done.
approach() plans its own route around tables, so prefer it over navigate_to.
If a skill fails, read the message and recover (e.g. search, approach again). Do not repeat a call that just failed with the same arguments. Never invent names: use names returned by detect or search."""


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
        "Drive to world position (x, y) and optionally face yaw_deg.",
        {"x": {"type": "number"}, "y": {"type": "number"}, "yaw_deg": {"type": "number"}},
        ["x", "y"],
    ),
    _fn("approach", "Drive to a manipulation pose in front of an object or bin.", {"name": {"type": "string"}}, ["name"]),
    _fn("pick", "Grasp and lift an object (must approach it first).", {"name": {"type": "string"}}, ["name"]),
    _fn("place", "Place the held object into a bin (must approach the bin first).", {"target": {"type": "string"}}, ["target"]),
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

    def __init__(self, calls: list[tuple[str, dict]]):
        self.calls = list(calls)

    def __call__(self, messages, tools) -> PlannerReply:
        if not self.calls:
            return PlannerReply("", [ToolCall("done", {"success": True, "summary": "script finished"})])
        name, args = self.calls.pop(0)
        return PlannerReply("", [ToolCall(name, args)])


def scripted_pick_place(obj: str, target: str = "bin") -> ScriptedPlanner:
    return ScriptedPlanner(
        [
            ("search", {"name": obj}),
            ("approach", {"name": obj}),
            ("pick", {"name": obj}),
            ("search", {"name": target}),
            ("approach", {"name": target}),
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
    ):
        self.skills, self.planner = skills, planner
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
            case "approach":
                return sk.approach(str(a["name"]))
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
        msgs: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
        first = {"role": "user", "content": f"Task: {instruction}\nCurrent state: {json.dumps(self.skills.get_state())}"}
        if self.use_images:
            first["images"] = self._image_msg("head", 0)["images"]
        msgs.append(first)
        llm_time, invalid, failed, nudges = 0.0, 0, 0, 0
        fail_counts: dict[str, int] = {}
        claimed, summary, trace = None, "step limit reached", []
        step = 0
        for step in range(1, self.max_steps + 1):
            reply = self.planner(msgs, TOOLS)
            llm_time += reply.latency_s
            assistant = {"role": "assistant", "content": reply.content}
            if reply.tool_calls:
                assistant["tool_calls"] = [{"function": {"name": c.name, "arguments": c.args}} for c in reply.tool_calls]
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
            err = validate_call(call)
            t0 = time.time()
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
                "thought": reply.content[:500],
                "llm_s": round(reply.latency_s, 2),
                "skill_s": round(time.time() - t0, 2),
                "sim_t": round(self.skills.sim.time, 1),
            }
            if not result.get("success", False):
                key = call.name + json.dumps(call.args, sort_keys=True)
                fail_counts[key] = fail_counts.get(key, 0) + 1
                if fail_counts[key] >= 2:
                    result["hint"] = (
                        f"this exact call has failed {fail_counts[key]} times; do something different "
                        "(e.g. approach(name) to route around tables, or search(name))"
                    )
            trace.append(rec)
            self._log(rec)
            msgs.append({"role": "tool", "tool_name": call.name, "content": json.dumps(result) + note})
            if call.name == "done":
                break
            if call.name == "look" and not err:
                msgs.append(self._image_msg(call.args["camera"], step))
            elif self.use_images and call.name in ("rotate", "search", "navigate_to", "approach", "pick", "place"):
                msgs.append(self._image_msg("head", step))
            self._drop_old_images(msgs)
            if self.skills.sim.fallen():
                summary = "robot fell over"
                break
            if self.skills.sim.time > self.max_sim_s:
                summary = "sim time limit reached"
                break
        return EpisodeResult(claimed, summary, step, self.skills.sim.time, time.time() - t_wall, llm_time, invalid, failed, trace)
