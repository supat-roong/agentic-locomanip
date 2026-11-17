"""Compose an annotated demo video from a run directory.

Needs a run recorded with `--video` (video.mp4 + frames.json) and trace.jsonl;
`--explain` adds the model's one-line reason per tool call. Layout (1280x720):
the robot video on the left and, on the right, the task prompt, the camera
image the model saw, its reason, the tool call it chose, the result and a step
log. Each model decision is shown as a short freeze ("thinking") followed by
the skill executing in sim time.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H, FPS = 1280, 720, 15
TOP = 52
VID_W, VID_H = 852, 639
PANEL_X = VID_W + 20
PANEL_W = W - PANEL_X - 18

BG = (16, 20, 25)
PANEL = (23, 30, 37)
FG = (228, 234, 238)
MUTED = (140, 155, 168)
ACCENT = (76, 194, 176)
OK = (92, 207, 138)
BAD = (240, 104, 96)
THINK = (246, 196, 92)


def _font(size: int, mono: bool = False, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = (
        ["/System/Library/Fonts/Menlo.ttc", "DejaVuSansMono.ttf"]
        if mono
        else ["/System/Library/Fonts/HelveticaNeue.ttc", "/System/Library/Fonts/Helvetica.ttc", "DejaVuSans.ttf"]
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size, index=(1 if bold and "Helvetica" in path else 0))
        except OSError:
            continue
    return ImageFont.load_default(size)


F_TITLE = _font(20, bold=True)
F_LABEL = _font(12, bold=True)
F_BODY = _font(16)
F_SMALL = _font(13)
F_MONO = _font(15, mono=True)
F_MONO_S = _font(12, mono=True)
F_PILL = _font(15, bold=True)


@dataclass
class Step:
    n: int
    tool: str
    args: dict
    reason: str
    result: dict
    llm_s: float
    t0: float
    t1: float
    seen_image: Path | None


def _call_str(tool: str, args: dict) -> str:
    def fmt(v):
        return f'"{v}"' if isinstance(v, str) else (f"{v:g}" if isinstance(v, float) else str(v).lower() if isinstance(v, bool) else str(v))

    if tool == "done":
        return f"done(success={fmt(args.get('success'))})"
    return f"{tool}(" + ", ".join(f"{k}={fmt(v)}" for k, v in args.items()) + ")"


def _wrap(text: str, font, width: int) -> list[str]:
    avg = max(font.getlength("abcdefghijklmnopqrstuvwxyz") / 26, 1)
    return textwrap.wrap(text, max(int(width / avg), 10)) or [""]


def _load_steps(run_dir: Path) -> list[Step]:
    images = sorted(run_dir.glob("step*_*.png"))

    def seen_before(n: int) -> Path | None:
        prior = [p for p in images if int(p.name[4:6]) < n]
        return prior[-1] if prior else None

    steps, prev_t = [], None
    for line in (run_dir / "trace.jsonl").read_text().splitlines():
        r = json.loads(line)
        if "tool" not in r:
            continue
        t1 = float(r["sim_t"])
        t0 = float(r.get("sim_t0", prev_t if prev_t is not None else t1))
        steps.append(
            Step(r["step"], r["tool"], r["args"], r.get("reason") or r.get("thought") or "", r["result"],
                 float(r.get("llm_s", 0)), t0, t1, seen_before(r["step"]))
        )
        prev_t = t1
    return steps


class Composer:
    def __init__(self, run_dir: Path, steps: list[Step], meta: dict):
        self.steps, self.meta = steps, meta
        self.times = np.array(json.loads((run_dir / "frames.json").read_text()))
        self.frames = imageio.get_reader(str(run_dir / "video.mp4"))
        self._cache: dict[int, np.ndarray] = {}
        self._thumbs: dict[Path, Image.Image] = {}

    def sim_frame(self, t: float) -> Image.Image:
        i = int(np.clip(np.searchsorted(self.times, t), 0, len(self.times) - 1))
        if i not in self._cache:
            self._cache = {i: self.frames.get_data(i)}
        return Image.fromarray(self._cache[i]).resize((VID_W, VID_H), Image.BILINEAR)

    def thumb(self, p: Path | None, w: int) -> Image.Image | None:
        if p is None:
            return None
        if p not in self._thumbs:
            im = Image.open(p).convert("RGB")
            self._thumbs[p] = im.resize((w, int(w * im.height / im.width)), Image.BILINEAR)
        return self._thumbs[p]

    # ------------------------------------------------------------- drawing
    def _pill(self, d: ImageDraw.ImageDraw, xy, text: str, color, dark=False) -> None:
        x, y = xy
        tw = d.textlength(text, font=F_PILL)
        d.rounded_rectangle([x, y, x + tw + 24, y + 30], radius=15, fill=(0, 0, 0) if dark else color)
        d.text((x + 12, y + 6), text, font=F_PILL, fill=color if dark else BG)

    def frame(self, t_sim: float, idx: int, phase: str, progress: float, done_card: bool = False) -> np.ndarray:
        img = Image.new("RGB", (W, H), BG)
        img.paste(self.sim_frame(t_sim), (0, TOP))
        d = ImageDraw.Draw(img)

        # top bar
        d.text((18, 15), "TRON2 WFYG  ·  local VLM agent", font=F_TITLE, fill=FG)
        right = f"{self.meta['model']}   sim {t_sim:5.1f} s"
        d.text((W - 18 - d.textlength(right, font=F_MONO), 18), right, font=F_MONO, fill=MUTED)

        st = self.steps[idx]
        # status pill on the video
        if done_card:
            pass
        elif phase == "think":
            self._pill(d, (16, TOP + 14), f"MODEL THINKING  ·  {st.llm_s:.0f} s real time (paused)", THINK, dark=True)
        else:
            self._pill(d, (16, TOP + 14), f"EXECUTING  {_call_str(st.tool, st.args)}", ACCENT, dark=True)

        # right panel
        d.rectangle([VID_W, TOP, W, H], fill=PANEL)
        x, y = PANEL_X, TOP + 14
        d.text((x, y), "TASK PROMPT", font=F_LABEL, fill=MUTED)
        y += 18
        for ln in _wrap(self.meta["instruction"], F_BODY, PANEL_W)[:2]:
            d.text((x, y), ln, font=F_BODY, fill=FG)
            y += 21
        y += 10

        d.text((x, y), "WHAT THE MODEL SEES (head camera)", font=F_LABEL, fill=MUTED)
        y += 18
        th = self.thumb(st.seen_image, 300)
        if th is not None:
            img.paste(th, (x, y))
            y += th.height
        y += 12

        n_tools = len(self.steps)
        label_color = THINK if phase == "think" else ACCENT
        d.text((x, y), f"STEP {st.n} / {n_tools}  ·  {'DECIDING' if phase == 'think' else 'RUNNING'}", font=F_LABEL, fill=label_color)
        y += 18
        reason = st.reason or "(no reason given)"
        if phase == "think":  # type the reason in
            reason = reason[: max(1, int(len(reason) * min(1.0, progress * 1.6)))]
        for ln in _wrap(f"“{reason}”", F_BODY, PANEL_W)[:3]:
            d.text((x, y), ln, font=F_BODY, fill=FG)
            y += 21
        y += 6
        if phase != "think" or progress > 0.6:
            call = _call_str(st.tool, st.args)
            d.rounded_rectangle([x - 6, y - 4, x + PANEL_W, y + 24], radius=6, fill=(13, 17, 21))
            d.text((x, y), ("→ " + call)[:44], font=F_MONO, fill=ACCENT)
        y += 34

        # result of this call once it finishes; while deciding, the previous result the model is reacting to
        shown = st if (phase == "exec" and progress > 0.85) or done_card else None
        if phase == "think" and idx > 0:
            shown = self.steps[idx - 1]
        if shown is not None:
            ok = shown.result.get("success", False)
            prefix = "" if shown is st else f"previous ({shown.tool}): "
            msg = prefix + str(shown.result.get("message", ""))
            color = OK if ok else BAD
            d.text((x, y), "✓" if ok else "✗", font=F_MONO_S, fill=color)
            for ln in _wrap(msg, F_MONO_S, PANEL_W - 22)[:2]:
                d.text((x + 18, y), ln, font=F_MONO_S, fill=color)
                y += 16
        else:
            y += 16
        y += 12

        # step log
        d.text((x, y), "TOOL CALL LOG", font=F_LABEL, fill=MUTED)
        y += 18
        first = max(0, idx - (H - y - 10) // 18 + 1)
        for j in range(first, idx + 1):
            s = self.steps[j]
            finished = j < idx or (phase == "exec" and progress > 0.85) or done_card
            mark, col = ("✓", OK) if s.result.get("success") else ("✗", BAD)
            if not finished:
                mark, col = "…", MUTED
            d.text((x, y), f"{s.n:>2} {mark}", font=F_MONO_S, fill=col)
            d.text((x + 42, y), _call_str(s.tool, s.args)[:42], font=F_MONO_S, fill=FG if j == idx else MUTED)
            y += 18

        if done_card:
            ok = self.meta.get("task_success")
            box = [140, TOP + 200, VID_W - 140, TOP + 420]
            d.rounded_rectangle(box, radius=14, fill=(10, 13, 16), outline=OK if ok else BAD, width=3)
            head = "TASK COMPLETE" if ok else "TASK FAILED"
            d.text((box[0] + 28, box[1] + 26), head, font=_font(30, bold=True), fill=OK if ok else BAD)
            summ = self.steps[-1].args.get("summary", "") if self.steps[-1].tool == "done" else self.meta.get("summary", "")
            yy = box[1] + 76
            for ln in _wrap(summ, F_BODY, box[2] - box[0] - 56)[:3]:
                d.text((box[0] + 28, yy), ln, font=F_BODY, fill=FG)
                yy += 22
            stats = (f"{len(self.steps)} tool calls  ·  {self.meta.get('sim_time_s', 0):.0f} s sim  ·  "
                     f"{self.meta.get('wall_time_s', 0) / 60:.1f} min wall on an M4")
            d.text((box[0] + 28, box[3] - 40), stats, font=F_SMALL, fill=MUTED)
        return np.asarray(img)


def make_demo(run_dir: Path, out: Path, think_s: float = 2.5, speed: float = 1.5) -> Path:
    steps = _load_steps(run_dir)
    if not steps:
        raise SystemExit(f"no tool calls in {run_dir / 'trace.jsonl'}")
    if not (run_dir / "frames.json").exists():
        raise SystemExit("frames.json missing: record the run with `--video` (current version)")
    meta = json.loads((run_dir / "result.json").read_text())
    comp = Composer(run_dir, steps, meta)
    w = imageio.get_writer(str(out), fps=FPS, macro_block_size=8, quality=7)
    n_think = int(think_s * FPS)
    for i, st in enumerate(steps):
        for k in range(n_think):
            w.append_data(comp.frame(st.t0, i, "think", k / max(n_think - 1, 1)))
        n_exec = max(int((st.t1 - st.t0) / speed * FPS), int(0.8 * FPS))
        for k in range(n_exec):
            p = k / max(n_exec - 1, 1)
            w.append_data(comp.frame(st.t0 + p * (st.t1 - st.t0), i, "exec", p))
    for _ in range(int(4 * FPS)):
        w.append_data(comp.frame(steps[-1].t1, len(steps) - 1, "exec", 1.0, done_card=True))
    w.close()
    return out
