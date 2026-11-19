"""Push the demo notebook to Kaggle (T4 GPU, internet on). It clones the source from GitHub. Private unless --public.

  uv run python tools/kaggle/push.py --user <kaggle-username>
  uv run python tools/kaggle/push.py --user <kaggle-username> --public

Uses the Kaggle CLI through `uvx --with six kaggle` (credentials in ~/.kaggle).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
KERNEL_FILE = "agentic-locomanip-demo.ipynb"
KERNEL_TITLE = "Agentic Loco-Manipulation with a Local VLM"
KERNEL_SLUG = "agentic-loco-manipulation-with-a-local-vlm"  # Kaggle derives the slug from the title


def kaggle(*args: str) -> str:
    cmd = ["uvx", "--with", "six", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print("$ kaggle", " ".join(args), "\n", out[-1500:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return out


def push_kernel(user: str, public: bool) -> None:
    subprocess.run(["python3", str(HERE / "build_notebook.py")], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(HERE / "notebook" / KERNEL_FILE, d)
        meta = {
            "id": f"{user}/{KERNEL_SLUG}",
            "title": KERNEL_TITLE,
            "code_file": KERNEL_FILE,
            "language": "python",
            "kernel_type": "notebook",
            "is_private": not public,
            "enable_gpu": True,
            "enable_internet": True,
            "machine_shape": "NvidiaTeslaT4",
            "dataset_sources": [],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        kaggle("kernels", "push", "-p", str(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--public", action="store_true", help="make the notebook public")
    a = ap.parse_args()
    push_kernel(a.user, a.public)


if __name__ == "__main__":
    main()
