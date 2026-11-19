"""Upload the repo source as a Kaggle dataset and push the demo notebook (T4 GPU). Both are private unless --public.

  uv run python tools/kaggle/push.py --user <kaggle-username>
  uv run python tools/kaggle/push.py --user <kaggle-username> --public
  uv run python tools/kaggle/push.py --user <kaggle-username> --skip-dataset   # notebook only

Uses the Kaggle CLI through `uvx --with six kaggle` (credentials in ~/.kaggle).
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
DATASET_SLUG = "agentic-locomanip-src"
KERNEL_FILE = "agentic-locomanip-demo.ipynb"
KERNEL_TITLE = "Agentic Loco-Manipulation with a Local VLM"
KERNEL_SLUG = "agentic-loco-manipulation-with-a-local-vlm"  # Kaggle derives the slug from the title
INCLUDE = ["pyproject.toml", "README.md", "LICENSE", "src", "scripts"]


def kaggle(*args: str) -> str:
    cmd = ["uvx", "--with", "six", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print("$ kaggle", " ".join(args), "\n", out[-1500:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return out


def push_dataset(user: str, public: bool) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        with tarfile.open(d / "agentic-locomanip.tar.gz", "w:gz") as tar:
            for name in INCLUDE:
                tar.add(ROOT / name, arcname=f"agentic-locomanip/{name}",
                        filter=lambda ti: None if "__pycache__" in ti.name else ti)
        meta = {"title": "agentic-locomanip source", "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
        (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
        exists = subprocess.run(["uvx", "--with", "six", "kaggle", "datasets", "status", f"{user}/{DATASET_SLUG}"],
                                capture_output=True, text=True).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", str(d), "-m", "update source")
        else:
            kaggle("datasets", "create", "-p", str(d), *(["--public"] if public else []))


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
            "dataset_sources": [f"{user}/{DATASET_SLUG}"],
            "competition_sources": [],
            "kernel_sources": [],
        }
        (d / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
        kaggle("kernels", "push", "-p", str(d))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--skip-dataset", action="store_true")
    ap.add_argument("--public", action="store_true", help="make the dataset and notebook public")
    a = ap.parse_args()
    if not a.skip_dataset:
        push_dataset(a.user, a.public)
    push_kernel(a.user, a.public)


if __name__ == "__main__":
    main()
