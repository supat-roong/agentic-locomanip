"""Upload the repo source as a private Kaggle dataset and push the demo notebook (private, T4 GPU).

  uv run python tools/kaggle/push.py --user <kaggle-username>
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
DATASET_SLUG = "tron2-agent-src"
KERNEL_SLUG = "tron2-agent-demo"
INCLUDE = ["pyproject.toml", "README.md", "src", "scripts"]


def kaggle(*args: str) -> str:
    cmd = ["uvx", "--with", "six", "kaggle", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = (r.stdout + r.stderr).strip()
    print("$ kaggle", " ".join(args), "\n", out[-1500:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    return out


def push_dataset(user: str) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        with tarfile.open(d / "tron2-agent.tar.gz", "w:gz") as tar:
            for name in INCLUDE:
                tar.add(ROOT / name, arcname=f"tron2-agent/{name}",
                        filter=lambda ti: None if "__pycache__" in ti.name else ti)
        meta = {"title": DATASET_SLUG, "id": f"{user}/{DATASET_SLUG}", "licenses": [{"name": "other"}]}
        (d / "dataset-metadata.json").write_text(json.dumps(meta, indent=1))
        exists = subprocess.run(["uvx", "--with", "six", "kaggle", "datasets", "status", f"{user}/{DATASET_SLUG}"],
                                capture_output=True, text=True).returncode == 0
        if exists:
            kaggle("datasets", "version", "-p", str(d), "-m", "update source")
        else:
            kaggle("datasets", "create", "-p", str(d))  # private unless --public is given


def push_kernel(user: str) -> None:
    subprocess.run(["python3", str(HERE / "build_notebook.py")], check=True)
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp)
        shutil.copy(HERE / "notebook" / f"{KERNEL_SLUG}.ipynb", d)
        meta = {
            "id": f"{user}/{KERNEL_SLUG}",
            "title": "TRON2 Agent Demo",
            "code_file": f"{KERNEL_SLUG}.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
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
    a = ap.parse_args()
    if not a.skip_dataset:
        push_dataset(a.user)
    push_kernel(a.user)


if __name__ == "__main__":
    main()
