from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "assets"
ROBOT_XML = ASSETS / "tron2-robot-description/tron2a/WFYG_TRON2A/xml/robot.xml"
POLICY_DIR = ASSETS / "policy/WF_TRON2A"
RUNS = ROOT / "runs"


def check_assets() -> None:
    missing = [p for p in (ROBOT_XML, POLICY_DIR / "policy.onnx") if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"missing assets {[str(p) for p in missing]}; run scripts/fetch_assets.sh"
        )
