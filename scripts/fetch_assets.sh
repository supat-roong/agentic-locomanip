#!/usr/bin/env bash
# Fetch third-party assets (Apache-2.0, LimX Dynamics) into assets/.
#   - TRON2A WFYG robot description (MJCF + meshes) via sparse checkout
#   - Pretrained WF_TRON2A wheel-leg locomotion policy (ONNX)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ASSETS="$ROOT/assets"
DESC_REF="${TRON2_DESC_REF:-main}"
DEPLOY_REF="${TRON2_DEPLOY_REF:-master}"

mkdir -p "$ASSETS"

if [ ! -d "$ASSETS/tron2-robot-description/.git" ]; then
  git clone --quiet --depth 1 --filter=blob:none --sparse --branch "$DESC_REF" \
    https://github.com/limxdynamics/tron2-robot-description.git \
    "$ASSETS/tron2-robot-description"
  git -C "$ASSETS/tron2-robot-description" sparse-checkout set \
    tron2a/WFYG_TRON2A/xml tron2a/WFYG_TRON2A/meshes
fi

POLICY_DIR="$ASSETS/policy/WF_TRON2A"
mkdir -p "$POLICY_DIR"
BASE="https://raw.githubusercontent.com/limxdynamics/tron2_rl_deploy_python/$DEPLOY_REF/controllers/model/WF_TRON2A"
for f in policy.onnx encoder.onnx params.yaml; do
  [ -f "$POLICY_DIR/$f" ] || curl -fsSL "$BASE/$f" -o "$POLICY_DIR/$f"
done

echo "assets ready in $ASSETS"
