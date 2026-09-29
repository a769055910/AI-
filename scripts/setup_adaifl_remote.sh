#!/usr/bin/env bash
# Install the official AdaIFL source and checkpoint for inference only.
set -euo pipefail

ROOT="${ADAIFL_ROOT:-/root/forge-detector/vendor/AdaIFL}"
PYTHON_BIN="${ADAIFL_PYTHON:-/root/miniconda3/bin/python}"
CHECKPOINT="${ADAIFL_MODEL_PATH:-/root/autodl-tmp/forge-detector-data/models/AdaIFL/AdaIFL_v0.pth}"
MODEL_ID="187SJ_O0YHP0DVBXgfob_o2BofzCf0TMP"

if [[ ! -d "$ROOT/.git" ]]; then
  mkdir -p "$(dirname "$ROOT")"
  git clone --depth 1 https://github.com/LMIAPC/AdaIFL.git "$ROOT"
else
  git -C "$ROOT" pull --ff-only
fi

# The server already has a CUDA-enabled PyTorch runtime.  Reusing it avoids a
# second multi-GB CUDA environment; AdaIFL's official test script only needs
# torch, torchvision, cv2, numpy and matplotlib at inference time.
"$PYTHON_BIN" -c "import torch, torchvision, cv2, numpy, matplotlib" 
if ! "$PYTHON_BIN" -c "import gdown"; then
  "$PYTHON_BIN" -m pip install --no-cache-dir gdown
fi

if [[ ! -s "$CHECKPOINT" ]]; then
  mkdir -p "$(dirname "$CHECKPOINT")"
  "$PYTHON_BIN" -m gdown --fuzzy "https://drive.google.com/file/d/$MODEL_ID/view" -O "$CHECKPOINT"
fi

test -s "$CHECKPOINT"
echo "AdaIFL ready: $ROOT ; checkpoint=$(du -h "$CHECKPOINT" | awk '{print $1}') ; python=$PYTHON_BIN"
