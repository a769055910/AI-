#!/usr/bin/env bash
# Run once on the AutoDL GPU host before restarting gpu_server.py.
# TruFor is restricted by its upstream licence to informational/non-profit use.
set -euo pipefail

ROOT="${TRUFOR_INSTALL_ROOT:-/root/forge-detector/vendor/TruFor}"
CONDA_BIN="${CONDA_BIN:-/root/miniconda3/bin/conda}"
if [ ! -d "$ROOT/.git" ]; then
  git clone --depth 1 https://github.com/grip-unina/TruFor.git "$ROOT"
fi
cd "$ROOT/test_docker"
# The upstream YAML builds mmcv-full/jpegio from source.  That build targets
# CUDA 11.3 and fails against this host's CUDA 12.1 compiler even though the
# inference path does not import either package.  Keep the inference-only env
# minimal and isolated from the active GPU service.
if [ "${TRUFOR_REBUILD:-0}" = "1" ] && "$CONDA_BIN" env list | awk '{print $1}' | grep -qx trufor; then
  "$CONDA_BIN" env remove -n trufor -y
fi
if ! "$CONDA_BIN" env list | awk '{print $1}' | grep -qx trufor; then
  "$CONDA_BIN" create -n trufor -y python=3.7 pytorch=1.11 torchvision=0.12 cudatoolkit=11.3 -c pytorch -c conda-forge
fi
"$CONDA_BIN" run -n trufor python -m pip install --no-cache-dir --timeout 60 \
  --index-url https://pypi.tuna.tsinghua.edu.cn/simple \
  albumentations==1.2.1 yacs==0.1.8 timm==0.5.4 opencv-python==4.5.5.64 \
  numpy==1.21.5 scipy==1.7.3 scikit-image==0.16.2 tqdm==4.64.0 pyyaml==6.0
WEIGHTS_DIR="$ROOT/TruFor_train_test/pretrained_models"
mkdir -p "$WEIGHTS_DIR"
if [ ! -f "$WEIGHTS_DIR/trufor.pth.tar" ]; then
  wget -O /tmp/TruFor_weights.zip https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
  unzip -o /tmp/TruFor_weights.zip -d "$WEIGHTS_DIR"
fi
if [ -f "$WEIGHTS_DIR/weights/trufor.pth.tar" ] && [ ! -f "$WEIGHTS_DIR/trufor.pth.tar" ]; then
  ln -s "weights/trufor.pth.tar" "$WEIGHTS_DIR/trufor.pth.tar"
fi
echo "TruFor environment and weights are ready. Use /root/miniconda3/envs/trufor/bin/python."
