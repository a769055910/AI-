"""Persistent, loopback-only TruFor inference worker.

This process deliberately runs in TruFor's original Conda environment.  It
loads the official checkpoint once at start-up, then accepts raw image bytes
from ``gpu_server.py`` over localhost.  Keeping the environments isolated
avoids dependency conflicts while removing per-image Python/model start-up.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F


TRUFOR_ROOT = Path(os.environ.get(
    "TRUFOR_ROOT", "/root/forge-detector/vendor/TruFor/TruFor_train_test"
))
TRUFOR_WEIGHTS = Path(os.environ.get(
    "TRUFOR_WEIGHTS", str(TRUFOR_ROOT / "pretrained_models" / "trufor.pth.tar")
))
# The public upload remains capped by the web app.  Internally gpu_server
# serializes a decoded JPEG as lossless PNG before sending it over localhost,
# which may grow to several times the original file size for camera photos.
MAX_IMAGE_BYTES = 64 * 1024 * 1024
# TruFor's official test script feeds native pixels directly to the model.
# Its activation memory grows rapidly with image area.  1024 px preserves
# localized editing traces at normal viewing scale while keeping the worker
# compatible with the GPU service's other resident models.  Originals remain
# untouched; this limit applies only to the model input.
MAX_INFERENCE_EDGE = max(256, int(os.environ.get("TRUFOR_MAX_EDGE", "1024")))


def _map_png(values: np.ndarray) -> str:
    """Return a bounded PNG data URL, matching the former API response."""
    array = np.asarray(values, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError("TruFor map is not two-dimensional")
    image = Image.fromarray(np.uint8(np.clip(array, 0.0, 1.0) * 255), mode="L")
    if max(image.size) > 512:
        scale = 512.0 / max(image.size)
        image = image.resize(
            (max(1, int(image.width * scale)), max(1, int(image.height * scale))),
            Image.Resampling.BILINEAR,
        )
    encoded = io.BytesIO()
    image.save(encoded, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(encoded.getvalue()).decode("ascii")


class TruForRuntime:
    def __init__(self) -> None:
        if not TRUFOR_ROOT.is_dir() or not TRUFOR_WEIGHTS.is_file():
            raise FileNotFoundError("TruFor source tree or checkpoint is not installed")

        # The upstream test.py imports ``lib`` from this directory and creates
        # its configuration from the trufor_ph3 experiment.  Reuse exactly that
        # setup and the official /256 input normalization.
        os.chdir(TRUFOR_ROOT)
        for path in (str(TRUFOR_ROOT), str(TRUFOR_ROOT.parent)):
            if path not in sys.path:
                sys.path.insert(0, path)
        from lib.config import config, update_config
        from lib.utils import get_model

        opts = ["TEST.MODEL_FILE", str(TRUFOR_WEIGHTS)]
        update_config(config, SimpleNamespace(experiment="trufor_ph3", gpu=0, opts=opts))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = bool(config.CUDNN.BENCHMARK)
            torch.backends.cudnn.deterministic = bool(config.CUDNN.DETERMINISTIC)
            torch.backends.cudnn.enabled = bool(config.CUDNN.ENABLED)

        checkpoint = torch.load(str(TRUFOR_WEIGHTS), map_location=self.device)
        self.model = get_model(config)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        self.lock = threading.Lock()

    def infer(self, raw_image: bytes) -> dict:
        image = Image.open(io.BytesIO(raw_image)).convert("RGB")
        original_size = image.size
        if max(original_size) > MAX_INFERENCE_EDGE:
            scale = MAX_INFERENCE_EDGE / max(original_size)
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        # TestDataset in the official test.py divides by 256.0, not 255.0.
        rgb = np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 256.0
        tensor = torch.from_numpy(rgb).unsqueeze(0).to(self.device)
        with self.lock, torch.inference_mode():
            pred, conf, det, _ = self.model(tensor, save_np=False)

        anomaly = F.softmax(torch.squeeze(pred, 0), dim=0)[1].detach().cpu().numpy()
        confidence = (torch.sigmoid(torch.squeeze(conf, 0))[0].detach().cpu().numpy()
                      if conf is not None else np.ones_like(anomaly, dtype=np.float32))
        score = float(torch.sigmoid(det).item()) if det is not None else float(np.mean(anomaly))
        anomaly = np.asarray(anomaly, dtype=np.float32)
        confidence = np.asarray(confidence, dtype=np.float32)
        return {
            "available": True,
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "reliability": round(float(np.mean(np.clip(confidence, 0.0, 1.0))), 4),
            "map_coverage": round(float(np.mean(np.clip(anomaly, 0.0, 1.0) >= 0.50)), 4),
            "map_png": _map_png(anomaly),
            "confidence_png": _map_png(confidence),
            "input_size": list(original_size),
            "inference_size": list(image.size),
            "detail": (
                "TruFor 常驻模型已输出完整性分数、疑似篡改区域与定位可靠性。"
                + (f" 原图已从 {original_size[0]}×{original_size[1]} 规范化到 {image.width}×{image.height} 推理。"
                   if image.size != original_size else "")
            ),
        }


class Handler(BaseHTTPRequestHandler):
    runtime: TruForRuntime | None = None

    def log_message(self, fmt: str, *args) -> None:
        # The parent service owns operational logging; avoid an access log per
        # image in the worker log.
        return

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send_json(200, {"ready": self.runtime is not None, "service": "trufor-worker"})
        else:
            self._send_json(404, {"detail": "not found"})

    def do_POST(self) -> None:
        if self.path != "/infer":
            self._send_json(404, {"available": False, "detail": "not found"})
            return
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= MAX_IMAGE_BYTES:
                raise ValueError("invalid image size")
            if self.runtime is None:
                raise RuntimeError("TruFor worker is not ready")
            self._send_json(200, self.runtime.infer(self.rfile.read(size)))
        except Exception as exc:
            print(f"[TruFor Worker] inference failed: {type(exc).__name__}: {exc}", flush=True)
            self._send_json(500, {"available": False, "detail": f"TruFor worker inference failed: {type(exc).__name__}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=6012)
    args = parser.parse_args()

    print("[TruFor Worker] loading official checkpoint...", flush=True)
    Handler.runtime = TruForRuntime()
    print("[TruFor Worker] model ready", flush=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    server.serve_forever()


if __name__ == "__main__":
    main()
