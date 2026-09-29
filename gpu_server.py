# -*- coding: utf-8 -*-
"""
AutoDL GPU 推理服务
三路基础检测 + 专项模型：
  - UniversalFakeDetect / UnivFD (CLIP ViT-L/14) → 跨生成器 AI 全图生成检测
  - AIDE (ICLR 2025, GenImage checkpoint) → 伪影 + DCT 噪声特征检测
  - Ateeqq/ai-vs-human-image-detector    (SigLIP2, 保留未使用)
  - prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-B, 92.12% 准确率) → 深度伪造检测

启动方式：python gpu_server.py --port 6006
"""
import os
import sys
import io
import math
import time
import gc
import argparse
import base64
import json
import re
import urllib.request
import subprocess
import tempfile
import threading
from pathlib import Path
import numpy as np
import requests
from PIL import Image
import torch
import torch.nn.functional as F
from flask import Flask, request, jsonify

# ──────────────────────────────────────────────────────────
# 配置 — 国内网络使用 HuggingFace 镜像
# ──────────────────────────────────────────────────────────
_HF_MIRROR = "https://hf-mirror.com"
if os.environ.get("HF_ENDPOINT", "") == "":
    os.environ["HF_ENDPOINT"] = _HF_MIRROR
    print(f"[Config] 已设置 HF_ENDPOINT={_HF_MIRROR}")

MODEL_CACHE = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

AI_HUMAN_MODEL_ID   = "Ateeqq/ai-vs-human-image-detector"   # 保留，不再使用
DEEPFAKE_MODEL_ID   = "prithivMLmods/Deep-Fake-Detector-v2-Model"
UNIVFD_CLIP_ARCH    = "ViT-L/14"
UNIVFD_WEIGHTS_URL  = "https://raw.githubusercontent.com/WisconsinAIVision/UniversalFakeDetect/main/pretrained_weights/fc_weights.pth"
UNIVFD_WEIGHTS_PATH = os.path.join(MODEL_CACHE, "univfd", "fc_weights.pth")
AIDE_ROOT           = os.environ.get("AIDE_ROOT", "/root/forge-detector/vendor/AIDE")
AIDE_CHECKPOINT     = os.environ.get("AIDE_CHECKPOINT", "/root/forge-detector/models/aide/GenImage_train.pth")
DEAR_ROOT           = os.environ.get("DEAR_ROOT", "/root/forge-detector/vendor/dear")
DEAR_R_CHECKPOINT   = os.environ.get("DEAR_R_CHECKPOINT", "/root/forge-detector/models/dear/dear_r/model_best.pth")
PROBE_DINO_MODEL_ID = "facebook/dinov2-with-registers-large"
PROBE_CHECKPOINT    = os.environ.get("PROBE_CHECKPOINT", "/root/forge-detector/models/probe/DINOv2_best_model_step_34999.pth")
TRUFOR_ROOT         = os.environ.get("TRUFOR_ROOT", "/root/forge-detector/vendor/TruFor/TruFor_train_test")
TRUFOR_PYTHON       = os.environ.get("TRUFOR_PYTHON", "/root/miniconda3/envs/trufor/bin/python")
TRUFOR_WEIGHTS      = os.environ.get("TRUFOR_WEIGHTS", os.path.join(TRUFOR_ROOT, "pretrained_models", "trufor.pth.tar"))
TRUFOR_TIMEOUT      = int(os.environ.get("TRUFOR_TIMEOUT", "240"))
# Match the worker default.  Resize before serializing the internal request so
# a high-resolution original cannot turn into an oversized lossless PNG or
# waste tunnel bandwidth.  This never changes the file retained by the web
# application; it is only the TruFor model input.
TRUFOR_MAX_EDGE     = max(256, int(os.environ.get("TRUFOR_MAX_EDGE", "1024")))
# The AI-generation/deepfake chain completes before the web app calls
# /trufor_analyze. Releasing its resident models at that boundary gives the
# separate TruFor process enough VRAM without changing any result already
# returned to the web app. They are loaded lazily again for the next request.
TRUFOR_RELEASE_GPU_MODELS = os.environ.get("TRUFOR_RELEASE_GPU_MODELS", "1") == "1"
# TruFor has its own Conda environment.  Keep that environment in a small
# loopback worker so that its model is loaded once instead of spawning Python
# and reloading the checkpoint for every image.
TRUFOR_WORKER_HOST  = os.environ.get("TRUFOR_WORKER_HOST", "127.0.0.1")
TRUFOR_WORKER_PORT  = int(os.environ.get("TRUFOR_WORKER_PORT", "6012"))
TRUFOR_WORKER_URL   = os.environ.get("TRUFOR_WORKER_URL", f"http://{TRUFOR_WORKER_HOST}:{TRUFOR_WORKER_PORT}")
TRUFOR_WORKER_FILE  = os.environ.get("TRUFOR_WORKER_FILE", os.path.join(os.path.dirname(os.path.abspath(__file__)), "trufor_worker.py"))
TRUFOR_WORKER_LOG   = os.environ.get("TRUFOR_WORKER_LOG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "trufor_worker.log"))
TRUFOR_WORKER_START_TIMEOUT = int(os.environ.get("TRUFOR_WORKER_START_TIMEOUT", "90"))
ADAIFL_ROOT         = os.environ.get("ADAIFL_ROOT", "/root/forge-detector/vendor/AdaIFL")
ADAIFL_MODEL_PATH   = os.environ.get("ADAIFL_MODEL_PATH", "/root/autodl-tmp/forge-detector-data/models/AdaIFL/AdaIFL_v0.pth")
ADAIFL_PYTHON       = os.environ.get("ADAIFL_PYTHON", "/root/miniconda3/bin/python")
ADAIFL_CALIBRATED   = os.environ.get("ADAIFL_CALIBRATED", "0") == "1"
ADAIFL_TIMEOUT      = int(os.environ.get("ADAIFL_TIMEOUT", "300"))

# 基础三路检测模型 ID（按你的实际情况修改）
EXISTING_MODEL_ID = os.environ.get("EXISTING_MODEL_ID", None)

app = Flask(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[GPU Server] device = {device}")

_trufor_worker_lock = threading.Lock()
_trufor_worker_last_launch = 0.0

DETECTION_MODELS = ("ai_generated", "deepfake", "tamper")

# GLM 请求只在 AutoDL 上发起：本机仅经已有的 6006 隧道取得结果。
# 密钥优先从环境变量读取；部署脚本会将其保存在同目录、仅服务账号可读的文件中，
# 避免把密钥写进 GPU 服务源码或日志。
GLM_API_URL = "https://api.siliconflow.cn/v1/chat/completions"
GLM_MODEL = "zai-org/GLM-4.5V"


def _load_glm_api_key():
    key = os.environ.get("SILICONFLOW_API_KEY", "").strip()
    if key:
        return key
    try:
        key_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".siliconflow_api_key")
        with open(key_path, "r", encoding="utf-8") as key_file:
            return key_file.read().strip()
    except OSError:
        return ""


GLM_API_KEY = _load_glm_api_key()


def parse_detection_models():
    """Read the selected special detectors; no field keeps old all-model behavior."""
    requested = set()
    for value in request.form.getlist("models"):
        requested.update(item.strip() for item in str(value).split(",") if item.strip())
    if not requested:
        return DETECTION_MODELS
    return tuple(model for model in DETECTION_MODELS if model in requested)


def _extract_glm_json(text):
    """Extract the JSON object from an occasionally fenced GLM response."""
    if not isinstance(text, str):
        return None
    fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    candidates = [fenced.group(1)] if fenced else []
    plain = re.search(r"\{[\s\S]*\}", text)
    if plain:
        candidates.append(plain.group())
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue
    return None


def detect_face_glm_remote(image):
    """Call GLM from the GPU host and return a stable face-detection payload."""
    if not GLM_API_KEY:
        return {
            "face_detected": False, "face_count": 0, "faces": [], "confidence": 0.0,
            "detail": "远端未配置 GLM API 密钥", "method": "gpu_glm_vision",
            "error": "SILICONFLOW_API_KEY is not configured on GPU server",
        }

    # Keep the payload bounded while retaining enough facial detail for recognition.
    image_copy = image.copy().convert("RGB")
    if max(image_copy.size) > 2048:
        scale = 2048 / max(image_copy.size)
        image_copy = image_copy.resize(
            (max(1, int(image_copy.width * scale)), max(1, int(image_copy.height * scale))),
            Image.Resampling.LANCZOS,
        )
    buffer = io.BytesIO()
    image_copy.save(buffer, format="JPEG", quality=85)
    image_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    prompt = (
        "请只返回 JSON，不要 Markdown。判断图中是否有真实或逼真的人脸（正脸、侧脸均算），"
        "格式：{\"has_face\":true/false,\"face_count\":整数,\"faces\":[],"
        "\"confidence\":0到1,\"reasoning\":\"简短依据\"}。"
    )
    payload = {
        "model": GLM_MODEL,
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "text", "text": prompt},
        ]}],
        "max_tokens": 400,
        "temperature": 0.1,
    }
    try:
        response = requests.post(
            GLM_API_URL,
            headers={"Authorization": f"Bearer {GLM_API_KEY}", "Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = _extract_glm_json(content)
        if not parsed:
            raise ValueError("GLM response did not contain valid JSON")
        return {
            "face_detected": bool(parsed.get("has_face", False)),
            "face_count": int(parsed.get("face_count", 0) or 0),
            "faces": parsed.get("faces", []) if isinstance(parsed.get("faces", []), list) else [],
            "confidence": float(parsed.get("confidence", 0.5) or 0.5),
            "detail": str(parsed.get("reasoning", "")),
            "method": "gpu_glm_vision",
        }
    except Exception as exc:
        print(f"[ERROR] remote GLM face detection failed: {type(exc).__name__}: {exc}")
        return {
            "face_detected": False, "face_count": 0, "faces": [], "confidence": 0.0,
            "detail": f"远端 GLM 人脸检测请求失败: {str(exc)[:160]}",
            "method": "gpu_glm_vision", "error": str(exc),
        }


# ──────────────────────────────────────────────────────────
# 模型加载器
# ──────────────────────────────────────────────────────────

class AIHumanDetector:
    """Ateeqq/ai-vs-human-image-detector — 基于 SigLIP2 的 AI/人类 二分类"""
    def __init__(self):
        print(f"[AIHumanDetector] loading {AI_HUMAN_MODEL_ID} ...")
        from transformers import AutoImageProcessor, SiglipForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(AI_HUMAN_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model = SiglipForImageClassification.from_pretrained(AI_HUMAN_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model.to(device).eval()
        self.labels = self.model.config.id2label  # {0: 'ai', 1: 'hum'}  注意: 0=ai, 1=hum
        print(f"[AIHumanDetector] loaded. labels={self.labels}")

    def predict(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        ai_score   = float(probs[0])   # label 0 = 'ai'
        human_score = float(probs[1])  # label 1 = 'hum'
        return {
            "label": "AI vs Human Detector",
            "ai_score": round(ai_score, 4),
            "human_score": round(human_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5
        }


class DeepFakeV2Detector:
    """prithivMLmods/Deep-Fake-Detector-v2-Model — 基于 ViT-B-patch16 的深度伪造检测"""
    def __init__(self):
        print(f"[DeepFakeV2Detector] loading {DEEPFAKE_MODEL_ID} ...")
        from transformers import ViTImageProcessor, ViTForImageClassification
        self.processor = ViTImageProcessor.from_pretrained(DEEPFAKE_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model = ViTForImageClassification.from_pretrained(DEEPFAKE_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model.to(device).eval()
        self.labels = self.model.config.id2label  # 预期: {0: 'Realism', 1: 'Deepfake'} 或类似
        print(f"[DeepFakeV2Detector] loaded. labels={self.labels}")

    def predict(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # 根据 id2label 确定 deepfake/real 概率
        id2label = self.labels
        real_idx, fake_idx = None, None
        for idx, lbl in id2label.items():
            low = lbl.lower()
            if 'real' in low or 'human' in low:
                real_idx = idx
            elif 'fake' in low or 'deepfake' in low or 'ai' in low:
                fake_idx = idx

        # 如果没找到语义标签，默认 0=real, 1=fake
        if real_idx is None and fake_idx is None:
            real_idx, fake_idx = 0, 1

        real_score   = float(probs[real_idx])   if real_idx is not None else 1.0 - float(probs[fake_idx])
        fake_score   = float(probs[fake_idx])   if fake_idx is not None else 1.0 - float(probs[real_idx])

        return {
            "label": "DeepFake Detector v2",
            "deepfake_score": round(fake_score, 4),
            "real_score": round(real_score, 4),
            "verdict": "疑似深度伪造" if fake_score >= 0.5 else "疑似真实人脸/图片",
            "is_deepfake": fake_score >= 0.5
        }


class AIRealNetDetector:
    """Modotte/AIRealNet — 基于 SwinV2 的 AI生成/真实 二分类，softmax 分数分布更平滑"""
    def __init__(self):
        print(f"[AIRealNetDetector] loading {AIREALNET_MODEL_ID} ...")
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        self.processor = AutoImageProcessor.from_pretrained(AIREALNET_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model = AutoModelForImageClassification.from_pretrained(AIREALNET_MODEL_ID, cache_dir=MODEL_CACHE)
        self.model.to(device).eval()
        self.labels = self.model.config.id2label  # 预期: {0: 'Artificial', 1: 'Human'}
        print(f"[AIRealNetDetector] loaded. labels={self.labels}")

    def predict(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].cpu().numpy()

        # 根据 id2label 确定 AI / Human 索引
        id2label = self.labels
        ai_idx, human_idx = None, None
        for idx, lbl in id2label.items():
            low = lbl.lower()
            if 'ai' in low or 'artificial' in low or 'fake' in low or 'generated' in low:
                ai_idx = idx
            elif 'human' in low or 'real' in low:
                human_idx = idx

        if ai_idx is None and human_idx is None:
            ai_idx, human_idx = 0, 1  # 兜底

        ai_score    = float(probs[ai_idx])    if ai_idx    is not None else 1.0 - float(probs[human_idx])
        human_score = float(probs[human_idx]) if human_idx is not None else 1.0 - float(probs[ai_idx])

        return {
            "label": "AIRealNet",
            "ai_score": round(ai_score, 4),
            "human_score": round(human_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5
        }


class UnivFDDetector:
    """UniversalFakeDetect 官方预训练线性头，使用 CLIP ViT-L/14 特征进行零训练推理。"""
    def __init__(self):
        print(f"[UnivFDDetector] loading UniversalFakeDetect ({UNIVFD_CLIP_ARCH}) ...")
        try:
            import clip
        except ImportError as exc:
            raise RuntimeError(
                "缺少 openai-clip 依赖，请执行: python -m pip install openai-clip"
            ) from exc

        os.makedirs(os.path.dirname(UNIVFD_WEIGHTS_PATH), exist_ok=True)
        if not os.path.exists(UNIVFD_WEIGHTS_PATH):
            print("[UnivFDDetector] downloading official fc_weights.pth ...")
            urllib.request.urlretrieve(UNIVFD_WEIGHTS_URL, UNIVFD_WEIGHTS_PATH)

        # clip.load 内置的预处理与 UnivFD 官方验证脚本一致：Resize/CenterCrop 224 + CLIP 归一化。
        self.model, self.preprocess = clip.load(
            UNIVFD_CLIP_ARCH, device=device, download_root=MODEL_CACHE
        )
        self.model.eval()
        self.fc = torch.nn.Linear(768, 1)
        try:
            state_dict = torch.load(UNIVFD_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容较早版本的 PyTorch
            state_dict = torch.load(UNIVFD_WEIGHTS_PATH, map_location="cpu")
        self.fc.load_state_dict(state_dict)
        self.fc.to(device).eval()
        print("[UnivFDDetector] loaded.")

    def predict(self, image: Image.Image):
        image_tensor = self.preprocess(image.convert("RGB")).unsqueeze(0).to(device)
        with torch.no_grad():
            # CLIP 在 CUDA 上通常以 fp16 运行；线性头统一转 fp32 计算以避免 dtype 不匹配。
            features = self.model.encode_image(image_tensor).float()
            logit = F.linear(features, self.fc.weight.float(), self.fc.bias.float())
            ai_score = float(torch.sigmoid(logit)[0, 0].item())

        return {
            "label": "UniversalFakeDetect (UnivFD)",
            "ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5,
            "available": True,
        }


class AIDEDetector:
    """AIDE（ICLR 2025）官方实现，使用 GenImage 训练的完整 checkpoint 做单图推理。"""
    def __init__(self):
        if not os.path.isfile(AIDE_CHECKPOINT):
            raise FileNotFoundError(f"未找到 AIDE checkpoint: {AIDE_CHECKPOINT}")
        if not os.path.isdir(AIDE_ROOT):
            raise FileNotFoundError(f"未找到 AIDE 官方代码目录: {AIDE_ROOT}")

        print(f"[AIDEDetector] loading checkpoint: {AIDE_CHECKPOINT} ...")
        if AIDE_ROOT not in sys.path:
            sys.path.insert(0, AIDE_ROOT)

        from torchvision import transforms
        import models.AIDE as aide_model
        from data.dct import DCT_base_Rec_Module

        # checkpoint 已包含 AIDE 的 ResNet 与 OpenCLIP ConvNeXt-XXLarge 全部参数；
        # 先以空预训练骨干构建，再使用安全的 weights_only 模式恢复官方权重。
        self.model = aide_model.AIDE(resnet_path=None, convnext_path=None)
        try:
            checkpoint = torch.load(AIDE_CHECKPOINT, map_location="cpu", weights_only=True)
        except TypeError:  # 兼容较早版本 PyTorch
            checkpoint = torch.load(AIDE_CHECKPOINT, map_location="cpu")
        state_dict = checkpoint.get("model", checkpoint)
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"AIDE checkpoint 与官方模型不匹配: missing={missing[:5]}, unexpected={unexpected[:5]}"
            )

        self.model.to(device).eval()
        self.dct = DCT_base_Rec_Module()
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Compose([
            transforms.Resize([256, 256]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[AIDEDetector] loaded.")

    def predict(self, image: Image.Image):
        # 严格复用官方 TestDataset：ToTensor → DCT 四路重建 → Resize/Normalize → 五路堆叠。
        source = self.to_tensor(image.convert("RGB"))
        x_minmin, x_maxmax, x_minmin1, x_maxmax1 = self.dct(source)
        sample = torch.stack([
            self.normalize(x_minmin), self.normalize(x_maxmax),
            self.normalize(x_minmin1), self.normalize(x_maxmax1), self.normalize(source),
        ], dim=0).unsqueeze(0).to(device)
        with torch.inference_mode():
            probabilities = torch.softmax(self.model(sample), dim=1)[0].float().cpu().numpy()

        # 官方数据集标签：0 = 真实图片，1 = AI 生成图片。
        human_score = float(probabilities[0])
        ai_score = float(probabilities[1])
        return {
            "label": "AIDE (GenImage)",
            "ai_score": round(ai_score, 4),
            "human_score": round(human_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5,
            "available": True,
        }


class DEARRDetector:
    """DEAR-r（ICML 2026）官方权重：针对压缩、缩放等后处理增强鲁棒性。"""
    def __init__(self):
        if not os.path.isfile(DEAR_R_CHECKPOINT):
            raise FileNotFoundError(f"未找到 DEAR-r checkpoint: {DEAR_R_CHECKPOINT}")
        if not os.path.isdir(DEAR_ROOT):
            raise FileNotFoundError(f"未找到 DEAR 官方代码目录: {DEAR_ROOT}")

        print(f"[DEARRDetector] loading checkpoint: {DEAR_R_CHECKPOINT} ...")
        if DEAR_ROOT not in sys.path:
            sys.path.insert(0, DEAR_ROOT)
        from torchvision import transforms
        from dear.detector.rajan_mask_gated_detector import RajanMaskGatedDetector

        self.model = RajanMaskGatedDetector(device=str(device))
        self.model.load(DEAR_R_CHECKPOINT)
        self.model.eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[DEARRDetector] loaded.")

    def predict(self, image: Image.Image):
        # 官方单图流程保持原始分辨率，不做固定裁剪；过大图像仅作安全缩放以避免占满显存。
        source = image.convert("RGB")
        max_side = max(source.size)
        if max_side > 1536:
            scale = 1536.0 / max_side
            source = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.LANCZOS)
        x = self.transform(source).unsqueeze(0).to(device)
        with torch.inference_mode():
            logit = self.model.predict(x).squeeze()
            ai_score = float(torch.sigmoid(logit).item())
        return {
            "label": "DEAR-r",
            "ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5,
            "available": True,
        }


class PROBEDinoV2Detector:
    """PROBE-DINOv2（ICML 2026）官方微调权重，按官方 336px 分块后汇总 logit。"""
    PATCH_SIZE = 336
    MAX_PATCHES = 9

    def __init__(self):
        if not os.path.isfile(PROBE_CHECKPOINT):
            raise FileNotFoundError(f"未找到 PROBE-DINOv2 checkpoint: {PROBE_CHECKPOINT}")
        print(f"[PROBEDinoV2Detector] loading checkpoint: {PROBE_CHECKPOINT} ...")
        from transformers import Dinov2WithRegistersConfig, Dinov2WithRegistersModel
        from torchvision import transforms

        config = Dinov2WithRegistersConfig.from_pretrained(PROBE_DINO_MODEL_ID, cache_dir=MODEL_CACHE)
        self.backbone = Dinov2WithRegistersModel(config)
        try:
            checkpoint = torch.load(PROBE_CHECKPOINT, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(PROBE_CHECKPOINT, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict", checkpoint.get("model", checkpoint))
        backbone_state = {key[len("backbone."):]: value for key, value in state_dict.items() if key.startswith("backbone.")}
        self.backbone.load_state_dict(backbone_state, strict=True)
        self.fc = torch.nn.Linear(config.hidden_size, 1)
        self.fc.load_state_dict({key[len("fc."):]: value for key, value in state_dict.items() if key.startswith("fc.")})
        self.backbone.to(device).eval()
        self.fc.to(device).eval()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        print("[PROBEDinoV2Detector] loaded.")

    def _patches(self, image: Image.Image):
        source = image.convert("RGB")
        if min(source.size) < self.PATCH_SIZE:
            scale = self.PATCH_SIZE / min(source.size)
            source = source.resize((round(source.width * scale), round(source.height * scale)), Image.Resampling.BICUBIC)
        tensor = self.transform(source)
        _, height, width = tensor.shape
        ys = list(range(0, height - self.PATCH_SIZE + 1, self.PATCH_SIZE)) or [0]
        xs = list(range(0, width - self.PATCH_SIZE + 1, self.PATCH_SIZE)) or [0]
        if ys[-1] != height - self.PATCH_SIZE:
            ys.append(height - self.PATCH_SIZE)
        if xs[-1] != width - self.PATCH_SIZE:
            xs.append(width - self.PATCH_SIZE)
        coords = [(y, x) for y in ys for x in xs]
        if len(coords) > self.MAX_PATCHES:
            indices = np.linspace(0, len(coords) - 1, self.MAX_PATCHES, dtype=int)
            coords = [coords[index] for index in indices]
        return torch.stack([tensor[:, y:y + self.PATCH_SIZE, x:x + self.PATCH_SIZE] for y, x in coords])

    def predict(self, image: Image.Image):
        patches = self._patches(image).to(device)
        with torch.inference_mode():
            features = self.backbone(patches).last_hidden_state[:, 0]
            logit = self.fc(features).mean()
            ai_score = float(torch.sigmoid(logit).item())
        return {
            "label": "PROBE-DINOv2",
            "ai_score": round(ai_score, 4),
            "human_score": round(1.0 - ai_score, 4),
            "verdict": "疑似AI生成" if ai_score >= 0.5 else "疑似真实图片",
            "is_ai": ai_score >= 0.5,
            "available": True,
        }


class ThreeWayDetector:
    """
    基础三路检测器 (AI生成 / 深度换脸 / 图像篡改)
    
    如果你已有自定义模型，修改这里的 _predict 方法即可。
    目前提供两种模式：
      MODE = "huggingface"  → 从 HuggingFace 加载一个 4 分类模型
      MODE = "custom"       → 替换 _predict_custom 为你自己的推理逻辑
    """
    MODE = "huggingface"

    def __init__(self):
        self._loaded = False
        self.model = None
        self.processor = None
        # 类标签与索引
        self.idx_map = {"ai_generated": 0, "deepfake": 1, "splicing": 2, "real": 3}
        self.idx_rev = {0: "ai_generated", 1: "deepfake", 2: "splicing", 3: "real"}

    def _load_huggingface(self):
        """从 HuggingFace 加载 4 分类模型（如果你有的话）"""
        from transformers import AutoImageProcessor, AutoModelForImageClassification
        model_id = EXISTING_MODEL_ID
        if not model_id:
            print("[ThreeWayDetector] 未设置 EXISTING_MODEL_ID，使用随机初始化（演示）")
            self._loaded = True
            return False
        print(f"[ThreeWayDetector] loading {model_id} ...")
        self.processor = AutoImageProcessor.from_pretrained(model_id, cache_dir=MODEL_CACHE)
        self.model = AutoModelForImageClassification.from_pretrained(model_id, cache_dir=MODEL_CACHE)
        self.model.to(device).eval()
        self._loaded = True
        return True

    def _predict_huggingface(self, image):
        if not self._loaded:
            self._load_huggingface()
        if self.model is None or self.processor is None:
            return self._fallback()
        inputs = self.processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = self.model(**inputs).logits[0].cpu().numpy()
        probs = self._softmax(logits)
        return self._build(probs)

    def _predict_custom(self, image):
        """在这里填写你现有的三路检测推理代码"""
        # 示例：直接返回你的已有结果
        # return {...}
        return self._fallback()

    def _fallback(self):
        """占位返回：未加载模型时的默认值"""
        print("[ThreeWayDetector] 未找到基础模型，返回演示数据")
        import random
        r = [random.random() for _ in range(4)]
        total = sum(r)
        probs = [v / total for v in r]
        return self._build(probs)

    def _softmax(self, x):
        e = np.exp(x - np.max(x))
        return e / e.sum()

    def _build(self, probs):
        ai, df, sp, real = float(probs[0]), float(probs[1]), float(probs[2]), float(probs[3])
        best_idx = np.argmax(probs)
        return {
            "result": self.idx_rev.get(int(best_idx), "real"),
            "confidence": round(float(probs[int(best_idx)]), 4),
            "details": {
                "ai_generated": round(ai, 4),
                "deepfake":     round(df, 4),
                "splicing":     round(sp, 4),
                "real":         round(real, 4),
            }
        }

    def predict(self, image: Image.Image):
        if self.MODE == "huggingface":
            return self._predict_huggingface(image)
        else:
            return self._predict_custom(image)


# ──────────────────────────────────────────────────────────
# 全局模型实例（懒加载）
# ──────────────────────────────────────────────────────────
_threeway   = None
_aihuman    = None
_deepfakev2 = None
_univfd     = None
_aide       = None
_dear_r     = None
_probe_dinov2 = None


def get_threeway():
    global _threeway
    if _threeway is None:
        _threeway = ThreeWayDetector()
    return _threeway


def get_aihuman():
    global _aihuman
    if _aihuman is None:
        _aihuman = AIHumanDetector()
    return _aihuman


def get_deepfakev2():
    global _deepfakev2
    if _deepfakev2 is None:
        _deepfakev2 = DeepFakeV2Detector()
    return _deepfakev2


def get_univfd():
    global _univfd
    if _univfd is None:
        _univfd = UnivFDDetector()
    return _univfd


def get_aide():
    global _aide
    if _aide is None:
        _aide = AIDEDetector()
    return _aide


def get_dear_r():
    global _dear_r
    if _dear_r is None:
        _dear_r = DEARRDetector()
    return _dear_r


def get_probe_dinov2():
    global _probe_dinov2
    if _probe_dinov2 is None:
        _probe_dinov2 = PROBEDinoV2Detector()
    return _probe_dinov2


def release_models_for_trufor():
    """Free completed GPU-specialty models before the separate TruFor worker.

    ``/detect`` has already serialized its scores to the caller when this is
    invoked, so releasing the process-local caches cannot affect the current
    result. It avoids two large model sets competing for the same GPU. The
    normal lazy getters reload a model on a later detection request.
    """
    if not TRUFOR_RELEASE_GPU_MODELS:
        return []

    global _threeway, _aihuman, _deepfakev2, _univfd, _aide, _dear_r, _probe_dinov2
    cache_names = (
        "_threeway", "_aihuman", "_deepfakev2", "_univfd",
        "_aide", "_dear_r", "_probe_dinov2",
    )
    released = []
    for cache_name in cache_names:
        if globals()[cache_name] is not None:
            globals()[cache_name] = None
            released.append(cache_name[1:])

    if released:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print(f"[TruFor] released completed GPU models: {', '.join(released)}")
    return released


# ──────────────────────────────────────────────────────────
# SRM 滤波器组（与 tamper_detector._SRM_FILTERS 完全一致）
# ──────────────────────────────────────────────────────────

_SRM_FILTERS_GPU = [
    (3, 3, np.array([[0,0,0],[-1,1,0],[0,0,0]], dtype=np.float32)),        # 1_h
    (3, 3, np.array([[0,-1,0],[0,1,0],[0,0,0]], dtype=np.float32)),        # 1_v
    (3, 3, np.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=np.float32)),        # 1_d1
    (3, 3, np.array([[0,0,-1],[0,1,0],[0,0,0]], dtype=np.float32)),        # 1_d2
    (1, 3, np.array([[1,-2,1]], dtype=np.float32)),                         # 2_h
    (3, 1, np.array([[1],[-2],[1]], dtype=np.float32)),                     # 2_v
    (3, 3, np.array([[1,0,0],[0,-2,0],[0,0,1]], dtype=np.float32)),        # 2_d
    (1, 4, np.array([[1,-3,3,-1]], dtype=np.float32)),                      # 3_h
    (4, 1, np.array([[1],[-3],[3],[-1]], dtype=np.float32)),                # 3_v
    (3, 3, np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)),    # e3_lap
    (3, 3, np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)),     # e3_sh
    (3, 3, np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)),     # e3_sv
    (3, 3, np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)),    # cross
    (5, 5, np.array([[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],                   # 5_lap
                     [-1,-1,24,-1,-1],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]], dtype=np.float32)),
    (5, 5, np.array([[0,0,-1,0,0],[0,-1,2,-1,0],                          # 5_gab
                     [-1,2,-4,2,-1],[0,-1,2,-1,0],[0,0,-1,0,0]], dtype=np.float32)),
]


def _srm_analyze_gpu(gray_tensor: torch.Tensor, windows) -> list:
    """
    GPU 加速完整 SRM 分析：卷积 → 窗口矩特征 → 马氏距离
    返回与 CPU 代码一致的 per-window 异常得分列表。
    """
    nf = 16
    nw = len(windows)
    if nw < 4:
        return [0.0] * nw

    # ── 1. 批量卷积产生残差图 ──
    responses = []
    inp = gray_tensor[None, None, :, :]  # (1, 1, H, W)
    for kh, kw, kern in _SRM_FILTERS_GPU:
        k = torch.from_numpy(kern).to(device=gray_tensor.device, dtype=torch.float32)
        k = k[None, None, :, :]  # (1, 1, kh, kw)
        r = F.conv2d(inp, k, padding='same').squeeze()  # (H, W), zero-pad; edge ~2px 不影响内部窗口
        responses.append(r)
    res_stack = torch.stack(responses, dim=0)  # (16, H, W)

    # ── 2. 逐窗口计算 4 矩特征 → (nw, 64) ──
    feats = torch.zeros(nw, nf * 4, device=gray_tensor.device)
    for wi, (y0, y1, x0, x1) in enumerate(windows):
        patch = res_stack[:, y0:y1, x0:x1]  # (16, hw, ww)
        npx = (y1 - y0) * (x1 - x0)
        if npx == 0:
            continue
        flat = patch.reshape(nf, -1)  # (16, npx)
        mu = flat.mean(dim=1)         # (16,)
        sd = flat.std(dim=1, unbiased=False) + 1e-10  # (16,)
        c = (flat - mu[:, None]) / sd[:, None]        # (16, npx)
        sk = (c ** 3).mean(dim=1)    # (16,)
        ku = (c ** 4).mean(dim=1)    # (16,) excess kurtosis
        feats[wi] = torch.stack([mu, sd, sk, ku], dim=1).flatten()  # (64,)

    # ── 3. 马氏距离（float64 精度以避免 rank-deficient 矩阵的 NaN）──
    feats_d = feats.double()           # (nw, 64) → float64
    mu_f = feats_d.mean(dim=0)         # (64,)
    c_f = feats_d - mu_f               # (nw, 64)
    # pinv 自然处理秩亏（nw < nf 时），无需 +eye 正则化
    cov = c_f.T @ c_f / (nw - 1)  # (64, 64) float64
    inv_cov = torch.linalg.pinv(cov)

    # 批量马氏距离: sqrt(clamp(diag(X @ inv_cov @ X.T), 0))
    maha_sq = (c_f @ inv_cov * c_f).sum(dim=1).clamp(min=0)
    maha = maha_sq.sqrt()              # (nw,)

    # 与 CPU 代码完全相同的归一化
    mx = maha.max()
    scores = maha / (mx * 1.5) if mx > 1e-8 else maha
    return scores.float().clamp(0, 1).cpu().tolist()


# ──────────────────────────────────────────────────────────
# GPU 加速 DCT 分析（图像篡改检测的 DCT 维度）
# ──────────────────────────────────────────────────────────

# 预计算 8×8 正交 DCT-II 变换矩阵（PyTorch 张量，懒加载到 GPU）
_dct_T_cache = {}

def _get_dct_T(device):
    """获取 8×8 DCT-II 正交矩阵 T 的缓存（按设备缓存）"""
    dev_str = str(device)
    if dev_str not in _dct_T_cache:
        N = 8
        k = torch.arange(N, dtype=torch.float32, device=device)[:, None]  # (8,1)
        v = torch.arange(N, dtype=torch.float32, device=device)[None, :]  # (1,8)
        T = torch.cos(math.pi / N * (v + 0.5) * k)  # (8,8)
        T[0] = T[0] / math.sqrt(2)
        T = T * math.sqrt(2.0 / N)
        _dct_T_cache[dev_str] = T
    return _dct_T_cache[dev_str]


def _build_windows(h, w, win_size=128, stride=64):
    """生成滑窗坐标（与 tamper_detector._build_windows 相同逻辑）"""
    win = min(win_size, h, w)
    st = max(stride, win // 4)
    ws = []
    for y0 in range(0, h - win + 1, st):
        for x0 in range(0, w - win + 1, st):
            ws.append((y0, y0 + win, x0, x0 + win))
    if h > win:
        for x0 in range(0, w - win + 1, st):
            ws.append((h - win, h, x0, x0 + win))
    if w > win:
        for y0 in range(0, h - win + 1, st):
            ws.append((y0, y0 + win, w - win, w))
    if h > win and w > win:
        ws.append((h - win, h, w - win, w))
    return list(set(ws))


def _dct_analyze_gpu(gray_tensor: torch.Tensor, windows) -> list:
    """
    在 GPU 上批量计算 DCT 滑窗分析。

    参数:
        gray_tensor: (H, W) float32 PyTorch 张量，已在 GPU 上
        windows:     [(y0, y1, x0, x1), ...]

    返回:
        [float, ...] 每个窗口的异常得分（顺序对应 windows）
    """
    if not windows:
        return []

    T = _get_dct_T(gray_tensor.device)  # (8,8)
    scores = torch.zeros(len(windows), device=gray_tensor.device)

    # 收集所有窗口的 8×8 块到统一批次
    all_blocks = []
    window_offsets = []  # (win_idx, start_idx, num_blocks)
    total = 0

    for wi, (y0, y1, x0, x1) in enumerate(windows):
        win = gray_tensor[y0:y1, x0:x1]
        hw, ww = win.shape
        bh, bw = hw // 8, ww // 8
        if bh < 2 or bw < 2:
            continue  # 窗口太小，score 留 0

        # 用 unfold 一次性提取窗口内所有 8×8 非重叠块
        # win shape (hw, ww) → (1, 1, hw, ww) → unfold(k=8, s=8) → (1, 64, nb)
        blocks = F.unfold(win[None, None, :, :], kernel_size=(8, 8), stride=(8, 8))
        nb = blocks.shape[2]
        blocks = blocks.squeeze(0).T.reshape(-1, 8, 8)  # (nb, 8, 8)
        blocks = blocks - 128.0  # DC 电平移位（与 CPU 代码一致）

        window_offsets.append((wi, total, nb))
        all_blocks.append(blocks)
        total += nb

    if total == 0:
        return [0.0] * len(windows)

    all_blocks_tensor = torch.cat(all_blocks, dim=0)  # (total_blocks, 8, 8)

    # ── 批量 DCT：T @ blocks @ T.T ──
    # (total_blocks, 8, 8) → bmm
    dct_temp = torch.bmm(all_blocks_tensor, T.T.expand(total, -1, -1))  # (N,8,8) @ (8,8)ᵗ
    dct_all = torch.bmm(T.expand(total, -1, -1), dct_temp)               # T @ result
    # dct_all: (total_blocks, 8, 8) — DCT-II 系数

    # ── 分块计算 DCT 特征 ──
    # ac = dct, 但 DC 系数置零
    ac_all = dct_all.clone()
    ac_all[:, 0, 0] = 0.0
    ac_flat = ac_all.reshape(total, 64)  # (total_blocks, 64)

    dct_abs = dct_all.abs().reshape(total, 64)
    ac_abs = ac_flat.abs()

    # --- 特征 1: AC 标准差 ---
    ac_std = ac_flat.std(dim=1, unbiased=False)  # (total_blocks,)

    # --- 特征 2: AC 能量占比 ---
    dct_sum = dct_abs.sum(dim=1)       # (total_blocks,)
    ac_sum = ac_abs.sum(dim=1)         # (total_blocks,)
    ac_pct = ac_sum / (dct_sum + 1e-10)

    # --- 特征 3: 高/中/低频能量比（按光栅扫描扁平索引）---
    # 低频: idx 1-10, 中频: idx 11-35, 高频: idx 36-63
    lo_mask = torch.zeros(64, dtype=torch.bool, device=ac_all.device)
    mi_mask = torch.zeros(64, dtype=torch.bool, device=ac_all.device)
    hi_mask = torch.zeros(64, dtype=torch.bool, device=ac_all.device)
    lo_mask[1:11] = True
    mi_mask[11:36] = True
    hi_mask[36:64] = True

    lo_energy = ac_abs[:, lo_mask].sum(dim=1)  # (total_blocks,)
    mi_energy = ac_abs[:, mi_mask].sum(dim=1)
    hi_energy = ac_abs[:, hi_mask].sum(dim=1)
    total_ac_energy = lo_energy + mi_energy + hi_energy + 1e-10
    low_r = lo_energy / total_ac_energy
    mid_r = mi_energy / total_ac_energy
    high_r = hi_energy / total_ac_energy

    # --- 特征 4: Benford 首位数字分布（完全向量化）---
    # 与 CPU 代码完全一致：先要求非零系数 >= 6，再要求首位有效位数 >= 4
    nonzero_mask = (ac_abs > 1e-10)          # (total_blocks, 64)，非零 AC 系数
    nonzero_counts = nonzero_mask.sum(dim=1)  # 每块非零系数个数
    valid_mask = (ac_abs >= 1.0)              # (total_blocks, 64)
    valid_counts = valid_mask.sum(dim=1)      # 每块 >=1.0 的系数个数

    benford_scores = torch.full((total,), 0.5, device=ac_all.device, dtype=torch.float32)

    # 外层门控 len(nz) >= 6 且内层门控 len(lds) >= 4
    outer_ok = nonzero_counts >= 6
    inner_ok = valid_counts >= 4
    have_enough = outer_ok & inner_ok
    if have_enough.any():
        # 收集所有有效块的有效系数值 + 对应的块索引
        batch_indices = torch.arange(total, device=ac_all.device)[:, None].expand_as(ac_abs)
        batch_indices = batch_indices[valid_mask]  # 每个有效系数所属的块索引
        valid_vals = ac_abs[valid_mask]  # 有效系数值

        # 计算首位数：floor(v / 10^floor(log10(v)))
        log10_vals = torch.log10(valid_vals)
        pow10 = torch.pow(10.0, torch.floor(log10_vals).float())
        first_digits = (valid_vals / pow10).long()  # 1-9

        # 逐块统计首位数频次
        # combined_idx = batch_idx * 10 + first_digit
        combined_idx = batch_indices * 10 + first_digits
        hist_flat = torch.bincount(combined_idx, minlength=total * 10).float()
        hist = hist_flat.reshape(total, 10)  # (total_blocks, 10), 第 0 列永远是 0

        # Benford 理论分布 (1-9)
        benford_probs = torch.tensor(
            [math.log10(1 + 1 / d) for d in range(1, 10)],
            device=ac_all.device, dtype=torch.float32
        )  # (9,)

        # 计算有效块的 JS 散度
        cnt_sum = hist[:, 1:].sum(dim=1) + 1e-10  # (total_blocks,) 第 0 位不计
        hist_norm = hist[:, 1:] / cnt_sum[:, None]  # (total_blocks, 9)

        m = (hist_norm + benford_probs[None, :]) / 2  # 平均分布
        kl_data = (hist_norm * torch.log(hist_norm / (m + 1e-10) + 1e-10)).sum(dim=1)
        kl_ben = (benford_probs[None, :] * torch.log(benford_probs[None, :] / (m + 1e-10) + 1e-10)).sum(dim=1)
        js = (kl_data + kl_ben) / 2

        # JS 越小越接近 Benford → 真实图片 → score 越高
        benford_block = torch.clamp(1 - js * 5, 0, 1)
        benford_scores[have_enough] = benford_block[have_enough]

    # ── 按窗口聚合 ──
    for wi, start, nb in window_offsets:
        if nb == 0:
            continue
        end = start + nb
        # 取每窗口内各块特征的平均值
        mac = ac_std[start:end].mean()
        mb = benford_scores[start:end].mean()
        mhr = high_r[start:end].mean()
        mapc = ac_pct[start:end].mean()

        # 与 CPU 代码完全相同的得分公式（torch.where 替代 Python if）
        acs = torch.where(mac < 15, torch.clamp(1 - mac / 15, 0, 1), torch.tensor(0.0, device=ac_all.device))
        ba = 1 - mb
        hfa = (mhr - 0.3).abs() * 2
        apa = torch.where(mapc < 0.3, torch.clamp(1 - mapc / 0.3, 0, 1), torch.tensor(0.0, device=ac_all.device))

        s = torch.clamp(acs * 0.3 + ba * 0.3 + hfa * 0.2 + apa * 0.2, 0, 1)
        scores[wi] = s

    return scores.cpu().tolist()


@app.route('/dct_analyze', methods=['POST'])
def dct_analyze():
    """
    GPU 加速 DCT 滑窗分析接口。

    请求: multipart/form-data
        file: 图片文件
        win_size: 可选，窗口大小（默认 128）
        stride: 可选，步长（默认 64）

    返回: JSON
        { code: 200, data: { scores: [float,...], window_count: int, elapsed: float } }
    """
    t0 = time.time()
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "缺少图片文件"}), 400

    file = request.files['file']
    win_size = int(request.form.get('win_size', 128))
    stride = int(request.form.get('stride', 64))

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"code": 400, "msg": f"图片解析失败: {str(e)}"}), 400

    # 转灰度 + 送 GPU
    gray_np = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray_np.shape
    gray = torch.from_numpy(gray_np).to(device).float()  # (H, W)

    # 生成滑窗
    windows = _build_windows(h, w, win_size, stride)
    nw = len(windows)

    if nw < 2:
        return jsonify({
            "code": 200,
            "data": {"scores": [0.0] * nw, "window_count": nw, "elapsed": round(time.time() - t0, 3)}
        })

    # GPU DCT 分析
    scores = _dct_analyze_gpu(gray, windows)

    elapsed = round(time.time() - t0, 3)
    print(f"[DCT] done in {elapsed}s  windows={nw}  mean_score={sum(scores)/max(len(scores),1):.3f}")

    return jsonify({
        "code": 200,
        "data": {"scores": scores, "window_count": nw, "elapsed": elapsed}
    })


@app.route('/srm_analyze', methods=['POST'])
def srm_analyze():
    """
    GPU 加速 SRM 滑窗分析接口（完整：16 滤波器卷积 + 4 矩 + 马氏距离）。

    请求: multipart/form-data
        file: 图片文件
        win_size: 可选，窗口大小（默认 128）
        stride:   可选，步长（默认 64）

    返回: JSON
        { code: 200, data: { scores: [float,...], window_count: int, elapsed: float } }
    """
    t0 = time.time()
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "缺少图片文件"}), 400

    file = request.files['file']
    win_size = int(request.form.get('win_size', 128))
    stride = int(request.form.get('stride', 64))

    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"code": 400, "msg": f"图片解析失败: {str(e)}"}), 400

    gray_np = np.array(image.convert("L"), dtype=np.float32)
    h, w = gray_np.shape
    gray = torch.from_numpy(gray_np).to(device).float()

    windows = _build_windows(h, w, win_size, stride)
    nw = len(windows)

    if nw < 4:
        return jsonify({
            "code": 200,
            "data": {"scores": [0.0] * nw, "window_count": nw, "elapsed": round(time.time() - t0, 3)}
        })

    scores = _srm_analyze_gpu(gray, windows)

    elapsed = round(time.time() - t0, 3)
    print(f"[SRM] done in {elapsed}s  windows={nw}  mean_score={sum(scores)/max(len(scores),1):.3f}")

    return jsonify({
        "code": 200,
        "data": {"scores": scores, "window_count": nw, "elapsed": elapsed}
    })


def _trufor_map_png(values):
    """Encode a [0,1] TruFor map as a bounded grayscale data URL."""
    array = np.asarray(values, dtype=np.float32).squeeze()
    if array.ndim != 2:
        raise ValueError("TruFor map is not two-dimensional")
    array = np.clip(array, 0.0, 1.0)
    image = Image.fromarray(np.uint8(array * 255), mode="L")
    if max(image.size) > 512:
        scale = 512.0 / max(image.size)
        image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BILINEAR)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def _trufor_worker_ready(timeout=1.5):
    """Return whether the persistent TruFor worker has completed model loading."""
    try:
        response = requests.get(f"{TRUFOR_WORKER_URL}/health", timeout=timeout)
        return response.status_code == 200 and bool(response.json().get("ready"))
    except Exception:
        return False


def _ensure_trufor_worker():
    """Start the isolated TruFor worker once and wait for its readiness.

    The main API uses a different Conda environment from TruFor.  Starting a
    separate local worker preserves the official runtime while avoiding the
    prior per-request process startup and checkpoint load.
    """
    global _trufor_worker_last_launch
    if _trufor_worker_ready():
        return True

    with _trufor_worker_lock:
        if _trufor_worker_ready():
            return True

        # Avoid repeatedly forking failed workers when a deployment is missing
        # a dependency.  The legacy one-shot path remains available below.
        now = time.monotonic()
        if now - _trufor_worker_last_launch < 15:
            return False
        _trufor_worker_last_launch = now

        if not (Path(TRUFOR_WORKER_FILE).is_file() and Path(TRUFOR_PYTHON).is_file() and Path(TRUFOR_WEIGHTS).is_file()):
            print("[TruFor] persistent worker prerequisites are unavailable; using legacy runner")
            return False

        try:
            Path(TRUFOR_WORKER_LOG).parent.mkdir(parents=True, exist_ok=True)
            worker_env = os.environ.copy()
            worker_env.update({
                "TRUFOR_ROOT": TRUFOR_ROOT,
                "TRUFOR_WEIGHTS": TRUFOR_WEIGHTS,
            })
            with open(TRUFOR_WORKER_LOG, "ab", buffering=0) as worker_log:
                subprocess.Popen(
                    [TRUFOR_PYTHON, TRUFOR_WORKER_FILE, "--host", TRUFOR_WORKER_HOST,
                     "--port", str(TRUFOR_WORKER_PORT)],
                    cwd=str(Path(TRUFOR_WORKER_FILE).parent), env=worker_env,
                    stdin=subprocess.DEVNULL, stdout=worker_log, stderr=worker_log,
                    start_new_session=True,
                )
        except Exception as exc:
            print(f"[TruFor] failed to launch persistent worker: {type(exc).__name__}: {exc}")
            return False

        deadline = time.monotonic() + TRUFOR_WORKER_START_TIMEOUT
        while time.monotonic() < deadline:
            if _trufor_worker_ready():
                print("[TruFor] persistent worker ready")
                return True
            time.sleep(0.5)

        print("[TruFor] persistent worker did not become ready; using legacy runner")
        return False


def _run_trufor_legacy(image: Image.Image):
    """Run the official TruFor script once; only used as a safe fallback."""
    root = Path(TRUFOR_ROOT)
    runner = root / "test.py"
    if not runner.is_file() or not Path(TRUFOR_WEIGHTS).is_file() or not Path(TRUFOR_PYTHON).is_file():
        raise FileNotFoundError("TruFor runtime, source tree, or checkpoint is not installed")
    with tempfile.TemporaryDirectory(prefix="trufor_") as temp_dir:
        work = Path(temp_dir)
        input_path, output_path = work / "image.png", work / "result.npz"
        image.save(input_path, format="PNG")
        command = [TRUFOR_PYTHON, str(runner), "-g", "0", "-in", str(input_path), "-out", str(output_path),
                   "-exp", "trufor_ph3", "TEST.MODEL_FILE", TRUFOR_WEIGHTS]
        completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True, timeout=TRUFOR_TIMEOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "TruFor inference failed")[-500:])
        if not output_path.is_file():
            raise FileNotFoundError("TruFor did not produce an NPZ result")
        with np.load(output_path) as result:
            score = float(np.asarray(result["score"]).squeeze())
            anomaly = np.asarray(result["map"], dtype=np.float32).squeeze()
            confidence = np.asarray(result["conf"], dtype=np.float32).squeeze()
        reliability = float(np.mean(np.clip(confidence, 0.0, 1.0)))
        coverage = float(np.mean(np.clip(anomaly, 0.0, 1.0) >= 0.50))
        return {
            "available": True,
            "score": round(float(np.clip(score, 0.0, 1.0)), 4),
            "reliability": round(reliability, 4),
            "map_coverage": round(coverage, 4),
            "map_png": _trufor_map_png(anomaly),
            "confidence_png": _trufor_map_png(confidence),
            "detail": "TruFor 已输出完整性分数、疑似篡改区域与定位可靠性。",
        }


def _run_trufor(image: Image.Image):
    """Infer through the persistent TruFor runtime, with legacy fallback."""
    # Do the same high-quality, aspect-ratio-preserving normalization as the
    # worker before encoding the loopback request.  Besides reducing latency,
    # this prevents camera originals from exceeding MAX_IMAGE_BYTES after PNG
    # serialization.  The worker repeats the size guard for direct requests.
    model_image = image.convert("RGB")
    if max(model_image.size) > TRUFOR_MAX_EDGE:
        scale = TRUFOR_MAX_EDGE / max(model_image.size)
        model_image = model_image.resize(
            (max(1, round(model_image.width * scale)), max(1, round(model_image.height * scale))),
            Image.Resampling.LANCZOS,
        )

    if _ensure_trufor_worker():
        try:
            image_data = io.BytesIO()
            model_image.save(image_data, format="PNG")
            response = requests.post(
                f"{TRUFOR_WORKER_URL}/infer", data=image_data.getvalue(),
                headers={"Content-Type": "image/png"}, timeout=TRUFOR_TIMEOUT,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get("available"):
                return payload
            raise RuntimeError(payload.get("detail", "persistent TruFor worker returned no result"))
        except Exception as exc:
            # Availability must not silently become a genuine-image verdict.
            # The established official-script path keeps service continuity if a
            # worker is interrupted after it has started.
            print(f"[TruFor] persistent worker request failed; falling back: {type(exc).__name__}: {exc}")

    return _run_trufor_legacy(model_image)


@app.route('/trufor_analyze', methods=['POST'])
def trufor_analyze():
    if 'file' not in request.files:
        return jsonify({"code": 400, "data": {"available": False, "detail": "缺少图片文件"}}), 400
    try:
        image = Image.open(request.files['file'].stream).convert("RGB")
        release_models_for_trufor()
        return jsonify({"code": 200, "data": _run_trufor(image)})
    except Exception as exc:
        print(f"[TruFor] unavailable: {type(exc).__name__}: {exc}")
        return jsonify({"code": 503, "data": {
            "available": False,
            "detail": "TruFor 模型未就绪或推理失败；未输出篡改结论。",
        }}), 503


def _run_adaifl(image: Image.Image):
    """Run the official AdaIFL test script and return its binary mask.

    The upstream script writes a 0/255 localization mask rather than an
    image-level probability.  Coverage is therefore exposed only for later
    acceptance-set calibration and cannot replace TruFor's primary score.
    """
    root = Path(ADAIFL_ROOT)
    inference = root / "test.py"
    if not inference.is_file() or not Path(ADAIFL_MODEL_PATH).is_file():
        raise FileNotFoundError("AdaIFL source tree or checkpoint is not installed")
    with tempfile.TemporaryDirectory(prefix="adaifl_") as temp_dir:
        work = Path(temp_dir)
        input_path, output = work / "image.png", work / "output"
        output.mkdir(parents=True, exist_ok=True)
        image.save(input_path, format="PNG")
        command = [ADAIFL_PYTHON, str(inference), "--image", str(input_path),
                   "--model", ADAIFL_MODEL_PATH, "--output", str(output)]
        completed = subprocess.run(command, cwd=str(root), capture_output=True, text=True,
                                   timeout=ADAIFL_TIMEOUT, check=False)
        if completed.returncode != 0:
            raise RuntimeError((completed.stderr or completed.stdout or "AdaIFL inference failed")[-500:])
        mask_path = output / input_path.name
        if not mask_path.is_file():
            raise FileNotFoundError("AdaIFL did not produce a localization mask")
        values = np.asarray(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
        coverage = float(np.mean(values >= 0.50))
        return {
            "available": True,
            "calibrated": ADAIFL_CALIBRATED,
            "score": round(coverage, 4),
            "map_coverage": round(coverage, 4),
            "map_png": _trufor_map_png(values),
            "detail": ("AdaIFL 定位完成，已启用独立验收集校准阈值。" if ADAIFL_CALIBRATED
                       else "AdaIFL 定位完成；输出为定位掩码，尚未校准为整图判定，未参与最终定性。"),
        }


@app.route('/adaifl_analyze', methods=['POST'])
def adaifl_analyze():
    if 'file' not in request.files:
        return jsonify({"code": 400, "data": {"available": False, "detail": "缺少图片文件"}}), 400
    try:
        image = Image.open(request.files['file'].stream).convert("RGB")
        return jsonify({"code": 200, "data": _run_adaifl(image)})
    except Exception as exc:
        print(f"[AdaIFL] unavailable: {type(exc).__name__}: {exc}")
        return jsonify({"code": 503, "data": {
            "available": False, "detail": "AdaIFL 未安装、权重未配置或推理失败。",
        }}), 503


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status": "ok", "device": str(device),
        "trufor_worker_ready": _trufor_worker_ready(),
        "adaifl_ready": Path(ADAIFL_ROOT, "test.py").is_file() and Path(ADAIFL_MODEL_PATH).is_file(),
        "adaifl_calibrated": ADAIFL_CALIBRATED,
    })


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({"code": 400, "msg": "缺少图片文件"}), 400

    file = request.files['file']
    selected_models = parse_detection_models()
    if not selected_models:
        return jsonify({"code": 400, "msg": "请至少选择一项检测模型"}), 400
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"code": 400, "msg": f"图片解析失败: {str(e)}"}), 400

    t0 = time.time()

    # 旧的三路基础模型仅提供页面兼容详情，专项融合不再依赖它；
    # 因此不加载/运行它，避免用户只选一个专项时产生额外 GPU 推理。
    base = {
        "result": "not_run", "confidence": 0.0,
        "details": {"ai_generated": 0, "deepfake": 0, "splicing": 0, "real": 0}
    }

    # ── AI vs Human 专项检测（保留，不再作为主模型）──
    aihuman = {"label": "AI vs Human Detector (已停用)", "ai_score": 0, "human_score": 0, "verdict": "已停用", "is_ai": False}

    # ── UniversalFakeDetect 专项检测（跨生成器 AI 全图生成）──
    univfd = None
    aide = None
    dear_r = None
    probe_dinov2 = None
    if "ai_generated" in selected_models:
        try:
            univfd = get_univfd().predict(image)
        except Exception as e:
            print(f"[ERROR] UnivFD检测失败: {e}")
            univfd = {"label": "UniversalFakeDetect (UnivFD)", "ai_score": None,
                      "human_score": None, "verdict": "检测不可用", "is_ai": False,
                      "available": False, "error": str(e)}

    # ── AIDE 专项检测（ICLR 2025，伪影 + DCT 噪声特征）──
        try:
            aide = get_aide().predict(image)
        except Exception as e:
            print(f"[ERROR] AIDE检测失败: {e}")
            aide = {"label": "AIDE (GenImage)", "ai_score": None,
                    "human_score": None, "verdict": "检测不可用", "is_ai": False,
                    "available": False, "error": str(e)}

    # ── DEAR-r 专项检测（ICML 2026，后处理鲁棒性）──
        try:
            dear_r = get_dear_r().predict(image)
        except Exception as e:
            print(f"[ERROR] DEAR-r检测失败: {e}")
            dear_r = {"label": "DEAR-r", "ai_score": None,
                      "human_score": None, "verdict": "检测不可用", "is_ai": False,
                      "available": False, "error": str(e)}

    # ── PROBE-DINOv2 专项检测（ICML 2026，未见生成器泛化）──
        try:
            probe_dinov2 = get_probe_dinov2().predict(image)
        except Exception as e:
            print(f"[ERROR] PROBE-DINOv2检测失败: {e}")
            probe_dinov2 = {"label": "PROBE-DINOv2", "ai_score": None,
                             "human_score": None, "verdict": "检测不可用", "is_ai": False,
                             "available": False, "error": str(e)}

    # ── DeepFake 专项检测 (ViT-B) ──
    deepfakev2 = None
    face_glm = None
    if "deepfake" in selected_models:
        try:
            deepfakev2 = get_deepfakev2().predict(image)
        except Exception as e:
            print(f"[ERROR] DeepFakeV2检测失败: {e}")
            deepfakev2 = {"label": "DeepFake Detector v2", "deepfake_score": 0,
                          "real_score": 0, "verdict": "检测失败", "is_deepfake": False}

        # GLM is a part of the selected deepfake chain and is never called otherwise.
        face_glm = detect_face_glm_remote(image)

    elapsed = round(time.time() - t0, 2)
    print(f"[Detect] done in {elapsed}s models={','.join(selected_models)}")

    return jsonify({
        "code": 200,
        "data": {
            **base,  # result, confidence, details
            "selected_models": list(selected_models),
            "face_glm": face_glm,
            "specialized_models": {
                "ai_vs_human": aihuman,
                "univfd": univfd,
                "aide": aide,
                "dear_r": dear_r,
                "probe_dinov2": probe_dinov2,
                "deepfake_detector": deepfakev2,
            }
        }
    })


# ──────────────────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--port', type=int, default=6006, help='服务端口')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='绑定地址')
    args = parser.parse_args()

    # ── 启动时预加载模型（避免首次请求超时）──
    import sys as _sys
    preload = os.environ.get("PRELOAD_MODELS", "1")
    if preload == "1":
        print("[Startup] 预加载 UnivFD（首次需下载 CLIP ViT-L/14，约 900MB）...")
        _sys.stdout.flush()
        try:
            get_univfd()
            print("[Startup] UnivFD 模型加载完成")
        except Exception as e:
            print(f"[Startup] UnivFD 预加载失败（不影响其他检测链路）: {e}")

        print("[Startup] 预加载 AIDE（GenImage checkpoint，约 3.4GB）...")
        _sys.stdout.flush()
        try:
            get_aide()
            print("[Startup] AIDE 模型加载完成")
        except Exception as e:
            print(f"[Startup] AIDE 预加载失败（不影响其他检测链路）: {e}")

        print("[Startup] 预加载 DEAR-r（ICML 2026，约 90MB）...")
        _sys.stdout.flush()
        try:
            get_dear_r()
            print("[Startup] DEAR-r 模型加载完成")
        except Exception as e:
            print(f"[Startup] DEAR-r 预加载失败（不影响其他检测链路）: {e}")

        print("[Startup] 预加载 PROBE-DINOv2（ICML 2026，约 1.2GB）...")
        _sys.stdout.flush()
        try:
            get_probe_dinov2()
            print("[Startup] PROBE-DINOv2 模型加载完成")
        except Exception as e:
            print(f"[Startup] PROBE-DINOv2 预加载失败（不影响其他检测链路）: {e}")

        print(f"[Startup] 预加载 DeepFakeV2 模型（首次需从 HuggingFace 下载 ~330MB，请耐心等待）...")
        _sys.stdout.flush()
        try:
            get_deepfakev2()
            print(f"[Startup] DeepFakeV2 模型加载完成")
        except Exception as e:
            print(f"[Startup] DeepFakeV2 预加载失败（将在首次请求时重试）: {e}")

    # TruFor depends on a dedicated Conda environment, so it is preloaded by a
    # loopback worker rather than into this process.  Its failure is non-fatal:
    # /trufor_analyze keeps the validated one-shot runner as a fallback.
    if os.environ.get("PRELOAD_TRUFOR", "1") == "1":
        print("[Startup] 预加载 TruFor 常驻模型...")
        if not _ensure_trufor_worker():
            print("[Startup] TruFor 常驻模型未就绪，将在首个篡改请求时重试并保留旧路径兜底")

    print(f"\n{'='*50}")
    print(f"  GPU 推理服务启动: http://{args.host}:{args.port}")
    print(f"  设备: {device}")
    print(f"  模型缓存: {MODEL_CACHE}")
    print(f"{'='*50}\n")

    app.run(host=args.host, port=args.port, debug=False)
