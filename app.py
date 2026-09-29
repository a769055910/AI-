# -*- coding: utf-8 -*-
"""
AI 伪造鉴别智能体 - Web 展示框架
基于 Flask，兼容 Windows Server 2016
"""
import os
import uuid
import json
import requests
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from flask import Flask, render_template, request, jsonify, url_for, Response, stream_with_context, send_from_directory
from npr_detector import npr_analyze
from deepfake_detector import deepfake_zero_model_analysis
from tamper_detector import tamper_zero_model_analysis
from camera_forensics import analyze_camera_imaging
from watermark_pipeline import detect_visible_ai_watermark
from hidden_watermark_detector import detect_hidden_watermark
from content_risk_analyzer import analyze_ai_image_content, should_analyze_content, normalize_crime_scene, CRIME_SCENES
from batch_detector import create_task, get_all_tasks, get_task_summary, get_task_images, get_image_detail, delete_task
from image_quality import validate_uploaded_file

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 512 * 1024 * 1024  # 最大上传 512MB（视频可能较大）


@app.after_request
def prevent_stale_html_cache(response):
    """避免浏览器继续使用已更新前端模板的旧页面副本。"""
    if response.content_type and response.content_type.startswith('text/html'):
        response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        response.headers['Pragma'] = 'no-cache'
        response.headers['Expires'] = '0'
    return response

# 图像专项鉴别 Skill：只允许页面读取本白名单中的技能，避免任意本机文件暴露。
_CODEX_SKILLS_DIR = os.path.join(
    os.environ.get('CODEX_HOME', os.path.join(os.path.expanduser('~'), '.codex')),
    'skills'
)
SKILL_CATALOG = {
    'common': {
        'title': 'AI 图片鉴定公共模块',
        'skill_name': 'ai-image-forensics-common',
        'filename': 'ai-image-forensics-common.md',
        'summary': '涉诈、涉谣、涉黄三类鉴别 Skill 共用的证据、研判、分级与安全原则。',
        'editable': False,
    },
    'fraud': {
        'title': '涉诈鉴别 Skill',
        'skill_name': 'fraud-ai-image-forensics',
        'summary': '鉴定涉诈语境中的 AI 生成、换脸、合成与 AI 驱动篡改风险。',
    },
    'rumor': {
        'title': '涉谣鉴别 Skill',
        'skill_name': 'rumor-ai-image-forensics',
        'summary': '鉴定公共事件和新闻配图中的 AI 合成与涉谣传播风险。',
    },
    'porn': {
        'title': '涉黄鉴别 Skill',
        'skill_name': 'porn-ai-image-forensics',
        'summary': '鉴定疑似不雅图像中的 AI 换脸、合成与侵害勒索风险。',
    },
}

# 上传目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# GPU 推理服务地址（AutoDL → SSH 隧道本地转发）。可用环境变量切换实例，默认保留原端口。
GPU_API = os.environ.get("GPU_API", "http://127.0.0.1:6006")

# 检测类型配置 — 仅保留图像检测
DETECT_TYPES = {
    'image': {
        'name': '图像检测',
        'icon': 'image',
        'accept': '.jpg,.jpeg,.png,.bmp,.webp',
        'tips': '请上传高清图像，大小 30KB ~ 20MB，最小尺寸 300px',
        'placeholder': '',
        'max_size': '20MB'
    }
}

# 三个专项的请求标识。未携带 models 的旧客户端保持“三项联合检测”兼容行为。
DETECTION_MODELS = ('ai_generated', 'deepfake', 'tamper')
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
# 深度换脸专项统一阈值：严格超过 60% 才判为 AI/DeepFake，60% 及以下为真实。
DEEPFAKE_AI_THRESHOLD = 0.60
# AI 全图生成与图像篡改专项统一阈值：达到 60% 即判为检出；低于 60% 为真实图片。
AI_GENERATED_THRESHOLD = 0.60
TAMPER_THRESHOLD = 0.60


def parse_detection_models(values=None):
    """规范化前端提交的专项选择，返回按固定顺序排列的元组。"""
    if values is None:
        values = request.form.getlist('models')
    if isinstance(values, str):
        values = [values]

    requested = set()
    for value in values or []:
        requested.update(item.strip() for item in str(value).split(',') if item.strip())

    # 兼容旧版页面：没有 models 字段时仍执行全部专项。
    if not requested:
        return DETECTION_MODELS
    return tuple(model for model in DETECTION_MODELS if model in requested)


def parse_crime_scene(value=None):
    """读取并校验用户选择的内容识别专项场景。"""
    if value is None:
        value = request.form.get('crime_scene')
    return normalize_crime_scene(value)


def validate_detection_image(file_obj):
    """在保存、推理前拦截不满足最低检测质量的上传图片。"""
    ext = os.path.splitext(file_obj.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return {'valid': False, 'reason': '仅支持 JPG、JPEG、PNG、BMP、WebP 图片。'}
    return validate_uploaded_file(file_obj)


def fuse_npr_with_physical_evidence(npr_result, camera_forensics):
    """将单图相机成像物证作为 NPR 的内部物理维度。

    该步骤仅服务 AI 全图生成：原始 NPR 保持主导，传感器噪声、CFA 去马赛克和
    采集痕迹只提供受限校正，避免截图/压缩图片因缺少相机信息而被直接判为 AI。
    """
    if not npr_result or not (camera_forensics or {}).get('available'):
        return npr_result

    raw_score = float(npr_result.get('score', 0.5) or 0.5)
    trace_score = float(camera_forensics.get('camera_trace_score', 0.5) or 0.5)
    noise = camera_forensics.get('noise') or {}
    cfa = camera_forensics.get('cfa') or {}
    exif = camera_forensics.get('exif') or {}

    # 相机物证的 AI 可疑度：弱采集痕迹、弱噪声一致性、弱 CFA 周期性和无相机
    # 元数据。它不独立定性，只以 20% 权重参与 NPR 物理融合。
    noise_weakness = 0.5
    if noise.get('available'):
        sigma = float(noise.get('noise_sigma', 0.0) or 0.0)
        consistency = float(noise.get('noise_consistency', 0.0) or 0.0)
        noise_presence = min(sigma / 0.012, 1.0)
        noise_weakness = 1.0 - (0.55 * noise_presence + 0.45 * consistency)
    cfa_weakness = 1.0 - float(cfa.get('cfa_trace_score', 0.0) or 0.0)
    metadata_weakness = 0.0 if exif.get('has_camera_tags') else 1.0
    camera_physical_score = (
        0.55 * (1.0 - trace_score) + 0.25 * noise_weakness +
        0.15 * cfa_weakness + 0.05 * metadata_weakness
    )
    fused_score = max(0.0, min(1.0, 0.80 * raw_score + 0.20 * camera_physical_score))

    # 保留内部审计字段，前端只展示最终 NPR 综合得分。
    npr_result['raw_noise_score'] = round(raw_score, 4)
    npr_result['score'] = round(fused_score, 4)
    npr_result['physical_fused'] = True
    npr_result['physical_dimensions'] = {
        'camera_trace_weakness': round(1.0 - trace_score, 4),
        'noise_weakness': round(noise_weakness, 4),
        'cfa_weakness': round(cfa_weakness, 4),
        'metadata_weakness': round(metadata_weakness, 4),
    }
    return npr_result


# ──────────────────────────────────────────────────────────
# NPR + AIRealNet + UnivFD + AIDE 融合判定逻辑
# ──────────────────────────────────────────────────────────
def _compute_combined_verdict_legacy(npr_result, specialized_models):
    """
    融合 NPR 噪声模式分析、AIRealNet、UnivFD 与 AIDE 结果。
    任一专项模型不可用时自动对可用链路重新归一化，避免单点故障中断检测。
    输出综合判定结论

    npr_result: { score (0~1), verdict, features, ... }
    specialized_models: {
        ai_realnet: { ai_score, human_score, verdict, ... },
        univfd: { ai_score, human_score, verdict, available, ... },
        aide: { ai_score, human_score, verdict, available, ... }
    }

    返回:
    {
        final_score: float,      # 综合 AI 概率 (0~1)
        final_verdict: str,      # 最终判定文本
        confidence: str,         # 置信度等级 (高/中/低)
        agreement: str,          # 检测链路一致性
        detail: str,             # 详细解释
        npr_contribution: float, # NPR 贡献权重
        model_contribution: float, # AIRealNet 模型贡献权重
        univfd_contribution: float # UnivFD 模型贡献权重（可用时）
        aide_contribution: float   # AIDE 模型贡献权重（可用时）
    }
    """
    # 初始融合权重，后续应使用验收集上的校准结果替代：
    # AIRealNet 擅长常见图像生成来源，UnivFD 强化跨生成器泛化，
    # AIDE 提供伪影 + DCT 噪声证据，NPR 用于补充轻量频域证据。
    W_NPR_FALLBACK = 0.4
    W_MODEL_FALLBACK = 0.6
    W_NPR = 0.20
    W_AIREALNET = 0.30
    W_UNIVFD = 0.20
    W_AIDE = 0.30
    W_DEAR_R = 0.15
    W_PROBE_DINOV2 = 0.30
    THRESH_HIGH = 0.75   # ≥75%：AI 生成
    THRESH_MID  = 0.55   # 55%-75%：疑似 AI 生成；<55%：真实图片

    # 提取 NPR 分数（0~1，越高越像 AI 生成）
    npr_score = npr_result.get('score', 0.5) if npr_result else 0.5
    npr_verdict = npr_result.get('verdict', 'uncertain') if npr_result else 'uncertain'

    # 提取 AIRealNet 模型分数
    ai_realnet_data = specialized_models.get('ai_realnet', {}) if specialized_models else {}
    model_ai_score = ai_realnet_data.get('ai_score', 0.5)
    univfd_data = specialized_models.get('univfd', {}) if specialized_models else {}
    univfd_ai_score = univfd_data.get('ai_score')
    univfd_available = (
        univfd_data.get('available', False) is True
        and isinstance(univfd_ai_score, (int, float))
    )
    aide_data = specialized_models.get('aide', {}) if specialized_models else {}
    aide_ai_score = aide_data.get('ai_score')
    aide_available = (
        aide_data.get('available', False) is True
        and isinstance(aide_ai_score, (int, float))
    )
    dear_r_data = specialized_models.get('dear_r', {}) if specialized_models else {}
    dear_r_ai_score = dear_r_data.get('ai_score')
    dear_r_available = (
        dear_r_data.get('available', False) is True
        and isinstance(dear_r_ai_score, (int, float))
    )
    probe_data = specialized_models.get('probe_dinov2', {}) if specialized_models else {}
    probe_ai_score = probe_data.get('ai_score')
    probe_available = (
        probe_data.get('available', False) is True
        and isinstance(probe_ai_score, (int, float))
    )

    # ── 加权融合 ──
    components = [("NPR", npr_score, W_NPR), ("AIRealNet", model_ai_score, W_AIREALNET)]
    if univfd_available:
        components.append(("UnivFD", univfd_ai_score, W_UNIVFD))
    if aide_available:
        components.append(("AIDE", aide_ai_score, W_AIDE))
    if dear_r_available:
        components.append(("DEAR-r", dear_r_ai_score, W_DEAR_R))
    if probe_available:
        components.append(("PROBE-DINOv2", probe_ai_score, W_PROBE_DINOV2))

    # 仅剩旧双路时保持原有 0.4 / 0.6 融合；其余情况下按可用模型重新归一化。
    if not univfd_available and not aide_available:
        components = [("NPR", npr_score, W_NPR_FALLBACK), ("AIRealNet", model_ai_score, W_MODEL_FALLBACK)]
    weight_total = sum(weight for _, _, weight in components)
    normalized = [(name, score, weight / weight_total) for name, score, weight in components]
    weights = {name: weight for name, _, weight in normalized}
    combined_score = round(sum(score * weight for _, score, weight in normalized), 4)
    W_NPR = weights["NPR"]
    W_AIREALNET = weights["AIRealNet"]
    W_UNIVFD = weights.get("UnivFD", 0.0)
    W_AIDE = weights.get("AIDE", 0.0)
    W_DEAR_R = weights.get("DEAR-r", 0.0)
    W_PROBE_DINOV2 = weights.get("PROBE-DINOv2", 0.0)

    # NPR 判定方向
    npr_is_ai = npr_score >= THRESH_MID
    model_is_ai = model_ai_score >= THRESH_MID

    votes = [("NPR", npr_is_ai), ("AIRealNet", model_is_ai)]
    if univfd_available:
        votes.append(("UnivFD", univfd_ai_score >= THRESH_MID))
    if aide_available:
        votes.append(("AIDE", aide_ai_score >= THRESH_MID))
    if dear_r_available:
        votes.append(("DEAR-r", dear_r_ai_score >= THRESH_MID))
    if probe_available:
        votes.append(("PROBE-DINOv2", probe_ai_score >= THRESH_MID))
    vote_values = [value for _, value in votes]
    vote_count_name = {2: "双路", 3: "三路", 4: "四路"}.get(len(votes), f"{len(votes)}路")
    if all(vote_values):
        agreement = f"{vote_count_name}一致"
        agreement_detail = f"{'、'.join(name for name, _ in votes)} 均判定为 AI 生成"
    elif not any(vote_values):
        agreement = f"{vote_count_name}一致"
        agreement_detail = f"{'、'.join(name for name, _ in votes)} 均判定为真实图片"
    elif len(votes) == 2 and npr_is_ai and model_is_ai:
        agreement = "双模型一致"
        agreement_detail = "NPR 噪声分析与 AIRealNet 检测器均判定为 AI 生成"
    else:
        agreement = "模型分歧"
        ai_names = [name for name, is_ai in votes if is_ai]
        real_names = [name for name, is_ai in votes if not is_ai]
        agreement_detail = f"AI 判定：{'、'.join(ai_names)}；真实判定：{'、'.join(real_names)}"

    # ── 最终判定（三档）──
    if combined_score >= THRESH_HIGH:
        final_verdict = "AI生成图片"
        confidence = "高" if "一致" in agreement else "中高"
    elif combined_score >= THRESH_MID:
        final_verdict = "疑似AI生成图片"
        confidence = "中" if "一致" in agreement else "中低"
    else:
        final_verdict = "真实图片"
        confidence = "高" if "一致" in agreement else "中"

    # 有分歧时降一级置信度
    if agreement == "模型分歧":
        conf_order = ["高", "中高", "中", "中低", "低"]
        idx = conf_order.index(confidence) if confidence in conf_order else 2
        confidence = conf_order[min(idx + 1, len(conf_order) - 1)]

    # ── 生成详细解释 ──
    detail_terms = [
        f"NPR 贡献 {npr_score:.2f} × {W_NPR}",
        f"AIRealNet 贡献 {model_ai_score:.2f} × {W_AIREALNET}",
    ]
    if univfd_available:
        detail_terms.append(f"UnivFD 贡献 {univfd_ai_score:.2f} × {W_UNIVFD}")
    if aide_available:
        detail_terms.append(f"AIDE 贡献 {aide_ai_score:.2f} × {W_AIDE}")
    if dear_r_available:
        detail_terms.append(f"DEAR-r 贡献 {dear_r_ai_score:.2f} × {W_DEAR_R}")
    if probe_available:
        detail_terms.append(f"PROBE-DINOv2 贡献 {probe_ai_score:.2f} × {W_PROBE_DINOV2}")
    detail = (
        f"{agreement_detail}。综合加权得分 {combined_score:.2f}（{' + '.join(detail_terms)}），"
        f"判定为「{final_verdict}」，置信度: {confidence}。"
    )

    return {
        "final_score": combined_score,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "agreement": agreement,
        "agreement_detail": agreement_detail,
        "detail": detail,
        "npr_contribution": round(npr_score * W_NPR, 4),
        "model_contribution": round(model_ai_score * W_AIREALNET, 4),
        # 所有神经网络检测器在融合结果中的合计贡献；单模型原始分数。
        # 仅在前端“神经网络分析”细节中展示。
        "neural_contribution": round(
            model_ai_score * W_AIREALNET
            + (univfd_ai_score * W_UNIVFD if univfd_available else 0.0)
            + (aide_ai_score * W_AIDE if aide_available else 0.0),
            4,
        ),
        "npr_score": npr_score,
        "model_ai_score": model_ai_score,
        "univfd_available": univfd_available,
        "univfd_ai_score": univfd_ai_score if univfd_available else None,
        "univfd_contribution": round(univfd_ai_score * W_UNIVFD, 4) if univfd_available else 0.0,
        "aide_available": aide_available,
        "aide_ai_score": aide_ai_score if aide_available else None,
        "aide_contribution": round(aide_ai_score * W_AIDE, 4) if aide_available else 0.0,
        "dear_r_available": dear_r_available,
        "dear_r_ai_score": dear_r_ai_score if dear_r_available else None,
        "dear_r_contribution": round(dear_r_ai_score * W_DEAR_R, 4) if dear_r_available else 0.0,
        "probe_available": probe_available,
        "probe_ai_score": probe_ai_score if probe_available else None,
        "probe_contribution": round(probe_ai_score * W_PROBE_DINOV2, 4) if probe_available else 0.0,
        "fusion_mode": " + ".join(name for name, _, _ in normalized),
    }


# ──────────────────────────────────────────────────────────
# DeepFake 零模型 + 纯模型 融合判定逻辑
# ──────────────────────────────────────────────────────────
def compute_combined_verdict(npr_result, specialized_models):
    """Fuse physical evidence with the available AI-image neural detectors."""
    specialized_models = specialized_models or {}
    npr_score = float((npr_result or {}).get("score", 0.5))
    npr_score = max(0.0, min(1.0, npr_score))

    configured_models = (
        ("UnivFD", "univfd", 0.20),
        ("AIDE", "aide", 0.35),
        ("DEAR-r", "dear_r", 0.15),
        ("PROBE-DINOv2", "probe_dinov2", 0.30),
    )
    components = [("NPR", npr_score, 0.20)]
    available = {}
    for display_name, key, weight in configured_models:
        item = specialized_models.get(key) or {}
        score = item.get("ai_score")
        if item.get("available", False) is True and isinstance(score, (int, float)):
            score = max(0.0, min(1.0, float(score)))
            available[key] = score
            components.append((display_name, score, weight))

    total_weight = sum(weight for _, _, weight in components)
    normalized = [(name, score, weight / total_weight) for name, score, weight in components]
    weights = {name: weight for name, _, weight in normalized}
    final_score = round(sum(score * weight for _, score, weight in normalized), 4)

    votes = [(name, score >= AI_GENERATED_THRESHOLD) for name, score, _ in normalized]
    ai_names = [name for name, is_ai in votes if is_ai]
    real_names = [name for name, is_ai in votes if not is_ai]
    vote_count = len(votes)
    route_name = {1: "单路", 2: "双路", 3: "三路", 4: "四路", 5: "五路"}.get(vote_count, f"{vote_count}路")
    if len(ai_names) == vote_count:
        agreement = f"{route_name}一致"
        agreement_detail = f"{'、'.join(ai_names)} 均判定为 AI 生成"
    elif not ai_names:
        agreement = f"{route_name}一致"
        agreement_detail = f"{'、'.join(real_names)} 均判定为真实图片"
    else:
        agreement = "模型分歧"
        agreement_detail = f"AI 判定：{'、'.join(ai_names)}；真实判定：{'、'.join(real_names)}"

    if final_score >= AI_GENERATED_THRESHOLD:
        final_verdict, confidence = "AI生成图片", "高" if agreement.endswith("一致") else "中"
    else:
        final_verdict, confidence = "真实图片", "高"
    if agreement == "模型分歧":
        confidence = "中低" if confidence == "中" else "中"

    detail_terms = [f"{name} 贡献 {score:.2f} × {weight:.2f}" for name, score, weight in normalized]
    detail = (
        f"{agreement_detail}。综合加权得分 {final_score:.2f}（{' + '.join(detail_terms)}），"
        f"判定为“{final_verdict}”，置信度：{confidence}。"
    )

    def model_payload(key, display_name):
        score = available.get(key)
        weight = weights.get(display_name, 0.0)
        return score is not None, score, round((score or 0.0) * weight, 4)

    univfd_available, univfd_score, univfd_contribution = model_payload("univfd", "UnivFD")
    aide_available, aide_score, aide_contribution = model_payload("aide", "AIDE")
    dear_available, dear_score, dear_contribution = model_payload("dear_r", "DEAR-r")
    probe_available, probe_score, probe_contribution = model_payload("probe_dinov2", "PROBE-DINOv2")
    neural_contribution = round(
        univfd_contribution + aide_contribution + dear_contribution + probe_contribution, 4
    )
    return {
        "final_score": final_score,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "agreement": agreement,
        "agreement_detail": agreement_detail,
        "detail": detail,
        "npr_contribution": round(npr_score * weights["NPR"], 4),
        "neural_contribution": neural_contribution,
        "npr_score": npr_score,
        "univfd_available": univfd_available,
        "univfd_ai_score": univfd_score,
        "univfd_contribution": univfd_contribution,
        "aide_available": aide_available,
        "aide_ai_score": aide_score,
        "aide_contribution": aide_contribution,
        "dear_r_available": dear_available,
        "dear_r_ai_score": dear_score,
        "dear_r_contribution": dear_contribution,
        "probe_available": probe_available,
        "probe_ai_score": probe_score,
        "probe_contribution": probe_contribution,
        "fusion_mode": " + ".join(name for name, _, _ in normalized),
    }


def compute_deepfake_combined_verdict(deepfake_zero, deepfake_model):
    """
    融合 DeepFake 零模型（ELA）与纯模型（ViT-B）结果
    输出综合 DeepFake 判定结论

    deepfake_zero:  { deepfake_score, verdict, detail, ela, face_detected }
    deepfake_model: { deepfake_score, real_score, verdict, is_deepfake }  (GPU 服务器返回)

    返回:
    {
        final_score: float,
        final_verdict: str,
        confidence: str,
        agreement: str,
        detail: str,
        contributions: {...}
    }
    """
    # 权重配置：零模型 30%，纯模型 70%
    W_ZERO  = 0.30
    W_MODEL = 0.70
    THRESHOLD = DEEPFAKE_AI_THRESHOLD

    # 提取零模型分数
    zero_score = deepfake_zero.get('deepfake_score', 0.0) if deepfake_zero else 0.0
    zero_verdict = deepfake_zero.get('verdict', 'uncertain') if deepfake_zero else 'uncertain'
    face_detected = deepfake_zero.get('face_detected', False) if deepfake_zero else False

    # 提取纯模型分数
    model_score = deepfake_model.get('deepfake_score', 0.0) if deepfake_model else 0.0
    model_verdict = deepfake_model.get('verdict', '未知') if deepfake_model else '未知'
    # GPU 模型自带的 is_deepfake 沿用旧阈值，不能用于最终业务判定。
    model_is_deepfake = model_score > THRESHOLD

    # ── 无可分析人脸：不把“未发现”误表述为“真实人脸”。 ──
    if not face_detected:
        return {
            "final_score": 0.0,
            "final_verdict": "未检测到可分析人脸",
            "confidence": "证据不足",
            "agreement": "人脸证据不足",
            "agreement_detail": "未检测到可用于换脸研判的人脸区域",
            "detail": "图片中未检测到可分析人脸，深度伪造专项不输出肯定或否定结论。",
            "zero_contribution": 0.0,
            "model_contribution": 0.0,
            "zero_score": 0.0,
            "model_score": 0.0,
            "face_detected": False,
        }

    # ── 加权融合 ──
    combined_score = round(W_ZERO * zero_score + W_MODEL * model_score, 4)

    # 零模型判定方向
    zero_is_fake = zero_score > THRESHOLD and zero_verdict != 'error'

    # ── 双模型一致性 ──
    if zero_is_fake and model_is_deepfake:
        agreement = "双模型一致"
        agreement_detail = "零模型（ELA）与纯模型（ViT-B）均判定为深度伪造"
        base_confidence_src = "双模型"
    elif (not zero_is_fake) and (not model_is_deepfake):
        agreement = "双模型一致"
        agreement_detail = "零模型与纯模型均未发现深度伪造痕迹"
        base_confidence_src = "双模型"
    else:
        agreement = "模型分歧"
        if zero_is_fake:
            agreement_detail = "零模型判定为深度伪造，纯模型（ViT-B）判定为真实"
        else:
            agreement_detail = "纯模型（ViT-B）判定为深度伪造，零模型判定为真实"
        base_confidence_src = "单模型"

    # ── 最终判定：超过 60% 为 AI/DeepFake，60% 及以下为真实。──
    if combined_score > THRESHOLD:
        final_verdict = "疑似深度伪造 (DeepFake)"
        if base_confidence_src == "双模型":
            confidence = "高"
        elif base_confidence_src == "纯模型":
            confidence = "中高"
        else:
            confidence = "中"
    else:
        final_verdict = "真实图片/人脸"
        confidence = "高" if base_confidence_src == "双模型" else "中高"

    # 分歧时降级
    if agreement == "模型分歧":
        conf_order = ["高", "中高", "中", "中低", "低"]
        idx = conf_order.index(confidence) if confidence in conf_order else 2
        confidence = conf_order[min(idx + 1, len(conf_order) - 1)]

    # ── 详细解释 ──
    detail = (
        f"{agreement_detail}。"
        f"综合加权得分 {combined_score:.2f}"
        f"（零模型 {zero_score:.2f} × {W_ZERO:.0%} + "
        f"纯模型 {model_score:.2f} × {W_MODEL:.0%}），"
        f"判定为「{final_verdict}」，置信度: {confidence}。"
    )

    return {
        "final_score": combined_score,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "agreement": agreement,
        "agreement_detail": agreement_detail,
        "detail": detail,
        "zero_contribution": round(zero_score * (0.30 if not face_detected else W_ZERO), 4),
        "model_contribution": round(model_score * (0.70 if not face_detected else W_MODEL), 4),
        "zero_score": zero_score,
        "model_score": model_score,
        "face_detected": face_detected,
    }


# ──────────────────────────────────────────────────────────
# 图像篡改 零模型 + 纯模型 融合判定逻辑
# ──────────────────────────────────────────────────────────
def compute_tamper_combined_verdict(tamper_result):
    """
    融合图像篡改零模型 + 纯模型 结果，输出综合判定结论

    tamper_result: {
        tamper_score, verdict, detail,
        copy_move:     { score, match_count, ... },    # 零模型-复制移动
        block_noise:   { score, dispersion, ... },     # 零模型-区块噪声
        patch_feature: { score, mean_similarity, ... }  # 纯模型-区块特征
    }

    返回:
    {
        final_score, final_verdict, confidence,
        agreement, detail, contributions: {...}
    }
    """
    if not tamper_result:
        return {
            "final_score": 0.0,
            "final_verdict": "真实图片",
            "confidence": "低",
            "agreement": "检测失败",
            "agreement_detail": "图像篡改检测模块异常",
            "detail": "图像篡改检测未返回有效结果。",
            "cm_contribution": 0.0,
            "bn_contribution": 0.0,
            "pf_contribution": 0.0,
            "cm_score": 0.0,
            "bn_score": 0.0,
            "pf_score": 0.0,
        }

    THRESH_HIGH = 0.70   # ≥70%：疑似篡改
    THRESH_MID  = 0.50   # 50%-70%：存疑；<50%：真实

    combined_score = tamper_result.get('tamper_score', 0.0)

    # 子模块得分
    cm_result = tamper_result.get('copy_move', {})
    bn_result = tamper_result.get('block_noise', {})
    pf_result = tamper_result.get('patch_feature', {})

    cm_score = cm_result.get('score', 0.0)
    bn_score = bn_result.get('score', 0.0)
    pf_score = pf_result.get('score', 0.0)

    # 一致性子模块判定
    cm_fake = cm_result.get('verdict') == 'fake_likely'
    bn_fake = bn_result.get('verdict') == 'fake_likely'
    pf_fake = pf_result.get('verdict') == 'fake_likely'

    fake_count = sum([cm_fake, bn_fake, pf_fake])

    if fake_count >= 2:
        agreement = "多模块一致"
        agreement_detail = "复制-移动检测、区块噪声分析与区块特征分析中多个模块均检测到异常"
        base_confidence_src = "多模块"
    elif fake_count == 1:
        agreement = "单一模块告警"
        modules = []
        if cm_fake: modules.append("复制-移动检测")
        if bn_fake: modules.append("区块噪声分析")
        if pf_fake: modules.append("区块特征分析")
        agreement_detail = f"仅有{'、'.join(modules)}检测到异常"
        base_confidence_src = "单模块"
    else:
        agreement = "多模块一致"
        agreement_detail = "所有检测模块均未发现明显图像篡改痕迹"
        base_confidence_src = "多模块"

    # 最终判定
    if combined_score >= THRESH_HIGH:
        final_verdict = "疑似图像篡改"
        if base_confidence_src == "多模块":
            confidence = "高"
        else:
            confidence = "中"
    elif combined_score >= THRESH_MID:
        final_verdict = "疑似图像篡改"
        confidence = "中" if base_confidence_src == "多模块" else "中低"
    else:
        final_verdict = "真实图片"
        confidence = "高" if base_confidence_src == "多模块" else "中高"

    # 单一告警降级
    if fake_count == 1:
        conf_order = ["高", "中高", "中", "中低", "低"]
        idx = conf_order.index(confidence) if confidence in conf_order else 2
        confidence = conf_order[min(idx + 1, len(conf_order) - 1)]

    # 详细解释
    detail = (
        f"{agreement_detail}。"
        f"综合加权得分 {combined_score:.2f}"
        f"（复制-移动 {cm_score:.2f} × 25% + "
        f"区块噪声 {bn_score:.2f} × 25% + "
        f"区块特征 {pf_score:.2f} × 50%），"
        f"判定为「{final_verdict}」，置信度: {confidence}。"
    )

    return {
        "final_score": combined_score,
        "final_verdict": final_verdict,
        "confidence": confidence,
        "agreement": agreement,
        "agreement_detail": agreement_detail,
        "detail": detail,
        "cm_contribution": round(cm_score * 0.25, 4),
        "bn_contribution": round(bn_score * 0.25, 4),
        "pf_contribution": round(pf_score * 0.50, 4),
        "cm_score": cm_score,
        "bn_score": bn_score,
        "pf_score": pf_score,
    }


# ──────────────────────────────────────────────────────────
# 三路专项融合得分比较 → 最终判定
# ──────────────────────────────────────────────────────────
def compute_tamper_combined_verdict(tamper_result):
    """Produce an auditable TruFor-led tamper verdict.

    Traditional copy-move/noise signals are intentionally displayed as
    supporting evidence only; they do not change a learned model's score.
    """
    trufor = (tamper_result or {}).get('trufor') or {}
    primary = trufor
    primary_name = 'TruFor'
    if not primary.get('available'):
        return {
            'final_score': 0.0,
            'final_verdict': '篡改模型不可用',
            'confidence': '不可用',
            'agreement': '无模型结论',
            'agreement_detail': trufor.get('detail', 'TruFor 篡改定位模型不可用。'),
            'detail': 'TruFor 未输出可用的篡改结论。',
            'model_contribution': 0.0,
            'model_score': None,
            'model_reliability': None,
        }

    score = float(primary.get('score', 0.0) or 0.0)
    reliability = primary.get('reliability')
    reliability = float(reliability) if isinstance(reliability, (int, float)) else None
    if score >= TAMPER_THRESHOLD:
        verdict, confidence = '疑似图像篡改', '高' if reliability is not None and reliability >= 0.75 else '中'
    else:
        verdict, confidence = '未见明显篡改痕迹', '中' if reliability is None or reliability >= 0.50 else '低'
    return {
        'final_score': round(score, 4),
        'final_verdict': verdict,
        'confidence': confidence,
        'agreement': primary_name + ' 主模型',
        'agreement_detail': primary.get('detail', primary_name + ' 推理完成。'),
        'detail': 'TruFor 完整性分数、定位图与可靠性图共同构成篡改研判结果。',
        'model_contribution': round(score, 4),
        'model_score': round(score, 4),
        'model_reliability': round(reliability, 4) if reliability is not None else None,
    }


def compute_final_comparison(combined, deepfake_combined, tamper_combined, selected_models=None):
    """
    综合三路专项检测得分，输出三档最终判定：
    - AI 全图生成/图像篡改：得分 ≥ 60% 检出；< 60% 为真实图片
    - 深度换脸：沿用严格超过 60% 才检出的专项规则

    combined:          NPR + AIRealNet 融合结果 → AI全图生成专项
    deepfake_combined: ELA + ViT-B 融合结果    → 深度伪造专项
    tamper_combined:   CM + BN + PF 融合结果    → 图像篡改专项

    返回:
    {
        final_label: str,        # ai_generated / suspected_ai / real
        final_verdict: str,      # 最终判定文本
        final_score: float,      # 最高得分
        confidence: float,       # 置信度
        verdict_detail: str,     # 详细解释
        ai_score: float,
        deepfake_score: float,
        tamper_score: float,
        is_real: bool,
    }
    """
    selected_models = tuple(selected_models or DETECTION_MODELS)
    ai_score = combined.get('final_score', 0) if combined and 'ai_generated' in selected_models else 0
    df_score = deepfake_combined.get('final_score', 0) if deepfake_combined and 'deepfake' in selected_models else 0
    tp_score = tamper_combined.get('final_score', 0) if tamper_combined and 'tamper' in selected_models else 0
    candidates = [
        ('ai_generated', ai_score, combined),
        ('deepfake', df_score, deepfake_combined),
        ('tamper', tp_score, tamper_combined),
    ]
    candidates = [item for item in candidates if item[0] in selected_models]
    source, max_score, source_result = max(candidates, key=lambda item: item[1])
    confidence = max_score
    source_threshold = DEEPFAKE_AI_THRESHOLD if source == 'deepfake' else (
        TAMPER_THRESHOLD if source == 'tamper' else AI_GENERATED_THRESHOLD
    )
    is_detected = max_score > source_threshold if source == 'deepfake' else max_score >= source_threshold

    if len(selected_models) == 1 and source_result:
        # 单项检测应保留专项结论，避免把“篡改”或“换脸”笼统显示成 AI 全图生成。
        final_verdict = source_result.get('final_verdict', '检测完成')
        if is_detected:
            final_label = 'ai_generated'
        else:
            final_label = 'real'
    elif is_detected and source == 'deepfake':
        final_verdict = 'AI生成图片（深度伪造）'
        final_label = 'ai_generated'
    elif is_detected:
        final_verdict = '疑似图像篡改' if source == 'tamper' else 'AI生成图片'
        final_label = 'ai_generated'
    else:
        final_verdict = '真实图片'
        final_label = 'real'

    score_labels = {'ai_generated': 'AI全图生成', 'deepfake': '深度伪造', 'tamper': '图像篡改'}
    score_values = {'ai_generated': ai_score, 'deepfake': df_score, 'tamper': tp_score}
    score_detail = '，'.join(
        f'{score_labels[model]} {score_values[model]:.2f}' for model in selected_models
    )
    verdict_detail = (
        f'{score_detail}。最高得分来自{score_labels[source]} {max_score:.2f}，'
        f'判定为「{final_verdict}」。'
    )

    return {
        'final_label': final_label,
        'final_verdict': final_verdict,
        'final_score': round(max_score, 4),
        'confidence': round(confidence, 4),
        'verdict_detail': verdict_detail,
        'ai_score': round(ai_score, 4),
        'deepfake_score': round(df_score, 4),
        'tamper_score': round(tp_score, 4),
        'selected_models': list(selected_models),
        'triggered_model': source,
        'threshold': source_threshold,
        'is_real': final_label == 'real',
    }


def c2pa_credential_present(hidden_watermark_result):
    """Return whether a C2PA credential/marker was found, regardless of trust validation."""
    hidden = hidden_watermark_result or {}
    verification = hidden.get('c2pa_verification') or {}
    metadata_c2pa = (hidden.get('metadata') or {}).get('c2pa') or {}
    return bool(verification.get('present') or metadata_c2pa.get('present'))


def apply_c2pa_ai_evidence(combined_verdict, hidden_watermark_result):
    """Treat any detected C2PA credential as AI-generation evidence.

    Other selected detectors still run; final type is chosen by the highest of
    AI full-image generation, deepfake, and tampering scores.
    """
    provenance = (hidden_watermark_result or {}).get('c2pa_verification') or {}
    metadata_c2pa = ((hidden_watermark_result or {}).get('metadata') or {}).get('c2pa') or {}
    if not c2pa_credential_present(hidden_watermark_result):
        return combined_verdict

    result = dict(combined_verdict or {})
    original_score = float(result.get('final_score', 0.0) or 0.0)
    verified = bool(provenance.get('trust_verified'))
    result.update({
        'final_score': max(original_score, 0.99),
        'final_verdict': 'AI生成图片',
        'confidence': '高',
        'provenance_verified': verified,
        'provenance_status': 'C2PA 凭证已验签' if verified else '检测到 C2PA 凭证/标记，未验证信任链',
        'detail': (
            ('C2PA 内容凭证已通过验证' if verified else '检测到 C2PA 内容凭证/标记（未验证签名、文件绑定或信任链）')
            + ('，来源声明为 AI 生成' if provenance.get('ai_source_declared') or metadata_c2pa.get('ai_source') else '')
            + '；按配置作为 AI 全图生成证据，将该项得分提升为 0.99；最终类别仍由三项检测中的最高分决定。'
            + (' ' + result.get('detail', '') if result.get('detail') else '')
        ),
    })
    return result


def run_content_risk_analysis(image_path, label, label_text, confidence, crime_scene=None):
    """仅在确认图片为 AI 生成后，执行内容识别与知识库研判。"""
    if not should_analyze_content(label):
        return {
            'analyzed': False,
            'status': 'skipped',
            'detail': '图片尚未确认是 AI 生成，未执行内容识别与风险研判。',
            'content_tags': [], 'ocr_text': [], 'risk_tags': [],
            'matched_knowledge': [], 'risk_level': '未知',
            'evidence': [], 'recommended_checks': [],
        }
    crime_scene = normalize_crime_scene(crime_scene)
    if not crime_scene:
        return {
            'analyzed': False,
            'status': 'skipped',
            'detail': '未选择犯罪场景，未执行针对性内容识别。',
            'content_tags': [], 'ocr_text': [], 'risk_tags': [],
            'matched_knowledge': [], 'risk_level': '未知',
            'evidence': [], 'recommended_checks': [],
        }
    return analyze_ai_image_content(image_path, {
        'label': label,
        'label_text': label_text,
        'confidence': confidence,
        'crime_scene': crime_scene,
    })


@app.route('/')
def index():
    """主页面：伪造鉴别仪表板（仅图像检测）"""
    return render_template('index.html')


@app.route('/api/skills/<skill_key>', methods=['GET', 'POST'])
def get_skill_content(skill_key):
    """读取或保存固定图像专项 Skill 的 Markdown。"""
    skill = SKILL_CATALOG.get(skill_key)
    if not skill:
        return jsonify({'error': '未找到该鉴别 Skill'}), 404

    skill_path = os.path.join(
        _CODEX_SKILLS_DIR,
        skill.get('filename') or os.path.join(skill['skill_name'], 'SKILL.md')
    )
    if request.method == 'POST':
        if not skill.get('editable', True):
            return jsonify({'error': '公共模块由三个专项 Skill 共用，当前页面仅支持只读查看。'}), 405
        payload = request.get_json(silent=True) or {}
        markdown = payload.get('markdown')
        if not isinstance(markdown, str) or not markdown.strip():
            return jsonify({'error': 'Skill 内容不能为空'}), 400
        if len(markdown.encode('utf-8')) > 512 * 1024:
            return jsonify({'error': 'Skill 内容不能超过 512KB'}), 400

        expected_name = f"name: {skill['skill_name']}"
        if not markdown.lstrip().startswith('---') or expected_name not in markdown.split('---', 2)[1]:
            return jsonify({'error': f"请保留 YAML 头部中的 {expected_name}"}), 400
        try:
            temp_path = f"{skill_path}.{uuid.uuid4().hex}.tmp"
            with open(temp_path, 'w', encoding='utf-8', newline='\n') as skill_file:
                skill_file.write(markdown)
            os.replace(temp_path, skill_path)
        except OSError:
            try:
                if 'temp_path' in locals() and os.path.exists(temp_path):
                    os.remove(temp_path)
            except OSError:
                pass
            return jsonify({'error': '保存 Skill 文件失败'}), 500
        return jsonify({'ok': True, 'markdown': markdown})

    try:
        with open(skill_path, 'r', encoding='utf-8') as skill_file:
            markdown = skill_file.read()
    except OSError:
        return jsonify({
            'error': '该 Skill 文件暂不可用',
            'title': skill['title'],
            'summary': skill['summary'],
        }), 503

    return jsonify({
        'key': skill_key,
        'title': skill['title'],
        'skill_name': skill['skill_name'],
        'file_name': skill.get('filename', 'SKILL.md'),
        'summary': skill['summary'],
        'editable': skill.get('editable', True),
        'markdown': markdown,
    })


@app.route('/api/detect/image/identifiers', methods=['POST'])
def api_detect_image_identifiers():
    """独立核验 AI 标识编码，不依赖 GPU 推理服务。"""
    file_obj = request.files.get('file')
    if not file_obj or not file_obj.filename:
        return jsonify({'code': 400, 'msg': '请上传图片文件'}), 400

    ext = os.path.splitext(file_obj.filename)[1].lower()
    if ext not in IMAGE_EXTENSIONS:
        return jsonify({'code': 400, 'msg': '仅支持 JPG、JPEG、PNG、BMP、WebP 图片'}), 400
    quality = validate_detection_image(file_obj)
    if not quality['valid']:
        return jsonify({'code': 400, 'msg': quality['reason'], 'quality': quality}), 400

    save_name = f"identifier_{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    file_obj.save(save_path)

    try:
        # 隐式标识在本地完成；可见水印复用原有 GLM 视觉检测。
        hidden_result = detect_hidden_watermark(save_path)
    except Exception as exc:
        hidden_result = {
            'detected': False, 'suspicious': False,
            'detail': f'隐式标识核验异常：{str(exc)[:160]}',
            'tc260': {'fields': {}},
            'c2pa_verification': {'present': False},
        }

    try:
        visible_result = detect_visible_ai_watermark(save_path)
    except Exception as exc:
        visible_result = {
            'detected': False, 'source': None, 'detected_text': None,
            'position': None, 'confidence': 0.0,
            'detail': f'显式水印核验异常：{str(exc)[:160]}',
        }

    return jsonify({
        'code': 200,
        'msg': 'ok',
        'data': {
            'hidden_watermark_result': hidden_result,
            'watermark_result': visible_result,
            'filename': file_obj.filename,
        },
    })


@app.route('/api/detect/image/stream', methods=['POST'])
def api_detect_image_stream():
    """图像检测接口 - SSE 流式返回进度"""
    file_obj = request.files.get('file')
    if not file_obj or not file_obj.filename:
        return jsonify({'code': 400, 'msg': '请上传图片文件'}), 400
    selected_models = parse_detection_models()
    if not selected_models:
        return jsonify({'code': 400, 'msg': '请至少选择一项检测模型'}), 400
    crime_scene = parse_crime_scene()

    ext = os.path.splitext(file_obj.filename)[1].lower()
    quality = validate_detection_image(file_obj)
    if not quality['valid']:
        return jsonify({'code': 400, 'msg': quality['reason'], 'quality': quality}), 400
    save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
    save_path = os.path.join(UPLOAD_FOLDER, save_name)
    file_obj.save(save_path)
    file_obj.seek(0)

    def generate():
        def send_event(event_type, data):
            return f"event: {event_type}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        # ── 相机成像物证（不直接判 AI/篡改）──
        # 与水印核验无数据依赖，后台并行运行。普通无水印图片会在进入
        # NPR 前取回结果；水印短路时不等待这个纯辅助任务。
        camera_forensics = None
        camera_executor = None
        camera_future = None
        if 'ai_generated' in selected_models:
            camera_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix='camera-forensics')
            camera_future = camera_executor.submit(analyze_camera_imaging, save_path)

        def discard_camera_forensics():
            """Do not hold a watermark shortcut open for an auxiliary task."""
            if camera_future and not camera_future.done():
                camera_future.cancel()
            if camera_executor:
                camera_executor.shutdown(wait=False)

        def collect_camera_forensics():
            """Collect the concurrent camera evidence before NPR needs it."""
            if not camera_future:
                return None
            try:
                return camera_future.result()
            except Exception as exc:
                return {'available': False, 'detail': f'相机成像物证分析异常：{exc}'}
            finally:
                if camera_executor:
                    camera_executor.shutdown(wait=False)

        # ── 步骤 1: 水印检测（本地隐式标识 + GLM 可见水印）──
        yield send_event("progress", {
            "step": 1, "total": 7, "percent": 3,
            "stepName": "水印检测",
            "message": "正在检测水印与AI平台签名...",
            "color": "#10b981"
        })

        # 1a. 本地隐式标识检测（GB 45438 TC260:AIGC + 国外 C2PA/元数据，毫秒级）
        hidden_result = None
        try:
            hidden_result = detect_hidden_watermark(save_path)
        except Exception:
            hidden_result = None

        # 发现任何 C2PA 凭证/标记时都运行完整专项，再按三项最高分判定。
        hidden_has_c2pa = c2pa_credential_present(hidden_result)
        if hidden_has_c2pa:
            selected_models = tuple(dict.fromkeys((*selected_models, 'ai_generated', 'deepfake', 'tamper')))
        if ('ai_generated' in selected_models and hidden_result and hidden_result.get('detected')
                and not hidden_has_c2pa):
            # 隐式标识命中 → 直接短路判定 AI 生成
            discard_camera_forensics()
            yield send_event("progress", {
                "step": 1, "total": 7, "percent": 15,
                "stepName": "水印检测",
                "message": f"检测到隐式标识！来源：{hidden_result.get('source', '未知')}",
                "color": "#10b981",
                "detail": hidden_result.get('detail', '')
            })

            TC260_SHORTCUT_SCORE = 0.97
            shortcut_result = {
                'type': 'image',
                'type_name': '图像检测',
                'label': 'ai_generated',
                'label_text': 'AI生成图片',
                'confidence': TC260_SHORTCUT_SCORE,
                'details': {
                    'final_label': 'AI生成图片',
                    'final_verdict': 'AI生成图片',
                    'watermark_detected': True,
                    'watermark_source': hidden_result.get('source'),
                    'watermark_text': '隐式元数据标识（TC260:AIGC / C2PA）',
                    'watermark_confidence': hidden_result.get('confidence'),
                    'verdict_detail': f"检测到国标/元数据隐式标识「{hidden_result.get('source', '未知平台')}」，直接判定为 AI 生成图片。",
                },
                'npr': None,
                'deepfake': None,
                'tamper': None,
                'specialized_models': None,
                'combined': None,
                'deepfake_combined': None,
                'tamper_combined': None,
                'final_comparison': {
                    'final_label': 'ai_generated',
                    'final_verdict': 'AI生成图片',
                    'final_score': TC260_SHORTCUT_SCORE,
                    'confidence': TC260_SHORTCUT_SCORE,
                    'verdict_detail': f"隐式标识检测命中「{hidden_result.get('source', '未知平台')}」，直接判定为 AI 生成图片。",
                    'ai_score': TC260_SHORTCUT_SCORE,
                    'deepfake_score': 0.0,
                    'tamper_score': 0.0,
                    'threshold': 0.50,
                    'is_real': False,
                },
                'filename': file_obj.filename,
                'saved_path': save_path,
                'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'task_id': str(uuid.uuid4())[:8],
                'hidden_watermark_result': hidden_result,
                'shortcut': True,  # 标记为短路结果
                'shortcut_reason': 'tc260',
            }

            yield send_event("progress", {
                "step": 6, "total": 7, "percent": 75,
                "stepName": "内容研判", "message": "正在识别图片内容并匹配犯罪知识库...",
                "color": "#7c3aed"
            })
            shortcut_result['content_analysis'] = run_content_risk_analysis(
                save_path, shortcut_result['label'], shortcut_result['label_text'], shortcut_result['confidence'], crime_scene
            )

            for pct, msg in [(85, "隐式标识与内容研判完成，正在生成检测报告..."), (95, "正在汇总检测结果..."), (100, "检测完成！")]:
                yield send_event("progress", {
                    "step": min(7, int(pct / 100 * 7) + 1),
                    "total": 7, "percent": pct,
                    "stepName": "水印判定" if pct < 100 else "完成",
                    "message": msg,
                    "color": "#10b981" if pct < 100 else "#22c55e",
                    "detail": f"已确认隐式标识来源: {hidden_result.get('source', '未知平台')}"
                })

            yield send_event("result", shortcut_result)
            yield send_event("done", {})
            return

        # 1b. 可见水印分层研判：本地 OCR → Qwen 快速复核 → GLM 仅处理不确定项。
        watermark_result = detect_visible_ai_watermark(save_path)

        if ('ai_generated' in selected_models and watermark_result and watermark_result.get('detected')
                and not hidden_has_c2pa):
            # 检测到水印 → 直接短路判定 AI 生成
            discard_camera_forensics()
            yield send_event("progress", {
                "step": 1, "total": 7, "percent": 15,
                "stepName": "水印检测",
                "message": f"检测到 AI 平台水印！来源：{watermark_result.get('source', '未知')}",
                "color": "#10b981",
                "detail": watermark_result.get('detail', '')
            })

            # 快速跳过后续步骤，直接构建 AI 生成结果
            AI_GENERATED_SCORE = 0.95
            shortcut_result = {
                'type': 'image',
                'type_name': '图像检测',
                'label': 'ai_generated',
                'label_text': 'AI生成图片',
                'confidence': AI_GENERATED_SCORE,
                'details': {
                    'final_label': 'AI生成图片',
                    'final_verdict': 'AI生成图片',
                    'watermark_detected': True,
                    'watermark_source': watermark_result.get('source'),
                    'watermark_position': watermark_result.get('position'),
                    'watermark_text': watermark_result.get('detected_text'),
                    'watermark_confidence': watermark_result.get('confidence'),
                },
                'npr': None,
                'deepfake': None,
                'tamper': None,
                'specialized_models': None,
                'combined': None,
                'deepfake_combined': None,
                'tamper_combined': None,
                'final_comparison': {
                    'final_label': 'ai_generated',
                    'final_verdict': 'AI生成图片',
                    'final_score': AI_GENERATED_SCORE,
                    'confidence': AI_GENERATED_SCORE,
                    'verdict_detail': f"可见水印检测命中「{watermark_result.get('source', '未知平台')}」水印，直接判定为 AI 生成图片。",
                    'ai_score': AI_GENERATED_SCORE,
                    'deepfake_score': 0.0,
                    'tamper_score': 0.0,
                    'threshold': 0.50,
                    'is_real': False,
                },
                'filename': file_obj.filename,
                'saved_path': save_path,
                'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'task_id': str(uuid.uuid4())[:8],
                'watermark_result': watermark_result,
                'shortcut': True,  # 标记为短路结果
            }

            yield send_event("progress", {
                "step": 6, "total": 7, "percent": 75,
                "stepName": "内容研判", "message": "正在识别图片内容并匹配犯罪知识库...",
                "color": "#7c3aed"
            })
            shortcut_result['content_analysis'] = run_content_risk_analysis(
                save_path, shortcut_result['label'], shortcut_result['label_text'], shortcut_result['confidence'], crime_scene
            )

            # 模拟快速进度条跳至完成
            for pct, msg in [(85, "水印与内容研判完成，正在生成检测报告..."), (95, "正在汇总检测结果..."), (100, "检测完成！")]:
                yield send_event("progress", {
                    "step": min(7, int(pct / 100 * 7) + 1),
                    "total": 7, "percent": pct,
                    "stepName": "水印判定" if pct < 100 else "完成",
                    "message": msg,
                    "color": "#10b981" if pct < 100 else "#22c55e",
                    "detail": f"已确认 AI 水印来源: {watermark_result.get('source', '未知平台')}"
                })

            yield send_event("result", shortcut_result)
            yield send_event("done", {})
            return

        # 未检测到水印，继续正常流程
        yield send_event("progress", {
            "step": 1, "total": 7, "percent": 8,
            "stepName": "水印检测",
            "message": "未检测到可见水印，进入深度学习推理...",
            "color": "#10b981",
            "detail": watermark_result.get('detail', '') if watermark_result else ''
        })

        # ── 步骤 2: GPU 推理 ──
        yield send_event("progress", {
            "step": 2, "total": 7, "percent": 10,
            "stepName": "GPU推理",
            "message": "正在上传图片并连接GPU推理服务...",
            "color": "#6366f1"
        })

        data = {'details': {}, 'specialized_models': {}}
        gpu_error = None
        # 仅篡改专项不需要远端 GPU；AI/换脸专项的 GPU 失败也不能中断本地检测链路。
        if {'ai_generated', 'deepfake'} & set(selected_models):
            try:
                resp = requests.post(
                    f"{GPU_API}/detect",
                    files={'file': (file_obj.filename, file_obj.read(), file_obj.content_type)},
                    data={'models': ','.join(selected_models)}, timeout=300
                )
                gpu_result = resp.json()
                if gpu_result.get('code') == 200:
                    data = gpu_result['data']
                else:
                    gpu_error = f"GPU 推理失败: {gpu_result.get('msg', gpu_result)}"
            except requests.exceptions.Timeout:
                gpu_error = "GPU 推理超时"
            except requests.exceptions.ConnectionError:
                gpu_error = "GPU 服务器连接失败"
            except Exception as e:
                gpu_error = f"GPU 推理异常: {str(e)}"

        yield send_event("progress", {
            "step": 2, "total": 7, "percent": 25,
            "stepName": "GPU推理",
            "message": "GPU推理完成（SwinV2 + ViT-B）" if not gpu_error else "GPU推理不可用，继续本地专项检测",
            "color": "#6366f1", "detail": "已获取双模型推理结果" if not gpu_error else gpu_error
        })

        npr_result = None
        if 'ai_generated' in selected_models:
            camera_forensics = collect_camera_forensics()
            yield send_event("progress", {
                "step": 3, "total": 7, "percent": 33, "stepName": "AI全图生成",
                "message": "AI全图生成检测（NPR噪声模式）中...", "color": "#8b5cf6"
            })
            try:
                npr_result = npr_analyze(save_path)
            except Exception as e:
                npr_result = {"score": 0.0, "verdict": "error", "features": {}, "error": str(e)}
            npr_result = fuse_npr_with_physical_evidence(npr_result, camera_forensics)
            yield send_event("progress", {
                "step": 3, "total": 7, "percent": 45, "stepName": "AI全图生成",
                "message": "AI全图生成检测完成", "color": "#8b5cf6",
                "detail": f"频域特征提取完成，NPR得分 {npr_result.get('score', 0):.2f}"
            })

        deepfake_result = None
        if 'deepfake' in selected_models:
            yield send_event("progress", {
                "step": 4, "total": 7, "percent": 50, "stepName": "DeepFake",
                "message": "DeepFake深度伪造检测中...", "color": "#f59e0b"
            })
            try:
                deepfake_result = deepfake_zero_model_analysis(
                    save_path, (data or {}).get('face_glm')
                )
            except Exception as e:
                deepfake_result = {"deepfake_score": 0.0, "verdict": "error", "detail": str(e)}
            yield send_event("progress", {
                "step": 4, "total": 7, "percent": 62, "stepName": "DeepFake",
                "message": "DeepFake检测完成", "color": "#f59e0b",
                "detail": f"ELA分析完成，人脸检测{'已' if deepfake_result.get('face_detected') else '未'}发现"
            })

        tamper_result = None
        if 'tamper' in selected_models:
            yield send_event("progress", {
                "step": 5, "total": 7, "percent": 66, "stepName": "图像篡改",
                "message": "图像篡改检测中...", "color": "#06b6d4"
            })
            try:
                tamper_result = tamper_zero_model_analysis(save_path)
            except Exception as e:
                tamper_result = {"tamper_score": 0.0, "verdict": "error", "detail": str(e)}
            yield send_event("progress", {
                "step": 5, "total": 7, "percent": 80, "stepName": "图像篡改",
                "message": "图像篡改检测完成", "color": "#06b6d4",
                "detail": "复制移动+区块噪声+特征一致性分析完成"
            })

        # ── 步骤 6: 融合判定 ──
        yield send_event("progress", {
            "step": 6, "total": 7, "percent": 85,
            "stepName": "融合判定",
            "message": "融合判定分析中...",
            "color": "#ec4899"
        })
        specialized_models = data.get('specialized_models') or {}
        combined_verdict = None
        if 'ai_generated' in selected_models:
            combined_verdict = compute_combined_verdict(npr_result, specialized_models)
            combined_verdict = apply_c2pa_ai_evidence(combined_verdict, hidden_result)
        deepfake_combined = None
        if 'deepfake' in selected_models:
            deepfake_combined = compute_deepfake_combined_verdict(
                deepfake_result, specialized_models.get('deepfake_detector')
            )
        tamper_combined = compute_tamper_combined_verdict(tamper_result) if 'tamper' in selected_models else None
        final_comparison = compute_final_comparison(
            combined_verdict, deepfake_combined, tamper_combined, selected_models
        )

        completed = []
        if 'ai_generated' in selected_models and npr_result and npr_result.get('verdict') != 'error':
            completed.append('AI全图生成')
        if 'deepfake' in selected_models and deepfake_result and deepfake_result.get('verdict') != 'error':
            completed.append('深度伪造')
        if 'tamper' in selected_models and tamper_result and tamper_result.get('verdict') != 'error':
            completed.append('图像篡改')
        if not completed:
            reasons = [gpu_error] if gpu_error else []
            for item in (npr_result, deepfake_result, tamper_result):
                if item and (item.get('error') or item.get('detail')):
                    reasons.append(str(item.get('error') or item.get('detail')))
            yield send_event("error", {
                "message": "所选检测链路均失败：" + "；".join(reasons[:3] or ['未返回有效结果'])
            })
            return

        content_confirmed = should_analyze_content(final_comparison['final_label'])
        yield send_event("progress", {
            "step": 7, "total": 7, "percent": 96,
            "stepName": "内容研判",
            "message": (
                "已确认 AI 生成，正在识别图片内容并匹配犯罪知识库..."
                if content_confirmed else "未确认 AI 生成，已跳过图片内容识别"
            ),
            "color": "#7c3aed" if content_confirmed else "#6b7280"
        })
        content_analysis = run_content_risk_analysis(
            save_path,
            final_comparison['final_label'],
            final_comparison['final_verdict'],
            final_comparison['final_score'],
            crime_scene,
        )

        yield send_event("progress", {
            "step": 6, "total": 7, "percent": 95,
            "stepName": "融合判定",
            "message": "融合判定完成",
            "color": "#ec4899",
            "detail": f"综合判定: {final_comparison['final_verdict']}"
        })

        # ── 步骤 7: 完成 ──
        result_data = {
            'type': 'image',
            'type_name': '图像检测',
            'label': final_comparison['final_label'],
            'label_text': final_comparison['final_verdict'],
            'confidence': final_comparison['final_score'],
            'details': data['details'],
            'selected_models': list(selected_models),
            'crime_scene': crime_scene,
            'crime_scene_label': CRIME_SCENES[crime_scene]['label'] if crime_scene else None,
            'npr': npr_result,
            'deepfake': deepfake_result,
            'tamper': tamper_result,
            'specialized_models': specialized_models,
            'combined': combined_verdict,
            'deepfake_combined': deepfake_combined,
            'tamper_combined': tamper_combined,
            'final_comparison': final_comparison,
            'filename': file_obj.filename,
            'saved_path': save_path,
            'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'task_id': str(uuid.uuid4())[:8],
            'hidden_watermark_result': hidden_result,
            'watermark_result': watermark_result,
            'c2pa_verification': hidden_result.get('c2pa_verification') if hidden_result else None,
            'content_analysis': content_analysis,
        }

        yield send_event("progress", {
            "step": 7, "total": 7, "percent": 100,
            "stepName": "完成",
            "message": "检测完成！",
            "color": "#22c55e",
            "detail": f"最终判定: {final_comparison['final_verdict']}（置信度 {final_comparison['final_score']:.0%}）"
        })
        yield send_event("result", result_data)
        yield send_event("done", {})

    return Response(
        stream_with_context(generate()),
        mimetype='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'Connection': 'keep-alive',
            'X-Accel-Buffering': 'no'
        }
    )


@app.route('/api/detect/<detect_type>', methods=['POST'])
def api_detect(detect_type):
    """检测接口"""
    if detect_type not in DETECT_TYPES:
        return jsonify({'code': 400, 'msg': '不支持的检测类型'}), 400

    file_obj = request.files.get('file')
    selected_models = parse_detection_models()
    if not selected_models:
        return jsonify({'code': 400, 'msg': '请至少选择一项检测模型'}), 400
    crime_scene = parse_crime_scene()

    # ====== 图像检测：调用 GPU 推理服务 ======
    if detect_type == 'image' and file_obj and file_obj.filename:
        ext = os.path.splitext(file_obj.filename)[1].lower()
        quality = validate_detection_image(file_obj)
        if not quality['valid']:
            return jsonify({'code': 400, 'msg': quality['reason'], 'quality': quality}), 400
        save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
        save_path = os.path.join(UPLOAD_FOLDER, save_name)
        file_obj.save(save_path)
        file_obj.seek(0)  # 重置指针以便重新读取

        camera_forensics = None
        if 'ai_generated' in selected_models:
            try:
                camera_forensics = analyze_camera_imaging(save_path)
            except Exception as exc:
                camera_forensics = {'available': False, 'detail': f'相机成像物证分析异常：{exc}'}

        # ── 统一来源凭证检测：C2PA/TC260/元数据（不跳过深度伪造、篡改专项） ──
        hidden_result = None
        try:
            hidden_result = detect_hidden_watermark(save_path)
        except Exception:
            hidden_result = None
        hidden_has_c2pa = c2pa_credential_present(hidden_result)
        if hidden_has_c2pa:
            selected_models = tuple(dict.fromkeys((*selected_models, 'ai_generated', 'deepfake', 'tamper')))

        # ── 可见水印分层快捷通道（本地 OCR → Qwen → GLM 兜底）──
        watermark_result = None
        try:
            watermark_result = detect_visible_ai_watermark(save_path)
            if ('ai_generated' in selected_models and watermark_result and watermark_result.get('detected')
                    and not hidden_has_c2pa):
                content_analysis = run_content_risk_analysis(
                    save_path, 'ai_generated', 'AI生成图片', 0.95, crime_scene
                )
                return jsonify({
                    'code': 200,
                    'msg': 'success (watermark shortcut)',
                    'data': {
                        'type': 'image',
                        'label': 'ai_generated',
                        'label_text': 'AI生成图片',
                        'confidence': 0.95,
                        'details': {
                            'final_label': 'AI生成图片',
                            'final_verdict': 'AI生成图片',
                            'watermark_detected': True,
                            'watermark_source': watermark_result.get('source'),
                            'watermark_text': watermark_result.get('detected_text'),
                            'watermark_position': watermark_result.get('position'),
                            'watermark_confidence': watermark_result.get('confidence'),
                            'verdict_detail': f"可见水印检测命中「{watermark_result.get('source', '未知平台')}」水印，直接判定为 AI 生成图片。",
                        },
                        'gpu': None,
                        'npr': None,
                        'deepfake': None,
                        'tamper': None,
                        'watermark_result': watermark_result,
                        'content_analysis': content_analysis,
                        'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    }
                })
        except Exception:
            watermark_result = {"detected": False, "detail": "水印检测跳过"}

        try:
            resp = requests.post(
                f"{GPU_API}/detect",
                files={'file': (file_obj.filename, file_obj.read(), file_obj.content_type)},
                data={'models': ','.join(selected_models)},
                timeout=300  # 首次请求需下载模型，给予充足时间
            )
            gpu_result = resp.json()
            if gpu_result.get('code') == 200:
                data = gpu_result['data']
                # 英文标签 → 中文映射（供参考）
                label_map = {
                    'ai_generated': 'AI生成图片',
                    'suspected_ai': '疑似AI生成图片',
                    'real': '真实图片',
                }

                # ── NPR 零模型专项验证（AI全图生成检测）──
                npr_result = None
                if 'ai_generated' in selected_models:
                    try:
                        npr_result = npr_analyze(save_path)
                    except Exception as e:
                        npr_result = {"score": 0.0, "verdict": "error", "features": {}, "error": str(e)}
                    npr_result = fuse_npr_with_physical_evidence(npr_result, camera_forensics)

                # ── DeepFake 零模型专项检测（深度伪造/换脸）──
                # deepfake_detector.py 内置 OpenCV Haar Cascade 人脸检测，可靠
                deepfake_result = None
                if 'deepfake' in selected_models:
                    try:
                        deepfake_result = deepfake_zero_model_analysis(
                            save_path, (data or {}).get('face_glm')
                        )
                    except Exception as e:
                        deepfake_result = {"deepfake_score": 0.0, "verdict": "error", "detail": str(e)}

                # ── 图像篡改 零模型专项检测 ──
                tamper_result = None
                if 'tamper' in selected_models:
                    try:
                        tamper_result = tamper_zero_model_analysis(save_path)
                    except Exception as e:
                        tamper_result = {"tamper_score": 0.0, "verdict": "error", "detail": str(e)}

                # ── NPR + AI-Human 融合判定 ──
                specialized_models = data.get('specialized_models') or {}
                combined_verdict = None
                if 'ai_generated' in selected_models:
                    combined_verdict = compute_combined_verdict(npr_result, specialized_models)
                    combined_verdict = apply_c2pa_ai_evidence(combined_verdict, hidden_result)

                # ── DeepFake 零模型 + 纯模型 融合判定 ──
                deepfake_combined = None
                if 'deepfake' in selected_models:
                    deepfake_combined = compute_deepfake_combined_verdict(
                        deepfake_result, specialized_models.get('deepfake_detector')
                    )

                # ── 图像篡改 零模型 + 纯模型 融合判定 ──
                tamper_combined = compute_tamper_combined_verdict(tamper_result) if 'tamper' in selected_models else None

                # ── 三路专项融合得分比较 → 最终判定 ──
                final_comparison = compute_final_comparison(
                    combined_verdict, deepfake_combined, tamper_combined, selected_models
                )
                content_analysis = run_content_risk_analysis(
                    save_path,
                    final_comparison['final_label'],
                    final_comparison['final_verdict'],
                    final_comparison['final_score'],
                    crime_scene,
                )

                return jsonify({
                    'code': 200,
                    'msg': '检测完成',
                    'data': {
                        'type': detect_type,
                        'type_name': DETECT_TYPES[detect_type]['name'],
                        'label': final_comparison['final_label'],
                        'label_text': final_comparison['final_verdict'],
                        'confidence': final_comparison['final_score'],
                        'details': data['details'],
                        'selected_models': list(selected_models),
                        'crime_scene': crime_scene,
                        'crime_scene_label': CRIME_SCENES[crime_scene]['label'] if crime_scene else None,
                        'npr': npr_result,                    # ← NPR 专项验证（AI 生成）
                        'deepfake': deepfake_result,        # ← DeepFake 零模型专项（深度伪造/换脸）
                        'tamper': tamper_result,            # ← 图像篡改 零模型+纯模型 专项
                        'specialized_models': specialized_models,
                        'combined': combined_verdict,       # ← NPR + AIRealNet 融合结果（AI 生成）
                        'deepfake_combined': deepfake_combined,  # ← DeepFake 零模型+纯模型 融合结果
                        'tamper_combined': tamper_combined,     # ← 图像篡改 零模型+纯模型 融合结果
                        'final_comparison': final_comparison,   # ← 三路比较综合判定
                        'hidden_watermark_result': hidden_result,
                        'watermark_result': watermark_result,
                        'c2pa_verification': hidden_result.get('c2pa_verification') if hidden_result else None,
                        'filename': file_obj.filename,
                        'saved_path': save_path,
                        'detect_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                        'task_id': str(uuid.uuid4())[:8],
                        'content_analysis': content_analysis,
                    }
                })
            else:
                return jsonify({'code': 500, 'msg': f"GPU 推理失败: {gpu_result}"}), 500
        except requests.exceptions.Timeout:
            return jsonify({'code': 500, 'msg': 'GPU 推理超时，请稍后重试'}), 500
        except requests.exceptions.ConnectionError:
            return jsonify({'code': 500, 'msg': 'GPU 服务器连接失败，请检查 GPU 实例是否已开机'}), 500
        except Exception as e:
            return jsonify({'code': 500, 'msg': f'GPU 推理异常: {str(e)}'}), 500

    # ====== 非图像类型（已移除，仅保留兼容返回）======
    return jsonify({
        'code': 400,
        'msg': f'不支持的检测类型: {detect_type}，当前仅支持图像检测',
    }), 400


@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    """提供上传目录中的文件访问（用于批量检测图片预览）"""
    return send_from_directory(UPLOAD_FOLDER, filename)


# ════════════════════════════════════════════════════════════
# 批量检测 API 路由
# ════════════════════════════════════════════════════════════

@app.route('/api/batch/detect', methods=['POST'])
def api_batch_detect():
    """创建批量检测任务"""
    task_name = request.form.get('task_name', '').strip()
    if not task_name:
        return jsonify({'code': 400, 'msg': '请输入任务名称'}), 400
    selected_models = parse_detection_models()
    if not selected_models:
        return jsonify({'code': 400, 'msg': '请至少选择一项检测模型'}), 400
    crime_scene = parse_crime_scene()

    files = request.files.getlist('files')
    if not files or len(files) == 0:
        return jsonify({'code': 400, 'msg': '请选择要检测的图片'}), 400

    image_exts = IMAGE_EXTENSIONS
    image_files = []
    skipped_files = []
    for index, f in enumerate(files):
        ext = os.path.splitext(f.filename)[1].lower()
        if ext in image_exts:
            quality = validate_detection_image(f)
            if quality['valid']:
                image_files.append(f)
            else:
                # 质量校验在文件落盘前执行。不可靠图片不会写入 uploads，
                # 也不会阻断同一批次中其余合格图片的检测。
                skipped_files.append({
                    'index': index,
                    'filename': f.filename,
                    'reason': quality['reason'],
                })

    if len(image_files) == 0:
        if skipped_files:
            return jsonify({
                'code': 400,
                'msg': f"已自动移除 {len(skipped_files)} 张无法可靠检测的图片；请重新选择清晰原图。",
                'skipped_files': skipped_files,
            }), 400
        return jsonify({'code': 400, 'msg': '未找到支持的图片格式（JPG/PNG/BMP/WebP）'}), 400

    try:
        task_id = create_task(task_name, image_files, selected_models, crime_scene)
        message = f'任务已创建，共 {len(image_files)} 张图片'
        if skipped_files:
            message += f'；已自动移除 {len(skipped_files)} 张无法可靠检测的图片'
        return jsonify({
            'code': 200,
            'msg': message,
            'data': {
                'task_id': task_id,
                'task_name': task_name,
                'total_count': len(image_files),
                'skipped_files': skipped_files,
                'selected_models': list(selected_models),
                'crime_scene': crime_scene,
                'crime_scene_label': CRIME_SCENES[crime_scene]['label'] if crime_scene else None,
            }
        })
    except Exception as e:
        return jsonify({'code': 500, 'msg': f'创建任务失败: {str(e)}'}), 500


@app.route('/api/batch/tasks', methods=['GET'])
def api_batch_tasks():
    """获取批量检测任务列表（支持搜索、筛选、分页）"""
    try:
        search = request.args.get('search')
        status = request.args.get('status')
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)
        result = get_all_tasks(search, status, page, page_size)
        return jsonify({'code': 200, 'msg': 'ok', 'data': result})
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)}), 500


@app.route('/api/batch/task/<task_id>', methods=['GET', 'DELETE'])
def api_batch_task_detail(task_id):
    """获取任务概要或删除任务"""
    if request.method == 'DELETE':
        try:
            delete_task(task_id)
            return jsonify({'code': 200, 'msg': '任务已删除'})
        except Exception as e:
            return jsonify({'code': 500, 'msg': str(e)}), 500

    # GET — 任务概要
    try:
        task = get_task_summary(task_id)
        if task is None:
            return jsonify({'code': 404, 'msg': '任务不存在'}), 404
        return jsonify({'code': 200, 'msg': 'ok', 'data': task})
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)}), 500


@app.route('/api/batch/task/<task_id>/images', methods=['GET'])
def api_batch_task_images(task_id):
    """获取任务内的图片结果列表（支持分页和筛选）"""
    try:
        label_filter = request.args.get('filter')
        page = request.args.get('page', 1, type=int)
        page_size = request.args.get('page_size', 20, type=int)

        if label_filter and label_filter not in ('ai_generated', 'suspected_ai', 'real'):
            return jsonify({'code': 400, 'msg': '不支持的筛选条件'}), 400

        result = get_task_images(task_id, label_filter, page, page_size)
        return jsonify({'code': 200, 'msg': 'ok', 'data': result})
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)}), 500


@app.route('/api/batch/task/<task_id>/image/<int:image_id>', methods=['GET'])
def api_batch_image_detail(task_id, image_id):
    """获取单张图片的完整检测结果"""
    try:
        img = get_image_detail(task_id, image_id)
        if img is None:
            return jsonify({'code': 404, 'msg': '图片不存在'}), 404
        return jsonify({'code': 200, 'msg': 'ok', 'data': img})
    except Exception as e:
        return jsonify({'code': 500, 'msg': str(e)}), 500


if __name__ == '__main__':
    # host='0.0.0.0' 支持外网访问；Windows Server 2016 可用
    app.run(host='0.0.0.0', port=5000, debug=False)
