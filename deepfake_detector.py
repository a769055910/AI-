"""
深度伪造专项检测 — 零模型方案（纯信号级，无需训练）

路径：ELA 误差分析（换脸/篡改图的压缩历史与原始图不一致）
     → 仅负责检测「深度伪造/换脸」，不与 AI生成/图像篡改混合

人脸检测：使用 GLM-4.5V 多模态视觉模型（SiliconFlow API）
依赖：numpy + opencv-python + requests（项目已有）
"""
import os
import cv2
import numpy as np
from siliconflow_client import detect_face_glm


# ══════════════════════════════════════════════════════════
# 模块一：ELA 误差等级分析
# ══════════════════════════════════════════════════════════

def ela_analyze(image_path: str) -> dict:
    """
    Error Level Analysis — JPEG 压缩误差分析

    原理：
      真实照片 → 各区域压缩历史一致 → ELA 误差均匀
      换脸/篡改图 → 不同来源区域压缩历史不同 → 局部 ELA 异常

    返回:
      {
        "score": float,         # 0~1，越高越可疑
        "anomaly_ratio": float, # 异常像素占比
        "mean_error": float,    # 平均误差
        "std_error": float,     # 误差标准差
        "verdict": str
      }
    """
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        return {"score": 0.0, "anomaly_ratio": 0.0, "mean_error": 0.0, "std_error": 0.0, "verdict": "error"}

    # 以 quality=90 重新编码 JPEG，读回得到参考图
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, 90]
    _, jpeg_buf = cv2.imencode('.jpg', img, encode_params)
    recompressed = cv2.imdecode(jpeg_buf, cv2.IMREAD_COLOR)

    # 计算逐像素差值的幅值（取三通道最大差异）
    diff = cv2.absdiff(img.astype(np.float32), recompressed.astype(np.float32))
    diff_magnitude = np.max(diff, axis=2)  # shape (H, W)

    mean_err = float(np.mean(diff_magnitude))
    std_err = float(np.std(diff_magnitude))

    # 异常阈值：超过 mean + 2*std 的像素视为 ELA 异常
    anomaly_threshold = mean_err + 2.0 * std_err
    anomaly_mask = diff_magnitude > anomaly_threshold
    anomaly_ratio = float(np.sum(anomaly_mask)) / (img.shape[0] * img.shape[1])

    # ── 空间聚集度分析 ──
    # 如果异常像素呈大块聚集（而非随机散布），更可疑
    kernel = np.ones((5, 5), np.uint8)
    anomaly_dilated = cv2.dilate(anomaly_mask.astype(np.uint8), kernel, iterations=2)
    cluster_area = float(np.sum(anomaly_dilated)) / (img.shape[0] * img.shape[1])

    # ── 综合评分 ──
    # 异常比例高 → 可疑；异常聚集度高 → 更可疑
    score_anomaly = min(anomaly_ratio * 8.0, 1.0)      # anomaly_ratio ~12.5% → 满分
    score_cluster = min(cluster_area * 5.0, 1.0)         # cluster ~20% → 满分
    score_mean   = min(mean_err / 15.0, 1.0)             # mean_err ~15 → 满分

    score = round(float(np.clip(
        score_anomaly * 0.40 + score_cluster * 0.35 + score_mean * 0.25,
        0.0, 1.0
    )), 4)

    if score > 0.60:
        verdict = "fake_likely"
    elif score <= 0.60:
        verdict = "real_likely"
    else:
        verdict = "uncertain"

    return {
        "score": score,
        "anomaly_ratio": round(anomaly_ratio, 6),
        "cluster_ratio": round(cluster_area, 6),
        "mean_error": round(mean_err, 4),
        "std_error": round(std_err, 4),
        "verdict": verdict
    }


def _clip01(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0))


def _load_bgr_image(image_path: str):
    """以支持中文路径的方式读取图像。"""
    data = np.fromfile(image_path, dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _detect_face_boxes(image: np.ndarray) -> list:
    """使用 OpenCV Haar Cascade 定位正脸；仅用于确定局部取证区域，不承担伪造判定。"""
    if image is None or image.size == 0:
        return []
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    height, width = gray.shape[:2]
    min_side = max(56, min(height, width) // 10)
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.10, minNeighbors=5,
        minSize=(min_side, min_side), flags=cv2.CASCADE_SCALE_IMAGE
    )
    return sorted([tuple(map(int, face)) for face in faces], key=lambda item: item[2] * item[3], reverse=True)[:5]


def _ellipse_mask(shape: tuple, box: tuple, scale: float) -> np.ndarray:
    """根据正脸框构造椭圆近似轮廓，用于取脸部轮廓内外窄带。"""
    height, width = shape[:2]
    x, y, face_w, face_h = box
    center = (int(x + face_w * 0.50), int(y + face_h * 0.53))
    axes = (max(2, int(face_w * 0.46 * scale)), max(2, int(face_h * 0.50 * scale)))
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    return mask


def _safe_mean(values: np.ndarray, mask: np.ndarray) -> float:
    selected = values[mask > 0]
    return float(np.mean(selected)) if selected.size else 0.0


def face_boundary_fusion_analyze(image_path: str) -> dict:
    """人脸边界融合一致性：比较脸部轮廓内外窄带的接缝、清晰度、颜色过渡和轮廓连续性。

    该方法是确定性图像取证，不使用深度伪造分类模型；Haar Cascade 只用于定位可分析人脸。
    """
    image = _load_bgr_image(image_path)
    if image is None:
        return {"available": False, "score": 0.0, "verdict": "error", "detail": "图像读取失败", "face_detected": False}

    faces = _detect_face_boxes(image)
    if not faces:
        return {
            "available": False, "score": 0.0, "verdict": "insufficient_evidence",
            "detail": "未定位到可用于边界取证的正脸", "face_detected": False,
            "face_count": 0, "analyzed_face_count": 0,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    gradient = cv2.magnitude(grad_x, grad_y)
    laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F, ksize=3))
    lab_gradients = []
    for channel in cv2.split(lab):
        channel_x = cv2.Sobel(channel, cv2.CV_32F, 1, 0, ksize=3)
        channel_y = cv2.Sobel(channel, cv2.CV_32F, 0, 1, ksize=3)
        lab_gradients.append(cv2.magnitude(channel_x, channel_y))
    lab_gradient = np.mean(np.stack(lab_gradients, axis=0), axis=0)
    canny = cv2.Canny(gray, 70, 150)

    face_scores = []
    for box in faces:
        x, y, face_w, face_h = box
        min_side = min(face_w, face_h)
        # 极小人脸的边缘指标不稳定，直接跳过。
        if min_side < 72:
            continue

        base = _ellipse_mask(gray.shape, box, 1.0)
        inner = _ellipse_mask(gray.shape, box, 0.88)
        outer = _ellipse_mask(gray.shape, box, 1.14)
        inner_band = cv2.subtract(base, inner)
        outer_band = cv2.subtract(outer, base)
        ring = cv2.subtract(cv2.dilate(base, np.ones((3, 3), np.uint8), iterations=1), cv2.erode(base, np.ones((3, 3), np.uint8), iterations=1))

        # 人脸尺寸和局部清晰度共同决定该脸是否足以形成可靠证据。
        size_quality = _clip01((min_side - 72.0) / 120.0)
        focus_quality = _clip01((_safe_mean(laplacian, base) - 8.0) / 70.0)
        quality = 0.55 * size_quality + 0.45 * focus_quality
        if quality < 0.30:
            continue

        edge_ref = (_safe_mean(gradient, inner_band) + _safe_mean(gradient, outer_band)) / 2.0 + 1e-6
        edge_ratio = _safe_mean(gradient, ring) / edge_ref
        seam_strength = _clip01((edge_ratio - 1.20) / 1.35)

        inner_sharpness = _safe_mean(laplacian, inner_band) + 1e-6
        outer_sharpness = _safe_mean(laplacian, outer_band) + 1e-6
        sharpness_gap = _clip01((abs(np.log(inner_sharpness / outer_sharpness)) - 0.18) / 1.25)

        color_ref = (_safe_mean(lab_gradient, inner_band) + _safe_mean(lab_gradient, outer_band)) / 2.0 + 1e-6
        color_ratio = _safe_mean(lab_gradient, ring) / color_ref
        color_transition = _clip01((color_ratio - 1.15) / 1.30)

        contour_ref = (_safe_mean(canny.astype(np.float32), inner_band) + _safe_mean(canny.astype(np.float32), outer_band)) / 2.0 + 1e-6
        contour_ratio = _safe_mean(canny.astype(np.float32), ring) / contour_ref
        contour_discontinuity = _clip01((contour_ratio - 1.15) / 1.40)

        raw_score = (
            0.35 * seam_strength + 0.25 * sharpness_gap +
            0.25 * color_transition + 0.15 * contour_discontinuity
        )
        # 低质量图保守降权，防止模糊、压缩造成的边界误报。
        score = _clip01(raw_score * (0.55 + 0.45 * quality))
        face_scores.append({
            "score": score, "quality": quality,
            "edge_seam": seam_strength, "sharpness_gap": sharpness_gap,
            "color_transition": color_transition, "contour_discontinuity": contour_discontinuity,
        })

    if not face_scores:
        return {
            "available": False, "score": 0.0, "verdict": "insufficient_evidence",
            "detail": "检测到人脸但尺寸、清晰度或姿态不足，无法形成可靠边界证据",
            "face_detected": True, "face_count": len(faces), "analyzed_face_count": 0,
        }

    # 多人脸场景下保留风险最高且质量合格的人脸，避免低风险背景脸稀释告警。
    result = max(face_scores, key=lambda item: item["score"])
    score = round(result["score"], 4)
    if score > 0.60:
        verdict = "fake_likely"
        detail = "人脸轮廓内外存在较明显的融合过渡异常"
    elif score <= 0.60:
        verdict = "real_likely"
        detail = "未发现明显的人脸边界融合不连续"
    else:
        verdict = "uncertain"
        detail = "人脸边界存在轻度不连续，建议与 ELA 和纯模型结果交叉研判"

    return {
        "available": True, "score": score, "verdict": verdict, "detail": detail,
        "face_detected": True, "face_count": len(faces), "analyzed_face_count": len(face_scores),
        "face_quality": round(result["quality"], 4),
        "edge_seam": round(result["edge_seam"], 4),
        "sharpness_gap": round(result["sharpness_gap"], 4),
        "color_transition": round(result["color_transition"], 4),
        "contour_discontinuity": round(result["contour_discontinuity"], 4),
    }


# ══════════════════════════════════════════════════════════
# 对外统一接口
# ══════════════════════════════════════════════════════════

def deepfake_zero_model_analysis(image_path: str, face_result_override: dict | None = None) -> dict:
    """
    深度伪造零模型检测统一入口（纯 ELA + 人脸检测）

    返回:
      {
        "deepfake_score": float,   # 综合深度伪造概率 (0~1)
        "verdict": str,            # fake_likely / real_likely / uncertain / error
        "ela": { ... },            # ELA 子结果
        "detail": str,             # 可读解释
        "face_detected": bool
      }
    """
    # ── 人脸检测：优先 GLM，服务异常时回退到本地 OpenCV。──
    # GLM 明确返回“无人脸”时尊重其结论；只有请求/解析失败才走回退，
    # 避免因云端服务短暂不可用把清晰人脸误写为“未检测到”。
    face_detected = False
    face_count = 0
    face_detail = ""
    face_method = "glm_vision"
    face_fallback_used = False
    face_detection_error = None
    try:
        # 单图/批量主链路会从现有 GPU 隧道带回远端 GLM 结果，避免受本机
        # 出站网络策略影响；独立调用时仍兼容直接请求 GLM。
        face_result = face_result_override if face_result_override is not None else detect_face_glm(image_path)
        face_detected = face_result.get("face_detected", False)
        face_count = face_result.get("face_count", 0)
        face_detail = face_result.get("detail", "")
        face_detection_error = face_result.get("error")
        if face_result_override is not None:
            face_method = face_result.get("method", "gpu_glm_vision")
    except Exception as exc:
        face_result = None
        face_detection_error = str(exc)

    if face_detection_error:
        face_fallback_used = True
        face_method = "opencv_haar_fallback"
        try:
            image = _load_bgr_image(image_path)
            face_boxes = _detect_face_boxes(image)
            face_detected = bool(face_boxes)
            face_count = len(face_boxes)
            if face_detected:
                face_detail = (
                    f"GLM-4.5V 人脸检测服务异常，已由本地 OpenCV 定位到 {face_count} 张人脸"
                )
            else:
                face_detail = "GLM-4.5V 人脸检测服务异常；本地 OpenCV 也未定位到可分析正脸"
        except Exception as exc:
            face_detected = False
            face_count = 0
            face_detail = "GLM-4.5V 人脸检测服务异常；本地 OpenCV 回退检测也失败"
            face_detection_error = f"{face_detection_error}; OpenCV: {exc}"

    # ── ELA 分析 ──
    ela_result = None
    try:
        ela_result = ela_analyze(image_path)
    except Exception as e:
        ela_result = {"score": 0.0, "verdict": "error", "error": str(e)}

    ela_score = ela_result.get("score", 0.0)

    # ── 判定：零模型仅保留 ELA 压缩误差分析。 ──
    combined_score = round(ela_score, 4)

    if combined_score > 0.60:
        verdict = "fake_likely"
    elif combined_score <= 0.60:
        verdict = "real_likely"
    else:
        verdict = "uncertain"

    if ela_result.get("verdict") == "fake_likely":
        detail = "ELA 压缩误差分布存在异常，疑似换脸/深度伪造"
    elif face_detected:
        detector_name = "本地 OpenCV 回退检测" if face_fallback_used else "GLM-4.5V"
        detail = f"{detector_name}检测到 {face_count} 张人脸，ELA 未发现明显异常"
    else:
        detail = f"GLM-4.5V 未检测到人脸区域: {face_detail}" if face_detail else "GLM-4.5V 未检测到人脸区域，ELA 未发现异常"

    return {
        "deepfake_score": combined_score,
        "verdict": verdict,
        "ela": ela_result,
        "detail": detail,
        "face_detected": face_detected,
        "face_count": face_count,
        "face_detection_method": face_method,
        "face_fallback_used": face_fallback_used,
        "face_detection_error": face_detection_error,
    }

