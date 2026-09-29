"""上传图片的大小与最低可检测质量校验。

该校验只拦截低分辨率或明显失焦的图片，避免把“检测不可靠”误报成“真实图片”。
它不是 AI 生成检测，也不用于判定图片真伪。
"""

from __future__ import annotations

import cv2
import numpy as np


MIN_SHORT_EDGE = 300
MIN_LAPLACIAN_VARIANCE = 25.0
MIN_IMAGE_BYTES = 30 * 1024
MAX_IMAGE_BYTES = 20 * 1024 * 1024


def validate_image_bytes(image_bytes: bytes) -> dict:
    """返回可安全展示给用户的质量校验结果。"""
    if not image_bytes:
        return {"valid": False, "reason": "图片文件为空，请重新选择。"}
    if len(image_bytes) < MIN_IMAGE_BYTES:
        return {"valid": False, "reason": "图片文件大小不能小于 30 KB，请上传清晰原图。", "file_size": len(image_bytes)}
    if len(image_bytes) > MAX_IMAGE_BYTES:
        return {"valid": False, "reason": "图片文件过大（超过 20 MB），请压缩后重新选择。", "file_size": len(image_bytes)}

    image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return {"valid": False, "reason": "无法读取图片，请上传完整的 JPG、PNG、BMP 或 WebP 文件。"}

    height, width = image.shape[:2]
    short_edge = min(width, height)
    if short_edge < MIN_SHORT_EDGE:
        return {
            "valid": False,
            "reason": f"上传图像最小尺寸不小于 {MIN_SHORT_EDGE}px（当前 {width} × {height}）。",
            "width": width,
            "height": height,
            "short_edge": short_edge,
        }

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if sharpness < MIN_LAPLACIAN_VARIANCE:
        return {
            "valid": False,
            "reason": "图片过于模糊，无法可靠检测。请上传清晰原图，避免失焦、严重压缩或二次截图。",
            "width": width,
            "height": height,
            "short_edge": short_edge,
            "sharpness": round(sharpness, 2),
        }

    return {
        "valid": True,
        "width": width,
        "height": height,
        "short_edge": short_edge,
        "sharpness": round(sharpness, 2),
    }


def validate_uploaded_file(file_obj) -> dict:
    """校验 Flask FileStorage，并将读取指针复位以供后续保存/推理。"""
    image_bytes = file_obj.read(MAX_IMAGE_BYTES + 1)
    file_obj.seek(0)
    return validate_image_bytes(image_bytes)
