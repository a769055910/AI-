# -*- coding: utf-8 -*-
"""单图相机成像物证分析。

本模块不把“缺少相机痕迹”等同于 AI：截图、社交平台重编码、夜景降噪和后期
都可能削弱这些痕迹。它输出的是可解释的来源物证，供 AI 全图生成、深度换脸和
图像篡改三条链路共同参考。

严格 PRNU 设备指纹需要同一设备的多张参考照片；本模块仅实现不依赖参考库的
单图层：元数据采集证据、CFA 去马赛克周期性、噪声-亮度关系、局部噪声一致性。
"""

import os

import cv2
import numpy as np
from PIL import Image, ExifTags


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_bgr(image_path):
    raw = np.fromfile(image_path, dtype=np.uint8)
    return cv2.imdecode(raw, cv2.IMREAD_COLOR)


def _exif_camera_evidence(image_path):
    """提取相机厂商/型号/镜头等采集声明；EXIF 可伪造，故仅为辅助物证。"""
    result = {'make': None, 'model': None, 'software': None, 'has_camera_tags': False}
    try:
        with Image.open(image_path) as image:
            exif = image.getexif() or {}
            values = {ExifTags.TAGS.get(key, str(key)): value for key, value in exif.items()}
        result['make'] = str(values.get('Make') or '').strip() or None
        result['model'] = str(values.get('Model') or '').strip() or None
        result['software'] = str(values.get('Software') or '').strip() or None
        result['has_camera_tags'] = bool(result['make'] and result['model'])
    except Exception:
        pass
    return result


def _noise_statistics(gray):
    """低纹理区域的噪声残差、亮度-噪声关系和空间一致性。"""
    grayf = gray.astype(np.float32) / 255.0
    smooth = cv2.GaussianBlur(grayf, (0, 0), 1.1)
    residual = grayf - smooth
    gradient = cv2.magnitude(cv2.Sobel(grayf, cv2.CV_32F, 1, 0), cv2.Sobel(grayf, cv2.CV_32F, 0, 1))
    low_texture = gradient < np.percentile(gradient, 45)

    # 以 32x32 小块统计，剔除纹理块，减少主体纹理被误认为传感器噪声。
    samples = []
    h, w = gray.shape
    for y in range(0, max(1, h - 31), 32):
        for x in range(0, max(1, w - 31), 32):
            mask = low_texture[y:y + 32, x:x + 32]
            if mask.size < 256 or np.mean(mask) < 0.55:
                continue
            patch = residual[y:y + 32, x:x + 32][mask]
            bright = grayf[y:y + 32, x:x + 32][mask]
            if patch.size >= 120:
                samples.append((float(np.mean(bright)), float(np.var(patch))))

    if len(samples) < 4:
        return {'available': False, 'sample_count': len(samples), 'noise_sigma': 0.0,
                'noise_consistency': 0.0, 'brightness_noise_correlation': 0.0}

    brightness = np.array([item[0] for item in samples])
    variance = np.array([item[1] for item in samples])
    noise_sigma = float(np.sqrt(np.median(variance)))
    coefficient_of_variation = float(np.std(variance) / (np.mean(variance) + 1e-10))
    consistency = float(np.clip(1.0 - coefficient_of_variation / 1.5, 0.0, 1.0))
    corr = float(np.corrcoef(brightness, variance)[0, 1]) if np.std(brightness) > 1e-6 else 0.0
    # 相机光子噪声通常随亮度存在正相关；这里仅作为弱物证，负相关不计入惩罚。
    positive_corr = float(np.clip(corr, 0.0, 1.0))
    return {
        'available': True,
        'sample_count': len(samples),
        'noise_sigma': round(noise_sigma, 5),
        'noise_consistency': round(consistency, 4),
        'brightness_noise_correlation': round(corr, 4),
        'positive_brightness_noise_relation': round(positive_corr, 4),
    }


def _cfa_periodicity(bgr):
    """估计 Bayer CFA 去马赛克残差的 2×2 周期线索。

    对彩色平滑区域的高频残差按像素奇偶位置分组。真实相机 ISP 常保留轻微的
    周期性，但压缩、缩放和去噪都会削弱它，所以该项权重受限。
    """
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    residual = rgb - cv2.GaussianBlur(rgb, (0, 0), 0.8)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad = cv2.magnitude(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    mask = grad < np.percentile(grad, 40)
    energies = []
    for py in (0, 1):
        for px in (0, 1):
            part_mask = mask[py::2, px::2]
            part = residual[py::2, px::2, :]
            if np.count_nonzero(part_mask) < 64:
                energies.append(0.0)
            else:
                energies.append(float(np.mean(np.square(part[part_mask]))))
    mean_energy = float(np.mean(energies))
    periodicity = float(np.std(energies) / (mean_energy + 1e-10))
    # 周期性太强也可能是压缩/纹理伪影，使用饱和函数限制为弱证据。
    score = float(np.clip(periodicity / 0.35, 0.0, 1.0))
    return {
        'available': mean_energy > 1e-10,
        'parity_energies': [round(value, 8) for value in energies],
        'periodicity': round(periodicity, 4),
        'cfa_trace_score': round(score, 4),
    }


def _region_signature(bgr, mask):
    """区域级物理签名：残差噪声能量与 CFA 奇偶周期性。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    residual = gray - cv2.GaussianBlur(gray, (0, 0), 1.0)
    valid = mask.astype(bool)
    if np.count_nonzero(valid) < 256:
        return None
    noise_energy = float(np.sqrt(np.mean(np.square(residual[valid]))))

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    color_residual = rgb - cv2.GaussianBlur(rgb, (0, 0), 0.8)
    parity = []
    for py in (0, 1):
        for px in (0, 1):
            sub_mask = valid[py::2, px::2]
            sub = color_residual[py::2, px::2]
            if np.count_nonzero(sub_mask) >= 64:
                parity.append(float(np.mean(np.square(sub[sub_mask]))))
    cfa_periodicity = float(np.std(parity) / (np.mean(parity) + 1e-10)) if len(parity) == 4 else 0.0
    return {'noise_energy': noise_energy, 'cfa_periodicity': cfa_periodicity}


def _detect_face_boxes(gray):
    """本地 Haar 仅用于确定局部物证区域；不参与人脸身份或换脸判定。"""
    if not hasattr(cv2, 'CascadeClassifier') or not hasattr(cv2, 'data'):
        return []
    cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        return []
    boxes = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(42, 42))
    return [tuple(map(int, box)) for box in boxes]


def _deepfake_camera_consistency(bgr):
    """换脸专项：比较人脸与其邻域是否服从同一噪声/CFA 成像链路。"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    boxes = _detect_face_boxes(gray)
    if not boxes:
        return {
            'available': False, 'face_regions': 0, 'consistency_score': 0.0,
            'detail': '未定位到可用于相机物证比较的人脸区域。',
        }

    h, w = gray.shape
    scores, comparisons = [], []
    for x, y, fw, fh in boxes[:5]:
        face_mask = np.zeros((h, w), dtype=np.uint8)
        face_mask[y:y + fh, x:x + fw] = 1
        # 取扩大矩形减去脸框，作为相同画面的背景近邻。
        pad_x, pad_y = max(12, fw // 3), max(12, fh // 3)
        x0, y0 = max(0, x - pad_x), max(0, y - pad_y)
        x1, y1 = min(w, x + fw + pad_x), min(h, y + fh + pad_y)
        ring_mask = np.zeros((h, w), dtype=np.uint8)
        ring_mask[y0:y1, x0:x1] = 1
        ring_mask[face_mask.astype(bool)] = 0
        face_sig = _region_signature(bgr, face_mask)
        ring_sig = _region_signature(bgr, ring_mask)
        if not face_sig or not ring_sig:
            continue
        noise_delta = abs(np.log((face_sig['noise_energy'] + 1e-7) / (ring_sig['noise_energy'] + 1e-7)))
        cfa_delta = abs(face_sig['cfa_periodicity'] - ring_sig['cfa_periodicity'])
        # 两项均是辅助物证，限制单脸最大贡献，避免美颜/景深造成过高分。
        score = float(np.clip(0.65 * noise_delta / 0.75 + 0.35 * cfa_delta / 0.35, 0.0, 1.0))
        scores.append(score)
        comparisons.append({
            'noise_delta': round(noise_delta, 4),
            'cfa_delta': round(cfa_delta, 4),
            'score': round(score, 4),
        })

    if not scores:
        return {
            'available': False, 'face_regions': len(boxes), 'consistency_score': 0.0,
            'detail': '人脸区域过小或背景不足，无法稳定比较相机物证。',
        }
    score = float(np.mean(scores))
    return {
        'available': True,
        'face_regions': len(scores),
        'consistency_score': round(score, 4),
        'comparisons': comparisons,
        'detail': ('人脸与近邻背景的成像痕迹存在差异，作为换脸辅助线索。'
                   if score >= 0.55 else '人脸与近邻背景的成像痕迹未见明显差异。'),
    }


def _tamper_camera_discontinuity(bgr):
    """篡改专项：以网格比较任意局部的噪声/CFA 签名，查找拼接与替换边界。"""
    h, w = bgr.shape[:2]
    tile = max(48, min(128, min(h, w) // 5))
    signatures = []
    for y in range(0, h - tile + 1, tile):
        for x in range(0, w - tile + 1, tile):
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[y:y + tile, x:x + tile] = 1
            sig = _region_signature(bgr, mask)
            if sig:
                signatures.append(sig)
    if len(signatures) < 4:
        return {
            'available': False, 'tile_count': len(signatures), 'discontinuity_score': 0.0,
            'detail': '可用局部块不足，无法进行相机物证不连续性分析。',
        }

    noise = np.array([item['noise_energy'] for item in signatures])
    cfa = np.array([item['cfa_periodicity'] for item in signatures])
    def robust_outlier_ratio(values):
        median = np.median(values)
        mad = np.median(np.abs(values - median)) + 1e-10
        z = 0.6745 * (values - median) / mad
        return float(np.mean(np.abs(z) > 2.7))
    noise_outliers = robust_outlier_ratio(noise)
    cfa_outliers = robust_outlier_ratio(cfa)
    dispersion = float(np.clip(np.std(noise) / (np.mean(noise) + 1e-10), 0.0, 2.0))
    score = float(np.clip(0.45 * min(noise_outliers * 3, 1) +
                          0.30 * min(cfa_outliers * 3, 1) +
                          0.25 * min(dispersion / 0.8, 1), 0.0, 1.0))
    return {
        'available': True,
        'tile_count': len(signatures),
        'discontinuity_score': round(score, 4),
        'noise_outlier_ratio': round(noise_outliers, 4),
        'cfa_outlier_ratio': round(cfa_outliers, 4),
        'detail': ('局部相机成像痕迹不连续，作为拼接/局部替换的辅助线索。'
                   if score >= 0.55 else '局部相机成像痕迹整体连续。'),
    }


def analyze_camera_imaging(image_path, include_local=False):
    """返回单图相机成像物证，不产出 AI/篡改二元结论。"""
    output = {
        'available': False,
        'camera_trace_score': 0.0,
        'acquisition_risk_score': 0.5,
        'verdict': '无法分析',
        'detail': '无法读取图像的相机成像痕迹。',
        'exif': _exif_camera_evidence(image_path),
        'noise': None,
        'cfa': None,
        'limitations': '严格 PRNU 设备指纹需要同一设备的参考照片；单图结果仅作辅助物证。',
    }
    try:
        bgr = _load_bgr(image_path)
        if bgr is None or min(bgr.shape[:2]) < 48:
            output['detail'] = '图片尺寸过小，无法稳定分析相机成像痕迹。'
            return output

        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        exif = output['exif']
        noise = _noise_statistics(gray)
        cfa = _cfa_periodicity(bgr)

        metadata_score = 1.0 if exif.get('has_camera_tags') else 0.0
        noise_score = 0.0
        if noise.get('available'):
            # 噪声存在、局部一致性和亮度关系共同形成弱-中等物证。
            sigma_score = float(np.clip(noise.get('noise_sigma', 0.0) / 0.012, 0.0, 1.0))
            noise_score = 0.45 * sigma_score + 0.40 * noise.get('noise_consistency', 0.0) + \
                          0.15 * noise.get('positive_brightness_noise_relation', 0.0)
        cfa_score = cfa.get('cfa_trace_score', 0.0) if cfa.get('available') else 0.0

        # EXIF 容易伪造，权重刻意低；压缩后没有 CFA/噪声也不会直接形成高风险。
        trace_score = float(np.clip(0.20 * metadata_score + 0.55 * noise_score + 0.25 * cfa_score, 0.0, 1.0))
        if trace_score >= 0.62:
            verdict = '存在较稳定的相机采集痕迹'
            detail = '检测到相机采集元数据或较稳定的噪声/CFA 去马赛克线索，可作为真实性辅助物证。'
        elif trace_score >= 0.38:
            verdict = '相机采集痕迹有限'
            detail = '只检测到部分相机成像线索；压缩、截图、降噪或后期处理均可能造成该结果。'
        else:
            verdict = '未发现稳定的相机采集痕迹'
            detail = '未发现足够的单图相机成像物证；这不等同于 AI 生成或图像篡改。'

        # AI 全图生成只使用全局采集痕迹的“缺失程度”作为低权重解释性物证。
        ai_evidence = {
            'available': True,
            'global_trace_weakness': round(1.0 - trace_score, 4),
            'detail': ('整体相机采集痕迹偏弱，仅作为 AI 全图生成的辅助物证。'
                       if trace_score < 0.38 else '整体存在一定相机采集痕迹，仅作为 AI 全图生成的反向辅助物证。'),
        }

        output.update({
            'available': True,
            'camera_trace_score': round(trace_score, 4),
            'acquisition_risk_score': round(1.0 - trace_score, 4),
            'verdict': verdict,
            'detail': detail,
            'exif': exif,
            'noise': noise,
            'cfa': cfa,
            'ai_generation_evidence': ai_evidence,
        })
        if include_local:
            output['deepfake_evidence'] = _deepfake_camera_consistency(bgr)
            output['tamper_evidence'] = _tamper_camera_discontinuity(bgr)
        return output
    except Exception as exc:
        output['detail'] = f'相机成像痕迹分析异常：{str(exc)[:180]}'
        return output
