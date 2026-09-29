"""Fast local visible-AI-watermark screening pipeline.

Only local OCR/signature checks run here.  A negative result never calls an
external vision model and never blocks the subsequent forgery detectors.
"""
from __future__ import annotations

from visible_watermark_detector import detect_visible_watermark_fast


def _failure(detail: str) -> dict:
    return {
        'detected': False, 'source': None, 'detected_text': None, 'position': None,
        'confidence': 0.0, 'method': 'watermark_pipeline', 'detail': detail,
        'review_chain': [],
    }


def detect_visible_ai_watermark(image_path: str) -> dict:
    """Run the local OCR/signature check only; no external VLM request."""
    try:
        local = detect_visible_watermark_fast(image_path)
    except Exception as exc:
        local = _failure(f'本地 OCR 快筛异常：{exc}')
        local['method'] = 'local_ocr_fast'
    review_chain = [{
        'stage': 'local_ocr', 'method': local.get('method'), 'detected': bool(local.get('detected')),
        'confidence': local.get('confidence', 0.0), 'detail': local.get('detail'),
    }]
    return {
        **local,
        'review_chain': review_chain,
        'review_status': 'confirmed' if local.get('detected') else 'local_clear',
        'detail': local.get('detail') or '本地 OCR 快筛完成，未发现高置信 AI 平台水印',
    }
