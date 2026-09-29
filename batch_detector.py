# -*- coding: utf-8 -*-
"""
批量图像检测核心模块
- 创建任务、异步并行检测、结果查询
- 使用 ThreadPoolExecutor 并发处理多张图片
"""
import os
import uuid
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

import requests

from db import get_db
from npr_detector import npr_analyze
from deepfake_detector import deepfake_zero_model_analysis
from tamper_detector import tamper_zero_model_analysis
from camera_forensics import analyze_camera_imaging
from watermark_pipeline import detect_visible_ai_watermark
from hidden_watermark_detector import detect_hidden_watermark
from content_risk_analyzer import analyze_ai_image_content, should_analyze_content, normalize_crime_scene

# GPU 推理服务地址（与 app.py 一致）
GPU_API = "http://127.0.0.1:6006"

# 上传目录
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')

# 并行处理数
MAX_WORKERS = 4
DETECTION_MODELS = ('ai_generated', 'deepfake', 'tamper')


def normalize_detection_models(values=None):
    """Normalize selected special detectors; old callers without a value run all."""
    if isinstance(values, str):
        values = [values]
    requested = set()
    for value in values or []:
        requested.update(item.strip() for item in str(value).split(',') if item.strip())
    if not requested:
        return DETECTION_MODELS
    return tuple(model for model in DETECTION_MODELS if model in requested)


# ══════════════════════════════════════════════════════════
# 核心函数：单张图片检测管线
# ══════════════════════════════════════════════════════════

def process_single_image(image_id, saved_path, selected_models=None, crime_scene=None):
    """
    对单张图片执行完整检测管线，返回检测结果 dict
    结果结构：
    {
        "label": "ai_generated" | "suspected_ai" | "real",
        "label_text": "AI生成图片" | ...,
        "confidence": float,
        "npr_result": {...} | None,
        "deepfake_result": {...} | None,
        "tamper_result": {...} | None,
        "specialized_models": {...} | None,
        "combined_result": {...} | None,
        "deepfake_combined_result": {...} | None,
        "tamper_combined_result": {...} | None,
        "final_comparison_result": {...} | None,
        "watermark_result": {...} | None,
        "content_analysis": {...} | None,
        "error": str | None
    }
    """
    selected_models = normalize_detection_models(selected_models)
    crime_scene = normalize_crime_scene(crime_scene)
    result = {
        "label": None,
        "label_text": None,
        "confidence": 0,
        "npr_result": None,
        "deepfake_result": None,
        "tamper_result": None,
        "specialized_models": None,
        "combined_result": None,
        "deepfake_combined_result": None,
        "tamper_combined_result": None,
        "final_comparison_result": None,
        "watermark_result": None,
        "content_analysis": None,
        "selected_models": list(selected_models),
        "error": None
    }

    # 相机成像物证为三条专项共用，先一次性计算，避免重复解码图片。
    camera_forensics = None
    if 'ai_generated' in selected_models:
        try:
            camera_forensics = analyze_camera_imaging(saved_path)
        except Exception as exc:
            camera_forensics = {"available": False, "detail": f"相机成像物证分析异常：{exc}"}

    # ── 1. 水印检测 ──
    watermark_result = None
    hidden = None
    has_c2pa = False
    try:
        # 隐式水印（TC260 / C2PA 元数据）
        hidden = detect_hidden_watermark(saved_path)
        c2pa_verification = (hidden or {}).get('c2pa_verification', {})
        c2pa_metadata = ((hidden or {}).get('metadata') or {}).get('c2pa', {})
        has_c2pa = bool(c2pa_verification.get('present') or c2pa_metadata.get('present'))
        if has_c2pa:
            selected_models = tuple(dict.fromkeys((*selected_models, 'ai_generated', 'deepfake', 'tamper')))
        # 任何 C2PA 凭证/标记都走三条检测链路，最后由最高分决定类别。
        if hidden and hidden.get('detected') and not has_c2pa:
            watermark_result = hidden
            watermark_result['method'] = 'hidden'
        elif hidden and has_c2pa:
            watermark_result = hidden
            watermark_result['method'] = 'c2pa'
        else:
            # 可见水印（本地 OCR → Qwen 快速复核 → GLM 仅处理不确定项）
            visible = detect_visible_ai_watermark(saved_path)
            if visible and visible.get('detected'):
                watermark_result = visible
                watermark_result['method'] = 'visible'
    except Exception:
        pass

    # 水印短路：检测到水印则直接判定
    if ('ai_generated' in selected_models and watermark_result and watermark_result.get('detected')
            and not has_c2pa):
        result["watermark_result"] = watermark_result
        result["label"] = "ai_generated"
        result["label_text"] = "AI生成图片"
        result["confidence"] = 0.98
        result["content_analysis"] = analyze_ai_image_content(saved_path, {
            "label": result["label"], "label_text": result["label_text"], "confidence": result["confidence"],
            "crime_scene": crime_scene,
        })
        return result

    # ── 2. GPU 推理 ──
    gpu_result = None
    gpu_error = None
    try:
        with open(saved_path, 'rb') as f:
            gpu_resp = requests.post(
                f"{GPU_API}/detect",
                files={'file': (os.path.basename(saved_path), f, 'image/jpeg')},
                data={'models': ','.join(selected_models)},
                timeout=300
            )
            if gpu_resp.status_code == 200:
                gpu_data = gpu_resp.json()
                if gpu_data.get('code') == 200:
                    gpu_result = gpu_data['data']
    except Exception as exc:
        gpu_result = None
        gpu_error = str(exc)

    # ── 3. NPR 零模型检测 ──
    npr_result = None
    if 'ai_generated' in selected_models:
        try:
            npr_result = npr_analyze(saved_path)
        except Exception:
            npr_result = {"score": 0.0, "verdict": "error", "features": {}}
    result["npr_result"] = npr_result

    # ── 4. DeepFake 零模型检测 ──
    deepfake_result = None
    if 'deepfake' in selected_models:
        try:
            deepfake_result = deepfake_zero_model_analysis(
                saved_path, (gpu_result or {}).get('face_glm')
            )
        except Exception:
            deepfake_result = {"deepfake_score": 0.0, "verdict": "error"}
    result["deepfake_result"] = deepfake_result

    # ── 5. 图像篡改检测 ──
    tamper_result = None
    if 'tamper' in selected_models:
        try:
            tamper_result = tamper_zero_model_analysis(saved_path)
        except Exception:
            tamper_result = {"tamper_score": 0.0, "verdict": "error"}
    result["tamper_result"] = tamper_result

    # ── 6. 融合判定 ──
    # GPU 纯模型是专项证据之一，不应在它短暂不可用时丢弃已完成的本地 NPR/ELA/篡改分析。
    from app import compute_combined_verdict, fuse_npr_with_physical_evidence, apply_c2pa_ai_evidence, compute_deepfake_combined_verdict, \
        compute_tamper_combined_verdict, compute_final_comparison

    specialized_models = (gpu_result or {}).get('specialized_models') or {}
    combined = None
    if 'ai_generated' in selected_models:
        npr_result = fuse_npr_with_physical_evidence(npr_result, camera_forensics)
        result["npr_result"] = npr_result
        combined = compute_combined_verdict(npr_result, specialized_models)
        combined = apply_c2pa_ai_evidence(combined, hidden)
    deepfake_combined = None
    if 'deepfake' in selected_models:
        deepfake_combined = compute_deepfake_combined_verdict(
            deepfake_result, specialized_models.get('deepfake_detector')
        )
    tamper_combined = compute_tamper_combined_verdict(tamper_result) if 'tamper' in selected_models else None
    final_comparison = compute_final_comparison(
        combined, deepfake_combined, tamper_combined, selected_models
    )

    result["specialized_models"] = specialized_models or None
    result["combined_result"] = combined
    result["deepfake_combined_result"] = deepfake_combined
    result["tamper_combined_result"] = tamper_combined
    result["final_comparison_result"] = final_comparison
    result["label"] = final_comparison['final_label']
    result["label_text"] = final_comparison['final_verdict']
    result["confidence"] = final_comparison['final_score']

    # C2PA 凭证（无论是否命中 AI 来源）是所有专项共用的来源证据，需要随批量明细保存。
    if watermark_result:
        result["watermark_result"] = watermark_result

    # 仅在用户所选的每一条专项都无法完成时标记任务失败；单个 GPU/GLM 服务故障
    # 不再覆盖其它已完成专项的结论为“50%”。
    completed = []
    if 'ai_generated' in selected_models and npr_result and npr_result.get('verdict') != 'error':
        completed.append('AI全图生成')
    if 'deepfake' in selected_models and deepfake_result and deepfake_result.get('verdict') != 'error':
        completed.append('深度伪造')
    if 'tamper' in selected_models and tamper_result and tamper_result.get('verdict') != 'error':
        completed.append('图像篡改')
    if not completed:
        reasons = []
        if gpu_error:
            reasons.append(f"GPU 推理不可用：{gpu_error[:120]}")
        if npr_result and npr_result.get('error'):
            reasons.append(f"AI全图生成：{npr_result['error']}")
        if deepfake_result and deepfake_result.get('detail'):
            reasons.append(f"深度伪造：{deepfake_result['detail']}")
        if tamper_result and tamper_result.get('detail'):
            reasons.append(f"图像篡改：{tamper_result['detail']}")
        result["error"] = "；".join(reasons) or "所选检测链路均未返回有效结果"

    if should_analyze_content(result["label"]):
        result["content_analysis"] = analyze_ai_image_content(saved_path, {
            "label": result["label"], "label_text": result["label_text"], "confidence": result["confidence"],
            "crime_scene": crime_scene,
        })
    else:
        result["content_analysis"] = {
            "analyzed": False,
            "status": "skipped",
            "detail": "图片尚未确认是 AI 生成，未执行内容识别与风险研判。",
        }

    return result


# ══════════════════════════════════════════════════════════
# 任务处理线程
# ══════════════════════════════════════════════════════════

def process_task(task_id, selected_models=None, crime_scene=None):
    """后台线程：使用线程池并行处理任务中的每张图片"""
    try:
        conn = get_db()
        cursor = conn.cursor()

        # 获取所有待处理图片
        cursor.execute(
            "SELECT image_id, saved_path FROM batch_images WHERE task_id = %s ORDER BY image_id",
            (task_id,)
        )
        images = cursor.fetchall()
        cursor.close()
        conn.close()

        if not images:
            return

        total = len(images)

        def _process_one(image_id, saved_path):
            """处理单张图片并更新 DB"""
            try:
                # 标记 processing
                conn2 = get_db()
                cur2 = conn2.cursor()
                cur2.execute(
                    "UPDATE batch_images SET status = 'processing' WHERE image_id = %s",
                    (image_id,)
                )
                conn2.commit()
                cur2.close()
                conn2.close()

                # 执行检测
                img_result = process_single_image(image_id, saved_path, selected_models, crime_scene)

                # 更新结果到 DB
                conn3 = get_db()
                cur3 = conn3.cursor()
                cur3.execute("""
                    UPDATE batch_images SET
                        status = %s,
                        label = %s,
                        label_text = %s,
                        confidence = %s,
                        npr_result = %s,
                        deepfake_result = %s,
                        tamper_result = %s,
                        specialized_models = %s,
                        combined_result = %s,
                        deepfake_combined_result = %s,
                        tamper_combined_result = %s,
                        final_comparison_result = %s,
                        watermark_result = %s,
                        content_analysis = %s
                    WHERE image_id = %s
                """, (
                    'error' if img_result['error'] else 'done',
                    img_result['label'],
                    img_result['label_text'],
                    img_result['confidence'],
                    json.dumps(img_result['npr_result']) if img_result['npr_result'] else None,
                    json.dumps(img_result['deepfake_result']) if img_result['deepfake_result'] else None,
                    json.dumps(img_result['tamper_result']) if img_result['tamper_result'] else None,
                    json.dumps(img_result['specialized_models']) if img_result['specialized_models'] else None,
                    json.dumps(img_result['combined_result']) if img_result['combined_result'] else None,
                    json.dumps(img_result['deepfake_combined_result']) if img_result['deepfake_combined_result'] else None,
                    json.dumps(img_result['tamper_combined_result']) if img_result['tamper_combined_result'] else None,
                    json.dumps(img_result['final_comparison_result']) if img_result['final_comparison_result'] else None,
                    json.dumps(img_result['watermark_result']) if img_result['watermark_result'] else None,
                    json.dumps(img_result['content_analysis']) if img_result['content_analysis'] else None,
                    image_id
                ))
                conn3.commit()
                cur3.close()
                conn3.close()

            except Exception as e:
                # 更新为错误状态
                try:
                    conn_e = get_db()
                    cur_e = conn_e.cursor()
                    cur_e.execute(
                        "UPDATE batch_images SET status = 'error' WHERE image_id = %s",
                        (image_id,)
                    )
                    conn_e.commit()
                    cur_e.close()
                    conn_e.close()
                except Exception:
                    pass

        # 并行处理
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            futures = [
                executor.submit(_process_one, img['image_id'], img['saved_path'])
                for img in images
            ]
            # 等待所有线程完成（不处理单个异常，已在 _process_one 内部处理）
            for _ in as_completed(futures):
                pass

        # 全部完成，更新任务统计 + 状态
        conn4 = get_db()
        cur4 = conn4.cursor()

        cur4.execute("""
            SELECT
                COUNT(*) AS total,
                SUM(CASE WHEN status = 'done' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
                SUM(CASE WHEN label = 'ai_generated' THEN 1 ELSE 0 END) AS ai,
                SUM(CASE WHEN label = 'suspected_ai' THEN 1 ELSE 0 END) AS suspected,
                SUM(CASE WHEN label = 'real' THEN 1 ELSE 0 END) AS real_count
            FROM batch_images WHERE task_id = %s
        """, (task_id,))
        stats = cur4.fetchone()

        cur4.execute("""
            UPDATE batch_tasks SET
                status = 'completed',
                total_count = %s,
                completed_count = %s,
                ai_count = %s,
                suspected_count = %s,
                real_count = %s,
                error_count = %s
            WHERE task_id = %s
        """, (
            stats['total'],
            stats['completed'],
            stats['ai'],
            stats['suspected'],
            stats['real_count'],
            stats['errors'],
            task_id
        ))
        conn4.commit()
        cur4.close()
        conn4.close()

    except Exception:
        # 任务级异常：更新任务状态为 failed
        try:
            conn_f = get_db()
            cur_f = conn_f.cursor()
            cur_f.execute(
                "UPDATE batch_tasks SET status = 'failed' WHERE task_id = %s",
                (task_id,)
            )
            conn_f.commit()
            cur_f.close()
            conn_f.close()
        except Exception:
            pass


# ══════════════════════════════════════════════════════════
# 对外接口函数
# ══════════════════════════════════════════════════════════

def create_task(task_name, files, selected_models=None, crime_scene=None):
    """
    创建批量检测任务
    - 插入 task 记录到 MySQL
    - 保存文件到 uploads/
    - 插入 image 记录到 MySQL
    - 启动后台处理线程
    返回 task_id
    """
    task_id = str(uuid.uuid4())

    ensure_content_analysis_column()
    conn = get_db()
    cursor = conn.cursor()

    try:
        # 创建任务记录
        cursor.execute("""
            INSERT INTO batch_tasks (task_id, task_name)
            VALUES (%s, %s)
        """, (task_id, task_name))
        conn.commit()

        # 保存文件并插入图片记录
        image_count = 0
        for f in files:
            if f.filename:
                ext = os.path.splitext(f.filename)[1].lower()
                save_name = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uuid.uuid4().hex[:8]}{ext}"
                save_path = os.path.join(UPLOAD_FOLDER, save_name)
                f.save(save_path)

                cursor.execute("""
                    INSERT INTO batch_images (task_id, filename, saved_path)
                    VALUES (%s, %s, %s)
                """, (task_id, f.filename, save_path))
                image_count += 1

        # 更新任务的总图片数
        cursor.execute(
            "UPDATE batch_tasks SET total_count = %s WHERE task_id = %s",
            (image_count, task_id)
        )
        conn.commit()

        # 启动后台处理线程
        selected_models = normalize_detection_models(selected_models)
        crime_scene = normalize_crime_scene(crime_scene)
        t = threading.Thread(target=process_task, args=(task_id, selected_models, crime_scene), daemon=True)
        t.start()

        return task_id

    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def ensure_content_analysis_column():
    """为已有部署做一次幂等迁移，保存批量任务的内容研判 JSON。"""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("SHOW COLUMNS FROM batch_images LIKE 'content_analysis'")
        if cursor.fetchone() is None:
            cursor.execute("ALTER TABLE batch_images ADD COLUMN content_analysis JSON NULL AFTER watermark_result")
            conn.commit()
    finally:
        cursor.close()
        conn.close()


def _get_live_task_stats(cursor, task_id):
    """从图片明细表计算运行中任务的实时进度和分类统计。"""
    cursor.execute("""
        SELECT
            COUNT(*) AS total,
            SUM(CASE WHEN status = 'done' OR status = 'error' THEN 1 ELSE 0 END) AS completed,
            SUM(CASE WHEN status = 'error' THEN 1 ELSE 0 END) AS errors,
            SUM(CASE WHEN label = 'ai_generated' THEN 1 ELSE 0 END) AS ai,
            SUM(CASE WHEN label = 'suspected_ai' THEN 1 ELSE 0 END) AS suspected,
            SUM(CASE WHEN label = 'real' THEN 1 ELSE 0 END) AS real_count
        FROM batch_images WHERE task_id = %s
    """, (task_id,))
    stats = cursor.fetchone()
    return {
        'total_count': stats['total'] or 0,
        'completed_count': stats['completed'] or 0,
        'ai_count': stats['ai'] or 0,
        'suspected_count': stats['suspected'] or 0,
        'real_count': stats['real_count'] or 0,
        'error_count': stats['errors'] or 0,
    }


def get_all_tasks(search=None, status=None, page=1, page_size=20):
    """获取任务列表（支持搜索、状态筛选、分页）
    返回 { "tasks": [...], "total": N, "page": N, "page_size": N, "total_pages": N }
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        conditions = []
        params = []

        if search:
            conditions.append("task_name LIKE %s")
            params.append(f"%{search}%")
        if status:
            conditions.append("status = %s")
            params.append(status)

        where_clause = ""
        if conditions:
            where_clause = "WHERE " + " AND ".join(conditions)

        # 查询总数
        cursor.execute(f"SELECT COUNT(*) AS cnt FROM batch_tasks {where_clause}", params)
        total = cursor.fetchone()['cnt']

        # 查询分页数据
        offset = (page - 1) * page_size
        sql = f"""
            SELECT task_id, task_name, status, total_count, completed_count,
                   ai_count, suspected_count, real_count, error_count,
                   created_at, updated_at
            FROM batch_tasks
            {where_clause}
            ORDER BY created_at DESC
            LIMIT %s OFFSET %s
        """
        cursor.execute(sql, params + [page_size, offset])
        tasks = cursor.fetchall()

        # 任务表的统计在整批完成时才落库；运行中则实时从图片明细表汇总，
        # 使前端每 3 秒的轮询能看到进度条和分类数量持续变化。
        for t in tasks:
            if t['status'] == 'running':
                t.update(_get_live_task_stats(cursor, t['task_id']))

        # 格式化日期
        for t in tasks:
            t['created_at'] = t['created_at'].strftime('%Y-%m-%d %H:%M:%S') if t['created_at'] else None
            t['updated_at'] = t['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if t['updated_at'] else None

        return {
            "tasks": tasks,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size) if total > 0 else 1
        }
    finally:
        cursor.close()
        conn.close()


def delete_task(task_id):
    """删除任务及其所有图片记录"""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM batch_images WHERE task_id = %s", (task_id,))
        cursor.execute("DELETE FROM batch_tasks WHERE task_id = %s", (task_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        cursor.close()
        conn.close()


def get_task_summary(task_id):
    """获取单个任务概要（含实时进度统计）"""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT task_id, task_name, status, total_count, completed_count,
                   ai_count, suspected_count, real_count, error_count,
                   created_at, updated_at
            FROM batch_tasks WHERE task_id = %s
        """, (task_id,))
        task = cursor.fetchone()
        if task:
            task['created_at'] = task['created_at'].strftime('%Y-%m-%d %H:%M:%S') if task['created_at'] else None
            task['updated_at'] = task['updated_at'].strftime('%Y-%m-%d %H:%M:%S') if task['updated_at'] else None

            # 如果任务正在运行，重新从 images 表计算当前进度
            if task['status'] == 'running':
                task.update(_get_live_task_stats(cursor, task_id))

        return task
    finally:
        cursor.close()
        conn.close()


def get_task_images(task_id, label_filter=None, page=1, page_size=20):
    """
    获取任务内的图片列表（分页）
    label_filter: None(全部) | 'ai_generated' | 'suspected_ai' | 'real'
    返回 { "images": [...], "total": N, "page": N, "page_size": N }
    """
    conn = get_db()
    cursor = conn.cursor()
    try:
        offset = (page - 1) * page_size

        if label_filter:
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM batch_images WHERE task_id = %s AND label = %s",
                (task_id, label_filter)
            )
            total = cursor.fetchone()['cnt']
            cursor.execute("""
                SELECT image_id, filename, saved_path, status, label, label_text, confidence
                FROM batch_images
                WHERE task_id = %s AND label = %s
                ORDER BY image_id ASC
                LIMIT %s OFFSET %s
            """, (task_id, label_filter, page_size, offset))
        else:
            cursor.execute(
                "SELECT COUNT(*) AS cnt FROM batch_images WHERE task_id = %s",
                (task_id,)
            )
            total = cursor.fetchone()['cnt']
            cursor.execute("""
                SELECT image_id, filename, saved_path, status, label, label_text, confidence
                FROM batch_images
                WHERE task_id = %s
                ORDER BY image_id ASC
                LIMIT %s OFFSET %s
            """, (task_id, page_size, offset))

        images = cursor.fetchall()
        # confidence 转为 float
        for img in images:
            if img['confidence'] is not None:
                img['confidence'] = float(img['confidence'])

        return {
            "images": images,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size)
        }
    finally:
        cursor.close()
        conn.close()


def get_image_detail(task_id, image_id):
    """获取单张图片的完整检测结果（含所有 JSON 字段）"""
    conn = get_db()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            SELECT * FROM batch_images
            WHERE task_id = %s AND image_id = %s
        """, (task_id, image_id))
        img = cursor.fetchone()
        if img is None:
            return None

        # 解析 JSON 字段
        json_fields = [
            'npr_result', 'deepfake_result', 'tamper_result',
            'specialized_models', 'combined_result',
            'deepfake_combined_result', 'tamper_combined_result',
            'final_comparison_result', 'watermark_result', 'content_analysis'
        ]
        for field in json_fields:
            if img.get(field) and isinstance(img[field], str):
                try:
                    img[field] = json.loads(img[field])
                except (json.JSONDecodeError, TypeError):
                    pass

        if img['confidence'] is not None:
            img['confidence'] = float(img['confidence'])

        return img
    finally:
        cursor.close()
        conn.close()
