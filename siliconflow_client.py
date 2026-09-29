# -*- coding: utf-8 -*-
"""
SiliconFlow API 客户端
使用 GLM-4.5V 多模态视觉模型进行水印检测和人脸检测

模型: zai-org/GLM-4.5V（多模态视觉语言模型，支持图片输入）
"""
import os
import re
import json
import base64
import requests
import numpy as np
import cv2

# ── 配置 ──
API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
API_URL  = "https://api.siliconflow.cn/v1/chat/completions"
MODEL    = "zai-org/GLM-4.5V"
# Watermark review is deliberately cheaper than the GLM content-analysis path.
# It can be overridden during evaluation without changing the content model.
WATERMARK_REVIEW_MODEL = os.environ.get("WATERMARK_REVIEW_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

# 发送前最大边长（像素），超出则等比缩放，避免 base64 过大
MAX_EDGE = 2048

# ──────────────────────────────────────────────────────────
# 图片预处理
# ──────────────────────────────────────────────────────────

def _load_and_resize(image_path):
    """
    读取图片（支持中文路径/大图自动缩放），返回 resized JPEG bytes
    """
    img_data = np.fromfile(image_path, dtype=np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"无法读取图片: {image_path}")

    h, w = img.shape[:2]
    if max(h, w) > MAX_EDGE:
        scale = MAX_EDGE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    _, jpeg_buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return jpeg_buf.tobytes()


def _encode_image(image_path):
    """
    将图片文件编码为 base64 data URL（含自动缩放）
    """
    jpeg_bytes = _load_and_resize(image_path)
    b64 = base64.b64encode(jpeg_bytes).decode('utf-8')
    return f"data:image/jpeg;base64,{b64}"


def _encode_scene_details(image_path):
    """为涉谣识别提供重叠的四象限细节，保留跨区域的排水/救援关系。"""
    image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    crop_w, crop_h = max(1, round(width * 0.65)), max(1, round(height * 0.65))
    origins = [(0, 0), (width - crop_w, 0), (0, height - crop_h),
               (width - crop_w, height - crop_h)]
    canvas = np.full((1536, 1536, 3), 245, dtype=np.uint8)
    for index, (left, top) in enumerate(origins):
        crop = image[top:top + crop_h, left:left + crop_w]
        scale = min(744 / crop.shape[1], 744 / crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)),
                                    max(1, round(crop.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC)
        row, col = divmod(index, 2)
        x = col * 768 + (768 - resized.shape[1]) // 2
        y = row * 768 + (768 - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("场景细节编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode('utf-8')


def _encode_watermark_corners(image_path):
    """Build a four-corner contact sheet so small watermarks survive low detail."""
    img_data = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"无法读取图片: {image_path}")
    height, width = image.shape[:2]
    crop_w, crop_h = max(1, round(width * 0.26)), max(1, round(height * 0.26))
    corners = [
        image[:crop_h, :crop_w], image[:crop_h, width - crop_w:],
        image[height - crop_h:, :crop_w], image[height - crop_h:, width - crop_w:],
    ]
    canvas = np.full((1024, 1024, 3), 245, dtype=np.uint8)
    for index, crop in enumerate(corners):
        scale = min(480 / crop.shape[1], 480 / crop.shape[0])
        resized = cv2.resize(crop, (max(1, round(crop.shape[1] * scale)), max(1, round(crop.shape[0] * scale))),
                             interpolation=cv2.INTER_AREA)
        row, col = divmod(index, 2)
        x = col * 512 + (512 - resized.shape[1]) // 2
        y = row * 512 + (512 - resized.shape[0]) // 2
        canvas[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    ok, encoded = cv2.imencode('.jpg', canvas, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise ValueError("水印角落联系表编码失败")
    return "data:image/jpeg;base64," + base64.b64encode(encoded.tobytes()).decode('utf-8')


# ──────────────────────────────────────────────────────────
# API 核心调用
# ──────────────────────────────────────────────────────────

def _call_api(image_path, prompt, max_tokens=512, temperature=0.1, *, model=None, image_url=None, detail=None,
              additional_image_urls=None):
    """
    调用 SiliconFlow Chat Completions API (GLM-4.5V 视觉)
    发送图片 + 文本 prompt，返回模型的文本回复
    """
    image_url = image_url or _encode_image(image_path)
    image_input = {"url": image_url}
    if detail:
        image_input["detail"] = detail
    content = [{"type": "image_url", "image_url": image_input}]
    for extra_url in additional_image_urls or []:
        extra_input = {"url": extra_url}
        if detail:
            extra_input["detail"] = detail
        content.append({"type": "image_url", "image_url": extra_input})
    content.append({"type": "text", "text": prompt})

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type":  "application/json",
    }

    payload = {
        "model": model or MODEL,
        "messages": [
            {
                "role": "user",
                "content": content,
            }
        ],
        "max_tokens":   max_tokens,
        "temperature":  temperature,
    }

    resp = requests.post(API_URL, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]


def _extract_json(text):
    """
    从模型回复中提取 JSON 对象
    支持 ```json ... ``` 代码块和纯 {...} 两种格式
    """
    # 先尝试匹配 ```json 代码块
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # 再尝试直接匹配大括号
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        try:
            return json.loads(m.group())
        except json.JSONDecodeError:
            pass
    return None


def _has_visible_watermark_evidence(*values):
    """Accept only affirmative, reproducible watermark evidence from a VLM."""
    empty_values = {
        '', 'none', 'null', 'nil', 'n/a', 'na', '未知', '无',
        '未见水印', '未检测到水印', '未发现水印', '没有水印', '无水印',
        '未见', '未检测到', '未发现', '没有',
    }
    normalized = [str(value or '').strip().lower() for value in values]
    negative_markers = ('未见', '未检测', '未发现', '没有', '无水印', 'not found', 'no watermark')
    return any(
        value not in empty_values and not any(marker in value for marker in negative_markers)
        for value in normalized
    )


# ──────────────────────────────────────────────────────────
# 对外接口一：水印检测
# ──────────────────────────────────────────────────────────

def detect_watermark_qwen(image_path):
    """Cheap VLM watermark review using an enlarged four-corner contact sheet.

    A valid high-confidence positive short-circuits the image flow.  A valid
    high-confidence negative skips GLM; all weak/invalid responses are marked
    ``uncertain`` for the GLM fallback layer.
    """
    prompt = """你是图像取证助手。输入是一张四角联系表，顺序为：左上、右上、左下、右下；每个角落都已放大。
仅检查 AI 图像生成平台的可见水印、Logo 或签名文字，例如豆包、即梦、通义、文心、可灵、混元、ChatGLM、DALL·E、Midjourney、Adobe Firefly、Bing Image Creator、AI Generated、AIGC。
普通页面文字、用户名、日期、图片主体文字不算水印。看不清时不要猜测。
只输出 JSON：
{"has_watermark":true,"platform":"平台名或null","watermark_text":"可见文字或null","position":"左上/右上/左下/右下或null","confidence":0到1,"decision_confidence":0到1,"reasoning":"不超过40字"}
当没有水印时 has_watermark 为 false；decision_confidence 表示你对“有或无”这个结论本身的把握。"""
    unavailable = {
        'detected': False, 'source': None, 'detected_text': None, 'position': None,
        'confidence': 0.0, 'decision_confidence': 0.0, 'review_status': 'uncertain',
        'method': 'qwen3_vl_8b_corner_review',
    }
    try:
        response_text = _call_api(
            image_path, prompt, max_tokens=96, temperature=0,
            model=WATERMARK_REVIEW_MODEL,
            image_url=_encode_watermark_corners(image_path), detail='low',
        )
        result = _extract_json(response_text)
        if not isinstance(result, dict):
            return {**unavailable, 'detail': 'Qwen 水印复核结果解析失败', 'raw_response': response_text[:500]}

        detected = bool(result.get('has_watermark', False))
        confidence = max(0.0, min(1.0, float(result.get('confidence', 0.0) or 0.0)))
        decision_confidence = max(0.0, min(1.0, float(result.get('decision_confidence', confidence) or confidence)))
        has_visible_evidence = _has_visible_watermark_evidence(
            result.get('platform'), result.get('watermark_text')
        )
        review_status = 'confirmed' if detected and has_visible_evidence and confidence >= 0.80 else (
            'clear' if not detected and decision_confidence >= 0.80 else 'uncertain'
        )
        return {
            'detected': detected,
            'source': result.get('platform') if detected and has_visible_evidence else None,
            'detected_text': result.get('watermark_text') if detected and has_visible_evidence else None,
            'position': result.get('position') if detected else None,
            'confidence': confidence if detected else 0.0,
            'decision_confidence': decision_confidence,
            'review_status': review_status,
            'detail': str(result.get('reasoning') or 'Qwen3-VL-8B 四角水印复核完成') + (
                '' if not detected or has_visible_evidence else '；未返回平台或可见文字，不作为快捷判定依据'
            ),
            'method': 'qwen3_vl_8b_corner_review',
            'review_model': WATERMARK_REVIEW_MODEL,
            'raw_response': response_text,
        }
    except requests.exceptions.RequestException as exc:
        return {**unavailable, 'detail': f'Qwen 水印复核请求失败: {str(exc)[:120]}', 'error': str(exc)}
    except Exception as exc:
        return {**unavailable, 'detail': f'Qwen 水印复核异常: {str(exc)[:120]}', 'error': str(exc)}


def detect_watermark_glm(image_path):
    """
    使用 GLM-4.5V 视觉模型检测图片中的 AI 平台水印/签名

    参数:
        image_path: str  图片文件路径

    返回:
        dict: {
            "detected":       bool,    # 是否检测到水印
            "source":         str,     # 平台名称（未检测到为 None）
            "detected_text":  str,     # 水印上的具体文字
            "position":       str,     # 水印位置描述
            "confidence":     float,   # 置信度 0~1
            "detail":         str,     # 判断依据/详细说明
            "method":         str,     # "glm_vision"
            "raw_response":   str,     # 模型原始回复（调试用）
        }
    """
    prompt = """你是一名专业的图像取证分析师。请仔细观察这张图片的全部区域，尤其是四个角落、底部边缘、顶部边缘，判断图片中是否存在任何AI生成平台的水印、Logo标志或签名文字。

需要重点排查的平台水印包括（但不限于）：

【国内AI平台】
豆包 / doubao、即梦 / jimeng、通义千问 / 通义万相 / tongyi、文心一格 / 文心一言、讯飞星火、智谱清影 / 清言 / ChatGLM、混元 / hunyuan、可灵 / kling、奇域、堆友、LiblibAI、吐司 / tusiart

【国外AI平台】
DALL·E / DALL-E、Midjourney / MJ、Stable Diffusion、Bing Image Creator / Microsoft Designer、Adobe Firefly、NovelAI、Ideogram、Flux / Flux.1 / Flux Pro、Leonardo AI、SeaArt、Playground AI

【通用AI标识】
AI生成、AIGC、AI Generated、Generated by AI、Created with AI、AI创作

请严格按照以下JSON格式返回结果（只输出JSON，不要任何额外说明文字）：
{
  "has_watermark": true,
  "platform": "豆包",
  "watermark_text": "豆包AI生成",
  "position": "右下角",
  "confidence": 0.95,
  "reasoning": "在图片右下角发现'豆包AI生成'白色半透明文字，确认为豆包平台水印"
}

如果没有发现任何水印，返回：
{
  "has_watermark": false,
  "platform": null,
  "watermark_text": null,
  "position": null,
  "confidence": 0.0,
  "reasoning": "经检查四角及边缘区域，未发现AI平台水印或签名文字"
}"""

    try:
        response_text = _call_api(image_path, prompt, max_tokens=512, temperature=0.1)
        result = _extract_json(response_text)

        if result and result.get("has_watermark"):
            return {
                "detected":      True,
                "source":        result.get("platform"),
                "detected_text": result.get("watermark_text"),
                "position":      result.get("position"),
                "confidence":    float(result.get("confidence", 0.85)),
                "detail":        result.get("reasoning", f"GLM-4.5V 检测到 {result.get('platform', '未知平台')} 水印"),
                "method":        "glm_vision",
                "raw_response":  response_text,
            }
        else:
            reasoning = result.get("reasoning", "") if result else ""
            detail = reasoning or "GLM-4.5V 视觉分析：未发现AI平台水印"
            return {
                "detected":      False,
                "source":        None,
                "detected_text": None,
                "position":      None,
                "confidence":    0.0,
                "detail":        detail,
                "method":        "glm_vision",
                "raw_response":  response_text if not result else None,
            }

    except requests.exceptions.RequestException as e:
        return {
            "detected":      False,
            "source":        None,
            "detected_text": None,
            "position":      None,
            "confidence":    0.0,
            "detail":        f"GLM-4.5V API 请求失败: {str(e)[:120]}",
            "method":        "glm_vision",
            "error":         str(e),
        }
    except Exception as e:
        return {
            "detected":      False,
            "source":        None,
            "detected_text": None,
            "position":      None,
            "confidence":    0.0,
            "detail":        f"GLM-4.5V 水印检测异常: {str(e)[:120]}",
            "method":        "glm_vision",
            "error":         str(e),
        }


# ──────────────────────────────────────────────────────────
# 对外接口二：人脸检测
# ──────────────────────────────────────────────────────────

def detect_face_glm(image_path):
    """
    使用 GLM-4.5V 视觉模型检测图片中的真实人脸

    参数:
        image_path: str  图片文件路径

    返回:
        dict: {
            "face_detected": bool,   # 是否检测到人脸
            "face_count":    int,    # 人脸数量
            "faces":         list,   # 人脸详情列表
            "confidence":    float,  # 置信度 0~1
            "detail":        str,    # 判断依据
            "method":        str,    # "glm_vision"
            "raw_response":  str,
        }
    """
    prompt = """你是一名专业的人脸检测分析师。请仔细观察这张图片，判断图片中是否存在人脸，并统计数量。

判断时请注意以下要点：
1. 识别所有人类面孔（正面、侧面、半侧面均需统计）
2. AI 生成的逼真虚拟人物面孔也需要统计为"检测到人脸"
3. 卡通风格、简笔画、明显非写实风格的面孔不计入
4. 多张人脸请全部统计，不要遗漏
5. 仔细检查图片的每个区域，包括远处的小面孔

请严格按照以下JSON格式返回结果（只输出JSON，不要任何额外说明文字）：

有检测到人脸时示例：
{
  "has_face": true,
  "face_count": 1,
  "faces": [
    {
      "position": "画面中央偏右",
      "size": "large",
      "is_human": true
    }
  ],
  "confidence": 0.95,
  "reasoning": "图片中央有一张清晰的女性面孔，约占画面30%面积，五官清晰可见"
}

无人脸时示例：
{
  "has_face": false,
  "face_count": 0,
  "faces": [],
  "confidence": 0.98,
  "reasoning": "图片为风景照，经全面检查未发现任何人脸"
}"""

    try:
        response_text = _call_api(image_path, prompt, max_tokens=512, temperature=0.1)
        result = _extract_json(response_text)

        if result:
            return {
                "face_detected": bool(result.get("has_face", False)),
                "face_count":    int(result.get("face_count", 0)),
                "faces":         result.get("faces", []),
                "confidence":    float(result.get("confidence", 0.5)),
                "detail":        result.get("reasoning", ""),
                "method":        "glm_vision",
                "raw_response":  response_text,
            }
        else:
            return {
                "face_detected": False,
                "face_count":    0,
                "faces":         [],
                "confidence":    0.0,
                "detail":        "GLM-4.5V 返回结果解析失败",
                "method":        "glm_vision",
                "raw_response":  response_text,
            }

    except requests.exceptions.RequestException as e:
        return {
            "face_detected": False,
            "face_count":    0,
            "faces":         [],
            "confidence":    0.0,
            "detail":        f"GLM-4.5V API 请求失败: {str(e)[:120]}",
            "method":        "glm_vision",
            "error":         str(e),
        }
    except Exception as e:
        return {
            "face_detected": False,
            "face_count":    0,
            "faces":         [],
            "confidence":    0.0,
            "detail":        f"GLM-4.5V 人脸检测异常: {str(e)[:120]}",
            "method":        "glm_vision",
            "error":         str(e),
        }


# ──────────────────────────────────────────────────────────
# 对外接口三：图片内容识别（供犯罪知识库研判）
# ──────────────────────────────────────────────────────────

CONTENT_CLUE_LABELS = {
    "sexual_content": "性化内容", "real_person_face": "人物面部",
    "identity_document": "身份证件", "medical_or_invoice": "医疗凭证或发票",
    "official_document": "正式文书", "emergency_or_accident": "灾害或事故线索",
    "flood_disaster": "洪水灾害", "factory_fire": "工厂火灾",
    "factory_discharge": "疑似工厂排污", "water_discharge": "疑似异常排水",
    "wage_arrears_protest": "讨薪诉求", "factory_flood": "厂区受淹",
    "traffic_accident": "交通事故", "financial_or_payment": "支付或交易",
    "investment_or_profit": "投资收益", "tool_or_tutorial": "工具或教程",
    "video_call_or_liveness": "视频通话或活体核验",
}


def content_clue_bool(value):
    """兼容模型返回的布尔值；字符串 false 不能成为阳性线索。"""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return isinstance(value, (int, float)) and value == 1


def normalize_content_tags(values):
    if not isinstance(values, list):
        return []
    tags = []
    for value in values:
        if not isinstance(value, str):
            continue
        value = value.strip()
        tag = CONTENT_CLUE_LABELS.get(value, value)[:120]
        if tag and tag not in tags:
            tags.append(tag)
    return tags[:12]


def normalize_water_discharge_observation(observations):
    observations = observations if isinstance(observations, dict) else {}
    water = observations.get("water_discharge", {})
    water = water if isinstance(water, dict) else {}
    keys = ("outlet_or_channel_visible", "water_discoloration_visible",
            "flow_into_waterbody_visible", "spreading_plume_visible", "industrial_source_visible")
    result = {key: content_clue_bool(water.get(key, False)) for key in keys}
    result.update({key: str(water.get(key) or "")[:240] for key in ("description", "location")})
    return result


def has_water_discharge_relationship(water):
    """要求排水通道、汇入关系和异常水流共同出现，避免仅凭水色触发。"""
    return (water.get("outlet_or_channel_visible", False)
            and water.get("flow_into_waterbody_visible", False)
            and (water.get("water_discoloration_visible", False)
                 or water.get("spreading_plume_visible", False)))


def analyze_image_content_glm(image_path, crime_scene=None):
    """提取图片可见内容、OCR 与风险线索，不对犯罪事实作结论。"""
    crime_scene_prompts = {
        "fraud": "本次为“涉诈图片”专项：重点识别身份冒用、虚假凭证、收付款/投资收益、引流话术和视频核验等线索；不输出涉黄或涉谣言研判。",
        "pornography": "本次为“涉黄图片”专项：重点识别裸露、性行为、性交易招嫖、换脸造黄及工具教程等线索；不输出涉诈或涉谣言研判。",
        "rumor": "本次为“涉谣言图片”专项：重点识别工厂着火、工厂排污、工厂讨薪、工厂/厂区被水淹、洪灾救援和交通事故场景。第一张为原图，第二张仅为同图四象限的重叠放大细节（从左上、右上、左下、右下排列），不是其他事件；OCR 只读取原图真实文字。先逐区查看水道、建筑周边、车辆和人群，再描述有关系的事件线索，不要只罗列建筑、河流、道路、农田。尤其逐条核查：是否可见管口/涵洞/沟渠；其中水流是否连续汇入河湖；汇入口附近是否存在与周边不同的水色、泡沫或扩散色带；是否有工业设施与该排水路径的直接联系。工业来源不明确时仍须记录可见异常排水，不要因为是高层建筑或看不到工厂招牌而忽略。water_discharge 仅表示可见的异常排水关系，factory_discharge 仅在工业来源与排水关系均有视觉支持时为 true。仅有浑浊河水、停放消防车或普通人群不能直接推断排污/火灾/讨薪。描述优先写可见流向、色带与位置，其次写建筑与救援活动；不得认定实际污染、违法排放、建筑用途或事件真假。",
    }
    scene_instruction = crime_scene_prompts.get(crime_scene, "")
    prompt = """你是一名图像内容识别助手。请只根据图片中可见内容，提取客观场景、文字和线索，供后续规则引擎进行风险研判。
__CRIME_SCENE_INSTRUCTION__

禁止把“图片像 AI 生成”直接推断为犯罪；禁止判断新闻真伪、证件真伪或人物身份。涉谣、伪造、违法等只能作为“需进一步核验的视觉线索”。

请关注：人物/人脸、身份证件或人脸核验、医疗证明/发票/正式文书、火灾事故/警情、投资收益/转账/收款码、短视频/聊天/视频通话界面、AI 出图/换脸工具或教程、裸露或性行为内容。涉谣专项额外检查工厂火灾、工厂排污、工人讨薪、厂区/城市洪水和交通事故；洪灾线索关注被淹建筑/车辆、积水水位、漂浮物、救援人员/船只及淹水范围，排污线索关注厂区出口/管道/沟渠与异常水体之间是否存在连续排放路径。仅提取画面能支持的内容，不要臆测建筑用途、污染来源、伤亡或事件真假。读取图片中清晰可见的重要文字；对可辨识的 App 名称、网址、域名、短链、二维码旁文字，按画面原样完整写入 ocr_text；看不清不要猜测，也不要补全网址。

严格只输出如下 JSON：
{
  "scene_description": "不超过120字的客观画面描述",
  "ocr_text": ["图片中清晰可见的文字，最多12条"],
  "content_tags": ["人物/身份证件/医疗凭证/工厂火灾/工厂排污/讨薪/洪水/救援/交通事故/投资收益/收款码/工具界面等，最多12个"],
  "risk_clues": {
    "sexual_content": false,
    "real_person_face": false,
    "identity_document": false,
    "medical_or_invoice": false,
    "official_document": false,
    "emergency_or_accident": false,
    "flood_disaster": false,
    "factory_fire": false,
    "factory_discharge": false,
    "water_discharge": false,
    "wage_arrears_protest": false,
    "factory_flood": false,
    "traffic_accident": false,
    "financial_or_payment": false,
    "investment_or_profit": false,
    "tool_or_tutorial": false,
    "video_call_or_liveness": false
  },
  __RUMOR_OBSERVATIONS__
  "confidence": 0.0,
  "limitations": ["只写与本图有关的识别局限，最多3条"]
}"""
    prompt = prompt.replace("__CRIME_SCENE_INSTRUCTION__", scene_instruction)
    water_schema = '''"rumor_observations": {
    "water_discharge": {
      "outlet_or_channel_visible": false,
      "water_discoloration_visible": false,
      "flow_into_waterbody_visible": false,
      "spreading_plume_visible": false,
      "industrial_source_visible": false,
      "description": "可见排水通道、流向与扩散关系；没有则写未见，不推测污染来源",
      "location": "相关线索在原图中的相对位置"
    }
  },''' if crime_scene == "rumor" else ""
    prompt = prompt.replace("__RUMOR_OBSERVATIONS__", water_schema)
    try:
        extra_images = [_encode_scene_details(image_path)] if crime_scene == "rumor" else None
        response_text = _call_api(image_path, prompt, max_tokens=1400 if crime_scene == "rumor" else 900,
                                  temperature=0.1, additional_image_urls=extra_images,
                                  detail="high" if crime_scene == "rumor" else None)
        result = _extract_json(response_text)
        if not result:
            return {"analyzed": False, "detail": "GLM-4.5V 内容识别结果解析失败", "raw_response": response_text}

        clue_keys = [
            "sexual_content", "real_person_face", "identity_document", "medical_or_invoice",
            "official_document", "emergency_or_accident", "flood_disaster",
            "factory_fire", "factory_discharge", "water_discharge", "wage_arrears_protest",
            "factory_flood", "traffic_accident", "financial_or_payment",
            "investment_or_profit", "tool_or_tutorial", "video_call_or_liveness",
        ]
        clues = result.get("risk_clues") if isinstance(result.get("risk_clues"), dict) else {}
        normalized_clues = {key: content_clue_bool(clues.get(key, False)) for key in clue_keys}
        water = normalize_water_discharge_observation(result.get("rumor_observations"))
        if crime_scene == "rumor":
            normalized_clues["water_discharge"] = has_water_discharge_relationship(water)
        tags = normalize_content_tags(result.get("content_tags", []))
        if crime_scene == "rumor" and normalized_clues["water_discharge"]:
            tags = list(dict.fromkeys(["疑似异常排水", "异常水体扩散"] + tags))[:12]
        return {
            "analyzed": True,
            "scene_description": str(result.get("scene_description", ""))[:240],
            "ocr_text": result.get("ocr_text", []) if isinstance(result.get("ocr_text"), list) else [],
            "content_tags": tags,
            "risk_clues": normalized_clues,
            "rumor_observations": {"water_discharge": water} if crime_scene == "rumor" else {},
            "confidence": max(0.0, min(1.0, float(result.get("confidence", 0.0) or 0.0))),
            "limitations": result.get("limitations", []) if isinstance(result.get("limitations"), list) else [],
            "raw_response": response_text,
        }
    except requests.exceptions.RequestException as exc:
        return {"analyzed": False, "detail": f"GLM-4.5V 内容识别请求失败: {str(exc)[:120]}", "error": str(exc)}
    except Exception as exc:
        return {"analyzed": False, "detail": f"GLM-4.5V 内容识别异常: {str(exc)[:120]}", "error": str(exc)}
