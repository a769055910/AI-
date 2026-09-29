# -*- coding: utf-8 -*-
"""AI 图像内容识别与犯罪知识库研判。

视觉模型只抽取可见内容、OCR 文本与风险线索；本模块再按
《AI犯罪鉴别知识库》的规则产生“风险提示”，不对图片作犯罪事实认定。
"""
import json
import os
from pathlib import Path

from siliconflow_client import (
    analyze_image_content_glm, content_clue_bool, normalize_content_tags,
    normalize_water_discharge_observation, has_water_discharge_relationship,
)


# 内容识别属于 AI 图片确认后的后续研判，不能把“疑似”结论当作前提。
CONFIRMED_AI_LABELS = {"ai_generated"}

# 内容研判必须由用户选择一个专项场景，避免把通用内容描述误作某类犯罪线索。
CRIME_SCENES = {
    "fraud": {"label": "涉诈图片", "focus": "身份冒用、虚假凭证、收付款、投资收益、引流话术等诈骗线索"},
    "pornography": {"label": "涉黄图片", "focus": "裸露、性行为、性交易招嫖、换脸造黄及相关工具线索"},
    "rumor": {"label": "涉谣言图片", "focus": "工厂着火、工厂排污、工厂讨薪、工厂被水淹、洪灾救援、伪造交通事故、警情、虚假通报及传播性误导线索；识别场景关系但不认定事件真假"},
}


# 涉诈平台/域名特征库随涉诈 Skill 维护。允许通过环境变量覆盖，便于部署时
# 将 Skill 目录与应用目录分离；读取失败时只跳过命中，不影响图片主流程。
_FRAUD_PLATFORM_LIBRARY_CACHE = {"path": None, "mtime": None, "data": None}


def _fraud_platform_library_path():
    configured = os.environ.get("FRAUD_PLATFORM_LIBRARY_PATH", "").strip()
    if configured:
        return Path(configured)
    return Path.home() / ".codex" / "skills" / "fraud-ai-image-forensics" / "fraud-platforms.json"


def load_fraud_platform_library():
    """读取外置涉诈平台库，返回合法的 categories[] 结构或 None。"""
    library_path = _fraud_platform_library_path()
    try:
        mtime = library_path.stat().st_mtime_ns
    except OSError:
        return None
    cache = _FRAUD_PLATFORM_LIBRARY_CACHE
    if cache["path"] == library_path and cache["mtime"] == mtime:
        return cache["data"]
    try:
        data = json.loads(library_path.read_text(encoding="utf-8-sig"))
        if not isinstance(data, dict) or not isinstance(data.get("categories"), list):
            data = None
    except (OSError, json.JSONDecodeError):
        data = None
    cache.update({"path": library_path, "mtime": mtime, "data": data})
    return data


def match_fraud_platforms(text):
    """在 OCR/URL/画面描述中查询外置库；命中是线索，不构成平台或犯罪事实认定。"""
    library = load_fraud_platform_library()
    normalized_text = str(text or "").casefold()
    matches = []
    if not library or not normalized_text:
        return matches

    seen = set()
    for category in library.get("categories", []):
        if not isinstance(category, dict):
            continue
        category_type = str(category.get("type", "")).strip()
        query_hint = str(category.get("query_hint") or library.get("query_hint") or "").strip()
        for field in ("apps", "domains"):
            values = category.get(field, [])
            if not isinstance(values, list):
                continue
            for value in values:
                value = str(value).strip()
                key = (category_type, field, value.casefold())
                if value and key not in seen and value.casefold() in normalized_text:
                    seen.add(key)
                    matches.append({
                        "type": category_type,
                        "field": field,
                        "matched": value,
                        "query_hint": query_hint,
                    })
    return matches


def normalize_crime_scene(scene):
    """返回受支持的犯罪场景代码；未知或未选择时返回 None。"""
    scene = str(scene or "").strip().lower()
    return scene if scene in CRIME_SCENES else None


def should_analyze_content(label):
    """仅对已确认的 AI 生成图片进行内容识别与风险研判。"""
    return label in CONFIRMED_AI_LABELS


def _unique_texts(values, limit=12):
    result = []
    for value in values or []:
        value = str(value).strip()
        if value and value not in result:
            result.append(value[:120])
        if len(result) >= limit:
            break
    return result


def _truthy(clues, key):
    return content_clue_bool((clues or {}).get(key, False))


def _contains_any(text, words):
    lowered = (text or "").lower()
    return any(word.lower() in lowered for word in words)


def _risk_level(risk_tags):
    levels = [tag.get("level", "低") for tag in risk_tags]
    if "高" in levels:
        return "高"
    if "中" in levels:
        return "中"
    return "低"


def map_to_knowledge(vision_result, forensic_context=None):
    """将结构化视觉线索映射到知识库中的案件模式与核查建议。"""
    forensic_context = forensic_context or {}
    scene = normalize_crime_scene(forensic_context.get("crime_scene"))
    clues = dict(vision_result.get("risk_clues", {}) or {})
    # 只保留当前专项需要的线索，确保三类研判结果彼此独立。
    scene_clues = {
        "fraud": {"identity_document", "medical_or_invoice", "official_document", "financial_or_payment", "investment_or_profit", "video_call_or_liveness", "real_person_face"},
        "pornography": {"sexual_content", "real_person_face", "tool_or_tutorial"},
        "rumor": {
            "emergency_or_accident", "official_document", "flood_disaster",
            "factory_fire", "factory_discharge", "water_discharge", "wage_arrears_protest",
            "factory_flood", "traffic_accident",
        },
    }
    allowed_clues = scene_clues.get(scene, set())
    clues = {key: content_clue_bool(value) if key in allowed_clues else False for key, value in clues.items()}
    tags = normalize_content_tags(vision_result.get("content_tags", []))
    ocr_text = _unique_texts(vision_result.get("ocr_text", []), limit=20)
    visible_text = " ".join(tags + [vision_result.get("scene_description", "")])
    searchable_text = " ".join([visible_text] + ocr_text)
    water = normalize_water_discharge_observation(vision_result.get("rumor_observations"))
    platform_matches = match_fraud_platforms(searchable_text) if scene == "fraud" else []

    risk_tags, matched_knowledge, evidence, recommended_checks = [], [], [], []

    if _truthy(clues, "sexual_content"):
        risk_tags.append({"tag": "涉黄/淫秽风险", "level": "高"})
        matched_knowledge.append("AI 制作淫秽物品类犯罪知识库")
        evidence.append("视觉识别发现裸露、性行为或明显性化内容线索")
        recommended_checks.extend(["复核原始图片/视频帧及传播范围", "核查发布账号、群组、网盘链接和支付记录"])
        if _truthy(clues, "real_person_face"):
            risk_tags.append({"tag": "疑似换脸造黄风险", "level": "高"})
            matched_knowledge.append("AI 换脸造黄（深度伪造敲诈）")
            evidence.append("不雅内容中出现疑似真实人物面部，需核验是否未经授权合成")
            recommended_checks.append("核验肖像来源、当事人授权情况和换脸工具操作痕迹")

    if _truthy(clues, "identity_document"):
        risk_tags.append({"tag": "伪造证件/身份冒用风险", "level": "高"})
        matched_knowledge.append("AI 深度伪造类犯罪知识库：伪造证件/素材绕过平台验证")
        evidence.append("画面包含身份证件、人脸核验或实名认证相关要素")
        recommended_checks.extend(["与签发机关或平台核验字段", "核查活体检测日志、账号实名信息及原始文件元数据"])

    if _truthy(clues, "medical_or_invoice") or _truthy(clues, "official_document"):
        risk_tags.append({"tag": "虚假凭证/文书风险", "level": "高"})
        matched_knowledge.append("AI 生成虚假信息类犯罪知识库：虚假凭证敲诈/虚假证据")
        evidence.append("画面包含医疗证明、发票、正式文书或官方通知类要素")
        recommended_checks.extend(["OCR 提取字段后向开具机构或官方系统交叉核验", "核验票据号码、印章、签发时间与原始文件元数据"])

    if scene == "rumor":
        factory_words = ["工厂", "厂房", "厂区", "车间", "工业园", "生产设施", "烟囱", "储罐"]
        flood_words = ["洪水", "洪涝", "水灾", "洪灾", "被淹", "淹没", "洪水救援"]
        flood_support = ["救援", "冲锋舟", "橡皮艇", "水位", "被淹车辆", "积水", "漂浮物"]
        fire_words = ["火灾", "着火", "起火", "浓烟", "火焰", "爆燃"]
        wage_words = ["讨薪", "欠薪", "拖欠工资", "要工资", "工人维权"]
        accident_words = ["交通事故", "车祸", "追尾", "碰撞", "翻车", "车辆相撞", "事故现场"]

        has_factory = _contains_any(visible_text, factory_words)
        has_flood = _truthy(clues, "flood_disaster") or _truthy(clues, "factory_flood") or (
            _contains_any(visible_text, flood_words)
            or (_contains_any(visible_text, ["积水", "水淹", "内涝"])
                and _contains_any(visible_text, flood_support))
        )
        has_discharge = has_water_discharge_relationship(water) or _truthy(clues, "factory_discharge")
        rumor_scenarios = []
        if _truthy(clues, "factory_fire") or (has_factory and _contains_any(visible_text, fire_words)):
            rumor_scenarios.append(("疑似工厂火灾/涉谣风险", "画面线索涉及厂房或工业设施起火、浓烟，需核实是否为真实事故现场。"))
        if has_discharge:
            industrial_source = water["industrial_source_visible"] if has_water_discharge_relationship(water) else _truthy(clues, "factory_discharge")
            risk_tags.append({
                "tag": "疑似工厂排污/涉谣风险" if industrial_source else "疑似排污/涉谣风险（来源待核）",
                "level": "中",
            })
            matched_knowledge.append("AI 生成虚假信息类犯罪知识库：疑似虚假排污/环境事件传播")
            if water["description"]:
                evidence.append(water["description"])
            if water["location"]:
                evidence.append("线索位置：" + water["location"])
            evidence.append("可见排水通道与异常水体的汇入/扩散线索；需核实水体性质、排水来源和建筑用途，不能据此认定违法排污。")
            tags = list(dict.fromkeys(["疑似异常排水", "异常水体扩散"] + tags))[:12]
            recommended_checks.extend([
                "核验原始图片、首次发布时间及排水口/沟渠地理位置",
                "核查排水来源、建筑用途，并与生态环境部门通报及水质监测记录交叉核验",
            ])
        if _truthy(clues, "wage_arrears_protest") or _contains_any(searchable_text, wage_words):
            rumor_scenarios.append(("疑似工厂讨薪/涉谣风险", "画面或文字线索涉及工人讨薪/欠薪诉求，需核实标语、地点和发布语境。"))
        if _truthy(clues, "factory_flood") or (has_factory and has_flood):
            rumor_scenarios.append(("疑似工厂受淹/涉谣风险", "画面线索涉及厂房或生产设施被水淹；需核实受淹范围及厂区属性。"))
        if has_flood:
            rumor_scenarios.append(("疑似洪灾/水淹现场涉谣风险", "画面呈现洪水、积水或救援线索；图像本身不能确认灾情的时间、地点或真实性。"))
        if _truthy(clues, "traffic_accident") or (
            _truthy(clues, "emergency_or_accident") and _contains_any(searchable_text, accident_words)
        ):
            rumor_scenarios.append(("疑似交通事故现场涉谣风险", "画面线索涉及车辆碰撞或翻覆；需核实是否为事故、演练、影视拍摄或旧图。"))
        if _truthy(clues, "emergency_or_accident") and not rumor_scenarios and not has_discharge:
            rumor_scenarios.append(("疑似突发事件/涉谣风险", "视觉识别提取到灾害、事故或警情线索，需结合具体场景及发布语境核验。"))

        for tag, clue in rumor_scenarios:
            risk_tags.append({"tag": tag, "level": "中"})
            matched_knowledge.append("AI 生成虚假信息类犯罪知识库：虚假险情传播")
            evidence.append(clue)
        if rumor_scenarios:
            recommended_checks.extend([
                "核验首次发布账号、时间和地理位置",
                "与权威部门通报、现场原图和可信新闻源交叉比对",
            ])

    if _truthy(clues, "investment_or_profit") or (_truthy(clues, "financial_or_payment") and _contains_any(searchable_text, ["收益", "盈利", "投资", "稳赚", "导师", "带单"])):
        risk_tags.append({"tag": "虚假投资/诈骗引流风险", "level": "中"})
        matched_knowledge.append("AI 深度伪造类犯罪知识库：AI 辅助网恋诈骗流水线")
        evidence.append("画面包含投资收益、交易账户、资金回报或引流营销要素")
        recommended_checks.extend(["核验投资平台资质、收款账户和资金流水", "保全聊天记录、开户链接及收益截图原件"])
    elif _truthy(clues, "financial_or_payment"):
        risk_tags.append({"tag": "支付/交易引流风险", "level": "低"})
        evidence.append("画面包含二维码、收款账号、转账或交易信息")
        recommended_checks.append("核查收款主体、二维码归属与交易背景")

    if platform_matches:
        hit_values = "、".join(match["matched"] for match in platform_matches[:4])
        hit_types = "、".join(_unique_texts([match["type"] for match in platform_matches], limit=3))
        risk_tags.append({"tag": "涉诈平台/域名特征命中", "level": "中"})
        matched_knowledge.append(f"涉诈平台/域名特征库：{hit_types}")
        evidence.append(f"OCR、网址或画面文字命中外置特征库：{hit_values}")
        recommended_checks.extend([
            "该命中仅为风险线索；请通过官方渠道独立核验 App、域名、收款账户及经营资质",
            "不要使用图片内链接、二维码或客服联系方式进行转账、下载或身份验证",
        ])

    tool_words = ["comfyui", "stable diffusion", "sd", "lora", "facefusion", "deepfacelab", "换脸", "工作流", "模型包", "教程"]
    if _truthy(clues, "tool_or_tutorial") or _contains_any(searchable_text, tool_words):
        if _truthy(clues, "sexual_content"):
            risk_tags.append({"tag": "AI 造黄工具/教程风险", "level": "高"})
            matched_knowledge.append("AI 制作淫秽物品类犯罪知识库：AI 制黄工具、教程售卖")
            evidence.append("出现 AI 图像生成/换脸工具或教程，并与涉黄内容同时出现")
            recommended_checks.extend(["保全工具界面、课程/模型包链接和收费页面", "核查社群成员、支付记录与交付文件"])
        else:
            risk_tags.append({"tag": "AI 生成工具线索", "level": "低"})
            evidence.append("画面出现 AI 出图、换脸、模型或工作流工具线索")

    if _truthy(clues, "video_call_or_liveness") and _truthy(clues, "real_person_face"):
        risk_tags.append({"tag": "人脸核验/冒充身份风险", "level": "中"})
        matched_knowledge.append("AI 深度伪造类犯罪知识库：模拟人脸识别/冒充高管会议诈骗")
        evidence.append("画面同时包含人脸与视频通话、活体检测或身份核验场景")
        recommended_checks.extend(["核验通话原始视频、邀请来源和账号身份", "复核活体检测记录及人脸素材来源"])

    if not risk_tags:
        risk_tags.append({"tag": "未发现知识库高风险内容", "level": "低"})
        evidence.append("已完成视觉内容与 OCR 线索提取，未命中当前知识库的高风险组合")
        recommended_checks.append("如需研判涉谣或证据真实性，应结合发布语境、原始文件和权威来源进一步核验")

    return {
        "content_tags": tags,
        "ocr_text": ocr_text,
        "platform_matches": platform_matches,
        "platform_library": {
            "name": "fraud-platforms.json",
            "matched": bool(platform_matches),
            "note": "命中结果仅用于风险提示，不构成涉诈平台或犯罪事实认定。",
        } if scene == "fraud" else None,
        "risk_tags": risk_tags,
        "matched_knowledge": _unique_texts(matched_knowledge),
        "risk_level": _risk_level(risk_tags),
        "evidence": _unique_texts(evidence, limit=10),
        "recommended_checks": _unique_texts(recommended_checks, limit=10),
        "forensic_context": {
            "label": forensic_context.get("label"),
            "label_text": forensic_context.get("label_text"),
            "confidence": forensic_context.get("confidence"),
            "crime_scene": scene,
            "crime_scene_label": CRIME_SCENES.get(scene, {}).get("label"),
        },
        "crime_scene": scene,
        "crime_scene_label": CRIME_SCENES.get(scene, {}).get("label"),
        "rumor_observations": {"water_discharge": water} if scene == "rumor" else {},
    }


def analyze_ai_image_content(image_path, forensic_context=None):
    """执行视觉内容识别并生成知识库风险研判；失败时返回可展示的降级结果。"""
    forensic_context = forensic_context or {}
    scene = normalize_crime_scene(forensic_context.get("crime_scene"))
    if not scene:
        return {
            "analyzed": False, "status": "skipped",
            "detail": "未选择犯罪场景，未执行针对性内容识别。",
            "content_tags": [], "ocr_text": [], "risk_tags": [], "matched_knowledge": [],
            "risk_level": "未知", "evidence": [], "recommended_checks": [],
        }
    forensic_context = {**forensic_context, "crime_scene": scene}
    try:
        vision_result = analyze_image_content_glm(image_path, crime_scene=scene)
    except Exception as exc:
        return {
            "analyzed": False, "status": "error", "detail": f"内容识别服务异常：{str(exc)[:120]}",
            "content_tags": [], "ocr_text": [], "risk_tags": [], "matched_knowledge": [],
            "risk_level": "未知", "evidence": [], "recommended_checks": [],
        }

    if not vision_result.get("analyzed"):
        return {
            "analyzed": False, "status": "unavailable",
            "detail": vision_result.get("detail", "内容识别服务暂不可用"),
            "content_tags": [], "ocr_text": [], "risk_tags": [], "matched_knowledge": [],
            "risk_level": "未知", "evidence": [], "recommended_checks": [],
        }

    result = map_to_knowledge(vision_result, forensic_context)
    result.update({
        "analyzed": True,
        "status": "ok",
        "method": "GLM-4.5V视觉识别 + AI犯罪鉴别知识库规则映射",
        "scene_description": vision_result.get("scene_description", ""),
        "vision_confidence": vision_result.get("confidence", 0.0),
        "limitations": _unique_texts(vision_result.get("limitations", []), limit=5) or [
            "内容标签仅反映图像可见线索，不单独证明图片所述事件或材料真实。"
        ],
        "detail": "已完成图片内容识别、OCR 提取与知识库风险映射。",
    })
    return result
