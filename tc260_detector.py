# -*- coding: utf-8 -*-
"""
阶段二：TC260:AIGC 隐式标识检测（国标 GB 45438，国内平台 AI 生成强证据）

规范依据：
    - GB 45438—2025《网络安全技术 人工智能生成合成内容标识方法》（强制性国标）
    - TC260《人工智能生成合成内容标识方法 文件元数据隐式标识》网络安全标准实践指南
    - 《人工智能生成合成内容标识办法》（2025-09-01 施行）

隐式标识格式（国内实测两种）：
    1. 直放形式（字节系：豆包 / 即梦，写入 PNG 文本 chunk / XMP）：
       TC260:AIGC={"Label":"1","ContentProducer":"doubao","ProduceID":"...",
                   "ReservedCode1":"","ContentPropagator":"","PropagateID":"","ReservedCode2":""}
    2. 嵌套 JSON 形式（腾讯混元 / 百度文心，写入 JPEG EXIF）：
       {"AIGC":{"Label":"1","ContentProducer":"...","ProduceID":"...",...}}

核心字段（GB 45438）：
    Label              内容标签（1 = 人工智能生成）
    ContentProducer    内容提供者标识（平台编码 / 名称）
    ProduceID          内容生产标识（一次生成任务的唯一 ID）
    ContentPropagator  内容传播者标识（可选）
    PropagateID        内容传播标识（可选）
    ReservedCode1/2    保留字段

设计原则：
    - 单向强证据：命中 TC260:AIGC 且字段完整 → 判 AI 生成（置信度高）；
      未命中 → 不代表不是 AI（元数据可被抹除 / 截图 / 二次压缩剥离）。
    - 与阶段一（国外元数据/C2PA）互补：阶段二覆盖国内头部平台（豆包、即梦、混元、文心等）。

依赖：Pillow；复用 hidden_watermark_detector 的基础字节解析工具。
"""

import json
import re

from hidden_watermark_detector import (
    _collect_metadata_texts,
    _read_png_chunks,
    _parse_png_text_chunks,
    _load_image_bytes,
)
from visible_watermark_detector import _normalize


# ═══════════════════════════════════════════════════════════════
# TC260 关键词 / 规则库
# ═══════════════════════════════════════════════════════════════

# 命名空间前缀（GB 45438 约定的元数据隐式标识命名空间）
TC260_NAMESPACE = 'TC260:AIGC'

# 隐式标识必备字段（GB 45438：Label + ContentProducer + ProduceID）
REQUIRED_FIELDS = ('Label', 'ContentProducer', 'ProduceID')
# 可选字段
OPTIONAL_FIELDS = ('ContentPropagator', 'PropagateID', 'ReservedCode1', 'ReservedCode2')

# Label 取值语义（GB 45438 附录）：1 = 人工智能生成内容
LABEL_AI_GENERATED = {'1', 'ai生成', 'aigc', 'true', '1.0'}

# 字节级兜底标记（用于「检测到命名空间但 JSON 解析失败」的弱证据分支）
BYTE_MARKERS = [b'TC260:AIGC', b'TC260', b'AIGC']

# 平台推断规则：ContentProducer 原文 / ProduceID 特征 → 中文平台名
# 注意：内容生产者编码由平台自行分配，本表为实测 + 公开资料的常见值，
#       识别不出时回退为「原始 ContentProducer 编码」本身，不强行猜测。
PLATFORM_RULES = [
    # (匹配串, 平台名, 匹配目标: 'producer' | 'produce_id' | 'any')
    ('doubao', '豆包', 'any'),
    ('jimeng', '即梦', 'any'),
    ('baichuan', '百川智能', 'any'),
    ('zhipu', '智谱清言', 'any'),
    ('qwen', '通义千问', 'any'),
    ('tongyi', '通义', 'any'),
    ('cogview', '智谱 CogView', 'any'),
    ('kling', '可灵', 'any'),
    ('kuaishou', '快手可灵', 'any'),
    ('minimax', 'MiniMax 海螺', 'any'),
    ('wanzhi', '腾讯智影', 'any'),
    ('hunyuan', '腾讯混元', 'any'),
    ('yuanbao', '腾讯元宝', 'any'),
    ('ernie', '文心一言', 'any'),
    ('wenxin', '文心', 'any'),
    ('xinference', '讯飞星火', 'any'),
    ('xunfei', '讯飞', 'any'),
    ('stepfun', '阶跃星辰', 'any'),
    ('kimi', '月之暗面 Kimi', 'any'),
    ('liblib', 'LiblibAI', 'any'),
    ('tusiart', '吐司 TusiArt', 'any'),
]

# 基于生产者编码前缀的平台推断（编码体系为公开资料汇总，识别不出则回退原文）
PRODUCER_PREFIX_RULES = [
    ('00119144030071526726', '腾讯混元'),
    ('001191110000802100433', '百度文心'),
    ('00119144030008867405', '字节系（即梦/豆包）'),
    ('00119114000000123456', '字节系（即梦/豆包）'),
    ('001191440101', '通义千问（阿里）'),
    ('0011913400007', '讯飞星火'),
]


# ═══════════════════════════════════════════════════════════════
# 隐式标识提取
# ═══════════════════════════════════════════════════════════════

def _clean_json_text(text):
    """清理可能包裹在 XML 实体 / 转义中的 JSON 文本"""
    t = text.strip()
    # 反转义 XML 实体（XMP / iTXt 常见）
    t = (t.replace('&quot;', '"').replace('&amp;', '&')
          .replace('&lt;', '<').replace('&gt;', '>').replace('&apos;', "'"))
    # 剥离行尾多余的 `,`（部分实现 JSON 末尾多逗号）
    t = re.sub(r',\s*}$', '}', t.strip())
    return t


def _extract_tc260_payload(value):
    """
    从单个字段值中提取 TC260:AIGC 隐式标识，返回 (json_obj, 形式描述) 或 (None, None)。

    兼容四种实测形式：
        1) XMP 元素：`<TC260:AIGC>{...}</TC260:AIGC>`（PNG iTXt / XMP 内，JSON 被 XML 实体转义）
        2) 直放：`TC260:AIGC={...}` —— 取 `=` 后 JSON
        3) 嵌套：`{"AIGC":{...}}`   —— 取 AIGC 键值
        4) 兜底：整个值就是一个含 Label/ContentProducer 的 JSON
    """
    if not value or not isinstance(value, str):
        return None, None
    t = value.strip()
    raw_unescaped = _clean_json_text(t)

    # 形式 1：XMP 元素形式 <TC260:AIGC>{...}</TC260:AIGC>
    # 注意：先于反转义做正则（XML 实体时 `&quot;` 会破坏 JSON 结构，需先定位元素再整体反转义）
    m = re.search(r'<TC260\s*:\s*AIGC[^>]*>\s*(.*?)\s*</TC260\s*:\s*AIGC>', t, re.IGNORECASE | re.DOTALL)
    if m:
        payload = _clean_json_text(m.group(1))
        try:
            return json.loads(payload), 'XMP 元素 <TC260:AIGC> 形式'
        except Exception:
            pass

    t2 = raw_unescaped

    # 形式 2：TC260:AIGC= 前缀（支持单引号 / 无引号包裹的 JSON）
    m = re.search(r'TC260\s*:\s*AIGC\s*=\s*(.+)', t2, re.IGNORECASE)
    if m:
        payload = m.group(1).strip().strip('"').strip()
        try:
            return json.loads(payload), 'TC260:AIGC 直放形式'
        except Exception:
            pass

    # 形式 2：{"AIGC": {...}} 嵌套（含大小写变体）
    m = re.search(r'\{\s*["\']?AIGC["\']?\s*:\s*(\{)', t, re.IGNORECASE)
    if m:
        # 从 AIGC 值开始，尝试找完整 JSON 对象
        start = m.start(1)
        try:
            obj, _end = json.JSONDecoder().raw_decode(t[start:])
            if isinstance(obj, dict):
                return obj, 'AIGC 嵌套 JSON 形式'
        except Exception:
            pass

    # 形式 3：整个值就是 JSON 对象，且含 TC260 必备字段
    try:
        obj = json.loads(t)
        if isinstance(obj, dict) and 'Label' in obj and ('ContentProducer' in obj or 'ProduceID' in obj):
            return obj, 'TC260 字段直存 JSON'
    except Exception:
        pass

    return None, None


def _parse_nested_aigc(obj):
    """
    处理可能的双层结构：{"AIGC": {...}} 或 {"Label": ...} 平铺。
    返回标准化后的字段 dict（键为 GB 45438 字段名）。
    """
    if not isinstance(obj, dict):
        return None
    fields = obj
    # 若整体被 AIGC / AI 包裹一层，剥开
    if 'AIGC' in obj and isinstance(obj['AIGC'], dict):
        fields = obj['AIGC']
    elif 'AI' in obj and isinstance(obj['AI'], dict):
        fields = obj['AI']
    # 提取字符串字段（部分实现写数字）
    out = {}
    for k, v in fields.items():
        out[k] = str(v).strip() if v is not None else ''
    return out


def _field_completeness(fields):
    """计算必备字段完整度，返回 (命中数, 必备字段存在列表)"""
    present = [f for f in REQUIRED_FIELDS if fields.get(f)]
    return len(present), present


def _infer_platform(fields):
    """
    依据 ContentProducer / ProduceID 推断平台名。
    返回 (platform_name, matched_key)。识别不出时平台名为 None（调用方回退原文）。
    """
    producer = fields.get('ContentProducer', '') or ''
    produce_id = fields.get('ProduceID', '') or ''
    np = _normalize(producer)
    nid = _normalize(produce_id)
    nall = np + '|' + nid

    # 优先精确规则（any 命中 ContentProducer 或 ProduceID）
    for sig, name, target in PLATFORM_RULES:
        s = _normalize(sig)
        if s and s in nall:
            return name, ('ContentProducer' if s in np else 'ProduceID')

    # 生产者编码前缀规则（ContentProducer 开头匹配）
    for prefix, name in PRODUCER_PREFIX_RULES:
        if producer.startswith(prefix):
            return name, 'ContentProducer'

    return None, None


# ═══════════════════════════════════════════════════════════════
# 阶段二主函数
# ═══════════════════════════════════════════════════════════════

def detect_tc260(image_path):
    """
    TC260:AIGC 隐式标识检测（阶段二，国内平台 AI 生成强证据）

    判定逻辑：
        1. 遍历全部元数据字段值，提取 TC260:AIGC 隐式标识（两种形式）
        2. 校验必备字段（Label / ContentProducer / ProduceID）完整度
        3. 必备字段命中 ≥2 且含 Label=AI 生成 → detected=True，confidence 0.93~0.97
        4. 仅字节级命中命名空间但解析失败 → 弱证据，detected=False（不误判）

    返回 dict：
        detected, source, evidence_type, confidence, matched_field,
        matched_value, detail, fields, format, producer, produce_id,
        label_is_ai, byte_hit, all_hits
    """
    try:
        texts, raw, fmt, png_chunks = _collect_metadata_texts(image_path)
        all_hits = []
        best = None  # 字段完整度最高的命中

        # ── 1. 扫描所有文本字段 ──
        for field, value in texts:
            payload, form = _extract_tc260_payload(value)
            if payload is None:
                continue
            fields = _parse_nested_aigc(payload)
            if not fields:
                continue
            n, present = _field_completeness(fields)
            hit = {
                'field': field, 'form': form, 'fields': fields,
                'completeness': n, 'present': present,
                'value': value[:500],
            }
            all_hits.append(hit)
            if best is None or n > best['completeness']:
                best = hit

        # ── 2. 字节级兜底（PNG iTXt / 未解析路径等）──
        byte_hit = []
        lower = raw.lower()
        for marker in BYTE_MARKERS:
            if marker in lower:
                byte_hit.append(marker.decode('latin-1'))

        if not all_hits and byte_hit:
            return {
                'detected': False,
                'source': None,
                'evidence_type': 'tc260_namespace_only',
                'confidence': 0.25,
                'matched_field': None,
                'matched_value': None,
                'detail': f'图片字节中存在 {", ".join(byte_hit)} 命名空间标记，但未解析出完整隐式标识字段（可能已部分剥离或被压缩）',
                'fields': {},
                'format': None,
                'producer': None,
                'produce_id': None,
                'label_is_ai': False,
                'byte_hit': byte_hit,
                'all_hits': [],
            }

        # ── 3. 判定：必备字段完整度 ──
        if best:
            fields = best['fields']
            n = best['completeness']
            present = best['present']
            label = fields.get('Label', '')
            label_is_ai = _normalize(label) in {_normalize(x) for x in LABEL_AI_GENERATED}
            platform, matched_key = _infer_platform(fields)
            producer = fields.get('ContentProducer', '')
            produce_id = fields.get('ProduceID', '')

            # 强证据：必备字段 ≥2 且 Label 为 AI 生成
            if n >= 2 and label_is_ai:
                confidence = 0.97 if n == 3 else 0.93
                source = platform or (producer or '国内 AI 平台')
                detail = (f'检测到国标 GB 45438 TC260:AIGC 隐式标识（{best["form"]}，位于 {best["field"]}），'
                          f'字段完整度 {n}/3（{"、".join(present)}），Label={label} 表明内容由 AI 生成')
                if platform and matched_key:
                    detail += f'，内容提供者特征指向「{platform}」'
            # 疑似：有必备字段但 Label 语义非 AI 生成（如合成/编辑类）
            elif n >= 2:
                confidence = 0.75
                source = platform or (producer or '国内 AI 平台')
                detail = (f'检测到 TC260:AIGC 隐式标识（{best["form"]}，位于 {best["field"]}），'
                          f'但 Label={label or "空"} 不直接声明「AI 生成」'
                          f'（GB 45438 中 Label 可能标记为合成/编辑/其他），判定为疑似 AI 生成')
            else:
                confidence = 0.50
                source = None
                detail = (f'检测到 TC260:AIGC 隐式标识残留字段（{best["field"]}），'
                          f'但必备字段仅命中 {n}/3（{"、".join(present) or "无"}），'
                          f'标识可能被部分剥离，仅作弱参考')

            return {
                'detected': n >= 2 and label_is_ai,
                'source': source,
                'evidence_type': 'tc260_aigc' if n >= 2 else 'tc260_partial',
                'confidence': confidence,
                'matched_field': best['field'],
                'matched_value': best['value'],
                'detail': detail,
                'fields': fields,
                'format': best['form'],
                'producer': producer or None,
                'produce_id': produce_id or None,
                'label_is_ai': label_is_ai,
                'byte_hit': byte_hit,
                'all_hits': all_hits,
            }

        return {
            'detected': False,
            'source': None,
            'evidence_type': None,
            'confidence': 0.0,
            'matched_field': None,
            'matched_value': None,
            'detail': '未检测到 GB 45438 TC260:AIGC 隐式标识',
            'fields': {},
            'format': None,
            'producer': None,
            'produce_id': None,
            'label_is_ai': False,
            'byte_hit': byte_hit,
            'all_hits': [],
        }
    except Exception as e:
        return {
            'detected': False,
            'source': None,
            'evidence_type': None,
            'confidence': 0.0,
            'matched_field': None,
            'matched_value': None,
            'detail': f'TC260 隐式标识检测异常: {e}',
            'fields': {},
            'format': None,
            'producer': None,
            'produce_id': None,
            'label_is_ai': False,
            'byte_hit': [],
            'all_hits': [],
        }


if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    if len(sys.argv) < 2:
        print('用法: python tc260_detector.py <图片路径>')
        sys.exit(1)
    r = detect_tc260(sys.argv[1])
    print('=== 阶段二：TC260:AIGC 隐式标识检测 ===')
    print(f"detected      : {r['detected']}")
    print(f"source        : {r['source']}")
    print(f"evidence_type : {r['evidence_type']}")
    print(f"confidence    : {r['confidence']}")
    print(f"matched_field : {r['matched_field']}")
    print(f"format        : {r['format']}")
    print(f"label_is_ai   : {r['label_is_ai']}")
    print(f"producer      : {r['producer']}")
    print(f"produce_id    : {r['produce_id']}")
    print(f"fields        : {r['fields']}")
    print(f"detail        : {r['detail']}")
    if r.get('byte_hit'):
        print(f"byte_hit      : {r['byte_hit']}")
