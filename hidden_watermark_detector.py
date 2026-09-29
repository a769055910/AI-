# -*- coding: utf-8 -*-
"""
隐藏水印检测模块（零模型、零 API、本地 CPU 毫秒级）

阶段一：国外平台元数据 / C2PA / PNG 生图参数检测（强证据，命中即可判定 AI 生成）
    - detect_metadata_watermark()  —— 元数据/C2PA/PNG 参数检测（本阶段主体）
后续阶段（暂未实现，占位）：
    - detect_lsb_steganography()   —— LSB 最低位隐写检测（阶段三）
    - detect_frequency_watermark() —— 频域水印异常粗检（阶段四）
    - detect_hidden_watermark()    —— 统一入口（阶段五汇总）

设计原则：
    - 只做「单向强证据」：命中 → 判 AI 生成；未命中 → 不代表不是 AI（元数据可被抹除/截图）。
    - 国外平台字段名公开且确定，误报率极低（锚定平台写入的确切字段与内容格式）。

依赖：Pillow（easyocr 已间接依赖）
"""

import io
import json
import re
import zlib

from visible_watermark_detector import _normalize


# ═══════════════════════════════════════════════════════════════
# 基础字节 / chunk 解析工具
# ═══════════════════════════════════════════════════════════════
def _load_image_bytes(path):
    """读取文件原始字节（兼容 Windows 中文路径）"""
    with open(path, 'rb') as f:
        return f.read()


def _read_png_chunks(raw):
    """解析 PNG chunk 列表，返回 [(ctype, cdata)]"""
    chunks = []
    if raw[:8] != b'\x89PNG\r\n\x1a\n':
        return chunks
    pos = 8
    while pos + 8 <= len(raw):
        length = int.from_bytes(raw[pos:pos + 4], 'big')
        ctype = raw[pos + 4:pos + 8].decode('latin-1')
        cdata = raw[pos + 8:pos + 8 + length]
        chunks.append((ctype, cdata))
        pos += 12 + length
        if ctype == 'IEND':
            break
    return chunks


def _parse_png_text_chunks(raw):
    """解析 PNG 文本类 chunk（tEXt/iTXt/zTXt），返回 [(keyword, value)]"""
    results = []
    for ctype, cdata in _read_png_chunks(raw):
        if ctype == 'tEXt':
            idx = cdata.find(b'\x00')
            if idx != -1:
                kw = cdata[:idx].decode('latin-1', 'ignore')
                val = cdata[idx + 1:].decode('utf-8', 'ignore')
                results.append((kw, val))
        elif ctype == 'iTXt':
            # iTXt: keyword \0 compflag compmethod \0 langtag \0 translated \0 text
            idx = cdata.find(b'\x00')
            if idx != -1:
                kw = cdata[:idx].decode('latin-1', 'ignore')
                rest = cdata[idx + 1:]
                # 未压缩时，文本从第 3 个 \0 之后开始
                parts = rest.split(b'\x00', 2)
                if len(parts) == 3:
                    val = parts[2].decode('utf-8', 'ignore')
                else:
                    val = rest.decode('utf-8', 'ignore')
                results.append((kw, val))
        elif ctype == 'zTXt':
            # zTXt: keyword \0 compmethod <zlib data>
            idx = cdata.find(b'\x00')
            if idx != -1:
                kw = cdata[:idx].decode('latin-1', 'ignore')
                comp_data = cdata[idx + 2:]  # 跳过 \0 和 compmethod(1字节)
                try:
                    val = zlib.decompress(comp_data).decode('utf-8', 'ignore')
                    results.append((kw, val))
                except Exception:
                    pass
    return results


def _extract_xmp_fields(raw):
    """从原始字节中提取 XMP 包内的关键字段，返回 [(field, value)]"""
    fields = []
    # 定位 XMP 包（JPEG APP1 / PNG iTXt / WebP chunk 内都可能存在）
    m = re.search(rb'<x:xmpmeta.*?</x:xmpmeta>', raw, re.DOTALL)
    if not m:
        m = re.search(rb'<rdf:RDF.*?</rdf:RDF>', raw, re.DOTALL)
    if not m:
        return fields
    segment = m.group(0)

    def _dec(v):
        try:
            return v.decode('utf-8', 'ignore').strip()
        except Exception:
            return str(v)

    # CreatorTool / Generator：属性形式 xmp:CreatorTool="..." 与元素形式 <xmp:CreatorTool>...</xmp:CreatorTool>
    attr_pat = rb'(?:xmp:|photoshop:)?(CreatorTool|Generator)\s*=\s*"([^"]*)"'
    for name, val in re.findall(attr_pat, segment):
        fields.append((f'xmp:{name.decode("latin-1")}', _dec(val)))
    elem_pat = rb'<(?:xmp:|photoshop:)?(CreatorTool|Generator)[^>]*>\s*([^<]+?)\s*</'
    for name, val in re.findall(elem_pat, segment):
        fields.append((f'xmp:{name.decode("latin-1")}', _dec(val)))

    # digitalSourceType（C2PA）：属性形式
    for v in re.findall(rb'(?:c2pa:)?digitalSourceType\s*=\s*"([^"]*)"', segment):
        fields.append(('digitalSourceType', _dec(v)))

    # dc:description / dc:creator（含 rdf:li / rdf:Alt）
    for v in re.findall(rb'<dc:(?:description|creator)[^>]*>\s*<rdf:(?:li|Alt)[^>]*>([^<]*)', segment):
        fields.append(('dc', _dec(v)))
    return fields


# ═══════════════════════════════════════════════════════════════
# 强证据规则库（国外平台，公开且确定）
# ═══════════════════════════════════════════════════════════════

# 1) C2PA / Content Credentials 标记
#    注意：不用过短的 'cai'（3 字母在 XMP/文本里极易误命中，如 xml 命名空间）
C2PA_BYTE_MARKERS = [
    b'c2pa', b'contentcredentials', b'content credentials',
    b'urn:c2pa', b'c2pa.org', b'c2pa.manifest',
]
# C2PA 里「数字来源类型 = AI 生成」的权威值（DALL·E / Firefly 等写入）
C2PA_AI_SOURCE = ['trainedalgorithmicmedia']

# 2) 生图软件指纹（用于 EXIF Software / XMP CreatorTool / Generator 等「软件类」字段）
#    value 为归一化（无空格小写）后的匹配串
SOFTWARE_SIGNATURES = {
    'Stable Diffusion': [
        'stablediffusion', 'stable-diffusion', 'automatic1111',
        'comfyui', 'invokeai', 'stabilityai',
    ],
    'Midjourney': ['midjourney'],
    'Adobe Firefly': ['adobefirefly', 'firefly'],
    'DALL·E': ['dall-e', 'dall·e', 'dalle', 'openai', 'chatgpt'],
    'Google Imagen/Gemini': ['imagen', 'gemini'],
}

# 3) Stable Diffusion 写入 PNG tEXt 的「parameters」参数指纹
#    （归一化后无空格小写匹配，如 'CFG scale:' -> 'cfgscale:'）
SD_PARAM_KEYS = [
    'Steps:', 'Sampler:', 'CFG scale:', 'Model hash:', 'Model:',
    'Denoising strength:', 'ADetailer', 'Clip skip:', 'Lora hashes:',
    'TI hashes:', 'VAE hash:', 'ENSD:',
]
# 极其独特、几乎只有 SD 会写的 key（单条命中即可判定）
SD_UNIQUE_KEYS = ['Model hash:', 'CFG scale:', 'Denoising strength:', 'ADetailer', 'VAE hash:']

# 4) Midjourney 写入 PNG chunk 的 JSON 字段指纹
MJ_UNIQUE_KEYS = ['job_id', 'jobid']


# ═══════════════════════════════════════════════════════════════
# 证据收集
# ═══════════════════════════════════════════════════════════════
def _collect_metadata_texts(path):
    """
    收集图片元数据中的所有文本字段。
    返回 (texts, raw, fmt, png_text_chunks)
        texts:            [(field_name, value)]
        raw:              原始字节
        fmt:              Pillow 识别到的格式
        png_text_chunks:  PNG 文本 chunk [(keyword, value)]
    """
    from PIL import Image
    from PIL.ExifTags import TAGS

    raw = _load_image_bytes(path)
    texts = []
    fmt = None

    try:
        img = Image.open(io.BytesIO(raw))
        fmt = img.format

        # EXIF（Software / ImageDescription / UserComment / Make / Model / Artist 等）
        try:
            exif = img.getexif()
            if exif:
                for tag_id, value in exif.items():
                    tag_name = TAGS.get(tag_id, str(tag_id))
                    if isinstance(value, str):
                        texts.append((f'EXIF:{tag_name}', value))
                    elif isinstance(value, bytes):
                        texts.append((f'EXIF:{tag_name}', value.decode('utf-8', 'ignore')))
        except Exception:
            pass

        # Pillow info 兜底（含 PNG tEXt / 部分 XMP）
        for k, v in img.info.items():
            if isinstance(v, str):
                texts.append((f'INFO:{k}', v))
            elif isinstance(v, bytes):
                texts.append((f'INFO:{k}', v.decode('utf-8', 'ignore')))
    except Exception:
        pass

    # XMP 字段
    try:
        texts.extend(_extract_xmp_fields(raw))
    except Exception:
        pass

    # PNG 文本 chunk（更精确地拿到 keyword，如 parameters / Description / Comment）
    png_text_chunks = []
    try:
        png_text_chunks = _parse_png_text_chunks(raw)
        for kw, val in png_text_chunks:
            texts.append((f'PNG:{kw}', val))
    except Exception:
        pass

    return texts, raw, fmt, png_text_chunks


def _detect_c2pa(raw, fmt):
    """检测 C2PA / Content Credentials 标记，返回 dict"""
    found = []
    ai_source_hit = False

    # PNG caBX chunk（C2PA 专用 chunk）
    if fmt == 'PNG':
        for ctype, _cdata in _read_png_chunks(raw):
            if ctype == 'caBX':
                found.append('PNG caBX chunk (C2PA)')

    # 字节级搜索
    lower = raw.lower()
    for marker in C2PA_BYTE_MARKERS:
        if marker in lower:
            found.append(marker.decode('latin-1'))

    # digitalSourceType = trainedAlgorithmicMedia（AI 生成强证据）
    for src in C2PA_AI_SOURCE:
        if src.encode('latin-1') in lower:
            ai_source_hit = True
            found.append(f'digitalSourceType={src}')

    return {
        'present': bool(found),
        'ai_source': ai_source_hit,
        'markers': list(dict.fromkeys(found)),
    }


def _c2pa_base_result(status='absent', detail='未发现 C2PA 内容凭证'):
    """统一 C2PA 验签结果结构，确保接口调用方始终能安全读取。"""
    return {
        'sdk_available': True,
        'present': status != 'absent',
        'verification_status': status,  # absent / valid / invalid / error / unavailable
        'signature_valid': False,
        'trust_verified': False,
        'ai_source_declared': False,
        'digital_source_types': [],
        'actions': [],
        'claim_generator': None,
        'signer': None,
        'validation_state': None,
        'validation_results': [],
        'detail': detail,
    }


def _collect_c2pa_actions(manifest_store):
    """从官方 SDK 解析出的 Manifest JSON 中提取创建/编辑动作和来源类型。"""
    actions, source_types = [], []
    manifests = manifest_store.get('manifests', {}) if isinstance(manifest_store, dict) else {}
    for manifest in manifests.values():
        for assertion in manifest.get('assertions', []) or []:
            data = assertion.get('data', {}) if isinstance(assertion, dict) else {}
            for action in data.get('actions', []) or []:
                if not isinstance(action, dict):
                    continue
                source_type = str(action.get('digitalSourceType', '') or '')
                if source_type:
                    source_types.append(source_type)
                actions.append({
                    'action': action.get('action'),
                    'digital_source_type': source_type or None,
                })
    return actions, list(dict.fromkeys(source_types))


def verify_c2pa_signature(image_path):
    """使用官方 c2pa-python SDK 验证 Manifest、文件绑定和信任状态。

    不获取远端 Manifest 或 OCSP，避免上传检测时因外网阻塞；嵌入式凭证仍会
    进行签名与信任配置验证。未携带 C2PA 的图片是中性结果，不是验签失败。
    """
    try:
        from c2pa import Context, Reader
    except ImportError:
        result = _c2pa_base_result('unavailable', 'C2PA 验签组件未安装，仅执行元数据标记检测')
        result['sdk_available'] = False
        return result

    try:
        context = Context.from_dict({
            'verify': {
                'verify_after_reading': True,
                'verify_trust': True,
                'verify_timestamp_trust': True,
                'remote_manifest_fetch': False,
                'ocsp_fetch': False,
            }
        })
        reader = Reader(str(image_path), context=context)
        manifest_store = json.loads(reader.json())
        validation_state = str(reader.get_validation_state())
        validation_results = reader.get_validation_results()
    except Exception as e:
        message = str(e)
        if 'ManifestNotFound' in message or 'no JUMBF data found' in message:
            return _c2pa_base_result('absent', '未发现可验证的 C2PA 内容凭证')
        result = _c2pa_base_result('invalid', f'C2PA 凭证存在但验签未通过：{message[:240]}')
        result['present'] = True
        return result

    active_id = manifest_store.get('active_manifest')
    active = (manifest_store.get('manifests', {}) or {}).get(active_id, {})
    actions, source_types = _collect_c2pa_actions(manifest_store)
    state_text = validation_state.lower()
    try:
        results_text = json.dumps(validation_results, ensure_ascii=False, default=str).lower()
    except Exception:
        results_text = str(validation_results).lower()

    has_error = any(token in state_text or token in results_text for token in ('invalid', 'error', 'fail'))
    has_untrusted = 'untrusted' in state_text or 'untrusted' in results_text
    signature_valid = (not has_error) and ('valid' in state_text or 'trusted' in state_text)
    trust_verified = signature_valid and not has_untrusted
    status = 'valid' if trust_verified else 'invalid'
    ai_source = any('trainedalgorithmicmedia' in value.lower() for value in source_types)
    signer = active.get('signature_info', {}) if isinstance(active, dict) else {}

    result = _c2pa_base_result(
        status,
        ('C2PA 签名、文件绑定和信任状态验证通过'
         if trust_verified else 'C2PA 凭证存在，但签名、绑定或信任链验证未通过')
    )
    result.update({
        'present': True,
        'signature_valid': signature_valid,
        'trust_verified': trust_verified,
        'ai_source_declared': ai_source,
        'digital_source_types': source_types,
        'actions': actions[:20],
        'claim_generator': active.get('claim_generator') if isinstance(active, dict) else None,
        'signer': {
            'issuer': signer.get('issuer'),
            'time': signer.get('time'),
            'alg': signer.get('alg'),
        } if isinstance(signer, dict) else None,
        'validation_state': validation_state,
        'validation_results': validation_results,
    })
    return result


def _match_software_signature(value):
    """在「软件类」字段值中匹配生图软件指纹，返回平台名或 None"""
    v = _normalize(value)
    if not v:
        return None
    for platform, sigs in SOFTWARE_SIGNATURES.items():
        for sig in sigs:
            if sig in v:
                return platform
    return None


def _match_sd_parameters(value):
    """检测 SD 参数指纹，返回命中的 key 列表"""
    t = _normalize(value)
    if not t:
        return []
    hits = []
    for k in SD_PARAM_KEYS:
        if _normalize(k) in t:
            hits.append(k)
    return hits


def _match_midjourney(value):
    """检测 MJ JSON 字段指纹，返回命中的 key 列表"""
    t = _normalize(value)
    if not t:
        return []
    return [k for k in MJ_UNIQUE_KEYS if k in t]


# ═══════════════════════════════════════════════════════════════
# 阶段一主函数：元数据 / C2PA / PNG 参数检测
# ═══════════════════════════════════════════════════════════════
def detect_metadata_watermark(image_path):
    """
    国外平台元数据 / C2PA / PNG 生图参数检测（强证据）

    判定优先级（命中即返回 detected=True）：
        1. C2PA digitalSourceType = trainedAlgorithmicMedia（AI 生成，置信最高）
        2. EXIF/XMP 软件字段命中生图软件指纹（Stable Diffusion / Midjourney / Firefly / DALL·E）
        3. PNG parameters 命中 SD 参数指纹组合（Steps/Sampler/CFG scale/Model hash 等）
        4. PNG JSON 命中 Midjourney 的 job_id 字段

    返回 dict：
        detected, source, evidence_type, confidence,
        matched_field, matched_value, detail, c2pa, all_evidence
    """
    try:
        texts, raw, fmt, png_chunks = _collect_metadata_texts(image_path)
        c2pa = _detect_c2pa(raw, fmt)
        c2pa_verification = verify_c2pa_signature(image_path)
        c2pa['verification'] = c2pa_verification
        c2pa['present'] = bool(c2pa['present'] or c2pa_verification.get('present'))
        evidence = []

        # ── 1. 已验签 C2PA 的 AI 来源声明 ──
        # 只有 Manifest 的签名、文件绑定及信任状态均通过时，才作为 AI 生成强证据。
        if c2pa_verification.get('trust_verified') and c2pa_verification.get('ai_source_declared'):
            return {
                'detected': True,
                'source': 'C2PA/Content Credentials',
                'evidence_type': 'c2pa_verified_ai_source',
                'confidence': 0.99,
                'matched_field': 'C2PA digitalSourceType',
                'matched_value': 'trainedAlgorithmicMedia',
                'detail': 'C2PA 内容凭证已完成签名与信任验证，且声明 digitalSourceType=trainedAlgorithmicMedia（可信 AI 生成来源）',
                'c2pa': c2pa,
                'all_evidence': evidence,
            }

        # ── 2. 生图软件指纹（仅匹配「软件类」字段，避免在 prompt 中误报） ──
        software_fields = ('EXIF:Software', 'xmp:CreatorTool', 'xmp:Generator',
                           'INFO:Software', 'INFO:software', 'INFO:CreatorTool')
        for field, value in texts:
            if field in software_fields:
                platform = _match_software_signature(value)
                if platform:
                    evidence.append({'type': 'software', 'field': field, 'value': value[:200], 'platform': platform})
                    return {
                        'detected': True,
                        'source': platform,
                        'evidence_type': 'software',
                        'confidence': 0.90,
                        'matched_field': field,
                        'matched_value': value[:300],
                        'detail': f"图片元数据「{field}」中出现 AI 生图软件指纹「{platform}」",
                        'c2pa': c2pa,
                        'all_evidence': evidence,
                    }

        # ── 3. Stable Diffusion 参数指纹（不限字段，参数组合本身极独特） ──
        for field, value in texts:
            hits = _match_sd_parameters(value)
            if not hits:
                continue
            unique_hits = [h for h in hits if h in SD_UNIQUE_KEYS]
            # 判定：命中任意「极独特 key」，或命中 ≥2 个普通 key
            if unique_hits or len(hits) >= 2:
                evidence.append({'type': 'sd_parameters', 'field': field, 'hits': hits})
                return {
                    'detected': True,
                    'source': 'Stable Diffusion',
                    'evidence_type': 'sd_parameters',
                    'confidence': 0.88,
                    'matched_field': field,
                    'matched_value': value[:300],
                    'detail': f"图片元数据「{field}」含 Stable Diffusion 生图参数：{', '.join(hits[:6])}",
                    'c2pa': c2pa,
                    'all_evidence': evidence,
                }

        # ── 4. Midjourney JSON 指纹 ──
        for field, value in texts:
            hits = _match_midjourney(value)
            if hits:
                evidence.append({'type': 'midjourney', 'field': field, 'hits': hits})
                return {
                    'detected': True,
                    'source': 'Midjourney',
                    'evidence_type': 'midjourney',
                    'confidence': 0.88,
                    'matched_field': field,
                    'matched_value': value[:300],
                    'detail': f"图片元数据「{field}」含 Midjourney 生成字段（{', '.join(hits)}）",
                    'c2pa': c2pa,
                    'all_evidence': evidence,
                }

        # ── 5. C2PA 状态说明：未验签的字段绝不作为 AI 生成直判依据 ──
        if c2pa_verification.get('present') and c2pa_verification.get('trust_verified'):
            return {
                'detected': False,
                'source': 'C2PA/Content Credentials',
                'evidence_type': 'c2pa_verified_provenance',
                'confidence': 0.0,
                'matched_field': None,
                'matched_value': None,
                'detail': 'C2PA 内容凭证验签通过，但未声明 AI 全图生成；将作为深度伪造和篡改研判的来源证据展示。',
                'c2pa': c2pa,
                'all_evidence': evidence,
            }

        # 仅存在 C2PA 标记或未验签 AI 字段 → 弱证据，不判定。
        if c2pa['present']:
            raw_ai_hint = c2pa['ai_source'] or c2pa_verification.get('ai_source_declared')
            return {
                'detected': False,
                'source': None,
                'evidence_type': 'c2pa_unverified_ai_claim' if raw_ai_hint else 'c2pa_marker_only',
                'confidence': 0.30,
                'matched_field': None,
                'matched_value': None,
                'detail': ('存在 AI 来源声明，但 C2PA 签名/信任验证未通过，不作为 AI 生成直判依据'
                           if raw_ai_hint else '存在 C2PA 标记，但无可验证的 AI 来源声明（不足以判定）'),
                'c2pa': c2pa,
                'all_evidence': evidence,
            }

        return {
            'detected': False,
            'source': None,
            'evidence_type': None,
            'confidence': 0.0,
            'matched_field': None,
            'matched_value': None,
            'detail': '未发现国外 AI 平台的元数据 / C2PA / PNG 生图参数强证据',
            'c2pa': c2pa,
            'all_evidence': evidence,
        }
    except Exception as e:
        return {
            'detected': False,
            'source': None,
            'evidence_type': None,
            'confidence': 0.0,
            'matched_field': None,
            'matched_value': None,
            'detail': f'元数据检测异常: {e}',
            'c2pa': {
                'present': False, 'ai_source': False, 'markers': [],
                'verification': _c2pa_base_result('error', 'C2PA 验签过程异常'),
            },
            'all_evidence': [],
        }


# ═══════════════════════════════════════════════════════════════
# 调试工具：打印图片全部元数据字段（阶段二实测国内字段时也会复用）
# ═══════════════════════════════════════════════════════════════
def dump_metadata(image_path):
    """打印图片的全部元数据文本字段 + PNG chunk + C2PA 标记，供实测排查"""
    texts, raw, fmt, png_chunks = _collect_metadata_texts(image_path)
    c2pa = _detect_c2pa(raw, fmt)
    print('=' * 60)
    print(f'文件: {image_path}')
    print(f'格式: {fmt}')
    print(f'C2PA: {c2pa}')
    print('-' * 60)
    if not texts:
        print('(无元数据文本字段)')
    for field, value in texts:
        # 截断超长字段，避免刷屏
        shown = value if len(value) <= 400 else value[:400] + f'... (共 {len(value)} 字符)'
        print(f'[{field}]\n  {shown}')
    print('=' * 60)


# ═══════════════════════════════════════════════════════════════
# 统一入口（阶段五完整版；当前含阶段一国外元数据 + 阶段二国内 TC260）
# ═══════════════════════════════════════════════════════════════
def detect_hidden_watermark(image_path):
    """
    隐藏水印统一入口（阶段一国外元数据/C2PA + 阶段二国内 TC260:AIGC 隐式标识）

    判定逻辑：
        - 任一阶段 detected=True → 整体判定 AI 生成（取置信度更高者描述）
        - 全部未命中 → 无隐藏水印强证据（不排除被剥离/截图）
        - 仅存在弱证据（C2PA 标记 / TC260 命名空间残留）→ suspicious=True 提示
    """
    from tc260_detector import detect_tc260  # 延迟导入，避免循环依赖

    metadata = detect_metadata_watermark(image_path)
    tc260 = detect_tc260(image_path)
    c2pa_verification = metadata.get('c2pa', {}).get('verification', _c2pa_base_result())

    if metadata['detected'] and tc260['detected']:
        # 双命中：取置信度更高者为 source 描述，detail 合并
        hi = metadata if metadata['confidence'] >= tc260['confidence'] else tc260
        lo = tc260 if hi is metadata else metadata
        return {
            'detected': True,
            'suspicious': False,
            'source': hi['source'],
            'confidence': hi['confidence'],
            'detail': f"[阶段一·国外元数据] {metadata['detail']} ｜ [阶段二·国内TC260] {tc260['detail']}",
            'methods': ['metadata', 'tc260'],
            'metadata': metadata,
            'tc260': tc260,
            'c2pa_verification': c2pa_verification,
            'lsb': None,
            'frequency': None,
        }
    elif metadata['detected']:
        return {
            'detected': True,
            'suspicious': False,
            'source': metadata['source'],
            'confidence': metadata['confidence'],
            'detail': f"[阶段一·国外元数据] {metadata['detail']} ｜ [阶段二·国内TC260] {tc260['detail']}",
            'methods': ['metadata', 'tc260'],
            'metadata': metadata,
            'tc260': tc260,
            'c2pa_verification': c2pa_verification,
            'lsb': None,
            'frequency': None,
        }
    elif tc260['detected']:
        return {
            'detected': True,
            'suspicious': False,
            'source': tc260['source'],
            'confidence': tc260['confidence'],
            'detail': f"[阶段一·国外元数据] {metadata['detail']} ｜ [阶段二·国内TC260] {tc260['detail']}",
            'methods': ['metadata', 'tc260'],
            'metadata': metadata,
            'tc260': tc260,
            'c2pa_verification': c2pa_verification,
            'lsb': None,
            'frequency': None,
        }

    # 均未命中：汇总弱证据提示（C2PA 标记 / TC260 命名空间残留）
    suspicious = bool(metadata.get('c2pa', {}).get('present')) or bool(tc260.get('byte_hit'))
    pieces = []
    if metadata.get('evidence_type'):
        pieces.append(f"[阶段一] {metadata['detail']}")
    if tc260.get('evidence_type'):
        pieces.append(f"[阶段二] {tc260['detail']}")
    detail = ' ｜ '.join(pieces) if pieces else '未检测到隐藏水印 / 元数据 AI 标识强证据'
    confidence = max(metadata['confidence'], tc260['confidence'])
    return {
        'detected': False,
        'suspicious': suspicious,
        'source': None,
        'confidence': confidence,
        'detail': detail,
        'methods': ['metadata', 'tc260'],
        'metadata': metadata,
        'tc260': tc260,
        'c2pa_verification': c2pa_verification,
        'lsb': None,
        'frequency': None,
    }


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('用法: python hidden_watermark_detector.py <图片路径> [--dump]')
        sys.exit(1)

    img_path = sys.argv[1]
    do_dump = '--dump' in sys.argv

    if do_dump:
        dump_metadata(img_path)

    result = detect_hidden_watermark(img_path)
    print('\n=== 隐藏水印统一入口结果（阶段一国外元数据 + 阶段二国内TC260） ===')
    print(f"detected  : {result['detected']}")
    print(f"suspicious: {result['suspicious']}")
    print(f"source    : {result['source']}")
    print(f"confidence: {result['confidence']}")
    print(f"methods   : {result['methods']}")
    print(f"detail    : {result['detail']}")

    m, t = result['metadata'], result['tc260']
    print(f"\n[阶段一 国外元数据] detected={m['detected']} evidence={m['evidence_type']} conf={m['confidence']}")
    print(f"[阶段二 国内TC260]   detected={t['detected']} evidence={t['evidence_type']} conf={t['confidence']}")
    if t.get('source'):
        print(f"  平台: {t['source']}  Label=AI生成: {t['label_is_ai']}  ContentProducer: {t['producer']}")
        print(f"  ProduceID: {t['produce_id']}")
