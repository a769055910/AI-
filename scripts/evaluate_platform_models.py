"""Evaluate collected platform originals using the project's current AI-image route.

No resize/re-encode, watermark shortcut, training, or threshold tuning. The last
definitions of the two app fusion functions are executed verbatim from its AST,
avoiding unrelated web application/database initialization. JSONL stores every
raw response; CSV and Markdown summarize the same measurements.
"""
from __future__ import annotations

import argparse
import ast
import copy
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from camera_forensics import analyze_camera_imaging
from image_quality import validate_image_bytes
from npr_detector import npr_analyze

MODELS = {
    'npr_raw': 'NPR 原始噪声分析',
    'npr_fused': 'NPR 加入相机物证',
    'probe_dinov2': 'PROBE-DINOv2',
    'univfd': 'UnivFD',
    'final': '当前项目综合评分',
    'final_without_camera': '综合评分（移除相机校正的对照）',
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scoring_snapshot(output):
    source = (ROOT / 'app.py').read_text(encoding='utf-8-sig')
    tree = ast.parse(source)
    wanted = ('fuse_npr_with_physical_evidence', 'compute_combined_verdict')
    functions = [next(n for n in reversed(tree.body)
                      if isinstance(n, ast.FunctionDef) and n.name == name)
                 for name in wanted]
    threshold_node = next(n for n in tree.body if isinstance(n, ast.Assign)
                          and any(isinstance(t, ast.Name) and t.id == 'AI_GENERATED_THRESHOLD'
                                  for t in n.targets))
    namespace = {}
    snapshot = ast.Module(body=[threshold_node, *functions], type_ignores=[])
    exec(compile(snapshot, str(ROOT / 'app.py'), 'exec'), namespace)
    (output / 'scoring_snapshot.py').write_text(ast.unparse(snapshot) + '\n', encoding='utf-8')
    return namespace


def samples_from(manifest, controls=False):
    accepted, excluded = [], []
    with manifest.open(encoding='utf-8-sig', newline='') as handle:
        for row in csv.DictReader(handle):
            if controls and row['expected_label'] != 'real':
                continue
            path = manifest.parent / row['file']
            quality = validate_image_bytes(path.read_bytes())
            actual_hash = digest(path)
            if row.get('sha256') and actual_hash != row['sha256']:
                raise RuntimeError(f'Source hash mismatch: {path.name}')
            item = {**row, 'image_path': str(path.resolve()), 'sha256': actual_hash,
                    'quality': quality, 'platform': 'real_control' if controls else row['platform'],
                    'manifest': str(manifest.resolve())}
            (accepted if quality['valid'] else excluded).append(item)
    return accepted, excluded


def valid_score(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_sample(sample, api, session, ns):
    path = Path(sample['image_path'])
    result = {'sample': sample, 'scores': {key: None for key in MODELS}, 'timings': {}, 'errors': []}
    start = time.perf_counter()
    raw = camera = fused = None
    for key, action in [('npr', lambda: npr_analyze(str(path))),
                        ('camera', lambda: analyze_camera_imaging(str(path), include_local=False))]:
        before = time.perf_counter()
        try:
            value = action()
            result[key] = value
            if key == 'npr':
                if value.get('verdict') == 'error' or not valid_score(value.get('score')):
                    raise RuntimeError('No valid NPR result')
                raw = value
                result['scores']['npr_raw'] = raw['score']
            else:
                camera = value
                if not value.get('available'):
                    result['errors'].append('Camera evidence unavailable')
        except Exception as exc:
            result['errors'].append(f'{key}: {type(exc).__name__}: {exc}')
        result['timings'][key] = round(time.perf_counter() - before, 4)
    if raw is not None:
        fused = ns['fuse_npr_with_physical_evidence'](copy.deepcopy(raw), camera)
        result['npr_fused'] = fused
        result['scores']['npr_fused'] = fused['score']
    before = time.perf_counter()
    try:
        with path.open('rb') as handle:
            response = session.post(api.rstrip('/') + '/detect',
                                    files={'file': (path.name, handle, 'application/octet-stream')},
                                    data={'models': 'ai_generated'}, timeout=(15, 300))
        response.raise_for_status()
        payload = response.json()
        if payload.get('code') != 200:
            raise RuntimeError(str(payload.get('msg', payload.get('code'))))
        result['gpu_response'] = payload
        specialized = payload['data'].get('specialized_models') or {}
        for key in ('probe_dinov2', 'univfd'):
            item = specialized.get(key) or {}
            if item.get('available') is True and valid_score(item.get('ai_score')):
                result['scores'][key] = item['ai_score']
            else:
                result['errors'].append(f"{key}: unavailable ({item.get('detail', item.get('error', 'not returned'))})")
        if fused is not None:
            result['combined'] = ns['compute_combined_verdict'](fused, specialized)
            result['scores']['final'] = result['combined']['final_score']
            result['combined_without_camera'] = ns['compute_combined_verdict'](raw, specialized)
            result['scores']['final_without_camera'] = result['combined_without_camera']['final_score']
    except Exception as exc:
        result['errors'].append(f'gpu: {type(exc).__name__}: {exc}')
    result['timings']['gpu'] = round(time.perf_counter() - before, 4)
    result['timings']['total'] = round(time.perf_counter() - start, 4)
    result['source_unchanged'] = digest(path) == sample['sha256']
    if not result['source_unchanged']:
        raise RuntimeError(f'Source changed during inference: {path}')
    return result


def metrics(rows, key, threshold):
    positive = [r for r in rows if r['sample']['expected_label'] == 'ai_generated']
    negative = [r for r in rows if r['sample']['expected_label'] == 'real']
    pos = [r['scores'][key] for r in positive if valid_score(r['scores'][key])]
    neg = [r['scores'][key] for r in negative if valid_score(r['scores'][key])]
    tp, fp = sum(s >= threshold for s in pos), sum(s >= threshold for s in neg)
    tn, fn = len(neg) - fp, len(pos) - tp
    # Pairwise empirical AUC; ties count half. No fitting or threshold search.
    auc = (sum((p > n) + .5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))) if pos and neg else None
    return {'positive_total': len(positive), 'negative_total': len(negative),
            'positive_available': len(pos), 'negative_available': len(neg),
            'tp': tp, 'fn': fn, 'fp': fp, 'tn': tn,
            'recall': tp / len(pos) if pos else None,
            'fpr': fp / len(neg) if neg else None,
            'accuracy': (tp + tn) / (len(pos) + len(neg)) if pos or neg else None,
            'balanced_accuracy': .5 * (tp / len(pos) + tn / len(neg)) if pos and neg else None,
            'auc': auc, 'ai_mean': statistics.mean(pos) if pos else None,
            'real_mean': statistics.mean(neg) if neg else None}


def pct(value):
    return '—' if value is None else f'{value * 100:.1f}%'


def summarize(rows, output, threshold):
    fields = ['file', 'platform', 'expected_label', 'generator', 'sha256', *MODELS,
              'camera_trace_score', 'camera_physical_score', 'camera_metadata',
              'npr_seconds', 'camera_seconds', 'gpu_seconds', 'total_seconds', 'source_unchanged', 'errors']
    with (output / 'scores.csv').open('w', encoding='utf-8-sig', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            camera = r.get('camera') or {}
            writer.writerow({**{k: r['sample'].get(k, '') for k in fields[:5]}, **r['scores'],
                             'camera_trace_score': camera.get('camera_trace_score'),
                             'camera_physical_score': (r.get('npr_fused') or {}).get('physical_score'),
                             'camera_metadata': (camera.get('exif') or {}).get('has_camera_tags'),
                             **{f'{k}_seconds': r['timings'].get(k) for k in ('npr', 'camera', 'gpu', 'total')},
                             'source_unchanged': r['source_unchanged'], 'errors': ' | '.join(r['errors'])})
    overall = {key: metrics(rows, key, threshold) for key in MODELS}
    by_platform = {platform: {key: metrics([r for r in rows if r['sample']['platform'] == platform], key, threshold)
                              for key in MODELS} for platform in ('doubao', 'jimeng', 'qwen')}
    summary = {'threshold': threshold, 'count': len(rows), 'overall': overall, 'by_platform': by_platform}
    summary['native_neural_threshold_0_5'] = {
        key: metrics(rows, key, .5) for key in ('probe_dinov2',)
    }
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    lines = ['# 豆包、即梦、千问 AI 全图生成模型实测', '',
             f'评测时间：{datetime.now().isoformat(timespec="seconds")}；固定阈值：{threshold * 100:g} 分（得分 ≥ 阈值判 AI）。', '',
             '## 实测结论', '',
             f"- 当前综合评分检出 {overall['final']['tp']}/30 张 AI 图片，漏检 {overall['final']['fn']}/30 张；9 张真实对照误报 {overall['final']['fp']} 张。",
             f"- 综合评分按平台检出：豆包 {by_platform['doubao']['final']['tp']}/10，即梦 {by_platform['jimeng']['final']['tp']}/10，千问 {by_platform['qwen']['final']['tp']}/10。当前配置在这些平台样本上漏检较多。",
             f"- NPR 原始噪声评分检出 {overall['npr_raw']['tp']}/30；加入相机物证后检出 {overall['npr_fused']['tp']}/30。物证校正改善了 NPR 子项，但最终综合结果仍有大量漏检。",
             f"- PROBE 在这批样本中的检出率为 {pct(overall['probe_dinov2']['recall'])}；逐图有效融合权重记录在原始 JSONL 的 combined.scoring.components。",
             f"- 有 {sum(r['scores']['probe_dinov2'] >= threshold and r['scores']['final'] < threshold for r in rows if valid_score(r['scores']['probe_dinov2']) and valid_score(r['scores']['final']))} 张图片达到 PROBE 阈值，最终综合分仍低于阈值，说明高分证据被其他低分项拉低。",
             '- 当前仅评测仍启用的模型；未返回的模型记为不可用，停用的 AI vs Human 占位结果不参与评分。', '',
             '## 样本与口径', '',
             '- 主测试集：30 张公开 AI 生成图片，豆包、即梦、千问各 10 张。千问来自官方 Qwen-Image 示例；豆包、即梦依据原文章生成说明标注。',
             '- 补充对照：项目已有公开数据集中通过当前质量门槛的 9 张真实照片；其余 16 张仅因质量条件排除，未按检测分数筛选。',
             '- 使用原始下载文件；未缩放、未转码。保留 SHA-256，推理前后检查文件未改变。直接请求 GPU AI 全图生成模型，绕过水印提前定性。',
             '- NPR 和相机物证使用当前本地代码，融合使用 app.py 最后生效的函数。没有训练、调参或修改阈值。详见 run_config.json 与 scoring_snapshot.py。',
             '- 各模型统一按项目 60 分比较；原生模型标签/阈值另保存在原始 JSONL，不混用。分数不是已校准的正确概率。', '',
             '## 检出与误报', '',
             '| 模型 | AI 有效/30 | 检出/有效 | 检出率 | 真实有效/9 | 误报/有效 | 误报率 | 本样本准确率 | 平衡准确率 | AUC |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key, name in MODELS.items():
        m = overall[key]
        lines.append(f"| {name} | {m['positive_available']}/{m['positive_total']} | {m['tp']}/{m['positive_available']} | {pct(m['recall'])} | {m['negative_available']}/{m['negative_total']} | {m['fp']}/{m['negative_available']} | {pct(m['fpr'])} | {pct(m['accuracy'])} | {pct(m['balanced_accuracy'])} | {m['auc']:.3f} |" if m['auc'] is not None else f"| {name} | {m['positive_available']}/30 | — | — | {m['negative_available']}/9 | — | — | — | — | — |")
    lines += ['', '不可用模型记为缺失，不按 0 分或真实图片处理；综合评分仍按项目逻辑对可用模型归一化加权。', '',
              '## 各平台检出率与平均分', '',
              '| 模型 | 豆包检出/有效；均分 | 即梦检出/有效；均分 | 千问检出/有效；均分 | 真实照片均分 |',
              '|---|---:|---:|---:|---:|']
    for key, name in MODELS.items():
        columns = []
        for platform in by_platform:
            m = by_platform[platform][key]
            columns.append(f"{m['tp']}/{m['positive_available']}；{m['ai_mean'] * 100:.2f}" if m['ai_mean'] is not None else '不可用')
        real_mean = overall[key]['real_mean']
        lines.append(f"| {name} | {' | '.join(columns)} | {real_mean * 100:.2f}" + ' |' if real_mean is not None else f"| {name} | {' | '.join(columns)} | — |")
    lines += ['', '## 分数尺度与原生模型标签', '',
              '项目综合评分及模型投票采用 60 分，GPU 神经模型的原生 is_ai 标签采用 50 分。下面单独对照原生阈值，主表仍使用项目当前 60 分。', '',
              '| 神经模型 | 原生 50 分检出/30 | 项目 60 分检出/30 | 原生 50 分真实误报/9 |',
              '|---|---:|---:|---:|']
    for key in ('probe_dinov2',):
        native = summary['native_neural_threshold_0_5'][key]
        lines.append(f"| {MODELS[key]} | {native['tp']}/30 | {overall[key]['tp']}/30 | {native['fp']}/9 |")
    raw_ai = [r['scores']['npr_raw'] for r in rows if r['sample']['expected_label'] == 'ai_generated' and valid_score(r['scores']['npr_raw'])]
    if raw_ai:
        lines += ['', f"NPR 原始 AI 样本分数范围为 {min(raw_ai) * 100:.2f}–{max(raw_ai) * 100:.2f} 分，全部低于当前 60 分。原始分数有分组差异，但当前阈值下均漏检。项目 NPR 是本地 FFT 特征与人工映射的启发式评分，并非本次加载的预训练神经网络。"]
    radial = [(r.get('npr') or {}).get('features', {}).get('radial_variance') for r in rows]
    radial = [v for v in radial if valid_score(v)]
    if radial:
        max_contribution = min(max(radial) * 50, 1) * .25 * 100
        lines.append(f"本次所有图片的径向方差项贡献最高仅 {max_contribution:.2f} 分（该项配置上限 25 分），提示现有特征映射尺度需要单独核查。未据此调整算法。")
    lines += ['', '## 相机成像物证', '',
              '相机物证为单图 EXIF、噪声及 CFA 辅助证据，并非独立 AI 分类模型，也不是需要相机参考照片的严格 PRNU 设备指纹。因此不给它单独宣称分类准确率。', '']
    for group, label in [('ai_generated', 'AI 图片'), ('real', '真实照片')]:
        group_rows = [r for r in rows if r['sample']['expected_label'] == group]
        trace = [(r.get('camera') or {}).get('camera_trace_score') for r in group_rows]
        trace = [v for v in trace if valid_score(v)]
        physical = [(r.get('npr_fused') or {}).get('physical_score') for r in group_rows]
        physical = [v for v in physical if valid_score(v)]
        tags = sum(bool(((r.get('camera') or {}).get('exif') or {}).get('has_camera_tags')) for r in group_rows)
        if trace and physical:
            lines.append(f'- {label}：相机痕迹平均 {statistics.mean(trace) * 100:.2f} 分（越高越支持相机采集），物证 AI 可疑度平均 {statistics.mean(physical) * 100:.2f} 分，相机 EXIF 标签 {tags}/{len(group_rows)}。')
    pairs = [r for r in rows if valid_score(r['scores']['final']) and valid_score(r['scores']['final_without_camera'])]
    changed = [r for r in pairs if (r['scores']['final'] >= threshold) != (r['scores']['final_without_camera'] >= threshold)]
    lines.append(f'- 固定所有神经模型输出，仅移除相机对 NPR 的校正：综合判定发生变化 {len(changed)}/{len(pairs)} 张。')
    for r in changed:
        lines.append(f"  - {r['sample']['file']}：无相机校正 {r['scores']['final_without_camera'] * 100:.2f} → 当前 {r['scores']['final'] * 100:.2f} 分；标签 {r['sample']['expected_label']}。")
    lines += ['', '## 综合评分漏检、误报清单', '', '| 文件 | 标注 | 当前综合分 | PROBE | NPR 原始 / 相机融合 |', '|---|---|---:|---:|---:|']
    def score(r, key):
        value = r['scores'][key]
        return '—' if value is None else f'{value * 100:.2f}'
    for r in rows:
        if r['scores']['final'] is None:
            continue
        if (r['scores']['final'] >= threshold) != (r['sample']['expected_label'] == 'ai_generated'):
            lines.append(f"| {r['sample']['file']} | {r['sample']['expected_label']} | {score(r, 'final')} | {score(r, 'probe_dinov2')} | {score(r, 'npr_raw')} / {score(r, 'npr_fused')} |")
    lines += ['', '## 耗时', '']
    for key in ('npr', 'camera', 'gpu', 'total'):
        times = [r['timings'][key] for r in rows]
        lines.append(f'- {key}：平均 {statistics.mean(times):.2f} 秒，中位数 {statistics.median(times):.2f} 秒，最大 {max(times):.2f} 秒。')
    lines += ['', '## 适用范围', '',
              '这是 30 张正样本与 9 张补充对照的探索性评测，不代表生产环境整体准确率。平台示例并非随机抽样，真实对照数量少、来源与题材不同；均分、AUC、误报率均可能受来源、题材和压缩差异影响。公开图片可能经平台重编码，不等同生成器原始导出。',
              '尤其不能把缺少 EXIF、弱噪声或弱 CFA 痕迹直接当作伪造证明；本次记录的是项目现有启发式分数表现。0 次误报也不代表真实误报率为零。', '',
              '## 后续验证方向', '',
              '先扩大同来源、同题材、同压缩条件的真实与 AI 对照，划分独立的标定集和测试集，再验证 NPR 特征映射、60 分阈值及融合权重。当前结果支持优先核查漏检与分数尺度，尚不足以据此直接改阈值、改权重或宣称某模型在所有平台更准确。', '',
              '完整逐图分数：scores.csv；原始模型与物证响应：results.jsonl；统计：summary.json；质量排除记录：excluded_controls.json。', '']
    (output / 'report.md').write_text('\n'.join(lines), encoding='utf-8')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT / 'test_data/ai_platform_web_20261008/manifest.csv')
    parser.add_argument('--controls', type=Path, default=ROOT / 'test_data/ai_platform_testset/manifest.csv')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-api', default='http://127.0.0.1:6006')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'results.jsonl').exists():
        raise SystemExit('Output already contains results; use a new directory to prevent overwriting.')
    ns = scoring_snapshot(args.output)
    samples, excluded_main = samples_from(args.manifest)
    controls, excluded_controls = samples_from(args.controls, controls=True)
    if excluded_main:
        raise RuntimeError('Collected positive images unexpectedly fail current quality gates')
    samples += controls
    (args.output / 'excluded_controls.json').write_text(json.dumps(excluded_controls, ensure_ascii=False, indent=2), encoding='utf-8')
    health = requests.get(args.gpu_api.rstrip('/') + '/health', timeout=15)
    health.raise_for_status()
    config = {'started_at': datetime.now().isoformat(), 'gpu_api': args.gpu_api,
              'health': health.json(), 'threshold': ns['AI_GENERATED_THRESHOLD'],
              'preprocessing': 'Original bytes; no evaluator resize/re-encode; no watermark shortcut',
              'camera_include_local': False, 'manifests': [str(args.manifest), str(args.controls)],
              'source_hashes': {name: digest(ROOT / name) for name in ('app.py', 'npr_detector.py', 'camera_forensics.py', 'image_quality.py', 'scripts/evaluate_platform_models.py')},
              'sample_count': len(samples), 'excluded_real_controls': len(excluded_controls)}
    (args.output / 'run_config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    rows = []
    with requests.Session() as session, (args.output / 'results.jsonl').open('w', encoding='utf-8') as handle:
        for index, sample in enumerate(samples, 1):
            result = run_sample(sample, args.gpu_api, session, ns)
            rows.append(result)
            handle.write(json.dumps(result, ensure_ascii=False) + '\n')
            handle.flush()
            scores = result['scores']
            print(f"[{index}/{len(samples)}] {sample['file']} final={scores['final']} PROBE={scores['probe_dinov2']} {result['timings']['total']:.2f}s", flush=True)
            actual_errors = [e for e in result['errors'] if not e.startswith('univfd:')]
            if actual_errors:
                print('ERROR: ' + ' | '.join(actual_errors), flush=True)
    summary = summarize(rows, args.output, ns['AI_GENERATED_THRESHOLD'])
    print(json.dumps(summary['overall'], ensure_ascii=False, indent=2), flush=True)
    print(f'Artifacts: {args.output.resolve()}', flush=True)


if __name__ == '__main__':
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    main()
