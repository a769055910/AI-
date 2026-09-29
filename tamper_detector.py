"""
图像篡改检测器 — NPR + SRM + DCT 三维滑窗分析
针对「AI 局部编辑/修改」场景，三个数学上正交的维度做滑窗分析：
  NPR  (空域纹理统计)  → AI 生成的频域异常纹理
  SRM  (多方向噪声残差)→ 缺失的相机传感器/ISP 管线痕迹
  DCT  (频域压缩痕迹)  → DCT 系数分布偏离自然图像统计
"""
import os
import requests
import cv2
import numpy as np
from scipy import stats as sp_stats
from trufor_client import analyze_trufor

# GPU 服务器 DCT 分析端点（SSH 隧道映射到本机 6006 端口）
GPU_DCT_API = "http://127.0.0.1:6006/dct_analyze"
GPU_SRM_API = "http://127.0.0.1:6006/srm_analyze"

# ═══════════════════════════════════════════════
# 工具
# ═══════════════════════════════════════════════

def _load_image(path, rgb=False):
    data = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None: return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if rgb else img

def _build_windows(h, w, win_size=128, stride=64):
    """生成滑窗坐标 [(y0,y1,x0,x1), ...]"""
    win = min(win_size, h, w)
    st = max(stride, win // 4)
    ws = []
    for y0 in range(0, h - win + 1, st):
        for x0 in range(0, w - win + 1, st):
            ws.append((y0, y0 + win, x0, x0 + win))
    if h > win:
        for x0 in range(0, w - win + 1, st):
            ws.append((h - win, h, x0, x0 + win))
    if w > win:
        for y0 in range(0, h - win + 1, st):
            ws.append((y0, y0 + win, w - win, w))
    if h > win and w > win:
        ws.append((h - win, h, w - win, w))
    return list(set(ws))

# ═══════════════════════════════════════════════
# 维度一：NPR 滑窗
# ═══════════════════════════════════════════════

def _npr_features(block):
    """灰度块 NPR 频域特征"""
    hb, wb = block.shape
    ks = max(3, min(hb, wb) // 30 | 1)
    if ks % 2 == 0: ks += 1
    blurred = cv2.GaussianBlur(block, (ks, ks), 0)
    noise = block.astype(np.float32) - blurred.astype(np.float32)

    fft = np.fft.fft2(noise)
    ffts = np.fft.fftshift(fft)
    mag = np.abs(ffts)
    total = hb * wb
    cy, cx = hb // 2, wb // 2

    # 峰值密度
    nz = mag[mag > 0]
    med = float(np.median(nz)) if len(nz) > 0 else 1.0
    peak_r = float(np.sum(mag > med * 8.0)) / total

    # 频谱平坦度
    if len(nz) > 0:
        gm = np.exp(np.mean(np.log(nz + 1e-10)))
        am = float(np.mean(mag))
        flatness = gm / am if am > 1e-10 else 0.0
    else:
        flatness = 0.0

    # 高频能量
    y, x = np.ogrid[:hb, :wb]
    dist = np.sqrt((y - cy)**2 + (x - cx)**2)
    rh = min(hb, wb) // 4
    hf_ratio = float(np.sum(mag[dist > rh])) / (float(np.sum(mag)) + 1e-10)

    # 径向方差
    mr = min(hb, wb) // 2
    nb = 30
    rad = np.zeros(nb)
    for i in range(nb):
        r0, r1 = mr * i / nb, mr * (i + 1) / nb
        msk = (dist >= r0) & (dist < r1)
        rad[i] = float(np.sum(mag[msk]))
    rad /= (np.sum(rad) + 1e-10)
    r_var = float(np.var(rad))

    return {"peak_ratio": peak_r, "spectral_flatness": flatness,
            "high_freq_ratio": hf_ratio, "radial_variance": r_var}

def _npr_score(f):
    p = np.clip(f["peak_ratio"] * 100.0, 0, 1)
    s = np.clip((1 - f["spectral_flatness"]) * 0.8, 0, 1)
    h = np.clip(f["high_freq_ratio"] * 1.5, 0, 1)
    v = np.clip(f["radial_variance"] * 50.0, 0, 1)
    return float(np.clip(p*0.30 + s*0.25 + h*0.20 + v*0.25, 0, 1))

def _npr_sliding(gray, windows):
    scores = []
    for y0, y1, x0, x1 in windows:
        block = gray[y0:y1, x0:x1]
        scores.append(_npr_score(_npr_features(block)))
    return np.array(scores)

# ═══════════════════════════════════════════════
# 维度二：SRM 残差分析
# ═══════════════════════════════════════════════

_SRM_FILTERS = [
    # 1阶残差
    ("1_h",  np.array([[0,0,0],[-1,1,0],[0,0,0]], dtype=np.float32)),
    ("1_v",  np.array([[0,-1,0],[0,1,0],[0,0,0]], dtype=np.float32)),
    ("1_d1", np.array([[-1,0,0],[0,1,0],[0,0,0]], dtype=np.float32)),
    ("1_d2", np.array([[0,0,-1],[0,1,0],[0,0,0]], dtype=np.float32)),
    # 2阶残差
    ("2_h",  np.array([[1,-2,1]], dtype=np.float32)),
    ("2_v",  np.array([[1],[-2],[1]], dtype=np.float32)),
    ("2_d",  np.array([[1,0,0],[0,-2,0],[0,0,1]], dtype=np.float32)),
    # 3阶残差
    ("3_h",  np.array([[1,-3,3,-1]], dtype=np.float32)),
    ("3_v",  np.array([[1],[-3],[3],[-1]], dtype=np.float32)),
    # 3x3边缘
    ("e3_lap", np.array([[-1,2,-1],[2,-4,2],[-1,2,-1]], dtype=np.float32)),
    ("e3_sh", np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)),
    ("e3_sv", np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)),
    # 十字形
    ("cross", np.array([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=np.float32)),
    # 5x5
    ("5_lap", np.array([[-1]*5,[-1]*5,[-1,-1,24,-1,-1],[-1]*5,[-1]*5], dtype=np.float32)),
    ("5_gab", np.array([[0,0,-1,0,0],[0,-1,2,-1,0],[-1,2,-4,2,-1],[0,-1,2,-1,0],[0,0,-1,0,0]], dtype=np.float32)),
]

def _srm_residuals(gray):
    """全图 SRM 残差 → (F, H, W)"""
    h, w = gray.shape
    nf = len(_SRM_FILTERS)
    res = np.zeros((nf, h, w), dtype=np.float32)
    for i, (_, k) in enumerate(_SRM_FILTERS):
        res[i] = cv2.filter2D(gray, cv2.CV_32F, k)
    return res

def _srm_win_feat(residuals, y0, y1, x0, x1):
    """单窗口 SRM 特征 (nf*4,)"""
    nf = residuals.shape[0]
    feat = np.zeros(nf * 4, dtype=np.float32)
    for i in range(nf):
        w = residuals[i, y0:y1, x0:x1].ravel()
        if len(w) == 0: continue
        mu = float(np.mean(w))
        sd = float(np.std(w)) + 1e-10
        sk = float(np.mean(((w - mu) / sd)**3))
        ku = float(np.mean(((w - mu) / sd)**4))
        feat[i*4:(i+1)*4] = [mu, sd, sk, ku]
    return feat

def _srm_sliding(gray, windows, image_path=None, win_size=128, stride=64):
    """SRM 滑窗 → 异常得分（GPU 优先，CPU 降级）"""
    # 尝试 GPU 远程加速
    if image_path is not None and os.path.isfile(image_path):
        try:
            with open(image_path, 'rb') as _f:
                _resp = requests.post(
                    GPU_SRM_API,
                    files={'file': _f},
                    data={'win_size': str(win_size), 'stride': str(stride)},
                    timeout=60
                )
            _result = _resp.json()
            if _result.get('code') == 200:
                _scores = _result['data']['scores']
                if len(_scores) == len(windows):
                    print(f"  [SRM] GPU 加速完成 ({len(windows)} 窗口)")
                    return np.array(_scores, dtype=np.float64)
        except Exception as _e:
            print(f"  [SRM] GPU 服务不可用，降级到 CPU: {_e}")
    # 降级：CPU 原实现
    return _srm_sliding_cpu(gray, windows)

def _srm_sliding_cpu(gray, windows):
    """SRM 滑窗 → 异常得分（CPU 原始实现）"""
    residuals = _srm_residuals(gray)
    feats = [_srm_win_feat(residuals, y0, y1, x0, x1) for y0, y1, x0, x1 in windows]
    if len(feats) < 4:
        return np.array([0.0]*len(feats))
    fm = np.stack(feats)  # (N, nf*4)
    gm = np.mean(fm, axis=0)
    fc = fm - gm
    cov = np.cov(fc.T) + np.eye(fc.shape[1])*1e-6
    try:
        ci = np.linalg.inv(cov)
    except np.linalg.LinAlgError:
        ci = np.linalg.pinv(cov)
    ds = [float(np.sqrt(np.dot(np.dot(fm[i]-gm, ci), fm[i]-gm))) for i in range(len(fm))]
    da = np.array(ds)
    dmx = np.max(da)
    scores = da / (dmx * 1.5) if dmx > 1e-8 else da
    return np.clip(scores, 0, 1)

# ═══════════════════════════════════════════════
# 维度三：DCT 系数分析
# ═══════════════════════════════════════════════

def _dct_block_feat(block):
    """8x8块 DCT 特征"""
    hb, wb = block.shape
    if hb != 8 or wb != 8:
        block = cv2.resize(block, (8, 8), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(block.astype(np.float32) - 128.0)
    ac = dct.copy(); ac[0,0] = 0.0
    aflat = ac.ravel()

    ac_std = float(np.std(aflat))
    tp = float(np.sum(np.abs(dct)))
    ap = float(np.sum(np.abs(ac)))
    ac_pct = ap / (tp + 1e-10)

    # 高中低频
    zz = [0,1,8,16,9,2,3,10,17,24,32,40,48,56,57,49,41,33,25,18,11,4,5,12,19,26,33,42,50,58,59,51,43,34,27,20,13,6,7,14,21,28,35,42,50,58,59,51,43,34,27,20,13,6,7,14,21,28,35,43,51,59,60,52,44,36,29,22,15,23,30,37,45,53,61,62,54,46,38,31,39,47,55,63]
    lo, mi, hi = 0.0, 0.0, 0.0
    for idx in range(1, 64):
        v = abs(dct.flat[idx])
        if idx <= 10: lo += v
        elif idx <= 35: mi += v
        else: hi += v
    total_ac = lo + mi + hi + 1e-10
    low_r, mid_r, high_r = lo/total_ac, mi/total_ac, hi/total_ac

    # Benford
    nz = np.abs(ac[ac != 0.0]).ravel()
    benford = 0.5
    if len(nz) >= 6:
        lds = [int(str(int(v))[0]) for v in nz if v >= 1.0]
        if len(lds) >= 4:
            be = np.array([np.log10(1+1/d) for d in range(1,10)])
            cnt = np.bincount(lds, minlength=10)[1:]
            ba = cnt/(np.sum(cnt)+1e-10)
            m = (be+ba)/2
            kle = np.sum(be*np.log((be+1e-10)/(m+1e-10)))
            kla = np.sum(ba*np.log((ba+1e-10)/(m+1e-10)))
            js = (kle+kla)/2
            benford = float(np.clip(1-js*5, 0, 1))

    return {"ac_std": ac_std, "ac_pct": ac_pct, "benford": benford,
            "low_r": low_r, "mid_r": mid_r, "high_r": high_r}

def _dct_sliding_cpu(gray, windows):
    """DCT 滑窗 → 异常得分（CPU 原始实现）"""
    scores = []
    for y0, y1, x0, x1 in windows:
        win = gray[y0:y1, x0:x1]
        hw, ww = win.shape
        bh, bw = hw//8, ww//8
        if bh < 2 or bw < 2:
            scores.append(0.0)
            continue
        bfs = []
        for r in range(bh):
            for c in range(bw):
                bfs.append(_dct_block_feat(win[r*8:(r+1)*8, c*8:(c+1)*8]))
        mac = float(np.mean([f["ac_std"] for f in bfs]))
        mb = float(np.mean([f["benford"] for f in bfs]))
        mhr = float(np.mean([f["high_r"] for f in bfs]))
        mapc = float(np.mean([f["ac_pct"] for f in bfs]))

        acs = np.clip(1-mac/15, 0, 1) if mac < 15 else 0
        ba = 1 - mb
        hfa = abs(mhr - 0.3) * 2
        apa = np.clip(1-mapc/0.3, 0, 1) if mapc < 0.3 else 0
        s = float(np.clip(acs*0.3 + ba*0.3 + hfa*0.2 + apa*0.2, 0, 1))
        scores.append(s)
    return np.array(scores)


def _dct_sliding(gray, windows, image_path=None, win_size=128, stride=64):
    """DCT 滑窗 → 异常得分（GPU 优先，CPU 降级）"""
    # 尝试 GPU 远程加速
    if image_path is not None and os.path.isfile(image_path):
        try:
            with open(image_path, 'rb') as _f:
                _resp = requests.post(
                    GPU_DCT_API,
                    files={'file': _f},
                    data={'win_size': str(win_size), 'stride': str(stride)},
                    timeout=60
                )
            _result = _resp.json()
            if _result.get('code') == 200:
                _scores = _result['data']['scores']
                if len(_scores) == len(windows):
                    print(f"  [DCT] GPU 加速完成 ({len(windows)} 窗口)")
                    return np.array(_scores, dtype=np.float64)
        except Exception as _e:
            print(f"  [DCT] GPU 服务不可用，降级到 CPU: {_e}")
    # 降级：CPU 原实现
    return _dct_sliding_cpu(gray, windows)

# ═══════════════════════════════════════════════
# 融合 & 热力图
# ═══════════════════════════════════════════════

def _build_heatmap_2d(windows, scores):
    y0s = sorted(set(y0 for y0, _, _, _ in windows))
    x0s = sorted(set(x0 for _, _, x0, _ in windows))
    ym = {v: i for i, v in enumerate(y0s)}
    xm = {v: i for i, v in enumerate(x0s)}
    nr, nc = max(len(y0s), 1), max(len(x0s), 1)
    hm = np.full((nr, nc), np.nan, dtype=np.float32)
    for (y0, _, x0, _), s in zip(windows, scores):
        ri, ci = ym.get(y0, 0), xm.get(x0, 0)
        if np.isnan(hm[ri, ci]): hm[ri, ci] = s
        else: hm[ri, ci] = max(hm[ri, ci], s)
    mask = np.isnan(hm)
    if not np.all(mask):
        hm[mask] = float(np.nanmean(hm))
    else:
        hm = np.zeros_like(hm)
    return hm

def _analyze_heatmap(hm):
    flat = hm.ravel()
    n = len(flat)
    if n == 0:
        return {"score": 0.0, "max_anomaly": 0.0, "mean_anomaly": 0.0,
                "contrast_ratio": 0.0, "coverage": 0.0, "dispersion": 0.0,
                "verdict": "real_likely"}
    mx = float(np.max(flat))
    mn = float(np.mean(flat))
    sd = float(np.std(flat))
    ct = mx/(mn+1e-6)
    cv = float(np.sum(flat > 0.50))/n

    sc = float(np.clip(mx*0.40 + np.clip(ct/3,0,1)*0.25 + np.clip(cv*4,0,1)*0.20 + np.clip(sd*3,0,1)*0.15, 0, 1))
    vd = "fake_likely" if sc >= 0.60 else ("uncertain" if sc >= 0.35 else "real_likely")
    return {"score": round(sc, 4), "max_anomaly": round(mx, 4),
            "mean_anomaly": round(mn, 4), "contrast_ratio": round(ct, 4),
            "coverage": round(cv, 4), "dispersion": round(sd, 4), "verdict": vd}

# ═══════════════════════════════════════════════
# 核心接口
# ═══════════════════════════════════════════════

def tamper_multidim_analysis(image_path, win_size=128, stride=64):
    """
    NPR + SRM + DCT 三维篡改检测核心接口
    返回: {tamper_score, verdict, npr/srm/dct/combined heatmaps, stats, detail}
    """
    img = _load_image(image_path)
    if img is None:
        return {"tamper_score": 0.0, "verdict": "error",
                "detail": "无法加载图片", "windows_count": 0}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    windows = _build_windows(h, w, win_size, stride)
    nw = len(windows)

    if nw < 2:
        return {"tamper_score": 0.0, "verdict": "real_likely",
                "detail": f"图片过小 ({w}x{h})，窗口数不足", "windows_count": nw}

    # 三维滑窗分析
    npr_scores = _npr_sliding(gray, windows)
    srm_scores = _srm_sliding(gray, windows, image_path, win_size, stride)
    dct_scores = _dct_sliding(gray, windows, image_path, win_size, stride)

    # 窗口级融合 (权重: 0.35 / 0.35 / 0.30)
    combined_scores = (npr_scores * 0.35 + srm_scores * 0.35 + dct_scores * 0.30)

    # 构建各自热力图
    hm_npr = _build_heatmap_2d(windows, npr_scores)
    hm_srm = _build_heatmap_2d(windows, srm_scores)
    hm_dct = _build_heatmap_2d(windows, dct_scores)
    hm_cmb = _build_heatmap_2d(windows, combined_scores)

    # 统计分析
    ns = _analyze_heatmap(hm_npr)
    ss = _analyze_heatmap(hm_srm)
    ds = _analyze_heatmap(hm_dct)
    cs = _analyze_heatmap(hm_cmb)

    # 多维度共识：至少两个维度认为 suspicious 才升高最终分
    dim_flags = [ns["score"] >= 0.35, ss["score"] >= 0.35, ds["score"] >= 0.35]
    consensus = sum(dim_flags)
    if consensus >= 2:
        final_score = cs["score"]
    elif consensus == 1:
        final_score = cs["score"] * 0.7
    else:
        final_score = cs["score"] * 0.5

    final_verdict = "fake_likely" if final_score >= 0.60 else ("uncertain" if final_score >= 0.35 else "real_likely")

    # 构建细节说明
    dim_names = ["NPR(空域纹理)", "SRM(噪声残差)", "DCT(频域系数)"]
    dim_scores = [ns["score"], ss["score"], ds["score"]]
    anomaly_dims = [dim_names[i] for i in range(3) if dim_flags[i]]
    detail_parts = [
        f"滑窗总数: {nw} ({hm_npr.shape[0]}×{hm_npr.shape[1]} 网格)",
        f"NPR维度: {ns['score']:.3f} (max={ns['max_anomaly']:.3f})",
        f"SRM维度: {ss['score']:.3f} (max={ss['max_anomaly']:.3f})",
        f"DCT维度: {ds['score']:.3f} (max={ds['max_anomaly']:.3f})",
    ]
    if anomaly_dims:
        detail_parts.append(f"异常维度: {', '.join(anomaly_dims)} (共识度 {consensus}/3)")
    else:
        detail_parts.append("三维度均未发现显著异常")
    detail_parts.append(f"综合判定: {final_verdict} ({final_score:.3f})")

    return {
        "tamper_score": round(final_score, 4),
        "verdict": final_verdict,
        "npr_heatmap": [round(v,4) for v in hm_npr.ravel().tolist()],
        "srm_heatmap": [round(v,4) for v in hm_srm.ravel().tolist()],
        "dct_heatmap": [round(v,4) for v in hm_dct.ravel().tolist()],
        "combined_heatmap": [round(v,4) for v in hm_cmb.ravel().tolist()],
        "heatmap_shape": list(hm_cmb.shape),
        "npr_stats": ns,
        "srm_stats": ss,
        "dct_stats": ds,
        "combined_stats": cs,
        "detail": " | ".join(detail_parts),
        "windows_count": nw,
    }

# ═══════════════════════════════════════════════
# 兼容旧版接口（保留传统模块 + 新版三维分析）
# ═══════════════════════════════════════════════

def copy_move_detect(image_path):
    """Copy-Move 检测 (ORB + Lowe's ratio test)——改进版"""
    img = _load_image(image_path)
    if img is None: return {"score": 0.0, "verdict": "real_likely",
                            "match_count": 0, "detail": "无法加载图片"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    orb = cv2.ORB_create(nfeatures=2000)
    kp, des = orb.detectAndCompute(gray, None)
    if des is None or len(kp) < 20:
        return {"score": 0.0, "verdict": "real_likely",
                "match_count": 0, "detail": f"检测到 {len(kp) if kp else 0} 个关键点,样本不足"}

    # 使用 knnMatch + Lowe's ratio test 替代 crossCheck 自匹配
    # k=3：最近邻总是自身(距离=0)，取第2和第3作为真正的候选匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des, des, k=3)

    match_pairs = []
    for matches in raw_matches:
        if len(matches) < 3:
            continue
        m_self, m1, m2 = matches[0], matches[1], matches[2]
        # 跳过自身匹配不成立的情况
        if m_self.queryIdx != m_self.trainIdx:
            continue
        # Lowe's ratio test on non-self matches
        if m1.distance < m2.distance * 0.75:
            p1 = kp[m1.queryIdx]
            p2 = kp[m1.trainIdx]
            spatial_dist = np.sqrt((p1.pt[0] - p2.pt[0])**2 + (p1.pt[1] - p2.pt[1])**2)
            if spatial_dist > 30:
                match_pairs.append((p1.pt, p2.pt, float(m1.distance), spatial_dist))

    n = len(match_pairs)
    h2, w2 = gray.shape
    area = h2 * w2

    # 对数归一化：少量匹配对(自然纹理重复)得分低，大量匹配对(疑似篡改)得分高
    if n < 3:
        score = 0.0
    else:
        score = min(1.0, np.log2(n / 3.0 + 1) * 0.35)

    verdict = "fake_likely" if n >= 8 else ("uncertain" if n >= 3 else "real_likely")
    density = n / (area / 10000) if area > 0 else 0.0
    return {"score": round(float(score), 4), "verdict": verdict,
            "match_count": n, "match_density": round(float(density), 4),
            "detail": f"发现 {n} 个疑似复制-移动匹配对" if n else "未发现复制-移动痕迹"}

def block_noise_analysis(image_path):
    """分块噪声分析——保留兼容"""
    img = _load_image(image_path)
    if img is None: return {"score": 0.0, "verdict": "real_likely",
                            "dispersion": 0.0, "detail": "无法加载图片"}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    bs = min(h, w) // 8
    if bs < 16:
        return {"score": 0.0, "verdict": "real_likely", "dispersion": 0.0, "detail": "图片过小"}
    noise_stds = []
    for y in range(0, h-bs+1, bs):
        for x in range(0, w-bs+1, bs):
            block = gray[y:y+bs, x:x+bs]
            b = cv2.GaussianBlur(block, (3,3), 0)
            noise = block.astype(np.float32) - b.astype(np.float32)
            noise_stds.append(float(np.std(noise)))
    arr = np.array(noise_stds)
    if len(arr) < 4:
        return {"score": 0.0, "verdict": "real_likely", "dispersion": 0.0, "detail": "分块不足"}
    med = np.median(arr)
    mad = np.median(np.abs(arr - med))
    z = 0.6745 * (arr - med) / (mad + 1e-10)
    outlier_r = float(np.sum(np.abs(z) > 2.5)) / len(arr)
    score = np.clip(outlier_r * 3.0, 0, 1)
    verdict = "fake_likely" if score >= 0.60 else "real_likely"
    return {"score": round(float(score), 4), "verdict": verdict,
            "dispersion": round(float(np.std(arr)), 4),
            "mean_score": round(float(np.mean(arr)), 4),
            "detail": f"分块噪声不一致比例: {outlier_r:.3f}"}

def patch_feature_analysis(image_path):
    """
    区块深层特征一致性分析 — 保留兼容的桩函数。
    原依赖 EfficientNet，现改为利用 NPR 滑窗的 patch 级别特征一致性。
    """
    result = tamper_multidim_analysis(image_path, win_size=64, stride=64)
    npr = result.get("combined_stats", {})
    score = npr.get("score", 0.0)
    verdict = "fake_likely" if score >= 0.60 else ("uncertain" if score >= 0.35 else "real_likely")
    return {"score": round(score, 4), "verdict": verdict,
            "mean_similarity": round(1.0 - npr.get("mean_anomaly", 0.0), 4),
            "detail": f"深层特征一致性得分: {1.0 - npr.get('mean_anomaly', 0.0):.3f}"}

def _legacy_tamper_zero_model_analysis(image_path):
    """
    零模型篡改分析——与 app.py 兼容的完整接口。

    返回包含：
      - multi_dim:  NPR+SRM+DCT 三维滑窗结果
      - copy_move: 传统 ORB 复制-移动检测
      - block_noise: 传统分块噪声不一致分析
      - patch_feature: 深层特征一致性
    """
    # 核心：三维滑窗分析
    md = tamper_multidim_analysis(image_path)

    # 传统模块
    cm = copy_move_detect(image_path)
    bn = block_noise_analysis(image_path)
    pf = patch_feature_analysis(image_path)

    # 融合：三维主分数 + 传统辅助
    trad_boost = 0.0
    if cm.get("score", 0) > 0.3:
        trad_boost += 0.05
    if bn.get("score", 0) > 0.3:
        trad_boost += 0.05
    final_score = min(md["tamper_score"] + trad_boost, 1.0)

    return {
        "multi_dim": md,
        "copy_move": cm,
        "block_noise": bn,
        "patch_feature": pf,
        "tamper_score": round(final_score, 4),
        "verdict": md["verdict"],
        "detail": md["detail"],
        "npr_score": md["npr_stats"]["score"],
        "srm_score": md["srm_stats"]["score"],
        "dct_score": md["dct_stats"]["score"],
    }


def tamper_zero_model_analysis(image_path):
    """TruFor-only tamper assessment.

    A model outage is explicit and never converted into a "real image" verdict.
    """
    trufor = analyze_trufor(image_path)
    primary = trufor if trufor.get("available") else {}
    if primary.get("available"):
        score = float(primary["score"])
        verdict = "fake_likely" if score >= 0.60 else "real_likely"
        detail = f"TruFor：{primary.get('detail', '推理完成。')}"
    else:
        score = 0.0
        verdict = "model_unavailable"
        detail = trufor.get("detail") or "TruFor 篡改定位模型不可用。"

    return {
        "trufor": trufor,
        "primary_model": "TruFor" if primary.get("available") else None,
        "tamper_score": round(score, 4),
        "verdict": verdict,
        "detail": detail,
    }
