# AI 伪造鉴别智能体

基于 Flask 的 AI 伪造内容鉴别系统，支持图像检测（含 AI 生成、深度伪造、图像篡改三类专项检测），并提供文本/视频/音频检测的展示框架。

## 架构说明

本项目分为**本地 Web 服务**和**远程 GPU 推理服务**两部分：

```
浏览器
  │
  ▼
本地 Flask (app.py, 端口 5000)
  │  ├─ 水印检测 (GLM-4.5V)
  │  ├─ NPR / DeepFake / 篡改 零模型分析（本地）
  │  └─ 调用 GPU 推理服务
  │
  ▼
SSH 隧道（本地 6006 → 远程 6006）
  │
  ▼
AutoDL GPU 服务器 (gpu_server.py, 端口 6006)
  │  ├─ Modotte/AIRealNet     (SwinV2, 主模型 → AI 生成检测)
  │  └─ prithivMLmods/Deep-Fake-Detector-v2-Model (ViT-B → 深度伪造检测)
```

## 一、AutoDL 服务器信息

| 项目 | 值 |
|------|------|
| SSH 地址 | [REDACTED: retrieve from AutoDL console]|
| SSH 端口 | [REDACTED: retrieve from AutoDL console]|
| 用户名 | [REDACTED: retrieve from AutoDL console]|
| 密码 | [REMOVED: keep credentials out of the repository]|
| 远程代码目录 | `/root/forge-detector` |
| 服务端口 | `6006` |
| Python 解释器 | `/root/miniconda3/bin/python3` |
| 模型缓存目录 | `~/.cache/huggingface/hub/` |
| 日志文件 | `/root/forge-detector/server.log` |

> ⚠️ 模型缓存在 `~/.cache/huggingface/hub/`，**只要实例是「关机」而非「已释放」，开机后不会重新下载模型**，几十秒即可就绪。

## 二、启动流程（三步）

### 第 1 步：开机 AutoDL 实例

在 [AutoDL 控制台](https://www.autodl.com/console)确认实例状态：

- 状态为**「关机」** → 点击「开机」即可，`/root` 下的代码和模型缓存都还在。
- 状态为**「已释放」** → 数据会被清空，需要重新部署（见下方「重新部署」章节）。

### 第 2 步：启动 GPU 推理服务

Local-only restart helper scripts are excluded from the repository. Use the manual SSH steps below.

**方式 B：SSH 手动启动**

```bash
ssh -p "$AUTODL_SSH_PORT" "$AUTODL_SSH_USER@$AUTODL_SSH_HOST"
```

登录后执行：

```bash
# 1. 清理旧进程（避免端口占用）
pkill -9 -f gpu_server.py 2>/dev/null; sleep 2

# 2. 启动服务（后台运行，日志写入 server.log）
cd /root/forge-detector && nohup /root/miniconda3/bin/python3 gpu_server.py --port 6006 > server.log 2>&1 &

# 3. 实时查看启动日志
tail -f /root/forge-detector/server.log
```

看到以下日志即启动成功：

```
[Startup] AIRealNet 模型加载完成
[Startup] DeepFakeV2 模型加载完成
Running on http://0.0.0.0:6006
```

### 第 3 步：验证服务 + 建立本地隧道

**验证 GPU 服务**（在服务器上执行）：

```bash
curl http://127.0.0.1:6006/health
```

TruFor 篡改定位默认将最长边超过 1024px 的图片以 Lanczos 高质量缩放后再推理；原始上传文件不会被改写。为避免它与已完成的专项模型争抢显存，服务默认在调用 TruFor 前释放这些模型缓存，并会在下次对应检测时按需重新加载。若显存充足且需要调整该行为，可在启动 GPU 服务前设置 `TRUFOR_MAX_EDGE` 或 `TRUFOR_RELEASE_GPU_MODELS=0`。

返回 `{"status":"ok","device":"cuda"}` 即为成功。

**建立 SSH 隧道**（把远程 6006 映射到本地 6006），然后本地验证：

```powershell
Invoke-WebRequest -Uri "http://127.0.0.1:6006/health" -UseBasicParsing
```

**启动本地 Web 服务**：

```powershell
python app.py
```

浏览器访问 `http://127.0.0.1:5000` 即可使用。

## 三、检测能力说明

### 水印检测（步骤 1，双通道）

1. **本地隐式标识检测**（毫秒级，优先执行，命中即短路）：
   - **阶段一 · 国外元数据/C2PA**（`hidden_watermark_detector.py`）：解析 PNG/JPEG/WebP 元数据（EXIF、XMP、PNG chunk、iCCP 等），检测国外主流 AI 平台（Midjourney、DALL·E、Stable Diffusion、Adobe Firefly、Flux 等）写入的生成签名。
   - **阶段二 · 国内 TC260:AIGC 隐式标识**（`tc260_detector.py`）：依据国标 **GB 45438—2025**《网络安全技术 人工智能生成合成内容标识方法》与 TC260《文件元数据隐式标识》实践指南，检测国内平台写入的 `TC260:AIGC` 隐式标识，兼容四种形式：
     - XMP 元素形式 `<TC260:AIGC>{...}</TC260:AIGC>`（PNG iTXt / XMP，JSON 经 XML 实体转义）
     - 直放形式 `TC260:AIGC={...}`
     - 嵌套 JSON 形式 `{"AIGC":{...}}`（JPEG EXIF，腾讯混元 / 百度文心等）
     - TC260 字段直存 JSON
   - 校验 GB 45438 必备字段完整度（`Label` / `ContentProducer` / `ProduceID`），`Label=1`（AI 生成）且字段完整时置信度 0.93~0.97，并依据 `ContentProducer` / `ProduceID` 特征推断平台（豆包、即梦、混元、通义千问、文心、讯飞星火等，识别不出时回退原始编码）。

2. **GLM-4.5V 可见水印检测**（`siliconflow_client.py`）：本地隐式标识未命中时，调用多模态视觉模型识别图片上的可见水印与 AI 平台签名。

### 检测流程（图像，7 步）

```
1. 水印检测（本地隐式标识 → GLM 可见水印，命中即短路判定 AI 生成）
2. 图像元数据 / 噪声模式（NPR）
3. DeepFake 深度伪造专项检测（零模型 + GPU ViT 模型）
4. 图像篡改专项检测（零模型 + GPU 模型）
5. 多模型联合判定
6. 综合评分与报告
7. 输出结果
```

## 四、脚本速查表

| 脚本 | 用途 |
|------|------|
| `python _deploy_fix.py` | 重新部署：上传 `gpu_server.py` 到远程并重启服务 |
| `python _deploy.py` | 完整部署脚本（上传 + 装依赖 + 启动 + 健康检查） |
| `python _status.py` | 查看远程进程、端口、日志、模型缓存、磁盘、健康状态 |
| `python _check_server.py` | 远程服务完整诊断（GPU / 进程 / 端口 / 日志 / 磁盘） |
| `python app.py` | 启动本地 Web 服务（端口 5000） |

## 四、重新部署（仅实例被「释放」后需要）

如果开机后发现 `/root/forge-detector` 目录为空、或模型要重新下载，说明实例被「释放」过，需要重新部署：

```powershell
python _deploy_fix.py
```

首次部署会从 HuggingFace 镜像（`https://hf-mirror.com`）下载模型，约 3–5 分钟。

## 五、常见问题

| 现象 | 原因 | 解决 |
|------|------|------|
| `GPU 服务器连接失败` | 实例未开机 / 隧道未建立 | 检查 AutoDL 实例状态 + 隧道工具 |
| 模型重新下载 | 实例被「释放」而非「关机」 | 跑 `_deploy_fix.py` 重新部署 |
| `Address already in use` | 端口 6006 被旧进程占用 | `pkill -9 -f gpu_server.py` 后重启 |
| 启动慢 | 首次加载模型 | 正常，等待日志出现「模型加载完成」 |

## 六、依赖安装

本地运行依赖：

```powershell
pip install flask pillow requests paramiko
```

远程 GPU 服务依赖（部署脚本会自动安装）：

```
flask pillow torch torchvision transformers
```

> API credentials must come from the `SILICONFLOW_API_KEY` environment variable. Revoke and replace any credentials previously stored in source or docs.
