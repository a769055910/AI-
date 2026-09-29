"""
图片分析辅助工具
用法一：双击运行，选择图片文件
用法二：将图片拖拽到 image_analyzer.bat 上
"""
import base64
import json
import os
import sys
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext

import httpx
import pyperclip

# === 配置 ===
SILICONFLOW_API_KEY = os.getenv("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

# 预设问题列表，方便切换分析视角
PRESET_QUESTIONS = {
    "通用描述": "请详细描述这张图片的内容",
    "文字提取": "请识别并提取图片中的所有文字内容",
    "结构分析": "请分析这张图中的架构、流程或结构关系",
    "关键信息": "请提炼这张图片中最关键的3-5条信息",
    "AI鉴别视角": "请从AI伪造/篡改鉴别的角度分析这张图片，指出可能存在的可疑之处",
}


def analyze_image(image_path: str, question: str) -> str:
    """调用硅基流动视觉模型分析图片"""
    if not os.path.isfile(image_path):
        return f"⚠️ 找不到图片文件：{image_path}"

    with open(image_path, "rb") as f:
        image_bytes = f.read()

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = MIME_MAP.get(ext, "image/png")
    image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    file_size_mb = len(image_bytes) / 1024 / 1024

    try:
        response = httpx.post(
            f"{BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {SILICONFLOW_API_KEY}",
                "Content-Type": "application/json",
            },
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}"
                                },
                            },
                            {"type": "text", "text": question},
                        ],
                    }
                ],
                "max_tokens": 1024,
                "temperature": 0.7,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        result = response.json()
    except Exception as e:
        return f"⚠️ API 调用失败：{e}"

    if "choices" not in result or len(result["choices"]) == 0:
        return f"⚠️ API 返回异常：{json.dumps(result, ensure_ascii=False)}"

    answer = result["choices"][0]["message"]["content"]
    usage = result.get("usage", {})
    t = usage.get("total_tokens", "?")
    p = usage.get("prompt_tokens", "?")
    c = usage.get("completion_tokens", "?")

    return (
        f"{'='*50}\n"
        f"  [Image] {os.path.basename(image_path)} ({file_size_mb:.1f}MB)\n"
        f"  [Mode]  {question[:30]}...\n"
        f"{'='*50}\n\n"
        f"{answer}\n\n"
        f"{'='*50}\n"
        f"  [Token] {t} (in {p} + out {c})\n"
        f"{'='*50}"
    )


def open_image_dialog():
    """打开文件选择对话框"""
    path = filedialog.askopenfilename(
        title="选择要分析的图片",
        filetypes=[
            ("图片文件", "*.png *.jpg *.jpeg *.webp *.gif *.bmp"),
            ("所有文件", "*.*"),
        ],
    )
    if path and os.path.isfile(path):
        return path
    return None


class AnalyzerApp:
    def __init__(self, image_path: str = None):
        self.root = tk.Tk()
        self.root.title("🖼️ 图片分析助手")
        self.root.geometry("700x600")
        self.root.minsize(500, 400)

        # 居中
        self.root.update_idletasks()
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = 700, 600
        x = (sw - w) // 2
        y = (sh - h) // 2
        self.root.geometry(f"{w}x{h}+{x}+{y}")

        # === 顶部：图片路径 ===
        top_frame = tk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=12, pady=(12, 0))

        tk.Label(top_frame, text="📁 图片路径：", font=("微软雅黑", 10)).pack(side=tk.LEFT)
        self.path_var = tk.StringVar(value=image_path or "")
        path_entry = tk.Entry(
            top_frame, textvariable=self.path_var, font=("Consolas", 10)
        )
        path_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)
        tk.Button(top_frame, text="选择...", command=self.select_file).pack(
            side=tk.LEFT
        )

        # === 中部：问题选择 ===
        q_frame = tk.Frame(self.root)
        q_frame.pack(fill=tk.X, padx=12, pady=(8, 4))

        tk.Label(q_frame, text="🔍 分析模式：", font=("微软雅黑", 10)).pack(
            side=tk.LEFT
        )
        self.q_mode = tk.StringVar(value="通用描述")
        self.q_combo = tk.OptionMenu(
            q_frame, self.q_mode, *PRESET_QUESTIONS.keys()
        )
        self.q_combo.config(font=("微软雅黑", 9))
        self.q_combo.pack(side=tk.LEFT, padx=4)

        tk.Label(q_frame, text="或自定义：", font=("微软雅黑", 9)).pack(
            side=tk.LEFT, padx=(8, 2)
        )
        self.custom_q = tk.Entry(q_frame, font=("微软雅黑", 9))
        self.custom_q.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=4)

        # === 按钮区 ===
        btn_frame = tk.Frame(self.root)
        btn_frame.pack(fill=tk.X, padx=12, pady=8)
        self.analyze_btn = tk.Button(
            btn_frame,
            text="🚀 开始分析",
            command=self.do_analyze,
            bg="#2563eb",
            fg="white",
            font=("微软雅黑", 11, "bold"),
            padx=20,
            pady=4,
        )
        self.analyze_btn.pack(side=tk.LEFT, padx=(0, 8))

        self.copy_btn = tk.Button(
            btn_frame,
            text="📋 复制结果",
            command=self.do_copy,
            state=tk.DISABLED,
            font=("微软雅黑", 9),
        )
        self.copy_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="准备就绪 — 选择图片后点击「开始分析」")
        status_label = tk.Label(
            btn_frame,
            textvariable=self.status_var,
            fg="gray",
            font=("微软雅黑", 9),
        )
        status_label.pack(side=tk.RIGHT)

        # === 结果展示区 ===
        self.result_text = scrolledtext.ScrolledText(
            self.root,
            wrap=tk.WORD,
            font=("Consolas", 10),
            padx=10,
            pady=10,
        )
        self.result_text.pack(fill=tk.BOTH, expand=True, padx=12, pady=(0, 12))
        self.result_text.insert(tk.END, "等待分析...\n")
        self.result_text.config(state=tk.DISABLED)

        self.current_result = ""

        # 如果传入了路径，自动分析
        if image_path and os.path.isfile(image_path):
            self.root.after(500, self.do_analyze)

        self.root.mainloop()

    def select_file(self):
        path = open_image_dialog()
        if path:
            self.path_var.set(path)

    def get_question(self) -> str:
        custom = self.custom_q.get().strip()
        if custom:
            return custom
        mode = self.q_mode.get()
        return PRESET_QUESTIONS.get(mode, PRESET_QUESTIONS["通用描述"])

    def do_analyze(self):
        image_path = self.path_var.get().strip()
        if not image_path:
            messagebox.showwarning("提示", "请先选择一张图片！")
            return
        if not os.path.isfile(image_path):
            messagebox.showerror("错误", f"文件不存在：\n{image_path}")
            return

        ext = os.path.splitext(image_path)[1].lower()
        if ext not in MIME_MAP:
            messagebox.showwarning("提示", f"不支持的图片格式：{ext}")
            return

        question = self.get_question()
        self.status_var.set("⏳ 正在调用视觉模型分析中...")
        self.analyze_btn.config(state=tk.DISABLED, text="⏳ 分析中...")
        self.root.update()

        try:
            result = analyze_image(image_path, question)
        except Exception as e:
            result = f"⚠️ 分析失败：{e}"

        self.current_result = result
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, result)
        self.result_text.config(state=tk.DISABLED)
        self.result_text.see("1.0")

        self.analyze_btn.config(state=tk.NORMAL, text="🚀 开始分析")
        self.copy_btn.config(state=tk.NORMAL)
        self.status_var.set(
            "✅ 分析完成！结果已准备就绪，可点击「复制结果」"
        )

    def do_copy(self):
        if self.current_result:
            pyperclip.copy(self.current_result)
            self.status_var.set("📋 已复制到剪贴板！粘贴到对话框即可使用。")
        else:
            messagebox.showwarning("提示", "暂无结果可复制")


def main():
    # 如果命令行传入了图片路径
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            AnalyzerApp(path)
        else:
            print(f"文件不存在：{path}")
            input("按回车键退出...")
            return
    else:
        AnalyzerApp()


if __name__ == "__main__":
    main()
