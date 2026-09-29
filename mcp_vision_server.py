"""
硅基流动 (SiliconFlow) 视觉理解 MCP Server
提供图片分析工具，让 DeepSeek 等 LLM 能够"看懂"图片
"""
import base64
import json
import os

import httpx
from mcp.server import MCPServer
from mcp.server.stdio import stdio_server

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Instruct"

MIME_MAP = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
    ".gif": "image/gif",
    ".bmp": "image/bmp",
}

mcp = MCPServer("siliconflow-vision")


@mcp.tool()
def analyze_image(
    image_path: str,
    question: str = "请详细描述这张图片的内容",
) -> str:
    """
    分析图片内容，可以识别图片中的物体、场景、文字、人物等。支持中文提问和中文回答。

    参数:
        image_path: 图片文件的绝对路径（支持 png/jpg/webp/gif/bmp）
        question: 关于图片的问题，默认为"请详细描述这张图片的内容"

    返回:
        图片分析结果（中文）
    """
    if not SILICONFLOW_API_KEY:
        return (
            "错误：未设置 SILICONFLOW_API_KEY 环境变量。\n"
            "请在 MCP 配置文件的 env 中填入你的硅基流动 API Key。\n"
            "获取地址：https://cloud.siliconflow.cn/account/ak"
        )

    if not os.path.isfile(image_path):
        return f"错误：找不到图片文件 → {image_path}"

    try:
        with open(image_path, "rb") as f:
            image_bytes = f.read()
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
    except Exception as e:
        return f"读取图片失败：{e}"

    ext = os.path.splitext(image_path)[1].lower()
    mime_type = MIME_MAP.get(ext, "image/png")
    file_size_kb = len(image_bytes) / 1024

    try:
        response = httpx.post(
            f"{SILICONFLOW_BASE_URL}/chat/completions",
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
    except httpx.HTTPError as e:
        return f"API 请求失败：{e}"
    except Exception as e:
        return f"未知错误：{e}"

    if "choices" in result and len(result["choices"]) > 0:
        answer = result["choices"][0]["message"]["content"]
        usage = result.get("usage", {})
        token_info = ""
        if usage:
            t = usage.get("total_tokens", "?")
            p = usage.get("prompt_tokens", "?")
            c = usage.get("completion_tokens", "?")
            token_info = f"\n\n---\n*Token: {t} (输入 {p} + 输出 {c})*"
        return f"**图片分析** ({os.path.basename(image_path)}, {file_size_kb:.1f}KB)\n\n{answer}{token_info}"
    else:
        return f"API 返回异常：\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```"


def main():
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
