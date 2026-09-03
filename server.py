# -*- coding: utf-8 -*-
"""
本地 Qwen3-VL-4B 图片识别 HTTP 服务（模型常驻，只加载一次）
环境: Windows + RTX 2060 (6GB) + conda 环境 vlm

启动:
  pip install fastapi uvicorn python-multipart
  python server.py

然后浏览器打开:  http://127.0.0.1:8000
或用 API:
  curl -X POST http://127.0.0.1:8000/recognize -F "file=@图.jpg" -F "prompt=识别图中文字"
"""

import io
import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import HTMLResponse, JSONResponse
from PIL import Image

# ===== 配置 =====
MODEL_DIR = r"D:\models\Qwen3-VL-4B-Instruct"
DEFAULT_PROMPT = "请识别并总结这张图里的全部内容，用中文清晰回答。"
MAX_PIXELS = 1280 * 28 * 28

# 全局变量，模型只加载一次，存这里给所有请求复用
STATE = {"model": None, "processor": None}


def load_model():
    """启动时加载一次模型。"""
    print("[信息] 正在加载模型（只需一次，几十秒）...")
    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    except ImportError:
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import AutoModelForImageTextToText as Qwen3VLForConditionalGeneration

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        MODEL_DIR,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        local_files_only=True,
    )
    processor = AutoProcessor.from_pretrained(
        MODEL_DIR, local_files_only=True, max_pixels=MAX_PIXELS
    )
    STATE["model"] = model
    STATE["processor"] = processor
    print("[信息] 模型加载完成，服务就绪！浏览器打开 http://127.0.0.1:8000")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()      # 服务启动时加载
    yield
    STATE.clear()     # 服务关闭时清理


app = FastAPI(title="Qwen3-VL 本地识别服务", lifespan=lifespan)


def run_inference(image: Image.Image, prompt: str) -> str:
    """核心推理：给一张 PIL 图 + 提示词，返回识别文本。"""
    from qwen_vl_utils import process_vision_info
    model = STATE["model"]
    processor = STATE["processor"]

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs, videos=video_inputs,
        padding=True, return_tensors="pt",
    ).to("cuda:0")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=1024)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    return processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]


@app.post("/recognize")
async def recognize(file: UploadFile = File(...), prompt: str = Form(DEFAULT_PROMPT)):
    """API 接口：上传图片 + 提示词，返回识别结果。"""
    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw)).convert("RGB")
        result = run_inference(image, prompt)
        return JSONResponse({"ok": True, "result": result})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.get("/", response_class=HTMLResponse)
async def index():
    """一个简单的网页界面，浏览器直接用。"""
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>Qwen3-VL 本地识别</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 760px; margin: 40px auto; padding: 0 16px; }
  h1 { font-size: 20px; }
  textarea { width: 100%; height: 70px; box-sizing: border-box; padding: 8px; }
  button { padding: 10px 20px; font-size: 15px; cursor: pointer; margin-top: 10px; }
  #preview { max-width: 300px; margin: 12px 0; display: none; border: 1px solid #ccc; }
  #result { white-space: pre-wrap; background: #f5f5f5; padding: 16px; border-radius: 8px;
            margin-top: 16px; min-height: 40px; }
  .hint { color: #888; font-size: 13px; }
</style>
</head>
<body>
  <h1>Qwen3-VL 本地图片识别</h1>
  <p class="hint">选一张图，改改提示词，点识别。模型已常驻显存，结果几秒返回。</p>

  <input type="file" id="file" accept="image/*"><br>
  <img id="preview">

  <p>提示词：</p>
  <textarea id="prompt">请识别并总结这张图里的全部内容，用中文清晰回答。</textarea><br>
  <button onclick="run()">开始识别</button>

  <div id="result"></div>

<script>
  const fileEl = document.getElementById('file');
  const preview = document.getElementById('preview');
  fileEl.onchange = () => {
    if (fileEl.files[0]) {
      preview.src = URL.createObjectURL(fileEl.files[0]);
      preview.style.display = 'block';
    }
  };
  async function run() {
    const f = fileEl.files[0];
    if (!f) { alert('请先选一张图片'); return; }
    const box = document.getElementById('result');
    box.textContent = '识别中，请稍候...';
    const fd = new FormData();
    fd.append('file', f);
    fd.append('prompt', document.getElementById('prompt').value);
    try {
      const resp = await fetch('/recognize', { method: 'POST', body: fd });
      const data = await resp.json();
      box.textContent = data.ok ? data.result : ('出错：' + data.error);
    } catch (e) {
      box.textContent = '请求失败：' + e;
    }
  }
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
