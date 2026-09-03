# -*- coding: utf-8 -*-
"""
本地 Qwen3-VL 文本多轮对话服务（模型常驻，只加载一次）
用同一个视觉模型做纯文本聊天，支持多轮上下文。

启动:
  pip install fastapi uvicorn
  python chat_server.py
浏览器打开: http://127.0.0.1:8001

API:
  POST /chat   {"session_id": "abc", "message": "你好"}       -> 返回回复
  POST /reset  {"session_id": "abc"}                          -> 清空该会话历史
"""

import torch
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# ===== 配置 =====
MODEL_DIR = r"D:\models\Qwen3-VL-4B-Instruct"
SYSTEM_PROMPT = "你是一个乐于助人的中文智能助手，回答简洁、准确。"
MAX_NEW_TOKENS = 1024
MAX_HISTORY_TURNS = 10   # 只保留最近 N 轮，防止上下文过长撑爆 6G 显存

STATE = {"model": None, "processor": None}
# 内存里保存每个会话的历史： {session_id: [ {role, content}, ... ]}
SESSIONS = {}


def load_model():
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
    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    STATE["model"] = model
    STATE["processor"] = processor
    print("[信息] 模型加载完成！浏览器打开 http://127.0.0.1:8001")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield
    STATE.clear()


app = FastAPI(title="Qwen3-VL 本地多轮对话", lifespan=lifespan)


class ChatIn(BaseModel):
    session_id: str
    message: str


class ResetIn(BaseModel):
    session_id: str


def build_messages(history):
    """把系统提示 + 历史拼成模型输入格式。"""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    return messages


def generate_reply(history):
    """纯文本推理：给对话历史，返回助手回复。"""
    model = STATE["model"]
    processor = STATE["processor"]

    messages = build_messages(history)
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # 纯文本，无图片，直接交给 processor 的分词器处理
    inputs = processor(text=[text], return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    reply = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return reply.strip()


@app.post("/chat")
async def chat(req: ChatIn):
    try:
        history = SESSIONS.setdefault(req.session_id, [])
        history.append({"role": "user", "content": req.message})

        # 只保留最近若干轮，防止上下文过长
        if len(history) > MAX_HISTORY_TURNS * 2:
            history = history[-MAX_HISTORY_TURNS * 2:]
            SESSIONS[req.session_id] = history

        reply = generate_reply(history)
        history.append({"role": "assistant", "content": reply})
        return JSONResponse({"ok": True, "reply": reply})
    except Exception as e:
        return JSONResponse({"ok": False, "error": f"{type(e).__name__}: {e}"}, status_code=500)


@app.post("/reset")
async def reset(req: ResetIn):
    SESSIONS.pop(req.session_id, None)
    return JSONResponse({"ok": True})


@app.get("/", response_class=HTMLResponse)
async def index():
    return """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Qwen3-VL 本地对话</title>
<style>
  body { font-family: system-ui, sans-serif; max-width: 720px; margin: 0 auto;
         height: 100vh; display: flex; flex-direction: column; }
  header { padding: 12px 16px; border-bottom: 1px solid #eee; display: flex;
           justify-content: space-between; align-items: center; }
  h1 { font-size: 17px; margin: 0; }
  #reset { font-size: 13px; padding: 6px 12px; cursor: pointer; }
  #chat { flex: 1; overflow-y: auto; padding: 16px; }
  .msg { margin: 10px 0; display: flex; }
  .msg.user { justify-content: flex-end; }
  .bubble { max-width: 78%; padding: 10px 14px; border-radius: 14px;
            white-space: pre-wrap; line-height: 1.5; }
  .user .bubble { background: #2563eb; color: #fff; border-bottom-right-radius: 4px; }
  .bot .bubble { background: #f1f1f3; color: #111; border-bottom-left-radius: 4px; }
  footer { display: flex; padding: 12px 16px; border-top: 1px solid #eee; gap: 8px; }
  #input { flex: 1; padding: 10px 12px; font-size: 15px; border: 1px solid #ccc;
           border-radius: 10px; resize: none; }
  #send { padding: 0 20px; font-size: 15px; cursor: pointer; border: none;
          background: #2563eb; color: #fff; border-radius: 10px; }
  #send:disabled { background: #9db8f0; cursor: default; }
</style>
</head>
<body>
  <header>
    <h1>Qwen3-VL 本地对话</h1>
    <button id="reset" onclick="resetChat()">新对话</button>
  </header>
  <div id="chat"></div>
  <footer>
    <textarea id="input" rows="1" placeholder="输入消息，回车发送（Shift+回车换行）"></textarea>
    <button id="send" onclick="send()">发送</button>
  </footer>

<script>
  const sid = 'sess_' + Math.random().toString(36).slice(2);
  const chat = document.getElementById('chat');
  const input = document.getElementById('input');
  const sendBtn = document.getElementById('send');

  function addMsg(role, text) {
    const div = document.createElement('div');
    div.className = 'msg ' + (role === 'user' ? 'user' : 'bot');
    div.innerHTML = '<div class="bubble"></div>';
    div.querySelector('.bubble').textContent = text;
    chat.appendChild(div);
    chat.scrollTop = chat.scrollHeight;
    return div.querySelector('.bubble');
  }

  async function send() {
    const text = input.value.trim();
    if (!text) return;
    input.value = '';
    addMsg('user', text);
    sendBtn.disabled = true;
    const botBubble = addMsg('bot', '思考中...');
    try {
      const resp = await fetch('/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sid, message: text })
      });
      const data = await resp.json();
      botBubble.textContent = data.ok ? data.reply : ('出错：' + data.error);
    } catch (e) {
      botBubble.textContent = '请求失败：' + e;
    }
    sendBtn.disabled = false;
    chat.scrollTop = chat.scrollHeight;
  }

  async function resetChat() {
    await fetch('/reset', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sid })
    });
    chat.innerHTML = '';
  }

  input.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); send(); }
  });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)