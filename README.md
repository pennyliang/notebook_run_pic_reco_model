笔记本本地部署视觉大模型
在一台普通笔记本 + 消费级英伟达显卡上，完全离线地部署 Qwen3-VL 视觉语言模型。它既能识别图片里的中文文字和画面内容（OCR + 图像理解），也能当作普通大语言模型做多轮文本对话。提供命令行工具、图片识别服务、文本对话服务三种形态，均带网页界面。
> 全程本地、离线、免费，数据不出本机。Qwen3-VL 是「能看图的大语言模型」，所以同一个模型既能识图，也能纯文本聊天。
✨ 特性
一模两用：同一个模型，既做图片识别，又做多轮文本对话
完全本地离线：跑在自己的显卡上，不联网、不花钱、数据不外传
中文友好：Qwen3-VL 对中文 OCR、画面理解、对话都很强
低显存可用：4-bit 量化，6GB 显存（如 RTX 2060）即可运行 4B 模型
三种服务形态：
命令行单张图片识别
图片识别 HTTP 服务 + 网页拖拽界面
文本多轮对话 HTTP 服务 + 网页聊天界面
可批量：附带批量识别脚本，适合成规模处理截图
踩坑手册：部署中真实遇到的每个坑和解法都记录在 docs/troubleshooting.md
🖥️ 环境要求
项目	要求
操作系统	Windows 10/11（Linux 见文末，流程几乎一致）
显卡	英伟达显卡，≥6GB 显存（4-bit 跑 4B 模型）
驱动	较新的 NVIDIA 驱动（支持 CUDA 12.x 及以上）
磁盘	至少 15GB 可用空间（模型约 9G + PyTorch 约 3G）
Python	3.11（用 conda 建独立环境，不要用 3.13/3.14，PyTorch 生态跟进慢）
🚀 快速开始
第 0 步：确认显卡与驱动
```bash
nvidia-smi
```
能出表格、显示显卡型号和 CUDA 版本即可。驱动太旧或命令找不到，见 踩坑手册 §1。
第 1 步：装 Miniconda 并建独立环境
去 Miniconda 官网 下载安装，然后：
```bash
conda create -n vlm python=3.11 -y
conda activate vlm
```
第 2 步：装 CUDA 版 PyTorch
关键：必须装 CUDA 版，不能是 CPU 版。
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
网络反复中断时，改用 curl 手动下 whl 再本地装（见 踩坑手册 §2）。装完自检：
```bash
python scripts/check_gpu.py
```
看到 `CUDA 可用: True` 和你的显卡型号才算成功。
第 3 步：装其余依赖
```bash
pip install transformers accelerate qwen-vl-utils pillow bitsandbytes
pip install fastapi uvicorn python-multipart   # 跑 HTTP 服务需要
```
第 4 步：下载模型
把 `scripts/download_model.bat` 放到想存模型的位置（建议 D 盘），双击运行。它用 curl 断点续传，稳定快速，断了重跑即可。
模型约 8.9G，含 2 个 `.safetensors` 权重 + 9 个配置文件，缺一不可。
> 不建议用 `hf download`：Qwen3-VL 权重走 Xet 存储，配国内镜像常卡死（见 [踩坑手册 §3](docs/troubleshooting.md)）。
第 5 步：改路径，开跑
把各脚本顶部的 `MODEL_DIR` / `DEFAULT_MODEL_DIR` 改成你的模型目录，然后三选一：
```bash
python recognize.py 图片.jpg      # 命令行识图
python server.py                  # 图片识别服务  -> http://127.0.0.1:8000
python chat_server.py             # 文本对话服务  -> http://127.0.0.1:8001
```
📖 三种用法
用法一：命令行识别单张图片
```bash
python recognize.py D:\path\to\image.jpg

# 自定义提示词（逐字识别、不猜测，精度更高）
python recognize.py image.jpg -p "严格逐字识别图中所有文字，模糊的用【?】标注，不要猜测"

# 指定输出 / 长度 / 看图清晰度
python recognize.py image.jpg -o result.txt --max-tokens 512 --max-pixels 2000000

python recognize.py --help        # 全部参数
```
用法二：图片识别服务（含网页界面）
模型只加载一次，之后每次识别秒级返回；浏览器里拖图、改提示词、看结果。
```bash
python server.py
```
浏览器打开 http://127.0.0.1:8000。API：
```bash
curl.exe -X POST http://127.0.0.1:8000/recognize -F "file=@image.jpg" -F "prompt=识别图中文字"
```
批量识别（先启动上面的服务）：
```bash
python scripts/batch_recognize.py D:\screenshots
```
用法三：文本多轮对话服务（含聊天界面）
用同一个模型做纯文本聊天，服务端保存每个会话的历史，支持多轮上下文。
```bash
python chat_server.py
```
浏览器打开 http://127.0.0.1:8001，像聊天软件一样对话，点「新对话」清空上下文。API：
```bash
curl.exe -X POST http://127.0.0.1:8001/chat -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"test\",\"message\":\"你好\"}"
```
同一个 `session_id` 连续发即多轮；换 `session_id` 即新对话；POST `/reset` 清空会话。
可在 `chat_server.py` 顶部改 `SYSTEM_PROMPT` 定制助手角色，改 `MAX_HISTORY_TURNS` 控制记忆轮数（6G 显存别设太大）。
> 两个服务端口错开（8000 / 8001），但 6G 显存**建议一次只开一个**，避免同时加载两份模型撑爆显存。
🎯 提升识别精度
小模型 + 4-bit 量化精度有限，可从这几处提升：
调大 `--max-pixels`：让模型看图更清晰，对密集小字最有效（更吃显存）
逐字提示词：要求「严格逐字、模糊标注、不要猜测」，减少脑补
换 8-bit 或更大模型：显存够时精度更高
上更大的卡跑大模型：见下节
📦 项目结构
```
local-vlm-ocr/
├── README.md
├── LICENSE
├── requirements.txt
├── recognize.py              # 命令行识别工具（4-bit 量化，带参数）
├── server.py                 # 图片识别 HTTP 服务 + 网页界面（端口 8000）
├── chat_server.py            # 文本多轮对话 HTTP 服务 + 聊天界面（端口 8001）
├── scripts/
│   ├── download_model.bat    # curl 断点续传下载模型
│   ├── check_gpu.py          # PyTorch 显卡自检
│   └── batch_recognize.py    # 批量识别客户端
└── docs/
    └── troubleshooting.md    # 踩坑排查手册（重点！）
```
⬆️ 扩展到更大的显卡 / 服务器
本项目在 6GB 的 RTX 2060 上「能跑通」，追求高精度可搬到多卡服务器：
换大模型：`Qwen3-VL-32B` 等，识别与对话质量是另一个量级（把模型目录换掉即可）
多卡推理：`device_map="auto"` 自动切分到多张卡；高吞吐批量场景推荐用 vLLM 起 OpenAI 兼容服务
注意 V100 等老架构不支持 bf16，统一用 fp16
🐧 Linux 说明
流程与 Windows 几乎一致，只是系统命令不同：Miniconda 用 `wget ... && bash` 安装，模型下载用 `curl` 或 `wget`，Python 代码完全通用。前提是先装好 Linux 版 NVIDIA 驱动（`nvidia-smi` 能出表格）。
📝 License
MIT
🙏 致谢
Qwen3-VL — 阿里通义千问视觉语言模型
HF-Mirror — HuggingFace 国内镜像
DDU — 驱动清理工具