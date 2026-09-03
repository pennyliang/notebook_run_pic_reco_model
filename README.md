# 笔记本本地部署视觉大模型识别图片

在一台**普通笔记本 + 消费级英伟达显卡**上，完全离线地部署 [Qwen3-VL](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) 视觉语言模型，用来识别图片里的**中文文字 + 画面内容**（OCR + 图像理解）。提供命令行工具、常驻 HTTP 服务和网页界面三种用法。

> 全程本地、离线、免费，数据不出本机。特别适合识别中文界面截图、社媒截图、文档图片等。

## ✨ 特性

- **完全本地离线**：模型跑在你自己的显卡上，不联网、不花钱、数据不外传
- **中文友好**：Qwen3-VL 对中文 OCR 和画面理解都很强
- **低显存可用**：4-bit 量化，**6GB 显存**（如 RTX 2060）即可运行 4B 模型
- **三种用法**：命令行单张识别 / 常驻 HTTP 服务 / 网页拖拽界面
- **可批量**：附带批量识别脚本，适合成规模处理截图
- **踩坑手册**：把部署中真实遇到的每个坑和解法都记录在 [docs/troubleshooting.md](docs/troubleshooting.md)

## 🖥️ 环境要求

| 项目 | 要求 |
|---|---|
| 操作系统 | Windows 10/11（Linux 见文末说明，流程几乎一致） |
| 显卡 | 英伟达显卡，**≥6GB 显存**（4-bit 跑 4B 模型） |
| 驱动 | 较新的 NVIDIA 驱动（支持 CUDA 12.x 及以上） |
| 磁盘 | 至少 15GB 可用空间（模型约 9G + PyTorch 约 3G） |
| Python | 3.11（用 conda 建独立环境，**不要用 3.13/3.14**，PyTorch 生态跟进慢） |

## 🚀 快速开始

### 第 0 步：确认显卡与驱动

```bash
nvidia-smi
```
能出表格、显示显卡型号和 CUDA 版本即可。驱动太旧或命令找不到，见 [踩坑手册 §1](docs/troubleshooting.md#1-nvidia-驱动装不上--装了一半失败)。

### 第 1 步：装 Miniconda 并建独立环境

去 [Miniconda 官网](https://www.anaconda.com/download/success) 下载安装，然后：

```bash
conda create -n vlm python=3.11 -y
conda activate vlm
```
> 行首出现 `(vlm)` 说明进入了独立环境。之后所有命令都在这个环境里执行。

### 第 2 步：装 CUDA 版 PyTorch

**关键：必须装 CUDA 版，不能是 CPU 版。**

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

网络不稳、反复中断时，改用 curl 手动下 whl 再本地装（见 [踩坑手册 §2](docs/troubleshooting.md#2-下载大文件反复中断pytorch-whl--模型权重)）。

装完自检：
```bash
python scripts/check_gpu.py
```
看到 `CUDA 可用: True` 和你的显卡型号才算成功。

### 第 3 步：装其余依赖

```bash
pip install transformers accelerate qwen-vl-utils pillow bitsandbytes
```

### 第 4 步：下载模型

把 `scripts/download_model.bat` 放到想存模型的位置（建议 D 盘），双击运行。它用 curl 断点续传，稳定快速，断了重跑即可。

也可手动下载，直链规律：
```
https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/main/{文件名}
```
> 不建议用 `hf download`：Qwen3-VL 权重走 Xet 存储，配国内镜像常卡死（见 [踩坑手册 §3](docs/troubleshooting.md#3-hf-download-下模型跑一整晚没成)）。

模型约 8.9G，含 2 个 `.safetensors` 权重 + 9 个配置文件，缺一不可。

### 第 5 步：识别！

改 `recognize.py` 顶部的 `DEFAULT_MODEL_DIR` 指向你的模型目录，然后：

```bash
python recognize.py 你的图片.jpg
```

## 📖 用法

### 方式一：命令行单张识别

```bash
# 识别指定图片
python recognize.py D:\path\to\image.jpg

# 自定义提示词（逐字识别、不猜测，精度更高）
python recognize.py image.jpg -p "严格逐字识别图中所有文字，模糊的用【?】标注，不要猜测"

# 指定输出、限制长度、调整看图清晰度
python recognize.py image.jpg -o result.txt --max-tokens 512 --max-pixels 2000000

# 查看全部参数
python recognize.py --help
```

### 方式二：常驻 HTTP 服务 + 网页界面（推荐）

模型只加载一次，之后每次识别秒级返回，还能在浏览器里拖图改提示词。

```bash
pip install fastapi uvicorn python-multipart
python server.py
```

启动后浏览器打开 **http://127.0.0.1:8000**，选图 → 改提示词 → 点识别。

API 调用：
```bash
curl.exe -X POST http://127.0.0.1:8000/recognize -F "file=@image.jpg" -F "prompt=识别图中文字"
```

局域网访问：把 `server.py` 末尾 `host="127.0.0.1"` 改为 `host="0.0.0.0"`，并放行防火墙端口（见 [踩坑手册 §8](docs/troubleshooting.md#8-局域网其它设备访问不到-http-服务)）。

### 方式三：批量识别

先启动服务，再对一个文件夹批量跑：
```bash
python scripts/batch_recognize.py D:\screenshots
```
结果写入每张图同目录的 `xxx_识别结果.txt`。

## 🎯 提升识别精度

小模型 + 4-bit 量化精度有限，可从这几处提升：

1. **调大 `--max-pixels`**：让模型看图更清晰，对密集小字最有效（代价是更吃显存）
2. **逐字提示词**：要求「严格逐字、模糊标注、不要猜测」，减少脑补
3. **换 8-bit 或更大模型**：显存够时精度更高
4. **上更大的卡跑大模型**：见下节

## 📦 项目结构

```
local-vlm-ocr/
├── README.md
├── LICENSE
├── requirements.txt
├── recognize.py              # 命令行识别工具（4-bit 量化，带参数）
├── server.py                 # FastAPI HTTP 服务 + 网页界面
├── scripts/
│   ├── download_model.bat    # curl 断点续传下载模型
│   ├── check_gpu.py          # PyTorch 显卡自检
│   └── batch_recognize.py    # 批量识别客户端
└── docs/
    └── troubleshooting.md    # 踩坑排查手册（重点！）
```

## ⬆️ 扩展到更大的显卡 / 服务器

本项目在 6GB 的 RTX 2060 上「能跑通」，追求高精度可搬到多卡服务器：

- **换大模型**：`Qwen3-VL-32B` 等，识别质量是另一个量级（把模型名/目录换掉即可）
- **多卡推理**：`device_map="auto"` 自动切分到多张卡；高吞吐批量场景推荐用 [vLLM](https://github.com/vllm-project/vllm) 起 OpenAI 兼容服务
- **注意 V100 等老架构不支持 bf16**，统一用 fp16

## 🐧 Linux 说明

流程与 Windows 几乎一致，只是系统命令不同：Miniconda 用 `wget ... && bash` 安装，模型下载用 `curl` 或 `wget`，Python 代码完全通用。前提是先装好 Linux 版 NVIDIA 驱动（`nvidia-smi` 能出表格）。

## 📝 License

[MIT](LICENSE)

## 🙏 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) — 阿里通义千问视觉语言模型
- [HF-Mirror](https://hf-mirror.com) — HuggingFace 国内镜像
- [DDU](https://www.wagnardsoft.com) — 驱动清理工具
