\# 踩坑排查手册



这份手册记录了本项目部署过程中真实遇到的每一个坑和解法。按现象查即可。



\## 1. NVIDIA 驱动装不上 / 装了一半失败



\*\*现象\*\*：运行官方驱动安装包，走到「安装」阶段报「NVIDIA 安装程序失败」，图形驱动程序状态为「失败」。若勾了「清洁安装」，旧驱动已被卸载、新驱动又没装上，外接显示器会黑屏没信号。



\*\*原因\*\*：旧驱动残留 + 安装时有进程占用，官方精简/自定义安装都清不干净。



\*\*解法（DDU 彻底清除，基本必成）\*\*：

1\. 先下好 DDU（Display Driver Uninstaller，官网 wagnardsoft.com 或 guru3d.com）和官方驱动包，放本地。

2\. 进安全模式：设置 → 系统 → 恢复 → 高级启动 → 疑难解答 → 高级选项 → 启动设置 → 重启 → 按 4。

3\. 运行 DDU → 选 GPU / NVIDIA → 「清除并重启」。

4\. 回正常模式，暂停 Windows Update、关杀毒，管理员运行官方驱动包安装。



\*\*备选\*\*：官方包始终失败时，用 \*\*NVIDIA App\*\* 自动检测安装驱动往往能成（前提是残留已被 DDU 清掉）。



\*\*验证\*\*：`nvidia-smi` 能出表格、显示正确 Driver 版本和 CUDA 版本即成功。外接屏信号会随驱动装好自动恢复。



\## 2. 下载大文件反复中断（PyTorch whl / 模型权重）



\*\*现象\*\*：`pip install torch` 下到一半断，反复续传到 attempt 5；或 `WinError 32 文件被占用`。



\*\*原因\*\*：海外源下大文件不稳；杀毒软件扫描落地文件时锁文件。



\*\*解法（用 curl 手动下，最稳）\*\*：

```

curl.exe -L -C - "文件直链" -o 输出文件名

```

\- `-C -` 断点续传，断了重跑同一条即可

\- `-L` 跟随跳转

\- 实测 2.35G 的 torch 用 curl 约 8 分钟下完，比 pip 挂一小时强



\*\*注意\*\*：PowerShell 里 `wget` 是 `Invoke-WebRequest` 的别名，参数不兼容（`-c` 会报「二义性」）。用 `curl.exe`（带 .exe 后缀绕开别名），或换 CMD 窗口。



\## 3. hf download 下模型跑一整晚没成



\*\*现象\*\*：`hf download`（旧命令 `huggingface-cli` 已废弃）跑一整晚，目录里只有 `.cache`，一个权重文件都没落地。



\*\*原因\*\*：Qwen3-VL 权重用 Xet 分块存储，`hf` 走 Xet 协议 + 国内镜像经常卡死。



\*\*解法\*\*：绕开 hf，直接用 curl 按文件下（见 `scripts/download\_model.bat`）。hf-mirror 直链规律：

```

https://hf-mirror.com/{模型名}/resolve/main/{文件名}

```



\## 4. 模型加载后识别卡住不动（好几分钟）



\*\*现象\*\*：日志出现 `Some parameters are on the meta device because they were offloaded to the disk and cpu`，然后卡住。



\*\*原因\*\*：fp16 的 4B 模型约需 8-9G，6G 显存装不下，`device\_map="auto"` 把部分权重甩到硬盘/CPU，推理时数据在显卡↔硬盘间搬运，慢到几乎不动。



\*\*解法\*\*：改用 \*\*4-bit 量化\*\*，把模型压到约 4G，全部塞进显卡：

```python

from transformers import BitsAndBytesConfig

quant\_config = BitsAndBytesConfig(

&#x20;   load\_in\_4bit=True,

&#x20;   bnb\_4bit\_compute\_dtype=torch.float16,   # 2060/V100 用 fp16，不支持 bf16

&#x20;   bnb\_4bit\_quant\_type="nf4",

&#x20;   bnb\_4bit\_use\_double\_quant=True,

)

model = ...from\_pretrained(..., quantization\_config=quant\_config, device\_map="cuda:0")

```

需先 `pip install bitsandbytes`。本项目 `recognize.py` 和 `server.py` 已默认走 4-bit。



\## 5. torchvision 导入报 No module named 'PIL'



\*\*原因\*\*：缺 Pillow。\*\*解法\*\*：`pip install pillow`（导入名是 `PIL`）。



\## 6. Bad metadata / pip 安装崩溃



\*\*现象\*\*：`invalid metadata entry 'name'` 一大段 traceback。



\*\*原因\*\*：旧版 pip 解析 bug，或中断残留。



\*\*解法\*\*：升级 pip `python -m pip install --upgrade pip`；检查 `site-packages` 里有无 `\~` 开头的残留文件夹，有就删。



\## 7. C 盘空间不够



\*\*解法\*\*：装 Miniconda / 存模型前，先把这些指到 D 盘：

```

set HF\_HOME=D:\\hf\_cache

```

conda 环境和模型都存 D 盘。切勿给 conda 目录设「只读」属性，否则后续安装会因写不进而失败。



\## 8. 局域网其它设备访问不到 HTTP 服务



\*\*现象\*\*：`http://本机IP:8000` 打不开。



\*\*原因一\*\*：`server.py` 里 `host="127.0.0.1"` 只允许本机。改成 `host="0.0.0.0"` 并重启服务。



\*\*原因二\*\*：Windows 防火墙拦截。管理员 PowerShell 放行端口：

```powershell

New-NetFirewallRule -DisplayName "Qwen-VL-8000" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow

```



\## 9. 跑脚本报 ModuleNotFoundError: No module named 'torch'（或 transformers 等）



\*\*现象\*\*：`python server.py` / `recognize.py` 报找不到 torch、transformers 等模块，但明明装过。命令行行首显示 `(base)`。



\*\*原因\*\*：不在正确的 conda 环境里。torch 等依赖装在 `vlm` 环境，`(base)` 里没有。关掉窗口重开后，默认会回到 `(base)`，最容易忘记激活。



\*\*解法\*\*：运行任何项目脚本前，先激活环境：

```bash

conda activate vlm

```

确认行首变成 `(vlm)` 再运行。这是每次开新窗口都要做的第一步。

