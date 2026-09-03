# -*- coding: utf-8 -*-
"""快速自检：确认 PyTorch 能用上显卡。装完 PyTorch 后先跑这个。"""
import torch

print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("显卡型号:", torch.cuda.get_device_name(0))
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"显存总量: {total:.1f} GB")
    print("CUDA 版本:", torch.version.cuda)
    print("\n✅ 显卡接通，可以继续部署模型。")
else:
    print("\n❌ 没接上显卡。常见原因：")
    print("  1) 装成了 CPU 版 PyTorch —— 用 cu124 的 index-url 重装")
    print("  2) NVIDIA 驱动太旧 —— 更新到最新版")
    print("  3) 不在正确的 conda 环境里 —— 先 conda activate vlm")
