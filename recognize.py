# -*- coding: utf-8 -*-
"""
本地 Qwen3-VL-4B 图片识别工具（4-bit 量化版，带命令行参数）
环境: Windows + RTX 2060 (6GB) + conda 环境 vlm

用法示例：
  python recognize_4bit.py 图片.jpg
  python recognize_4bit.py 图片.jpg -p "只逐字识别图中所有文字，不要猜测"
  python recognize_4bit.py 图片.jpg -o 结果.txt --max-tokens 512
  python recognize_4bit.py --help        # 查看全部参数
"""

import os
import sys
import argparse
import torch

# 默认值（不传参时用这些）
DEFAULT_MODEL_DIR = r"D:\models\Qwen3-VL-4B-Instruct"
DEFAULT_IMAGE = r"D:\esp32\ffmpeg\output2.jpg"
DEFAULT_PROMPT = (
    "请识别并总结这张图里的全部内容："
    "包括顶部搜索关键词、所有标签栏文字、每条笔记的标题和作者、点赞数，"
    "以及图片里花束的样子。用中文分条清晰回答。"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="本地 Qwen3-VL 图片识别工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "image", nargs="?", default=DEFAULT_IMAGE,
        help="要识别的图片路径",
    )
    parser.add_argument(
        "-p", "--prompt", default=DEFAULT_PROMPT,
        help="提问内容 / 提示词",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="结果保存的 txt 路径（默认存到图片同目录，文件名加 _识别结果）",
    )
    parser.add_argument(
        "-m", "--model", default=DEFAULT_MODEL_DIR,
        help="本地模型目录",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=1024,
        help="生成的最大长度，显存紧可调小（如 512）",
    )
    parser.add_argument(
        "--max-pixels", type=int, default=1280 * 28 * 28,
        help="模型看图的像素上限，调大更清晰但更吃显存",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.model):
        print(f"[错误] 找不到模型目录：{args.model}")
        sys.exit(1)
    if not os.path.isfile(args.image):
        print(f"[错误] 找不到图片：{args.image}")
        sys.exit(1)
    if not torch.cuda.is_available():
        print("[错误] 没检测到 CUDA 显卡。")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"[信息] 显卡：{gpu_name}，显存 {total_mem:.1f} GB")
    print(f"[信息] 识别图片：{args.image}")

    try:
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
    except ImportError:
        from transformers import AutoProcessor, BitsAndBytesConfig
        from transformers import AutoModelForImageTextToText as Qwen3VLForConditionalGeneration

    from qwen_vl_utils import process_vision_info

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    print("[信息] 正在以 4-bit 量化加载模型...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        local_files_only=True,
    )
    # max_pixels 控制看图清晰度，对密集小字的识别精度影响很大
    processor = AutoProcessor.from_pretrained(
        args.model, local_files_only=True, max_pixels=args.max_pixels
    )
    print("[信息] 模型加载完成，开始识别...")

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": args.image},
                {"type": "text", "text": args.prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    ).to("cuda:0")

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)

    trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, generated_ids)]
    result = processor.batch_decode(
        trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    print("\n" + "=" * 40)
    print("识别结果")
    print("=" * 40 + "\n")
    print(result)

    out_txt = args.output or (os.path.splitext(args.image)[0] + "_识别结果.txt")
    try:
        with open(out_txt, "w", encoding="utf-8") as f:
            f.write(result)
        print(f"\n[信息] 结果已保存到：{out_txt}")
    except Exception as e:
        print(f"\n[提示] 结果保存失败（不影响识别）：{e}")


if __name__ == "__main__":
    try:
        main()
    except torch.cuda.OutOfMemoryError:
        print("\n[显存不足] 把 --max-tokens 调小（如 512），或把 --max-pixels 调小。")
    except Exception as e:
        print(f"\n[出错] {type(e).__name__}: {e}")
        print("把上面这行报错发给我，我帮你定位。")
