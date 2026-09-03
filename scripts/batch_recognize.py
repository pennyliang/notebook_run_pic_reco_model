# -*- coding: utf-8 -*-
"""
批量识别：调用已启动的 HTTP 服务（server.py），批量处理一个文件夹里的图片。
先启动服务：python server.py
再运行本脚本：python scripts/batch_recognize.py D:\screenshots
结果写入每张图同目录的 xxx_识别结果.txt。
"""
import os
import sys
import requests

API = "http://127.0.0.1:8000/recognize"
PROMPT = "请识别并总结这张图里的全部内容，用中文清晰回答。"
EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    if not os.path.isdir(folder):
        print(f"[错误] 不是有效目录：{folder}")
        sys.exit(1)

    images = [f for f in os.listdir(folder) if f.lower().endswith(EXTS)]
    if not images:
        print(f"[提示] {folder} 里没找到图片。")
        return

    print(f"[信息] 共 {len(images)} 张图，开始批量识别...")
    for i, name in enumerate(images, 1):
        path = os.path.join(folder, name)
        print(f"\n[{i}/{len(images)}] {name}")
        try:
            with open(path, "rb") as f:
                resp = requests.post(
                    API, files={"file": f}, data={"prompt": PROMPT}, timeout=300
                )
            data = resp.json()
            if data.get("ok"):
                result = data["result"]
                out = os.path.splitext(path)[0] + "_识别结果.txt"
                with open(out, "w", encoding="utf-8") as g:
                    g.write(result)
                print("  ✅ 已保存:", out)
            else:
                print("  ❌ 出错:", data.get("error"))
        except Exception as e:
            print("  ❌ 请求失败:", e)

    print("\n[信息] 批量识别完成。")


if __name__ == "__main__":
    main()
