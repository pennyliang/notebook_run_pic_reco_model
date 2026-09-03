@echo off
chcp 65001 >nul
REM =====================================================================
REM  下载 Qwen3-VL-4B-Instruct 模型到本地（用 curl，断点续传，稳定快速）
REM  用法：把本文件放到想存模型的目录，双击运行，或命令行执行
REM  断了重跑本脚本即可：已下好的文件会自动跳过
REM =====================================================================

REM 模型保存目录（按需修改）
set TARGET=D:\models\Qwen3-VL-4B-Instruct
REM 国内镜像地址前缀
set BASE=https://hf-mirror.com/Qwen/Qwen3-VL-4B-Instruct/resolve/main

if not exist "%TARGET%" mkdir "%TARGET%"
cd /d "%TARGET%"

echo ========== 下载权重文件（大，约 8.9G） ==========
curl.exe -L -C - "%BASE%/model-00001-of-00002.safetensors" -o model-00001-of-00002.safetensors
curl.exe -L -C - "%BASE%/model-00002-of-00002.safetensors" -o model-00002-of-00002.safetensors

echo ========== 下载配置文件（小，缺一不可） ==========
curl.exe -L "%BASE%/config.json"                     -o config.json
curl.exe -L "%BASE%/generation_config.json"          -o generation_config.json
curl.exe -L "%BASE%/model.safetensors.index.json"    -o model.safetensors.index.json
curl.exe -L "%BASE%/chat_template.json"              -o chat_template.json
curl.exe -L "%BASE%/tokenizer_config.json"           -o tokenizer_config.json
curl.exe -L "%BASE%/tokenizer.json"                  -o tokenizer.json
curl.exe -L "%BASE%/vocab.json"                      -o vocab.json
curl.exe -L "%BASE%/merges.txt"                      -o merges.txt
curl.exe -L "%BASE%/preprocessor_config.json"        -o preprocessor_config.json

echo.
echo ========== 完成，当前目录文件列表 ==========
dir
echo.
echo 检查：应有 2 个 .safetensors 大文件 + 9 个小文件，总大小约 8.9G
pause
