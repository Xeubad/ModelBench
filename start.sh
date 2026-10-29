#!/bin/bash
# 启动模型测试平台（只启动服务，不自动打开浏览器；地址会打印在终端里，自己在浏览器打开）。
# 端口和监听地址见 config.json（当前 127.0.0.1:18889；18888 被 VS Code 占用），Ctrl+C 退出。
# 想启动后自动打开浏览器：./start.sh --open
cd "$(dirname "$0")" && exec python3 -m model_test serve "$@"
