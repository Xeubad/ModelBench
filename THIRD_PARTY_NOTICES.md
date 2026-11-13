# 第三方声明 / Third-party notices

## ModelTrace

本项目的 `model_test/modeltrace.py` 依据 ModelTrace 的 `fingerprint.py` 和
`static/fingerprint-core.js` 改写为 Python 标准库实现，挑战文本改编自其
`generate_challenges`。`model_test/modeltrace_suite.py` 保留上游 36 条采集挑战，
`model_test/modeltrace_training.py` 基于上游 `bank_builder.py` 改写并增加本地溯源与留出评估。
`model_test/assets/modeltrace_unified_bank.json` 为上游参考库快照。

- 上游：https://github.com/xqy2006/ModelTrace
- 上游提交：`6cd4a8f`（2026-10-09）
- 参考库 SHA-256：`f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d`
- 许可：MIT，原始许可全文如下。

```text
MIT License

Copyright (c) 2026 xqy2006

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## meow-llm-detector

本项目参考了该项目的测量思想和公开技术文档；未包含其软件代码、探针或基准数据。
上游：https://github.com/chen-006/meow-llm-detector ，许可为 PolyForm Noncommercial 1.0.0。
本项目新增的短问题行为分布场景不是该项目的官方基准或评分实现。
`model_test/meow_bridge.py` 通过本机 HTTP API 与用户独立运行的 meow 通信，不分发、安装或复制 meow 的代码或数据；
使用独立 meow 及其基准时须遵守其许可。
