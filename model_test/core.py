#!/usr/bin/env python3
"""模型测试引擎（网页和命令行共用）：支持 OpenAI、Anthropic、Gemini、Azure 和 Bedrock 接口。

做了什么：
- 接口类型：responses、chat、anthropic、gemini、azure、bedrock；按协议构造路径与鉴权。
- 公平性：每个接口复用一条连接（TCP/TLS 握手单独记录，不算进请求耗时）；每个模型先发一次不计入的预热请求；
  每一轮把“全部模型 × 全部题目”随机打乱顺序再发（种子保存在结果里，可复现）；同一接口两次请求之间留间隔，避免 429。
- 计时拆开：连接、响应头、首个事件、首个思考、首个正文字、总耗时、输出速度（tokens/秒）。
- 场景：speed（自定义提示词）、quiz（10 道有标准答案的题，自动判分）、tool（两步工具调用，含把 reasoning 项回放）、
  needle10k / needle30k（大海捞针）、ModelTrace 候选和短问题分布。判分不让模型自评。
- 统计：中位数、p95、最小、最大；样本 < 5 标“样本不足”；固定题集仅报告描述性正确率和一致性。
直接请求用户配置的模型接口，不依赖其他应用；引擎本身不保存密钥。
时间度量一律用单调时钟 time.perf_counter()（不受系统校时影响），只有记录“几点几分”用墙上时钟。
"""
import datetime as dt
import concurrent.futures
import hashlib
import http.client
import json
import math
import platform
import random
import re
import statistics
import sys
import threading
import time
import urllib.parse

from . import azure, bedrock, behavior, custom_http, gemini, modeltrace, modeltrace_suite

VERSION = "2.11.0"
SCHEMA_VERSION = 3
METHOD_VERSION = "7"  # 变更请求、判分或测量口径时更新，旧结果不自动合并
CONTEXT_CHOICES = (0, 10000, 30000)  # 自定义提示词前的背景文字长度（约 tokens）；0 = 不加
LOW_SAMPLE = 5
P95_MIN_N = 20  # 样本少于这个数，p95 基本等于最大值，不展示
SLO_SECONDS = 10  # 描述性口径：10 秒内最终完成且有判分时答对
BUFFER_MIN_TOTAL = 3.0  # 总耗时达到这个秒数才判断“非流式”（太短的请求流式与否分不出）
EFFORTS = ("", "low", "medium", "high")  # 推理强度；空 = 不指定，由模型自己决定
# 固定采样参数：让“同一模型在不同接口/中转”之间可比（否则差异可能来自接口默认值而不是模型）。
# temperature=0 让输出尽量确定，速度与正确率都不受随机性放大影响；max_tokens 给足，避免长回答被截断误判成失败。
SAMPLING = {"temperature": 0}
# 输出上限：chat 用 max_tokens，responses 用 max_output_tokens；给足长度，避免长回答被截断后误判成“答案不符”
MAX_TOKENS = 4096
SAMPLING_CHAT = dict(SAMPLING, max_tokens=MAX_TOKENS)
SAMPLING_RESPONSES = dict(SAMPLING, max_output_tokens=MAX_TOKENS)
DEFAULT_PROMPT = "只回复：好"
SCENARIOS = {
    "speed": "自定义提示词（速度）",
    "gen": "生成速度（固定约 300 字）",
    "quiz": "题库（正确率）",
    "quiz_extended": "多领域题集（8题）",
    "tool": "工具调用（两步）",
    "needle10k": "大海捞针（约 1 万 tokens）",
    "needle30k": "大海捞针（约 3 万 tokens）",
    "needle_matrix": "长上下文矩阵（12题，费用较高）",
    "fingerprint": "模型指纹（数值分布对账）",
    "modeltrace": "ModelTrace 模型候选（每轮 3 次长数字挑战）",
    "modeltrace_enroll": "ModelTrace 参考库采集（每模型 36 次）",
    "behavior": "短问题行为分布（每轮 4 次）",
}
_FILLER_UNIT = "长汀县汀江流域河田段农业面源污染治理项目采用MET混合滤料包与土建安装相结合的方式，招标文件要求投标人提供CMA检测报告并加盖公章。"  # 约 41 tokens
NEEDLE = "【内部备忘】本次测试的暗号是：青竹-7392，请牢记。"

# 题库：ans 是标准答案（自检和单元测试用它验证判分器本身），chk 是判分方式
QUIZ = [
    {"id": "q01", "name": "金额大写", "prompt": "把“2,520,000.00 元”改写成中文大写金额，只输出结果，不要解释。",
     "chk": ("re", r"(人民币)?贰佰伍拾贰万元整"), "ans": "贰佰伍拾贰万元整"},
    {"id": "q02", "name": "数字提取", "prompt": "从下面文字中提取投标报价，只输出数字，不带单位和千分位逗号：“本项目投标报价为人民币1,386,500.50元（含税）。”",
     "chk": ("re", r"1386500\.50?"), "ans": "1386500.50"},
    {"id": "q03", "name": "JSON 格式", "prompt": "只输出一个 JSON 对象，不要代码块和任何其他文字：name 为“张三”，age 为数字 30。",
     "chk": ("json", {"name": "张三", "age": 30}), "ans": '{"name": "张三", "age": 30}', "strict": True},
    {"id": "q04", "name": "指令遵循", "prompt": "请只输出苹果、香蕉、橙子这三种水果的名称，用英文逗号分隔，顺序不限，不要有其他任何文字、空格或标点。",
     "chk": ("set", ["苹果", "香蕉", "橙子"]), "ans": "苹果,香蕉,橙子", "strict": True},
    {"id": "q05", "name": "千分位不拆列", "prompt": "按竖线 | 拆分成列，输出 JSON 字符串数组，不要代码块和其他文字：设备名称|1,200.00|台",
     "chk": ("json", ["设备名称", "1,200.00", "台"]), "ans": '["设备名称", "1,200.00", "台"]', "strict": True},
    {"id": "q06", "name": "算术", "prompt": "17 乘以 23 等于多少？只输出数字。", "chk": ("re", r"391"), "ans": "391"},
    {"id": "q07", "name": "日期推算", "prompt": "2026年10月12日是星期一，那么2026年10月19日是星期几？只输出星期几。",
     "chk": ("re", r"(星期一|周一)"), "ans": "星期一"},
    {"id": "q08", "name": "逻辑推理", "prompt": "小明比小红高，小红比小刚高。三个人里最矮的是谁？只输出名字。", "chk": ("re", r"小刚"), "ans": "小刚"},
    {"id": "q09", "name": "日期抽取", "prompt": "招标文件规定：投标保证金为人民币伍万元整，开标时间为2026年11月3日上午9时30分。请只输出开标日期，格式 YYYY-MM-DD。",
     "chk": ("re", r"2026-11-03"), "ans": "2026-11-03"},
    {"id": "q10", "name": "严格输出", "prompt": "请只回复一个大写字母 A，不要有其他任何字符。", "chk": ("re", r"A"), "ans": "A", "strict": True},
]
QUIZ_EXTENDED = [
    {"id": "e01", "name": "百分比计算", "prompt": "原价 240 元，打八折后是多少元？只输出数字。", "chk": ("re", r"192"), "ans": "192"},
    {"id": "e02", "name": "单位换算", "prompt": "3.5 小时等于多少分钟？只输出数字。", "chk": ("re", r"210"), "ans": "210"},
    {"id": "e03", "name": "集合逻辑", "prompt": "所有方块都是图形，甲是方块。甲是不是图形？只输出是或否。", "chk": ("re", r"是"), "ans": "是"},
    {"id": "e04", "name": "日期边界", "prompt": "2024 年 2 月 28 日的下一天是哪一天？只输出 YYYY-MM-DD。", "chk": ("re", r"2024-02-29"), "ans": "2024-02-29"},
    {"id": "e05", "name": "英文阅读", "prompt": "Read: The red box is empty. The blue box contains a key. Which box contains the key? Reply with only its color in lowercase English.", "chk": ("re", r"blue"), "ans": "blue", "strict": True},
    {"id": "e06", "name": "数组处理", "prompt": "对数字 3,1,2 从小到大排序，仅输出 JSON 数组。", "chk": ("json", [1, 2, 3]), "ans": "[1,2,3]", "strict": True},
    {"id": "e07", "name": "否定识别", "prompt": "记录：会议不在周二举行，改为周三。会议在哪天举行？只输出星期几。", "chk": ("re", r"(星期三|周三)"), "ans": "周三"},
    {"id": "e08", "name": "信息缺失", "prompt": "记录中只有：小王有两支笔。小王的笔是什么颜色？如果未提及，只输出未知。", "chk": ("re", r"未知"), "ans": "未知", "strict": True},
]
GEN_PROMPT = "请写一段约 300 字的中文，说明污水处理中 A/O（缺氧/好氧）工艺的基本原理。直接写正文，不要标题和列表。"
TOOL_DEF = {"name": "lookup_price", "description": "查询某种材料的单价",
            "parameters": {"type": "object", "properties": {"item": {"type": "string", "description": "材料名称"}}, "required": ["item"]}}
TOOL_PROMPT = '请调用工具查询“MET混合滤料包”的单价。拿到结果后只输出 JSON 对象，恰好包含数字 price 和字符串 unit 两个字段，不要其他内容。'
TOOL_RESULT = {"item": "MET混合滤料包", "price": 128.5, "unit": "元/包"}

# ---------------------------------------------------------------- 模型指纹（数值分布对账）
# 思路参考开源项目 ModelTrace（MIT）与 meow-llm-detector（只借思想、自写实现、不引入依赖）：
# 输出数值分布的 Hellinger 距离仅作启发式描述，阈值未经真实模型校准，不可认证身份。
FP_COUNT = 40          # 每条挑战让模型输出的整数个数
FP_RANGE = (1, 355)    # 整数取值范围
FP_BINS = 20           # 把 [1, 355] 分成多少个等宽区间统计分布
FP_MIN_NUMBERS = 120   # 一个模型的有效数字少于这个数 → 拒判（约一轮 3×40）
FP_NOISE_FACTOR = 2.0  # 判定线 = max(自身抽样噪声 × 这个系数, FP_ABS_FLOOR)
FP_ABS_FLOOR = 0.10    # 判定线下限：样本再多，“抽样噪声 ≈ 0”时也不把微小差异当证据
_FP_ANTI = ("必须由当前模型直接完成，不要调用任何工具、代码或外部随机数。每个位置独立凭第一反应选择，"
            "不要计数、递增、递减、等差、循环或套用任何规律；偶然重复是有效的，写出后不要排序、去重或修改。")
FP_PROMPTS = [
    f"这是一次无语义数值测试。请为 {FP_COUNT} 个位置各凭第一反应选一个 {FP_RANGE[0]} 到 {FP_RANGE[1]} 之间（含端点）的整数。{_FP_ANTI}只输出 {FP_COUNT} 个整数，用逗号或空格分隔，不要输出任何其他文字。",
    f"请连续给出 {FP_COUNT} 个 {FP_RANGE[0]} 到 {FP_RANGE[1]}（含端点）的整数，每一项都独立快速决定。{_FP_ANTI}只输出数字序列，不要解释。",
    f"现在做 {FP_COUNT} 次无语义取值：每次在闭区间 [{FP_RANGE[0]}, {FP_RANGE[1]}] 内任选一个整数。{_FP_ANTI}输出 {FP_COUNT} 个整数即可，不要有其他内容。",
]


# ---------------------------------------------------------------- 地址与连接

def endpoint_parts(base_url):
    """(scheme, host, port, path 前缀)。没写版本号补 /v1；误带了 /chat/completions 等结尾会去掉。"""
    u = urllib.parse.urlsplit((base_url or "").strip())
    if u.scheme not in ("http", "https") or not u.hostname:
        raise ValueError("地址必须以 http:// 或 https:// 开头")
    if u.username is not None or u.password is not None or u.query or u.fragment:
        raise ValueError("接口地址不能包含用户信息、查询参数或片段，请把密钥填在密钥栏")
    path = u.path.rstrip("/")
    for tail in ("/chat/completions", "/responses", "/models", "/messages"):
        if path.endswith(tail):
            path = path[: -len(tail)]
    if not re.search(r"/v\d+\w*$", path):
        path += "/v1"
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), path


def clip(text, n=300):
    return (text or "").replace("\n", " ")[:n]


def redact_text(value, secret):
    """隐藏已知密钥的原文和 URL 编码；短密钥只按独立词匹配以降低误伤。"""
    if not isinstance(value, str) or not secret:
        return value
    variants = {secret, urllib.parse.quote(secret, safe=""), urllib.parse.quote_plus(secret, safe="")}
    for token in sorted(variants, key=len, reverse=True):
        if len(token) < 8:
            value = re.sub(r"(?<!\w)" + re.escape(token) + r"(?!\w)", "[已隐藏密钥]", value)
        else:
            value = value.replace(token, "[已隐藏密钥]")
    return value


def _redact_value(value, secret, mask_keys=False):
    if isinstance(value, str):
        return redact_text(value, secret)
    if isinstance(value, list):
        return [_redact_value(item, secret, mask_keys) for item in value]
    if isinstance(value, dict):
        return {(redact_text(key, secret) if mask_keys and isinstance(key, str) else key): _redact_value(item, secret, mask_keys)
                for key, item in value.items()}
    return value


def _response_fragments(value):
    """汇集常见流协议的文本分片，检测跨事件拆分的密钥。"""
    if isinstance(value, list):
        for item in value:
            yield from _response_fragments(item)
    elif isinstance(value, dict):
        for key, item in value.items():
            if isinstance(item, str) and key in ("content", "text", "reasoning", "reasoning_content", "thinking", "partial_json", "message", "arguments", "delta"):
                yield item
            elif isinstance(item, (dict, list)):
                yield from _response_fragments(item)


def _redact_record(rec, secret):
    if not secret:
        return rec
    events = rec.get("events") or []
    replay = rec.get("_replay") or {}
    fragments = "".join(_response_fragments(events))
    candidates = (rec.get("text") or "", fragments, json.dumps(events, ensure_ascii=False),
                  json.dumps(replay, ensure_ascii=False), json.dumps(rec.get("tool_calls") or [], ensure_ascii=False))
    sensitive_output = any(redact_text(text, secret) != text for text in candidates)
    if sensitive_output:
        # 流事件可能把密钥拆成多个片段，逐项替换不能保证无法重组。
        rec["events"] = [{"t": event.get("t"), "event": {"type": "redacted"}} for event in events]
        rec["ok"] = False
        rec["error"] = "响应包含已隐藏的密钥；为避免污染评测，本次不计成功"
        if rec.get("pass") is not None:
            rec["pass"], rec["grade_note"] = False, rec["error"]
        rec["sensitive_output_redacted"] = True
        rec["_replay"] = {"items": [], "reasoning": "", "text": ""}
    for key in list(rec):
        if not key.startswith("_"):
            rec[key] = _redact_value(rec[key], secret, mask_keys=True)
    return rec


def _new_conn(ep, timeout):
    scheme, host, port, _ = endpoint_parts(ep["base_url"])
    conn = (http.client.HTTPSConnection if scheme == "https" else http.client.HTTPConnection)(host, port, timeout=timeout)
    t = time.perf_counter()
    conn.connect()
    return conn, round(time.perf_counter() - t, 3)


class Pool:
    """每个主机一条复用连接。服务端断开后自动重建（重建耗时记为 connect_s，不算进请求耗时）。"""

    def __init__(self):
        self.conns = {}

    def _key(self, ep):
        s, h, p, _ = endpoint_parts(ep["base_url"])
        return (s, h, p)

    def get(self, ep, timeout):
        key = self._key(ep)
        c = self.conns.get(key)
        if c is not None and c.sock is not None:
            c.sock.settimeout(timeout)
            return c, True, 0.0
        c, cs = _new_conn(ep, timeout)
        self.conns[key] = c
        return c, False, cs

    def drop(self, ep):
        c = self.conns.pop(self._key(ep), None)
        if c is not None:
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass

    def close(self):
        for c in list(self.conns.values()):
            try:
                c.close()
            except Exception:  # noqa: BLE001
                pass
        self.conns.clear()


def label_of(ep, model):
    return f"{ep['name']}/{model}"


def _auth_headers(ep):
    """本地兼容服务可不使用密钥；有密钥时按接口协议传递。"""
    key = ep.get("api_key") or ""
    if ep.get("api") == "anthropic":
        return {**({"x-api-key": key} if key else {}), "anthropic-version": "2023-06-01"}
    if ep.get("api") == "gemini":
        return {"x-goog-api-key": key} if key else {}
    if ep.get("api") == "bedrock":
        return {}  # AWS SigV4 在请求准备时生成。
    if ep.get("api") == "azure" and ep.get("azure_auth", "api_key") == "api_key":
        return {"api-key": key} if key else {}
    if ep.get("api") in ("chat", "responses", "auto"):
        mode = ep.get("auth_mode", "bearer")
        if mode == "custom":
            custom_http.validate_auth(mode, ep.get("auth_header", ""), ep.get("auth_prefix", ""))
            return {ep["auth_header"]: ep.get("auth_prefix", "") + key} if key else {}
        if mode in ("api-key", "x-api-key"):
            return {mode: key} if key else {}
    return {"Authorization": "Bearer " + key} if key else {}


def fetch_models(ep, timeout=20):
    """读取完整模型列表；分页失败或游标循环时抛错，不返回可能误导的部分结果。"""
    _, _, _, prefix = endpoint_parts(ep["base_url"])
    styles = ("responses", "anthropic") if ep.get("api") == "auto" else (ep.get("api"),)
    deadline = time.perf_counter() + timeout

    def request(style, path):
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            raise TimeoutError("读取模型列表超时")
        conn, _ = _new_conn(ep, remaining)
        try:
            if ep.get("api") in ("auto", "chat", "responses", "anthropic"):
                path = custom_http.append_query(path, ep.get("query_params"))
            conn.request("GET", path, headers={**_auth_headers(dict(ep, api=style)), "Accept": "application/json",
                                               "User-Agent": "modelbench/" + VERSION})
            resp = conn.getresponse()
            raw = resp.read(2_000_001)
            if len(raw) > 2_000_000:
                raise ValueError("单页模型列表过大")
            return resp.status, raw.decode("utf-8", "replace")
        finally:
            conn.close()

    if ep.get("api") == "gemini":
        return gemini.fetch_models(ep, request, redact_text)
    if ep.get("api") == "azure":
        raise ValueError("Azure 请手动填写部署名称；模型目录不能代表可用部署")
    if ep.get("api") == "bedrock":
        return bedrock.fetch_models(ep, timeout, _new_conn, redact_text)
    for index, style in enumerate(styles):
        status, body = request(style, prefix + "/models")
        if status == 200:
            break
        if index + 1 == len(styles) or status not in (400, 401, 403, 404, 405):
            raise RuntimeError(f"HTTP {status} {clip(redact_text(body, ep.get('api_key')), 160)}")
    found, seen_cursors = set(), set()
    for _ in range(50):
        data = json.loads(body)
        if isinstance(data, list):
            items = data
        elif isinstance(data, dict) and isinstance(data.get("data"), list):
            items = data["data"]
        elif isinstance(data, dict) and isinstance(data.get("models"), list):
            items = data["models"]
        else:
            raise ValueError("接口返回的模型列表格式不正确")
        for item in items:
            model = (item.get("id") or item.get("name")) if isinstance(item, dict) else item
            if isinstance(model, str) and model and redact_text(model, ep.get("api_key")) == model:
                found.add(model)
        if len(found) > 5000:
            raise ValueError("模型列表超过 5000 个，请手动配置")
        if not isinstance(data, dict) or not data.get("has_more"):
            return sorted(found)
        cursor = data.get("last_id")
        if not isinstance(cursor, str) or not cursor or len(cursor) > 512 or cursor in seen_cursors or redact_text(cursor, ep.get("api_key")) != cursor:
            raise ValueError("模型分页游标缺失、无效或重复")
        seen_cursors.add(cursor)
        key = "after_id" if style == "anthropic" else "after"
        status, body = request(style, prefix + "/models?" + urllib.parse.urlencode({key: cursor}))
        if status != 200:
            raise RuntimeError(f"读取模型列表后续页面失败：HTTP {status} {clip(redact_text(body, ep.get('api_key')), 160)}")
    raise ValueError("模型列表页数超过上限")


# ---------------------------------------------------------------- 单次流式请求

def with_effort(body, api, effort):
    """推理强度：responses 传 reasoning.effort，chat 传 reasoning_effort；空字符串 = 不传。"""
    if effort:
        if api in ("chat", "azure"):
            body["reasoning_effort"] = effort
        elif api == "anthropic":
            body["output_config"] = {"effort": effort}
        else:
            body["reasoning"] = {"effort": effort}
    return body


def make_body(api, model, prompt, system=None, effort=""):
    """构造请求体。
    采样参数显式固定：不同中转站/接口的默认 temperature、max_tokens 往往不同，
    不固定的话“同一模型不同接口”的差异里会混进采样参数差异，不是模型本身差异。
    固定值同时写进环境快照（SAMPLING），复现和对比时可核对。"""
    if api == "gemini":
        if effort:
            raise ValueError("Gemini 原生接口暂不支持统一推理强度参数")
        return gemini.make_body(model, prompt, system, MAX_TOKENS)
    if api == "bedrock":
        if effort:
            raise ValueError("Bedrock 原生接口暂不支持统一推理强度参数")
        return bedrock.make_body(model, prompt, system, MAX_TOKENS)
    if api in ("chat", "azure"):
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return with_effort({"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True}, **SAMPLING_CHAT}, api, effort)
    if api == "anthropic":
        body = {"model": model, "messages": [{"role": "user", "content": prompt}], "stream": True, **SAMPLING_CHAT}
        if system:
            body["system"] = system
        return with_effort(body, api, effort)
    return with_effort({"model": model, "stream": True,
                        "input": ([{"role": "developer", "content": system}, {"role": "user", "content": prompt}] if system else prompt),
                        **SAMPLING_RESPONSES}, api, effort)


def _new_rec(ep, model, api):
    return {"time": dt.datetime.now().isoformat(timespec="seconds"), "endpoint": ep["name"], "model": model,
            "label": label_of(ep, model), "api": api, "ok": False, "status": None, "error": "", "reused": None, "first_reused": None,
            "connect_s": None, "headers_s": None, "first_event_s": None, "first_think_s": None, "first_token_s": None,
            "total_s": None, "tps": None, "tps_overall": None, "buffered": False, "non_stream": False, "queue_s": None,
             "reasoning_tokens": None, "response_status": None, "text": "", "tool_calls": [], "usage": None, "events": []}


def _chat_cached_tokens(usage):
    if usage.get("prompt_cache_hit_tokens") is not None:
        return usage["prompt_cache_hit_tokens"]
    return (usage.get("prompt_tokens_details") or {}).get("cached_tokens")


def _anthropic_usage(existing, data):
    """Messages 的 input_tokens 不包含缓存读取和创建量；output_tokens 增量覆盖。"""
    if not isinstance(data, dict):
        return existing
    usage = dict(existing or {})
    if "input_tokens" in data:
        usage["input_tokens"] = (data["input_tokens"] + (data.get("cache_read_input_tokens") or 0)
                                 + (data.get("cache_creation_input_tokens") or 0))
        usage["cached_tokens"] = data.get("cache_read_input_tokens")
        usage["cache_creation_input_tokens"] = data.get("cache_creation_input_tokens")
    if "output_tokens" in data:
        usage["output_tokens"] = data["output_tokens"]
    details = data.get("output_tokens_details") or {}
    if "thinking_tokens" in details:
        usage["reasoning_tokens"] = details["thinking_tokens"]
    return usage


def _consume_anthropic(rec, ev, acc, now):
    kind = ev.get("type")
    blocks = acc.setdefault("anthropic_blocks", {})
    opened = acc.setdefault("anthropic_open", set())
    index = ev.get("index", 0)
    if kind == "message_start":
        if acc.get("anthropic_started"):
            rec["error"] = "重复的 message_start 事件"
        acc["anthropic_started"] = True
        message = ev.get("message") or {}
        rec["returned_model"] = message.get("model")
        rec["usage"] = _anthropic_usage(rec["usage"], message.get("usage"))
    elif kind == "content_block_start":
        if not acc.get("anthropic_started") or index in blocks:
            rec["error"] = "无效的 content_block_start 事件"
            return
        block = dict(ev.get("content_block") or {})
        blocks[index] = block
        opened.add(index)
        if block.get("type") == "text" and block.get("text"):
            rec["first_token_s"] = rec["first_token_s"] if rec["first_token_s"] is not None else now
            acc["text"].append(block["text"])
        elif block.get("type") == "tool_use":
            acc["calls"][index] = {"id": block.get("id", ""), "name": block.get("name", ""), "arguments": ""}
    elif kind == "content_block_delta":
        if index not in opened:
            rec["error"] = "内容增量缺少对应的内容块"
            return
        delta = ev.get("delta") or {}
        block = blocks[index]
        delta_type = delta.get("type")
        if delta_type == "text_delta":
            text = delta.get("text") or ""
            if text:
                rec["first_token_s"] = rec["first_token_s"] if rec["first_token_s"] is not None else now
                acc["text"].append(text)
                block["text"] = (block.get("text") or "") + text
        elif delta_type == "thinking_delta":
            rec["first_think_s"] = rec["first_think_s"] if rec["first_think_s"] is not None else now
            block["thinking"] = (block.get("thinking") or "") + (delta.get("thinking") or "")
        elif delta_type == "signature_delta":
            block["signature"] = (block.get("signature") or "") + (delta.get("signature") or "")
        elif delta_type == "input_json_delta" and index in acc["calls"]:
            acc["calls"][index]["arguments"] += delta.get("partial_json") or ""
    elif kind == "content_block_stop":
        if index not in opened:
            rec["error"] = "内容块结束事件无对应的开始事件"
            return
        opened.remove(index)
        if index in acc["calls"]:
            call, block = acc["calls"][index], blocks[index]
            try:
                args = json.loads(call["arguments"]) if call["arguments"] else block.get("input", {})
                if not isinstance(args, dict):
                    raise ValueError("tool input must be an object")
                block["input"] = args
                call["arguments"] = json.dumps(args, ensure_ascii=False)
            except (ValueError, TypeError):
                rec["error"] = "工具参数不是合法 JSON 对象"
    elif kind == "message_delta":
        acc["anthropic_reason"] = (ev.get("delta") or {}).get("stop_reason")
        rec["usage"] = _anthropic_usage(rec["usage"], ev.get("usage"))
    elif kind == "message_stop":
        if not acc.get("anthropic_started") or opened:
            rec["error"] = "消息结束时仍有未完成的内容块"
        reason = acc.get("anthropic_reason")
        rec["stop_reason"] = reason
        rec["response_status"] = "completed" if reason in ("end_turn", "stop_sequence", "tool_use") else "incomplete"
    elif kind == "error":
        rec["error"] = clip(json.dumps(ev.get("error") or ev, ensure_ascii=False))


def _consume(rec, ev, api, acc, now):
    if rec["first_event_s"] is None:
        rec["first_event_s"] = now

    def think():
        if rec["first_think_s"] is None:
            rec["first_think_s"] = now

    def token():
        if rec["first_token_s"] is None:
            rec["first_token_s"] = now

    if api == "anthropic":
        _consume_anthropic(rec, ev, acc, now)
        return
    if api == "gemini":
        gemini.consume(rec, ev, acc, now)
        return
    if api in ("chat", "azure"):
        rec["returned_model"] = ev.get("model") or rec.get("returned_model")
        rec["system_fingerprint"] = ev.get("system_fingerprint") or rec.get("system_fingerprint")
        if ev.get("error"):
            rec["error"] = clip(json.dumps(ev["error"], ensure_ascii=False))
        if ev.get("usage"):
            u = ev["usage"]
            rec["usage"] = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                            "cached_tokens": _chat_cached_tokens(u),
                            "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        choice = (ev.get("choices") or [{}])[0]
        delta = choice.get("delta") or {}
        if delta.get("reasoning_content") or delta.get("reasoning"):
            think()
            acc["reasoning"].append(delta.get("reasoning_content") or delta.get("reasoning"))
        if delta.get("content"):
            token()
            acc["text"].append(delta["content"])
        for tc in delta.get("tool_calls") or []:
            slot = acc["calls"].setdefault(tc.get("index", 0), {"id": "", "name": "", "arguments": ""})
            f = tc.get("function") or {}
            slot["id"] = tc.get("id") or slot["id"]
            slot["name"] = f.get("name") or slot["name"]
            slot["arguments"] += f.get("arguments") or ""
        if choice.get("finish_reason"):
            rec["response_status"] = "completed" if choice["finish_reason"] in ("stop", "tool_calls") else "incomplete"
        return
    kind = ev.get("type")
    if isinstance(ev.get("response"), dict):
        rec["returned_model"] = ev["response"].get("model") or rec.get("returned_model")
        rec["system_fingerprint"] = ev["response"].get("system_fingerprint") or rec.get("system_fingerprint")
    if kind == "response.output_text.delta":
        token()
        acc["text"].append(ev.get("delta", ""))
    elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        think()
    elif kind == "response.output_item.done" and ev.get("item"):
        acc["items"].append(ev["item"])
    elif kind in ("response.completed", "response.incomplete", "response.failed"):
        r = ev.get("response", {})
        rec["returned_model"] = r.get("model") or rec.get("returned_model")
        rec["system_fingerprint"] = r.get("system_fingerprint") or rec.get("system_fingerprint")
        rec["response_status"] = r.get("status")
        rec["usage"] = r.get("usage")
        if r.get("output"):
            acc["items"] = r["output"]
        if kind == "response.failed":
            rec["error"] = clip(json.dumps(r.get("error"), ensure_ascii=False))
    elif kind == "error":
        rec["error"] = clip(ev.get("message") or json.dumps(ev, ensure_ascii=False))


def _consume_json_body(rec, acc, api, payload):
    """中转站忽略 stream:true 时返回 application/json 一次性响应。正常解析出正文/工具调用/usage，
    标 non_stream=True（首字不是真首字，与 buffered 同口径）。没有这一步，这类中转站会被误判成“流在完成事件之前断开”。"""
    rec["returned_model"] = payload.get("model")
    rec["system_fingerprint"] = payload.get("system_fingerprint")
    if api == "gemini":
        rec["first_token_s"] = None
        gemini.consume(rec, payload, acc, rec["first_event_s"])
        rec["text"] = "".join(acc["text"])
        return
    if api == "bedrock":
        rec["first_token_s"] = None
        bedrock.consume_json(rec, payload, acc, rec["first_event_s"])
        return
    if api == "anthropic":
        acc["items"] = payload.get("content") or []
        rec["text"] = "".join(it.get("text", "") for it in acc["items"] if it.get("type") == "text")
        for i, item in enumerate(acc["items"]):
            if item.get("type") == "tool_use":
                acc["calls"][i] = {"id": item.get("id", ""), "name": item.get("name", ""),
                                   "arguments": json.dumps(item.get("input"), ensure_ascii=False)}
        rec["usage"] = _anthropic_usage(None, payload.get("usage"))
        rec["stop_reason"] = payload.get("stop_reason")
        rec["response_status"] = "completed" if rec["stop_reason"] in ("end_turn", "stop_sequence", "tool_use") else "incomplete"
        return
    if api in ("chat", "azure"):
        choice = (payload.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        rec["text"] = msg.get("content") or ""
        if msg.get("reasoning_content"):
            acc["reasoning"].append(msg["reasoning_content"])
        for tc in msg.get("tool_calls") or []:
            f = tc.get("function") or {}
            acc["calls"][tc.get("index", 0)] = {"id": tc.get("id") or "", "name": f.get("name") or "", "arguments": f.get("arguments") or ""}
        rec["response_status"] = "completed" if choice.get("finish_reason") in ("stop", "tool_calls") else "incomplete"
        u = payload.get("usage") or {}
        rec["usage"] = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                        "cached_tokens": _chat_cached_tokens(u),
                        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        return
    out = payload.get("output") or payload.get("output_text") or []
    rec["response_status"] = payload.get("status")
    rec["usage"] = payload.get("usage")
    acc["items"] = out
    texts = [(o.get("text") or "") for o in out if isinstance(o, dict) and o.get("type") == "message" for o in o.get("content") or []]
    if isinstance(payload.get("output_text"), str):
        texts.append(payload["output_text"])
    rec["text"] = "".join(texts)


def stream_request(pool, ep, body, timeout, redaction_keys=None):
    """发一次请求，不自动重发可能已送达的 POST；记录送达状态和端到端时间。"""
    api = ep["api"]
    rec = _new_rec(ep, body["model"], api)
    rec["request_config"] = {k: body[k] for k in ("model", "stream", "temperature", "max_tokens", "max_output_tokens", "reasoning", "reasoning_effort", "output_config", "stream_options", "generationConfig", "inferenceConfig") if k in body}
    acc = {"text": [], "reasoning": [], "items": [], "calls": {}}
    if api == "gemini":
        path = gemini.request_path(ep["base_url"], body["model"])
    elif api == "azure":
        path = azure.request_path(ep["base_url"], body["model"], ep.get("azure_api_version", azure.DEFAULT_API_VERSION))
    elif api == "bedrock":
        path = bedrock.request_path(ep["base_url"], body["model"])
    else:
        path = endpoint_parts(ep["base_url"])[3] + {"chat": "/chat/completions", "responses": "/responses", "anthropic": "/messages"}[api]
        path = custom_http.append_query(path, ep.get("query_params"))
    wire_body = {k: v for k, v in body.items() if k != "model"} if api in ("gemini", "azure", "bedrock") else body
    data = json.dumps(wire_body).encode()
    headers = {**_auth_headers(ep), "Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": "modelbench/" + VERSION}
    logical_start = t0 = time.perf_counter()
    rec["attempts"] = []
    send_started = False
    aws_secrets = []
    try:
        if api == "bedrock":
            credentials = ep.get("_aws_credentials") or bedrock.resolve_credentials(ep.get("aws_profile", ""))
            aws_secrets = bedrock.secrets_of(credentials)
            parsed = urllib.parse.urlsplit(ep["base_url"])
            url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, path, "", ""))
            signed = bedrock.sign_headers("POST", url, data, credentials, ep.get("aws_region", bedrock.DEFAULT_REGION))
            headers.update(signed)
            headers["Accept"] = "application/vnd.amazon.eventstream"
            aws_secrets.extend([signed["Authorization"], signed["Authorization"].split("Signature=")[-1]])
        conn, reused, cs = pool.get(ep, timeout)
        rec["first_reused"] = reused
        rec["reused"], rec["connect_s"] = reused, cs
        t0 = time.perf_counter()
        # request/getresponse 任一阶段出错都无法证明上游没收到请求，不能自动重发。
        send_started = True
        conn.request("POST", path, body=data, headers=headers)
        resp = conn.getresponse()
        rec["status"] = resp.status
        rec["headers_s"] = round(time.perf_counter() - t0, 3)
        if resp.status != 200:
            rec["error"] = clip(redact_text(resp.read().decode("utf-8", "replace"), ep.get("api_key")))
        elif api == "bedrock" and "application/vnd.amazon.eventstream" in (resp.getheader("Content-Type") or ""):
            for kind, value, message_type in bedrock.read_events(resp):
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f"超过 {timeout} 秒仍未结束")
                now = round(time.perf_counter() - t0, 3)
                rec["events"].append({"t": now, "event": {"type": kind, "message_type": message_type, "payload": value}})
                bedrock.consume(rec, kind, value, message_type, acc, now)
            if resp.length not in (None, 0):
                raise http.client.IncompleteRead(b"", resp.length)
            resp.close()
            rec["text"] = "".join(acc["text"])
            acc["items"] = [acc.get("bedrock_blocks", {})[i] for i in sorted(acc.get("bedrock_blocks", {}))]
            calls = [acc["calls"][k] for k in sorted(acc["calls"])]
            rec["tool_calls"] = calls
            rec["ok"] = rec["response_status"] == "completed" and bool(rec["text"] or calls) and not rec["error"]
            if not rec["response_status"] and not rec["error"]:
                rec["error"] = "Bedrock 流在消息结束事件之前断开"
            elif rec["response_status"] != "completed" and not rec["error"]:
                rec["error"] = "Bedrock 响应未完整完成"
        elif "text/event-stream" not in (resp.getheader("Content-Type") or ""):
            # 中转站忽略 stream:true，一次性返回 JSON：能正常读出回答，但“首字”不是模型真正的首字
            raw = resp.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
            except ValueError:
                raise ValueError(f"返回的不是流也不是 JSON（Content-Type {resp.getheader('Content-Type')}）：{clip(redact_text(raw, ep.get('api_key')), 120)}")
            now = round(time.perf_counter() - t0, 3)
            rec["first_event_s"] = rec["first_token_s"] = now
            rec["non_stream"] = True
            rec["events"].append({"t": now, "event": {"type": "non_stream_json", "payload": payload}})
            _consume_json_body(rec, acc, api, payload)
            calls = ([{"id": it.get("call_id"), "name": it.get("name"), "arguments": it.get("arguments")}
                      for it in acc["items"] if isinstance(it, dict) and it.get("type") == "function_call"]
                     if api == "responses" else [acc["calls"][k] for k in sorted(acc["calls"])])
            rec["tool_calls"] = calls
            rec["ok"] = rec["response_status"] == "completed" and bool(rec["text"] or calls) and not rec["error"]
            if not rec["ok"] and not rec["error"]:
                rec["error"] = "响应未完整完成" if rec["response_status"] != "completed" else "非流式响应里没有正文或工具调用"
        else:
            while True:
                raw = resp.readline()
                if not raw:
                    break
                if time.perf_counter() - t0 > timeout:
                    raise TimeoutError(f"超过 {timeout} 秒仍未结束")
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload in ("", "[DONE]"):
                    continue
                ev = json.loads(payload)
                now = round(time.perf_counter() - t0, 3)
                rec["events"].append({"t": now, "event": ev})
                _consume(rec, ev, api, acc, now)
            if resp.length not in (None, 0):
                raise http.client.IncompleteRead(b"", resp.length)
            # readline() 对 Content-Length 恰好读完时不会关闭 HTTPResponse；
            # 不关闭会让下一个 getresponse() 报 ResponseNotReady，并可能重复发送请求。
            resp.close()
            rec["text"] = "".join(acc["text"])
            if api == "anthropic":
                acc["items"] = [acc.get("anthropic_blocks", {})[i] for i in sorted(acc.get("anthropic_blocks", {}))]
            calls = ([{"id": it.get("call_id"), "name": it.get("name"), "arguments": it.get("arguments")}
                      for it in acc["items"] if isinstance(it, dict) and it.get("type") == "function_call"]
                     if api == "responses" else [acc["calls"][k] for k in sorted(acc["calls"])])
            rec["tool_calls"] = calls
            rec["ok"] = rec["response_status"] == "completed" and bool(rec["text"] or calls) and not rec["error"]
            if not rec["response_status"] and not rec["error"]:
                rec["error"] = "流在完成事件之前断开"
            elif rec["response_status"] != "completed" and not rec["error"]:
                rec["error"] = "响应未完整完成"
        if resp.will_close:
            pool.drop(ep)
    except Exception as e:  # noqa: BLE001
        rec["error"] = clip(redact_text(f"{type(e).__name__}: {e}", ep.get("api_key")))
        rec["delivery_uncertain"] = bool(send_started and rec["status"] is None)
        pool.drop(ep)
    rec["total_s"] = round(time.perf_counter() - t0, 3)
    rec["wall_total_s"] = round(time.perf_counter() - logical_start, 3)
    rec["attempts"].append({"reused": rec["reused"], "elapsed_s": rec["total_s"], "status": rec["status"],
                            "error": rec["error"], "delivery_uncertain": rec.get("delivery_uncertain", False)})
    # 客户端观测值，不能归因于服务端排队。
    if rec["headers_s"] is not None and rec["first_event_s"] is not None:
        rec["queue_s"] = round(max(rec["first_event_s"] - rec["headers_s"], 0), 3)
    usage = rec["usage"] or {}
    if usage and "cached_tokens" not in usage:
        usage["cached_tokens"] = (usage.get("input_tokens_details") or {}).get("cached_tokens")
    out = usage.get("output_tokens")
    rec["reasoning_tokens"] = usage.get("reasoning_tokens") or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    start = rec["first_token_s"] if rec["first_token_s"] is not None else rec["first_event_s"]
    visible = (out or 0) - (rec["reasoning_tokens"] or 0)  # 输出速度只算“可见回答”的 tokens：推理 tokens 是首个可见字之前产生的，时间不在分母里，算进去会虚高
    if visible > 0 and not rec.get("reasoning_tokens_unknown") and start is not None and rec["total_s"] - start > 0.05:
        rec["tps"] = round(visible / (rec["total_s"] - start), 1)
    if out and rec["total_s"] > 0.05:
        rec["tps_overall"] = round(out / rec["total_s"], 1)  # 含等待的整体速度；中转站攒满返回时 tps 算不出，用它
    # 非流式：响应几乎在最后一刻才一次性到达（首个事件≈总耗时），此时“首字”不是模型真正的首字
    rec["buffered"] = bool(rec["ok"] and rec["first_event_s"] is not None and rec["total_s"] >= BUFFER_MIN_TOTAL
                           and rec["total_s"] - rec["first_event_s"] <= 0.03 * rec["total_s"] + 0.1) or rec["non_stream"]
    rec["_replay"] = {"items": acc["items"], "reasoning": "".join(acc["reasoning"]), "text": rec["text"]}
    for secret in dict.fromkeys([ep.get("api_key")] + list(redaction_keys or []) + aws_secrets):
        _redact_record(rec, secret)
    return rec


# ---------------------------------------------------------------- 任务与判分

def context_text(tokens, needle=False):
    """约 tokens 个 token 的背景文字；needle=True 时在正中间藏一句暗号。"""
    if not tokens:
        return None
    n = max(1, round(tokens / 44))  # 实测每个填充单元约 44 tokens（66 个中文字符 ÷ 1.5）
    parts = [f"第{i + 1}条：{_FILLER_UNIT}" for i in range(n)]
    if needle:
        parts.insert(n // 2, NEEDLE)
    return "".join(parts)


def build_tasks(scenarios, prompt=DEFAULT_PROMPT, context=0):
    tasks = []
    for sc in scenarios:
        if sc == "speed":
            tasks.append({"id": "speed", "scenario": "speed", "kind": "plain", "name": "自定义提示词", "prompt": prompt, "system": context_text(context)})
        elif sc == "gen":
            tasks.append({"id": "gen", "scenario": "gen", "kind": "plain", "name": "生成约 300 字", "prompt": GEN_PROMPT})
        elif sc in ("quiz", "quiz_extended"):
            tasks += [{"id": q["id"], "scenario": sc, "kind": "plain", "name": q["name"], "prompt": q["prompt"], "chk": q["chk"],
                       **({"strict": True} if q.get("strict") else {})} for q in (QUIZ if sc == "quiz" else QUIZ_EXTENDED)]
        elif sc == "tool":
            tasks.append({"id": "tool", "scenario": "tool", "kind": "tool", "name": "两步工具调用", "prompt": TOOL_PROMPT})
        elif sc in ("needle10k", "needle30k"):
            tasks.append({"id": sc, "scenario": sc, "kind": "plain", "name": SCENARIOS[sc], "system": context_text(10000 if sc == "needle10k" else 30000, True),
                          "prompt": "上文背景材料里写了“本次测试的暗号”，它是什么？只输出暗号本身。", "chk": ("contains", "青竹-7392")})
        elif sc == "needle_matrix":
            for length in (10000, 30000):
                for depth in (10, 50, 90):
                    for present in (True, False):
                        tid = f"nm{length}-{depth}-{'yes' if present else 'no'}"
                        tasks.append({"id": tid, "scenario": sc, "kind": "needle_matrix", "name": f"约{length} tokens · 深度{depth}% · {'有答案' if present else '无答案'}",
                                      "prompt": "根据文档，目标编号是多少？未出现目标编号则只输出未知。只输出编号或未知。",
                                      "length": length, "depth": depth, "present": present})
        elif sc == "fingerprint":
            tasks += [{"id": f"fp{i + 1}", "scenario": "fingerprint", "kind": "plain", "name": f"数值指纹 {i + 1}", "prompt": p} for i, p in enumerate(FP_PROMPTS)]
        elif sc == "modeltrace":
            tasks += [dict(item, scenario=sc, kind="modeltrace", name=f"长数字挑战 {i + 1}")
                      for i, item in enumerate(modeltrace.challenges(0))]
        elif sc == "modeltrace_enroll":
            for item in modeltrace_suite.fingerprint_suite():
                prompt = ((item["user_prefix"] + "\n\nFinal task:\n") if item["user_prefix"] else "") + item["prompt"]
                tasks.append({"id": item["challenge_id"], "scenario": sc, "kind": sc,
                              "name": item["condition"] + " · " + item["challenge_id"],
                              "condition_id": item["condition"], "transport": item["transport"],
                              "expected_count": item["expected_count"], "prompt": prompt, "system": item["system"]})
        elif sc == "behavior":
            tasks += behavior.tasks()
    return tasks


def materialize_task(task, seed, round_no):
    """按任务和轮次派生实例；相同轮次的各模型收到完全相同的实例。"""
    digest = hashlib.sha256(f"{seed}:{round_no}:{task['id']}".encode()).digest()
    rng = random.Random(int.from_bytes(digest[:8], "big"))
    result = dict(task)
    if task["kind"] == "modeltrace":
        index = int(task["id"][2:]) - 1
        result.update(modeltrace.challenges(seed, round_no)[index])
    elif task["kind"] == "tool":
        result["tool_result"] = dict(TOOL_RESULT, price=rng.randrange(1000, 99999) / 100)
    elif task["kind"] == "needle_matrix":
        n = max(1, round(task["length"] / 44))
        parts = [f"第{i + 1}条：{_FILLER_UNIT}" for i in range(n)]
        target = f"MB-{rng.randrange(100000, 999999)}"
        if task["present"]:
            parts.insert(round(n * task["depth"] / 100), f"【目标记录】目标编号：{target}。")
        distractor = f"MB-{rng.randrange(100000, 999999)}"
        while distractor == target:
            distractor = f"MB-{rng.randrange(100000, 999999)}"
        parts.insert(round(n * (100 - task["depth"]) / 100), f"【干扰记录】其他编号：{distractor}。")
        # 文档作为用户内容，而非 system/developer 指令。
        result["prompt"] = "文档：\n" + "".join(parts) + "\n问题：" + task["prompt"]
        result["chk"] = ("re", re.escape(target if task["present"] else "未知"))
        result["strict"] = True
        result["case"] = {"target": target if task["present"] else None, "distractor": distractor, "length_estimate": task["length"], "depth_percent": task["depth"], "present": task["present"]}
    return result


def _norm(t):
    return re.sub(r"\s+", "", t or "")


# 判分宽松化：模型答对但带标点/“答案是…”这类前缀，不该算错（否则正确率系统性偏低）。
# 格式是否合规由“指令遵循”这类题目专门判，不连坐所有题。strict 记录“剥掉这些之后才判对”的情况。
_ANSWER_PREFIX = re.compile(r"^(答案(?:是|为)|结果(?:是|为)|它是|应该是|可以说|那就是)[:：\s]*")
_ANSWER_TAIL = re.compile(r"[。.;；,，!！?？\s]+$")


def normalize_answer(text):
    """剥掉首尾标点/空白和常见前缀后缀，返回用于判对的文本；second 为 True 表示做了宽松化处理。"""
    t = (text or "").strip()
    t2 = _ANSWER_TAIL.sub("", t)
    t2 = _ANSWER_PREFIX.sub("", t2).strip()
    if t2 == t2[:0] or not t2:  # 前缀吃掉整段就别剥（比如模型只回“结果”）
        t2 = t
    t2 = t2.strip("“”\"'`（）()【】[]{}<>《》")
    return t2, t2 != t


def check_answer(chk, text, strict=False):
    """判分。strict=True 时不做宽松化（用于“指令遵循/严格输出”类题目）。"""
    kind, arg = chk
    t = (text or "") if strict else (text or "").strip()
    if not strict:
        t = normalize_answer(t)[0]
    if kind == "re":
        return re.fullmatch(arg, t) is not None
    if kind == "json":
        try:
            def unique_object(pairs):
                obj = {}
                for key, value in pairs:
                    if key in obj:
                        raise ValueError("duplicate key")
                    obj[key] = value
                return obj
            def equal(actual, expected):
                if isinstance(expected, dict):
                    return isinstance(actual, dict) and actual.keys() == expected.keys() and all(equal(actual[k], v) for k, v in expected.items())
                if isinstance(expected, list):
                    return isinstance(actual, list) and len(actual) == len(expected) and all(equal(a, b) for a, b in zip(actual, expected))
                if isinstance(expected, (int, float)) and not isinstance(expected, bool):
                    return isinstance(actual, (int, float)) and not isinstance(actual, bool) and actual == expected
                return type(actual) is type(expected) and actual == expected
            return equal(json.loads(t, object_pairs_hook=unique_object), arg)
        except (ValueError, TypeError):
            return False
    if kind == "set":
        vals = t.split(",")
        return len(vals) == len(arg) and len(set(vals)) == len(arg) and set(vals) == set(arg)
    return arg in t


def _grade(rec, task):
    rec["scenario"], rec["task"], rec["task_name"] = task["scenario"], task["id"], task["name"]
    if task.get("case"):
        rec["case"] = task["case"]
    if "chk" in task:
        strict = bool(task.get("strict"))
        rec["strict"] = strict
        loose, did = normalize_answer(rec["text"]) if not strict else (rec["text"], False)
        rec["answer_loose"] = did
        rec["pass"] = bool(rec["ok"] and check_answer(task["chk"], rec["text"], strict))
        rec["grade_note"] = "" if rec["pass"] else ("请求失败" if not rec["ok"] else "答案不符：" + clip(rec["text"], 60))
        if not rec["pass"] and did and check_answer(task["chk"], loose, strict) and not strict:
            # 宽松化之后仍对不上，但和“严格口径”一致：说明差在格式不在答案
            rec["grade_note"] += "（答案可能对，格式不符）"


def _tool_args_ok(call):
    try:
        args = json.loads(call.get("arguments") or "")
    except ValueError:
        return False
    return call.get("name") == TOOL_DEF["name"] and isinstance(args, dict) and args.get("item") == TOOL_RESULT["item"]


def run_tool(pool, ep, model, task, timeout, effort="", redaction_keys=None):
    """验证工具调用以及对本次工具返回结果的使用。"""
    logical_start = time.perf_counter()
    api = ep["api"]
    if api in ("chat", "azure"):
        tool = {"type": "function", "function": TOOL_DEF}
    elif api == "gemini":
        tool = {"functionDeclarations": [{"name": TOOL_DEF["name"], "description": TOOL_DEF["description"], "parametersJsonSchema": TOOL_DEF["parameters"]}]}
    elif api == "bedrock":
        tool = {"toolSpec": {"name": TOOL_DEF["name"], "description": TOOL_DEF["description"], "inputSchema": {"json": TOOL_DEF["parameters"]}}}
    elif api == "anthropic":
        tool = {"name": TOOL_DEF["name"], "description": TOOL_DEF["description"], "input_schema": TOOL_DEF["parameters"]}
    else:
        tool = dict(TOOL_DEF, type="function")
    user = {"role": "user", "content": task["prompt"]}
    body1 = make_body(api, model, task["prompt"], effort=effort)
    if api == "responses":
        body1["input"] = [user]
    if api == "bedrock":
        body1["toolConfig"] = {"tools": [tool]}
    else:
        body1["tools"] = [tool]
    r1 = stream_request(pool, ep, body1, timeout, redaction_keys=redaction_keys)
    _grade(r1, task)
    r1["tool_result"] = task.get("tool_result", TOOL_RESULT)
    r1["steps"] = [{"step": 1, "ok": r1["ok"], "status": r1["status"], "total_s": r1["total_s"], "error": r1["error"], "request_config": r1["request_config"]}]
    call = next((c for c in r1["tool_calls"] if c.get("name")), None)
    if not r1["ok"] or not call or not _tool_args_ok(call) or (api == "bedrock" and not call.get("id")):
        r1["wall_total_s"] = round(time.perf_counter() - logical_start, 3)
        r1["pass"] = False
        r1["grade_note"] = ("第一步请求失败：" + (r1["error"] or "") if not r1["ok"] else
                            "第一步没有调用工具" if not call else "第一步工具参数不合法：" + clip(str(call.get("arguments")), 80))
        return r1
    expected = task.get("tool_result", TOOL_RESULT)
    result = json.dumps(expected, ensure_ascii=False)
    rp = r1["_replay"]
    if api in ("chat", "azure"):
        asst = {"role": "assistant", "content": rp["text"] or None,
                "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}
        if rp["reasoning"]:
            asst["reasoning_content"] = rp["reasoning"]
        body2 = make_body(api, model, task["prompt"], effort=effort)
        body2.update(messages=[user, asst, {"role": "tool", "tool_call_id": call["id"], "content": result}], tools=[tool])
    elif api == "anthropic":
        body2 = make_body(api, model, task["prompt"], effort=effort)
        body2.update(tools=[tool], messages=[user, {"role": "assistant", "content": rp["items"]},
                                             {"role": "user", "content": [{"type": "tool_result", "tool_use_id": call["id"], "content": result}]}])
    elif api == "bedrock":
        body2 = make_body(api, model, task["prompt"], effort=effort)
        body2.update(toolConfig={"tools": [tool]}, messages=[{"role": "user", "content": [{"text": task["prompt"]}]},
                     {"role": "assistant", "content": rp["items"]},
                     {"role": "user", "content": [{"toolResult": {"toolUseId": call["id"], "content": [{"json": expected}]}}]}])
    elif api == "gemini":
        response = {"name": call["name"], "response": expected}
        if call.get("id"):
            response["id"] = call["id"]
        body2 = make_body(api, model, task["prompt"], effort=effort)
        body2.update(tools=[tool], contents=[{"role": "user", "parts": [{"text": task["prompt"]}]},
                                            {"role": "model", "parts": rp["items"]},
                                            {"role": "user", "parts": [{"functionResponse": response}]}])
    else:
        body2 = make_body(api, model, task["prompt"], effort=effort)
        body2.update(tools=[tool], input=[user] + rp["items"] + [{"type": "function_call_output", "call_id": call["id"], "output": result}])
    r2 = stream_request(pool, ep, body2, timeout, redaction_keys=redaction_keys)
    rec = dict(r2)
    rec["time"] = r1["time"]
    rec["total_s"] = round(r1["total_s"] + r2["total_s"], 3)
    rec["wall_total_s"] = round(time.perf_counter() - logical_start, 3)
    rec["gen_s"] = r2["total_s"]  # 第二步（拿到工具结果后生成答案）单独一份：和单次请求的总耗时可比；两步合计用于看端到端代价
    rec["headers_s"], rec["queue_s"] = r1["headers_s"], r1["queue_s"]
    rec["first_event_s"] = r1["first_event_s"]
    rec["first_token_s"] = r1["first_token_s"] if r1["first_token_s"] is not None else (round(r1["total_s"] + r2["first_token_s"], 3) if r2["first_token_s"] is not None else None)
    rec["first_think_s"] = r1["first_think_s"] if r1["first_think_s"] is not None else (round(r1["total_s"] + r2["first_think_s"], 3) if r2["first_think_s"] is not None else None)
    rec["buffered"] = r1["buffered"] or r2["buffered"]
    rec["non_stream"] = r1["non_stream"] or r2["non_stream"]
    rec["events"] = [dict(e, step=1) for e in r1["events"]] + [dict(e, step=2) for e in r2["events"]]
    rec["steps"] = [dict(r1["steps"][0], attempts=r1["attempts"], wall_total_s=r1["wall_total_s"]), {"step": 2, "ok": r2["ok"], "status": r2["status"], "total_s": r2["total_s"], "wall_total_s": r2["wall_total_s"], "attempts": r2["attempts"], "error": r2["error"], "request_config": r2["request_config"]}]
    rec["ok"] = r1["ok"] and r2["ok"]
    rec["error"] = ("第二步：" + r2["error"]) if r2["error"] else ""
    _grade(rec, task)
    rec["tool_result"] = expected
    rec["pass"] = bool(rec["ok"] and check_answer(("json", {"price": expected["price"], "unit": expected["unit"]}), rec["text"], True))
    rec["grade_note"] = "" if rec["pass"] else ("第二步请求失败：" + (r2["error"] or f"HTTP {r2['status']}") if not r2["ok"] else "未找到要求的价格和单位：" + clip(rec["text"], 60))
    return rec


def run_task(pool, ep, model, task, timeout, effort="", redaction_keys=None):
    if task["kind"] == "tool":
        return run_tool(pool, ep, model, task, timeout, effort, redaction_keys)
    body = make_body(ep["api"], model, task["prompt"], task.get("system"), effort)
    if task["kind"] in ("modeltrace", "modeltrace_enroll", "behavior"):
        # 分布测试使用服务商默认 temperature，不强行设置为 0。
        body.pop("temperature", None)
        if ep["api"] == "gemini":
            body["generationConfig"].pop("temperature", None)
        elif ep["api"] == "bedrock":
            body["inferenceConfig"].pop("temperature", None)
    rec = stream_request(pool, ep, body, timeout, redaction_keys=redaction_keys)
    _grade(rec, task)
    if task["kind"] in ("modeltrace", "modeltrace_enroll"):
        rec["expected_count"] = task["expected_count"]
        rec["probe_prompt"] = task["prompt"]
        if task["kind"] == "modeltrace_enroll":
            rec["reference"] = {"condition_id": task["condition_id"], "challenge_id": task["id"], "transport": task["transport"],
                                "system_sha256": hashlib.sha256((task.get("system") or "").encode()).hexdigest()}
    return rec


def public_rec(rec, text_limit=4000):
    r = {k: v for k, v in rec.items() if k != "events" and not k.startswith("_")}
    r["event_count"] = len(rec.get("events", []))
    r["text"] = (rec.get("text") or "")[:text_limit]
    return r


# ---------------------------------------------------------------- 模型指纹分析

def extract_numbers(text):
    """从回答里提取整数序列：取最长的连续数字段（用字母/汉字分隔或超出范围会断开）。用来防模型回显提示词里的 1、355 等数字干扰。"""
    runs, current, prev_end = [], [], 0
    for m in re.finditer(r"\d+", text or ""):
        if current and any(c.isalpha() for c in text[prev_end:m.start()]):
            runs.append(current)
            current = []
        v = int(m.group())
        if FP_RANGE[0] <= v <= FP_RANGE[1]:
            current.append(v)
        prev_end = m.end()
    if current:
        runs.append(current)
    return max(runs, key=len) if runs else []


def _fp_counts(nums):
    c = [0] * FP_BINS
    for v in nums:
        c[min(FP_BINS - 1, (v - FP_RANGE[0]) * FP_BINS // (FP_RANGE[1] - FP_RANGE[0] + 1))] += 1
    return c


def hellinger(a, b):
    """两个分箱计数的 Hellinger 距离（0 = 同分布，1 = 完全不重叠）；加 0.5 平滑防零格。"""
    k = FP_BINS
    pa = [(x + 0.5) / (sum(a) + 0.5 * k) for x in a]
    pb = [(x + 0.5) / (sum(b) + 0.5 * k) for x in b]
    return round(math.sqrt(sum((math.sqrt(x) - math.sqrt(y)) ** 2 for x, y in zip(pa, pb)) / 2), 4)


def fp_verdict(dist, noise):
    """(结论, 判定线)。noise 是折算到整样本的自身抽样噪声；判定线 = max(noise × FP_NOISE_FACTOR, FP_ABS_FLOOR)，
    样本再多也不低于 FP_ABS_FLOOR；介于 1× 和 2× 判定线之间的差异不作判断（可拒判，不硬给结论）。"""
    thr = round(max(FP_NOISE_FACTOR * (noise or 0), FP_ABS_FLOOR), 4)
    if dist <= thr:
        return "数值分布接近（不证明身份）", thr
    if dist >= 2 * thr:
        return "数值分布差异较大（不证明身份）", thr
    return "差异不明显（不作判断）", thr


def fp_analysis(records):
    """按模型汇总数值指纹：分箱分布、有效数字个数、自身抽样噪声（奇偶半分距离），以及两两 Hellinger 距离和结论。
    比较用的噪声底线把半分距离除以 √2 折算到整样本量（两半各 n/2，对比两侧各 n，噪声 ∝ √(1/n)）。
    没有指纹记录返回 None。结论只是启发式证据，不是身份认证。"""
    nums_by_label = {}
    for r in records:
        if r.get("scenario") == "fingerprint" and r.get("ok"):
            nums_by_label.setdefault(r.get("label") or r["model"], []).extend(extract_numbers(r.get("text") or ""))
    if not nums_by_label:
        return None
    labels = {}
    for lb, nums in nums_by_label.items():
        half = len(nums) // 2
        self_d = hellinger(_fp_counts(nums[:half]), _fp_counts(nums[half:])) if half >= FP_MIN_NUMBERS // 2 else None
        labels[lb] = {"n_numbers": len(nums), "counts": _fp_counts(nums), "self_dist": self_d}
    pairs, keys = [], list(labels)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = keys[i], keys[j]
            na, nb = labels[a]["n_numbers"], labels[b]["n_numbers"]
            if na < FP_MIN_NUMBERS or nb < FP_MIN_NUMBERS:
                pairs.append({"a": a, "b": b, "n_a": na, "n_b": nb, "dist": None, "thr": None, "noise": None, "verdict": "样本不足（不作判断）"})
                continue
            dist = hellinger(labels[a]["counts"], labels[b]["counts"])
            noise = round(max(labels[a]["self_dist"] or 0, labels[b]["self_dist"] or 0) / math.sqrt(2), 4)
            verdict, thr = fp_verdict(dist, noise)
            pairs.append({"a": a, "b": b, "n_a": na, "n_b": nb, "dist": dist, "thr": thr, "noise": round(noise, 4), "verdict": verdict})
    return {"labels": labels, "pairs": pairs, "min_numbers": FP_MIN_NUMBERS}


# ---------------------------------------------------------------- 混合多次测试（提高样本量）

def cond_of(env, scenario):
    """返回可验证的场景条件键；历史数据缺字段时返回 None，不假定未知条件相同。"""
    env = env or {}
    fp = (env.get("scenario_fingerprints") or {}).get(scenario)
    required = ("method_version", "sampling", "timeout_s", "gap_s", "warmup", "effort")
    sampling = env.get("sampling")
    if not fp or any(k not in env or env[k] is None for k in required) or not isinstance(sampling, dict) or any(k not in sampling for k in ("temperature", "max_tokens")):
        return None
    data = {k: env[k] for k in required}
    if env.get("method_version") in ("4", "5", "6", "7"):
        if "concurrency" not in env or "connection_mode" not in env:
            return None
        data["concurrency"] = env["concurrency"]
        data["connection_mode"] = env["connection_mode"]
        if scenario in ("tool", "needle_matrix", "modeltrace"):
            if env.get("seed") is None:
                return None
            data["seed"] = env["seed"]
    if scenario == "behavior":
        if not env.get("behavior"):
            return None
        data["behavior"] = env["behavior"]
    if scenario == "modeltrace":
        identity = env.get("modeltrace") or {}
        if not identity.get("bank_sha256"):
            return None
        data["modeltrace"] = identity
    data.update(scenario=scenario, fingerprint=fp)
    return hashlib.sha256(json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def service_of(env, label):
    """合并同名目标时必须核对原始接口地址与协议；不保存或哈希密钥。"""
    endpoint = (label or "").split("/", 1)[0]
    matches = [e for e in (env.get("endpoints") or []) if e.get("name") == endpoint and e.get("base_url") and e.get("api")]
    if len(matches) != 1:
        return None
    e = matches[0]
    identity = [e["base_url"], e["api"]]
    if e["api"] == "azure":
        if not e.get("azure_api_version") or not e.get("azure_auth"):
            return None
        identity.extend((e["azure_api_version"], e["azure_auth"]))
    elif e["api"] == "bedrock":
        if not e.get("aws_region") or "aws_profile" not in e or not e.get("aws_credential_source"):
            return None
        identity.extend((e["aws_region"], e["aws_profile"], e["aws_credential_source"]))
    elif e["api"] in ("chat", "responses") and e.get("auth_mode", "bearer") != "bearer":
        identity.append(e["auth_mode"])
        if e["auth_mode"] == "custom":
            if not e.get("auth_header") or "auth_prefix" not in e:
                return None
            identity.extend((e["auth_header"], e["auth_prefix"]))
    if e["api"] in ("chat", "responses", "anthropic") and e.get("query_params"):
        if not isinstance(e["query_params"], dict):
            return None
        identity.append(sorted(e["query_params"].items()))
    return hashlib.sha256(json.dumps(identity, ensure_ascii=False).encode()).hexdigest()


def cond_label(env, scenario):
    """人能看懂的同一组条件说明，用于页面展示。"""
    env = env or {}
    eff = env.get("effort") or "未指定"
    sm = env.get("sampling")
    parts = [f"强度 {eff}"]
    if scenario == "speed":
        parts.insert(0, f"提示词「{(env.get('prompt') or '')[:12]}」")
    if scenario in ("modeltrace", "modeltrace_enroll", "behavior"):
        parts.append("采样 temperature=服务商默认")
    else:
        parts.append(f"采样 temperature={sm['temperature']}" if sm else "采样未记录（旧结果）")
    if env.get("context_tokens"):
        parts.append(f"背景 {env['context_tokens']}")
    return " · ".join(parts)


def merge_analysis(runs):
    """多次测试合并：把条件与接口身份已知且相同的请求合在一起做描述性统计。
    条件不同的组不合并、单独成组并注明——绝不把不同条件的结果混在一起算。"""
    groups, sources, group_env = {}, {}, {}
    for run in runs or []:
        env = run.get("env") or {}
        rid = run.get("id")
        for rec in run.get("records") or []:
            sc = rec.get("scenario", "speed")
            label = rec.get("label") or rec.get("model")
            ck = cond_of(env, sc)
            service = service_of(env, label)
            # 未知条件不能和任何其他运行合并或作跨运行推断。
            ck = ck or "unknown:" + str(rid) + ":" + sc
            service = service or "unknown:" + str(rid)
            key = (label, sc, ck, service)
            groups.setdefault(key, []).append(dict(rec, cond=ck, service=service))
            group_env.setdefault(key, env)  # 记住这一组自己的环境（不同条件组的 env 不同，标签才不会都一样）
            sources.setdefault(key, {})[rid] = sources.setdefault(key, {}).get(rid, 0) + 1
    if not groups:
        return None
    merged = []
    for key, recs in groups.items():
        label, sc, cond, service = key
        env = group_env.get(key) or {}
        merged.append({"label": label, "scenario": sc, "cond": cond, "service": service, "cond_label": cond_label(env, sc),
                       "runs": sorted(k for k in sources[key] if k), "n_per_run": [sources[key][k] for k in sorted(k for k in sources[key] if k)],
                       "records": recs})
    # 合并后的记录重新汇总（复用同一套统计，保证与单次测试口径一致）
    all_recs = [r for g in merged for r in g["records"]]
    rows = summarize(all_recs)
    # 把每个汇总行指回它的来源测试，便于展示“这行是几次测试合并来的”
    by_key = {(g["label"], g["scenario"], g["cond"], g["service"]): g for g in merged}
    for row in rows:
        g = by_key.get((row["label"], row["scenario"], row.get("cond", ""), row.get("service", "")))
        if g:
            row["merged_runs"], row["merged_n_per_run"], row["merged_cond"] = g["runs"], g["n_per_run"], g["cond_label"]
    # 条件不一致的提示：同一（模型, 场景）出现多个条件时，说明它们没被合并
    split = {}
    for g in merged:
        split.setdefault((g["label"], g["scenario"]), set()).add(g["cond"])
    warnings = [f"{lb} 在「{sc}」里有 {len(cs)} 种不同条件，已分开统计、未合并"
                 for (lb, sc), cs in split.items() if len(cs) > 1]
    services = {}
    for g in merged:
        services.setdefault((g["label"], g["scenario"]), set()).add(g["service"])
    warnings += [f"{lb} 在「{sc}」的接口地址或协议不同或未知，已分开统计" for (lb, sc), ss in services.items() if len(ss) > 1]
    return {"rows": rows, "groups": [{k: v for k, v in g.items() if k != "records"} for g in merged],
            "conclusion": conclusion(rows), "warnings": warnings, "run_ids": [r.get("id") for r in runs or [] if r.get("id")]}


# ---------------------------------------------------------------- 统计

def percentile(vals, p):
    s = sorted(vals)
    if not s:
        return None
    k = (len(s) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def wilson(k, n, z=1.96):
    """成功率的 Wilson 95% 置信区间。"""
    if n == 0:
        return None
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [round(max(0.0, centre - half), 3), round(min(1.0, centre + half), 3)]


def dist(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "median": round(statistics.median(vals), 3), "p95": round(percentile(vals, 0.95), 3),
            "p95_ok": len(vals) >= P95_MIN_N, "min": round(min(vals), 3), "max": round(max(vals), 3)}


def mwu_p(a, b):
    """Mann-Whitney U 双侧检验（正态近似，含并列校正和连续性校正）：两组耗时的差异是否超出随机波动。任一组少于 2 个返回 None。"""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return None
    allv = sorted([(v, 0) for v in a] + [(v, 1) for v in b])
    ranks, i, tie = [0.0] * len(allv), 0, 0
    while i < len(allv):
        j = i
        while j + 1 < len(allv) and allv[j + 1][0] == allv[i][0]:
            j += 1
        for k in range(i, j + 1):
            ranks[k] = (i + j) / 2 + 1
        tie += (j - i + 1) ** 3 - (j - i + 1)
        i = j + 1
    u = sum(r for r, (_, g) in zip(ranks, allv) if g == 0) - n1 * (n1 + 1) / 2
    n = n1 + n2
    sigma2 = n1 * n2 / 12 * ((n + 1) - tie / (n * (n - 1)))
    if sigma2 <= 0:
        return 1.0
    z = max((abs(u - n1 * n2 / 2) - 0.5) / math.sqrt(sigma2), 0.0)
    return round(math.erfc(z / math.sqrt(2)), 4)


def _norm(t):
    return re.sub(r"\s+", "", t or "")


def summarize(records):
    """按（模型, 场景）汇总。合并多次测试时记录会带 cond 字段，此时按（模型, 场景, 条件）分组，
    保证条件不同的结果不会混在一起统计（条件相同才谈得上合并）。"""
    rows = []
    keys = list(dict.fromkeys((r.get("label") or r["model"], r.get("scenario", "speed"), r.get("cond", ""), r.get("service", "")) for r in records))
    for label, sc, cond, service in keys:
        rs = [r for r in records if (r.get("label") or r["model"]) == label and r.get("scenario", "speed") == sc
              and r.get("cond", "") == cond and r.get("service", "") == service]
        good = [r for r in rs if r["ok"]]
        streaming = [r for r in good if not r.get("buffered") and not r.get("non_stream")]
        codes = {}
        for r in rs:
            if not r["ok"]:
                k = str(r["status"] or "无响应")
                codes[k] = codes.get(k, 0) + 1
        graded = [r for r in rs if r.get("pass") is not None]
        apis = sorted({r.get("api") for r in rs if r.get("api")})
        row = {"label": label, "scenario": sc, "cond": cond, "service": service, "api": apis[0] if len(apis) == 1 else "", "n": len(rs), "ok": len(good), "ok_ci": wilson(len(good), len(rs)), "codes": codes,
               "total": dist([r["total_s"] for r in good]), "first_token": dist([r["first_token_s"] for r in streaming]),
               "first_event": dist([r["first_event_s"] for r in good]), "first_think": dist([r.get("first_think_s") for r in good]),
               "tps": dist([r.get("tps") for r in streaming]), "connect": dist([r["connect_s"] for r in rs if r.get("first_reused") is False]),
               "tps_overall": dist([r.get("tps_overall") for r in good]), "total_vals": [round(r["total_s"], 3) for r in good if r["total_s"] is not None],
               "first_token_vals": [round(r["first_token_s"], 3) for r in streaming if r["first_token_s"] is not None],
               "tps_vals": [r["tps"] for r in streaming if r.get("tps") is not None],
               "gen": dist([r.get("gen_s") for r in good]), "gen_vals": [round(r["gen_s"], 3) for r in good if r.get("gen_s") is not None],
               "queue": dist([r.get("queue_s") for r in good]),
               "queue_vals": [round(r["queue_s"], 3) for r in good if r.get("queue_s") is not None],
               "think_share": None, "reasoning_tokens": dist([r.get("reasoning_tokens") for r in good]), "reasoning_share": None,
               "buffered_n": sum(1 for r in good if r.get("buffered")),
               "low_sample": len(rs) < LOW_SAMPLE,  # 请求数不足
               "low_ok_sample": 0 < len(good) < LOW_SAMPLE <= len(rs),  # 请求够多，但成功的太少：耗时只统计成功请求，实际用来算耗时的样本不足
               "pass_n": None, "pass_rate": None, "pass_ci": None,
               "consistency": None, "by_task": {}, "tin": None, "tout": None, "cached": None, "queue_median": None, "cache_trend": None, "sec_per_1k": None, "sec_per_1k_vals": []}
        wall_known = [r for r in rs if isinstance(r.get("wall_total_s"), (int, float))]
        row["wall_total"] = dist([r["wall_total_s"] for r in good if r.get("wall_total_s") is not None])
        row["slo_seconds"] = SLO_SECONDS
        row["slo_n"] = sum(bool(r["ok"] and r.get("pass") is not False and r["wall_total_s"] <= SLO_SECONDS) for r in wall_known)
        row["slo_observed"] = len(wall_known)
        row["slo_rate"] = round(row["slo_n"] / len(rs), 3) if len(wall_known) == len(rs) else None
        row["chars_per_second"] = dist([len(r.get("text") or "") / r["wall_total_s"] for r in good if r.get("text") and (r.get("wall_total_s") or 0) > 0])
        row["retry_n"] = sum(bool(r.get("conn_retry") or any(len(s.get("attempts", [])) > 1 for s in r.get("steps", []))) for r in rs)
        row["uncertain_delivery_n"] = sum(bool(r.get("delivery_uncertain")) for r in rs)
        row["returned_models"] = sorted({r["returned_model"] for r in rs if isinstance(r.get("returned_model"), str)})
        row["system_fingerprints"] = sorted({r["system_fingerprint"] for r in rs if isinstance(r.get("system_fingerprint"), str)})
        shares = [max(r["first_token_s"] - r["first_event_s"], 0) / r["total_s"] for r in good
                  if r["first_token_s"] is not None and r["first_event_s"] is not None and r["total_s"]]
        row["think_share"] = round(statistics.median(shares), 3) if shares else None
        rshare = [r["reasoning_tokens"] / r["usage"]["output_tokens"] for r in good
                  if r.get("reasoning_tokens") is not None and (r.get("usage") or {}).get("output_tokens")]
        row["reasoning_share"] = round(statistics.median(rshare), 3) if rshare else None
        for field, output in (("input_tokens", "tin"), ("output_tokens", "tout")):
            values = [(r.get("usage") or {}).get(field) for r in good]
            values = [v for v in values if isinstance(v, (int, float))]
            row[output] = round(statistics.median(values), 1) if values else None
        cached_all = [(r.get("usage") or {}).get("cached_tokens") for r in good]
        cached_all = [c for c in cached_all if c is not None]
        row["cached"] = round(statistics.median(cached_all)) if cached_all else None  # 取中位：前缀缓存命中随轮次递增，第一次的值不代表整体
        row["cached_known"], row["cached_hit_n"] = len(cached_all), sum(v > 0 for v in cached_all)
        # 前缀缓存趋势：首轮几乎不命中、后续命中越来越多时，后几轮的耗时偏快，中位数会被拉低。
        # 用“首轮 vs 末轮”的耗时比和 cached 增量暴露它，避免把缓存优势当成模型性能。
        row["cache_trend"] = None
        if len(good) >= 4 and len({r.get("task") for r in good}) == 1:
            def _c(r):
                return (r.get("usage") or {}).get("cached_tokens")
            withc = [r for r in good if _c(r) is not None]
            if len(withc) >= 4:
                first, last = withc[0], withc[-1]
                c0, c1 = _c(first) or 0, _c(last) or 0
                t0, t1 = first.get("total_s"), last.get("total_s")
                if c1 > c0 and t0 and t1 and t0 > 0.5:
                    row["cache_trend"] = {"cached_first": c0, "cached_last": c1,
                                           "total_first": round(t0, 3), "total_last": round(t1, 3),
                                           "speedup": round(t0 / t1, 2) if t1 > 0.05 else None}
        row["queue_median"] = row["queue"]["median"] if row["queue"] else None
        # 每千输出 tokens 的耗时：仅为描述性指标，固定等待和 tokenizer 差异未消除。
        per1k = [r["total_s"] / (u / 1000.0) for r in good
                 for u in [(r.get("usage") or {}).get("output_tokens") or 0] if u >= 20 and r.get("total_s")]
        row["sec_per_1k"] = dist([round(v, 2) for v in per1k])
        row["sec_per_1k_vals"] = [round(v, 2) for v in per1k]
        if graded:
            p = sum(1 for r in graded if r["pass"])
            row.update(pass_n=p, pass_rate=round(p / len(graded), 3), graded=len(graded))
            for r in graded:
                t = row["by_task"].setdefault(r["task"], {"name": r.get("task_name", r["task"]), "pass": 0, "n": 0})
                t["n"] += 1
                t["pass"] += 1 if r["pass"] else 0
            # 固定题目重复测量不满足普通二项 Wilson 区间假设；先只显示描述性得分。
            row.update(pass_ci=None, pass_ci_basis=None, pass_units=len(row["by_task"]))
            if sc == "quiz":
                cons = []
                for tid in {r["task"] for r in graded}:
                    ans = [_norm(r["text"]) for r in good if r.get("task") == tid]
                    if len(ans) >= 2:
                        cons.append(max(ans.count(a) for a in set(ans)) / len(ans))
                row["consistency"] = round(sum(cons) / len(cons), 3) if cons else None
        rows.append(row)
    return rows


def fmt(v, unit="s"):
    return "-" if v is None else f"{v:.1f}{unit}"


def conclusion(rows, fp=None):
    parts = []
    for sc, cond, api in dict.fromkeys((r["scenario"], r.get("cond", ""), r.get("api", "")) for r in rows):
        rs = [r for r in rows if r["scenario"] == sc and r.get("cond", "") == cond and r.get("api", "") == api]
        seg = []
        if cond.startswith("unknown:"):
            parts.append(f"【{SCENARIOS.get(sc, sc)}】测试条件未知，仅显示统计数据，不作排名")
            continue
        # 同名目标若对应多个服务身份，只展示分组信息，不能当成两个模型排名。
        duplicate_labels = len({r["label"] for r in rs}) != len(rs)
        if duplicate_labels:
            parts.append(f"【{SCENARIOS.get(sc, sc)}】同名目标的接口身份不同或未知，未作排名")
            continue
        if sc in ("modeltrace", "behavior"):
            seg.append(f"请求成功 {sum(r['ok'] for r in rs)}/{sum(r['n'] for r in rs)}；分布结果见下方，不用于身份认证")
        if sc == "modeltrace_enroll":
            seg.append(f"参考采集请求成功 {sum(r['ok'] for r in rs)}/{sum(r['n'] for r in rs)}；有效性以导出记录为准")
        full = sorted((r for r in rs if r["total"] and r["ok"] / r["n"] >= 0.8), key=lambda r: r["total"]["median"]) if sc not in ("fingerprint", "modeltrace", "modeltrace_enroll", "behavior") else []  # 指纹挑战的输出长度不同，不排名速度
        if full:
            f = full[0]
            prefix = "成功请求耗时中位数最低" if len(full) > 1 else "成功请求耗时中位数"
            txt = f"{prefix} {f['label']}（{fmt(f['total']['median'])}" + (f"，成功率 {f['ok']}/{f['n']}，只统计成功请求" if f["ok"] < f["n"] else "") + "）"
            # 输出长度差异大时，“总耗时”不可比：输出短的天然占优。给出可归一化参照（tokens/秒），并明确提示
            outs = [r["tout"] for r in full if r.get("tout")]
            if len(outs) >= 2 and max(outs) > 2 * max(1, min(outs)):
                txt += (f"。注意：各模型输出长度差 {max(outs) / max(1, min(outs)):.0f} 倍（{min(outs)}–{max(outs)} tokens），"
                        "总耗时受输出长度影响，不能直接当性能")
                fast_tps = [r for r in full if r.get("tps")]
                if fast_tps:
                    best = max(fast_tps, key=lambda r: r["tps"]["median"])
                    txt += f"；按输出速度看最高的是 {best['label']}（{best['tps']['median']} tokens/s，跨模型 token 口径可能不同）"
            if len(full) > 1:
                txt += f"；第二名 {full[1]['label']}（{fmt(full[1]['total']['median'])}）；仅描述本次样本，不能推断显著性或等效"
            seg.append(txt)
        graded = sorted((r for r in rs if r["pass_rate"] is not None), key=lambda r: -r["pass_rate"])
        if graded:
            b = graded[0]
            prefix = "正确率最高" if len(graded) > 1 else "正确率"
            txt = f"{prefix} {b['label']}（{b['pass_n']}/{b['graded']}）"
            txt += "，仅描述固定题集和本次请求"
            seg.append(txt)
        for r in rs:
            if r["ok"] < r["n"]:
                seg.append(f"{r['label']} 失败 {r['n'] - r['ok']}/{r['n']} 次：" + "，".join(f"{k}×{v}" for k, v in r["codes"].items()))
            if r.get("uncertain_delivery_n"):
                seg.append(f"{r['label']} 有 {r['uncertain_delivery_n']} 次送达状态未知，未自动重发")
        for r in rs:
            if r["ok"] and r["buffered_n"] * 2 >= r["ok"]:
                seg.append(f"{r['label']} 疑似非流式（中转站攒满后一次返回），首字不可信")
        if seg:
            parts.append(f"【{SCENARIOS.get(sc, sc)}】" + "；".join(seg))
    if any(r["pass_rate"] is not None for r in rows):
        parts.append("正确率来自少量固定题目，仅供参考，不是权威评测")
    if any(r["low_sample"] for r in rows):
        parts.append(f"部分结果样本 < {LOW_SAMPLE} 次，仅供参考")
    for r in rows:
        ct = r.get("cache_trend")
        if ct and (ct.get("speedup") or 0) >= 1.3:
            parts.append(f"⚠ {r['label']}（{SCENARIOS.get(r['scenario'], r['scenario'])}）疑似前缀缓存污染："
                         f"首条成功请求 {fmt(ct['total_first'])} → 末条 {fmt(ct['total_last'])}（快 {ct['speedup']}×，缓存命中 {ct['cached_first']}→{ct['cached_last']} tokens）。"
                         "这只是相关现象，不能确认缓存造成了变化；请核对缓存条件")
    if fp:
        for p in fp.get("pairs", []):
            if p.get("dist") is None:
                parts.append(f"模型指纹 {p['a']} vs {p['b']}：有效数字不足（{p['n_a']}/{p['n_b']}，至少 {fp['min_numbers']}），不作判断")
            else:
                parts.append(f"模型指纹 {p['a']} vs {p['b']}：距离 {p['dist']}（判定线 {p['thr']}）→ {p['verdict']}")
        if fp.get("labels") and not fp.get("pairs"):
            parts.append("模型指纹：只有一个模型的数据（有效数字 " + "、".join(f"{lb} {v['n_numbers']}" for lb, v in fp["labels"].items()) +
                         "）；再测一次（如直连 vs 中转）后到“历史与趋势”勾选两次结果，在“对比”里看指纹距离")
        parts.append("模型指纹是启发式证据，不是身份认证")
    return "\n".join(parts) or "没有测到任何模型"


# ---------------------------------------------------------------- 执行计划

def run_plan(plan, emit, stop):
    """plan: endpoints{id: ep(含 api_key, 已确定 api)}、models[(ep_id, model)]、tasks、rounds、timeout、gap、warmup、seed。
    emit(dict) 随时回报进度；stop 是 threading.Event。返回 (records, env_extra)。"""
    rng = random.Random(plan["seed"])
    pool = Pool()
    timeout, gap = plan["timeout"], plan["gap"]
    targets = [(plan["endpoints"][eid], m) for eid, m in plan["models"]]
    tasks = plan["tasks"]
    extra = {"handshake_s": {}, "warmup": {}}
    last_end = {}

    redaction_keys = plan.get("redaction_keys")
    if redaction_keys is None:
        redaction_keys = [endpoint.get("api_key") for endpoint in plan["endpoints"].values()]

    def redact_run_secrets(rec):
        for secret in redaction_keys:
            _redact_record(rec, secret)
        return rec

    def pace(ep):
        wait = last_end.get(ep["name"], -1e9) + gap - time.perf_counter()
        if wait > 0:
            stop.wait(wait)

    for ep in plan["endpoints"].values():  # 冷连接：握手耗时单独记录，之后复用这条连接
        try:
            extra["handshake_s"][ep["name"]] = pool.get(ep, timeout)[2]
        except Exception as e:  # noqa: BLE001
            extra["handshake_s"][ep["name"]] = None
            emit({"note": f"{ep['name']} 连接失败：{type(e).__name__}: {e}"})
    if plan.get("warmup"):
        for ep, m in targets:
            if stop.is_set():
                break
            emit({"phase": "warmup", "current": f"预热 {label_of(ep, m)}"})
            pace(ep)
            r = stream_request(pool, ep, make_body(ep["api"], m, DEFAULT_PROMPT, None, plan.get("effort", "")), timeout, redaction_keys=redaction_keys)
            last_end[ep["name"]] = time.perf_counter()
            extra["warmup"][label_of(ep, m)] = {"ok": r["ok"], "total_s": r["total_s"], "status": r["status"]}
    records = []
    total = plan["rounds"] * len(targets) * len(tasks)
    emit({"phase": "run", "total": total, "done": 0})
    concurrency = plan.get("concurrency", 1)
    if concurrency > 1:
        local, pools, lock = threading.local(), [], threading.Lock()
        next_start = {name: end + gap for name, end in last_end.items()}

        def worker(item):
            rd, seq, ep, model, task = item
            if stop.is_set():
                return None
            with lock:
                start = max(time.perf_counter(), next_start.get(ep["name"], 0))
                next_start[ep["name"]] = start + gap
            if stop.wait(max(0, start - time.perf_counter())):
                return None
            if not hasattr(local, "pool"):
                local.pool = Pool()
                with lock:
                    pools.append(local.pool)
            if plan.get("connection_mode", "reuse") == "cold":
                local.pool.drop(ep)
            instance = materialize_task(task, plan["seed"], rd)
            rec = run_task(local.pool, ep, model, instance, timeout, plan.get("effort", ""), redaction_keys=redaction_keys)
            redact_run_secrets(rec)
            rec["round"], rec["seq"] = rd, seq
            return rec

        seq = 0
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
                for rd in range(1, plan["rounds"] + 1):
                    items = [(ep, m, t) for ep, m in targets for t in tasks]
                    rng.shuffle(items)
                    pending = set()
                    for ep, m, task in items:
                        if stop.is_set():
                            break
                        while len(pending) >= concurrency:
                            done, pending = concurrent.futures.wait(pending, return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in done:
                                rec = f.result()
                                if rec:
                                    records.append(rec)
                                    emit({"records": records, "done": len(records)})
                        if stop.is_set():
                            break
                        seq += 1
                        pending.add(executor.submit(worker, (rd, seq, ep, m, task)))
                    for f in concurrent.futures.as_completed(pending):
                        rec = f.result()
                        if rec:
                            records.append(rec)
                            emit({"records": records, "done": len(records)})
                    if stop.is_set():
                        break
        finally:
            for p in pools:
                p.close()
            pool.close()
        records.sort(key=lambda r: r["seq"])
        return records, extra
    for rd in range(plan["rounds"]):
        items = [(ep, m, t) for ep, m in targets for t in tasks]
        rng.shuffle(items)
        for ep, m, t in items:
            if stop.is_set():
                break
            emit({"current": f"第 {rd + 1}/{plan['rounds']} 轮 · {label_of(ep, m)} · {t['name']}"})
            pace(ep)
            if stop.is_set():
                break
            instance = materialize_task(t, plan["seed"], rd + 1)
            if plan.get("connection_mode", "reuse") == "cold":
                pool.drop(ep)
            rec = run_task(pool, ep, m, instance, timeout, plan.get("effort", ""), redaction_keys=redaction_keys)
            redact_run_secrets(rec)
            last_end[ep["name"]] = time.perf_counter()
            rec["round"], rec["seq"] = rd + 1, len(records) + 1
            records.append(rec)
            emit({"records": records, "done": len(records)})
    pool.close()
    return records, extra


def make_env(plan, extra, meta):
    eps = {eid: {"name": e["name"], "base_url": e["base_url"], "api": e["api"],
                 **({"azure_api_version": e.get("azure_api_version", azure.DEFAULT_API_VERSION), "azure_auth": e.get("azure_auth", "api_key")} if e["api"] == "azure" else {}),
                 **({"aws_region": e.get("aws_region", bedrock.DEFAULT_REGION), "aws_profile": e.get("aws_profile", ""),
                     "aws_credential_source": (e.get("_aws_credentials") or {}).get("source", "unknown")} if e["api"] == "bedrock" else {}),
                 **({"auth_mode": e.get("auth_mode", "bearer")} if e["api"] in ("chat", "responses") else {}),
                 **({"auth_header": e.get("auth_header", ""), "auth_prefix": e.get("auth_prefix", "")} if e["api"] in ("chat", "responses") and e.get("auth_mode") == "custom" else {}),
                 **({"query_params": e["query_params"]} if e["api"] in ("chat", "responses", "anthropic") and e.get("query_params") else {})}
           for eid, e in plan["endpoints"].items()}
    scen = list(dict.fromkeys(t["scenario"] for t in plan["tasks"]))
    prompts = {t["id"]: hashlib.sha256((t["prompt"] + (t.get("system") or "")).encode()).hexdigest()[:12] for t in plan["tasks"]}
    scenario_fingerprints = {}
    for sc in scen:
        manifest = {"method_version": METHOD_VERSION, "tasks": [t for t in plan["tasks"] if t["scenario"] == sc]}
        if sc == "tool":
            manifest.update(tool=TOOL_DEF, result=TOOL_RESULT)
        scenario_fingerprints[sc] = hashlib.sha256(json.dumps(manifest, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()
    env = {"schema_version": SCHEMA_VERSION, "tool_version": VERSION, "created": meta["created"], "host": platform.node(),
            "python": sys.version.split()[0], "seed": plan["seed"], "rounds": plan["rounds"], "timeout_s": plan["timeout"],
            "gap_s": plan["gap"], "warmup": plan.get("warmup", False), "effort": plan.get("effort", ""), "concurrency": plan.get("concurrency", 1), "connection_mode": plan.get("connection_mode", "reuse"), "scenarios": scen, "name": meta.get("name", ""), "prompt": meta.get("prompt", ""),
            "context_tokens": meta.get("context", 0), "prompt_fingerprints": prompts, "scenario_fingerprints": scenario_fingerprints, "method_version": METHOD_VERSION, "proxy": False,
            "sampling": {"temperature": SAMPLING["temperature"], "max_tokens": MAX_TOKENS},
            "endpoints": list(eps.values()), "models": [label_of(plan["endpoints"][e], m) for e, m in plan["models"]],
            "handshake_s": extra.get("handshake_s", {}), "warmup_results": extra.get("warmup", {}),
            "note": "请求耗时不含连接建立；另记录端到端耗时。POST 送达不确定时不自动重发。普通场景采样参数固定（temperature=0、输出上限 " + str(MAX_TOKENS) + " tokens），不保证上游执行一致；重复请求可能相关，成功率区间仅供参考，正确率不计算 Wilson 区间。"}
    if "modeltrace" in scen:
        artifact = plan.get("modeltrace_artifact") or modeltrace.load_bank()
        env["modeltrace"] = {"bank_sha256": artifact["sha256"], "bank_id": plan.get("modeltrace_bank_id", "builtin"),
                             "bank_source": artifact.get("source", "upstream"),
                             "upstream_commit": artifact.get("upstream_commit", modeltrace.UPSTREAM_COMMIT),
                             "temperature": "provider_default", "challenges_per_round": 3}
        env["note"] += " ModelTrace 场景单独使用服务商默认 temperature。"
    if "modeltrace_enroll" in scen:
        env["modeltrace_enroll"] = {"suite_commit": modeltrace.UPSTREAM_COMMIT, "temperature": "provider_default", "challenges_per_model": 36}
        env["note"] += " ModelTrace 参考库采集使用服务商默认 temperature。"
    if "behavior" in scen:
        env["behavior"] = {"version": 1, "temperature": "provider_default", "probes_per_round": len(behavior.PROBES)}
        env["note"] += " 短问题行为分布场景使用服务商默认 temperature。"
    env["condition_keys"] = {sc: cond_of(env, sc) for sc in scen}
    redaction_keys = plan.get("redaction_keys")
    if redaction_keys is None:
        redaction_keys = [endpoint.get("api_key") for endpoint in plan["endpoints"].values()]
    for secret in redaction_keys:
        env = _redact_value(env, secret)
        for field in ("handshake_s", "warmup_results"):
            env[field] = _redact_value(env[field], secret, mask_keys=True)
    return env


def write_report(path, result):
    env, rows, recs = result["env"], result["summary"], result["records"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 模型测试报告 {result['id']}\n\n- 时间：{env['created']}\n- 场景：{'、'.join(SCENARIOS.get(s, s) for s in env['scenarios'])}\n"
                f"- 推理强度：{env.get('effort') or '未指定（模型默认）'}\n- 轮数：{env['rounds']}（每轮随机交替，种子 {env['seed']}），超时 {env['timeout_s']}s，逻辑任务间隔 {env['gap_s']}s，并发 {env.get('concurrency', 1)}，连接 {env.get('connection_mode', 'reuse')}，预热 {'是' if env['warmup'] else '否'}\n"
                f"- 接口：{'；'.join(e['name'] + ' ' + e['base_url'] + ' (' + e['api'] + ')' for e in env['endpoints'])}\n"
                f"- 冷连接握手：{json.dumps(env.get('handshake_s'), ensure_ascii=False)}\n- 脚本版本：{env['tool_version']}\n\n## 结论\n\n{result['conclusion']}\n\n## 汇总\n\n")
        f.write("| 场景 | 模型 | 成功 | 正确 | 总耗时 中位 / p95(n≥20)或最大 | 首字 中位 | 首个事件 中位 | 速度 tokens/s |\n|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            t, ft, fe, sp = r["total"], r["first_token"], r["first_event"], r["tps"]
            f.write(f"| {SCENARIOS.get(r['scenario'], r['scenario'])} | {r['label']} | {r['ok']}/{r['n']}{' *' if r['low_sample'] else ''} | "
                    f"{'-' if r['pass_rate'] is None else str(r['pass_n']) + '/' + str(r['graded'])} | "
                    f"{fmt(t and t['median'])} / {fmt(t and (t['p95'] if t['p95_ok'] else t['max']))} | {fmt(ft and ft['median'])}{' (非流式)' if r['buffered_n'] else ''} | {fmt(fe and fe['median'])} | "
                    f"{sp['median'] if sp else (str(r['tps_overall']['median']) + '(含等待)' if r['tps_overall'] else '-')} |\n")
        f.write("\n（* = 样本不足）\n")
        f.write(f"\n## {SLO_SECONDS} 秒限时完成\n\n逻辑任务开始后计算；对有判分的任务须答对，失败和超时计入分母，不含调度等待。\n\n")
        for r in rows:
            if r.get("slo_rate") is not None:
                f.write(f"- {r['label']} [{r['scenario']}]：{r['slo_n']}/{r['n']}；历史重试任务 {r['retry_n']}；送达状态未知 {r.get('uncertain_delivery_n', 0)}\n")
        behavior_results = result.get("behavior")
        if behavior_results:
            f.write("\n## 短问题行为分布\n\n分布距离仅供描述，没有校准后的身份判定线。\n\n")
            for label, probes in behavior_results["profiles"].items():
                f.write(f"- {label}：\n")
                for probe in probes.values():
                    counts = "、".join(f"{clip(answer, 100)} × {count}" for answer, count in probe["counts"].items()) or "无有效回答"
                    f.write(f"  - {probe['name']}：有效 {probe['valid']}/{probe['planned']}；{counts}\n")
            for pair in behavior_results["pairs"]:
                distance = "样本不足" if pair["mean_jsd"] is None else f"平均 JSD {pair['mean_jsd']:.3f}"
                f.write(f"- {pair['a']} / {pair['b']}：{distance}\n")
                for probe in pair["probes"]:
                    value = "样本不足" if probe["jsd"] is None else f"JSD {probe['jsd']:.3f}"
                    f.write(f"  - {probe['name']}：{value}（有效 {probe['a_valid']}/{probe['b_valid']}）\n")
        mt_results = result.get("modeltrace") or []
        if mt_results:
            f.write("\n## ModelTrace 候选参考\n\n")
            mt_env = env.get("modeltrace") or {}
            source_note = "Claude 参考样本来自第三方接口。" if mt_env.get("bank_source", "upstream") == "upstream" else "用户导入参考库的来源与真实性未验证。"
            f.write("分数是已收录候选之间的闭集相对权重，不是身份真实性概率；" + source_note + "\n\n")
            f.write(f"参考库 SHA-256：{mt_env.get('bank_sha256', '未知')}；上游提交：{mt_env.get('upstream_commit') or '未记录'}。\n\n")
            for item in mt_results:
                f.write(f"- {item['label']} 第 {item['round']} 轮：有效回答 {item['used_outputs']}/3")
                if item['status'] == 'ranked':
                    f.write("；候选：" + "、".join(f"{r['display_name']} {r['probability']:.1%}" for r in item['results'][:3]))
                    calibration = item.get("calibration") or {}
                    if calibration.get("fallback") or calibration.get("cv_samples") == 0:
                        f.write("；三题权重未校准（当前参考数据缺少交叉验证）")
                else:
                    reasons = {"duplicate_outputs": "回答序列重复", "patterned_output": "回答呈明显规律", "too_few_valid": "有效样本不足"}
                    f.write("；" + reasons.get(item.get("reason"), "无法评分") + "，不给候选排名")
                f.write("\n")
                for index, diag in enumerate(item["diagnostics"], 1):
                    f.write(f"  - 挑战 {index}：有效数字 {diag['parsed_numbers']}/{diag['minimum_numbers']}；{'计入' if diag['accepted'] else '未计入'}\n")
            for pair in result.get("modeltrace_pairs") or []:
                value = f"Hellinger 距离 {pair['distance']:.4f}" if pair["distance"] is not None else "样本不足或题目不一致"
                f.write(f"- 第 {pair['round']} 轮 {pair['a']} / {pair['b']}：{value}（只作描述，不能认证身份）\n")
        fp = result.get("fingerprint")
        if fp:
            f.write("\n## 模型指纹（数值分布对账）\n\n")
            for lb, v in (fp.get("labels") or {}).items():
                f.write(f"- {lb}：有效数字 {v['n_numbers']} 个，自身抽样噪声（半分距离）{v['self_dist']}\n")
            for p in fp.get("pairs", []):
                f.write(f"- {p['a']} vs {p['b']}：距离 {p['dist']}（判定线 {p['thr']}，噪声底线 {p['noise']}）→ {p['verdict']}\n")
            f.write("\n指纹是启发式证据，不是身份认证。\n")
        f.write("\n## 每次请求\n\n")
        for r in recs:
            f.write(f"- #{r.get('seq')} {r['time'][11:]} {r['label']} [{r.get('task_name', '')}] "
                    f"{'成功' if r['ok'] else '失败'} HTTP {r['status']} 总 {fmt(r['total_s'])} 首字 {fmt(r['first_token_s'])}"
                    f"{'' if r.get('pass') is None else (' 答对' if r['pass'] else ' 答错：' + r.get('grade_note', ''))}"
                    f"{' 错误：' + r['error'] if r['error'] else ''}\n")


def main():
    import argparse
    from . import store
    ap = argparse.ArgumentParser(prog="python3 -m model_test run", description="命令行跑一次测试（接口来自 data/endpoints.json，结果存 results/日期-次数/）")
    ap.add_argument("--endpoint", required=True, help="接口名称")
    ap.add_argument("--model", action="append", help="模型，可多次；默认该接口全部")
    ap.add_argument("--scenario", action="append", choices=list(SCENARIOS), help="场景，可多次；默认 speed")
    ap.add_argument("--modeltrace-bank", default="builtin", help="ModelTrace 参考库 ID：builtin 或已导入库的 SHA-256")
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--context", type=int, default=0, choices=CONTEXT_CHOICES)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--concurrency", type=int, choices=range(1, 5), default=1, help="并发逻辑任务数 1–4；可能增加费用与限流")
    ap.add_argument("--connection-mode", choices=("reuse", "cold"), default="reuse", help="每工作线程复用连接或每任务新建连接；不控制上游缓存")
    ap.add_argument("--seed", type=int, help="随机种子；复用后可生成相同题目实例和调度顺序")
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--effort", default="", choices=[e for e in EFFORTS], help="推理强度（不写 = 不指定）")
    ap.add_argument("--name", help="测试名称，文件夹变成 日期-序号_名称（中文、字母、数字、-、_，最长 30 字）")
    a = ap.parse_args()
    ep = next((e for e in store.list_endpoints(public=False) if e["name"] == a.endpoint), None)
    if not ep:
        sys.exit("找不到接口：" + a.endpoint)
    params = {"selection": [{"endpoint": ep["id"], "models": a.model or ep["models"]}], "scenarios": a.scenario or ["speed"], "prompt": a.prompt,
              "rounds": a.rounds, "timeout": a.timeout, "context": a.context, "gap": a.gap, "warmup": not a.no_warmup, "name": a.name or "", "effort": a.effort, "concurrency": a.concurrency, "connection_mode": a.connection_mode, "seed": a.seed, "modeltrace_bank": a.modeltrace_bank}
    from . import server
    params = server.validate(params)
    res = store.execute(params, lambda u: print(u.get("current", ""), flush=True) if u.get("current") else None, threading.Event())
    print("\n" + res["conclusion"] + "\n\n结果目录：" + store.run_dir(res["id"]))
