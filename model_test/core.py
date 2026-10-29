#!/usr/bin/env python3
"""模型测试引擎（网页和命令行共用）：直连任意 OpenAI 兼容接口，流式、不重试，记录状态码、分段耗时、用量、回答全文和全部流事件。

做了什么：
- 接口类型两种：responses（POST {base}/v1/responses）和 chat（POST {base}/v1/chat/completions）；地址没写版本号时自动补 /v1。
- 公平性：每个接口复用一条连接（TCP/TLS 握手单独记录，不算进请求耗时）；每个模型先发一次不计入的预热请求；
  每一轮把“全部模型 × 全部题目”随机打乱顺序再发（种子保存在结果里，可复现）；同一接口两次请求之间留间隔，避免 429。
- 计时拆开：连接、响应头、首个事件、首个思考、首个正文字、总耗时、输出速度（tokens/秒）。
- 场景：speed（自定义提示词）、quiz（10 道有标准答案的题，自动判分）、tool（两步工具调用，含把 reasoning 项回放）、
  needle10k / needle30k（大海捞针）。判分只用精确匹配、正则、JSON 比对，不让模型自评。
- 统计：中位数、p95、最小、最大；样本 < 5 标“样本不足”；成功率和正确率附 Wilson 95% 置信区间；题库另算一致性。
不改任何 OpenClaw 配置，不经过网关，不走终端代理，不保存密钥。
时间度量一律用单调时钟 time.perf_counter()（不受系统校时影响），只有记录“几点几分”用墙上时钟。
"""
import datetime as dt
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

VERSION = "2.5.0"
SCHEMA_VERSION = 2
CONTEXT_CHOICES = (0, 10000, 30000)  # 自定义提示词前的背景文字长度（约 tokens）；0 = 不加
LOW_SAMPLE = 5
P95_MIN_N = 20  # 样本少于这个数，p95 基本等于最大值，不展示
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
    "tool": "工具调用（两步）",
    "needle10k": "大海捞针（约 1 万 tokens）",
    "needle30k": "大海捞针（约 3 万 tokens）",
    "fingerprint": "模型指纹（数值分布对账）",
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
    {"id": "q04", "name": "指令遵循", "prompt": "请只输出三种水果的中文名，用英文逗号分隔，不要有其他任何文字、空格或标点。",
     "chk": ("re", r"[^,，。\s]+(,[^,，。\s]+){2}"), "ans": "苹果,香蕉,橙子"},
    {"id": "q05", "name": "千分位不拆列", "prompt": "按竖线 | 拆分成列，输出 JSON 字符串数组，不要代码块和其他文字：设备名称|1,200.00|台",
     "chk": ("json", ["设备名称", "1,200.00", "台"]), "ans": '["设备名称", "1,200.00", "台"]', "strict": True},
    {"id": "q06", "name": "算术", "prompt": "17 乘以 23 等于多少？只输出数字。", "chk": ("re", r"391"), "ans": "391"},
    {"id": "q07", "name": "日期推算", "prompt": "2026年10月12日是星期一，那么2026年10月19日是星期几？只输出星期几。",
     "chk": ("re", r"(星期一|周一)"), "ans": "星期一"},
    {"id": "q08", "name": "逻辑推理", "prompt": "小明比小红高，小红比小刚高。三个人里最矮的是谁？只输出名字。", "chk": ("re", r"小刚"), "ans": "小刚"},
    {"id": "q09", "name": "日期抽取", "prompt": "招标文件规定：投标保证金为人民币伍万元整，开标时间为2026年11月3日上午9时30分。请只输出开标日期，格式 YYYY-MM-DD。",
     "chk": ("re", r"2026-11-03"), "ans": "2026-11-03"},
    {"id": "q10", "name": "严格输出", "prompt": "请只回复一个大写字母 A，不要有其他任何字符。", "chk": ("re", r"A"), "ans": "A"},
]
GEN_PROMPT = "请写一段约 300 字的中文，说明污水处理中 A/O（缺氧/好氧）工艺的基本原理。直接写正文，不要标题和列表。"
TOOL_DEF = {"name": "lookup_price", "description": "查询某种材料的单价",
            "parameters": {"type": "object", "properties": {"item": {"type": "string", "description": "材料名称"}}, "required": ["item"]}}
TOOL_PROMPT = "请调用工具查询“MET混合滤料包”的单价，拿到结果后告诉我单价是多少（给出数字和单位即可）。"
TOOL_RESULT = {"item": "MET混合滤料包", "price": 128.5, "unit": "元/包"}

# ---------------------------------------------------------------- 模型指纹（数值分布对账）
# 思路参考开源项目 ModelTrace（MIT）与 meow-llm-detector（只借思想、自写实现、不引入依赖）：
# 让模型输出无语义随机整数，用数值分布的 Hellinger 距离判断“是不是同一个模型”——典型用法是
# 同一模型“直连官方 vs 走中转”各测一次，在“对比”里看指纹距离，对账中转有没有偷换模型。
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
    path = u.path.rstrip("/")
    for tail in ("/chat/completions", "/responses", "/models"):
        if path.endswith(tail):
            path = path[: -len(tail)]
    if not re.search(r"/v\d+\w*$", path):
        path += "/v1"
    return u.scheme, u.hostname, u.port or (443 if u.scheme == "https" else 80), path


def clip(text, n=300):
    return (text or "").replace("\n", " ")[:n]


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


def fetch_models(ep, timeout=20):
    """GET {base}/models，返回模型 id 列表（OpenAI 兼容格式）。失败抛 RuntimeError。"""
    _, _, _, prefix = endpoint_parts(ep["base_url"])
    conn, _ = _new_conn(ep, timeout)
    try:
        conn.request("GET", prefix + "/models", headers={"Authorization": "Bearer " + ep["api_key"], "Accept": "application/json",
                                                          "User-Agent": "modelbench/" + VERSION})
        resp = conn.getresponse()
        body = resp.read().decode("utf-8", "replace")
    finally:
        conn.close()
    if resp.status != 200:
        raise RuntimeError(f"HTTP {resp.status} {clip(body, 160)}")
    data = json.loads(body)
    items = data.get("data") if isinstance(data, dict) and isinstance(data.get("data"), list) else (
        data.get("models") if isinstance(data, dict) else data)
    ids = [(x.get("id") or x.get("name")) if isinstance(x, dict) else x for x in (items or [])]
    return sorted({i for i in ids if isinstance(i, str) and i})


# ---------------------------------------------------------------- 单次流式请求

def with_effort(body, api, effort):
    """推理强度：responses 传 reasoning.effort，chat 传 reasoning_effort；空字符串 = 不传。"""
    if effort:
        if api == "chat":
            body["reasoning_effort"] = effort
        else:
            body["reasoning"] = {"effort": effort}
    return body


def make_body(api, model, prompt, system=None, effort=""):
    """构造请求体。
    采样参数显式固定：不同中转站/接口的默认 temperature、max_tokens 往往不同，
    不固定的话“同一模型不同接口”的差异里会混进采样参数差异，不是模型本身差异。
    固定值同时写进环境快照（SAMPLING），复现和对比时可核对。"""
    if api == "chat":
        msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
        return with_effort({"model": model, "messages": msgs, "stream": True, "stream_options": {"include_usage": True}, **SAMPLING_CHAT}, api, effort)
    return with_effort({"model": model, "stream": True,
                        "input": ([{"role": "developer", "content": system}, {"role": "user", "content": prompt}] if system else prompt),
                        **SAMPLING_RESPONSES}, api, effort)


def _new_rec(ep, model, api):
    return {"time": dt.datetime.now().isoformat(timespec="seconds"), "endpoint": ep["name"], "model": model,
            "label": label_of(ep, model), "api": api, "ok": False, "status": None, "error": "", "reused": None, "first_reused": None,
            "connect_s": None, "headers_s": None, "first_event_s": None, "first_think_s": None, "first_token_s": None,
            "total_s": None, "tps": None, "tps_overall": None, "buffered": False, "non_stream": False, "queue_s": None,
            "reasoning_tokens": None, "response_status": None, "text": "", "tool_calls": [], "usage": None, "events": []}


def _consume(rec, ev, api, acc, now):
    if rec["first_event_s"] is None:
        rec["first_event_s"] = now

    def think():
        if rec["first_think_s"] is None:
            rec["first_think_s"] = now

    def token():
        if rec["first_token_s"] is None:
            rec["first_token_s"] = now

    if api == "chat":
        if ev.get("error"):
            rec["error"] = clip(json.dumps(ev["error"], ensure_ascii=False))
        if ev.get("usage"):
            u = ev["usage"]
            rec["usage"] = {"input_tokens": u.get("prompt_tokens"), "output_tokens": u.get("completion_tokens"),
                            "cached_tokens": u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
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
    if kind == "response.output_text.delta":
        token()
        acc["text"].append(ev.get("delta", ""))
    elif kind in ("response.reasoning_summary_text.delta", "response.reasoning_text.delta"):
        think()
    elif kind == "response.output_item.done" and ev.get("item"):
        acc["items"].append(ev["item"])
    elif kind in ("response.completed", "response.incomplete", "response.failed"):
        r = ev.get("response", {})
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
    if api == "chat":
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
                        "cached_tokens": u.get("prompt_cache_hit_tokens") or (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                        "reasoning_tokens": (u.get("completion_tokens_details") or {}).get("reasoning_tokens")}
        return
    out = payload.get("output") or payload.get("output_text") or []
    rec["response_status"] = payload.get("status") or "completed"
    rec["usage"] = payload.get("usage")
    acc["items"] = out
    texts = [(o.get("text") or "") for o in out if isinstance(o, dict) and o.get("type") == "message" for o in o.get("content") or []]
    if isinstance(payload.get("output_text"), str):
        texts.append(payload["output_text"])
    rec["text"] = "".join(texts)


def stream_request(pool, ep, body, timeout):
    """发一次流式请求（不重试）。例外：复用连接上的请求失败（服务端已把这条 keep-alive 连接关掉），
    换新连接再发一次并标 conn_retry——这是基础设施问题，不是模型/接口问题，不该计入失败率。"""
    api = ep["api"]
    rec = _new_rec(ep, body["model"], api)
    acc = {"text": [], "reasoning": [], "items": [], "calls": {}}
    path = endpoint_parts(ep["base_url"])[3] + ("/chat/completions" if api == "chat" else "/responses")
    data = json.dumps(body).encode()
    headers = {"Authorization": "Bearer " + ep["api_key"], "Content-Type": "application/json", "Accept": "text/event-stream",
               "User-Agent": "modelbench/" + VERSION}
    # 复用连接上的失败才算基础设施问题（可换连接重发）；新连接上的失败是真实的接口/网络问题，必须计入
    conn_errors = (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError,
                   http.client.ResponseNotReady, http.client.CannotSendRequest, http.client.BadStatusLine)
    try:
        for attempt in (0, 1):
            conn, reused, cs = pool.get(ep, timeout)
            if attempt == 0:
                rec["first_reused"] = reused  # 重试前的连接状态：连接复用率看这个，reused 会因重试变成 False
            rec["reused"], rec["connect_s"] = reused, cs
            t0 = time.perf_counter()
            try:
                conn.request("POST", path, body=data, headers=headers)
                resp = conn.getresponse()
                break
            except conn_errors:
                pool.drop(ep)
                if reused and attempt == 0:
                    rec["conn_retry"] = True
                    continue
                raise
        rec["status"] = resp.status
        rec["headers_s"] = round(time.perf_counter() - t0, 3)
        if resp.status != 200:
            rec["error"] = clip(resp.read().decode("utf-8", "replace"))
        elif "text/event-stream" not in (resp.getheader("Content-Type") or ""):
            # 中转站忽略 stream:true，一次性返回 JSON：能正常读出回答，但“首字”不是模型真正的首字
            raw = resp.read().decode("utf-8", "replace")
            try:
                payload = json.loads(raw)
            except ValueError:
                raise ValueError(f"返回的不是流也不是 JSON（Content-Type {resp.getheader('Content-Type')}）：{clip(raw, 120)}")
            now = round(time.perf_counter() - t0, 3)
            rec["first_event_s"] = rec["first_token_s"] = now
            rec["non_stream"] = True
            rec["events"].append({"t": now, "event": {"type": "non_stream_json", "payload": payload}})
            _consume_json_body(rec, acc, api, payload)
            calls = ([{"id": it.get("call_id"), "name": it.get("name"), "arguments": it.get("arguments")}
                      for it in acc["items"] if isinstance(it, dict) and it.get("type") == "function_call"]
                     if api == "responses" else [acc["calls"][k] for k in sorted(acc["calls"])])
            rec["tool_calls"] = calls
            rec["ok"] = rec["response_status"] in ("completed", "incomplete") and bool(rec["text"] or calls)
            if not rec["ok"] and not rec["error"]:
                rec["error"] = "非流式响应里没有正文"
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
            rec["text"] = "".join(acc["text"])
            calls = ([{"id": it.get("call_id"), "name": it.get("name"), "arguments": it.get("arguments")}
                      for it in acc["items"] if isinstance(it, dict) and it.get("type") == "function_call"]
                     if api == "responses" else [acc["calls"][k] for k in sorted(acc["calls"])])
            rec["tool_calls"] = calls
            rec["ok"] = rec["response_status"] == "completed" and bool(rec["text"] or calls)
            if not rec["response_status"] and not rec["error"]:
                rec["error"] = "流在完成事件之前断开"
        if resp.will_close:
            pool.drop(ep)
    except Exception as e:  # noqa: BLE001
        rec["error"] = clip(f"{type(e).__name__}: {e}")
        pool.drop(ep)
    rec["total_s"] = round(time.perf_counter() - t0, 3)
    # 排队时间：响应头到手之后、首个数据事件之前的等待（服务端/中转站内部排队，不是网络也不是模型）
    if rec["headers_s"] is not None and rec["first_event_s"] is not None:
        rec["queue_s"] = round(max(rec["first_event_s"] - rec["headers_s"], 0), 3)
    usage = rec["usage"] or {}
    out = usage.get("output_tokens")
    rec["reasoning_tokens"] = usage.get("reasoning_tokens") or (usage.get("output_tokens_details") or {}).get("reasoning_tokens")
    start = rec["first_token_s"] if rec["first_token_s"] is not None else rec["first_event_s"]
    visible = (out or 0) - (rec["reasoning_tokens"] or 0)  # 输出速度只算“可见回答”的 tokens：推理 tokens 是首个可见字之前产生的，时间不在分母里，算进去会虚高
    if visible > 0 and start is not None and rec["total_s"] - start > 0.05:
        rec["tps"] = round(visible / (rec["total_s"] - start), 1)
    if out and rec["total_s"] > 0.05:
        rec["tps_overall"] = round(out / rec["total_s"], 1)  # 含等待的整体速度；中转站攒满返回时 tps 算不出，用它
    # 非流式：响应几乎在最后一刻才一次性到达（首个事件≈总耗时），此时“首字”不是模型真正的首字
    rec["buffered"] = bool(rec["ok"] and rec["first_event_s"] is not None and rec["total_s"] >= BUFFER_MIN_TOTAL
                           and rec["total_s"] - rec["first_event_s"] <= 0.03 * rec["total_s"] + 0.1) or rec["non_stream"]
    rec["_replay"] = {"items": acc["items"], "reasoning": "".join(acc["reasoning"]), "text": rec["text"]}
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
        elif sc == "quiz":
            tasks += [{"id": q["id"], "scenario": "quiz", "kind": "plain", "name": q["name"], "prompt": q["prompt"], "chk": q["chk"],
                       **({"strict": True} if q.get("strict") else {})} for q in QUIZ]
        elif sc == "tool":
            tasks.append({"id": "tool", "scenario": "tool", "kind": "tool", "name": "两步工具调用", "prompt": TOOL_PROMPT})
        elif sc in ("needle10k", "needle30k"):
            tasks.append({"id": sc, "scenario": sc, "kind": "plain", "name": SCENARIOS[sc], "system": context_text(10000 if sc == "needle10k" else 30000, True),
                          "prompt": "上文背景材料里写了“本次测试的暗号”，它是什么？只输出暗号本身。", "chk": ("contains", "青竹-7392")})
        elif sc == "fingerprint":
            tasks += [{"id": f"fp{i + 1}", "scenario": "fingerprint", "kind": "plain", "name": f"数值指纹 {i + 1}", "prompt": p} for i, p in enumerate(FP_PROMPTS)]
    return tasks


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
    t = (text or "").strip()
    if not strict:
        t = normalize_answer(t)[0]
    if kind == "re":
        return re.fullmatch(arg, t) is not None
    if kind == "json":
        try:
            return json.loads(t) == arg
        except ValueError:
            return False
    return arg in t


def _grade(rec, task):
    rec["scenario"], rec["task"], rec["task_name"] = task["scenario"], task["id"], task["name"]
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
    return call.get("name") == TOOL_DEF["name"] and isinstance(args, dict) and "MET" in str(args.get("item", ""))


def run_tool(pool, ep, model, task, timeout, effort=""):
    """两步工具调用：①模型必须调用 lookup_price 且参数是合法 JSON；②把工具结果（和第一步输出的推理项）回传后，模型要给出 128.5。"""
    api = ep["api"]
    tool = ({"type": "function", "function": TOOL_DEF} if api == "chat" else dict(TOOL_DEF, type="function"))
    user = {"role": "user", "content": task["prompt"]}
    body1 = with_effort({"model": model, "messages": [user], "tools": [tool], "stream": True, "stream_options": {"include_usage": True}} if api == "chat"
                        else {"model": model, "input": [user], "tools": [tool], "stream": True}, api, effort)
    r1 = stream_request(pool, ep, body1, timeout)
    _grade(r1, task)
    r1["steps"] = [{"step": 1, "ok": r1["ok"], "status": r1["status"], "total_s": r1["total_s"], "error": r1["error"]}]
    call = next((c for c in r1["tool_calls"] if c.get("name")), None)
    if not r1["ok"] or not call or not _tool_args_ok(call):
        r1["pass"] = False
        r1["grade_note"] = ("第一步请求失败：" + (r1["error"] or "") if not r1["ok"] else
                            "第一步没有调用工具" if not call else "第一步工具参数不合法：" + clip(str(call.get("arguments")), 80))
        return r1
    result = json.dumps(TOOL_RESULT, ensure_ascii=False)
    rp = r1["_replay"]
    if api == "chat":
        asst = {"role": "assistant", "content": rp["text"] or None,
                "tool_calls": [{"id": call["id"], "type": "function", "function": {"name": call["name"], "arguments": call["arguments"]}}]}
        if rp["reasoning"]:
            asst["reasoning_content"] = rp["reasoning"]
        body2 = with_effort({"model": model, "messages": [user, asst, {"role": "tool", "tool_call_id": call["id"], "content": result}],
                             "tools": [tool], "stream": True, "stream_options": {"include_usage": True}}, api, effort)
    else:
        body2 = with_effort({"model": model, "tools": [tool], "stream": True,
                             "input": [user] + rp["items"] + [{"type": "function_call_output", "call_id": call["id"], "output": result}]}, api, effort)
    r2 = stream_request(pool, ep, body2, timeout)
    rec = dict(r2)
    rec["time"] = r1["time"]
    rec["total_s"] = round(r1["total_s"] + r2["total_s"], 3)
    rec["gen_s"] = r2["total_s"]  # 第二步（拿到工具结果后生成答案）单独一份：和单次请求的总耗时可比；两步合计用于看端到端代价
    rec["first_event_s"] = r1["first_event_s"]
    rec["events"] = [dict(e, step=1) for e in r1["events"]] + [dict(e, step=2) for e in r2["events"]]
    rec["steps"] = r1["steps"] + [{"step": 2, "ok": r2["ok"], "status": r2["status"], "total_s": r2["total_s"], "error": r2["error"]}]
    rec["ok"] = r1["ok"] and r2["ok"]
    rec["error"] = ("第二步：" + r2["error"]) if r2["error"] else ""
    _grade(rec, task)
    rec["pass"] = bool(rec["ok"] and "128.5" in rec["text"])
    rec["grade_note"] = "" if rec["pass"] else ("第二步请求失败：" + (r2["error"] or f"HTTP {r2['status']}") if not r2["ok"] else "回答里没有 128.5：" + clip(rec["text"], 60))
    return rec


def run_task(pool, ep, model, task, timeout, effort=""):
    if task["kind"] == "tool":
        return run_tool(pool, ep, model, task, timeout, effort)
    rec = stream_request(pool, ep, make_body(ep["api"], model, task["prompt"], task.get("system"), effort), timeout)
    _grade(rec, task)
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
        return "指纹一致", thr
    if dist >= 2 * thr:
        return "指纹不一致", thr
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
    """条件指纹：两次测试只有这个指纹完全相同才能合并样本，否则差值的来源不只是随机波动。
    包含：推理强度 + 提示词/背景指纹 + 采样参数（v2.4 起显式固定；旧结果记“未记录”）。
    采样参数没记录与已固定不能互合——一个用接口默认 temperature，一个是 0，
    否则会把“采样差异”当成模型差异。"""
    env = env or {}
    eff = env.get("effort") or ""
    fp = (env.get("prompt_fingerprints") or {}).get(scenario, "")
    sm = env.get("sampling")
    smk = f"t{sm.get('temperature')}/m{sm.get('max_tokens')}" if sm else "未记录采样"
    ctx = env.get("context_tokens") or 0
    return f"{eff}|{fp}|{smk}|ctx{ctx}"


def cond_label(env, scenario):
    """人能看懂的同一组条件说明，用于页面展示。"""
    env = env or {}
    eff = env.get("effort") or "未指定"
    sm = env.get("sampling")
    parts = [f"强度 {eff}"]
    if scenario == "speed":
        parts.insert(0, f"提示词「{(env.get('prompt') or '')[:12]}」")
    parts.append(f"采样 temperature={sm['temperature']}" if sm else "采样未记录（旧结果）")
    if env.get("context_tokens"):
        parts.append(f"背景 {env['context_tokens']}")
    return " · ".join(parts)


def merge_analysis(runs):
    """多次测试合并：把同一（模型, 场景, 条件）的请求合在一起重算统计，样本量上去后结论更可靠。
    条件不同的组不合并、单独成组并注明——绝不把不同条件的结果混在一起算。"""
    groups, sources, group_env = {}, {}, {}
    for run in runs or []:
        env = run.get("env") or {}
        rid = run.get("id")
        for rec in run.get("records") or []:
            sc = rec.get("scenario", "speed")
            label = rec.get("label") or rec.get("model")
            ck = cond_of(env, sc) + "|" + sc
            key = (label, sc, ck)
            groups.setdefault(key, []).append(dict(rec, cond=ck))  # 打上条件标记，summarize 按它分组，不同条件不会混算
            group_env.setdefault(key, env)  # 记住这一组自己的环境（不同条件组的 env 不同，标签才不会都一样）
            sources.setdefault(key, {})[rid] = sources.setdefault(key, {}).get(rid, 0) + 1
    if not groups:
        return None
    merged = []
    for key, recs in groups.items():
        label, sc, cond = key
        env = group_env.get(key) or {}
        merged.append({"label": label, "scenario": sc, "cond": cond, "cond_label": cond_label(env, sc),
                       "runs": sorted(k for k in sources[key] if k), "n_per_run": [sources[key][k] for k in sorted(k for k in sources[key] if k)],
                       "records": recs})
    # 合并后的记录重新汇总（复用同一套统计，保证与单次测试口径一致）
    all_recs = [r for g in merged for r in g["records"]]
    rows = summarize(all_recs)
    # 把每个汇总行指回它的来源测试，便于展示“这行是几次测试合并来的”
    by_key = {(g["label"], g["scenario"], g["cond"]): g for g in merged}
    for row in rows:
        g = by_key.get((row["label"], row["scenario"], row.get("cond", "")))
        if g:
            row["merged_runs"], row["merged_n_per_run"], row["merged_cond"] = g["runs"], g["n_per_run"], g["cond_label"]
    # 条件不一致的提示：同一（模型, 场景）出现多个条件时，说明它们没被合并
    split = {}
    for g in merged:
        split.setdefault((g["label"], g["scenario"]), set()).add(g["cond"])
    warnings = [f"{lb} 在「{sc}」里有 {len(cs)} 种不同条件，已分开统计、未合并"
                for (lb, sc), cs in split.items() if len(cs) > 1]
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
    keys = list(dict.fromkeys((r.get("label") or r["model"], r.get("scenario", "speed"), r.get("cond", "")) for r in records))
    for label, sc, cond in keys:
        rs = [r for r in records if (r.get("label") or r["model"]) == label and r.get("scenario", "speed") == sc
              and r.get("cond", "") == cond]
        good = [r for r in rs if r["ok"]]
        codes = {}
        for r in rs:
            if not r["ok"]:
                k = str(r["status"] or "无响应")
                codes[k] = codes.get(k, 0) + 1
        graded = [r for r in rs if r.get("pass") is not None]
        row = {"label": label, "scenario": sc, "cond": cond, "n": len(rs), "ok": len(good), "ok_ci": wilson(len(good), len(rs)), "codes": codes,
               "total": dist([r["total_s"] for r in good]), "first_token": dist([r["first_token_s"] for r in good]),
               "first_event": dist([r["first_event_s"] for r in good]), "first_think": dist([r.get("first_think_s") for r in good]),
               "tps": dist([r.get("tps") for r in good]), "connect": dist([r["connect_s"] for r in rs if r.get("first_reused") is False]),
               "tps_overall": dist([r.get("tps_overall") for r in good]), "total_vals": [round(r["total_s"], 3) for r in good if r["total_s"] is not None],
               "first_token_vals": [round(r["first_token_s"], 3) for r in good if r["first_token_s"] is not None],
               "tps_vals": [r["tps"] for r in good if r.get("tps") is not None],
               "gen": dist([r.get("gen_s") for r in good]), "gen_vals": [round(r["gen_s"], 3) for r in good if r.get("gen_s") is not None],
               "queue": dist([r.get("queue_s") for r in good]),
               "queue_vals": [round(r["queue_s"], 3) for r in good if r.get("queue_s") is not None],
               "think_share": None, "reasoning_tokens": dist([r.get("reasoning_tokens") for r in good]), "reasoning_share": None,
               "buffered_n": sum(1 for r in good if r.get("buffered")),
               "low_sample": len(rs) < LOW_SAMPLE,  # 请求数不足
               "low_ok_sample": 0 < len(good) < LOW_SAMPLE <= len(rs),  # 请求够多，但成功的太少：耗时只统计成功请求，实际用来算耗时的样本不足
               "pass_n": None, "pass_rate": None, "pass_ci": None,
               "consistency": None, "by_task": {}, "tin": None, "tout": None, "cached": None, "queue_median": None, "cache_trend": None, "sec_per_1k": None, "sec_per_1k_vals": []}
        shares = [max(r["first_token_s"] - r["first_event_s"], 0) / r["total_s"] for r in good
                  if r["first_token_s"] is not None and r["first_event_s"] is not None and r["total_s"]]
        row["think_share"] = round(statistics.median(shares), 3) if shares else None
        rshare = [r["reasoning_tokens"] / r["usage"]["output_tokens"] for r in good
                  if r.get("reasoning_tokens") is not None and (r.get("usage") or {}).get("output_tokens")]
        row["reasoning_share"] = round(statistics.median(rshare), 3) if rshare else None
        u = (good[0]["usage"] or {}) if good else {}
        row["tin"], row["tout"] = u.get("input_tokens"), u.get("output_tokens")
        cached_all = [(r.get("usage") or {}).get("cached_tokens") for r in good]
        cached_all = [c for c in cached_all if c is not None]
        row["cached"] = round(statistics.median(cached_all)) if cached_all else None  # 取中位：前缀缓存命中随轮次递增，第一次的值不代表整体
        # 前缀缓存趋势：首轮几乎不命中、后续命中越来越多时，后几轮的耗时偏快，中位数会被拉低。
        # 用“首轮 vs 末轮”的耗时比和 cached 增量暴露它，避免把缓存优势当成模型性能。
        row["cache_trend"] = None
        if len(good) >= 4:
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
        # 每千输出 tokens 的耗时：输出长度不同的模型之间，这个比“总耗时”更公
        # （总耗时 = 首字等待 + 生成时间，输出越长越吃亏；每千 tokens 把长度归一化掉）
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
            if len(row["by_task"]) > 1:
                # 同一道题重复多轮不是独立样本：区间按“题”计（每题取各轮答对比例），避免 10 题测 10 轮被当成 100 个独立样本、区间虚窄
                units = len(row["by_task"])
                row.update(pass_ci=wilson(sum(t["pass"] / t["n"] for t in row["by_task"].values()), units), pass_ci_basis="题目", pass_units=units)
            else:
                row.update(pass_ci=wilson(p, len(graded)), pass_ci_basis="请求", pass_units=len(graded))
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
    for sc in dict.fromkeys(r["scenario"] for r in rows):
        rs = [r for r in rows if r["scenario"] == sc]
        seg = []
        full = sorted((r for r in rs if r["total"] and r["ok"] / r["n"] >= 0.8), key=lambda r: r["total"]["median"]) if sc != "fingerprint" else []  # 成功率 ≥80% 才参与比速度；耗时只统计成功请求；指纹场景不比速度
        if full:
            f = full[0]
            txt = f"最快 {f['label']}（总耗时中位数 {fmt(f['total']['median'])}" + (f"，成功率 {f['ok']}/{f['n']}，只统计成功请求" if f["ok"] < f["n"] else "") + "）"
            # 输出长度差异大时，“总耗时”不可比：输出短的天然占优。给出可归一化参照（tokens/秒），并明确提示
            outs = [r["tout"] for r in full if r.get("tout")]
            if len(outs) >= 2 and max(outs) > 2 * max(1, min(outs)):
                fast_tps = [r for r in full if r.get("tps")]
                if fast_tps:
                    best = max(fast_tps, key=lambda r: r["tps"]["median"])
                    txt += (f"。注意：各模型输出长度差 {max(outs) / max(1, min(outs)):.0f} 倍（{min(outs)}–{max(outs)} tokens），"
                            f"总耗时受输出长度影响，不能直接当性能；按输出速度看最高的是 {best['label']}（{best['tps']['median']} tokens/s）")
            if len(full) > 1:
                s2 = full[1]
                p = mwu_p(f["total_vals"], s2["total_vals"])
                n_small = min(len(f["total_vals"]), len(s2["total_vals"]))
                if p is not None and n_small >= 4:
                    txt += f"，显著快于第二名 {s2['label']}（p={p}）" if p < 0.05 else f"，但与 {s2['label']} 差异不明显（p={p}）"
                    # 检验力提示：小样本下“差异不明显”常常是样本不够，不是真的没差别
                    if p >= 0.05:
                        rel = abs(f["total"]["median"] - s2["total"]["median"]) / max(s2["total"]["median"], 1e-9)
                        if n_small < 8 or rel < 0.35:
                            txt += f"；注意每边仅 {n_small} 个样本，检验力低，真实差距在 30% 以内时很难检出——多测几轮再到“历史与趋势”勾选多次结果合并对比"
                elif f["total"]["max"] >= s2["total"]["min"]:
                    txt += f"，与 {s2['label']} 范围重叠，差异不明显（样本少）"
            seg.append(txt)
        graded = sorted((r for r in rs if r["pass_rate"] is not None), key=lambda r: -r["pass_rate"])
        if graded:
            b = graded[0]
            txt = f"正确率最高 {b['label']}（{b['pass_n']}/{b['graded']}）"
            if len(graded) > 1 and b["pass_ci"] and graded[1]["pass_ci"] and b["pass_ci"][0] <= graded[1]["pass_ci"][1]:
                txt += f"，与 {graded[1]['label']} 的区间重叠，差异不明显"
            seg.append(txt)
        for r in rs:
            if r["ok"] < r["n"]:
                seg.append(f"{r['label']} 失败 {r['n'] - r['ok']}/{r['n']} 次：" + "，".join(f"{k}×{v}" for k, v in r["codes"].items()))
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
                         f"首轮 {fmt(ct['total_first'])} → 末轮 {fmt(ct['total_last'])}（快 {ct['speedup']}×，缓存命中 {ct['cached_first']}→{ct['cached_last']} tokens）。"
                         f"后几轮偏快，中位数被拉低；要公平比较请让各模型的缓存条件一致，或看首轮耗时")
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
            r = stream_request(pool, ep, make_body(ep["api"], m, DEFAULT_PROMPT, None, plan.get("effort", "")), timeout)
            last_end[ep["name"]] = time.perf_counter()
            extra["warmup"][label_of(ep, m)] = {"ok": r["ok"], "total_s": r["total_s"], "status": r["status"]}
    records = []
    total = plan["rounds"] * len(targets) * len(tasks)
    emit({"phase": "run", "total": total, "done": 0})
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
            rec = run_task(pool, ep, m, t, timeout, plan.get("effort", ""))
            last_end[ep["name"]] = time.perf_counter()
            rec["round"], rec["seq"] = rd + 1, len(records) + 1
            records.append(rec)
            emit({"records": records, "done": len(records)})
    pool.close()
    return records, extra


def make_env(plan, extra, meta):
    eps = {eid: {"name": e["name"], "base_url": e["base_url"], "api": e["api"]} for eid, e in plan["endpoints"].items()}
    scen = list(dict.fromkeys(t["scenario"] for t in plan["tasks"]))
    prompts = {t["id"]: hashlib.sha256((t["prompt"] + (t.get("system") or "")).encode()).hexdigest()[:12] for t in plan["tasks"]}
    return {"schema_version": SCHEMA_VERSION, "tool_version": VERSION, "created": meta["created"], "host": platform.node(),
            "python": sys.version.split()[0], "seed": plan["seed"], "rounds": plan["rounds"], "timeout_s": plan["timeout"],
            "gap_s": plan["gap"], "warmup": plan.get("warmup", False), "effort": plan.get("effort", ""), "scenarios": scen, "name": meta.get("name", ""), "prompt": meta.get("prompt", ""),
            "context_tokens": meta.get("context", 0), "prompt_fingerprints": prompts, "proxy": False,
            "sampling": {"temperature": SAMPLING["temperature"], "max_tokens": MAX_TOKENS},
            "endpoints": list(eps.values()), "models": [label_of(plan["endpoints"][e], m) for e, m in plan["models"]],
            "handshake_s": extra.get("handshake_s", {}), "warmup_results": extra.get("warmup", {}),
            "note": "耗时不含连接建立（握手单列）；每轮随机交替；成功率/正确率区间为 Wilson 95%；采样参数已固定（temperature=0、输出上限 " + str(MAX_TOKENS) + " tokens），不同接口之间可比"}


def write_report(path, result):
    env, rows, recs = result["env"], result["summary"], result["records"]
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"# 模型测试报告 {result['id']}\n\n- 时间：{env['created']}\n- 场景：{'、'.join(SCENARIOS.get(s, s) for s in env['scenarios'])}\n"
                f"- 推理强度：{env.get('effort') or '未指定（模型默认）'}\n- 轮数：{env['rounds']}（每轮随机交替，种子 {env['seed']}），超时 {env['timeout_s']}s，间隔 {env['gap_s']}s，预热 {'是' if env['warmup'] else '否'}\n"
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
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--context", type=int, default=0, choices=CONTEXT_CHOICES)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--timeout", type=int, default=90)
    ap.add_argument("--gap", type=float, default=1.5)
    ap.add_argument("--no-warmup", action="store_true")
    ap.add_argument("--effort", default="", choices=[e for e in EFFORTS], help="推理强度（不写 = 不指定）")
    ap.add_argument("--name", help="测试名称，文件夹变成 日期-序号_名称（中文、字母、数字、-、_，最长 30 字）")
    a = ap.parse_args()
    ep = next((e for e in store.list_endpoints(public=False) if e["name"] == a.endpoint), None)
    if not ep:
        sys.exit("找不到接口：" + a.endpoint)
    params = {"selection": [{"endpoint": ep["id"], "models": a.model or ep["models"]}], "scenarios": a.scenario or ["speed"], "prompt": a.prompt,
              "rounds": a.rounds, "timeout": a.timeout, "context": a.context, "gap": a.gap, "warmup": not a.no_warmup, "name": a.name or "", "effort": a.effort}
    res = store.execute(params, lambda u: print(u.get("current", ""), flush=True) if u.get("current") else None, threading.Event())
    print("\n" + res["conclusion"] + "\n\n结果目录：" + store.run_dir(res["id"]))


