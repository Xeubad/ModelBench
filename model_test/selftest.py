#!/usr/bin/env python3
"""仪器自检：本机起一个假模型服务，故意设定延迟、故障、工具调用，验证测试平台自己测得准不准。

假服务的模型名决定行为：delay-0.5 / delay-2（首个事件延迟 0.5s / 2s）、err-503 等（直接返回该状态码）、
quiz-good / quiz-bad（题库全对 / 全错）、tool-ok（两步工具调用；第二步若没带回 reasoning 项就返回 400，复现 reasoning_text 回归）、needle-good。
用法：python3 -m model_test selftest        （网页“仪器自检”按钮调用同一套检查）
"""
import json
import random
import re
import threading
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from . import core

KEY = "testkey"
MODELS = ["delay-0.5", "delay-2", "echo-effort", "slow-gen", "reason-gen", "quiz-good", "quiz-bad", "tool-ok", "needle-good",
          "fp-flat", "fp-flat2", "fp-peak", "fp-few", "err-503", "err-504", "err-429", "err-401", "err-404", "err-400"]
NONSTREAM = "non-stream"  # 忽略 stream:true、一次性返回 application/json 的假模型（中转站常见行为）
TIMING_TOLERANCE_S = 0.05
_fp_seq = {}  # 模型指纹假模型的请求计数（每次请求换一批确定性抽样）


def _user_text(body):
    if "messages" in body:
        users = [m for m in body["messages"] if m.get("role") == "user"]
        return users[-1]["content"] if users else ""
    inp = body.get("input")
    if isinstance(inp, str):
        return inp
    users = [i for i in inp if isinstance(i, dict) and i.get("role") == "user"]
    return users[-1]["content"] if users else ""


class Fake(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # 长连接，才能验证连接复用

    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        data = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def _chunk(self, s):
        b = s.encode()
        self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
        self.wfile.flush()

    def _authed(self):
        if self.headers.get("Authorization") != "Bearer " + KEY:
            self._json(401, {"error": {"message": "bad key"}})
            return False
        return True

    def do_GET(self):
        if self._authed():
            self._json(200, {"object": "list", "data": [{"id": m} for m in MODELS]} if self.path.endswith("/models") else {"error": "not found"})

    def _non_stream(self, body):
        chat = self.path.endswith("/chat/completions")
        user = _user_text(body)
        content = "青竹-7392" if "暗号" in user else "单价是 128.5 元/包" if "单价" in user else "好"
        if chat:
            payload = {"id": "ns", "object": "chat.completion", "model": body.get("model"),
                       "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                       "usage": {"prompt_tokens": 12, "completion_tokens": 8}}
        else:
            payload = {"id": "ns", "status": "completed", "output": [{"type": "message", "role": "assistant",
                       "content": [{"type": "output_text", "text": content}]}], "usage": {"input_tokens": 12, "output_tokens": 8}}
        return self._json(200, payload)

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        if not self._authed():
            return
        model, chat = body.get("model", ""), self.path.endswith("/chat/completions")
        m = re.fullmatch(r"err-(\d{3})", model)
        if m:
            return self._json(int(m.group(1)), {"error": {"message": "fake " + m.group(1)}})
        user, text, items, calls = _user_text(body), "好", [], []
        effort = (body.get("reasoning") or {}).get("effort") or body.get("reasoning_effort") or "none"
        think = model == "reason-gen"  # 先“思考”0.5 秒（不可见），再每 0.1 秒吐一个字；usage 里 100 个输出 tokens 其中 60 个是推理
        slow = model in ("slow-gen", "reason-gen")
        if model == "quiz-good":
            text = next((q["ans"] for q in core.QUIZ if q["prompt"] == user), "好")
        elif model == "quiz-bad":
            text = "不知道"
        elif model == "echo-effort":
            text = effort
        elif slow:
            text = "一二三四五"
        elif model == "needle-good":
            text = "青竹-7392"
        elif model == NONSTREAM:  # 非流式 JSON 响应：不走 SSE 分支，直接在下面一次性返回
            return self._non_stream(body)
        elif model in ("fp-flat", "fp-flat2", "fp-peak", "fp-few"):
            # 模型指纹挑战：fp-flat / fp-flat2 同分布（fp-flat2 用不同抽样）；fp-peak 集中在 1–20；fp-few 数量不够（应拒判）
            lo, hi = (1, 20) if model == "fp-peak" else core.FP_RANGE
            n = 5 if model == "fp-few" else core.FP_COUNT
            _fp_seq[model] = _fp_seq.get(model, 0) + 1  # 每次请求换一批抽样（仍确定性），否则各轮输出完全一样会让半分噪声失真
            rnd = random.Random(zlib.crc32(user.encode()) + 1000 * _fp_seq[model] + (0 if model == "fp-flat" else 1))
            text = ", ".join(str(rnd.randint(lo, hi)) for _ in range(n))
        elif model == "tool-ok":
            msgs = body.get("messages") or body.get("input") or []
            second = any((isinstance(i, dict) and (i.get("type") == "function_call_output" or i.get("role") == "tool")) for i in msgs)
            if second:
                if not chat and not any(isinstance(i, dict) and i.get("type") == "reasoning" for i in msgs):
                    return self._json(400, {"error": {"message": "reasoning_text must be passed back"}})
                text = "单价是 128.5 元/包"
            else:
                text = ""
                calls = [{"id": "call_1", "name": "lookup_price", "arguments": json.dumps({"item": "MET混合滤料包"}, ensure_ascii=False)}]
                items = [{"type": "reasoning", "id": "rs_1", "summary": []},
                         {"type": "function_call", "call_id": "call_1", "name": "lookup_price", "arguments": calls[0]["arguments"], "status": "completed"}]
        delay = float(model[6:]) if model.startswith("delay-") else 0.0
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()
        time.sleep(delay)
        ev = lambda o: self._chunk("data: " + json.dumps(o, ensure_ascii=False) + "\n\n")  # noqa: E731
        out_tok = 100 if think else 50 if slow else 8
        reason_tok = 60 if think else 0
        chunks = list(text) if slow else [text]

        def deltas(make):
            for i, c in enumerate(chunks):
                if think and i == 0:
                    time.sleep(0.5)
                if c:
                    ev(make(c))
                    if slow and i < len(chunks) - 1:
                        time.sleep(0.1)  # 每 0.1 秒一个字，第一个字之后共 0.4 秒 → 50 tokens ≈ 125 tokens/s

        usage_r = {"input_tokens": 12, "output_tokens": out_tok, "input_tokens_details": {"cached_tokens": 0}, "output_tokens_details": {"reasoning_tokens": reason_tok}}
        if chat:
            ev({"choices": [{"delta": {"role": "assistant"}}]})
            if calls:
                ev({"choices": [{"delta": {"tool_calls": [{"index": 0, "id": calls[0]["id"], "function": {"name": calls[0]["name"], "arguments": calls[0]["arguments"]}}]}}]})
            deltas(lambda c: {"choices": [{"delta": {"content": c}}]})
            ev({"choices": [{"delta": {}, "finish_reason": "tool_calls" if calls else "stop"}]})
            ev({"choices": [], "usage": {"prompt_tokens": 12, "completion_tokens": out_tok, "completion_tokens_details": {"reasoning_tokens": reason_tok}}})
            self._chunk("data: [DONE]\n\n")
        else:
            ev({"type": "response.created", "response": {"status": "in_progress"}})
            for it in items:
                ev({"type": "response.output_item.done", "item": it})
            deltas(lambda c: {"type": "response.output_text.delta", "delta": c})
            ev({"type": "response.completed", "response": {"status": "completed", "usage": usage_r, "output": items}})
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def start_fake():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def fake_endpoint(base, api="responses"):
    return {"name": "fake", "base_url": base, "api_key": KEY, "api": api}


def run_checks():
    """返回 [{name, ok, detail}]。"""
    srv, base = start_fake()
    out = []

    def check(name, fn):
        try:
            ok, detail = fn()
        except Exception as e:  # noqa: BLE001
            ok, detail = False, f"{type(e).__name__}: {e}"
        out.append({"name": name, "ok": bool(ok), "detail": detail})

    ep, pool = fake_endpoint(base), core.Pool()
    req = lambda model, e=None: core.stream_request(pool, e or ep, core.make_body((e or ep)["api"], model, "只回复：好"), 20)  # noqa: E731

    def timing(delay):
        def f():
            req("delay-0.5")  # 先建好连接，计时不含握手
            r = req(f"delay-{delay}")
            err = abs(r["first_event_s"] - delay)
            return err <= TIMING_TOLERANCE_S, f"设定 {delay}s，测得首个事件 {r['first_event_s']}s，误差 {err * 1000:.0f}ms（允许 {TIMING_TOLERANCE_S * 1000:.0f}ms）"
        return f

    check("计时误差：延迟 0.5s", timing(0.5))
    check("计时误差：延迟 2s", timing(2))

    def reuse():
        a, b = req("delay-0.5"), req("delay-0.5")
        return b["reused"] is True and b["connect_s"] == 0.0, f"第二次请求 reused={b['reused']}，connect_s={b['connect_s']}"
    check("连接复用（握手不计入请求耗时）", reuse)

    def codes():
        bad = []
        for c in (503, 504, 429, 401, 404, 400):
            r = req(f"err-{c}")
            if r["status"] != c or r["ok"]:
                bad.append(f"{c}→{r['status']}")
        return not bad, "6 种状态码均被正确识别" if not bad else "识别错误：" + "，".join(bad)
    check("状态码分类：503/504/429/401/404/400", codes)

    def nonstream(api):
        def f():
            e = fake_endpoint(base, api)
            t = core.build_tasks(["needle10k"])[0]
            core.run_task(pool, e, NONSTREAM, t, 20)  # 第一次建好连接
            r = core.run_task(pool, e, NONSTREAM, t, 20)
            ok = (r["ok"] and r["pass"] and r["non_stream"] and r["buffered"]
                  and r["text"] == "青竹-7392" and r["status"] == 200)
            return ok, ("忽略 stream:true 返回 JSON 的中转站：能正常读出回答并判对，标记 non_stream/buffered"
                        if ok else f"ok={r['ok']} pass={r.get('pass')} non_stream={r['non_stream']} text={r['text']!r} err={r['error']!r}")
        return f
    check("非流式 JSON 响应（中转站攒满返回）", nonstream("responses"))
    check("非流式 JSON 响应（chat）", nonstream("chat"))

    def conn_retry():
        """服务端每服务一个请求就关连接：下一次复用这条连接必然失败。
        平台应换新连接重发（标 conn_retry）而不是把这次记成失败——keep-alive 超时后的连接复用是基础设施问题。"""
        last = ""
        for _ in range(3):  # 时序敏感：偶发服务端线程没来得及关连接，重试几次再判定
            hits = {"n": 0}

            class CloseAfterOne(Fake):
                daemon_threads = True

                def do_POST(self):
                    Fake.do_POST(self)
                    hits["n"] += 1
                    self.close_connection = True  # 每服务一个请求就关连接

            srv2 = ThreadingHTTPServer(("127.0.0.1", 0), CloseAfterOne)
            srv2.daemon_threads = True
            threading.Thread(target=srv2.serve_forever, daemon=True).start()
            try:
                ep2 = fake_endpoint(f"http://127.0.0.1:{srv2.server_address[1]}")
                p2 = core.Pool()
                body = core.make_body("responses", "delay-0.5", "x")
                rs = [core.stream_request(p2, ep2, body, 20) for _ in range(3)]
                p2.close()
                retried = sum(1 for r in rs if r.get("conn_retry"))
                last = "服务端每请求后关连接：3 次请求 ok=" + str([r["ok"] for r in rs]) + "，其中 " + str(retried) + \
                       " 次标了 conn_retry（复用连接失败→换新连接重发，不计入失败率）"
                if all(r["ok"] for r in rs) and retried >= 1:
                    return True, last
                last += "；错误：" + "；".join(r["error"] for r in rs if r["error"])[:120]
            finally:
                srv2.shutdown()
                srv2.server_close()
        return False, last
    check("连接 keep-alive 竞态：换连接重发而非记失败", conn_retry)

    def quiz(api):
        def f():
            e = fake_endpoint(base, api)
            good = [core.run_task(pool, e, "quiz-good", t, 20) for t in core.build_tasks(["quiz"])]
            bad = [core.run_task(pool, e, "quiz-bad", t, 20) for t in core.build_tasks(["quiz"])]
            g, b = sum(1 for r in good if r["pass"]), sum(1 for r in bad if r["pass"])
            return g == len(good) == 10 and b == 0, f"标准答案判对 {g}/10，错误答案判对 {b}/10"
        return f
    check("判分器：题库（responses）", quiz("responses"))
    check("判分器：题库（chat）", quiz("chat"))

    def needle():
        t = core.build_tasks(["needle10k"])[0]
        g, b = core.run_task(pool, ep, "needle-good", t, 20), core.run_task(pool, ep, "quiz-bad", t, 20)
        return g["pass"] and not b["pass"], f"藏暗号的背景长度 {len(t['system'])} 字符；正确答案 pass={g['pass']}，错误答案 pass={b['pass']}"
    check("判分器：大海捞针", needle)

    def fingerprint():
        tasks = core.build_tasks(["fingerprint"])
        recs = []
        for m in ("fp-flat", "fp-flat2", "fp-peak", "fp-few"):
            recs += [core.run_task(pool, ep, m, t, 20) for _ in range(2) for t in tasks]
        fp = core.fp_analysis(recs)
        v = {(p["a"], p["b"]): p["verdict"] for p in fp["pairs"]}
        got = {
            "同分布": v.get(("fake/fp-flat", "fake/fp-flat2")),
            "异分布": v.get(("fake/fp-flat", "fake/fp-peak")),
            "样本不足": v.get(("fake/fp-flat", "fake/fp-few")),
        }
        ok = got == {"同分布": "指纹一致", "异分布": "指纹不一致", "样本不足": "样本不足（不作判断）"}
        return ok, f"同分布→{got['同分布']}，异分布→{got['异分布']}，少样本→{got['样本不足']}（有效数字 {fp['labels']['fake/fp-few']['n_numbers']} 个）"
    check("模型指纹：数值分布距离与拒判", fingerprint)

    def tool(api):
        def f():
            r = core.run_task(pool, fake_endpoint(base, api), "tool-ok", core.build_tasks(["tool"])[0], 20)
            return r["pass"], f"pass={r['pass']} {r.get('grade_note', '')}（步骤：{[s['status'] for s in r.get('steps', [])]}）"
        return f
    check("工具调用两步 + reasoning 回放（responses）", tool("responses"))
    check("工具调用两步（chat）", tool("chat"))

    def stats():
        w, p, u = core.wilson(5, 10), core.percentile([1, 2, 3, 4, 100], 0.95), core.mwu_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10])
        return (abs(w[0] - 0.237) < 0.002 and abs(w[1] - 0.763) < 0.002 and abs(p - 80.8) < 0.01 and abs(u - 0.0122) < 0.0005,
                f"wilson(5,10)={w}（应约 [0.237, 0.763]），p95={p}（应 80.8），Mann-Whitney p={u}（应约 0.0122）")
    check("统计：Wilson 区间、p95、显著性检验", stats)

    def effort():
        bad = []
        for api in ("responses", "chat"):
            e = fake_endpoint(base, api)
            with_, without = (core.stream_request(pool, e, core.make_body(api, "echo-effort", "x", effort=v), 20)["text"] for v in ("medium", ""))
            if (with_, without) != ("medium", "none"):
                bad.append(f"{api}: 传 medium 得 {with_!r}，不传得 {without!r}")
        return not bad, "responses 传 reasoning.effort、chat 传 reasoning_effort，不选则不传" if not bad else "；".join(bad)
    check("推理强度参数传递", effort)

    def buffered():
        a, b = req("delay-3.2"), req("delay-0.5")
        return a["buffered"] is True and b["buffered"] is False, f"延迟 3.2s 后一次性返回 → buffered={a['buffered']}（应 True）；正常流式 0.5s → buffered={b['buffered']}（应 False）"
    check("识别非流式（攒满后一次返回）", buffered)

    def speed():
        r = core.stream_request(pool, ep, core.make_body("responses", "slow-gen", "x"), 20)
        return r["tps"] is not None and 106 <= r["tps"] <= 144, f"50 tokens、首字后 0.4s 吐完，应约 125 tokens/s，测得 {r['tps']}"
    check("输出速度计算（tokens/秒）", speed)

    def speed_visible_only():
        out = []
        for api in ("responses", "chat"):
            r = core.stream_request(pool, fake_endpoint(base, api), core.make_body(api, "reason-gen", "x"), 20)
            out.append((api, r["tps"], r["reasoning_tokens"]))
        ok = all(t is not None and 80 <= t <= 125 and rt == 60 for _, t, rt in out)
        return ok, f"100 个输出 tokens 中 60 个是推理，可见 40 个、首字后 0.4s 吐完 → 应约 100 tokens/s（若把推理也算进去会是约 250）；测得 {out}"
    check("输出速度只算可见回答（不含推理 tokens）", speed_visible_only)

    def order():
        eps = {"x": dict(ep, name="a"), "y": dict(ep, name="b")}
        tasks = core.build_tasks(["quiz"])[:4]
        plan = {"endpoints": eps, "models": [("x", "quiz-good"), ("y", "quiz-good")], "tasks": tasks, "rounds": 2, "timeout": 20, "gap": 0, "warmup": False, "seed": 7}
        runs = []
        for _ in range(2):
            recs, _ = core.run_plan(plan, lambda u: None, threading.Event())
            runs.append([(r["round"], r["label"], r["task"]) for r in recs])
        each = all(sorted((r[1], r[2]) for r in runs[0] if r[0] == rd) == sorted((e, t["id"]) for e in ("a/quiz-good", "b/quiz-good") for t in tasks) for rd in (1, 2))
        blocks = [r[1] for r in runs[0][:8]]
        interleaved = blocks != sorted(blocks)
        return each and runs[0] == runs[1] and interleaved, f"每轮每个（模型,题）恰好一次={each}；同种子顺序可复现={runs[0] == runs[1]}；模型交替={interleaved}"
    check("交替顺序：每轮齐全、随机打乱、种子可复现", order)
    pool.close()
    srv.shutdown()
    srv.server_close()
    return out


def main():
    res = run_checks()
    for r in res:
        print(("通过 " if r["ok"] else "失败 ") + r["name"] + " —— " + r["detail"])
    raise SystemExit(0 if all(r["ok"] for r in res) else 1)
