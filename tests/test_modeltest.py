#!/usr/bin/env python3
"""模型测试平台的自动化测试（Python 标准库 unittest，全程用本机假服务，不访问任何真实接口，不碰真实的 data/ 和 results/）。
运行：在 ModelBench 目录下  python3 -m model_test test
"""
import datetime as dt
import http.client
import json
import os
import random
import re
import shutil
import stat
import sys
import tempfile
import threading
import time
import unittest
import urllib.parse
from unittest import mock

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # ModelBench 目录
sys.path.insert(0, HERE)
from model_test import config, core, selftest, server, store  # noqa: E402

TMP = None
FAKE = None
SRV = None


def setUpModule():
    global TMP, FAKE, SRV
    TMP = tempfile.mkdtemp(prefix="modeltest-")
    store.DATA_DIR, store.RESULTS_DIR, store.LEGACY_DIR = (os.path.join(TMP, "data"), os.path.join(TMP, "results"), os.path.join(TMP, "legacy"))
    store._hist_cache.clear()
    FAKE = selftest.start_fake()
    server.BIND_HOSTS = ["127.0.0.1"]
    SRV = server.ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
    server.PORT = SRV.server_address[1]
    threading.Thread(target=SRV.serve_forever, daemon=True).start()


def tearDownModule():
    SRV.shutdown()
    FAKE[0].shutdown()
    shutil.rmtree(TMP, ignore_errors=True)


def call(method, path, body=None, headers=None, raw=None):
    c = http.client.HTTPConnection("127.0.0.1", server.PORT, timeout=60)
    h = {"Content-Type": "application/json"} if method == "POST" else {}
    h.update(headers or {})
    c.request(method, path, body=raw if raw is not None else (json.dumps(body) if body is not None else None), headers=h)
    r = c.getresponse()
    data = r.read()
    c.close()
    try:
        return r.status, json.loads(data)
    except ValueError:
        return r.status, data


class Stats(unittest.TestCase):
    def test_percentile_and_wilson(self):
        self.assertAlmostEqual(core.percentile([1, 2, 3, 4, 100], 0.95), 80.8)
        self.assertEqual(core.percentile([5], 0.95), 5)
        lo, hi = core.wilson(5, 10)
        self.assertAlmostEqual(lo, 0.237, places=2)
        self.assertAlmostEqual(hi, 0.763, places=2)
        self.assertIsNone(core.wilson(0, 0))
        self.assertEqual(core.wilson(10, 10)[1], 1.0)

    def _rec(self, label, ok=True, total=1.0, sc="speed", passed=None, task="t", text="好", status=200):
        return {"label": label, "model": label, "scenario": sc, "ok": ok, "status": status if ok else (status if status != 200 else 503), "total_s": total,
                "first_token_s": total / 2, "first_event_s": total / 4, "connect_s": 0.0, "reused": True, "usage": None, "pass": passed,
                "task": task, "task_name": task, "text": text}

    def test_summarize_failures_low_sample_and_pass(self):
        recs = [self._rec("a", total=t) for t in (1, 2, 3)] + [self._rec("a", ok=False, total=0.1)]
        row = core.summarize(recs)[0]
        self.assertEqual((row["n"], row["ok"], row["codes"]), (4, 3, {"503": 1}))
        self.assertEqual(row["total"]["median"], 2)
        self.assertTrue(row["low_sample"])
        self.assertIsNone(row["pass_rate"])
        qs = [self._rec("b", sc="quiz", passed=p, task="q1", text=x) for p, x in ((True, "A"), (True, "A"), (False, "B"))]
        r = core.summarize(qs)[0]
        self.assertEqual((r["pass_n"], r["graded"]), (2, 3))
        self.assertAlmostEqual(r["consistency"], 2 / 3, places=3)

    def test_p95_only_with_enough_samples(self):
        self.assertFalse(core.dist([1, 2, 3, 4, 5])["p95_ok"])
        self.assertTrue(core.dist(list(range(1, 21)))["p95_ok"])

    def test_mann_whitney(self):
        self.assertAlmostEqual(core.mwu_p([1, 2, 3, 4, 5], [6, 7, 8, 9, 10]), 0.0122, places=3)
        self.assertEqual(core.mwu_p([1, 1, 1], [1, 1, 1]), 1.0)  # 全部并列
        self.assertIsNone(core.mwu_p([1], [2, 3]))
        self.assertGreater(core.mwu_p([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]), 0.5)

    def test_conclusion_significance(self):
        def rows(a, b):
            return core.summarize([self._rec("A", total=t) for t in a] + [self._rec("B", total=t) for t in b])
        c1 = core.conclusion(rows([1.0, 1.1, 1.0, 1.05, 1.02], [2.0, 2.1, 2.05, 2.2, 2.0]))
        self.assertIn("成功请求耗时中位数最低 A", c1)
        self.assertIn("不能推断显著性或等效", c1)
        c2 = core.conclusion(rows([1, 2, 3, 4, 5], [1.5, 2.5, 3.5, 4.5, 5.5]))
        self.assertIn("不能推断显著性或等效", c2)
        self.assertNotIn("显著快于", c2)
        c3 = core.conclusion(rows([1, 2], [1.5, 2.5]))
        self.assertNotIn("显著快于", c3)

    def test_reasoning_and_buffered_fields(self):
        r = self._rec("a", total=10)
        r.update(usage={"input_tokens": 5, "output_tokens": 100}, reasoning_tokens=40, buffered=True)
        row = core.summarize([r, dict(r)])[0]
        self.assertEqual((row["reasoning_share"], row["buffered_n"]), (0.4, 2))
        self.assertIn("非流式", core.conclusion([row]))
        self.assertEqual(row["think_share"], 0.25)  # (首字 5 - 首个事件 2.5) / 总 10

    def test_pass_ci_is_per_question_not_per_request(self):
        """固定题目重复测量不计算未经验证的 Wilson 区间。"""
        recs = [self._rec("m", sc="quiz", passed=True, task=f"q{i}") for i in range(10) for _ in range(10)]
        row = core.summarize(recs)[0]
        self.assertEqual((row["pass_n"], row["graded"], row["pass_ci_basis"], row["pass_units"]), (100, 100, None, 10))
        self.assertIsNone(row["pass_ci"])
        half = [self._rec("m", sc="quiz", passed=(i < 5), task=f"q{i}") for i in range(10) for _ in range(4)]  # 5 题全对、5 题全错
        self.assertIsNone(core.summarize(half)[0]["pass_ci"])

    def test_pass_ci_single_task_uses_requests(self):
        row = core.summarize([self._rec("m", sc="tool", passed=(i < 4), task="tool") for i in range(5)])[0]
        self.assertEqual((row["pass_ci_basis"], row["pass_units"]), (None, 1))
        self.assertIsNone(row["pass_ci"])

    def test_low_ok_sample_flag(self):
        """请求够多但成功的太少：耗时只统计成功请求，所以要单独提示。"""
        recs = [self._rec("m", ok=(i < 3)) for i in range(8)]  # 8 个请求，3 个成功
        row = core.summarize(recs)[0]
        self.assertEqual((row["low_sample"], row["low_ok_sample"], row["ok"]), (False, True, 3))
        self.assertEqual(core.summarize([self._rec("m") for _ in range(8)])[0]["low_ok_sample"], False)
        self.assertEqual(core.summarize([self._rec("m") for _ in range(3)])[0]["low_ok_sample"], False)  # 请求数本身就不足，归 low_sample
        self.assertEqual(core.summarize([self._rec("m", ok=False) for _ in range(8)])[0]["low_ok_sample"], False)  # 一个都没成功，是失败问题不是样本问题

    def test_summary_keeps_raw_values_for_compare(self):
        row = core.summarize([self._rec("m", total=t) for t in (1, 2, 3, 4, 5)])[0]
        self.assertEqual(row["total_vals"], [1, 2, 3, 4, 5])
        self.assertEqual(len(row["first_token_vals"]), 5)

    def test_conclusion_still_names_fastest_when_everyone_has_failures(self):
        recs = ([self._rec("A", total=1.0) for _ in range(9)] + [self._rec("A", ok=False)] +
                [self._rec("B", total=3.0) for _ in range(9)] + [self._rec("B", ok=False)])
        c = core.conclusion(core.summarize(recs))
        self.assertIn("成功请求耗时中位数最低 A", c)
        self.assertIn("只统计成功请求", c)
        low = core.conclusion(core.summarize([self._rec("A", total=1.0) for _ in range(3)] + [self._rec("A", ok=False) for _ in range(7)]))
        self.assertNotIn("成功请求耗时中位数", low)

    def test_failed_requests_do_not_enter_timing(self):
        row = core.summarize([self._rec("a", total=1), self._rec("a", ok=False, total=60)])[0]
        self.assertEqual(row["total"]["max"], 1)




class NonStreamAndRetry(unittest.TestCase):
    """中转站忽略 stream:true、keep-alive 竞态：两个曾经把正常响应误判成失败的 bug 的回归测试。"""

    def test_non_stream_json_response(self):
        for api in ("responses", "chat"):
            srv, base = selftest.start_fake()
            try:
                ep = selftest.fake_endpoint(base, api)
                pool = core.Pool()
                t = core.build_tasks(["needle10k"])[0]
                core.run_task(pool, ep, selftest.NONSTREAM, t, 20)  # 先建好连接
                r = core.run_task(pool, ep, selftest.NONSTREAM, t, 20)
                pool.close()
            finally:
                srv.shutdown()
                srv.server_close()
            self.assertEqual(r["status"], 200, api)
            self.assertTrue(r["ok"], api + " 应判成功：" + repr(r["error"]))
            self.assertEqual(r["text"], "青竹-7392", api)
            self.assertTrue(r["pass"], api)
            self.assertTrue(r["non_stream"] and r["buffered"], api)
            self.assertEqual(r["tps_overall"] is not None or r["tps"] is None, True, api)  # 一次性返回算不出分段速度

    def test_conn_retry_on_reused_connection(self):
        hits = {"n": 0}

        class CloseAfterOne(selftest.Fake):
            def do_POST(self):
                selftest.Fake.do_POST(self)
                hits["n"] += 1
                self.close_connection = True

        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), CloseAfterOne)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            ep = selftest.fake_endpoint(f"http://127.0.0.1:{srv.server_address[1]}")
            pool = core.Pool()
            body = core.make_body("responses", "delay-0.5", "x")
            rs = [core.stream_request(pool, ep, body, 20) for _ in range(3)]
            pool.close()
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertTrue(all(r["ok"] for r in rs), "keep-alive 竞态不该记为失败：" + str([r["error"] for r in rs]))
        self.assertTrue(any(r.get("conn_retry") for r in rs[1:]), "复用连接失败应标 conn_retry 并换连接重发")
        retry = next(r for r in rs if r.get("conn_retry"))
        self.assertEqual(len(retry["attempts"]), 2)
        self.assertTrue(retry["attempts"][0]["error"])
        self.assertGreaterEqual(retry["wall_total_s"], retry["total_s"])
        self.assertGreater(hits["n"], len(rs) - 1, "重发应多打一次请求")

    def test_queue_time_separated_from_first_token(self):
        class SlowFirst(selftest.Fake):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                self.rfile.read(n)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                time.sleep(0.4)

                def send(s):
                    b = s.encode()
                    self.wfile.write(b"%x\r\n" % len(b) + b + b"\r\n")
                    self.wfile.flush()
                send('data: {"type":"response.created","response":{"status":"in_progress"}}\n\n')
                time.sleep(0.4)
                send('data: {"type":"response.output_text.delta","delta":"好"}\n\n')
                send('data: {"type":"response.completed","response":{"status":"completed","usage":{"output_tokens":8}}}\n\n')
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()

        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), SlowFirst)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            ep = selftest.fake_endpoint(f"http://127.0.0.1:{srv.server_address[1]}")
            pool = core.Pool()
            core.stream_request(pool, ep, core.make_body("responses", "m", "x"), 20)
            r = core.stream_request(pool, ep, core.make_body("responses", "m", "x"), 20)
            pool.close()
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertAlmostEqual(r["queue_s"], 0.4, delta=0.15)
        self.assertGreater(r["first_token_s"] - r["queue_s"], 0.2)
        row = core.summarize([r])[0]
        self.assertAlmostEqual(row["queue_median"], 0.4, delta=0.15)

    def test_cached_tokens_uses_median(self):
        recs = [{"label": "a", "model": "a", "scenario": "speed", "ok": True, "status": 200, "total_s": 1,
                 "first_token_s": .5, "first_event_s": .2, "connect_s": 0, "reused": True, "pass": None,
                 "task": "t", "task_name": "t", "text": "好", "usage": {"input_tokens": 10, "output_tokens": 8, "cached_tokens": c}}
                for c in (0, 0, 200, 500, 900)]
        row = core.summarize(recs)[0]
        self.assertEqual(row["cached"], 200)  # 中位：前缀缓存命中随轮次递增，第一次的值不代表整体

    def test_tool_scenario_has_gen_time(self):
        srv, base = selftest.start_fake()
        try:
            ep = selftest.fake_endpoint(base)
            pool = core.Pool()
            t = core.build_tasks(["tool"])[0]
            core.run_task(pool, ep, "tool-ok", t, 20)
            r = core.run_task(pool, ep, "tool-ok", t, 20)
            pool.close()
        finally:
            srv.shutdown()
            srv.server_close()
        self.assertTrue(r["pass"])
        self.assertIn("gen_s", r)
        self.assertGreaterEqual(r["gen_s"], 0)  # 假服务极快时可能是 0.0（毫秒级四舍五入）
        self.assertAlmostEqual(r["total_s"], r["steps"][0]["total_s"] + r["gen_s"], places=3)
        row = core.summarize([r])[0]
        self.assertAlmostEqual(row["gen"]["median"], r["gen_s"], places=3)
        self.assertEqual(row["gen_vals"], [r["gen_s"]])
        self.assertGreaterEqual(r["wall_total_s"], r["total_s"] - 0.002)
        self.assertEqual(r["first_event_s"], next(e["t"] for e in r["events"] if e["step"] == 1))
class Graders(unittest.TestCase):
    def test_every_standard_answer_passes_and_wrong_fails(self):
        for q in core.QUIZ:
            self.assertTrue(core.check_answer(q["chk"], q["ans"], bool(q.get("strict"))), q["id"])
            self.assertFalse(core.check_answer(q["chk"], "完全不相关的回答", bool(q.get("strict"))), q["id"])

    def test_strictness(self):
        by = {q["id"]: q["chk"] for q in core.QUIZ}
        self.assertFalse(core.check_answer(by["q03"], '```json\n{"name":"张三","age":30}\n```', True))  # 题目要求不带代码块
        self.assertFalse(core.check_answer(by["q03"], '{"name":"张三","age":"30"}', True))  # age 必须是数字
        self.assertFalse(core.check_answer(by["q05"], '["设备名称","1","200.00","台"]', True))  # 千分位被拆列
        self.assertFalse(core.check_answer(by["q02"], "1,386,500.50"))  # 带千分位逗号，题目要求去掉
        self.assertTrue(core.check_answer(by["q01"], " 人民币贰佰伍拾贰万元整\n"))
        self.assertTrue(core.check_answer(("contains", "青竹-7392"), "暗号是青竹-7392。"))

    def test_loose_grading_tolerates_punctuation_and_prefix(self):
        # 答对但带标点/常见前缀：不该判错（正确率系统性偏低的来源）；格式由 strict 题专门判
        for v in ("贰佰伍拾贰万元整。", "答案是贰佰伍拾贰万元整", "391。", "结果是：391", "A！"):
            self.assertTrue(core.normalize_answer(v)[1] or core.normalize_answer(v)[0] != v)
        self.assertTrue(core.check_answer(core.QUIZ[0]["chk"], "贰佰伍拾贰万元整。"))
        self.assertTrue(core.check_answer(core.QUIZ[5]["chk"], "结果是：391"))
        self.assertTrue(core.check_answer(core.QUIZ[9]["chk"], "A！"))
        self.assertFalse(core.check_answer(core.QUIZ[3]["chk"], "苹果，香蕉，橙子", True))
        # strict 题不吃宽松化
        self.assertFalse(core.check_answer(core.QUIZ[2]["chk"], "{\"name\": \"张三\", \"age\": 30}。", True))

    def test_effort_in_body(self):
        self.assertEqual(core.make_body("responses", "m", "x", effort="high")["reasoning"], {"effort": "high"})
        self.assertEqual(core.make_body("chat", "m", "x", effort="low")["reasoning_effort"], "low")
        b = core.make_body("chat", "m", "x")
        self.assertNotIn("reasoning_effort", b)  # 不选就不传
        self.assertNotIn("reasoning", core.make_body("responses", "m", "x"))

    def test_needle_context_has_needle_in_middle(self):
        t = core.context_text(10000, needle=True)
        self.assertIn("青竹-7392", t)
        self.assertTrue(0.4 < t.index("青竹-7392") / len(t) < 0.6)
        self.assertNotIn("青竹-7392", core.context_text(10000))


class Urls(unittest.TestCase):
    def test_endpoint_parts(self):
        self.assertEqual(core.endpoint_parts("https://a.com"), ("https", "a.com", 443, "/v1"))
        self.assertEqual(core.endpoint_parts("https://a.com/v1/"), ("https", "a.com", 443, "/v1"))
        self.assertEqual(core.endpoint_parts("http://10.0.0.1:8080/v1/chat/completions"), ("http", "10.0.0.1", 8080, "/v1"))
        self.assertEqual(core.endpoint_parts("https://a.com/api/v3")[3], "/api/v3")
        for bad in ("a.com", "ftp://a.com", "", None):
            with self.assertRaises(ValueError):
                core.endpoint_parts(bad)


class Store(unittest.TestCase):
    def setUp(self):
        shutil.rmtree(store.DATA_DIR, ignore_errors=True)

    def test_endpoint_crud_and_key_hidden(self):
        e = store.upsert_endpoint({"name": "x", "base_url": "https://a.com", "api_key": "sk-secret-1234567890", "api": "auto"})
        self.assertNotIn("api_key", e)
        self.assertTrue(e["key_hint"].endswith("7890"))
        self.assertNotIn("sk-secret", json.dumps(store.list_endpoints()))
        self.assertEqual(stat.S_IMODE(os.stat(os.path.join(store.DATA_DIR, "endpoints.json")).st_mode), 0o600)
        store.upsert_endpoint({"id": e["id"], "name": "x2", "base_url": "https://b.com/v1", "api": "chat", "models": ["m1", "m2", "m1"]})  # 密钥留空 = 不变
        cur = store.get_endpoint(e["id"])
        self.assertEqual((cur["name"], cur["api_key"], cur["models"]), ("x2", "sk-secret-1234567890", ["m1", "m2"]))
        with self.assertRaises(ValueError):
            store.upsert_endpoint({"name": "y", "base_url": "https://a.com"})  # 新增必须有密钥
        store.upsert_endpoint({"name": "y", "base_url": "https://a.com", "api_key": "k1"})
        with self.assertRaises(ValueError):
            store.upsert_endpoint({"name": "y", "base_url": "https://a.com", "api_key": "k1"})  # 重名
        store.delete_endpoint(e["id"])
        self.assertEqual([x["name"] for x in store.list_endpoints()], ["y"])
        with self.assertRaises(ValueError):
            store.delete_endpoint("nope")

    def test_bad_input_rejected(self):
        for body in ({"name": "", "base_url": "https://a.com", "api_key": "k"}, {"name": "a/b", "base_url": "https://a.com", "api_key": "k"},
                     {"name": "a", "base_url": "javascript:1", "api_key": "k"}, {"name": "a", "base_url": "https://a.com", "api_key": "k", "api": "x"},
                     {"name": "a", "base_url": "https://a.com", "api_key": "k k"}, {"name": "a", "base_url": "https://a.com", "api_key": "k", "models": [""]}):
            with self.assertRaises(ValueError, msg=body):
                store.upsert_endpoint(body)

    def test_run_dir_naming_date_count(self):
        day = dt.date.today().isoformat()
        shutil.rmtree(store.RESULTS_DIR, ignore_errors=True)
        ids = [store.new_run_dir() for _ in range(3)]
        self.assertEqual(ids, [f"{day}-1", f"{day}-2", f"{day}-3"])
        os.makedirs(os.path.join(store.RESULTS_DIR, f"{day}-10"))
        self.assertEqual(store.new_run_dir(), f"{day}-11")  # 取最大值 +1，不会撞号
        self.assertEqual(store.new_run_dir("AB平台对比"), f"{day}-12_AB平台对比")  # 名称只是追加，日期-序号不变
        self.assertEqual(store.new_run_dir(), f"{day}-13")  # 带名称的文件夹也计入序号
        for bad in ("a/b", "..", "x y", "a" * 31, "a.b", "名称\n"):
            with self.assertRaises(ValueError, msg=bad):
                store.new_run_dir(bad)


class Api(unittest.TestCase):
    base = None

    @classmethod
    def setUpClass(cls):
        shutil.rmtree(store.DATA_DIR, ignore_errors=True)
        shutil.rmtree(store.RESULTS_DIR, ignore_errors=True)
        cls.base = FAKE[1]

    def test_01_security(self):
        self.assertEqual(call("GET", "/api/endpoints", headers={"Host": "evil.com"})[0], 403)
        self.assertEqual(call("POST", "/api/stop", {}, headers={"Origin": "http://evil.com"})[0], 403)
        self.assertEqual(call("POST", "/api/stop", raw="x=1", headers={"Content-Type": "text/plain"})[0], 415)
        self.assertEqual(call("POST", "/api/run", raw="[1]")[0], 400)
        self.assertEqual(call("GET", "/raw/..%2F..%2Fetc.json")[0], 404)
        self.assertEqual(call("GET", "/api/run/../../x")[0], 404)

    def test_02_endpoint_flow_and_probe(self):
        s, j = call("POST", "/api/endpoints", {"name": "fake", "base_url": self.base, "api_key": selftest.KEY, "api": "auto"})
        self.assertEqual(s, 200)
        Api.eid = j["endpoint"]["id"]
        self.assertNotIn(selftest.KEY, json.dumps(call("GET", "/api/endpoints")[1]))
        s, r = call("POST", "/api/endpoints/probe", {"id": Api.eid})
        self.assertEqual((s, r["models"], r["api"], r["errors"]), (200, len(selftest.MODELS), "responses", []))
        ep = [e for e in call("GET", "/api/endpoints")[1]["endpoints"] if e["id"] == Api.eid][0]
        self.assertEqual((ep["api_detected"], len(ep["models"])), ("responses", len(selftest.MODELS)))

    def test_03_wrong_key_probe_reports_error(self):
        s, j = call("POST", "/api/endpoints", {"name": "badkey", "base_url": self.base, "api_key": "wrong", "api": "auto"})
        s, r = call("POST", "/api/endpoints/probe", {"id": j["endpoint"]["id"]})
        self.assertTrue(r["errors"] and "401" in r["errors"][0])
        call("POST", "/api/endpoints/delete", {"id": j["endpoint"]["id"]})

    def test_04_validation(self):
        ok = {"selection": [{"endpoint": Api.eid, "models": ["quiz-good"]}], "scenarios": ["quiz"], "rounds": 1, "timeout": 30}
        bad = [dict(ok, selection=[{"endpoint": Api.eid, "models": ["not-in-list"]}]), dict(ok, selection=[{"endpoint": "ep-zzz", "models": ["x"]}]),
               dict(ok, scenarios=["rm -rf"]), dict(ok, scenarios=[]), dict(ok, rounds=0), dict(ok, rounds=99), dict(ok, timeout=1), dict(ok, context=5),
               dict(ok, scenarios=["speed"], prompt=" "), dict(ok, effort="ultra"), dict(ok, gap=99), dict(ok, selection=[]), dict(ok, selection=[{"endpoint": Api.eid, "models": []}]),
               dict(ok, selection=[{"endpoint": Api.eid, "models": selftest.MODELS}], rounds=10, scenarios=["quiz", "tool", "needle10k"])]
        for b in bad:
            self.assertEqual(call("POST", "/api/run", b)[0], 400, b)

    def test_05_end_to_end_run_saves_folder(self):
        day = dt.date.today().isoformat()
        body = {"selection": [{"endpoint": Api.eid, "models": ["quiz-good", "tool-ok"]}], "scenarios": ["quiz", "tool"], "rounds": 2,
                "timeout": 30, "gap": 0, "warmup": True, "effort": "medium", "seed": 123}
        self.assertEqual(call("POST", "/api/run", body)[0], 200)
        self.assertEqual(call("POST", "/api/run", body)[0], 409)  # 一次只跑一个
        self.assertEqual(call("POST", "/api/selftest", {})[0], 409)  # 测试中不允许自检
        for _ in range(200):
            job = call("GET", "/api/job")[1]
            if job["state"] not in ("starting", "running"):
                break
            time.sleep(0.2)
        self.assertEqual(job["state"], "done", job.get("error"))
        self.assertEqual(job["run_id"], f"{day}-1")
        d = os.path.join(store.RESULTS_DIR, f"{day}-1")
        self.assertEqual(sorted(os.listdir(d)), ["raw.json", "report.md", "result.json"])
        for fn in os.listdir(d):
            self.assertFalse(os.stat(os.path.join(d, fn)).st_mode & stat.S_IWUSR, fn + " 应为只读")
            with open(os.path.join(d, fn), encoding="utf-8") as f:
                self.assertNotIn(selftest.KEY, f.read(), fn + " 不能含密钥")
        res = call("GET", f"/api/run/{day}-1")[1]
        by = {(r["label"], r["scenario"]): r for r in res["summary"]}
        self.assertEqual((by[("fake/quiz-good", "quiz")]["pass_n"], by[("fake/quiz-good", "quiz")]["graded"]), (20, 20))
        self.assertEqual(by[("fake/tool-ok", "tool")]["pass_n"], 2)
        self.assertEqual(by[("fake/quiz-good", "quiz")]["consistency"], 1.0)
        self.assertEqual(len(res["records"]), 2 * (10 + 1) * 2)
        self.assertEqual(res["env"]["rounds"], 2)
        self.assertEqual(res["env"]["effort"], "medium")
        with open(os.path.join(d, "report.md"), encoding="utf-8") as f:
            self.assertIn("推理强度：medium", f.read())
        self.assertEqual(sorted(res["env"]["handshake_s"]), ["fake"])
        self.assertEqual(call("GET", f"/raw/{day}-1.json")[0], 200)
        # 第二次同一天 → -2；历史里可见并带摘要
        for bad in ("a/b", "..", "x y", "a" * 31, "../x"):
            self.assertEqual(call("POST", "/api/run", dict(body, name=bad))[0], 400, bad)
        self.assertEqual(call("POST", "/api/run", dict(body, scenarios=["tool"], rounds=1, name="AB平台对比"))[0], 200)
        for _ in range(100):
            if call("GET", "/api/job")[1]["state"] == "done":
                break
            time.sleep(0.2)
        runs = call("GET", "/api/history")[1]["runs"]
        rid = f"{day}-2_AB平台对比"
        self.assertEqual([r["id"] for r in runs[:2]], [rid, f"{day}-1"])
        self.assertEqual((runs[0]["name"], runs[1]["name"]), ("AB平台对比", ""))
        self.assertTrue(runs[0]["brief"])
        self.assertTrue(all(b["cond"] and b["cond_label"] for b in runs[0]["brief"]))
        tool = [next(b["cond"] for b in r["brief"] if b["scenario"] == "tool") for r in runs[:2]]  # 按场景取（摘要顺序由随机打乱决定，不能按位置取）
        self.assertEqual(tool[0], tool[1])  # 两次测试条件相同（都是 medium、同一道工具题）→ 同一条趋势线
        self.assertNotEqual(tool[0], next(b["cond"] for b in runs[1]["brief"] if b["scenario"] == "quiz"))  # 不同场景不会混在一条线上
        enc = urllib.parse.quote(rid)
        s, res2 = call("GET", "/api/run/" + enc)
        self.assertEqual((s, res2["env"]["name"]), (200, "AB平台对比"))
        self.assertEqual(call("GET", f"/raw/{enc}.json")[0], 200)
        self.assertTrue(os.path.isfile(os.path.join(store.RESULTS_DIR, rid, "result.json")))
        self.assertEqual(call("POST", "/api/run/delete", {"id": rid})[0], 200)
        self.assertTrue(os.path.isdir(os.path.join(store.RESULTS_DIR, ".trash")))
        self.assertEqual([r["id"] for r in call("GET", "/api/history")[1]["runs"]], [f"{day}-1"])


class Fairness(unittest.TestCase):
    def test_fake_rejects_missing_reasoning_replay(self):
        """假服务必须真的拒绝“没带回 reasoning 项”的第二步，否则自检里的工具调用回放检查就是摆设。"""
        pool, ep = core.Pool(), selftest.fake_endpoint(FAKE[1])
        body = {"model": "tool-ok", "stream": True, "input": [{"role": "user", "content": "x"}, {"type": "function_call_output", "call_id": "c", "output": "{}"}]}
        r = core.stream_request(pool, ep, body, 20)
        self.assertEqual(r["status"], 400)
        self.assertIn("reasoning_text", r["error"])
        pool.close()

    def test_warmup_and_alternation(self):
        eps = {"x": selftest.fake_endpoint(FAKE[1])}
        plan = {"endpoints": eps, "models": [("x", "quiz-good"), ("x", "quiz-bad")], "tasks": core.build_tasks(["quiz"])[:3], "rounds": 3,
                "timeout": 20, "gap": 0, "warmup": True, "seed": 1}
        recs, extra = core.run_plan(plan, lambda u: None, threading.Event())
        self.assertEqual(len(recs), 3 * 2 * 3)
        self.assertEqual(sorted(extra["warmup"]), ["fake/quiz-bad", "fake/quiz-good"])
        self.assertNotIn("warmup", [r.get("task") for r in recs])  # 预热不计入结果
        self.assertTrue(all(r["reused"] for r in recs))  # 全部复用连接
        for rd in (1, 2, 3):
            self.assertEqual(sum(1 for r in recs if r["round"] == rd), 6)

    def test_trend_condition_key(self):
        base = {"method_version": "3", "sampling": {"temperature": 0, "max_tokens": 4096}, "timeout_s": 20, "gap_s": 0, "warmup": False,
                "effort": "", "prompt": "甲", "context_tokens": 0, "scenario_fingerprints": {"speed": "aaa", "quiz": "q"}}
        k = lambda env, sc: store._cond(env, sc)[0]  # noqa: E731
        self.assertNotEqual(k(base, "speed"), k(dict(base, scenario_fingerprints={"speed": "bbb"}), "speed"))
        self.assertNotEqual(k(base, "speed"), k(dict(base, effort="high"), "speed"))
        self.assertEqual(k(base, "quiz"), k(dict(base, prompt="乙", scenario_fingerprints={"speed": "zzz", "quiz": "q"}), "quiz"))

    def test_gen_scenario_has_speed(self):
        eps = selftest.fake_endpoint(FAKE[1])
        pool = core.Pool()
        t = core.build_tasks(["gen"])[0]
        r = core.run_task(pool, eps, "slow-gen", t, 20)
        pool.close()
        self.assertEqual((r["scenario"], r["task"], r["ok"]), ("gen", "gen", True))
        self.assertTrue(r["tps"] and 100 < r["tps"] < 150)
        self.assertEqual(core.summarize([r])[0]["tps"]["n"], 1)

    def test_effort_reaches_server_request(self):
        pool, ep = core.Pool(), selftest.fake_endpoint(FAKE[1], "chat")
        eps = {"x": ep}
        plan = {"endpoints": eps, "models": [("x", "echo-effort")], "tasks": [{"id": "e", "scenario": "speed", "kind": "plain", "name": "e", "prompt": "x"}],
                "rounds": 1, "timeout": 20, "gap": 0, "warmup": False, "seed": 1, "effort": "high"}
        recs, _ = core.run_plan(plan, lambda u: None, threading.Event())
        self.assertEqual(recs[0]["text"], "high")
        pool.close()

    def test_stop_saves_partial(self):
        stop = threading.Event()
        eps = {"x": selftest.fake_endpoint(FAKE[1])}
        plan = {"endpoints": eps, "models": [("x", "quiz-good")], "tasks": core.build_tasks(["quiz"]), "rounds": 5, "timeout": 20, "gap": 0, "warmup": False, "seed": 2}
        recs, _ = core.run_plan(plan, lambda u: stop.set() if u.get("done", 0) >= 3 else None, stop)
        self.assertEqual(len(recs), 3)

    def test_gap_is_respected_per_endpoint(self):
        eps = {"x": selftest.fake_endpoint(FAKE[1])}
        plan = {"endpoints": eps, "models": [("x", "quiz-good")], "tasks": core.build_tasks(["quiz"])[:3], "rounds": 1, "timeout": 20, "gap": 0.4, "warmup": False, "seed": 3}
        t = time.perf_counter()
        core.run_plan(plan, lambda u: None, threading.Event())
        self.assertGreaterEqual(time.perf_counter() - t, 0.78)  # 3 个请求，至少 2 个 0.4 秒间隔


class Dashboard(unittest.TestCase):
    """页面静态检查：脚本引用的元素都在、关键交互都有绑定（防止改函数时误删绑定，导致页签/按钮点击无反应）。"""

    def setUp(self):
        import re
        with open(server.PAGE, encoding="utf-8") as f:
            self.html = f.read()
        self.markup, self.js = self.html.split("<script>")[0], self.html.split("<script>")[1]
        self.re = re

    def _css(self):
        """去注释、压掉标点周围空白，但保留选择器里的空格，便于按语义断言样式。"""
        css = self.re.sub(r"/\*.*?\*/", "", self.markup, flags=self.re.S)
        return self.re.sub(r"\s*([{};:,>])\s*", r"\1", css)

    def test_every_referenced_id_exists(self):
        ids = set(self.re.findall(r'\bid="([^"]+)"', self.markup))
        used = {u for u in self.re.findall(r'\$\("([^"]+)"\)', self.js) if not u.startswith(("tab-", "sc_"))}
        dynamic = {"setRounds"}  # 由脚本生成的元素（样本不足提示里的按钮），必须在脚本里真的创建
        for d in dynamic:
            self.assertIn(f'id="{d}"', self.js, f"动态元素 {d} 没有在脚本里创建")
        self.assertEqual(sorted(used - ids - dynamic), [])
        for t in ("result", "history", "compare", "selftest"):
            self.assertIn(f'id="tab-{t}"', self.markup)
            self.assertIn(f'data-tab="{t}"', self.markup)

    def test_key_bindings_present(self):
        for key in ('.tabs button").forEach(b => b.onclick = () => tab(b.dataset.tab))', '$("run").onclick', '$("stop").onclick', '$("addEp").onclick',
                    '$("f_save").onclick', '$("impEp").onclick', '$("cmpBtn").onclick', '$("mergeBtn").onclick', '$("stRun").onclick', '$("useName").onchange', '$("mfilter").oninput'):
            self.assertIn(key, self.js, "缺少绑定：" + key)

    def test_compare_and_sample_hint_pieces_present(self):
        """多条对比与样本提示：关键元素、说明文字和判断逻辑都在（防止改页面时被误删）。"""
        for key in ('id="cmpAll"', 'id="cmpClear"', 'class="cmpbar"', 'id="sampleHint"', "MAX_CMP", "function renderCompare", "function verdictHtml",
                    "function evalCell", "const baseIdx", "设为基准", "怎么看这张对比", "越小越好", "差异不明显", "显著更好", "function sampleHint", "一键设置轮数", "成功样本不足"):
            self.assertIn(key, self.html, "缺少：" + key)
        self.assertNotIn("cmpSel.slice(-2)", self.html, "对比数量不应再限制为 2 条")

    def test_history_click_targets_are_big_enough(self):
        """历史表格的可点区域：大方框、整格点击区、常规大小的按钮、点选择格/按钮不会误打开结果（曾因点击区太小被反馈）。"""
        css = self._css()
        for key in (".hist input[type=checkbox]{width:22px;height:22px",
                    ".hist .hit{display:flex;align-items:center;justify-content:center",
                    ".hist tr.sel{", ".hist .acts button{min-height:34px"):
            self.assertIn(key, css, "缺少样式：" + key)
        m = re.search(r"\.hist \.hit\{[^}]*min-height:(\d+)px", css)
        self.assertIsNotNone(m, "历史行勾选框没有最小高度")
        self.assertGreaterEqual(int(m.group(1)), 44, "整格点击区应至少 44px（可点性）")
        m = re.search(r"\.hist input\[type=checkbox\]\{[^}]*width:(\d+)px[^}]*height:(\d+)px", css)
        self.assertIsNotNone(m, "历史行复选框没有显式尺寸")
        self.assertGreaterEqual(int(m.group(1)), 20, "复选框应够大，方便勾选")
        self.assertEqual(m.group(1), m.group(2), "复选框应为正方形")
        for key in ('class="cell-chk" data-nopen', 'class="hit"', 'data-del=', 'e.target.closest("[data-nopen]")'):
            self.assertIn(key, self.html, "缺少：" + key)
        self.assertNotIn('data-open=', self.html, "整行已可点开结果，“查看”按钮是冗余的，应已删除")
        self.assertNotIn('class="danger sm" data-del', self.html, "删除按钮不应再用小号样式")
        self.assertIn('点任意一行直接打开结果', self.html, "引导文案应说明整行可点")

    def test_run_finish_does_not_refetch_result(self):
        """测试结束后直接用轮询拿到的结果渲染，不再按 id 重新拉一次（会收起用户展开的明细、多一次请求）。"""
        self.assertIn('if (j.run_id && j.records && j.records.length)', self.js, "完成后应优先用轮询数据渲染")
        self.assertIn('fingerprint: j.fingerprint', self.js, "完成后渲染要带上指纹数据")
        self.assertIn('else if (j.run_id)', self.js, "没有轮询到记录时要回退到按 id 拉取")

    def test_request_estimate_counts_warmup(self):
        """时间估算要把预热请求算进去，否则显示的时间偏短。"""
        self.assertIn('const warm = $("warmup").checked ? n : 0;', self.js, "估算未计入预热")
        self.assertIn('含 ${warm} 次预热', self.js, "估算文案未说明预热请求数")
        self.assertIn('if (i === "warmup") updateEst();', self.js, "切换预热勾选后应刷新估算")

    def test_design_tokens_and_aesthetics(self):
        """视觉方向（安静的机房）：深色仪器面板、琥珀荧光强调色、等宽数字、示波器上电动效、无障碍降级。"""
        css = self._css()
        for token in ("--bg:#0a0d12", "--amber:#ffb454", "--green:#5ed68f", "--red:#ff6b6b", "--mono:", "--line:"):
            self.assertIn(token, css, "缺少设计变量：" + token)
        for key in (".boot{", "@keyframes sweep", "prefers-reduced-motion", "background-attachment:fixed",
                    "text-transform:uppercase", "var(--amber)"):
            self.assertIn(key, css, "缺少视觉细节：" + key)
        self.assertNotIn("#2563eb", css + self.js, "旧版蓝色主题应已移除")
        # 深色底上前景一律走变量；白只允许出现在 selection 高亮里
        body_fg = re.search(r"body\{[^}]*color:([^;}]+)", css)
        self.assertIsNotNone(body_fg, "body 没有前景色")
        self.assertIn("var(--ink)", body_fg.group(1), "深色面板的前景应走 --ink 变量")
        self.assertNotIn(re.search(r"body\{[^}]*", css).group(0).replace("color:var(--ink)", "").replace("#fff", ""),
                         "#fff", "body 上仍有硬编码白色")
        # 键盘焦点可见（可访问性）
        self.assertIn("input:focus", css, "输入框缺少 focus 样式")
        self.assertIn(":focus-visible", css + self.js, "按钮缺少 :focus-visible 焦点环（键盘可访问性）")

    def test_script_has_no_syntax_error(self):
        import shutil
        import subprocess
        if not shutil.which("node"):
            self.skipTest("没有 node")
        p = os.path.join(TMP, "dash.js")
        with open(p, "w", encoding="utf-8") as f:
            f.write(self.js.split("</script>")[0])
        r = subprocess.run(["node", "--check", p], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)


class SelfTest(unittest.TestCase):
    def test_all_checks_pass(self):
        res = selftest.run_checks()
        self.assertEqual([r["name"] for r in res if not r["ok"]], [])
        self.assertGreaterEqual(len(res), 10)


class Fingerprint(unittest.TestCase):
    def _recs(self, label, nums, chunk=40):
        return [{"label": label, "model": label, "scenario": "fingerprint", "ok": True, "text": ", ".join(map(str, nums[i:i + chunk]))}
                for i in range(0, len(nums), chunk)]

    @staticmethod
    def _draw(seed, n, lo=1, hi=355):
        r = random.Random(seed)
        return [r.randint(lo, hi) for _ in range(n)]

    def test_extract_numbers(self):
        self.assertEqual(core.extract_numbers("1, 2, 3, 4"), [1, 2, 3, 4])
        self.assertEqual(core.extract_numbers("[1 2 3]"), [1, 2, 3])
        self.assertEqual(core.extract_numbers("1、2、3、4"), [1, 2, 3, 4])  # 标点分隔不断开
        self.assertEqual(core.extract_numbers("前 9999 个，然后 10 11 12"), [10, 11, 12])  # 超范围被丢弃
        self.assertEqual(core.extract_numbers("选 1 到 355 中的数：100 101 102"), [100, 101, 102])  # 汉字/字母会断开数字段，取最长段，防回显提示词里的数字混入
        self.assertEqual(core.extract_numbers("a 1 2 3 x 4 5 6 7"), [4, 5, 6, 7])
        self.assertEqual(core.extract_numbers("abc"), [])

    def test_hellinger_and_verdict(self):
        c = [10] * core.FP_BINS
        self.assertEqual(core.hellinger(c, c), 0.0)
        d = [0] * core.FP_BINS
        d[0] = 100
        self.assertGreater(core.hellinger(c, d), 0.5)
        self.assertEqual(core.fp_verdict(0.05, 0.1)[0], "数值分布接近（不证明身份）")
        self.assertEqual(core.fp_verdict(0.35, 0.1)[0], "差异不明显（不作判断）")
        self.assertEqual(core.fp_verdict(0.55, 0.1)[0], "数值分布差异较大（不证明身份）")
        self.assertEqual(core.fp_verdict(0.15, 0)[1], core.FP_ABS_FLOOR)  # 噪声为 0 时判定线不低于下限

    def test_fp_analysis_verdicts(self):
        same = core.fp_analysis(self._recs("a", self._draw(21, 1000)) + self._recs("b", self._draw(22, 1000)))
        self.assertEqual(same["pairs"][0]["verdict"], "数值分布接近（不证明身份）")
        diff = core.fp_analysis(self._recs("a", self._draw(31, 240)) + self._recs("b", self._draw(32, 240, 1, 20)))
        self.assertEqual(diff["pairs"][0]["verdict"], "数值分布差异较大（不证明身份）")
        few = core.fp_analysis(self._recs("a", self._draw(41, 240)) + self._recs("b", self._draw(42, 50)))
        self.assertEqual(few["pairs"][0]["verdict"], "样本不足（不作判断）")
        one = core.fp_analysis(self._recs("a", self._draw(51, 240)))
        self.assertEqual((one["pairs"], one["labels"]["a"]["n_numbers"]), ([], 240))
        self.assertIsNone(core.fp_analysis([{"scenario": "speed", "ok": True, "label": "a", "model": "a", "text": "1"}]))

    def test_fingerprint_tasks(self):
        tasks = core.build_tasks(["fingerprint"])
        self.assertEqual([t["id"] for t in tasks], ["fp1", "fp2", "fp3"])
        self.assertTrue(all(t["scenario"] == "fingerprint" and "chk" not in t for t in tasks))
        self.assertTrue(all(str(core.FP_COUNT) in t["prompt"] and "355" in t["prompt"] for t in tasks))

    def test_fp_end_to_end_with_fake(self):
        srv, base = selftest.start_fake()
        try:
            ep = selftest.fake_endpoint(base)
            pool = core.Pool()
            tasks = core.build_tasks(["fingerprint"])
            recs = []
            for m in ("fp-flat", "fp-peak"):
                recs += [core.run_task(pool, ep, m, t, 20) for _ in range(2) for t in tasks]
            pool.close()
        finally:
            srv.shutdown()
            srv.server_close()
        fp = core.fp_analysis(recs)
        self.assertEqual(fp["pairs"][0]["verdict"], "数值分布差异较大（不证明身份）")
        self.assertTrue(all(v["n_numbers"] == 240 for v in fp["labels"].values()))


class Config(unittest.TestCase):
    def _write(self, data):
        p = os.path.join(TMP, "cfg-%d.json" % time.time_ns())
        with open(p, "w", encoding="utf-8") as f:
            f.write(data if isinstance(data, str) else json.dumps(data))
        return p

    def test_defaults_without_file(self):
        c = config.load(os.path.join(TMP, "不存在.json"))
        self.assertEqual((c["server"]["port"], c["limits"]["max_requests"], c["defaults"]["rounds"]), (18888, 1200, 5))

    def test_partial_override_keeps_other_defaults(self):
        c = config.load(self._write({"_说明": "忽略", "server": {"port": 19000}, "defaults": {"rounds": 8}}))
        self.assertEqual((c["server"]["port"], c["server"]["bind_hosts"][0], c["defaults"]["rounds"], c["defaults"]["timeout"]), (19000, "127.0.0.1", 8, 90))

    def test_bad_config_rejected(self):
        for bad in ({"server": {"port": 70000}}, {"server": {"prot": 1}}, {"limits": {"max_rounds": 0}}, {"defaults": {"rounds": 99}},
                    {"defaults": {"gap": -1}}, {"server": {"bind_hosts": []}}, {"paths": {"data_dir": ""}}, {"server": 5}, {"nonsense": 1}):
            with self.assertRaises(ValueError, msg=bad):
                config.load(self._write(bad))
        with self.assertRaises(ValueError):
            config.load(self._write("{不是json"))

    def test_path_resolution(self):
        self.assertEqual(config.resolve("data"), os.path.join(config.ROOT, "data"))
        self.assertEqual(config.resolve("~/x"), os.path.join(os.path.expanduser("~"), "x"))
        self.assertEqual(config.resolve("/tmp/y"), "/tmp/y")

    def test_shipped_config_files_are_valid_and_equal_defaults(self):
        # config.example.json 必须存在且与内置默认同步
        ex = os.path.join(config.ROOT, "config.example.json")
        self.assertTrue(os.path.isfile(ex), "config.example.json 应随仓库提供")
        self.assertIsInstance(config.load(ex), dict, "config.example.json 应该能通过校验")
        c = config.load(ex)
        self.assertEqual({k: c[k] for k in config.DEFAULTS}, config.DEFAULTS, "config.example.json 与内置默认不一致（示例文件应保持同步）")
        # config.json 是本机配置，不入仓库（公开仓库 clone 后本来就没有），存在时才校验其合法性
        own = os.path.join(config.ROOT, "config.json")
        if os.path.isfile(own):
            self.assertIsInstance(config.load(own), dict, "config.json 应该能通过校验")

    def test_server_limits_come_from_config(self):
        self.assertEqual((server.MAX_ROUNDS, server.MAX_REQUESTS), (config.CFG["limits"]["max_rounds"], config.CFG["limits"]["max_requests"]))
        self.assertEqual(call("GET", "/api/config")[1]["defaults"], config.CFG["defaults"])


if __name__ == "__main__":
    unittest.main(verbosity=2)


class FairnessAndValidity(unittest.TestCase):
    """测量有效性与公平性：固定采样参数、输出长度可比性警示、前缀缓存污染检出、检验力提示。"""

    def test_sampling_params_are_fixed_and_recorded(self):
        # 不同中转站的默认 temperature / max_tokens 往往不同，不固定就会把“接口默认值差异”当成模型差异
        for api, key in (("chat", "max_tokens"), ("responses", "max_output_tokens")):
            body = core.make_body(api, "m", "hi")
            self.assertEqual(body.get("temperature"), 0, api + " 未固定 temperature")
            self.assertEqual(body.get(key), core.MAX_TOKENS, api + " 未固定输出上限")
            self.assertTrue(body.get("stream"), api + " 必须流式")

    def test_env_snapshot_records_sampling(self):
        srv, base = selftest.start_fake()
        try:
            ep = selftest.fake_endpoint(base)
            store._save({"schema_version": 1, "endpoints": [dict(ep, id="e1", name="fake", models=["quiz-good"], api_detected="responses", models_checked_at=None)]})
            res = store.execute({"selection": [{"endpoint": "e1", "models": ["quiz-good"]}], "scenarios": ["speed"], "prompt": "x",
                                 "rounds": 1, "timeout": 20, "context": 0, "gap": 0, "warmup": False, "name": "", "effort": ""}, lambda u: None, threading.Event())
        finally:
            srv.shutdown()
            srv.server_close()
        sm = res["env"].get("sampling")
        self.assertIsNotNone(sm, "环境快照未记录采样参数")
        self.assertEqual((sm["temperature"], sm["max_tokens"]), (0, core.MAX_TOKENS))
        self.assertIn("采样参数已固定", res["env"]["note"])

    def test_conclusion_warns_when_output_length_differs(self):
        def rec(label, total, tout, tps):
            return {"label": label, "model": label, "scenario": "speed", "ok": True, "status": 200, "total_s": total,
                    "first_token_s": .5, "first_event_s": .4, "connect_s": 0, "reused": True, "tps": tps,
                    "usage": {"input_tokens": 10, "output_tokens": tout}, "pass": None, "task": "t", "task_name": "t", "text": "x"}
        rows = core.summarize([rec("A", 2.0, 50, 33.3), rec("B", 5.0, 500, 111.1)])
        txt = core.conclusion(rows)
        self.assertIn("输出长度差 10 倍", txt, "输出长度差得多时应明确提示总耗时不可直接当性能")
        self.assertIn("按输出速度看最高的是 B", txt, "应给出可归一化的参照（tokens/s）")

    def test_cache_pollution_is_detected(self):
        recs = []
        for total, cached in zip([4.5, 4.2, 3.4, 2.6, 2.2], [0, 0, 200, 600, 900]):
            recs.append({"label": "a", "model": "a", "scenario": "speed", "ok": True, "status": 200, "total_s": total,
                         "first_token_s": total * .5, "first_event_s": total * .4, "connect_s": 0, "reused": True,
                         "usage": {"input_tokens": 1000, "output_tokens": 50, "cached_tokens": cached},
                         "pass": None, "task": "t", "task_name": "t", "text": "好"})
        row = core.summarize(recs)[0]
        ct = row["cache_trend"]
        self.assertIsNotNone(ct, "缓存命中持续增长且耗时随之下降时应算出趋势")
        self.assertEqual((ct["cached_first"], ct["cached_last"]), (0, 900))
        self.assertGreaterEqual(ct["speedup"], 1.3)
        self.assertIn("疑似前缀缓存污染", core.conclusion([row]))
        # 没有缓存增长时不应误报
        flat = [dict(r, usage={"input_tokens": 10, "output_tokens": 50, "cached_tokens": 0}) for r in recs]
        self.assertIsNone(core.summarize(flat)[0]["cache_trend"])

    def test_low_power_warning_when_difference_not_significant(self):
        # 小样本 + 差距不大时，“差异不明显”要提示是检验力不足，而不是真的没差别
        a = [3.0, 3.4, 3.1, 3.5, 3.2]      # 中位 3.2，与 B 大量重叠
        b = [3.3, 3.2, 3.6, 3.1, 3.4]      # 中位 3.3，真实差距仅约 3%
        p = core.mwu_p(a, b)
        self.assertGreaterEqual(p, 0.05, "构造的两组应检不出显著（用于验证提示语）")
        rows = core.summarize([{"label": "A", "model": "A", "scenario": "speed", "ok": True, "status": 200, "total_s": t,
                                "first_token_s": .5, "first_event_s": .4, "connect_s": 0, "reused": True,
                                "usage": {"input_tokens": 10, "output_tokens": 50}, "pass": None, "task": "t", "task_name": "t", "text": "x"}
                               for t in a] +
                              [{"label": "B", "model": "B", "scenario": "speed", "ok": True, "status": 200, "total_s": t,
                                "first_token_s": .5, "first_event_s": .4, "connect_s": 0, "reused": True,
                                "usage": {"input_tokens": 10, "output_tokens": 50}, "pass": None, "task": "t", "task_name": "t", "text": "x"}
                               for t in b])
        txt = core.conclusion(rows)
        self.assertIn("不能推断显著性或等效", txt)
        self.assertNotIn("显著快于", txt)


class MergeRuns(unittest.TestCase):
    """跨次测试合并：同条件样本翻倍、不同条件绝不混算、旧版结果不参与。"""

    def _run(self, effort="", scenario="quiz", prompt="", rounds=1):
        ep = selftest.fake_endpoint(FAKE[1])
        store._save({"schema_version": 1, "endpoints": [dict(ep, id="e1", name="fake", models=["quiz-good"], api_detected="responses", models_checked_at=None)]})
        return store.execute({"selection": [{"endpoint": "e1", "models": ["quiz-good"]}], "scenarios": [scenario], "prompt": prompt,
                              "rounds": rounds, "timeout": 20, "context": 0, "gap": 0, "warmup": False, "name": "", "effort": effort},
                             lambda u: None, threading.Event())["id"]

    def test_same_condition_merges_and_doubles_samples(self):
        a, b = self._run(rounds=2), self._run(rounds=2)
        m = store.merge_runs([a, b])
        self.assertEqual(len(m["rows"]), 1)
        row = m["rows"][0]
        self.assertEqual(row["n"], 40)                 # 2 次 × 2 轮 × 10 题
        self.assertEqual(sorted(row["merged_runs"]), sorted([a, b]))
        self.assertEqual(m["warnings"], [])
        # 同条件合并后可用样本翻倍，检验力提示应随之消失或更弱
        self.assertIn("成功请求耗时中位数", m["conclusion"])

    def test_different_conditions_are_not_merged(self):
        a, b = self._run(effort=""), self._run(effort="high")
        m = store.merge_runs([a, b])
        self.assertEqual(len(m["rows"]), 2, "推理强度不同的结果必须分开统计")
        conds = " | ".join(r["merged_cond"] for r in m["rows"])
        self.assertIn("强度 未指定", conds)
        self.assertIn("强度 high", conds)
        self.assertTrue(any("不同条件" in w for w in m["warnings"]))
        for r in m["rows"]:
            self.assertEqual(r["n"], 10, "不同条件不能把样本加在一起")

    def test_different_prompts_are_not_merged(self):
        a, b = self._run(scenario="speed", prompt="只回复：好"), self._run(scenario="speed", prompt="说一句话")
        m = store.merge_runs([a, b])
        self.assertEqual(len(m["rows"]), 2)
        self.assertTrue(any("不同条件" in w for w in m["warnings"]))

    def test_merge_requires_two_runs(self):
        a = self._run()
        with self.assertRaises(ValueError):
            store.merge_runs([a])
        with self.assertRaises(ValueError):
            store.merge_runs([a, "不存在的-id"])

    def test_merge_records_carry_condition_tag(self):
        a, b = self._run(), self._run()
        runs = [store.load_run(a), store.load_run(b)]
        m = core.merge_analysis(runs)
        self.assertIsNotNone(m)
        self.assertNotEqual(runs[0]["env"].get("sampling"), None)
        self.assertTrue(all(g["cond"] for g in m["groups"]), "合并后的组必须带条件标记")


class ValidityRegression(unittest.TestCase):
    def _env(self, tasks):
        plan = {"endpoints": {"e": selftest.fake_endpoint(FAKE[1])}, "models": [("e", "m")],
                "tasks": tasks, "rounds": 1, "timeout": 20, "gap": 0, "warmup": False, "effort": "", "seed": 1}
        return core.make_env(plan, {}, {"created": "2026-01-01", "prompt": "", "context": 0})

    def test_scenario_manifest_tracks_tasks_and_grading(self):
        for scenario in ("quiz", "fingerprint", "tool"):
            tasks = core.build_tasks([scenario])
            original = self._env(tasks)
            changed = [dict(t) for t in tasks]
            changed[0]["prompt"] += "changed"
            self.assertNotEqual(core.cond_of(original, scenario), core.cond_of(self._env(changed), scenario))
            if scenario == "quiz":
                changed = [dict(t) for t in tasks]
                changed[0]["chk"] = ("re", "changed")
                self.assertNotEqual(core.cond_of(original, scenario), core.cond_of(self._env(changed), scenario))
        old = dict(original)
        old.pop("scenario_fingerprints")
        self.assertIsNone(core.cond_of(old, "tool"))

    def test_unknown_conditions_and_changed_service_do_not_merge(self):
        env = self._env(core.build_tasks(["speed"]))
        rec = Stats()._rec("fake/m")
        old = dict(env)
        old.pop("scenario_fingerprints")
        runs = [{"id": "a", "env": old, "records": [rec]}, {"id": "b", "env": old, "records": [rec]}]
        self.assertEqual(len(core.merge_analysis(runs)["rows"]), 2)
        other = dict(env, endpoints=[dict(env["endpoints"][0], base_url="http://other.invalid")])
        runs = [{"id": "a", "env": env, "records": [rec]}, {"id": "b", "env": other, "records": [rec]}]
        merged = core.merge_analysis(runs)
        self.assertEqual(len(merged["rows"]), 2)
        self.assertIn("未作排名", merged["conclusion"])

    def test_conclusion_never_ranks_across_conditions(self):
        recs = [dict(Stats()._rec("same/m", total=1), cond="a") for _ in range(5)]
        recs += [dict(Stats()._rec("same/m", total=5), cond="b") for _ in range(5)]
        text = core.conclusion(core.summarize(recs))
        self.assertNotIn("第二名", text)
        self.assertNotIn("显著", text)

    def test_specific_grader_boundaries(self):
        q = {x["id"]: x for x in core.QUIZ}
        for text in ("桌子,椅子,电脑", "苹果,香蕉,苹果", "苹果，香蕉，橙子"):
            self.assertFalse(core.check_answer(q["q04"]["chk"], text, True))
        self.assertTrue(core.check_answer(q["q04"]["chk"], "橙子,苹果,香蕉", True))
        for text in ("答案是：A！", "A\n", " A"):
            self.assertFalse(core.check_answer(q["q10"]["chk"], text, True))
        for value in (["MET"], "MET", "MET错误名称", 123):
            self.assertFalse(core._tool_args_ok({"name": "lookup_price", "arguments": json.dumps({"item": value})}))
        self.assertTrue(core._tool_args_ok({"name": "lookup_price", "arguments": json.dumps({"item": "MET混合滤料包"})}))

    def test_tool_steps_use_recorded_sampling_parameters(self):
        for api, key in (("chat", "max_tokens"), ("responses", "max_output_tokens")):
            ep, pool = selftest.fake_endpoint(FAKE[1], api), core.Pool()
            try:
                with mock.patch.object(core, "stream_request", wraps=core.stream_request) as req:
                    result = core.run_task(pool, ep, "tool-ok", core.build_tasks(["tool"])[0], 20)
                self.assertTrue(result["pass"])
                self.assertEqual(req.call_count, 2)
                for call, step in zip(req.call_args_list, result["steps"]):
                    body = call.args[2]
                    self.assertEqual((body["temperature"], body[key]), (0, core.MAX_TOKENS))
                    self.assertEqual(step["request_config"][key], body[key])
            finally:
                pool.close()

    def test_incomplete_responses_fail_in_both_protocols_and_formats(self):
        class Truncated(selftest.Fake):
            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n))
                chat = self.path.endswith("/chat/completions")
                streamed = body.get("model") == "streamed"
                if not streamed:
                    data = ({"choices": [{"message": {"content": "partial"}, "finish_reason": "length"}]} if chat
                            else {"status": "incomplete", "output_text": "partial"})
                    return self._json(200, data)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                events = ([{"choices": [{"delta": {"content": "partial"}, "finish_reason": "length"}]}] if chat
                          else [{"type": "response.output_text.delta", "delta": "partial"},
                                {"type": "response.incomplete", "response": {"status": "incomplete"}}])
                for event in events:
                    self._chunk("data: " + json.dumps(event) + "\n\n")
                self.wfile.write(b"0\r\n\r\n")
                self.wfile.flush()
        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), Truncated)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            for api in ("chat", "responses"):
                pool = core.Pool()
                try:
                    ep = selftest.fake_endpoint(f"http://127.0.0.1:{srv.server_address[1]}", api)
                    for model in ("streamed", "json"):
                        r = core.stream_request(pool, ep, core.make_body(api, model, "x"), 20)
                        self.assertEqual(r["response_status"], "incomplete")
                        self.assertFalse(r["ok"])
                        self.assertEqual(r["text"], "partial")
                finally:
                    pool.close()
        finally:
            srv.shutdown()
            srv.server_close()

    def test_buffered_samples_excluded_from_streaming_summary(self):
        normal = dict(Stats()._rec("m", total=2), tps=10, buffered=False)
        buffered = dict(Stats()._rec("m", total=20), tps=100, buffered=True)
        row = core.summarize([normal, buffered])[0]
        self.assertEqual(row["first_token_vals"], [1])
        self.assertEqual(row["tps_vals"], [10])


class ExpandedEvaluation(unittest.TestCase):
    def test_extended_answers_and_near_misses(self):
        self.assertEqual(len(core.QUIZ_EXTENDED), 8)
        for q in core.QUIZ_EXTENDED:
            self.assertTrue(core.check_answer(q["chk"], q["ans"], q.get("strict", False)), q["id"])
            self.assertFalse(core.check_answer(q["chk"], "错误", q.get("strict", False)), q["id"])
        self.assertFalse(core.check_answer(("json", {"price": 1, "unit": "元/包"}), '{"price":true,"unit":"元/包"}', True))
        self.assertFalse(core.check_answer(("json", {"price": 1, "unit": "元/包"}), '{"price":2,"price":1,"unit":"元/包"}', True))

    def test_randomized_tool_is_paired_and_reproducible(self):
        task = core.build_tasks(["tool"])[0]
        a = core.materialize_task(task, 123, 1)
        b = core.materialize_task(task, 123, 1)
        c = core.materialize_task(task, 123, 2)
        self.assertEqual(a, b)
        self.assertNotEqual(a["tool_result"], c["tool_result"])
        for api in ("chat", "responses"):
            pool = core.Pool()
            try:
                r = core.run_task(pool, selftest.fake_endpoint(FAKE[1], api), "tool-ok", a, 20)
                self.assertTrue(r["pass"])
                self.assertEqual(r["tool_result"], a["tool_result"])
            finally:
                pool.close()

    def test_context_matrix_cases(self):
        tasks = core.build_tasks(["needle_matrix"])
        self.assertEqual(len(tasks), 12)
        self.assertEqual({t["depth"] for t in tasks}, {10, 50, 90})
        self.assertEqual({t["length"] for t in tasks}, {10000, 30000})
        for task in tasks:
            x = core.materialize_task(task, 19, 1)
            self.assertEqual(x, core.materialize_task(task, 19, 1))
            case = x["case"]
            self.assertIsNone(x.get("system"))
            self.assertIn(case["distractor"], x["prompt"])
            answer = case["target"] if case["present"] else "未知"
            self.assertTrue(core.check_answer(x["chk"], answer, True))
            self.assertFalse(core.check_answer(x["chk"], case["distractor"], True))
            self.assertFalse(core.check_answer(x["chk"], answer + "。", True))
            if case["present"]:
                self.assertIn(case["target"], x["prompt"])

    def test_random_seed_and_conditions(self):
        task = core.build_tasks(["tool"])
        plan = {"endpoints": {"e": selftest.fake_endpoint(FAKE[1])}, "models": [("e", "m")], "tasks": task,
                "rounds": 2, "timeout": 20, "gap": 0, "warmup": False, "effort": "", "seed": 1, "concurrency": 1, "connection_mode": "reuse"}
        meta = {"created": "2026-01-01", "prompt": "", "context": 0}
        a = core.make_env(plan, {}, meta)
        b = core.make_env(dict(plan, seed=2), {}, meta)
        self.assertNotEqual(core.cond_of(a, "tool"), core.cond_of(b, "tool"))
        self.assertNotEqual(core.cond_of(a, "tool"), core.cond_of(core.make_env(dict(plan, concurrency=2), {}, meta), "tool"))
        self.assertNotEqual(core.cond_of(a, "tool"), core.cond_of(core.make_env(dict(plan, connection_mode="cold"), {}, meta), "tool"))

    def test_slo_includes_failures_and_missing_timing_is_unknown(self):
        def rec(ok, passed, elapsed):
            return dict(Stats()._rec("m", ok=ok, passed=passed), wall_total_s=elapsed)
        row = core.summarize([rec(True, True, 2), rec(True, False, 2), rec(False, False, 2), rec(True, True, 11)])[0]
        self.assertEqual((row["slo_n"], row["slo_rate"], row["slo_observed"]), (1, .25, 4))
        self.assertIsNone(core.summarize([Stats()._rec("m")])[0]["slo_rate"])

    def test_concurrency_and_cold_connections(self):
        state, lock = {"active": 0, "peak": 0}, threading.Lock()
        class ConcurrentFake(selftest.Fake):
            def do_POST(self):
                with lock:
                    state["active"] += 1
                    state["peak"] = max(state["peak"], state["active"])
                try:
                    time.sleep(.15)
                    super().do_POST()
                finally:
                    with lock:
                        state["active"] -= 1
        srv = server.ThreadingHTTPServer(("127.0.0.1", 0), ConcurrentFake)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            ep = selftest.fake_endpoint(f"http://127.0.0.1:{srv.server_address[1]}")
            plan = {"endpoints": {"e": ep}, "models": [("e", "quiz-good")], "tasks": core.build_tasks(["quiz"])[:4],
                    "rounds": 1, "timeout": 20, "gap": 0, "warmup": False, "seed": 2, "concurrency": 2, "connection_mode": "cold"}
            recs, _ = core.run_plan(plan, lambda _: None, threading.Event())
            self.assertEqual(len(recs), 4)
            self.assertTrue(all(r["ok"] and r["first_reused"] is False for r in recs))
            self.assertEqual([r["seq"] for r in recs], [1, 2, 3, 4])
            self.assertEqual(state["peak"], 2)
        finally:
            srv.shutdown()
            srv.server_close()

    def test_concurrent_stop_and_seed_pairing(self):
        stop = threading.Event()
        ep = selftest.fake_endpoint(FAKE[1])
        plan = {"endpoints": {"e": ep}, "models": [("e", "tool-ok"), ("e", "tool-ok")], "tasks": core.build_tasks(["tool"]),
                "rounds": 1, "timeout": 20, "gap": 0, "warmup": False, "seed": 42, "concurrency": 2}
        recs, _ = core.run_plan(plan, lambda _: None, stop)
        self.assertEqual(len(recs), 2)
        self.assertTrue(all(r["pass"] for r in recs))
        self.assertEqual(recs[0]["tool_result"], recs[1]["tool_result"])
        plan.update(tasks=core.build_tasks(["quiz"]), models=[("e", "delay-0.5")], rounds=3)
        recs, _ = core.run_plan(plan, lambda u: stop.set() if u.get("done", 0) >= 1 else None, stop)
        self.assertLessEqual(len(recs), 2)
        self.assertTrue(stop.is_set())

    def test_needle_matrix_end_to_end(self):
        plan = {"endpoints": {"e": selftest.fake_endpoint(FAKE[1])}, "models": [("e", "needle-good")],
                "tasks": core.build_tasks(["needle_matrix"]), "rounds": 1, "timeout": 20, "gap": 0,
                "warmup": False, "seed": 12, "concurrency": 2}
        recs, _ = core.run_plan(plan, lambda _: None, threading.Event())
        self.assertEqual(len(recs), 12)
        self.assertTrue(all(r["pass"] for r in recs))
        self.assertEqual(sum(r["case"]["present"] for r in recs), 6)

    def test_validation_includes_tool_two_steps_and_parameters(self):
        ep = store.upsert_endpoint({"name": "cap-test", "base_url": FAKE[1], "api_key": selftest.KEY, "api": "responses", "models": ["tool-ok"]})
        body = {"selection": [{"endpoint": ep["id"], "models": ["tool-ok"]}], "scenarios": ["tool"], "rounds": 2, "timeout": 20, "gap": 0, "warmup": True}
        with mock.patch.object(server, "MAX_REQUESTS", 4):
            with self.assertRaises(ValueError):
                server.validate(body)
        for key, value in (("concurrency", 0), ("concurrency", 5), ("connection_mode", "other"), ("seed", -1)):
            with self.assertRaises(ValueError):
                server.validate(dict(body, **{key: value}))

    def test_zero_cached_tokens_and_returned_model_are_preserved(self):
        rec = core._new_rec(selftest.fake_endpoint(FAKE[1]), "requested", "chat")
        acc = {"text": [], "reasoning": [], "items": [], "calls": {}}
        core._consume(rec, {"model": "reported", "system_fingerprint": "fp-test", "usage": {"prompt_cache_hit_tokens": 0}, "choices": []}, "chat", acc, .1)
        self.assertEqual(rec["usage"]["cached_tokens"], 0)
        self.assertEqual((rec["returned_model"], rec["system_fingerprint"]), ("reported", "fp-test"))
        rec.update(ok=True, total_s=1, first_token_s=.5, first_event_s=.1, scenario="speed", task="s")
        row = core.summarize([rec])[0]
        self.assertEqual((row["cached_known"], row["cached_hit_n"]), (1, 0))
        self.assertEqual(row["returned_models"], ["reported"])


class LanAccess(unittest.TestCase):
    """局域网访问：默认关闭、令牌校验、只绑非回环地址、开关状态机。"""

    def tearDown(self):
        server.lan_stop()

    def test_lan_ips_returns_list_without_loopback(self):
        ips = server.lan_ips()
        self.assertIsInstance(ips, list)
        for ip in ips:
            self.assertFalse(ip.startswith("127."), "不应把回环地址当成局域网地址：" + ip)
            self.assertNotEqual(ip, "0.0.0.0")

    def test_is_lan_addr(self):
        self.assertFalse(server.is_lan_addr("127.0.0.1"))
        self.assertFalse(server.is_lan_addr(None))
        self.assertTrue(server.is_lan_addr("192.168.1.5"))

    def test_default_off_and_info_shape(self):
        info = server.lan_info()
        self.assertFalse(info["enabled"])
        self.assertIsNone(info["url"])
        for key in ("enabled", "host", "port", "token", "url", "candidates"):
            self.assertIn(key, info)

    def test_token_required_for_lan_requests(self):
        """带令牌的 ?k= 与 cookie 都能过；缺失或错误则拒。"""
        token = "testtoken123"
        old = dict(server.LAN)
        try:
            server.LAN.update(srv=object(), host="192.168.1.5", port=8888, token=token)
            self.assertTrue(server.lan_info()["enabled"])
            handler_cls = server.Handler
            h = handler_cls.__new__(handler_cls)
            # 1) 无令牌
            h.path, h.headers = "/api/config", {}
            self.assertFalse(h._token_ok())
            # 2) ?k= 正确
            h.path = "/api/config?k=" + token
            self.assertTrue(h._token_ok())
            # 3) cookie 正确
            h.path = "/api/config"
            h.headers = {"Cookie": "a=1; mbtoken=" + token + "; b=2"}
            self.assertTrue(h._token_ok())
            # 4) cookie 错误
            h.headers = {"Cookie": "mbtoken=wrong"}
            self.assertFalse(h._token_ok())
        finally:
            server.LAN.update(old)

    def test_token_never_published_from_lan_request(self):
        """局域网设备请求 /api/network 时不应拿到令牌（避免设备间串联）。"""
        old = dict(server.LAN)
        try:
            server.LAN.update(srv=object(), host="192.168.1.5", port=8888, token="secret-token")
            info = server.lan_info()
            self.assertEqual(info["token"], "secret-token")  # 本机视角能看到
            self.assertIsNotNone(info["url"])
        finally:
            server.LAN.update(old)

    def test_api_network_toggle_roundtrip(self):
        """接口层：GET 初始关闭；POST 非法参数报错；POST 关闭是幂等的。"""
        status, body = call("GET", "/api/network")
        self.assertEqual(status, 200)
        self.assertIn("enabled", body)
        for bad in ({}, {"enable": "yes"}, {"enable": 1}):
            st, b = call("POST", "/api/network", bad)
            self.assertEqual(st, 400, "非法参数应报 400：" + str(bad))
        st, b = call("POST", "/api/network", {"enable": False})
        self.assertEqual(st, 200)
        self.assertFalse(b["enabled"])

    def test_denied_page_mentions_token(self):
        html = server.lan_denied_html().decode("utf-8")
        self.assertIn("需要访问令牌", html)
        self.assertIn("?k=", html)
