"""Gemini 原生协议契约：仅本机假服务，不访问真实模型。"""
import json
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store

KEY = "gemini-fixture-key"


class GeminiFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, value, stream=False):
        raw = ("".join("data: " + json.dumps(v, ensure_ascii=False) + "\n\n" for v in value).encode() if stream
               else json.dumps(value, ensure_ascii=False).encode())
        self.send_response(status)
        self.send_header("Content-Type", "text/event-stream" if stream else "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        parsed = urllib.parse.urlsplit(self.path)
        self.server.calls.append(("GET", self.path, dict(self.headers), None))
        if parsed.path != "/v1beta/models" or self.headers.get("x-goog-api-key") != KEY:
            return self.reply(401, {"error": {"message": "bad key"}})
        token = urllib.parse.parse_qs(parsed.query).get("pageToken", [None])[0]
        if not token:
            return self.reply(200, {"models": [{"name": "models/gemini-one", "supportedGenerationMethods": ["generateContent"]},
                                               {"name": "models/embedding", "supportedGenerationMethods": ["embedContent"]}],
                                    "nextPageToken": "next"})
        if self.server.mode == "page-error":
            return self.reply(503, {"error": {"message": "unavailable"}})
        if self.server.mode == "page-loop":
            return self.reply(200, {"models": [], "nextPageToken": "next"})
        return self.reply(200, {"models": [{"name": "models/gemini-two", "supportedGenerationMethods": ["generateContent"]}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(("POST", self.path, dict(self.headers), body))
        parsed = urllib.parse.urlsplit(self.path)
        if not parsed.path.startswith("/v1beta/models/") or not parsed.path.endswith(":streamGenerateContent") or self.headers.get("x-goog-api-key") != KEY:
            return self.reply(401, {"error": {"message": "bad endpoint or auth"}})
        if "model" in body or urllib.parse.parse_qs(parsed.query).get("alt") != ["sse"]:
            return self.reply(400, {"error": {"message": "bad wire format"}})
        model = parsed.path.split("/")[-1].split(":")[0]
        if model == "gemini-tool":
            if len(body["contents"]) == 1:
                ev = {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "call-1", "name": "lookup_price", "args": {"item": "MET混合滤料包"}}, "thoughtSignature": "c2ln"}]}, "finishReason": "STOP"}]}
                return self.reply(200, [ev], stream=True)
            previous = body["contents"][-2]["parts"][0]
            function_response = body["contents"][-1]["parts"][0]["functionResponse"]
            if previous.get("thoughtSignature") != "c2ln" or function_response.get("id") != "call-1" or function_response.get("name") != "lookup_price":
                return self.reply(400, {"error": {"message": "thought signature or tool replay missing"}})
            result = function_response["response"]
            text = json.dumps({"price": result["price"], "unit": result["unit"]}, ensure_ascii=False)
            return self.reply(200, [{"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": "STOP"}]}], stream=True)
        if model == "gemini-echo":
            return self.reply(200, [{"candidates": [{"content": {"parts": [{"text": KEY[:8]}]}}]},
                                    {"candidates": [{"content": {"parts": [{"text": KEY[8:]}]}, "finishReason": "STOP"}]}], stream=True)
        reason = "MAX_TOKENS" if model == "gemini-truncated" else "SAFETY" if model == "gemini-blocked" else "STOP"
        result = {"modelVersion": "gemini-actual", "candidates": [{"content": {"parts": [{"text": "你好"}]}, "finishReason": reason}],
                  "usageMetadata": {"promptTokenCount": 20, "cachedContentTokenCount": 3, "candidatesTokenCount": 5, "thoughtsTokenCount": 7}}
        if model == "gemini-nonstream":
            return self.reply(200, result)
        return self.reply(200, [{"candidates": [{"content": {"parts": [{"text": "想", "thought": True}]}}]}, result], stream=True)


class GeminiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), GeminiFixture)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ep = {"name": "fixture", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}/v1beta", "api_key": KEY, "api": "gemini"}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.calls, self.srv.mode = [], "ok"

    def test_model_listing_paginates_and_filters_unsupported_models(self):
        self.assertEqual(core.fetch_models(self.ep, 5), ["gemini-one", "gemini-two"])
        self.assertEqual(len(self.srv.calls), 2)
        self.assertNotIn(KEY, self.srv.calls[1][1])
        self.assertIn("pageToken=next", self.srv.calls[1][1])

    def test_failed_or_looping_pagination_does_not_overwrite_models(self):
        for mode in ("page-error", "page-loop"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
                endpoint = store.upsert_endpoint({**self.ep, "models": ["manual"]})
                self.srv.mode = mode
                result = store.discover(endpoint["id"])
                self.assertTrue(result["errors"])
                self.assertEqual(store.get_endpoint(endpoint["id"])["models"], ["manual"])

    def request(self, model):
        pool = core.Pool()
        try:
            return core.stream_request(pool, self.ep, core.make_body("gemini", model, "hi", system="system"), 5)
        finally:
            pool.close()

    def test_streaming_usage_thought_and_system(self):
        record = self.request("gemini-one")
        self.assertTrue(record["ok"], record["error"])
        self.assertEqual(record["text"], "你好")
        self.assertEqual(record["returned_model"], "gemini-actual")
        self.assertIsNotNone(record["first_think_s"])
        self.assertIsNotNone(record["first_token_s"])
        self.assertEqual((record["usage"]["input_tokens"], record["usage"]["output_tokens"], record["usage"]["reasoning_tokens"], record["usage"]["cached_tokens"]), (20, 12, 7, 3))
        body = self.srv.calls[-1][3]
        self.assertEqual(body["systemInstruction"]["parts"][0]["text"], "system")
        self.assertNotIn("model", body)

    def test_nonstream_and_incomplete_or_refused_outputs(self):
        result = self.request("gemini-nonstream")
        self.assertTrue(result["ok"], result["error"])
        self.assertTrue(result["non_stream"])
        for model in ("gemini-truncated", "gemini-blocked"):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertFalse(result["ok"])
                self.assertTrue(result["error"])

    def test_tool_call_replays_opaque_thought_signature(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 123, 1)
            result = core.run_task(pool, self.ep, "gemini-tool", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            self.assertEqual(len(self.srv.calls), 2)
            self.assertEqual(self.srv.calls[0][3]["tools"][0]["functionDeclarations"][0]["name"], "lookup_price")
        finally:
            pool.close()

    def test_split_key_echo_is_not_saved(self):
        result = self.request("gemini-echo")
        self.assertFalse(result["ok"])
        self.assertNotIn(KEY, json.dumps(core.public_rec(result), ensure_ascii=False))
        self.assertNotIn(KEY, json.dumps(result["events"], ensure_ascii=False))

    def test_distribution_task_uses_provider_default_temperature(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["modeltrace"])[0], 123, 1)
            core.run_task(pool, self.ep, "gemini-one", task, 5)
            self.assertNotIn("temperature", self.srv.calls[-1][3]["generationConfig"])
        finally:
            pool.close()

    def test_effort_override_rejected_before_request(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({**self.ep, "models": ["gemini-one"]})
            with self.assertRaisesRegex(ValueError, "推理强度"):
                server.validate({"selection": [{"endpoint": endpoint["id"], "models": ["gemini-one"]}],
                                 "scenarios": ["speed"], "prompt": "hi", "rounds": 1, "timeout": 10,
                                 "context": 0, "gap": 0, "warmup": False, "effort": "high"})
            self.assertEqual(self.srv.calls, [])
