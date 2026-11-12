"""Anthropic Messages 协议契约，全部使用本机假服务，不访问真实接口。"""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, store


class AnthropicFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def send_body(self, status, data, content_type):
        body = data if isinstance(data, bytes) else json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def authorized(self):
        return self.headers.get("x-api-key") == "fixture-key" and self.headers.get("anthropic-version") == "2023-06-01"

    def do_GET(self):
        if self.path != "/v1/models" or not self.authorized():
            return self.send_body(401, {"error": "invalid auth or path"}, "application/json")
        self.send_body(200, {"data": [{"id": "claude-test"}]}, "application/json")

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append((self.path, body, self.authorized()))
        if self.path != "/v1/messages" or not self.authorized():
            return self.send_body(401, {"error": "invalid auth or path"}, "application/json")
        model = body["model"]
        if model == "nonstream":
            return self.send_body(200, {"type": "message", "model": "claude-actual", "content": [{"type": "text", "text": "好"}],
                                        "stop_reason": "end_turn", "usage": {"input_tokens": 3, "output_tokens": 2,
                                        "cache_read_input_tokens": 4, "cache_creation_input_tokens": 5}}, "application/json")
        if model == "nonstream-tool":
            content = body["messages"][-1]["content"]
            if isinstance(content, list):
                result = json.loads(content[0]["content"])
                blocks = [{"type": "text", "text": json.dumps({"price": result["price"], "unit": result["unit"]}, ensure_ascii=False)}]
                reason = "end_turn"
            else:
                blocks = [{"type": "tool_use", "id": "call-1", "name": "lookup_price", "input": {"item": "MET混合滤料包"}}]
                reason = "tool_use"
            return self.send_body(200, {"type": "message", "model": model, "content": blocks, "stop_reason": reason,
                                        "usage": {"input_tokens": 3, "output_tokens": 8}}, "application/json")
        events = [
            {"type": "message_start", "message": {"type": "message", "model": "claude-actual", "usage": {
                "input_tokens": 3, "output_tokens": 0, "cache_read_input_tokens": 4, "cache_creation_input_tokens": 5}}},
        ]
        is_tool = model == "tool"
        second = is_tool and isinstance(body["messages"][-1]["content"], list)
        if is_tool and not second:
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "think"}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "signature_delta", "signature": "sig"}},
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "call-1", "name": "lookup_price", "input": {}}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '{"item":"MET'}},
                {"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": '混合滤料包"}'}},
                {"type": "content_block_stop", "index": 1},
            ]
            reason = "tool_use"
        else:
            if second:
                prior = body["messages"][-2]["content"]
                result = body["messages"][-1]["content"][0]
                if result.get("tool_use_id") != "call-1" or prior[0].get("signature") != "sig":
                    return self.send_body(400, {"error": "bad tool replay"}, "application/json")
                price = json.loads(result["content"])
                text = json.dumps({"price": price["price"], "unit": price["unit"]}, ensure_ascii=False)
            elif model in ("behavior-a", "behavior-b"):
                text = "alpha" if model == "behavior-a" else "beta"
            elif model == "modeltrace":
                import re
                match = re.search(r" (\d+) 个 1 到 355", body["messages"][0]["content"])
                count = int(match.group(1)) if match else 300
                import random
                generator = random.Random(count)
                text = ",".join(str(generator.randint(1, 355)) for _ in range(count))
            else:
                text = "好"
            events += [
                {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}},
                {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": text}},
                {"type": "content_block_stop", "index": 0},
            ]
            reason = "max_tokens" if model == "truncated" else "end_turn"
        events.append({"type": "message_delta", "delta": {"stop_reason": reason},
                       "usage": {"output_tokens": 8, "output_tokens_details": {"thinking_tokens": 2}}})
        if model != "missing-stop":
            events.append({"type": "message_stop"})
        payload = "".join("event: " + e["type"] + "\ndata: " + json.dumps(e, ensure_ascii=False) + "\n\n" for e in events).encode()
        if model == "truncated-wire":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(payload) + 10))
            self.end_headers()
            self.wfile.write(payload)
            self.close_connection = True
            return
        self.send_body(200, payload, "text/event-stream")


class AnthropicMessagesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), AnthropicFixture)
        cls.srv.daemon_threads = True
        cls.srv.requests = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.endpoint = {"name": "fixture", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}", "api_key": "fixture-key", "api": "anthropic"}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def request(self, model):
        pool = core.Pool()
        try:
            return core.stream_request(pool, self.endpoint, core.make_body("anthropic", model, "hi"), 5)
        finally:
            pool.close()

    def test_models_use_anthropic_auth_and_endpoint_can_be_saved(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "Claude", "base_url": self.endpoint["base_url"], "api_key": "fixture-key", "api": "anthropic"})
            self.assertEqual(ep["api"], "anthropic")
            self.assertNotIn("api_key", ep)
            self.assertEqual(core.fetch_models(store.get_endpoint(ep["id"])), ["claude-test"])

    def test_auto_discovery_detects_anthropic_models_and_protocol(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "Claude", "base_url": self.endpoint["base_url"], "api_key": "fixture-key", "api": "auto"})
            result = store.discover(ep["id"])
            self.assertEqual(result["api"], "anthropic")
            self.assertEqual(result["models"], 1)
            self.assertEqual(store.get_endpoint(ep["id"])["models"], ["claude-test"])

    def test_editing_auto_endpoint_clears_old_protocol_detection(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "Claude", "base_url": self.endpoint["base_url"], "api_key": "fixture-key", "api": "auto"})
            store.discover(ep["id"])
            updated = store.upsert_endpoint({"id": ep["id"], "name": "Claude", "base_url": "http://127.0.0.1:12345", "api": "auto"})
            self.assertIsNone(updated["api_detected"])

    def test_stream_collects_text_usage_and_normal_completion(self):
        result = self.request("claude-test")
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["text"], "好")
        self.assertEqual(result["returned_model"], "claude-actual")
        self.assertEqual(result["usage"]["input_tokens"], 12)
        self.assertEqual(result["usage"]["cached_tokens"], 4)
        self.assertEqual(result["usage"]["output_tokens"], 8)
        self.assertEqual(result["reasoning_tokens"], 2)
        self.assertIsNotNone(result["first_token_s"])

    def test_content_length_stream_can_reuse_connection_without_duplicate_requests(self):
        pool = core.Pool()
        before = len(self.srv.requests)
        try:
            results = [core.stream_request(pool, self.endpoint, core.make_body("anthropic", "claude-test", "hi"), 5)
                       for _ in range(3)]
            self.assertTrue(all(r["ok"] for r in results))
            self.assertEqual([r["reused"] for r in results], [False, True, True])
            self.assertFalse(any(r.get("conn_retry") for r in results))
            self.assertEqual(len(self.srv.requests) - before, 3)
        finally:
            pool.close()

    def test_nonstream_json_is_not_misrepresented_as_stream(self):
        result = self.request("nonstream")
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["text"], "好")
        self.assertTrue(result["non_stream"])
        self.assertEqual(result["usage"]["input_tokens"], 12)

    def test_truncation_and_missing_message_stop_are_failures(self):
        for model in ("truncated", "missing-stop", "truncated-wire"):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertFalse(result["ok"])
                self.assertTrue(result["error"])

    def test_tool_stream_reassembles_json_and_replays_thinking(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 123, 1)
            result = core.run_task(pool, self.endpoint, "tool", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            self.assertEqual(len(result["steps"]), 2)
            self.assertEqual(result["steps"][1]["status"], 200)
            requests = [item for item in self.srv.requests if item[1]["model"] == "tool"]
            self.assertEqual(requests[-2][1]["tools"][0]["input_schema"]["type"], "object")
            self.assertTrue(requests[-1][2])
        finally:
            pool.close()

    def test_nonstream_tool_call_is_replayed(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 123, 1)
            result = core.run_task(pool, self.endpoint, "nonstream-tool", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            self.assertTrue(result["non_stream"])
            self.assertEqual(len(result["steps"]), 2)
        finally:
            pool.close()

    def test_system_and_effort_use_anthropic_fields(self):
        body = core.make_body("anthropic", "claude-test", "hi", system="instructions", effort="low")
        self.assertEqual(body["system"], "instructions")
        self.assertEqual(body["output_config"]["effort"], "low")
        self.assertNotIn("reasoning", body)
        self.assertEqual(body["max_tokens"], 4096)


if __name__ == "__main__":
    unittest.main()
