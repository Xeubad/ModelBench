"""密钥回显与请求重发的回归测试；只连接本机假服务。"""
import json
import os
import socket
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store

KEY = "fixture-sensitive-key"
OTHER_KEY = "other-fixture-secret"


class SafetyFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def reply(self, status, payload, content_type="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.startswith("/v1/models"):
            if getattr(self.server, "other_key_as_model", False):
                return self.reply(200, {"data": [{"id": OTHER_KEY}, {"id": "clean-model"}]})
            return self.reply(401, {"error": {"message": "invalid key " + KEY}})
        self.reply(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.received += 1
        self.server.payloads.append(body)
        model = body["model"]
        if model == "error-echo":
            return self.reply(401, {"error": {"message": "invalid key " + KEY}})
        if model == "json-echo":
            return self.reply(200, {"choices": [{"message": {"content": KEY}, "finish_reason": "stop"}]})
        if model == "json-key-echo":
            return self.reply(200, {"status": "completed", "output_text": "ok", "usage": {KEY: 3, "output_tokens": 1}})
        if model == "tool-other-key" and len(body.get("messages", [])) == 1:
            args = json.dumps({"item": core.TOOL_RESULT["item"], "extra": OTHER_KEY}, ensure_ascii=False)
            return self.reply(200, {"choices": [{"message": {"tool_calls": [{"id": "c1", "function": {"name": "lookup_price", "arguments": args}}]}, "finish_reason": "tool_calls"}]})
        if model == "disconnect" and self.server.received == 2:
            self.close_connection = True
            self.connection.shutdown(socket.SHUT_RDWR)
            self.connection.close()
            return
        if model == "stream-echo":
            chunks = [KEY[:8], KEY[8:]]
        elif model == "other-key":
            chunks = [OTHER_KEY]
        else:
            chunks = ["ok"]
        events = [{"choices": [{"delta": {"content": part}}]} for part in chunks]
        events.append({"choices": [{"delta": {}, "finish_reason": "stop"}]})
        payload = "".join("data: " + json.dumps(ev) + "\n\n" for ev in events).encode()
        self.reply(200, payload, "text/event-stream")


class TransportSafetyTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), SafetyFixture)
        cls.srv.daemon_threads = True
        cls.srv.received = 0
        cls.srv.payloads = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ep = {"name": "fake", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}", "api_key": KEY, "api": "chat"}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.received = 0
        self.srv.payloads = []

    def test_error_and_nonstream_echo_are_redacted(self):
        for model in ("error-echo", "json-echo"):
            with self.subTest(model=model):
                pool = core.Pool()
                try:
                    rec = core.stream_request(pool, self.ep, core.make_body("chat", model, "hi"), 5)
                    visible = {k: v for k, v in rec.items() if not k.startswith("_")}
                    self.assertNotIn(KEY, json.dumps(visible, ensure_ascii=False))
                    self.assertFalse(rec["ok"])
                    self.assertIn("隐藏", rec["error"])
                finally:
                    pool.close()

    def test_key_echoed_as_json_object_key_is_not_persisted(self):
        pool = core.Pool()
        try:
            rec = core.stream_request(pool, dict(self.ep, api="responses"), core.make_body("responses", "json-key-echo", "hi"), 5)
            self.assertNotIn(KEY, json.dumps(core.public_rec(rec), ensure_ascii=False))
            self.assertNotIn(KEY, json.dumps(rec.get("events"), ensure_ascii=False))
        finally:
            pool.close()

    def test_split_stream_echo_is_not_reconstructable_from_raw_events(self):
        pool = core.Pool()
        try:
            rec = core.stream_request(pool, self.ep, core.make_body("chat", "stream-echo", "hi"), 5)
            self.assertNotIn(KEY, json.dumps({k: v for k, v in rec.items() if not k.startswith("_")}))
            joined = "".join(ev.get("event", {}).get("choices", [{}])[0].get("delta", {}).get("content", "")
                             for ev in rec["events"])
            self.assertNotIn(KEY, joined)
            self.assertFalse(rec["ok"])
            self.assertIn("隐藏", rec["error"])
        finally:
            pool.close()

    def test_endpoint_discovery_error_does_not_expose_key(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat"})
            result = store.discover(ep["id"])
            self.assertTrue(result["errors"])
            self.assertNotIn(KEY, json.dumps(result, ensure_ascii=False))

    def test_discovery_never_saves_another_endpoint_key_as_model(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            a = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat"})
            store.upsert_endpoint({"name": "other", "base_url": self.ep["base_url"], "api_key": OTHER_KEY, "api": "chat"})
            self.srv.other_key_as_model = True
            try:
                result = store.discover(a["id"])
            finally:
                self.srv.other_key_as_model = False
            self.assertNotIn(OTHER_KEY, json.dumps(result))
            self.assertNotIn(OTHER_KEY, store.get_endpoint(a["id"])["models"])
            self.assertIn("clean-model", store.get_endpoint(a["id"])["models"])

    def test_echo_never_reaches_result_raw_or_report(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            ep = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY,
                                        "api": "chat", "models": ["stream-echo"]})
            params = {"selection": [{"endpoint": ep["id"], "models": ["stream-echo"]}],
                      "scenarios": ["speed"], "prompt": "hi", "rounds": 1, "timeout": 5, "context": 0,
                      "gap": 0, "warmup": False, "effort": "", "name": ""}
            result = store.execute(params, lambda _: None, threading.Event())
            for filename in ("result.json", "raw.json", "report.md"):
                with open(os.path.join(store.run_dir(result["id"]), filename), encoding="utf-8") as handle:
                    self.assertNotIn(KEY, handle.read(), filename)

    def test_prompt_does_not_store_configured_key_in_environment(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            ep = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY,
                                        "api": "chat", "models": ["normal"]})
            params = {"selection": [{"endpoint": ep["id"], "models": ["normal"]}], "scenarios": ["speed"],
                      "prompt": "test " + KEY, "rounds": 1, "timeout": 5, "context": 0, "gap": 0,
                      "warmup": False, "effort": "", "name": ""}
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.execute(params, lambda _: None, threading.Event())
            self.assertEqual(self.srv.received, 0)
            self.assertFalse(os.path.exists(store.RESULTS_DIR))

    def test_output_is_redacted_against_all_keys_in_same_run(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            a = store.upsert_endpoint({"name": "a", "base_url": self.ep["base_url"], "api_key": KEY,
                                       "api": "chat", "models": ["other-key"]})
            b = store.upsert_endpoint({"name": "b", "base_url": self.ep["base_url"], "api_key": OTHER_KEY,
                                       "api": "chat", "models": ["normal"]})
            params = {"selection": [{"endpoint": a["id"], "models": ["other-key"]}, {"endpoint": b["id"], "models": ["normal"]}],
                      "scenarios": ["speed"], "prompt": "hi", "rounds": 1, "timeout": 5, "context": 0,
                      "gap": 0, "warmup": False, "effort": "", "name": ""}
            result = store.execute(params, lambda _: None, threading.Event())
            for filename in ("result.json", "raw.json", "report.md"):
                with open(os.path.join(store.run_dir(result["id"]), filename), encoding="utf-8") as handle:
                    self.assertNotIn(OTHER_KEY, handle.read(), filename)
            record = next(r for r in result["records"] if r["model"] == "other-key")
            self.assertFalse(record["ok"])

    def test_unselected_saved_key_is_redacted_from_echo_and_saved_prompt(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            a = store.upsert_endpoint({"name": "a", "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat", "models": ["other-key"]})
            store.upsert_endpoint({"name": "b", "base_url": self.ep["base_url"], "api_key": OTHER_KEY, "api": "chat", "models": ["normal"]})
            params = {"selection": [{"endpoint": a["id"], "models": ["other-key"]}], "scenarios": ["speed"],
                      "prompt": "hi", "rounds": 1, "timeout": 5, "context": 0, "gap": 0,
                      "warmup": False, "effort": "", "name": ""}
            result = store.execute(params, lambda _: None, threading.Event())
            for filename in ("result.json", "raw.json", "report.md"):
                with open(os.path.join(store.run_dir(result["id"]), filename), encoding="utf-8") as handle:
                    self.assertNotIn(OTHER_KEY, handle.read(), filename)
            before = self.srv.received
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.execute(dict(params, prompt="test " + OTHER_KEY), lambda _: None, threading.Event())
            self.assertEqual(self.srv.received, before)

    def test_tool_replay_does_not_send_unselected_endpoint_secret(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            a = store.upsert_endpoint({"name": "a", "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat", "models": ["tool-other-key"]})
            store.upsert_endpoint({"name": "b", "base_url": self.ep["base_url"], "api_key": OTHER_KEY, "api": "chat", "models": ["normal"]})
            params = {"selection": [{"endpoint": a["id"], "models": ["tool-other-key"]}], "scenarios": ["tool"],
                      "prompt": "", "rounds": 1, "timeout": 5, "context": 0, "gap": 0,
                      "warmup": False, "effort": "", "name": ""}
            result = store.execute(params, lambda _: None, threading.Event())
            self.assertEqual(self.srv.received, 1, "含密钥的工具回答不得作为第二步回放")
            self.assertNotIn(OTHER_KEY, json.dumps(self.srv.payloads, ensure_ascii=False))
            self.assertFalse(result["records"][0]["ok"])

    def test_validate_rejects_prompt_containing_saved_key(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY,
                                        "api": "chat", "models": ["normal"]})
            body = {"selection": [{"endpoint": ep["id"], "models": ["normal"]}], "scenarios": ["speed"],
                    "prompt": "test " + KEY, "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False}
            with self.assertRaisesRegex(ValueError, "密钥"):
                server.validate(body)

    def test_base_url_rejects_embedded_credentials_and_query(self):
        for url in ("https://name:password@example.com/v1", "https://example.com/v1?token=secret", "https://example.com/v1#key"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    core.endpoint_parts(url)

    def test_endpoint_metadata_and_run_directory_reject_saved_keys(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.upsert_endpoint({"name": KEY, "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat"})
            ep = store.upsert_endpoint({"name": "fake", "base_url": self.ep["base_url"], "api_key": KEY, "api": "chat", "models": ["normal"]})
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.upsert_endpoint({"name": "other", "base_url": self.ep["base_url"], "api_key": OTHER_KEY, "api": "chat", "models": [KEY]})
            params = {"selection": [{"endpoint": ep["id"], "models": ["normal"]}], "scenarios": ["speed"],
                      "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0,
                      "warmup": False, "effort": "", "name": KEY}
            with self.assertRaisesRegex(ValueError, "密钥"):
                server.validate(params)
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.execute(params, lambda _: None, threading.Event())

    def test_endpoint_path_does_not_embed_its_saved_key(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.upsert_endpoint({"name": "fake", "base_url": "https://example.com/" + KEY,
                                       "api_key": KEY, "api": "chat"})

    def test_no_retry_when_server_received_post_and_dropped_response(self):
        pool = core.Pool()
        try:
            first = core.stream_request(pool, self.ep, core.make_body("chat", "disconnect", "first"), 5)
            self.assertTrue(first["ok"])
            second = core.stream_request(pool, self.ep, core.make_body("chat", "disconnect", "second"), 5)
            self.assertEqual(self.srv.received, 2, "同一逻辑请求不得自动重发 POST")
            self.assertFalse(second["ok"])
            self.assertTrue(second["delivery_uncertain"])
            self.assertEqual(len(second["attempts"]), 1)
            second.update(scenario="speed", task="speed", task_name="速度")
            row = core.summarize([second])[0]
            self.assertEqual(row["uncertain_delivery_n"], 1)
            self.assertIn("送达状态未知", core.conclusion([row]))
        finally:
            pool.close()


if __name__ == "__main__":
    unittest.main()
