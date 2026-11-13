"""通用接口接入：只使用临时目录和本机假模型服务。"""
import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, store


class GenericFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, value):
        payload = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        self.server.requests.append((self.path, dict(self.headers)))
        if getattr(self.server, "hold_models", False):
            self.server.models_started.set()
            self.server.models_release.wait(5)
        if self.path == "/v1/models":
            return self.reply(200, {"data": [{"id": "local-model"}]})
        return self.reply(404, {})

    def do_POST(self):
        self.server.requests.append((self.path, dict(self.headers)))
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/v1/chat/completions":
            return self.reply(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
        if self.path == "/v1/responses":
            if self.server.chat_only:
                return self.reply(404, {"error": "unsupported"})
            return self.reply(200, {"status": "completed", "output_text": "ok"})
        if self.path == "/v1/messages":
            return self.reply(200, {"type": "message", "content": [{"type": "text", "text": "ok"}],
                                    "stop_reason": "end_turn", "model": body["model"]})
        return self.reply(404, {})


class GenericEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), GenericFixture)
        cls.srv.daemon_threads = True
        cls.srv.chat_only = False
        cls.srv.requests = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}/v1"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.chat_only = False
        self.srv.requests = []

    def test_keyless_endpoints_run_all_supported_protocols_without_openclaw(self):
        for style in ("responses", "chat", "anthropic"):
            with self.subTest(style=style), tempfile.TemporaryDirectory() as root, \
                 mock.patch.object(store, "DATA_DIR", root + "/data"), mock.patch.object(store, "RESULTS_DIR", root + "/results"), \
                 mock.patch.object(store, "OPENCLAW_CONFIG", root + "/missing.json"), mock.patch.object(store, "LEGACY_DIR", root + "/missing-history"):
                endpoint = store.upsert_endpoint({"name": style, "base_url": self.base, "api": style, "models": ["local-model"]})
                self.assertEqual(endpoint["key_hint"], "未设置")
                self.assertEqual(self.srv.requests, [], "保存接口不应发送网络请求")
                params = {"selection": [{"endpoint": endpoint["id"], "models": ["local-model"]}], "scenarios": ["speed"],
                          "prompt": "hi", "rounds": 1, "timeout": 5, "context": 0, "gap": 0, "warmup": False, "name": ""}
                result = store.execute(params, lambda _: None, threading.Event())
                self.assertTrue(result["records"][0]["ok"], result["records"][0]["error"])
                self.assertTrue(store.history())
                headers = self.srv.requests[-1][1]
                self.assertNotIn("Authorization", headers)
                self.assertNotIn("x-api-key", headers)
                if style == "anthropic":
                    self.assertEqual(headers["anthropic-version"], "2023-06-01")
                self.srv.requests = []

    def test_keyless_auto_detection_can_fallback_to_chat(self):
        self.srv.chat_only = True
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({"name": "local", "base_url": self.base, "api": "auto", "models": []})
            result = store.discover(endpoint["id"])
            self.assertEqual(result["api"], "chat")
            self.assertIn("local-model", store.get_endpoint(endpoint["id"])["models"])
            self.assertTrue(all("Authorization" not in headers and "x-api-key" not in headers for _, headers in self.srv.requests))

    def test_discovery_does_not_save_if_authentication_changes_mid_request(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({"name": "local", "base_url": self.base, "api": "chat", "api_key": "fixture-long-key", "models": ["manual"]})
            self.srv.hold_models = True
            self.srv.models_started, self.srv.models_release = threading.Event(), threading.Event()
            outcome = []
            def discover():
                try:
                    outcome.append(store.discover(endpoint["id"]))
                except Exception as exc:
                    outcome.append(exc)
            worker = threading.Thread(target=discover)
            worker.start()
            try:
                self.assertTrue(self.srv.models_started.wait(3))
                store.upsert_endpoint({"id": endpoint["id"], "name": "local", "base_url": self.base, "api": "chat", "auth_mode": "api-key"})
            finally:
                self.srv.models_release.set()
                worker.join(6)
                self.srv.hold_models = False
            self.assertEqual(len(outcome), 1)
            self.assertIsInstance(outcome[0], ValueError)
            self.assertEqual(store.get_endpoint(endpoint["id"])["models"], ["manual"])

    def test_edit_can_preserve_replace_and_explicitly_clear_key(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({"name": "local", "base_url": self.base, "api": "auto", "api_key": "fixture-long-key", "models": ["local-model"]})
            store.discover(endpoint["id"])
            saved = store.get_endpoint(endpoint["id"])
            self.assertIsNotNone(saved["api_detected"])
            store.upsert_endpoint({"id": endpoint["id"], "name": "local", "base_url": self.base, "api": "auto", "api_key": ""})
            self.assertEqual(store.get_endpoint(endpoint["id"])["api_key"], "fixture-long-key")
            cleared = store.upsert_endpoint({"id": endpoint["id"], "name": "local", "base_url": self.base, "api": "auto", "clear_api_key": True})
            self.assertEqual(cleared["key_hint"], "未设置")
            saved = store.get_endpoint(endpoint["id"])
            self.assertEqual(saved["api_key"], "")
            self.assertIsNone(saved["api_detected"])
            self.assertIsNone(saved["models_checked_at"])
            self.assertEqual(saved["models"], ["local-model"])
            for extra in ({"clear_api_key": "yes"}, {"clear_api_key": True, "api_key": "new-key"}):
                with self.assertRaises(ValueError):
                    store.upsert_endpoint({"id": endpoint["id"], "name": "local", "base_url": self.base, "api": "auto", **extra})


if __name__ == "__main__":
    unittest.main()
