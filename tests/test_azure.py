"""Azure OpenAI 部署路径和鉴权契约，本机假服务。"""
import json
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store

KEY = "azure-fixture-key"


class AzureFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append((self.path, dict(self.headers), body))
        path = urllib.parse.urlsplit(self.path)
        version = urllib.parse.parse_qs(path.query).get("api-version")
        classic = path.path == "/openai/deployments/deploy-a/chat/completions" and version == ["2024-10-21"] and "model" not in body
        v1 = path.path == "/openai/v1/chat/completions" and not path.query and body.get("model") == "deploy-a"
        authenticated = self.headers.get("api-key") == KEY or self.headers.get("Authorization") == "Bearer " + KEY
        if not (classic or v1) or not authenticated:
            return self.reply(400, {"error": "invalid request"})
        if body.get("tools") and len(body["messages"]) == 1:
            call = {"id": "call-1", "function": {"name": "lookup_price", "arguments": json.dumps({"item": "MET混合滤料包"}, ensure_ascii=False)}}
            return self.reply(200, {"choices": [{"message": {"tool_calls": [call]}, "finish_reason": "tool_calls"}]})
        if body.get("tools"):
            tool_response = json.loads(body["messages"][-1]["content"])
            text = json.dumps({"price": tool_response["price"], "unit": tool_response["unit"]}, ensure_ascii=False)
        else:
            text = "ok"
        events = [{"model": "reported-model", "choices": [{"delta": {"content": text}}]},
                  {"choices": [{"delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 4, "completion_tokens": 2}}]
        raw = "".join("data: " + json.dumps(item, ensure_ascii=False) + "\n\n" for item in events).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def reply(self, status, obj):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class AzureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), AzureFixture)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ep = {"name": "azure", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}", "api": "azure",
                  "api_key": KEY, "azure_api_version": "2024-10-21", "azure_auth": "api_key"}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.calls = []

    def request(self, ep=None):
        pool = core.Pool()
        try:
            return core.stream_request(pool, ep or self.ep, core.make_body("azure", "deploy-a", "hi"), 5)
        finally:
            pool.close()

    def test_api_key_auth_uses_deployment_path_and_does_not_put_key_in_url(self):
        result = self.request()
        self.assertTrue(result["ok"], result["error"])
        path, headers, body = self.srv.calls[-1]
        self.assertEqual(headers["api-key"], KEY)
        self.assertNotIn("Authorization", headers)
        self.assertNotIn(KEY, path)
        self.assertNotIn("model", body)
        self.assertEqual(result["returned_model"], "reported-model")
        self.assertEqual(result["usage"]["output_tokens"], 2)

    def test_entra_bearer_auth_is_optional_and_explicit(self):
        result = self.request(dict(self.ep, azure_auth="bearer"))
        self.assertTrue(result["ok"], result["error"])
        _, headers, _ = self.srv.calls[-1]
        self.assertEqual(headers["Authorization"], "Bearer " + KEY)
        self.assertNotIn("api-key", headers)

    def test_openai_v1_allows_api_key_header_for_compatible_services(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({"name": "v1", "base_url": self.ep["base_url"] + "/openai/v1", "api": "chat",
                                              "api_key": KEY, "auth_mode": "api-key", "models": ["deploy-a"]})
            saved = store.get_endpoint(endpoint["id"])
            self.assertEqual(saved["auth_mode"], "api-key")
            pool = core.Pool()
            try:
                result = core.stream_request(pool, saved, core.make_body("chat", "deploy-a", "hi"), 5)
            finally:
                pool.close()
            self.assertTrue(result["ok"], result["error"])
            self.assertEqual(self.srv.calls[-1][1].get("api-key"), KEY)
            self.assertNotIn("Authorization", self.srv.calls[-1][1])

    def test_tool_call_uses_azure_deployment_in_both_steps(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 10, 1)
            result = core.run_task(pool, self.ep, "deploy-a", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            self.assertEqual(len(self.srv.calls), 2)
        finally:
            pool.close()

    def test_models_are_manual_and_settings_are_validated(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({**self.ep, "models": ["deploy-a"]})
            result = store.discover(endpoint["id"])
            self.assertTrue(result["errors"])
            self.assertEqual(store.get_endpoint(endpoint["id"])["models"], ["deploy-a"])
            self.assertEqual(self.srv.calls, [])
            for settings in ({"azure_api_version": "2024-10-21&key=oops"}, {"azure_auth": "unknown"}):
                with self.assertRaises(ValueError):
                    store.upsert_endpoint({**self.ep, **settings, "name": "other"})

    def test_api_version_participates_in_service_identity(self):
        def env(version):
            return {"endpoints": [{"name": "azure", "base_url": self.ep["base_url"], "api": "azure", "azure_api_version": version, "azure_auth": "api_key"}]}
        self.assertNotEqual(core.service_of(env("2024-10-21"), "azure/deploy-a"), core.service_of(env("2025-01-01-preview"), "azure/deploy-a"))

    def test_sensitive_key_never_in_public_endpoint_or_environment(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), mock.patch.object(store, "RESULTS_DIR", root + "/results"):
            endpoint = store.upsert_endpoint({**self.ep, "models": ["deploy-a"]})
            self.assertNotIn(KEY, json.dumps(endpoint))
            params = server.validate({"selection": [{"endpoint": endpoint["id"], "models": ["deploy-a"]}], "scenarios": ["speed"],
                                      "prompt": "hi", "rounds": 1, "timeout": 10, "gap": 0, "context": 0, "warmup": False})
            result = store.execute(params, lambda _: None, threading.Event())
            self.assertNotIn(KEY, json.dumps(result, ensure_ascii=False))
            self.assertEqual(result["env"]["endpoints"][0]["azure_api_version"], "2024-10-21")
