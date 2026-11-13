"""自定义鉴权头和非密钥查询参数：仅本机假服务。"""
import json
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store

KEY = "custom-fixture-secret"


class CustomFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def reply(self, status, obj):
        raw = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.server.calls.append(("GET", self.path, dict(self.headers)))
        parsed = urllib.parse.urlsplit(self.path)
        query = urllib.parse.parse_qs(parsed.query)
        if parsed.path != "/v1/models" or self.headers.get("X-Service-Token") != "ApiKey " + KEY:
            return self.reply(401, {"error": "bad auth"})
        if not query.get("after"):
            return self.reply(200, {"data": [{"id": "first"}], "has_more": True, "last_id": "first"})
        return self.reply(200, {"data": [{"id": "second"}], "has_more": False})

    def do_POST(self):
        self.server.calls.append(("POST", self.path, dict(self.headers)))
        self.rfile.read(int(self.headers["Content-Length"]))
        return self.reply(200, {"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})


class CustomEndpointTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), CustomFixture)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.calls = []

    def config(self):
        return {"name": "custom", "base_url": self.base, "api": "chat", "api_key": KEY, "auth_mode": "custom",
                "auth_header": "X-Service-Token", "auth_prefix": "ApiKey ", "query_params": {"api-version": "2025-01-01", "tenant": "demo"}, "models": ["manual"]}

    def test_custom_header_and_public_query_work_for_discovery_and_requests(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), mock.patch.object(store, "RESULTS_DIR", root + "/results"):
            endpoint = store.upsert_endpoint(self.config())
            result = store.discover(endpoint["id"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(store.get_endpoint(endpoint["id"])["models"], ["manual", "first", "second"])
            self.assertEqual(len(self.srv.calls), 2)
            for _, path, headers in self.srv.calls:
                self.assertNotIn(KEY, path)
                self.assertEqual(headers["X-Service-Token"], "ApiKey " + KEY)
                self.assertNotIn("Authorization", headers)
                query = urllib.parse.parse_qs(urllib.parse.urlsplit(path).query)
                self.assertEqual(query["api-version"], ["2025-01-01"])
                self.assertEqual(query["tenant"], ["demo"])
            self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(self.srv.calls[-1][1]).query)["after"], ["first"])
            params = server.validate({"selection": [{"endpoint": endpoint["id"], "models": ["manual"]}], "scenarios": ["speed"],
                                      "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False})
            run = store.execute(params, lambda _: None, threading.Event())
            self.assertTrue(run["records"][0]["ok"], run["records"][0]["error"])
            self.assertNotIn(KEY, json.dumps(run, ensure_ascii=False))
            self.assertEqual(run["env"]["endpoints"][0]["query_params"]["tenant"], "demo")
            self.assertEqual(self.srv.calls[-1][0], "POST")
            self.assertEqual(urllib.parse.parse_qs(urllib.parse.urlsplit(self.srv.calls[-1][1]).query)["tenant"], ["demo"])

    def test_unsafe_headers_and_secret_query_parameters_are_rejected(self):
        cases = [{"auth_header": "Host"}, {"auth_header": "X-Bad\r\nInjected"}, {"auth_prefix": "Bad\n"},
                 {"query_params": {"api_key": "secret"}}, {"query_params": {"authorization": "x"}},
                 {"query_params": {"auth": "x"}}, {"query_params": {"sig": "x"}}, {"query_params": {"access_token": "x"}},
                 {"query_params": {"tenant": KEY}}, {"query_params": {"after": "collision"}}]
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            for values in cases:
                with self.subTest(values=list(values)), self.assertRaises(ValueError):
                    store.upsert_endpoint({**self.config(), **values})
            self.assertEqual(self.srv.calls, [])

    def test_query_and_custom_auth_are_part_of_comparison_identity(self):
        endpoint = {"name": "custom", "base_url": self.base, "api": "chat", "auth_mode": "custom",
                    "auth_header": "X-Service-Token", "auth_prefix": "ApiKey ", "query_params": {"tenant": "a"}}
        other = dict(endpoint, query_params={"tenant": "b"})
        self.assertNotEqual(core.service_of({"endpoints": [endpoint]}, "custom/model"), core.service_of({"endpoints": [other]}, "custom/model"))
