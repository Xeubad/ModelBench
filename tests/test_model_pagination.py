"""模型列表分页与安全更新，使用本机假服务。"""
import json
import tempfile
import threading
import unittest
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import core, store


class PaginationFixture(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        self.server.paths.append(self.path)
        parsed = urllib.parse.urlsplit(self.path)
        if parsed.path != "/v1/models" or self.headers.get("x-api-key") != "fixture-key":
            status, data = 401, {"error": "bad auth"}
        else:
            query = urllib.parse.parse_qs(parsed.query)
            cursor = query.get("after_id", [None])[0]
            if self.server.mode == "bad-shape":
                status, data = 200, {"error": "not a model list"}
            elif self.server.mode == "missing-cursor":
                status, data = 200, {"data": [{"id": "first"}], "has_more": True}
            elif not cursor:
                status, data = 200, {"data": [{"id": "first"}], "has_more": True, "last_id": "first"}
            elif self.server.mode == "fail-next":
                status, data = 503, {"error": "temporarily unavailable"}
            elif self.server.mode == "loop":
                status, data = 200, {"data": [{"id": "first"}], "has_more": True, "last_id": "first"}
            else:
                status, data = 200, {"data": [{"id": "second"}], "has_more": False, "last_id": "second"}
        payload = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


class PaginationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), PaginationFixture)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ep = {"name": "fake", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}", "api_key": "fixture-key", "api": "anthropic"}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.mode = "ok"
        self.srv.paths = []

    def test_anthropic_cursor_fetches_all_pages(self):
        self.assertEqual(core.fetch_models(self.ep, timeout=5), ["first", "second"])
        self.assertEqual(len(self.srv.paths), 2)
        self.assertIn("after_id=first", self.srv.paths[1])

    def test_incomplete_or_cyclic_pages_raise_instead_of_returning_partial(self):
        for mode in ("loop", "missing-cursor", "fail-next", "bad-shape"):
            with self.subTest(mode=mode):
                self.srv.mode = mode
                self.srv.paths = []
                with self.assertRaises((ValueError, RuntimeError)):
                    core.fetch_models(self.ep, timeout=5)
                self.assertLessEqual(len(self.srv.paths), 3)

    def test_discovery_retains_manual_models_and_adds_all_pages(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({**self.ep, "models": ["manual"]})
            result = store.discover(ep["id"])
            self.assertEqual(result["errors"], [])
            self.assertEqual(store.get_endpoint(ep["id"])["models"], ["manual", "first", "second"])

    def test_failed_second_page_does_not_overwrite_existing_models(self):
        self.srv.mode = "fail-next"
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({**self.ep, "models": ["manual"]})
            result = store.discover(ep["id"])
            self.assertTrue(result["errors"])
            self.assertEqual(store.get_endpoint(ep["id"])["models"], ["manual"])


if __name__ == "__main__":
    unittest.main()
