"""meow 独立进程适配器的本机假服务契约。"""
import http.client
import json
import socket
import tempfile
import threading
import unittest
from unittest import mock
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from model_test import meow_bridge, server, store


class FakeMeow(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass

    def send_json(self, value, status=200):
        raw = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/api/bootstrap":
            return self.send_json({"token": "fixture-meow-token", "version": "4.5.1"})
        if self.headers.get("X-Meow-Token") != "fixture-meow-token":
            return self.send_json({"error": {"code": "session_token_required"}}, 403)
        if self.path == "/api/snapshot":
            return self.send_json({"brand": "meow LLM Detector", "version": "4.5.1", "packages": [
                {"id": "fixture", "version": "1.0.0", "name": "Test", "mode": "claude", "publisher": "local",
                 "content_sha256": "a" * 64, "models": [{"id": "candidate", "name": "Candidate"}], "secret": "should-not-escape"}],
                "endpoints": [{"id": "endpoint-1", "name": "fixture", "mode": "claude", "model": "model-alias",
                               "credential_saved": True, "credential_ref": "should-not-escape"}], "sessions": [], "active": []})
        if self.path == "/api/report/" + "a" * 32:
            return self.send_json({"product": "meow LLM Detector", "version": "4.5.1", "session_id": "a" * 32,
                "operational_status": "completed", "claimed_model": "candidate", "request_model": "model-alias",
                "benchmark": {"id": "fixture", "version": "1.0.0", "content_sha256": "a" * 64, "publisher": "local"},
                "fingerprint": {"verdict": "insufficient", "model": None, "reasons": ["samples_incomplete"], "valid_samples": 2,
                                "matches": {"modelbench-secret": 0.5}},
                "progress": {"planned": 4, "logical_completed": 4, "valid_samples": 2, "retries": 0},
                "results": [{"answer": "private-raw-response"}]})
        return self.send_json({"error": {"code": "not_found"}}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.posted.append((self.path, body, self.headers.get("X-Meow-Token")))
        if self.path == "/api/run/estimate":
            return self.send_json({"logical_requests": 4, "retry_budget": body["runtime"]["retry_budget"],
                                   "maximum_http_attempts": 4 + body["runtime"]["retry_budget"], "known_price": False})
        if self.path == "/api/run/start":
            if getattr(self.server, "drop_start", False):
                self.connection.shutdown(socket.SHUT_RDWR)
                self.connection.close()
                return
            return self.send_json({"session_id": "a" * 32})
        if self.path == "/api/run/stop":
            return self.send_json({"stopped": True})
        return self.send_json({"error": {"code": "not_found"}}, 404)


class MeowBridgeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), FakeMeow)
        cls.srv.daemon_threads = True
        cls.srv.posted = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.port = cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def config(self):
        return {"port": self.port, "endpoint_id": "endpoint-1", "endpoint_sha256": meow_bridge.status(self.port)["endpoints"][0]["config_sha256"], "package_id": "fixture", "package_version": "1.0.0", "benchmark_sha256": "a" * 64,
                "claimed_model": "candidate", "request_model": "model-alias", "tier": "low", "workers": 1, "retry_budget": 0}

    def test_status_exposes_only_expected_metadata(self):
        snapshot = meow_bridge.status(self.port)
        text = json.dumps(snapshot)
        self.assertNotIn("fixture-meow-token", text)
        self.assertNotIn("should-not-escape", text)
        self.assertEqual(snapshot["packages"][0]["id"], "fixture")
        self.assertEqual(snapshot["endpoints"][0]["id"], "endpoint-1")

    def test_estimate_then_start_uses_only_meow_saved_endpoint(self):
        config = self.config()
        estimate = meow_bridge.estimate(config)
        self.assertEqual(estimate["logical_requests"], 4)
        self.assertEqual(estimate["maximum_http_attempts"], 4)
        result = meow_bridge.start(dict(config, expected_max_http_attempts=4))
        self.assertEqual(result["session_id"], "a" * 32)
        request = [p for p in self.srv.posted if p[0] == "/api/run/start"][-1]
        self.assertEqual(request[1]["endpoint_id"], "endpoint-1")
        self.assertNotIn("key", request[1])
        self.assertNotIn("base_url", request[1])
        self.assertEqual(request[1]["runtime"]["retry_budget"], 0)
        self.assertEqual(request[2], "fixture-meow-token")
        with self.assertRaisesRegex(ValueError, "估算"):
            meow_bridge.start(dict(config, expected_max_http_attempts=3))

    def test_uncertain_start_is_not_replayed_and_changed_bank_is_rejected(self):
        config = self.config()
        with self.assertRaisesRegex(ValueError, "基准包已变化"):
            meow_bridge.start(dict(config, benchmark_sha256="b" * 64, expected_max_http_attempts=4))
        before = sum(p[0] == "/api/run/start" for p in self.srv.posted)
        self.srv.drop_start = True
        try:
            with self.assertRaisesRegex(ValueError, "送达状态未知"):
                meow_bridge.start(dict(config, expected_max_http_attempts=4))
        finally:
            self.srv.drop_start = False
        after = sum(p[0] == "/api/run/start" for p in self.srv.posted)
        self.assertEqual(after - before, 1)

    def test_report_returns_summary_without_raw_answers(self):
        report = meow_bridge.report(self.port, "a" * 32)
        self.assertEqual(report["fingerprint"]["verdict"], "insufficient")
        self.assertNotIn("private-raw-response", json.dumps(report))
        self.assertEqual(report["benchmark"]["content_sha256"], "a" * 64)

    def test_web_bridge_uses_local_meow_configuration(self):
        app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        app.daemon_threads = True
        threading.Thread(target=app.serve_forever, daemon=True).start()
        port = app.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(store, "DATA_DIR", directory), \
                 mock.patch.object(server, "PORT", port), mock.patch.object(server, "BIND_HOSTS", ["127.0.0.1"]):
                store.upsert_endpoint({"name": "own", "base_url": "http://127.0.0.1:9", "api_key": "modelbench-secret", "api": "chat"})
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                conn.request("GET", f"/api/meow/status?port={self.port}")
                response = conn.getresponse()
                self.assertEqual(response.status, 200, response.read())
                body = dict(self.config(), expected_max_http_attempts=4)
                conn.request("POST", "/api/meow/start", json.dumps(body), {"Content-Type": "application/json"})
                response = conn.getresponse()
                payload = response.read().decode()
                self.assertEqual(response.status, 200, payload)
                self.assertNotIn("modelbench-secret", payload)
                self.assertNotIn("modelbench-secret", json.dumps(self.srv.posted))
                conn.request("GET", f"/api/meow/report?port={self.port}&session_id={'a' * 32}")
                response = conn.getresponse()
                report = response.read().decode()
                self.assertEqual(response.status, 200, report)
                self.assertNotIn("modelbench-secret", report)
                conn.close()
        finally:
            app.shutdown()
            app.server_close()

    def test_rejects_nonlocal_port_formats_and_untrusted_identifiers(self):
        for value in ("localhost:1", -1, 65536, True):
            with self.subTest(value=value), self.assertRaises(ValueError):
                meow_bridge.status(value)
        with self.assertRaises(ValueError):
            meow_bridge.report(self.port, "../snapshot")
        with self.assertRaises(ValueError):
            meow_bridge.start(dict(self.config(), endpoint_id="../../bad", expected_max_http_attempts=4))


if __name__ == "__main__":
    unittest.main()
