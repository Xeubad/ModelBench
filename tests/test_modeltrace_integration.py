"""从 Anthropic 本机假接口到结果文件、手动分析接口的集成测试。"""
import http.client
import json
import os
import random
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store
from tests.test_anthropic import AnthropicFixture


class ModelTraceIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), AnthropicFixture)
        cls.srv.daemon_threads = True
        cls.srv.requests = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_claude_run_saves_ranked_results_and_does_not_set_temperature(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            ep = store.upsert_endpoint({"name": "fixture", "base_url": f"http://127.0.0.1:{self.srv.server_address[1]}",
                                        "api_key": "fixture-key", "api": "anthropic", "models": ["modeltrace"]})
            params = server.validate({"selection": [{"endpoint": ep["id"], "models": ["modeltrace"]}],
                                      "scenarios": ["modeltrace"], "prompt": "", "rounds": 1, "timeout": 10,
                                      "context": 0, "gap": 0, "warmup": False, "effort": "", "name": ""})
            result = store.execute(params, lambda _: None, threading.Event())
            saved = store.load_run(result["id"])
            self.assertEqual(len(saved["records"]), 3)
            self.assertEqual(len(saved["modeltrace"]), 1)
            self.assertEqual(saved["modeltrace"][0]["status"], "ranked")
            self.assertEqual(saved["modeltrace"][0]["used_outputs"], 3)
            self.assertEqual(saved["env"]["modeltrace"]["temperature"], "provider_default")
            requests = [body for _, body, _ in self.srv.requests if body.get("model") == "modeltrace"][-3:]
            self.assertEqual(len(requests), 3)
            self.assertTrue(all("temperature" not in body for body in requests))
            with open(os.path.join(store.run_dir(result["id"]), "result.json"), encoding="utf-8") as handle:
                self.assertNotIn("fixture-key", handle.read())
            with open(os.path.join(store.run_dir(result["id"]), "report.md"), encoding="utf-8") as handle:
                report = handle.read()
            self.assertIn("ModelTrace", report)
            self.assertIn("第三方", report)

    def test_automatic_same_round_reference_comparison_is_saved(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            endpoints = [store.upsert_endpoint({"name": name, "base_url": f"http://127.0.0.1:{self.srv.server_address[1]}",
                                                "api_key": "fixture-key", "api": "anthropic", "models": ["modeltrace"]})
                         for name in ("官方参考", "待测线路")]
            params = server.validate({"selection": [{"endpoint": ep["id"], "models": ["modeltrace"]} for ep in endpoints],
                                      "scenarios": ["modeltrace"], "rounds": 1, "timeout": 10, "context": 0,
                                      "gap": 0, "warmup": False, "effort": "", "name": ""})
            result = store.execute(params, lambda _: None, threading.Event())
            saved = store.load_run(result["id"])
            self.assertEqual(len(saved["modeltrace_pairs"]), 1)
            self.assertEqual(saved["modeltrace_pairs"][0]["status"], "descriptive")
            self.assertAlmostEqual(saved["modeltrace_pairs"][0]["distance"], 0)

    def test_behavior_run_saves_observed_distributions_without_identity_claims(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path + "/data"), \
             mock.patch.object(store, "RESULTS_DIR", path + "/results"):
            ep = store.upsert_endpoint({"name": "fixture", "base_url": f"http://127.0.0.1:{self.srv.server_address[1]}",
                                        "api_key": "fixture-key", "api": "anthropic", "models": ["behavior-a", "behavior-b"]})
            params = server.validate({"selection": [{"endpoint": ep["id"], "models": ["behavior-a", "behavior-b"]}],
                                      "scenarios": ["behavior"], "rounds": 5, "timeout": 10, "context": 0,
                                      "gap": 0, "warmup": False, "effort": "", "name": ""})
            result = store.execute(params, lambda _: None, threading.Event())
            saved = store.load_run(result["id"])
            self.assertEqual(len(saved["records"]), 40)
            self.assertEqual(saved["behavior"]["pairs"][0]["status"], "descriptive")
            self.assertAlmostEqual(saved["behavior"]["pairs"][0]["mean_jsd"], 1)
            with open(os.path.join(store.run_dir(result["id"]), "report.md"), encoding="utf-8") as handle:
                report = handle.read()
            self.assertIn("alpha", report)
            self.assertIn("beta", report)
            self.assertIn("5/5", report)
            requests = [body for _, body, _ in self.srv.requests if body.get("model") in ("behavior-a", "behavior-b")]
            self.assertTrue(all("temperature" not in body for body in requests))

    def test_manual_challenges_and_analyze_api(self):
        srv = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        srv.daemon_threads = True
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        port = srv.server_address[1]
        try:
            with mock.patch.object(server, "PORT", port), mock.patch.object(server, "BIND_HOSTS", ["127.0.0.1"]):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                conn.request("GET", "/api/modeltrace/challenges")
                response = conn.getresponse()
                challenges = json.loads(response.read())["challenges"]
                self.assertEqual(response.status, 200)
                self.assertEqual(len(challenges), 3)
                self.assertTrue(all(292 <= c["expected_count"] <= 332 for c in challenges))
                outputs = []
                for k, challenge in enumerate(challenges):
                    generator = random.Random(k)
                    outputs.append({"text": ",".join(str(generator.randint(1, 355)) for _ in range(300)),
                                    "expected_count": challenge["expected_count"]})
                conn.request("POST", "/api/modeltrace/analyze", json.dumps({"outputs": outputs}), {"Content-Type": "application/json"})
                response = conn.getresponse()
                result = json.loads(response.read())
                self.assertEqual(response.status, 200)
                self.assertEqual(result["status"], "ranked")
                conn.request("POST", "/api/modeltrace/analyze", json.dumps({"outputs": [{"text": "x"}] * 4}), {"Content-Type": "application/json"})
                response = conn.getresponse()
                response.read()
                self.assertEqual(response.status, 400)
                conn.close()
        finally:
            srv.shutdown()
            srv.server_close()

    def test_reference_bank_is_checked_before_any_test_requests(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "fixture", "base_url": "http://127.0.0.1:8000", "api_key": "fixture-key",
                                        "api": "anthropic", "models": ["claude-test"]})
            with mock.patch("model_test.modeltrace.load_bank", side_effect=ValueError("参考库损坏")):
                with self.assertRaisesRegex(ValueError, "参考库损坏"):
                    server.validate({"selection": [{"endpoint": ep["id"], "models": ["claude-test"]}],
                                     "scenarios": ["modeltrace"], "rounds": 1, "timeout": 10,
                                     "context": 0, "gap": 0, "effort": ""})

    def test_modeltrace_rejects_effort_override_for_reference_compatibility(self):
        with tempfile.TemporaryDirectory() as path, mock.patch.object(store, "DATA_DIR", path):
            ep = store.upsert_endpoint({"name": "fixture", "base_url": "http://127.0.0.1:8000", "api_key": "fixture-key",
                                        "api": "anthropic", "models": ["claude-test"]})
            with self.assertRaises(ValueError):
                server.validate({"selection": [{"endpoint": ep["id"], "models": ["claude-test"]}],
                                 "scenarios": ["modeltrace"], "rounds": 1, "timeout": 10, "context": 0,
                                 "gap": 0, "effort": "high"})


if __name__ == "__main__":
    unittest.main()
