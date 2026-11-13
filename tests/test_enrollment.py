"""ModelTrace 基准采集与可追溯导出（本机假服务）。"""
import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from model_test import core, server, store
from tests.test_anthropic import AnthropicFixture


class EnrollmentTest(unittest.TestCase):
    def test_suite_has_matching_twelve_environments(self):
        tasks = core.build_tasks(["modeltrace_enroll"])
        self.assertEqual(len(tasks), 36)
        self.assertEqual(len({t["condition_id"] for t in tasks}), 12)
        self.assertEqual(len({t["id"] for t in tasks}), 36)
        self.assertTrue(any(t.get("system") for t in tasks))
        self.assertTrue(any(t.get("transport") == "user" for t in tasks))

    def test_collect_and_export_preserves_evidence_and_excludes_key(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), AnthropicFixture)
        upstream.daemon_threads = True
        upstream.requests = []
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), \
                 mock.patch.object(store, "RESULTS_DIR", root + "/results"):
                ep = store.upsert_endpoint({"name": "fixture", "base_url": f"http://127.0.0.1:{upstream.server_address[1]}",
                                            "api_key": "fixture-key", "api": "anthropic", "models": ["modeltrace"]})
                params = server.validate({"selection": [{"endpoint": ep["id"], "models": ["modeltrace"]}],
                    "scenarios": ["modeltrace_enroll"], "rounds": 1, "timeout": 10, "context": 0, "gap": 0,
                    "warmup": False, "effort": "", "name": ""})
                result = store.execute(params, lambda _: None, threading.Event())
                exported = store.export_reference_rows(result["id"])
                self.assertEqual(len(exported), 36)
                self.assertIn("服务商默认", core.cond_label(result["env"], "modeltrace_enroll"))
                self.assertIn("普通场景", result["env"]["note"])
                self.assertTrue(all(row["strict_valid"] for row in exported))
                self.assertEqual(len({r["condition_id"] for r in exported}), 12)
                self.assertEqual(len({r["challenge_id"] for r in exported}), 36)
                self.assertTrue(all(row["run_id"] == result["id"] for row in exported))
                self.assertTrue(all(row["system_sha256"] for row in exported))
                self.assertNotIn("fixture-key", json.dumps(exported))
                self.assertEqual(len(upstream.requests), 36)
                self.assertTrue(all("temperature" not in call[1] for call in upstream.requests))
                with self.assertRaises(ValueError):
                    server.validate(dict(params, rounds=2))
        finally:
            upstream.shutdown()
            upstream.server_close()


if __name__ == "__main__":
    unittest.main()
