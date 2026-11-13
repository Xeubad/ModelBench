"""用户参考库导入、校验与固定版本分析，全部使用临时目录。"""
import base64
import copy
import hashlib
import http.client
import json
import os
import random
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from unittest import mock

from model_test import banks, modeltrace, server, store
from tests.test_anthropic import AnthropicFixture


class ReferenceBankTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = modeltrace.load_bank()["bank"]

    def modified(self):
        bank = copy.deepcopy(self.original)
        for item in bank["models"]:
            item["display_name"] = "本地-" + item["display_name"]
        bank["source_scope"] = "user supplied fixture"
        bank["calibration"]["3"]["fallback"] = True
        return bank

    def test_import_and_analyze_uses_selected_immutable_bank(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = json.dumps(self.modified(), ensure_ascii=False).encode()
            imported = banks.import_bank(directory, raw, [])
            self.assertEqual(imported["id"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(imported["source"], "local")
            self.assertEqual(len(banks.list_banks(directory)), 2)  # 内置 + 本地
            artifact = banks.load_bank(directory, imported["id"])
            outputs = []
            for i in range(3):
                rng = random.Random(i)
                outputs.append({"text": ",".join(str(rng.randint(1, 355)) for _ in range(300)), "expected_count": 300})
            result = modeltrace.analyze_outputs(outputs, artifact=artifact)
            self.assertEqual(result["status"], "ranked")
            self.assertTrue(result["results"][0]["display_name"].startswith("本地-"))
            self.assertEqual(result["bank_sha256"], imported["id"])
            self.assertEqual(modeltrace.load_bank()["sha256"], modeltrace.BANK_SHA256)
            self.assertEqual(os.stat(banks.bank_path(directory, imported["id"])).st_mode & 0o777, 0o600)

    def test_import_supports_long_bedrock_inference_profile_labels(self):
        bank = self.modified()
        label = "aws/arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + "a" * 180
        bank["models"][0]["id"] = label
        bank["models"][0]["display_name"] = label
        bank["robust"]["model_order"][0] = label
        with tempfile.TemporaryDirectory() as directory:
            imported = banks.import_bank(directory, json.dumps(bank).encode(), [])
            self.assertEqual(banks.load_bank(directory, imported["id"])["bank"]["models"][0]["id"], label)

    def test_rejects_malformed_vectors_or_missing_calibration(self):
        bad = self.modified()
        bad["robust"]["hellinger"]["feature_scale"][0] = 0
        bad2 = self.modified()
        bad2["calibration"].pop("3")
        for bank in (bad, bad2):
            with self.subTest(case="invalid"), tempfile.TemporaryDirectory() as directory:
                with self.assertRaises(ValueError):
                    banks.import_bank(directory, json.dumps(bank).encode(), [])
                self.assertEqual(banks.list_banks(directory)[0]["id"], "builtin")

    def test_rejects_credentials_and_untrusted_path(self):
        bank = self.modified()
        bank["source_scope"] = "fixture-sensitive-key"
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                banks.import_bank(directory, json.dumps(bank).encode(), ["fixture-sensitive-key"])
            encoded = json.dumps(bank).replace("fixture-sensitive-key", r"\u0066ixture-sensitive-key").encode()
            with self.assertRaises(ValueError):
                banks.import_bank(directory, encoded, ["fixture-sensitive-key"])
            with self.assertRaises(ValueError):
                banks.load_bank(directory, "../../example")

    def test_web_import_lists_and_analyzes_with_custom_bank(self):
        app = ThreadingHTTPServer(("127.0.0.1", 0), server.Handler)
        app.daemon_threads = True
        threading.Thread(target=app.serve_forever, daemon=True).start()
        port = app.server_address[1]
        try:
            with tempfile.TemporaryDirectory() as directory, mock.patch.object(store, "DATA_DIR", directory), \
                 mock.patch.object(server, "PORT", port), mock.patch.object(server, "BIND_HOSTS", ["127.0.0.1"]):
                conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                original = b"\xef\xbb\xbf" + (json.dumps(self.modified(), ensure_ascii=False, indent=2) + "\n").encode()
                body = json.dumps({"bank_base64": base64.b64encode(original).decode("ascii")}).encode()
                conn.request("POST", "/api/modeltrace/banks", body, {"Content-Type": "application/json"})
                resp = conn.getresponse()
                payload = json.loads(resp.read())
                self.assertEqual(resp.status, 200, payload)
                bank_id = payload["bank"]["id"]
                self.assertEqual(bank_id, hashlib.sha256(original).hexdigest(), "必须冻结原文件字节的哈希")
                conn.request("GET", "/api/modeltrace/banks")
                resp = conn.getresponse()
                self.assertEqual(len(json.loads(resp.read())["banks"]), 2)
                outputs = []
                for i in range(3):
                    rng = random.Random(i)
                    outputs.append({"text": ",".join(str(rng.randint(1, 355)) for _ in range(300)), "expected_count": 300})
                conn.request("POST", "/api/modeltrace/analyze", json.dumps({"outputs": outputs, "bank_id": bank_id}), {"Content-Type": "application/json"})
                resp = conn.getresponse()
                result = json.loads(resp.read())
                self.assertEqual(resp.status, 200, result)
                self.assertEqual(result["bank_sha256"], bank_id)
                self.assertEqual(result["bank_source"], "local")
                self.assertTrue(result["results"][0]["display_name"].startswith("本地-"))
                conn.close()
        finally:
            app.shutdown()
            app.server_close()

    def test_run_freezes_selected_bank_identity_and_result(self):
        upstream = ThreadingHTTPServer(("127.0.0.1", 0), AnthropicFixture)
        upstream.daemon_threads = True
        upstream.requests = []
        threading.Thread(target=upstream.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), \
                 mock.patch.object(store, "RESULTS_DIR", root + "/results"):
                bank_id = banks.import_bank(store.DATA_DIR, json.dumps(self.modified()).encode(), [])["id"]
                ep = store.upsert_endpoint({"name": "fake", "base_url": f"http://127.0.0.1:{upstream.server_address[1]}",
                                            "api_key": "fixture-key", "api": "anthropic", "models": ["modeltrace"]})
                params = server.validate({"selection": [{"endpoint": ep["id"], "models": ["modeltrace"]}],
                                          "scenarios": ["modeltrace"], "rounds": 1, "timeout": 10, "context": 0,
                                          "gap": 0, "warmup": False, "effort": "", "name": "", "modeltrace_bank": bank_id})
                result = store.execute(params, lambda _: None, threading.Event())
                saved = store.load_run(result["id"])
                self.assertEqual(saved["env"]["modeltrace"]["bank_sha256"], bank_id)
                self.assertEqual(saved["modeltrace"][0]["bank_sha256"], bank_id)
                self.assertTrue(saved["modeltrace"][0]["results"][0]["display_name"].startswith("本地-"))
                with open(os.path.join(store.run_dir(result["id"]), "report.md"), encoding="utf-8") as handle:
                    report = handle.read()
                    self.assertIn(bank_id, report)
                    self.assertIn("未校准", report)
        finally:
            upstream.shutdown()
            upstream.server_close()

    def test_tampered_custom_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            raw = json.dumps(self.modified()).encode()
            bank_id = banks.import_bank(directory, raw, [])["id"]
            with open(banks.bank_path(directory, bank_id), "w", encoding="utf-8") as handle:
                handle.write("{}")
            with self.assertRaises(ValueError):
                banks.load_bank(directory, bank_id)


if __name__ == "__main__":
    unittest.main()
