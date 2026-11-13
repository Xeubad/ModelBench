"""Amazon Bedrock ConverseStream/SigV4：仅使用本机假服务和合成凭据。"""
import datetime
import json
import os
import struct
import tempfile
import threading
import unittest
import urllib.parse
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock

from model_test import bedrock, core, server, store

ACCESS = "AKIDEXAMPLE"
SECRET = "SECRETKEYEXAMPLE"
TOKEN = "SESSIONEXAMPLE"
MODEL = "anthropic.claude-v2:0"


def event_frame(kind, value, message_type="event"):
    headers = b""
    for name, text in ((":message-type", message_type), (":event-type" if message_type == "event" else ":exception-type", kind), (":content-type", "application/json")):
        name, text = name.encode(), text.encode()
        headers += bytes([len(name)]) + name + bytes([7]) + struct.pack(">H", len(text)) + text
    payload = json.dumps(value, ensure_ascii=False).encode()
    head = struct.pack(">II", 16 + len(headers) + len(payload), len(headers))
    prelude = head + struct.pack(">I", zlib.crc32(head) & 0xffffffff)
    raw = prelude + headers + payload
    return raw + struct.pack(">I", zlib.crc32(raw) & 0xffffffff)


class BedrockFixture(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *_):
        pass

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def reply(self, status, raw, content_type="application/json"):
        if not isinstance(raw, bytes):
            raw = json.dumps(raw, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        self.server.calls.append(("GET", self.path, dict(self.headers), None))
        auth = self.headers.get("Authorization", "")
        if self.path != "/foundation-models" or not auth.startswith("AWS4-HMAC-SHA256 "):
            return self.reply(403, {"message": "invalid auth"})
        self.reply(200, {"modelSummaries": [
            {"modelId": MODEL, "outputModalities": ["TEXT"], "responseStreamingSupported": True},
            {"modelId": "image-only", "outputModalities": ["IMAGE"], "responseStreamingSupported": True},
            {"modelId": "no-stream", "outputModalities": ["TEXT"], "responseStreamingSupported": False}]})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.calls.append(("POST", self.path, dict(self.headers), body))
        auth = self.headers.get("Authorization", "")
        parsed = urllib.parse.urlsplit(self.path)
        if not parsed.path.startswith("/model/") or not parsed.path.endswith("/converse-stream") or not auth.startswith("AWS4-HMAC-SHA256 ") or self.headers.get("X-Amz-Security-Token") != TOKEN or "model" in body:
            return self.reply(400, {"message": "invalid wire format"})
        model = urllib.parse.unquote(parsed.path[len("/model/"):-len("/converse-stream")])
        if model == "error-echo":
            return self.reply(403, {"message": "credential " + TOKEN})
        if model == "nonstream":
            return self.reply(200, {"output": {"message": {"role": "assistant", "content": [{"text": "好"}]}}, "stopReason": "end_turn", "usage": {"inputTokens": 11, "outputTokens": 4}})
        frames = [event_frame("messageStart", {"role": "assistant"})]
        if model in ("tool", "tool-no-signature") and len(body["messages"]) == 1:
            frames.append(event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"text": "思考"}}}))
            if model == "tool":
                frames.append(event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"reasoningContent": {"signature": "signed-thinking"}}}))
            frames += [event_frame("contentBlockStop", {"contentBlockIndex": 0}),
                       event_frame("contentBlockStart", {"contentBlockIndex": 1, "start": {"toolUse": {"toolUseId": "call-1", "name": "lookup_price"}}}),
                       event_frame("contentBlockDelta", {"contentBlockIndex": 1, "delta": {"toolUse": {"input": '{"item":"MET混合滤料包"}'}}}),
                       event_frame("contentBlockStop", {"contentBlockIndex": 1}),
                       event_frame("messageStop", {"stopReason": "tool_use"})]
        else:
            if model in ("tool", "tool-no-signature"):
                blocks = body["messages"][-2]["content"]
                tool_result = body["messages"][-1]["content"][0]["toolResult"]
                thinking = blocks[0].get("reasoningContent", {}).get("reasoningText", {})
                valid_signature = thinking.get("signature") == "signed-thinking" if model == "tool" else "signature" not in thinking
                if not valid_signature or tool_result.get("toolUseId") != "call-1":
                    return self.reply(400, {"message": "reasoning or tool replay missing"})
                value = tool_result["content"][0]["json"]
                text = json.dumps({"price": value["price"], "unit": value["unit"]}, ensure_ascii=False)
            elif model == "echo":
                text = SECRET
            else:
                text = "好"
            for chunk in ([text[:8], text[8:]] if model == "echo" else [text]):
                frames.append(event_frame("contentBlockDelta", {"contentBlockIndex": 0, "delta": {"text": chunk}}))
            frames.append(event_frame("contentBlockStop", {"contentBlockIndex": 0}))
            if model != "missing-stop":
                frames.append(event_frame("messageStop", {"stopReason": "max_tokens" if model == "truncated" else "end_turn"}))
        if model == "exception":
            frames = [event_frame("modelStreamErrorException", {"message": "temporary"}, "exception")]
        frames.append(event_frame("metadata", {"usage": {"inputTokens": 11, "outputTokens": 4, "cacheReadInputTokens": 2}}))
        raw = b"".join(frames)
        if model == "bad-crc":
            raw = raw[:-1] + bytes([raw[-1] ^ 0xff])
        if model == "cut-wire":
            raw = raw[:-5]
        self.reply(200, raw, "application/vnd.amazon.eventstream")


class BedrockTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = ThreadingHTTPServer(("127.0.0.1", 0), BedrockFixture)
        cls.srv.daemon_threads = True
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.ep = {"name": "aws", "base_url": f"http://127.0.0.1:{cls.srv.server_address[1]}", "api": "bedrock", "api_key": "",
                  "aws_region": "us-east-1", "aws_profile": ""}

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def setUp(self):
        self.srv.calls = []
        self.env = mock.patch.dict(os.environ, {"AWS_ACCESS_KEY_ID": ACCESS, "AWS_SECRET_ACCESS_KEY": SECRET, "AWS_SESSION_TOKEN": TOKEN}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_sigv4_matches_independent_aws_sdk_fixed_vectors(self):
        now = datetime.datetime(2024, 1, 2, 3, 4, 5, tzinfo=datetime.timezone.utc)
        creds = {"access_key": ACCESS, "secret_key": SECRET, "session_token": TOKEN}
        samples = [("POST", "https://bedrock-runtime.us-east-1.amazonaws.com/model/anthropic.claude-v2%3A0/converse-stream", b'{"messages":[]}', "e79223576f5c27b2179f0712d4d41bb7517fd773129e3ac98f749c92a400eba0"),
                   ("GET", "https://bedrock.us-east-1.amazonaws.com/foundation-models?byOutputModality=TEXT&byProvider=Anthropic", b"", "5418bb169edcffe69e7dba01317600471a136ca8bd42d1358805b0f5716fc3d3")]
        for method, url, payload, expected in samples:
            with self.subTest(method=method):
                headers = bedrock.sign_headers(method, url, payload, creds, "us-east-1", now=now)
                self.assertEqual(headers["Authorization"].split("Signature=")[1], expected)
                self.assertEqual(headers["X-Amz-Security-Token"], TOKEN)

    def request(self, model=MODEL):
        pool = core.Pool()
        try:
            return core.stream_request(pool, self.ep, core.make_body("bedrock", model, "hi", system="system"), 5)
        finally:
            pool.close()

    def test_foundation_models_discovery_and_signing(self):
        self.assertEqual(core.fetch_models(self.ep, 5), [MODEL])
        method, path, headers, _ = self.srv.calls[0]
        self.assertEqual((method, path), ("GET", "/foundation-models"))
        self.assertIn("Credential=" + ACCESS + "/", headers["Authorization"])
        self.assertEqual(headers["X-Amz-Security-Token"], TOKEN)

    def test_binary_stream_usage_and_system(self):
        result = self.request()
        self.assertTrue(result["ok"], result["error"])
        self.assertEqual(result["text"], "好")
        self.assertEqual(result["usage"]["output_tokens"], 4)
        self.assertEqual(result["usage"]["cached_tokens"], 2)
        self.assertIsNotNone(result["first_token_s"])
        path, headers, body = self.srv.calls[0][1:]
        self.assertIn("anthropic.claude-v2%3A0", path)
        self.assertEqual(body["system"], [{"text": "system"}])
        self.assertNotIn(SECRET, json.dumps(core.public_rec(result)))
        self.assertNotIn(TOKEN, json.dumps(result["events"]))

    def test_tool_call_replays_reasoning_signature(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 123, 1)
            result = core.run_task(pool, self.ep, "tool", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            self.assertEqual(len(self.srv.calls), 2)
            self.assertEqual(self.srv.calls[0][3]["toolConfig"]["tools"][0]["toolSpec"]["name"], "lookup_price")
        finally:
            pool.close()

    def test_tool_replay_does_not_invent_missing_reasoning_signature(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["tool"])[0], 123, 1)
            result = core.run_task(pool, self.ep, "tool-no-signature", task, 5)
            self.assertTrue(result["pass"], result.get("grade_note"))
            thinking = self.srv.calls[-1][3]["messages"][-2]["content"][0]["reasoningContent"]["reasoningText"]
            self.assertNotIn("signature", thinking)
        finally:
            pool.close()

    def test_nonstream_and_corrupt_or_incomplete_streams(self):
        result = self.request("nonstream")
        self.assertTrue(result["ok"], result["error"])
        self.assertTrue(result["non_stream"])
        for model in ("bad-crc", "cut-wire", "missing-stop", "truncated", "exception"):
            with self.subTest(model=model):
                result = self.request(model)
                self.assertFalse(result["ok"])
                self.assertTrue(result["error"])

    def test_echoed_aws_secret_is_removed_from_record_and_stream(self):
        result = self.request("echo")
        self.assertFalse(result["ok"])
        self.assertNotIn(SECRET, json.dumps(core.public_rec(result), ensure_ascii=False))
        self.assertNotIn(SECRET, json.dumps(result["events"], ensure_ascii=False))

    def test_aws_session_token_echoed_in_error_is_redacted(self):
        result = self.request("error-echo")
        self.assertFalse(result["ok"])
        self.assertNotIn(TOKEN, json.dumps(core.public_rec(result), ensure_ascii=False))
        self.assertIn("隐藏", result["error"])

    def test_profile_credentials_and_missing_profile_are_handled_locally(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "credentials")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("[fixture]\naws_access_key_id=" + ACCESS + "\naws_secret_access_key=" + SECRET + "\naws_session_token=" + TOKEN + "\n")
            with mock.patch.dict(os.environ, {"AWS_SHARED_CREDENTIALS_FILE": path}, clear=True):
                credentials = bedrock.resolve_credentials("fixture")
                self.assertEqual(credentials["secret_key"], SECRET)
                self.assertEqual(credentials["source"], "profile:fixture")
                with self.assertRaises(ValueError):
                    bedrock.resolve_credentials("missing")

    def test_stored_endpoint_uses_aws_environment_without_exposing_credentials(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), mock.patch.object(store, "RESULTS_DIR", root + "/results"):
            endpoint = store.upsert_endpoint({**self.ep, "models": [MODEL]})
            self.assertNotIn(ACCESS, json.dumps(endpoint))
            params = server.validate({"selection": [{"endpoint": endpoint["id"], "models": [MODEL]}], "scenarios": ["speed"],
                                      "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False})
            result = store.execute(params, lambda _: None, threading.Event())
            self.assertTrue(result["records"][0]["ok"], result["records"][0]["error"])
            text = json.dumps(result, ensure_ascii=False)
            for secret in (ACCESS, SECRET, TOKEN):
                self.assertNotIn(secret, text)
            self.assertEqual(result["env"]["endpoints"][0]["aws_region"], "us-east-1")
            self.assertEqual(result["env"]["endpoints"][0]["aws_credential_source"], "environment")
            with self.assertRaisesRegex(ValueError, "密钥"):
                store.execute(dict(params, prompt="accidental " + SECRET), lambda _: None, threading.Event())

    def test_aws_secret_in_model_id_is_rejected_before_network(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root + "/data"), mock.patch.object(store, "RESULTS_DIR", root + "/results"):
            endpoint = store.upsert_endpoint({**self.ep, "models": [SECRET]})
            params = server.validate({"selection": [{"endpoint": endpoint["id"], "models": [SECRET]}], "scenarios": ["speed"],
                                      "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False})
            with self.assertRaisesRegex(ValueError, "AWS 凭据"):
                store.execute(params, lambda _: None, threading.Event())
            self.assertEqual(self.srv.calls, [])
            self.assertFalse(os.path.exists(store.RESULTS_DIR))

    def test_credential_source_participates_in_service_identity(self):
        endpoint = {"name": "aws", "base_url": self.ep["base_url"], "api": "bedrock", "aws_region": "us-east-1", "aws_profile": ""}
        env_a = {"endpoints": [dict(endpoint, aws_credential_source="environment")]}
        env_b = {"endpoints": [dict(endpoint, aws_credential_source="profile:fixture")]}
        self.assertNotEqual(core.service_of(env_a, "aws/model"), core.service_of(env_b, "aws/model"))
        self.assertIsNone(core.service_of({"endpoints": [endpoint]}, "aws/model"))

    def test_distribution_uses_provider_default_temperature(self):
        pool = core.Pool()
        try:
            task = core.materialize_task(core.build_tasks(["modeltrace"])[0], 1, 1)
            core.run_task(pool, self.ep, MODEL, task, 5)
            self.assertNotIn("temperature", self.srv.calls[-1][3]["inferenceConfig"])
        finally:
            pool.close()

    def test_long_inference_profile_arn_can_be_saved_and_encoded(self):
        model = "arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + "a" * 130
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({**self.ep, "models": [model]})
            params = server.validate({"selection": [{"endpoint": endpoint["id"], "models": [model]}], "scenarios": ["speed"],
                                      "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False})
            self.assertEqual(params["selection"][0]["models"], [model])
            self.assertIn("inference-profile%2F", bedrock.request_path(self.ep["base_url"], model))
            self.assertEqual(self.srv.calls, [])

    def test_official_endpoint_region_mismatch_is_rejected(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            with self.assertRaisesRegex(ValueError, "区域"):
                store.upsert_endpoint({"name": "aws", "base_url": "https://bedrock-runtime.us-east-1.amazonaws.com", "api": "bedrock",
                                       "aws_region": "eu-west-1", "models": [MODEL]})

    def test_aws_credentials_require_https_except_loopback_fixtures(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            with self.assertRaisesRegex(ValueError, "HTTPS"):
                store.upsert_endpoint({"name": "aws", "base_url": "http://192.0.2.10", "api": "bedrock", "aws_region": "us-east-1"})
            endpoint = store.upsert_endpoint({**self.ep, "models": [MODEL]})
            self.assertEqual(endpoint["api"], "bedrock")

    def test_effort_override_is_rejected_before_request(self):
        with tempfile.TemporaryDirectory() as root, mock.patch.object(store, "DATA_DIR", root):
            endpoint = store.upsert_endpoint({**self.ep, "models": [MODEL]})
            with self.assertRaisesRegex(ValueError, "推理强度"):
                server.validate({"selection": [{"endpoint": endpoint["id"], "models": [MODEL]}], "scenarios": ["speed"],
                                 "prompt": "hi", "rounds": 1, "timeout": 10, "context": 0, "gap": 0, "warmup": False, "effort": "high"})
            self.assertEqual(self.srv.calls, [])
