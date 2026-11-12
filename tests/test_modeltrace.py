"""ModelTrace 基准和评分回归，不调用任何真实模型。"""
import hashlib
import math
import os
import tempfile
import unittest
from unittest import mock

from model_test import core
from model_test import modeltrace


class ModelTraceTest(unittest.TestCase):
    def test_bank_is_pinned_and_has_claude_candidates(self):
        bank = modeltrace.load_bank()
        self.assertEqual(bank["sha256"], "f267d0513aa06981d024ec5394746fceddf44e4e8a75fa6bf8304255f4b32d6d")
        models = bank["bank"]["models"]
        self.assertEqual(len(models), 18)
        self.assertTrue(any(m.get("family") == "claude" for m in models))
        self.assertEqual(bank["bank"]["built_at"], "2026-10-09T06:24:27.469312+00:00")

    def test_tampered_reference_bank_is_rejected(self):
        with tempfile.TemporaryDirectory() as path:
            damaged = os.path.join(path, "bank.json")
            with open(damaged, "w", encoding="utf-8") as handle:
                handle.write("{}")
            with mock.patch.object(modeltrace, "BANK_FILE", damaged):
                with self.assertRaisesRegex(ValueError, "校验失败"):
                    modeltrace.load_bank.__wrapped__()

    def test_scoring_matches_upstream_javascript_fixed_vector(self):
        # 固定输入和预期由 ModelTrace static/fingerprint-core.js @ 6cd4a8f 独立计算。
        outputs = [{"text": ",".join(str(int.from_bytes(hashlib.sha256(f"{k}:{i}".encode()).digest()[:4], "big") % 355 + 1) for i in range(300)),
                    "expected_count": 300} for k in range(3)]
        result = modeltrace.analyze_outputs(outputs)
        self.assertEqual(result["status"], "ranked")
        self.assertEqual(result["used_outputs"], 3)
        self.assertEqual(result["results"][0]["model"], "gpt-5.4")
        self.assertAlmostEqual(result["results"][0]["probability"], 0.6456373362910393, places=7)
        self.assertAlmostEqual(result["results"][1]["probability"], 0.3543037959067897, places=7)
        self.assertAlmostEqual(result["family_probabilities"][0]["probability"], 0.6456386143724809, places=7)
        self.assertAlmostEqual(sum(m["probability"] for m in result["results"]), 1, places=10)

    def test_insufficient_or_failed_outputs_do_not_produce_ranking(self):
        sequence = ",".join(str(i % 355 + 1) for i in range(300))
        for outputs in ([{"text": sequence, "expected_count": 300}],
                        [{"text": sequence, "expected_count": 300}] * 2 + [{"text": sequence, "expected_count": 300, "ok": False}],
                        [{"text": "拒绝", "expected_count": 300}] * 3):
            result = modeltrace.analyze_outputs(outputs)
            self.assertEqual(result["status"], "insufficient")
            self.assertEqual(result["results"], [])
            self.assertFalse(result.get("prediction"))

    def test_three_identical_sequences_are_not_treated_as_independent_evidence(self):
        text = ",".join(str((i * 17) % 355 + 1) for i in range(300))
        result = modeltrace.analyze_outputs([{"text": text, "expected_count": 300}] * 3)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "duplicate_outputs")
        self.assertEqual(result["results"], [])

    def test_obvious_arithmetic_sequences_are_not_ranked(self):
        outputs = [{"text": ",".join(str((i * 17 + k * 31) % 355 + 1) for i in range(300)),
                    "expected_count": 300} for k in range(3)]
        result = modeltrace.analyze_outputs(outputs)
        self.assertEqual(result["status"], "insufficient")
        self.assertEqual(result["reason"], "patterned_output")
        self.assertEqual(result["results"], [])

    def test_number_parser_ignores_prompt_echo_and_checks_length(self):
        self.assertEqual(modeltrace.parse_numbers("范围 1 至 355。答案：10, 20, 30, 40"), [10, 20, 30, 40])
        # 与上游一致：只有标点分隔时，范围末尾的 355 会混入；不能声称能完全排除回显。
        self.assertEqual(modeltrace.parse_numbers("范围 1 至 355。10, 20, 30, 40"), [355, 10, 20, 30, 40])
        sequence = ",".join(["7"] * 164)
        result = modeltrace.analyze_outputs([{"text": sequence, "expected_count": 300}] * 3)
        self.assertEqual(result["status"], "insufficient")  # 至少 ceil(300*0.55)=165
        self.assertEqual(result["diagnostics"][0]["parsed_numbers"], 164)

    def test_challenges_vary_by_round_but_match_same_seed(self):
        tasks = core.build_tasks(["modeltrace"])
        self.assertEqual(len(tasks), 3)
        a = [core.materialize_task(t, 7, 1) for t in tasks]
        b = [core.materialize_task(t, 7, 1) for t in tasks]
        c = [core.materialize_task(t, 7, 2) for t in tasks]
        self.assertEqual(a, b)
        self.assertNotEqual([t["prompt"] for t in a], [t["prompt"] for t in c])
        self.assertEqual(len({t["expected_count"] for t in a}), 3)
        self.assertTrue(all(292 <= t["expected_count"] <= 332 for t in a))

    def test_modeltrace_condition_includes_seed_and_connection_mode(self):
        tasks = core.build_tasks(["modeltrace"])
        plan = {"endpoints": {"e": {"name": "test", "base_url": "http://127.0.0.1", "api": "anthropic"}},
                "models": [("e", "claude")], "tasks": tasks, "rounds": 1, "timeout": 10, "gap": 0,
                "warmup": False, "effort": "", "seed": 123, "concurrency": 1, "connection_mode": "reuse"}
        env = core.make_env(plan, {}, {"created": "2026-01-01", "prompt": "", "context": 0})
        key = core.cond_of(env, "modeltrace")
        self.assertIsNotNone(key)
        self.assertNotEqual(key, core.cond_of(dict(env, seed=124), "modeltrace"))
        self.assertNotEqual(key, core.cond_of(dict(env, connection_mode="cold"), "modeltrace"))
        missing = dict(env)
        missing.pop("seed")
        self.assertIsNone(core.cond_of(missing, "modeltrace"))

    def test_same_round_reference_comparison_uses_the_same_challenges(self):
        import random
        tasks = [core.materialize_task(t, 9, 1) for t in core.build_tasks(["modeltrace"])]
        records = []
        for label in ("official/claude", "relay/claude"):
            for i, task in enumerate(tasks):
                rng = random.Random(i)
                text = ",".join(str(rng.randint(1, 355)) for _ in range(task["expected_count"]))
                records.append({"scenario": "modeltrace", "label": label, "model": "claude", "round": 1,
                                "task": task["id"], "probe_prompt": task["prompt"], "ok": True,
                                "text": text, "expected_count": task["expected_count"]})
        pairs = modeltrace.compare_records(records)
        self.assertEqual(len(pairs), 1)
        self.assertEqual(pairs[0]["status"], "descriptive")
        self.assertAlmostEqual(pairs[0]["distance"], 0)
        records[-1]["probe_prompt"] = "不同的题目"
        self.assertEqual(modeltrace.compare_records(records)[0]["status"], "incompatible")
        records[-1]["ok"] = False
        self.assertEqual(modeltrace.compare_records(records)[0]["status"], "insufficient")

    def test_grouping_never_combines_partial_or_different_rounds(self):
        text = ",".join(str(i % 355 + 1) for i in range(300))
        records = [dict(label="target/claude", model="claude", scenario="modeltrace", round=1, task=f"mt{i}",
                        ok=True, text=text, expected_count=300) for i in (1, 2)]
        records.append(dict(label="target/claude", model="claude", scenario="modeltrace", round=2, task="mt3",
                            ok=True, text=text, expected_count=300))
        result = modeltrace.analyze_records(records)
        self.assertEqual(len(result), 2)
        self.assertTrue(all(r["status"] == "insufficient" for r in result))


if __name__ == "__main__":
    unittest.main()
