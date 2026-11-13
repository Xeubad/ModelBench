"""可选 NumPy 训练工具的契约；常规标准库测试不安装额外依赖。"""
import hashlib
import importlib.util
import json
import os
import random
import tempfile
import unittest
import warnings

from model_test import banks, modeltrace, modeltrace_suite


@unittest.skipUnless(importlib.util.find_spec("numpy"), "参考库训练需要可选 NumPy")
class ReferenceTrainingTest(unittest.TestCase):
    def rows(self, run_prefix):
        rows = []
        suite = modeltrace_suite.fingerprint_suite()
        for source_index, source in enumerate(("known/a", "known/b")):
            for index, item in enumerate(suite):
                rng = random.Random(source_index * 10000 + index + (100000 if run_prefix == "holdout" else 0))
                prompt = ((item["user_prefix"] + "\n\nFinal task:\n") if item["user_prefix"] else "") + item["prompt"]
                rows.append({"source": source, "run_id": run_prefix + source, "condition_id": item["condition"],
                             "nuisance_condition_id": item["condition"], "challenge_id": item["challenge_id"],
                             "requested_count": item["expected_count"], "strict_valid": True, "prompt": prompt,
                             "system_sha256": hashlib.sha256(item["system"].encode()).hexdigest(),
                             "suite_commit": modeltrace.UPSTREAM_COMMIT, "text": ",".join(str(rng.randint(1, 355)) for _ in range(item["expected_count"]))})
        return rows

    def test_train_validate_and_heldout_evaluation(self):
        from model_test import modeltrace_training
        training, holdout = self.rows("train"), self.rows("holdout")
        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)
            bank = modeltrace_training.train_bank(training, holdout)
        banks.validate_bank(bank)
        self.assertEqual(bank["provenance"]["training_rows"], 72)
        self.assertEqual(bank["evaluation"]["known_groups"], 24)
        self.assertTrue(bank["evaluation"]["independent_run_ids"])
        self.assertEqual(len(bank["provenance"]["training_sha256"]), 64)
        self.assertNotIn("text", json.dumps(bank["provenance"]))
        with tempfile.TemporaryDirectory() as directory:
            imported = banks.import_bank(directory, json.dumps(bank).encode(), [])
            self.assertEqual(banks.load_bank(directory, imported["id"])["bank"]["provenance"]["training_rows"], 72)

    def test_training_accepts_long_bedrock_inference_profile_label(self):
        from model_test import modeltrace_training
        training = self.rows("train")
        label = "aws/arn:aws:bedrock:us-east-1:123456789012:inference-profile/" + "a" * 180
        for row in training:
            if row["source"] == "known/a":
                row["source"] = label
        bank = modeltrace_training.train_bank(training)
        self.assertIn(label, [model["id"] for model in bank["models"]])

    def test_same_run_or_duplicate_challenges_cannot_leak_into_evaluation(self):
        from model_test import modeltrace_training
        training = self.rows("train")
        with self.assertRaisesRegex(ValueError, "独立"):
            modeltrace_training.train_bank(training, training)
        with self.assertRaisesRegex(ValueError, "重复"):
            modeltrace_training.train_bank(training + training[:1], [])

    def test_unknown_model_is_not_counted_as_rejected_when_closed_set_ranks_it(self):
        from model_test import modeltrace_training
        training, holdout = self.rows("train"), self.rows("holdout")
        for row in holdout[36:]:
            row["source"] = "unknown/model"
        bank = modeltrace_training.train_bank(training, holdout)
        self.assertEqual(bank["evaluation"]["unknown_groups"], 12)
        self.assertEqual(bank["evaluation"]["unknown_ranked"] + bank["evaluation"]["unknown_abstained"], 12)
        self.assertNotIn("false_positive_rate", bank["evaluation"])


if __name__ == "__main__":
    unittest.main()
