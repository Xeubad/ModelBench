"""短问题行为分布：仅作描述，不把小样本强行解释成模型身份。"""
import unittest

from model_test import behavior, core


class BehaviorTest(unittest.TestCase):
    @staticmethod
    def records(label, answers, ok=True):
        return [{"scenario": "behavior", "label": label, "model": label, "task": probe, "round": i + 1,
                 "ok": ok, "text": value} for probe, values in answers.items() for i, value in enumerate(values)]

    def test_equal_and_disjoint_distributions(self):
        self.assertAlmostEqual(behavior.js_divergence({"a": 5}, {"a": 3}), 0)
        self.assertAlmostEqual(behavior.js_divergence({"a": 5}, {"b": 3}), 1)
        self.assertIsNone(behavior.js_divergence({}, {"b": 1}))

    def test_profiles_keep_failed_and_missing_attempts_out_of_votes(self):
        tasks = core.build_tasks(["behavior"])
        self.assertGreaterEqual(len(tasks), 3)
        answers = {task["id"]: ["A", "a ", "B", "A", "A"] for task in tasks}
        records = self.records("a", answers) + self.records("a", {tasks[0]["id"]: ["A"]}, ok=False)
        result = behavior.analyze(records, planned_rounds=6)
        probe = result["profiles"]["a"][tasks[0]["id"]]
        self.assertEqual((probe["valid"], probe["attempted"], probe["planned"]), (5, 6, 6))
        self.assertEqual(probe["counts"], {"a": 4, "b": 1})

    def test_comparison_requires_enough_valid_answers_per_probe(self):
        tasks = core.build_tasks(["behavior"])
        aid = {task["id"]: ["a"] * 5 for task in tasks}
        bid = {task["id"]: ["b"] * 5 for task in tasks}
        full = behavior.analyze(self.records("a", aid) + self.records("b", bid), planned_rounds=5)
        self.assertEqual(full["pairs"][0]["status"], "descriptive")
        self.assertAlmostEqual(full["pairs"][0]["mean_jsd"], 1)
        short = behavior.analyze(self.records("a", aid) + self.records("b", {k: v[:2] for k, v in bid.items()}), planned_rounds=5)
        self.assertEqual(short["pairs"][0]["status"], "insufficient")
        self.assertIsNone(short["pairs"][0]["mean_jsd"])
        self.assertNotIn("prediction", short["pairs"][0])

    def test_conclusion_for_successful_distribution_tests_is_not_empty(self):
        for scenario in ("behavior", "modeltrace"):
            record = {"scenario": scenario, "label": "a", "model": "a", "task": "probe", "ok": True,
                      "status": 200, "total_s": 1, "first_token_s": .5, "first_event_s": .2,
                      "connect_s": 0, "reused": True, "usage": None, "text": "x"}
            conclusion = core.conclusion(core.summarize([record]))
            self.assertNotIn("没有测到任何模型", conclusion)
            self.assertIn("1/1", conclusion)

    def test_stopped_run_uses_planned_not_observed_denominator(self):
        tasks = core.build_tasks(["behavior"])
        answers = {task["id"]: ["a"] * 5 for task in tasks}
        result = behavior.analyze(self.records("a", answers) + self.records("b", answers), planned_rounds=10)
        self.assertEqual(result["pairs"][0]["status"], "insufficient")
        self.assertTrue(all(v["required"] == 6 for v in result["profiles"]["a"].values()))


if __name__ == "__main__":
    unittest.main()
