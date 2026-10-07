import unittest
from evaluation_metrics import aggregate_scores, score_prediction, select_rows


class MetricsTests(unittest.TestCase):
    def test_missing_is_not_success(self):
        row = score_prediction("Q4", "{}", '{"r2":0.9}')
        self.assertIsNone(row["action_match"])
        self.assertIsNone(row["next_tool_match"])
        self.assertIsNone(aggregate_scores([row])["action_match_rate"])

    def test_invalid_prediction_counts_as_failure_if_eligible(self):
        row = score_prediction("POLICY", "not json", '{"parameter_action":{"action":"repeat"},"next_tool":"sq.s21"}')
        self.assertFalse(row["action_match"])
        self.assertFalse(row["next_tool_match"])

    def test_task_mapping(self):
        for task, answer in (("Q3", '{"next_step":{"action":"repeat"}}'),
                             ("Q6", '{"corrective_action":{"action":"repeat"}}'),
                             ("Q7", '{"action":"repeat"}'), ("Q8", '{"decision":"repeat"}')):
            self.assertTrue(score_prediction(task, answer, answer)["action_match"])

    def test_denominators(self):
        rows = [score_prediction("Q4", "{}", "{}"),
                score_prediction("Q7", '{"action":"repeat"}', '{"action":"repeat"}')]
        metrics = aggregate_scores(rows)
        self.assertEqual(metrics["action_match_eligible"], 1)
        self.assertEqual(metrics["action_match_rate"], 1)
        self.assertEqual(metrics["next_tool_match_eligible"], 0)

    def test_sampling_reproducible_and_order_independent(self):
        rows = [{"id": str(i), "task": "Q1" if i % 2 else "Q2"} for i in range(30)]
        a = select_rows(rows, 2, 123)
        self.assertEqual(a, select_rows(list(reversed(rows)), 2, 123))
        self.assertEqual(len(a), 4)
        self.assertEqual(len(select_rows(rows, 0, 123)), 30)


if __name__ == "__main__":
    unittest.main()
