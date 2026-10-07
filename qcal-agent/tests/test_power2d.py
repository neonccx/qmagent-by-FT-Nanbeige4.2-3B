import copy
import unittest

from qmagent.analysis_tools import analyze
from qmagent.contracts import ContractError, DEFAULT_STATE, decision, validate_observation, validate_scan
from qmagent.parameter_tools import calculate_fit_updates
from qmagent.physical_backend import PhysicalSimulator
from qmagent.runtime import AgentRunner
from qmagent.policies import RulePolicy


class Power2DTests(unittest.TestCase):
    def test_grid_fit_and_contract(self):
        backend = PhysicalSimulator(2026091301, noise_scale=0.2)
        raw = backend.acquire("sq.s21_power2d", DEFAULT_STATE, {})
        self.assertEqual(len(raw["measurement"]["i"]), 31)
        self.assertEqual(len(raw["measurement"]["i"][0]), 241)
        result = raw | analyze(raw)
        validate_observation(result, "sq.s21_power2d", DEFAULT_STATE)
        self.assertTrue(result["quality"]["reliable"])
        self.assertLess(abs(result["fit_result"]["readout_power_dbm"]-
                            backend._truth["readout_optimum_dbm"]), 4.0)

    def test_scan_rejects_reversed_power_range(self):
        with self.assertRaisesRegex(ContractError, "minimum power"):
            validate_scan("sq.s21_power2d", DEFAULT_STATE,
                          {"power_min_dbm": -10.0, "power_max_dbm": -30.0})

    def test_fit_tool_updates_power_and_frequency(self):
        backend = PhysicalSimulator(2026091302, noise_scale=0.1)
        observation = backend.measure("sq.s21_power2d", DEFAULT_STATE, {})
        update = calculate_fit_updates(observation, copy.deepcopy(DEFAULT_STATE))["updates"]
        self.assertEqual(set(update), {"readout_power_dbm", "readout_frequency_hz"})

    def test_rule_flow_contains_power2d(self):
        result = AgentRunner(PhysicalSimulator(2026091303), RulePolicy(), max_steps=45).run()
        self.assertEqual(result["status"], "accepted")
        self.assertGreaterEqual(result["tool_counts"]["sq.s21_power2d"], 1)


if __name__ == "__main__":
    unittest.main()
