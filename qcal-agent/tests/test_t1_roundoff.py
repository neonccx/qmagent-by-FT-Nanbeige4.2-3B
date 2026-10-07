from qmagent.runtime import stage_passed


def test_t1_accepts_machine_roundoff_not_short_delay():
    t1 = 35.96765297361837
    observation = {"tool": "sq.t1", "quality": {"reliable": True},
                   "fit_result": {"t1_us": t1}, "current_parameters": {}}
    assert stage_passed(observation, {"t1_us": t1, "relaxation_delay_us": 179.83826486809184})
    assert not stage_passed(observation, {"t1_us": t1, "relaxation_delay_us": 5 * t1 - 1e-8})
