import copy
import pytest
from qmagent.protocol import public_context


@pytest.mark.parametrize("correction,within", [(49999, True), (50000, False), (-50000, False)])
def test_public_constraints_are_signed_boundary_safe_and_nonmutating(correction, within):
    context = {"budget": {"remaining_experiments": 0, "tool_counts": {"sq.ramsey_df": 10},
                          "max_calls_per_tool": 10},
               "observation": {"tool": "sq.ramsey_df", "quality": {"reliable": True},
                               "fit_result": {"frequency_correction_hz": correction}}}
    before = copy.deepcopy(context)
    result = public_context(context)["execution_constraints"]
    assert context == before
    assert result["experiments_available"] is False
    assert result["retry_exhausted_tools"] == ["sq.ramsey_df"]
    assert result["ramsey_frequency_within_tolerance"] is within
    assert "next_tool" not in result


def test_confirmation_flags_require_public_evidence():
    context = {"budget": {"remaining_experiments": 30, "tool_counts": {}, "max_calls_per_tool": 10},
               "observation": {"tool": "sq.piamp", "round_in_experiment": 1}}
    flags = public_context(context)["execution_constraints"]
    assert not flags["piamp_confirmation_acquired"]
    assert not flags["finish_permitted"]
