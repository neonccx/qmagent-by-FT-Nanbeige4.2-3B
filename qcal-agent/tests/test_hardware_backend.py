import copy
from types import SimpleNamespace

import pytest

from qmagent.contracts import DEFAULT_STATE
from qmagent.hardware_backend import (HardwareApprovalError, HardwareSafetyError,
                                      RegisteredHardwareBackend, discover_hardware_backend)
from qmagent.physical_backend import PhysicalSimulator


def raw_s21(request):
    return PhysicalSimulator(7, noise_scale=0).acquire(
        request["tool"], request["state"], request["scan"]
    )


def test_registered_hardware_backend_requires_action_time_approval():
    backend = RegisteredHardwareBackend({"sq.s21": raw_s21}, lambda request: True, lambda request: False)
    with pytest.raises(HardwareApprovalError):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})


def test_registered_hardware_backend_returns_measured_fit_and_audit_trace():
    requests = []
    backend = RegisteredHardwareBackend(
        {"sq.s21": raw_s21}, lambda request: True, lambda request: requests.append(request) or True
    )
    observation = backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert observation["synthetic"] is False
    assert observation["quality"]["reliable"] is True
    assert observation["tool_trace"][0]["operation"] == "acquire"
    assert requests[0]["sequence"] == 1


def test_hardware_failure_invokes_safe_shutdown():
    shutdowns = []

    def fail(_request):
        raise RuntimeError("instrument timeout")

    backend = RegisteredHardwareBackend(
        {"sq.s21": fail}, lambda request: True, lambda request: True, lambda: shutdowns.append(True)
    )
    with pytest.raises(RuntimeError, match="instrument timeout"):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert shutdowns == [True]


def test_hardware_sequence_is_never_reused_after_failed_acquisition_and_restore():
    sequences = []
    shutdowns = []

    def sometimes_fails(request):
        sequences.append(request["sequence"])
        if len(sequences) == 1:
            raise RuntimeError("instrument timeout")
        return raw_s21(request)

    backend = RegisteredHardwareBackend(
        {"sq.s21": sometimes_fails}, lambda request: True, lambda request: True,
        lambda: shutdowns.append(True),
    )
    checkpoint = backend.checkpoint()
    with pytest.raises(RuntimeError, match="instrument timeout"):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    backend.restore(checkpoint)
    backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert sequences == [1, 2]
    assert backend.checkpoint() == {"sequence": 2}
    assert shutdowns


def test_hardware_restore_rejects_negative_sequence():
    backend = RegisteredHardwareBackend({"sq.s21": raw_s21}, lambda request: True, lambda request: True)
    with pytest.raises(ValueError, match="Invalid hardware-backend checkpoint"):
        backend.restore({"sequence": -1})


def test_invalid_hardware_observation_invokes_safe_shutdown():
    shutdowns = []
    backend = RegisteredHardwareBackend(
        {"sq.s21": lambda request: None}, lambda request: True, lambda request: True,
        lambda: shutdowns.append(True),
    )
    with pytest.raises(TypeError, match="raw observation"):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert shutdowns == [True]


def test_hardware_analysis_error_invokes_safe_shutdown():
    shutdowns = []
    backend = RegisteredHardwareBackend(
        {"sq.s21": lambda request: {"measurement": {"i": [], "q": []}}}, lambda request: True,
        lambda request: True, lambda: shutdowns.append(True),
    )
    with pytest.raises((ValueError, KeyError)):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert shutdowns == [True]


class EntryPoints(list):
    def select(self, **kwargs):
        return self if kwargs == {"group": "qcal_agent.hardware"} else []


def test_discovery_returns_none_when_no_provider_is_installed(monkeypatch):
    monkeypatch.setattr("qmagent.hardware_backend.metadata.entry_points", lambda: EntryPoints())
    assert discover_hardware_backend() is None


def test_discovery_selects_one_healthy_registered_provider(monkeypatch):
    backend = RegisteredHardwareBackend({"sq.s21": raw_s21}, lambda request: True, lambda request: True)
    provider = SimpleNamespace(probe=lambda: True, create_backend=lambda: backend)
    entry = SimpleNamespace(name="test-lab", load=lambda: provider)
    monkeypatch.setattr("qmagent.hardware_backend.metadata.entry_points", lambda: EntryPoints([entry]))
    assert discover_hardware_backend() is backend
    assert backend.synthetic is False
    assert backend.provider_name == "test-lab"


def test_laboratory_safety_check_precedes_human_approval():
    approvals = []
    backend = RegisteredHardwareBackend({"sq.s21": raw_s21}, lambda request: False,
                                        lambda request: approvals.append(request) or True)
    with pytest.raises(HardwareSafetyError):
        backend.measure("sq.s21", copy.deepcopy(DEFAULT_STATE), {})
    assert approvals == []


def test_broken_installed_provider_does_not_silently_fall_back(monkeypatch):
    provider = SimpleNamespace(probe=lambda: (_ for _ in ()).throw(OSError("controller offline")),
                               create_backend=lambda: None)
    entry = SimpleNamespace(name="broken-lab", load=lambda: provider)
    monkeypatch.setattr("qmagent.hardware_backend.metadata.entry_points", lambda: EntryPoints([entry]))
    with pytest.raises(RuntimeError, match="broken-lab"):
        discover_hardware_backend()
