from types import SimpleNamespace

import pytest

from qmagent.backends import make_backend
from qmagent.hardware_backend import RegisteredHardwareBackend
from qmagent.physical_backend import PhysicalSimulator
from qmagent.settings import Settings


def test_auto_falls_back_to_physical_simulator_when_hardware_is_absent(monkeypatch):
    monkeypatch.setattr("qmagent.hardware_backend.discover_hardware_backend", lambda: None)
    backend = make_backend(Settings(backend="auto"))
    assert isinstance(backend, PhysicalSimulator)
    assert backend.synthetic is True


def test_auto_uses_detected_hardware(monkeypatch):
    hardware = RegisteredHardwareBackend({"sq.s21": lambda request: {}}, lambda request: True,
                                         lambda request: True)
    monkeypatch.setattr("qmagent.hardware_backend.discover_hardware_backend", lambda: hardware)
    assert make_backend(Settings(backend="auto")) is hardware


def test_strict_hardware_mode_refuses_simulator_fallback(monkeypatch):
    monkeypatch.setattr("qmagent.hardware_backend.discover_hardware_backend", lambda: None)
    with pytest.raises(RuntimeError, match="no healthy"):
        make_backend(Settings(backend="hardware"))
