"""Guarded adapter for user-supplied real-hardware acquisition functions.

This module intentionally contains no vendor driver and never discovers callable
names dynamically.  A deployment must register every permitted experiment and
approve every acquisition at action time.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from importlib import metadata
from typing import Any

from .analysis_tools import analyze
from .contracts import TOOLS, validate_scan
from .storage import digest


class HardwareApprovalError(RuntimeError):
    """Raised when a physical acquisition was not explicitly approved."""


class HardwareSafetyError(RuntimeError):
    """Raised when laboratory-specific deterministic limits reject a request."""


class RegisteredHardwareBackend:
    """Adapt an allow-list of acquisition callables to the Agent backend contract.

    Each handler receives one deep-copied request object with ``tool``, ``state``,
    ``scan`` and a monotonically increasing ``sequence``.  It must return the raw
    observation schema accepted by :func:`qmagent.analysis_tools.analyze`.
    """

    backend_name = "registered-hardware-1.0"
    synthetic = False

    def __init__(
        self,
        handlers: Mapping[str, Callable[[dict[str, Any]], dict[str, Any]]],
        safety_check: Callable[[dict[str, Any]], bool],
        approve: Callable[[dict[str, Any]], bool],
        safe_shutdown: Callable[[], None] | None = None,
        provider_name: str = "laboratory",
    ) -> None:
        unknown = set(handlers) - set(TOOLS)
        if unknown:
            raise ValueError(f"Unregistered experiment names: {sorted(unknown)}")
        if not handlers or not callable(safety_check) or not callable(approve):
            raise ValueError("Handlers, a safety callback and an approval callback are required")
        if safe_shutdown is not None and not callable(safe_shutdown):
            raise TypeError("safe_shutdown must be callable")
        if not isinstance(provider_name, str) or not provider_name.strip():
            raise ValueError("provider_name must be a non-empty string")
        self._handlers = dict(handlers)
        self._safety_check = safety_check
        self._approve = approve
        self._safe_shutdown = safe_shutdown
        self.provider_name = provider_name.strip()
        self._sequence = 0
        self._closed = False

    def checkpoint(self) -> dict[str, int]:
        """Return logical bookkeeping only; physical hardware is never rewound."""
        return {"sequence": self._sequence}

    def restore(self, checkpoint: dict[str, int]) -> None:
        if (set(checkpoint) != {"sequence"} or type(checkpoint["sequence"]) is not int
                or checkpoint["sequence"] < 0):
            raise ValueError("Invalid hardware-backend checkpoint")
        if self._safe_shutdown is not None:
            self._safe_shutdown()
        # A physical acquisition cannot be rolled back. Never reuse its audit ID.
        self._sequence = max(self._sequence, checkpoint["sequence"])

    def acquire(self, tool: str, state: dict, scan: dict) -> dict:
        if self._closed:
            raise RuntimeError("Hardware backend is closed")
        if tool not in self._handlers:
            raise ValueError(f"No hardware handler registered for {tool}")
        validate_scan(tool, state, scan)
        request = {"tool": tool, "state": copy.deepcopy(state), "scan": copy.deepcopy(scan),
                   "sequence": self._sequence + 1}
        if self._safety_check(copy.deepcopy(request)) is not True:
            raise HardwareSafetyError(f"Laboratory safety policy rejected acquisition: {tool}")
        if self._approve(copy.deepcopy(request)) is not True:
            raise HardwareApprovalError(f"Physical acquisition was not approved: {tool}")
        self._sequence += 1
        try:
            raw = self._handlers[tool](copy.deepcopy(request))
            if not isinstance(raw, dict):
                raise TypeError("Hardware handler must return a raw observation object")
            raw = copy.deepcopy(raw)
            raw.setdefault("tool", tool)
            raw.setdefault("current_parameters", copy.deepcopy(state))
            raw.setdefault("scan", copy.deepcopy(scan))
            raw["synthetic"] = False
            raw["backend"] = self.backend_name
            return raw
        except BaseException:
            if self._safe_shutdown is not None:
                self._safe_shutdown()
            raise

    def measure(self, tool: str, state: dict, scan: dict) -> dict:
        raw = self.acquire(tool, state, scan)
        try:
            result = analyze(raw)
            raw_hash = digest(raw)
            return raw | result | {
                "raw_artifact": {"id": "sha256:" + raw_hash, "sha256": raw_hash},
                "tool_trace": [
                    {"tool": tool, "operation": "acquire", "output_sha256": raw_hash},
                    {"tool": result["analysis_tool"], "operation": "analyze", "input_sha256": raw_hash},
                ],
            }
        except BaseException:
            if self._safe_shutdown is not None:
                self._safe_shutdown()
            raise

    def close(self) -> None:
        """Put outputs in their provider-defined safe state exactly once."""
        if not self._closed:
            self._closed = True
            if self._safe_shutdown is not None:
                self._safe_shutdown()


HARDWARE_ENTRY_POINT_GROUP = "qcal_agent.hardware"


def discover_hardware_backend() -> RegisteredHardwareBackend | None:
    """Return the one healthy installed hardware provider, or ``None``.

    Providers are installed Python entry points in ``qcal_agent.hardware``.
    The loaded object must expose ``probe()`` and ``create_backend()``.  A probe
    returning ``False`` means no device is attached.  A broken provider fails
    closed; it is never treated as an absent device and silently replaced after
    claiming availability.
    """
    entry_points = metadata.entry_points()
    providers = list(entry_points.select(group=HARDWARE_ENTRY_POINT_GROUP))
    available = []
    for entry_point in sorted(providers, key=lambda item: item.name):
        try:
            provider = entry_point.load()
            if not callable(getattr(provider, "probe", None)) or not callable(
                    getattr(provider, "create_backend", None)):
                raise TypeError("provider must expose probe() and create_backend()")
            ready = provider.probe()
            if type(ready) is not bool:
                raise TypeError("provider probe() must return bool")
            if ready:
                available.append((entry_point.name, provider))
        except Exception as exc:
            raise RuntimeError(f"Hardware provider {entry_point.name!r} failed its startup probe: {exc}") from exc
    if not available:
        return None
    if len(available) != 1:
        raise RuntimeError("Multiple hardware providers are available; select only one laboratory connection")
    name, provider = available[0]
    try:
        backend = provider.create_backend()
    except Exception as exc:
        raise RuntimeError(f"Hardware provider {name!r} was detected but could not start: {exc}") from exc
    if not isinstance(backend, RegisteredHardwareBackend):
        raise TypeError(f"Hardware provider {name!r} did not return RegisteredHardwareBackend")
    backend.provider_name = name
    return backend
