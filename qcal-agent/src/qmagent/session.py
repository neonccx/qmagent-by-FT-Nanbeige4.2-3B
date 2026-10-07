"""Durable sessions shared by simulated and explicitly registered hardware backends."""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re

from . import __version__
from .model_worker import ManagedPolicy
from .policies import RulePolicy
from .runtime import AgentRunner
from .settings import Settings, load_settings, save_settings
from .backends import make_backend
from .storage import Journal, SessionLock, atomic_json, digest, read_json, session_directory


def source_hashes():
    return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path(__file__).parent.glob("*.py"))}


def make_policy(settings):
    if settings.policy == "rule":
        return RulePolicy()
    if settings.policy == "remote":
        raise ValueError("Remote-model sessions must be opened with qm-agent connect")
    for name in ("model", "adapter"):
        value = getattr(settings, name)
        if value and not Path(value).is_dir():
            raise ValueError(f"Missing local {name} directory: {value}")
    return ManagedPolicy(settings)


class CalibrationSession:
    def __init__(self, directory, metadata, lock, journal, policy=None, backend=None):
        self.directory, self.metadata, self.lock, self.journal = directory, metadata, lock, journal
        self.settings = Settings.from_dict(metadata["settings"])
        self.policy = policy if policy is not None else make_policy(self.settings)
        self.runner = AgentRunner(backend if backend is not None else make_backend(self.settings), self.policy,
                                  self.settings.max_steps, self.settings.max_tool_calls)
        self.saved_events = 0
        self.broken = False

    @classmethod
    def create(cls, home: Path, settings: Settings, policy=None, qubit_name="unassigned"):
        settings = Settings.from_dict(settings.to_dict())
        if not isinstance(qubit_name, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,31}", qubit_name):
            raise ValueError("Qubit name must use 1-32 ASCII letters, digits, underscores or hyphens")
        beijing = timezone(timedelta(hours=8))
        base = datetime.now(beijing).strftime("%Y%m%dT%H%M%S-UTCp8-") + qubit_name
        session_id = base
        suffix = 2
        while session_directory(home, session_id).exists():
            session_id = f"{base}-{suffix:02d}"
            suffix += 1
        directory = session_directory(home, session_id)
        directory.mkdir(mode=0o700, parents=True)
        lock = SessionLock(directory)
        backend = None
        session = None
        try:
            backend = make_backend(settings)
            synthetic = bool(getattr(backend, "synthetic", True))
            metadata = {"version": 1, "package_version": __version__, "session_id": session_id,
                        "created_at": datetime.now(timezone.utc).isoformat(), "settings": settings.to_dict(),
                        "qubit_name": qubit_name, "display_timezone": "UTC+08:00",
                        "source_sha256": source_hashes(), "backend": backend.backend_name,
                        "hardware_provider": getattr(backend, "provider_name", None),
                        "synthetic": synthetic, "runtime_schema": "runtime-1.0"}
            atomic_json(directory / "session_config.json", metadata)
            session = cls(directory, metadata, lock, Journal(directory, create=True), policy, backend)
            session.save()
            return session
        except BaseException:
            if session:
                session.close()
            else:
                if backend is not None and hasattr(backend, "close"):
                    backend.close()
                lock.close()
            raise

    @classmethod
    def resume(cls, home: Path, session_id: str, policy=None):
        directory = session_directory(home, session_id)
        if not directory.is_dir():
            raise ValueError("Session does not exist")
        lock = SessionLock(directory)
        session = None
        backend = None
        try:
            metadata = read_json(directory / "session_config.json")
            settings = Settings.from_dict(metadata["settings"])
            backend = make_backend(settings)
            if (metadata["version"] != 1 or metadata["package_version"] != __version__
                    or metadata["source_sha256"] != source_hashes()
                    or metadata["session_id"] != session_id
                    or metadata["backend"] != backend.backend_name
                    or metadata["synthetic"] is not bool(getattr(backend, "synthetic", True))
                    or metadata.get("hardware_provider") != getattr(backend, "provider_name", None)):
                raise ValueError("Session/code version mismatch; keep old records and start a new session")
            journal = Journal(directory)
            latest = journal.recover()
            if latest["metadata_sha256"] != digest(metadata):
                raise ValueError("Session configuration changed; refusing mixed-policy resume")
            session = cls(directory, metadata, lock, journal, policy, backend)
            session.runner.restore(latest["snapshot"])
            return session
        except BaseException:
            if session:
                session.close()
            else:
                if backend is not None and hasattr(backend, "close"):
                    backend.close()
                lock.close()
            raise

    def save(self):
        if self.broken:
            raise RuntimeError("Previous persistence failure; reopen the last committed session")
        try:
            self.journal.append(self.runner.snapshot(), self.runner.events[self.saved_events:], digest(self.metadata))
            self.saved_events = len(self.runner.events)
        except BaseException:
            self.broken = True  # Never continue measuring after a failed commit.
            raise

    def plan(self):
        if self.broken:
            raise RuntimeError("Session persistence failed; reopen it")
        if self.runner.pending is not None or self.runner.status != "active":
            return self.runner.plan()
        try:
            result = self.runner.plan()
        except KeyboardInterrupt:
            self.pause("Interrupted during model planning")
            raise
        self.save()
        return result

    def step(self):
        self.plan()  # Proposal is committed before execution; /plan doesn't measure.
        if self.runner.status != "active":
            return self.runner.result()
        try:
            result = self.runner.step()
        except KeyboardInterrupt:
            self.pause("Interrupted; simulator measurement rolled back")
            raise
        self.save()
        return result

    def diagnostic(self, tool):
        if self.broken:
            raise RuntimeError("Session persistence failed; reopen it")
        try:
            result = self.runner.diagnostic(tool)
        except KeyboardInterrupt:
            self.pause("Interrupted during diagnostic measurement")
            raise
        self.save()
        return result

    def pause(self, reason="User paused the session"):
        self.runner.emit({"event": "pause", "reason": reason})
        self.save()
        if hasattr(self.policy, "close"):
            self.policy.close()

    def export(self, directory: Path):
        directory = directory.expanduser().resolve()
        directory.mkdir(mode=0o700, parents=True, exist_ok=False)
        result = self.runner.result()
        result.pop("events")
        result.update(session_id=self.metadata["session_id"], synthetic=self.metadata["synthetic"],
                      backend=self.metadata["backend"], policy=self.settings.policy)
        atomic_json(directory / "result.json", result)
        atomic_json(directory / "run_config.json", self.metadata)
        # Export public audit events only, never backend RNG or internal truth.
        with (directory / "trajectory.jsonl").open("x", encoding="utf-8") as handle:
            for record, _ in self.journal.records():
                for event in record["events"]:
                    handle.write(json.dumps(event, ensure_ascii=False, allow_nan=False) + "\n")
        gate = result["final_iq_gate"] or {}
        report = (f"# Calibration report\n\nSession: `{self.metadata['session_id']}`\n\n"
                  f"Policy: `{self.settings.policy}`; backend: `{self.metadata['backend']}`; "
                  f"data: `{'synthetic' if self.metadata['synthetic'] else 'REAL HARDWARE'}`.\n\n"
                  f"Status: `{result['status']}`; experiments: {result['experiment_count']}; "
                  f"consecutive IQ passes: {result['consecutive_iq_passes']}.\n\n"
                  "IQ metrics (held-out shots):\n\n```json\n" +
                  json.dumps(gate.get("metrics", {}), indent=2, allow_nan=False) +
                  "\n```\n\nRule results are not LLM performance. Two-step LoRA is not a trained policy.\n")
        (directory / "report.md").write_text(report, encoding="utf-8")
        if any(item["tool"] == "sq.iqraw" for item in self.runner.history):
            try:
                from .iq_report import from_trajectory
                from_trajectory(directory / "trajectory.jsonl", directory, policy=self.settings.policy,
                                status=result["status"], consecutive_passes=result["consecutive_iq_passes"],
                                synthetic=self.metadata["synthetic"])
                report += "\n## Recorded IQ measurement\n\n![IQ analysis](iq_report.png)\n\nSee iq_metrics.json for definitions and provenance.\n"
                (directory / "report.md").write_text(report, encoding="utf-8")
            except (ImportError, ValueError, OSError) as exc:
                # Numeric/audit export remains available if optional plotting fails.
                atomic_json(directory / "plot_error.json", {"error": str(exc)})
        return directory

    def close(self):
        try:
            if hasattr(self.policy, "close"):
                self.policy.close()
        finally:
            try:
                if hasattr(self.runner.backend, "close"):
                    self.runner.backend.close()
            finally:
                self.lock.close()


def list_sessions(home: Path):
    root = home / "sessions"
    if not root.is_dir():
        return []
    rows = []
    for directory in sorted(root.iterdir(), reverse=True):
        if not directory.is_dir() or directory.is_symlink():
            continue
        try:
            metadata = read_json(directory / "session_config.json")
            latest = Journal(directory).recover()["snapshot"]
            rows.append({"id": directory.name, "policy": metadata["settings"]["policy"],
                         "status": latest["status"], "steps": sum(latest["counts"].values())})
        except (OSError, ValueError, KeyError, TypeError):
            rows.append({"id": directory.name, "status": "unreadable", "policy": "?", "steps": "?"})
    return rows
