"""SSH model client; Agent state, experiments and reports remain local."""

from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import subprocess
import time

from .storage import atomic_json, read_json

MAX_FRAME = 2 * 1024**2


def _decode(line):
    if len(line) > MAX_FRAME:
        raise ValueError("Model RPC frame too large")
    value = json.loads(line)
    if not isinstance(value, dict):
        raise ValueError("Expected model RPC object")
    return value


@dataclass(frozen=True)
class RemoteProfile:
    host: str
    project: str
    control_path: str | None = None
    home: str | None = None
    local_results: str | None = None  # Legacy remote.json compatibility; unused.

    @classmethod
    def from_dict(cls, value):
        allowed = {"host", "project", "control_path", "home", "local_results"}
        if not isinstance(value, dict) or set(value) - allowed or not {"host", "project"} <= set(value):
            raise ValueError("Invalid remote profile; credentials/commands are not supported")
        result = cls(**value)
        if (not isinstance(result.host, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.@-]{0,200}", result.host)
                or result.host.count("@") > 1):
            raise ValueError("Invalid SSH host/account")
        for name in ("project", "home"):
            path = getattr(result, name)
            if path is not None and (not isinstance(path, str) or not PurePosixPath(path).is_absolute()
                                     or any(ord(char) < 32 for char in path)):
                raise ValueError("Remote paths must be absolute, without control characters")
        if not result.project:
            raise ValueError("Missing remote project")
        if result.control_path is not None and (not isinstance(result.control_path, str)
                                                or not Path(result.control_path).is_absolute()):
            raise ValueError("Control socket path must be absolute")
        if result.local_results is not None and (not isinstance(result.local_results, str)
                                                 or not Path(result.local_results).is_absolute()):
            raise ValueError("Local results path must be absolute")
        return result

    def command(self):
        command = ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                   "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
        if self.control_path:
            command += ["-S", self.control_path]
        remote_args = ["./qm-agent", "model-rpc"]
        if self.home:
            remote_args += ["--home", self.home]
        return command + [self.host, "cd " + shlex.quote(self.project) + " && exec " + shlex.join(remote_args)]

    def login_command(self):
        if not self.control_path:
            raise ValueError("Configure --control-path for reusable interactive SSH login")
        return ["ssh", "-o", "ControlMaster=auto", "-o", "ControlPersist=3600",
                "-o", "ControlPath=" + self.control_path, self.host]


def save_remote(home, profile):
    home.mkdir(mode=0o700, parents=True, exist_ok=True)
    atomic_json(home / "remote.json", asdict(profile), replace=True)


def load_remote(home):
    path = home / "remote.json"
    if not path.exists():
        raise ValueError("Remote model not configured; run qm-agent remote first")
    return RemoteProfile.from_dict(read_json(path))


class RemoteModelPolicy:
    """Policy-compatible proxy whose only remote operations are decide and chat."""

    supports_streaming = True

    def __init__(self, profile, on_wait=None, command=None):
        self.profile = profile
        self.process = subprocess.Popen(command or profile.command(), stdin=subprocess.PIPE,
                                        stdout=subprocess.PIPE, bufsize=0, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = bytearray()
        self.next_id = 0
        self.connected = True
        self.timeout = 240.0
        self.on_wait = on_wait
        self.last_generation = None
        try:
            info = self._call("info")
            self.server_settings = info["settings"]
            self.timeout = float(self.server_settings["request_timeout"]) + 60
        except BaseException:
            self.close()
            raise

    def _send(self, value):
        data = (json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n").encode()
        if len(data) > MAX_FRAME:
            raise ValueError("Model request too large")
        self.process.stdin.write(data)
        self.process.stdin.flush()

    def _receive(self, request_id, on_text=None):
        started, last_notice = time.monotonic(), 0.0
        while time.monotonic() - started < self.timeout:
            if b"\n" in self.buffer:
                line, _, remainder = self.buffer.partition(b"\n")
                self.buffer = bytearray(remainder)
                result = _decode(line)
                if result.get("id") != request_id:
                    raise RuntimeError("Unexpected model response ID")
                if result.get("event") == "text":
                    if on_text is None or not isinstance(result.get("text"), str):
                        raise ValueError("Invalid model stream event")
                    on_text(result["text"])
                    continue
                return result
            if self.selector.select(timeout=0.2):
                chunk = self.process.stdout.read(65536)
                if not chunk:
                    raise ConnectionError("Model server disconnected")
                self.buffer.extend(chunk)
                if len(self.buffer) > MAX_FRAME:
                    raise ValueError("Model response too large")
            elapsed = time.monotonic() - started
            if self.on_wait and elapsed - last_notice >= 20:
                self.on_wait(f"Model server processing; waited {int(elapsed)} s; Ctrl-C cancels locally.")
                last_notice = elapsed
        raise TimeoutError("Remote model request timed out")

    def _call(self, method, on_text=None, **params):
        if not self.connected:
            raise ConnectionError("Remote model connection is closed")
        if method not in {"info", "decide", "chat", "close"}:
            raise ValueError("Unlisted model method")
        self.next_id += 1
        request_id = self.next_id
        try:
            self._send({"id": request_id, "method": method, "params": params})
            response = self._receive(request_id, on_text=on_text)
        except (Exception, KeyboardInterrupt):
            self.close()
            raise
        if response.get("ok") is not True:
            raise RuntimeError(response.get("error", "Remote model failed"))
        self.last_generation = response.get("generation")
        return response.get("result")

    def decide(self, context):
        return self._call("decide", context=context)

    def chat(self, messages, context, on_text=None):
        return self._call("chat", messages=messages, context=context, stream=on_text is not None,
                          on_text=on_text)

    def close(self):
        if not getattr(self, "connected", False):
            return
        self.connected = False
        try:
            if self.process.poll() is None:
                try:
                    self.next_id += 1
                    data = (json.dumps({"id": self.next_id, "method": "close", "params": {}}) + "\n").encode()
                    self.process.stdin.write(data)
                    self.process.stdin.flush()
                except OSError:
                    pass
                try:
                    self.process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    self.process.terminate()
                    self.process.wait(timeout=2)
        finally:
            self.selector.close()
            for stream in (self.process.stdin, self.process.stdout):
                if stream:
                    stream.close()
