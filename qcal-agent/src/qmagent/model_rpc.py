"""SSH stdio endpoint exposing model inference only."""

import argparse
import json
import os
import signal
import sys

MAX_FRAME = 2 * 1024**2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home")
    args = parser.parse_args(argv)
    output = os.fdopen(os.dup(sys.stdout.fileno()), "w", encoding="utf-8", buffering=1)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    from .model_worker import ManagedPolicy
    from .settings import app_home, load_settings
    settings = load_settings(app_home(args.home))
    if settings.policy != "hf":
        raise ValueError("model-rpc requires server config policy=hf")
    policy = ManagedPolicy(settings)

    def send(value):
        line = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")) + "\n"
        if len(line.encode()) > MAX_FRAME:
            raise ValueError("Model response too large")
        output.write(line)
        output.flush()

    def shutdown(signum, frame):
        raise SystemExit(128 + signum)
    for name in ("SIGTERM", "SIGHUP"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), shutdown)
    try:
        for raw in sys.stdin.buffer:
            request_id = None
            try:
                if len(raw) > MAX_FRAME:
                    raise ValueError("Model request too large")
                request = json.loads(raw)
                request_id = request.get("id") if isinstance(request, dict) else None
                if (not isinstance(request, dict) or set(request) != {"id", "method", "params"}
                        or type(request_id) is not int or request_id < 1 or not isinstance(request["params"], dict)):
                    raise ValueError("Invalid model request")
                method, params = request["method"], dict(request["params"])
                if method == "close":
                    send({"id": request_id, "ok": True, "result": {"closing": True}})
                    break
                if method == "info":
                    if params:
                        raise ValueError("Invalid info request")
                    result = {"protocol": 1, "settings": settings.to_dict(),
                              "capabilities": {"decide": True, "chat": True, "text_streaming": True}}
                elif method == "decide":
                    if set(params) != {"context"}:
                        raise ValueError("Invalid decide request")
                    result = policy.decide(params["context"])
                elif method == "chat":
                    if set(params) != {"messages", "context", "stream"} or type(params["stream"]) is not bool:
                        raise ValueError("Invalid chat request")
                    callback = ((lambda text: send({"id": request_id, "event": "text", "text": text}))
                                if params.pop("stream") else None)
                    result = policy.chat(params["messages"], params["context"], on_text=callback)
                else:
                    raise ValueError("Unlisted model method")
                send({"id": request_id, "ok": True, "result": result,
                      "generation": policy.last_generation})
            except Exception as exc:
                send({"id": request_id, "ok": False, "error": f"{type(exc).__name__}: {exc}"})
    except (KeyboardInterrupt, BrokenPipeError):
        return 130
    finally:
        policy.close()
        output.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
