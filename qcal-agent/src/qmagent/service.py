"""Local Agent application core for sessions, experiments and reports."""

import copy
from pathlib import Path
import secrets

from . import __version__
from .conversation import ReplyStream, explicit_intent, parse_reply, validate_message
from .contracts import TOOLS
from .session import CalibrationSession, list_sessions
from .settings import Settings, load_settings
from .storage import digest


class AgentService:
    def __init__(self, home, session, policy_factory=None, settings_factory=None):
        self.home, self.session = home, session
        self.policy_factory = policy_factory
        self.settings_factory = settings_factory
        self.ticket = None
        self._restore_chat()

    def _restore_chat(self):
        self.messages, self.total_turns = [], 0
        for record, _ in self.session.journal.records():
            for event in record["events"]:
                if event["event"] == "chat_clear":
                    self.messages, self.total_turns = [], 0
                if event["event"] == "chat":
                    self.messages.extend([{"role": "user", "content": event["user"]},
                                          {"role": "assistant", "content": event["assistant"]}])
                    self.messages = self.messages[-24:]
                    self.total_turns += 1

    def info(self):
        backend = self.session.runner.backend
        synthetic = bool(getattr(backend, "synthetic", True))
        return {"version": __version__, "protocol": 1, "session_id": self.session.metadata["session_id"],
                "directory": str(self.session.directory), "settings": self.session.settings.to_dict(),
                "backend": backend.backend_name, "synthetic": synthetic,
                "hardware_provider": getattr(backend, "provider_name", None),
                "conversation_turns": self.total_turns,
                "broken": self.session.broken, "capabilities": {"text_streaming": True}}

    def status(self):
        result = self.session.runner.result()
        result.pop("events")
        result["max_steps"] = self.session.settings.max_steps
        return result

    def history(self):
        return copy.deepcopy(self.session.runner.history)

    def why(self):
        action = self.session.runner.pending
        if action is None:
            for record, _ in self.session.journal.records():
                for event in record["events"]:
                    if event["event"] == "decision":
                        action = event["decision"]
        return {"reason": action["reason"] if action else "No decision yet. Request a preview of the next action.",
                "fit_result": (self.session.runner.observation or {}).get("fit_result", {})}

    def plan(self):
        self.ticket = None
        return {"plan": self.session.plan(), "status": self.status()}

    def prepare(self, mode="step", limit=None, target=None):
        """Issue a scoped challenge, with no execution or state writes to instruments."""
        self.ticket = None
        if mode not in {"step", "run", "goal", "diagnostic"} or (limit is not None and (type(limit) is not int or not 1 <= limit <= 1000)):
            raise ValueError("Invalid execution request")
        if mode == "step" and limit is not None:
            raise ValueError("step does not accept a limit")
        if mode in {"goal", "diagnostic"} and target not in TOOLS:
            raise ValueError("Goal/diagnostic mode requires a registered experiment target")
        if mode not in {"goal", "diagnostic"} and target is not None:
            raise ValueError("Only goal/diagnostic mode accepts a target")
        if self.session.runner.status != "active":
            return {"token": None, "status": self.status()}
        plan = self.session.plan() if mode == "step" else None
        if mode == "step" and plan is None:
            return {"token": None, "status": self.status()}
        remaining = self.session.settings.max_steps - sum(self.session.runner.counts.values())
        allowance = 1 if mode in {"step", "diagnostic"} else min(remaining, limit if limit is not None else remaining)
        token = secrets.token_urlsafe(24)
        self.ticket = {"token": token, "mode": mode, "remaining": allowance,
                       "state": digest(self.session.runner.snapshot()), "target": target,
                       "target_start_count": self.session.runner.counts.get(target, 0) if target else None}
        required = ([tool for index, tool in enumerate(TOOLS[:TOOLS.index(target) + 1])
                     if index not in self.session.runner.completed] if target else [])
        return {"token": token, "mode": mode, "limit": allowance, "plan": plan,
                "target": target, "required_tools": required, "status": self.status()}

    def execute(self, token):
        """A chat reply cannot call this: only the UI-confirmed capability can."""
        ticket = self.ticket
        if (not isinstance(token, str) or not ticket or not secrets.compare_digest(ticket["token"], token)
                or ticket["state"] != digest(self.session.runner.snapshot())):
            self.ticket = None
            raise ValueError("Execution confirmation expired; preview and confirm the plan again")
        before_count = sum(self.session.runner.counts.values())
        before_events = len(self.session.runner.events)
        try:
            executed_action = None
            if ticket["mode"] == "diagnostic":
                self.session.diagnostic(ticket["target"])
                ticket["remaining"] = 0
                target_reached, done = True, True
            else:
                action = self.session.plan()
                executed_action = copy.deepcopy(action)
                done = action is None
                if action is not None and ticket["remaining"] == 0 and action["next_tool"].startswith("sq."):
                    done = True
                elif action is not None:
                    self.session.step()
                    ticket["remaining"] -= sum(self.session.runner.counts.values()) - before_count
                    target_reached = (ticket["mode"] == "goal" and
                        self.session.runner.counts.get(ticket["target"], 0) > ticket["target_start_count"])
                    done = self.session.runner.status != "active" or ticket["mode"] == "step" or target_reached
            target_reached = (ticket["mode"] == "diagnostic" or (ticket["mode"] == "goal" and
                self.session.runner.counts.get(ticket["target"], 0) > ticket["target_start_count"])
                )
            next_plan = None
            measured = sum(self.session.runner.counts.values()) > before_count
            if measured and not target_reached and self.session.runner.status == "active":
                next_plan = self.session.plan()
                if next_plan is None or self.session.runner.status != "active":
                    done = True
            if ticket["remaining"] == 0:
                done = True
            ticket["state"] = digest(self.session.runner.snapshot())
            if done:
                self.ticket = None
            observations = [event for event in self.session.runner.events[before_events:] if event["event"] == "experiment"]
            last = observations[-1] if observations else None
            summary = ({"step": last["step"], "tool": last["tool"], "fit_result": last["observation"]["fit_result"],
                        "quality": last["observation"]["quality"], "iq_passes": last["consecutive_iq_passes"]} if last else None)
            return {"status": self.status(), "experiment": summary, "done": done,
                    "remaining": ticket["remaining"], "goal_completed": target_reached,
                    "target": ticket["target"], "executed_decision": executed_action,
                    "next_plan": next_plan}
        except BaseException:
            self.ticket = None
            raise

    def cancel(self):
        self.ticket = None
        self.session.pause("Operation cancelled; no automatic retry")
        return self.status()

    def ask(self, text, on_text=None):
        validate_message(text)
        self.ticket = None
        proposal = explicit_intent(text)
        source, warning, generation = "controller", None, None
        if proposal and proposal["command"] == "status":
            status = self.status()
            message = (f"Current status: {status['status']}; executed {status['experiment_count']} simulated experiments; "
                       f"consecutive IQ passes {status['consecutive_iq_passes']}/2. "
                       f"Completed stages: {', '.join(status['completed_stages']) or 'none'}.")
            proposal = None
        elif proposal:
            message = "I will show the operation scope and wait for confirmation. Simulation only."
        elif not hasattr(self.session.policy, "chat"):
            source = "rule_notice"
            message = "RULE mode uses no language model and cannot answer open-ended questions. Configure HF or connect to a model server. This is a controller message."
        else:
            source = "model"
            stream = ReplyStream(on_text) if on_text is not None else None
            context = self.status()
            context.update(latest_fit=(self.session.runner.observation or {}).get("fit_result", {}),
                           history_turns_shown=len(self.messages) // 2, history_turns_total=self.total_turns,
                           backend={"name": self.info()["backend"], "synthetic": self.info()["synthetic"]})
            try:
                options = ({"on_text": stream.feed} if stream is not None and
                           getattr(self.session.policy, "supports_streaming", False) else {})
                raw = self.session.policy.chat(self.messages + [{"role": "user", "content": text}], context, **options)
                message, proposal, warning = parse_reply(raw)
                if stream is not None:
                    if not stream.finish(message):
                        self.session.runner.emit({"event": "stream_reconciled", "preview_chars": len(stream.emitted),
                                                  "final_chars": len(message)})
                        warning = (warning + "；" if warning else "") + "Streaming preview differed in formatting; the complete final response was preserved"
                generation = getattr(self.session.policy, "last_generation", None)
                if generation and generation.get("hit_output_limit"):
                    warning = (warning + "；" if warning else "") + (
                        f"Generated {generation.get('generated_tokens', '?')} tokens, reaching the configured limit; "
                        "ask 'continue' or increase --chat-max-new-tokens (maximum 4096)")
            except (Exception, KeyboardInterrupt) as exc:
                self.session.runner.emit({"event": "chat_error", "error": str(exc) or "Cancelled",
                    "user": text, "partial_assistant": stream.emitted if stream else "",
                    "partial_only": True})
                self.session.save()
                raise  # Chat failure never changes calibration status or executes a fallback.
        self.session.runner.emit({"event": "chat", "user": text, "assistant": message, "proposal": proposal,
                                  "source": source, "generation": generation, "warning": warning})
        self.session.save()
        self.messages.extend([{"role": "user", "content": text}, {"role": "assistant", "content": message}])
        self.messages = self.messages[-24:]
        self.total_turns += 1
        return {"message": message, "proposal": proposal, "source": source, "warning": warning,
                "generation": generation, "status": self.status()}

    def clear_chat(self):
        self.ticket = None
        self.session.runner.emit({"event": "chat_clear"})
        self.session.save()
        self.messages, self.total_turns = [], 0
        return {"message": "Conversation context cleared; audit records, calibration state and budgets are unchanged."}

    def pause(self):
        self.ticket = None
        self.session.pause()
        return {"message": "Saved and paused; model process released.", "status": self.status()}

    def report(self, target=None):
        if target is not None and (not isinstance(target, str) or not target):
            raise ValueError("Invalid report directory")
        if target:
            destination = Path(target)
        else:
            sequence = self.session.journal.sequence
            destination = self.session.directory / f"export-{sequence:06d}"
            while destination.exists():
                sequence += 1
                destination = self.session.directory / f"export-{sequence:06d}"
        destination = self.session.export(destination)
        result = {"directory": str(destination), "session_id": self.session.metadata["session_id"],
                  "status": self.status()}
        png = destination / "iq_report.png"
        if (destination / "plot_error.json").exists():
            result["plot_note"] = "Numerical report saved, but plotting failed. Inspect plot_error.json; report dependencies may be missing."
        elif png.is_file() and png.stat().st_size <= 1024 * 1024:
            import base64
            import hashlib
            data = png.read_bytes()
            result["preview"] = {"name": "iq_report.png", "sha256": hashlib.sha256(data).hexdigest(),
                                 "base64": base64.b64encode(data).decode("ascii")}
        else:
            result["plot_note"] = "No recorded IQ measurement; no image generated" if not png.exists() else "Image saved on server; exceeds inline preview size limit"
        return result

    def sessions(self):
        return list_sessions(self.home)

    def _switch(self, new):
        try:
            self.session.pause("Switching session")
        except BaseException:
            new.close()
            raise
        self.session.close()
        self.session = new
        self.ticket = None
        self._restore_chat()
        return self.info()

    def new(self, seed=None, qubit_name="unassigned"):
        settings = self.settings_factory() if self.settings_factory else load_settings(self.home)
        if seed is not None:
            settings = Settings.from_dict(settings.to_dict() | {"seed": seed})
        policy = self.policy_factory() if self.policy_factory else None
        try:
            session = CalibrationSession.create(self.home, settings, policy=policy,
                                                qubit_name=qubit_name)
        except BaseException:
            if policy is not None and hasattr(policy, "close"):
                policy.close()
            raise
        return self._switch(session)

    def resume(self, session_id):
        policy = self.policy_factory() if self.policy_factory else None
        try:
            session = CalibrationSession.resume(self.home, session_id, policy=policy)
        except BaseException:
            if policy is not None and hasattr(policy, "close"):
                policy.close()
            raise
        return self._switch(session)

    def close(self):
        self.ticket = None
        try:
            if not self.session.broken:
                self.session.pause("Client disconnected / terminal closed")
        finally:
            self.session.close()
