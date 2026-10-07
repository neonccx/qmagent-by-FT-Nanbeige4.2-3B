"""Interactive UI for the local Agent and optional remote model."""

import argparse
import importlib.metadata
import json
import re
from pathlib import Path
import shlex
import subprocess
import sys
import unicodedata
from contextlib import ExitStack, nullcontext

from . import __version__
from .conversation import resolve_experiment_target, validate_message
from .settings import Settings, app_home, load_settings, save_settings


HELP = """/help                Show commands
/theme [amber|cyan|mono] Change the theme for this session
/status              Current parameters, budgets, stages and IQ metrics
/model               Current policy and model
/plan                Preview the next action without executing
/step                Preview, confirm and execute one step
/run                Choose a registered experiment from a menu
/run <all|steps|experiment> Confirm and calibrate; Ctrl-C cancels
/goal <experiment> [--report] Run prerequisites through a named experiment
/why                 Latest decision rationale
/history             Recent experiment summaries
/clear               Clear conversation context; keep calibration state and audit
/pause               Save state and release the model
/report [new-dir]       Export reports on the Agent host
/sessions            List sessions on the Agent host
/new [qubit-name] [seed] Create a Beijing-time session labelled with its qubit
/resume <session-id>       Resume a session
/quit                Save and quit

Ask naturally; responses and interface use English. Operations require confirmation; no arbitrary shell.
The banner identifies whether this session uses synthetic data or registered hardware."""

RUN_MENU = (
    ("s21", "sq.s21", "Readout resonator search / 读出谐振腔搜索"),
    ("power2d", "sq.s21_power2d", "Readout frequency-power scan / 读出频率-功率二维扫描"),
    ("zpa2d", "sq.s21_zpa2d", "Flux/ZPA sweet-spot scan / 磁通甜点二维扫描"),
    ("spectroscopy", "sq.spectroscopy", "Qubit spectroscopy / 比特能谱"),
    ("rabi", "sq.piamp", "Rabi pi-amplitude calibration / 拉比 π 脉冲幅度"),
    ("ramsey", "sq.ramsey_df", "Ramsey frequency refinement / Ramsey 频率精调"),
    ("t1", "sq.t1", "Energy relaxation T1 / 能量弛豫"),
    ("echo", "sq.t2_echo", "Echo coherence T2 / 回波相干"),
    ("xeb", "sq.xeb", "Single-qubit XEB proxy / 单比特 XEB 代理"),
    ("iq", "sq.iqraw", "Raw IQ acquisition and discrimination / IQ 采集与分割"),
)


def safe_text(value):
    return "".join(char if char in "\n\t" or not unicodedata.category(char).startswith("C")
                   else f"\\u{ord(char):04x}" for char in str(value))


class Terminal:
    def __init__(self, home, session=None, input_fn=input, output_fn=print, service=None, write_fn=None,
                 markdown_factory=None):
        if service is None:
            from .service import AgentService
            service = AgentService(home, session)
        self.home, self.service = home, service
        self.input, self.output = input_fn, output_fn
        self.write = write_fn or (self._stdout_write if output_fn is print else output_fn)
        if markdown_factory is None and output_fn is print and write_fn is None:
            from .markdown_view import create_markdown_view
            markdown_factory = create_markdown_view
        self.markdown_factory = markdown_factory
        self.design = None
        if output_fn is print and write_fn is None:
            from .terminal_design import create_design
            self.design = create_design()
        self.running = True

    @staticmethod
    def _stdout_write(text):
        sys.stdout.write(text)
        sys.stdout.flush()

    @property
    def session(self):
        return self.service.session

    def say(self, value=""):
        self.output(safe_text(value))

    def banner(self):
        info = self.service.info()
        if self.design:
            self.design.banner(info, remote_model=info["settings"]["policy"] == "remote")
            return
        location = "local machine"
        mode = "SIMULATION" if info["synthetic"] else f"REAL HARDWARE ({info['hardware_provider']})"
        self.say(f"QM Calibration Agent {info['version']} · Agent host: {location} · {mode}")
        self.say(f"Session: {info['session_id']}")
        self.model(info)
        self.say("Ask a question; /help lists commands, /quit exits.")
        self.say(f"Local record directory: {info['directory']}")

    def model(self, info=None):
        settings = (info or self.service.info())["settings"]
        if settings["policy"] == "rule":
            self.say("Policy: RULE (deterministic demo, no language model or GPU)")
        else:
            self.say(f"Model: {settings['model']}; adapter: {settings['adapter'] or 'none (unmodified model)'}")
            self.say(f"Request timeout: {settings['request_timeout']:g}s; chat output limit: {settings['chat_max_new_tokens']} tokens; loaded on demand")
            self.say(f"Chat decoding: {settings['decode_backend']} (auto uses CUDA Graph for verified chat models; decisions use HF)")
            self.say(f"Calibration prompt: {settings['prompt_profile']} (minimal is the B0/F0 ablation arm)")
        self.say("System SSH handles server login. This program does not collect passwords or tokens.")

    def show_status(self, result):
        if self.design:
            self.design.status(result)
            return
        self.say(f"Status: {result['status']}; experiments: {result['experiment_count']}/{result['max_steps']}；"
                 f"Consecutive IQ passes: {result['consecutive_iq_passes']}/2")
        self.say("Completed stages: " + (", ".join(result["completed_stages"]) or "none"))
        self.say(json.dumps(result["final_state"], ensure_ascii=False, indent=2))
        if result["final_iq_gate"]:
            self.say("Independent IQ gate: " + json.dumps(result["final_iq_gate"], ensure_ascii=False))
        if result["reason"]:
            self.say("Stop reason: " + result["reason"])

    def status(self):
        self.show_status(self.service.status())

    def plan(self):
        if not self.design:
            self.say("Generating and validating a plan (first request loads the model)...")
        with self.design.waiting("Generating and validating plan") if self.design else nullcontext():
            result = self.service.plan()
        if result["plan"]:
            if self.design:
                self.design.plan(result["plan"])
            else:
                self.say("Pending plan (no experiment executed): ")
                self.say(json.dumps(result["plan"], ensure_ascii=False, indent=2))
        else:
            self.show_status(result["status"])
        return result["plan"]

    def confirm(self, prompt):
        return self.input(safe_text(prompt) + " [y/N] ").strip().lower() in {"y", "yes", "是", "确认"}

    def choose_run(self):
        self.say("Direct diagnostic measurements using current parameters (no prerequisites are completed):")
        for index, (name, tool, description) in enumerate(RUN_MENU, 1):
            self.say(f"  {index:>2}. {name:<12} {tool:<18} {description}")
        self.say("   A. all          Complete calibration loop / 完整闭环")
        self.say("   0. cancel")
        choice = self.input("Choose [1-10, name, A, 0]: ").strip()
        if choice.lower() in {"", "0", "cancel", "取消"}:
            self.say("Cancelled. No experiment was executed.")
            return
        if choice.lower() in {"a", "all", "全部", "完整"}:
            return self.execute(continuous=True)
        if choice.isdigit() and 1 <= int(choice) <= len(RUN_MENU):
            target = RUN_MENU[int(choice) - 1][1]
        else:
            target = resolve_experiment_target(choice)
        if not target:
            raise ValueError("Choose 1-10, a registered experiment name, A/all, or 0/cancel")
        return self.execute(target=target, diagnostic=True)

    def execute(self, continuous=False, limit=None, target=None, report=False, diagnostic=False):
        self.say("Preparing operation scope...")
        mode = "diagnostic" if diagnostic else ("goal" if target else ("run" if continuous else "step"))
        prepared = self.service.prepare(mode=mode, limit=limit, target=target)
        if not prepared["token"]:
            self.show_status(prepared["status"])
            return
        if prepared["plan"]:
            if self.design:
                self.design.plan(prepared["plan"])
            else:
                self.say(json.dumps(prepared["plan"], ensure_ascii=False, indent=2))
        if diagnostic:
            self.say(f"Direct diagnostic: {target} using current parameters; no prerequisite completion or calibration acceptance.")
        elif target:
            required = " → ".join(prepared["required_tools"]) or target
            self.say(f"Goal: {target}; planned registered stages: {required}")
        hardware = not self.service.info()["synthetic"]
        location = "REAL HARDWARE" if hardware else "the simulator"
        prompt = (f"Measure {target} now as diagnostic baseline on {location}?" if diagnostic else
                  (f"Run registered stages through {target} (at most {prepared['limit']} experiments) on {location}?"
                   if target else f"Execute at most {prepared['limit']} experiments on {location}?"))
        if not self.confirm(prompt):
            self.service.cancel()
            self.say("Cancelled. No experiment was executed.")
            return
        final = None
        while True:
            label = "Hardware acquisition running" if hardware else "Simulation running"
            with self.design.waiting(label) if self.design else nullcontext():
                result = self.service.execute(token=prepared["token"])
            experiment = result["experiment"]
            if experiment:
                if self.design:
                    self.design.experiment(experiment)
                else:
                    self.say(f"#{experiment['step']} {experiment['tool']} · reliable={experiment['quality']['reliable']} · "
                             f"Consecutive IQ passes {experiment['iq_passes']}/2")
                    self.say(json.dumps(experiment["fit_result"], ensure_ascii=False))
                if result.get("next_plan"):
                    if self.design:
                        self.design.rationale(result["next_plan"])
                    else:
                        self.say(f"Next decision: {result['next_plan']['next_tool']}")
                        self.say(f"Why: {result['next_plan']['reason']}")
                        action = result["next_plan"]["parameter_action"]
                        if action["updates"] or action["scan"]:
                            self.say("Planned changes: " + json.dumps(action, ensure_ascii=False))
            if result["done"]:
                final = result
                self.say(f"Operation finished: {result['status']['status']}; state saved.")
                if result["status"]["reason"]:
                    self.say(result["status"]["reason"])
                if target and result.get("goal_completed"):
                    label = "Diagnostic recorded" if diagnostic else "Goal reached"
                    self.say(f"{label}: recorded a new {target} measurement.")
                    if report:
                        self.dispatch("/report")
                return final

    def ask(self, text):
        validate_message(text)
        if not self.design:
            self.say("Requesting Agent; response streams live. Ctrl-C cancels...")
        displayed = []
        waiting = ExitStack()
        view = self.markdown_factory() if self.markdown_factory else None
        def on_text(delta):
            waiting.close()
            if view is not None:
                view.feed(delta)
            else:
                if not displayed:
                    self.write("Nanbeige: ")
                self.write(safe_text(delta))
            displayed.append(delta)
        try:
            with waiting:
                if self.design:
                    waiting.enter_context(self.design.waiting("Waiting for first output (first request loads model)"))
                result = self.service.ask(text=text, on_text=on_text)
        except (Exception, KeyboardInterrupt):
            if view is not None:
                view.finish()
            if displayed:
                if view is None:
                    self.write("\n")
                self.say("Response interrupted; text above is incomplete. No proposed operation was executed.")
            raise
        if view is not None:
            if not displayed and result["source"] == "model":
                on_text(result["message"])
            if displayed and "".join(displayed) != result["message"]:
                view.finish(result["message"])
            else:
                view.finish()
        if displayed and view is None:
            self.write("\n")
        label = "Nanbeige" if result["source"] == "model" else "Agent"
        if not displayed:
            self.say(f"{label}：{result['message']}")
        elif view is None and "".join(displayed) != result["message"]:
            self.say("Final response (format reconciled): ")
            self.say(result["message"])
        if result["warning"]:
            self.say("Note: " + result["warning"])
        if result["source"] == "model" and re.search(r"校准|实验|\biq\b|calibrat", text, re.I):
            status = result.get("status", {})
            if "experiment_count" in status:
                self.say(f"Controller: this answer executed no experiments; total {status['experiment_count']} experiments, "
                         f"consecutive IQ passes {status['consecutive_iq_passes']}/2; status {status['status']}。")
        proposal = result["proposal"]
        if proposal:
            self.say("Proposed operation (not executed): " + json.dumps(proposal, ensure_ascii=False))
            command = proposal["command"]
            if command in {"run", "step"}:
                self.execute(continuous=command == "run", limit=proposal.get("limit"))
            elif command == "goal":
                self.execute(continuous=True, target=proposal["target"], report=proposal.get("report", False))
            elif command == "diagnostic":
                self.execute(target=proposal["target"], report=proposal.get("report", False), diagnostic=True)
            elif self.confirm("Perform this view/export operation?"):
                self.dispatch("/" + command)

    def dispatch(self, line):
        line = line.strip()
        line = {"退出": "/quit", "暂停": "/pause", "帮助": "/help"}.get(line, line)
        if not line:
            return
        if not line.startswith("/"):
            self.ask(line)
            return
        command, *args = shlex.split(line)
        no_args = {"/help", "/status", "/model", "/plan", "/step", "/why", "/history",
                   "/clear", "/pause", "/sessions", "/quit", "/exit"}
        if command in no_args and args:
            raise ValueError(f"{command} does not accept extra arguments")
        if command == "/help":
            if self.design:
                self.design.help(HELP)
            else:
                self.say(HELP)
        elif command == "/theme":
            if len(args) > 1:
                raise ValueError("Usage: /theme [amber|cyan|mono]")
            if self.design:
                if args:
                    self.design.set_theme(args[0])
                self.say(f"Current theme: {self.design.theme}; amber / cyan / mono (session only)")
            else:
                self.say("Plain-text mode; install Rich for themes in an interactive terminal.")
        elif command == "/status":
            self.status()
        elif command == "/model":
            self.model()
        elif command == "/plan":
            self.plan()
        elif command == "/step":
            self.execute()
        elif command == "/run":
            if not args:
                self.choose_run()
            elif len(args) == 1 and args[0].lower() in {"a", "all", "全部", "完整"}:
                self.execute(continuous=True)
            elif len(args) == 1 and args[0].isdigit() and 1 <= int(args[0]) <= 1000:
                self.execute(continuous=True, limit=int(args[0]))
            else:
                target = resolve_experiment_target(" ".join(args))
                if not target:
                    raise ValueError("Usage: /run [1..1000|experiment]")
                self.execute(target=target, diagnostic=True)
        elif command == "/goal":
            wants_report = "--report" in args
            names = [item for item in args if item != "--report"]
            target = resolve_experiment_target(" ".join(names))
            if not target or len(names) != len(args) - int(wants_report):
                raise ValueError("Usage: /goal <experiment> [--report]")
            self.execute(continuous=True, target=target, report=wants_report)
        elif command == "/why":
            self.say(json.dumps(self.service.why(), ensure_ascii=False, indent=2))
        elif command == "/history":
            self.say(json.dumps(self.service.history(), ensure_ascii=False, indent=2))
        elif command == "/clear":
            self.say(self.service.clear_chat()["message"])
        elif command == "/pause":
            self.say(self.service.pause()["message"])
        elif command == "/report":
            if len(args) > 1:
                raise ValueError("Usage: /report [new-output-directory]")
            result = self.service.report(target=args[0] if args else None)
            self.say("Report exported locally: " + result["directory"])
            if result.get("plot_note"):
                self.say(result["plot_note"])
            from .report_preview import save_preview
            page = save_preview(result, self.home)
            if page:
                self.say("IQ report (local): " + str(page))
                if sys.platform == "darwin" and self.output is print:
                    import webbrowser
                    webbrowser.open(page.resolve().as_uri())
        elif command == "/sessions":
            self.say(json.dumps(self.service.sessions(), ensure_ascii=False, indent=2))
        elif command == "/new":
            if len(args) > 2:
                raise ValueError("Usage: /new [qubit-name] [seed]")
            qubit_name = args[0] if args else self.input("Qubit name (for example q0): ").strip()
            if not qubit_name:
                raise ValueError("Qubit name is required")
            if len(args) == 2 and not args[1].isdigit():
                raise ValueError("Seed must be a nonnegative integer")
            self.service.new(qubit_name=qubit_name, seed=int(args[1]) if len(args) == 2 else None)
            self.banner()
        elif command == "/resume":
            if len(args) != 1:
                raise ValueError("Usage: /resume <session-id>")
            self.service.resume(session_id=args[0])
            self.banner()
        elif command in {"/quit", "/exit"}:
            self.running = False
        else:
            self.say("Unknown command; nothing executed. Use /help. Arbitrary shell commands are not supported.")

    def loop(self):
        try:
            self.banner()
            try:
                import readline
                commands = [line.split()[0] for line in HELP.splitlines() if line.startswith("/")]
                readline.set_completer(lambda text, index: ([item for item in commands if item.startswith(text)] + [None])[index])
                readline.parse_and_bind("tab: complete")
                # Never persist raw terminal input: it may contain pasted secrets.
            except ImportError:
                pass
            while self.running:
                try:
                    if self.design and self.input is input:
                        line = self.design.read_input(HELP)
                    else:
                        line = self.input(self.design.prompt() if self.design else "You> ")
                    self.dispatch(line)
                except EOFError:
                    self.running = False
                except KeyboardInterrupt:
                    try:
                        self.service.cancel()
                        self.say("\nCancelled and saved. Continue chatting or /quit.")
                    except (OSError, ValueError, RuntimeError):
                        self.say("\nConnection closed. Resume the session to inspect the last committed state.")
                        return 1
                except (ValueError, OSError, RuntimeError, KeyError) as exc:
                    self.say(f"Error: {exc}")
                    if not getattr(self.service, "connected", True) or self.service.info()["broken"]:
                        return 1
            self.say("Saving and releasing the model...")
            return 0
        finally:
            self.service.close()


def add_options(parser):
    parser.add_argument("--policy", choices=("rule", "hf"))
    parser.add_argument("--decode-backend", choices=("auto", "hf", "cuda_graph"))
    parser.add_argument("--prompt-profile", choices=("minimal", "skill"))
    parser.add_argument("--model", type=Path)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--base-only", action="store_true", help="Clear configured adapter")
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=None)
    for name in ("max-input-tokens", "max-new-tokens", "chat-max-new-tokens", "seed", "max-steps", "max-tool-calls"):
        parser.add_argument("--" + name, type=int)
    parser.add_argument("--request-timeout", type=float)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--backend", choices=("auto", "hardware", "physical", "legacy"))
    parser.add_argument("--simulation-profile", choices=("nominal", "drift", "quasistatic", "ambiguous"))


def merged_settings(home, args):
    values = load_settings(home).to_dict()
    if getattr(args, "policy", None) == "rule":
        values.update(model=None, adapter=None, trust_remote_code=False, prompt_profile="skill")
    for key in list(values):
        value = getattr(args, key, None)
        if value is not None:
            values[key] = str(value.expanduser().resolve()) if isinstance(value, Path) else value
    if getattr(args, "base_only", False):
        if getattr(args, "adapter", None):
            raise ValueError("--base-only conflicts with --adapter")
        values["adapter"] = None
    return Settings.from_dict(values)


def doctor(home):
    settings = load_settings(home)
    from .backends import make_backend
    backend = make_backend(settings)
    report = {"version": __version__, "home": str(home), "policy": settings.policy,
              "backend": backend.backend_name,
              "synthetic": bool(getattr(backend, "synthetic", True)),
              "hardware_provider": getattr(backend, "provider_name", None),
              "authentication": "SSH; no credentials stored",
              "gpu_probe_performed": False, "packages": {}, "errors": []}
    names = ["numpy", "scipy"] + (["torch", "transformers", "peft", "accelerate"] if settings.policy == "hf" else [])
    for name in names:
        try:
            report["packages"][name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][name] = None
            report["errors"].append("Missing package: " + name)
    if settings.policy == "hf":
        for key in ("model", "adapter"):
            value = getattr(settings, key)
            report[key] = value
            if value and not Path(value).is_dir():
                report["errors"].append(f"Missing local {key} directory")
        report["note"] = "No model/GPU allocation performed; this is not a successful inference test"
    print(safe_text(json.dumps(report, ensure_ascii=False, indent=2)))
    if hasattr(backend, "close"):
        backend.close()
    return int(bool(report["errors"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description="QM Calibration Agent — simulation or explicitly registered hardware")
    parser.add_argument("--home", type=Path, help="Config/session root (or QM_AGENT_HOME)")
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command")
    add_options(sub.add_parser("shell", aliases=["chat"], help="Server-local Agent terminal"))
    add_options(sub.add_parser("configure", help="Save local model settings"))
    sub.add_parser("doctor")
    sub.add_parser("sessions")
    sub.add_parser("resume").add_argument("session_id")
    remote = sub.add_parser("remote", help="Configure an SSH model server; Agent remains local")
    remote.add_argument("--host", required=True)
    remote.add_argument("--project", required=True)
    remote.add_argument("--control-path", required=True, type=Path)
    remote.add_argument("--server-home")
    remote.add_argument("--local-results", type=Path, help=argparse.SUPPRESS)
    sub.add_parser("connect", help="Run the local Agent with a remote model").add_argument("--resume", dest="session_id")
    sub.add_parser("login", help="Open system SSH; enter credentials there yourself")
    args = parser.parse_args(argv)
    home = app_home(args.home)
    try:
        if args.command == "remote":
            from .remote import RemoteProfile, save_remote
            profile = RemoteProfile.from_dict({"host": args.host, "project": args.project,
                "control_path": str(args.control_path.expanduser().resolve()), "home": args.server_home,
                "local_results": str(args.local_results.expanduser().resolve()) if args.local_results else None})
            save_remote(home, profile)
            print(safe_text(f"Remote configuration saved: {home / 'remote.json'}; run qm-agent or qm-agent connect."))
            return 0
        if args.command in {"connect", "login"} or (args.command is None and (home / "remote.json").exists()):
            from .remote import RemoteModelPolicy, load_remote
            profile = load_remote(home)
            if args.command == "login":
                return subprocess.call(profile.login_command())
            notice = lambda message: print(safe_text(message), flush=True)
            def policy_factory():
                return RemoteModelPolicy(profile, on_wait=notice)
            policy = policy_factory()
            local = load_settings(home).to_dict()
            for key in ("model", "adapter", "trust_remote_code", "max_input_tokens",
                        "max_new_tokens", "chat_max_new_tokens", "decode_backend",
                        "prompt_profile", "fit_update_tool", "request_timeout"):
                local[key] = policy.server_settings[key]
            local["policy"] = "remote"
            settings = Settings.from_dict(local)
            from .service import AgentService
            from .session import CalibrationSession
            session_id = getattr(args, "session_id", None)
            try:
                session = (CalibrationSession.resume(home, session_id, policy=policy) if session_id
                           else CalibrationSession.create(home, settings, policy=policy))
            except BaseException:
                policy.close()
                raise
            service = AgentService(home, session, policy_factory=policy_factory,
                                   settings_factory=lambda: settings)
            return Terminal(home, service=service).loop()
        if args.command == "doctor":
            return doctor(home)
        if args.command == "configure":
            settings = merged_settings(home, args)
            if settings.policy == "hf":
                for name in ("model", "adapter"):
                    value = getattr(settings, name)
                    if value and not Path(value).is_dir():
                        raise ValueError(f"Missing local {name} directory")
            save_settings(home, settings)
            print(safe_text(f"Configuration saved: {home / 'config.json'}; applies to new sessions only; existing sessions keep their configuration."))
            return 0
        from .session import CalibrationSession, list_sessions
        if args.command == "sessions":
            print(safe_text(json.dumps(list_sessions(home), ensure_ascii=False, indent=2)))
            return 0
        session = (CalibrationSession.resume(home, args.session_id) if args.command == "resume"
                   else CalibrationSession.create(home, merged_settings(home, args)))
        return Terminal(home, session).loop()
    except (ValueError, OSError, RuntimeError, KeyError) as exc:
        print(safe_text(f"qm-agent: {exc}"), file=sys.stderr)
        return 1
