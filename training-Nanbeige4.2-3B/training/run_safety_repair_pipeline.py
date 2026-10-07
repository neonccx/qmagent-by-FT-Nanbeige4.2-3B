#!/usr/bin/env python3
"""Continue an audited adapter on a train-only safety curriculum and gate release."""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--curriculum", type=Path, required=True)
    parser.add_argument("--initial-run", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--eval-limit", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--closed-loop-seed", type=int, required=True)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--resume-evaluation", action="store_true",
                        help="Resume a failed evaluation after successful training; never retrain")
    args = parser.parse_args()
    training_root = Path(__file__).resolve().parents[1]
    agent_root = training_root.parent / "qcal-agent"
    if not (agent_root / "src/qmagent/__init__.py").is_file():
        raise ValueError("Missing ../qcal-agent/src; both folders are required for training")
    model = args.model.resolve()
    curriculum = args.curriculum.resolve()
    initial_run = args.initial_run.resolve()
    root = args.run_dir.resolve()
    initial_adapter = initial_run / "training" / "final_adapter"
    if not (initial_adapter / "adapter_model.safetensors").is_file():
        raise ValueError("Initial run has no final adapter")
    manifest = json.loads((curriculum / "manifest.json").read_text())
    if manifest.get("curriculum_schema") != "qcal-repair-curriculum-1.0":
        raise ValueError("Expected an audited repair curriculum")
    if args.epochs <= 0 or args.learning_rate <= 0:
        raise ValueError("Epochs and learning rate must be positive")
    root.mkdir(parents=True, exist_ok=args.resume_evaluation)
    audit_file = root / "tokenizer_audit.json"
    state = {
        "schema": "qcal-safety-repair-run-1.0",
        "status": "running",
        "pid": os.getpid(),
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model),
        "curriculum": str(curriculum),
        "initial_run": str(initial_run),
        "initial_adapter_sha256": sha256(initial_adapter / "adapter_model.safetensors"),
        "curriculum_manifest_sha256": sha256(curriculum / "manifest.json"),
        "closed_loop_seed": args.closed_loop_seed,
        "gpu": args.gpu,
        "stages": [],
        "scope": "Synthetic single-qubit safety curriculum; frozen evaluation; no hardware access",
    }
    if args.resume_evaluation:
        previous = json.loads((root / "status.json").read_text())
        for key in ("model", "curriculum", "initial_run", "initial_adapter_sha256",
                    "curriculum_manifest_sha256", "closed_loop_seed"):
            if previous.get(key) != state[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
        if not any(item["stage"] == "training" and item["status"] == "completed"
                   for item in previous["stages"]):
            raise ValueError("Evaluation resume requires completed training")
        if not (root / "training" / "final_adapter" / "adapter_model.safetensors").is_file():
            raise ValueError("Completed adapter is missing")
        state = previous
        state.update(status="running", pid=os.getpid(), gpu=args.gpu)
        state.pop("error", None)
        state.setdefault("resume_history", []).append({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "adapter_sha256": sha256(root / "training" / "final_adapter" / "adapter_model.safetensors"),
            "mode": "evaluation_only"})
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", PYTHONDONTWRITEBYTECODE="1",
               OMP_NUM_THREADS="4", MKL_NUM_THREADS="4", PYTHONUNBUFFERED="1",
               PYTHONPATH=f"{agent_root / 'src'}:{training_root / 'training'}")

    def save() -> None:
        temporary = root / "status.json.tmp"
        temporary.write_text(json.dumps(state, indent=2) + "\n")
        temporary.replace(root / "status.json")

    def run(name: str, command: list[object], scientific_failure: bool = False) -> None:
        if args.resume_evaluation and any(item["stage"] == name and item["status"] == "completed"
                                          for item in state["stages"]):
            return
        item = {"stage": name, "command": list(map(str, command)), "status": "running"}
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        attempt = sum(item["stage"] == name for item in state["stages"])
        log_path = root / (f"{name}.log" if attempt == 1 else f"{name}.attempt_{attempt}.log")
        with log_path.open("x") as log:
            result = subprocess.run(list(map(str, command)), cwd=training_root, env=env,
                                    stdout=log, stderr=subprocess.STDOUT)
        item.update(returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {name}.log")

    python = sys.executable
    adapter = root / "training" / "final_adapter"
    try:
        run("dataset_audit", [python, training_root / "training" / "audit_dataset.py", curriculum])
        run("tokenizer_audit", [python, training_root / "training" / "audit_tokens.py",
            "--dataset", curriculum, "--model", model, "--output", audit_file])
        run("training", [python, training_root / "training" / "train_lora.py", "--model", model,
            "--initial-adapter", initial_adapter,
            "--train-file", curriculum / "train.jsonl",
            "--validation-file", curriculum / "validation.jsonl",
            "--output-dir", root / "training",
            "--baseline-metrics", initial_run / "adapted_test" / "metrics.json",
            "--dataset-audit", audit_file, "--max-length", "20480",
            "--epochs", args.epochs, "--learning-rate", args.learning_rate,
            "--micro-batch", "2", "--gradient-accumulation", "5",
            "--logging-steps", "5", "--eval-steps", "25", "--save-steps", "25",
            "--assistant-only-projection", "--no-load-best-model"])
        if not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError("Safety continuation did not produce final_adapter")
        for split in ("test", "ood"):
            adapted = root / f"adapted_{split}"
            evaluation_command = [python, training_root / "training" / "evaluate_policy.py",
                "--model", model, "--adapter", adapter, "--trust-remote-code",
                "--test-file", curriculum / f"{split}.jsonl", "--output", adapted,
                "--limit", args.eval_limit, "--batch-size", args.batch_size]
            if args.resume_evaluation and (adapted / "config.json").is_file():
                evaluation_command += ["--resume"]
            run(f"adapted_{split}", evaluation_command)
            run(f"adapted_controller_{split}", [python, training_root / "training" / "score_policy_predictions.py",
                "--test-file", curriculum / f"{split}.jsonl",
                "--predictions", adapted / "predictions.jsonl",
                "--output", adapted / "controller_score.json"])
            run(f"comparison_{split}", [python, training_root / "training" / "compare_policy.py",
                "--comparison", "continuation",
                "--baseline", initial_run / f"adapted_{split}", "--adapted", adapted,
                "--output", root / f"comparison_{split}.json"])
        run("closed_loop_adapted", [python, "-m", "qmagent.cli", "run", "--policy", "hf",
            "--decode-backend", "hf", "--model", model, "--adapter", adapter,
            "--trust-remote-code", "--backend", "physical", "--seed", args.closed_loop_seed,
            "--episodes", "3", "--max-steps", "45", "--max-tool-calls", "10",
            "--request-timeout", "300", "--output-dir", root / "closed_loop_adapted"],
            scientific_failure=True)
        acceptance = {"schema": "qcal-release-gate-1.0", "checks": {}}
        for split in ("test", "ood"):
            metrics = json.loads((root / f"adapted_{split}" / "metrics.json").read_text())
            controller = json.loads((root / f"adapted_{split}" / "controller_score.json").read_text())
            comparison = json.loads((root / f"comparison_{split}.json").read_text())
            acceptance["checks"][split] = {
                "valid_call_rate": metrics["valid_rate"] >= 0.98,
                "next_tool_accuracy": metrics["next_tool_correct_rate"] >= 0.90,
                "argument_accuracy": metrics["arguments_correct_rate"] >= 0.85,
                "controller_executable_rate": controller["controller_executable_rate"] >= 0.98,
                "no_next_tool_regression": comparison["metrics"]["next_tool_correct"]["delta_percentage_points"] >= 0,
            }
        closed = json.loads((root / "closed_loop_adapted" / "summary.json").read_text())
        acceptance["checks"]["closed_loop"] = {
            "at_least_two_of_three_accepted": closed["statuses"]["accepted"] >= 2,
            "no_invalid_policy_actions": closed["statuses"]["invalid_action"] == 0,
            "no_tool_or_policy_errors": (closed["statuses"]["tool_error"] == 0
                                          and closed["statuses"]["policy_error"] == 0),
        }
        acceptance["all_passed"] = all(value for group in acceptance["checks"].values()
                                            for value in group.values())
        (root / "release_acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        if not acceptance["all_passed"]:
            raise RuntimeError("Release acceptance gates failed; do not package or publish this adapter")
        release = root / "release"
        release.mkdir()
        artifact = release / "qcal-agent-1.0.0-lora.safetensors"
        artifact.write_bytes((adapter / "adapter_model.safetensors").read_bytes())
        evidence = {
            "version": "1.0.0",
            "model": str(model),
            "adapter_sha256": sha256(artifact),
            "parent_adapter_sha256": state["initial_adapter_sha256"],
            "curriculum_manifest_sha256": state["curriculum_manifest_sha256"],
            "release_acceptance_sha256": sha256(root / "release_acceptance.json"),
            "comparison_test_sha256": sha256(root / "comparison_test.json"),
            "comparison_ood_sha256": sha256(root / "comparison_ood.json"),
        }
        (release / "release_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
        state["status"] = "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
