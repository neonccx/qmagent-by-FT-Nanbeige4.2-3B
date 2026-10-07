#!/usr/bin/env python3
"""Run the audited QCal Agent 1.0 baseline, LoRA, evaluation and packaging pipeline."""

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


def next_log_path(root: Path, name: str, attempt: int) -> Path:
    """Keep copied historical logs intact when resuming on a different host."""
    while True:
        path = root / (f"{name}.log" if attempt == 1 else f"{name}.attempt_{attempt}.log")
        if not path.exists():
            return path
        attempt += 1


def reconcile_interrupted_stages(state: dict) -> None:
    """Record a stage left running by a pipeline launch error as failed."""
    for item in state["stages"]:
        if item["status"] == "running":
            item.update(status="failed", error=state.get("error", "prior pipeline interrupted"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--gpu", default="0", help="One CUDA device index; the pipeline is single-process")
    parser.add_argument("--eval-limit", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--closed-loop-seed", type=int, default=2026091370,
                        help="First seed for the three closed-loop release episodes")
    parser.add_argument("--save-steps", type=int, default=100,
                        help="LoRA checkpoint interval; use a smaller value on preemptible hosts")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--eval-steps", type=int, default=150)
    parser.add_argument("--load-best-model", action="store_true",
                        help="Select minimum validation-loss checkpoint; eval/save intervals must match")
    parser.add_argument("--resume-from-checkpoint", type=Path,
                        help="Resume an interrupted training stage in this run; preserve completed stages")
    parser.add_argument("--resume-training-source-sha256",
                        help="Explicit reviewed loader-fix source hash; records old/new training code on resume")
    args = parser.parse_args()
    training_root = Path(__file__).resolve().parents[1]
    agent_root = training_root.parent / "qcal-agent"
    if not (agent_root / "src/qmagent/__init__.py").is_file():
        raise ValueError("Missing ../qcal-agent/src; both folders are required for training")
    model, dataset, root = args.model.resolve(), args.dataset.resolve(), args.run_dir.resolve()
    if not (model / "config.json").is_file():
        raise ValueError(f"Model checkpoint is missing: {model}")
    if not (dataset / "manifest.json").is_file():
        raise ValueError(f"Dataset is missing: {dataset}")
    if args.eval_limit < 1 or args.batch_size < 1 or args.save_steps < 1:
        raise ValueError("Evaluation limit, batch size and save steps must be positive")
    if args.epochs <= 0 or args.eval_steps < 1:
        raise ValueError("Epochs and evaluation interval must be positive")
    if args.load_best_model and args.save_steps != args.eval_steps:
        raise ValueError("Best-model selection requires matching save/evaluation intervals")
    root.mkdir(parents=True, exist_ok=bool(args.resume_from_checkpoint))
    source_paths = [training_root / "training" / "train_lora.py",
        training_root / "training" / "assistant_loss.py", Path(__file__).resolve(),
        training_root / "training" / "evaluate_policy.py", agent_root / "src" / "qmagent" / "policies.py"]
    state = {"schema": "qcal-training-run-1.0", "pid": os.getpid(), "status": "running",
        "started": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "model": str(model), "dataset": str(dataset), "gpu": args.gpu,
        "closed_loop_seed": args.closed_loop_seed, "stages": [],
        "scope": "Synthetic single-qubit research; no hardware or coupler access",
        "source_sha256": {str(path.relative_to(training_root.parent)): sha256(path) for path in source_paths}}
    if args.resume_from_checkpoint:
        previous = json.loads((root / "status.json").read_text())
        if previous.get("status") != "failed" or previous.get("current_stage") != "training":
            raise ValueError("Checkpoint resume requires a failed training stage")
        reconcile_interrupted_stages(previous)
        for key in ("model", "dataset", "closed_loop_seed"):
            if previous.get(key) != state[key]:
                raise ValueError(f"Resume provenance mismatch: {key}")
        for path, checksum in previous["source_sha256"].items():
            source = training_root.parent / path
            if not path.endswith("training/run_training_pipeline.py") and sha256(source) != checksum:
                if not (path.endswith("training/train_lora.py") and args.resume_training_source_sha256
                        == sha256(source)):
                    raise ValueError(f"Resume source changed: {path}")
        checkpoint = args.resume_from_checkpoint.resolve()
        if checkpoint.parent != root / "training":
            raise ValueError("Checkpoint must belong to this run")
        for name in ("adapter_model.safetensors", "optimizer.pt", "scheduler.pt", "trainer_state.json"):
            if not (checkpoint / name).is_file():
                raise ValueError(f"Incomplete checkpoint: {name}")
        state = previous
        state.update(status="running", pid=os.getpid(), gpu=args.gpu)
        state.pop("error", None)
        state.pop("finished", None)
        state.setdefault("resume_history", []).append({
            "at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "checkpoint": str(checkpoint), "adapter_sha256": sha256(checkpoint / "adapter_model.safetensors"),
            "original_training_source_sha256": next(value for key, value in previous["source_sha256"].items()
                                                        if key.endswith("training/train_lora.py")),
            "resumed_training_source_sha256": sha256(training_root / "training/train_lora.py"),
            "rng_loader_sha256": sha256(training_root / "training/checkpoint_rng.py")
                if (training_root / "training/checkpoint_rng.py").is_file() else None,
            "pipeline_sha256": sha256(Path(__file__).resolve())})
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
        if args.resume_from_checkpoint:
            prior = [item for item in state["stages"] if item["stage"] == name]
            if any(item["status"] == "completed" for item in prior):
                return
            if name == "training":
                if not prior or prior[0]["command"] != list(map(str, command)):
                    raise ValueError("Training parameters changed during resume")
                command = command + ["--resume-from-checkpoint", checkpoint]
        attempt = 1 + sum(item["stage"] == name for item in state["stages"])
        log_path = next_log_path(root, name, attempt)
        item = {"stage": name, "command": list(map(str, command)), "status": "running",
                "log": str(log_path)}
        state["stages"].append(item)
        state["current_stage"] = name
        save()
        try:
            with log_path.open("x") as log:
                result = subprocess.run(list(map(str, command)), cwd=training_root, env=env,
                                        stdout=log, stderr=subprocess.STDOUT)
        except BaseException as error:
            item.update(status="failed", error=f"{type(error).__name__}: {error}")
            save()
            raise
        item.update(returncode=result.returncode,
                    status="completed" if result.returncode == 0 else "failed")
        save()
        if result.returncode and not scientific_failure:
            raise RuntimeError(f"{name} exited {result.returncode}; inspect {name}.log")

    python = sys.executable
    adapter = root / "training" / "final_adapter"
    audit_file = root / "tokenizer_audit.json"
    try:
        run("dataset_audit", [python, training_root / "training" / "audit_dataset.py", dataset])
        run("tokenizer_audit", [python, training_root / "training" / "audit_tokens.py",
            "--dataset", dataset, "--model", model, "--output", audit_file])
        run("assistant_loss_verification", [python, training_root / "training" / "verify_assistant_loss.py",
            "--model", model, "--dataset", dataset / "validation.jsonl",
            "--output", root / "assistant_loss_verification.json"])
        for split in ("test", "ood"):
            run(f"baseline_{split}", [python, training_root / "training" / "evaluate_policy.py",
                "--model", model, "--trust-remote-code", "--test-file", dataset / f"{split}.jsonl",
                "--output", root / f"baseline_{split}", "--limit", args.eval_limit,
                "--batch-size", args.batch_size])
            run(f"baseline_controller_{split}", [python, training_root / "training" / "score_policy_predictions.py",
                "--test-file", dataset / f"{split}.jsonl",
                "--predictions", root / f"baseline_{split}" / "predictions.jsonl",
                "--output", root / f"baseline_{split}" / "controller_score.json"])
        run("training", [python, training_root / "training" / "train_lora.py", "--model", model,
            "--train-file", dataset / "train.jsonl", "--validation-file", dataset / "validation.jsonl",
            "--output-dir", root / "training", "--baseline-metrics", root / "baseline_test" / "metrics.json",
            "--dataset-audit", audit_file, "--max-length", "20480", "--epochs", str(args.epochs),
            "--micro-batch", "2", "--gradient-accumulation", "5", "--logging-steps", "5",
            "--eval-steps", str(args.eval_steps), "--save-steps", str(args.save_steps), "--assistant-only-projection",
            "--load-best-model" if args.load_best_model else "--no-load-best-model"])
        if not (adapter / "adapter_config.json").is_file():
            raise RuntimeError("Training did not produce final_adapter")
        for split in ("test", "ood"):
            adapted = root / f"adapted_{split}"
            run(f"adapted_{split}", [python, training_root / "training" / "evaluate_policy.py",
                "--model", model, "--adapter", adapter, "--trust-remote-code",
                "--test-file", dataset / f"{split}.jsonl", "--output", adapted,
                "--limit", args.eval_limit, "--batch-size", args.batch_size])
            run(f"adapted_controller_{split}", [python, training_root / "training" / "score_policy_predictions.py",
                "--test-file", dataset / f"{split}.jsonl", "--predictions", adapted / "predictions.jsonl",
                "--output", adapted / "controller_score.json"])
            run(f"comparison_{split}", [python, training_root / "training" / "compare_policy.py",
                "--baseline", root / f"baseline_{split}", "--adapted", adapted,
                "--output", root / f"comparison_{split}.json"])
        for arm in ("base", "adapted"):
            command = [python, "-m", "qmagent.cli", "run", "--policy", "hf",
                "--decode-backend", "hf", "--model", model, "--trust-remote-code",
                "--backend", "physical", "--seed", str(args.closed_loop_seed), "--episodes", "3",
                "--max-steps", "45", "--max-tool-calls", "10", "--request-timeout", "300",
                "--output-dir", root / f"closed_loop_{arm}"]
            if arm == "adapted":
                command += ["--adapter", adapter]
            run(f"closed_loop_{arm}", command, scientific_failure=True)
        acceptance = {"schema": "qcal-release-gate-1.0", "checks": {}}
        for split in ("test", "ood"):
            metrics = json.loads((root / f"adapted_{split}" / "metrics.json").read_text())
            controller = json.loads((root / f"adapted_{split}" / "controller_score.json").read_text())
            comparison = json.loads((root / f"comparison_{split}.json").read_text())
            checks = {
                "valid_call_rate": metrics["valid_rate"] >= 0.98,
                "next_tool_accuracy": metrics["next_tool_correct_rate"] >= 0.90,
                "argument_accuracy": metrics["arguments_correct_rate"] >= 0.85,
                "controller_executable_rate": controller["controller_executable_rate"] >= 0.98,
                "no_next_tool_regression": comparison["metrics"]["next_tool_correct"]["delta_percentage_points"] >= 0,
            }
            acceptance["checks"][split] = checks
        closed_loop = json.loads((root / "closed_loop_adapted" / "summary.json").read_text())
        acceptance["checks"]["closed_loop"] = {
            "at_least_two_of_three_accepted": closed_loop["statuses"]["accepted"] >= 2,
            "no_invalid_policy_actions": closed_loop["statuses"]["invalid_action"] == 0,
            "no_tool_or_policy_errors": (closed_loop["statuses"]["tool_error"] == 0
                                          and closed_loop["statuses"]["policy_error"] == 0),
        }
        acceptance["all_passed"] = all(
            value for group in acceptance["checks"].values() for value in group.values())
        (root / "release_acceptance.json").write_text(json.dumps(acceptance, indent=2) + "\n")
        if not acceptance["all_passed"]:
            raise RuntimeError("Release acceptance gates failed; do not package or publish this adapter")
        release = root / "release"
        release.mkdir()
        artifact = release / "qcal-agent-1.0.0-lora.safetensors"
        artifact.write_bytes((adapter / "adapter_model.safetensors").read_bytes())
        evidence = {"version": "1.0.0", "model": str(model), "dataset": str(dataset),
            "adapter_sha256": sha256(artifact), "dataset_manifest_sha256": sha256(dataset / "manifest.json"),
            "release_acceptance_sha256": sha256(root / "release_acceptance.json"),
            "comparison_test_sha256": sha256(root / "comparison_test.json"),
            "comparison_ood_sha256": sha256(root / "comparison_ood.json")}
        (release / "release_manifest.json").write_text(json.dumps(evidence, indent=2) + "\n")
        state["status"] = "completed_with_scientific_failures" if any(
            item["status"] == "failed" for item in state["stages"]) else "completed"
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        state["finished"] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save()


if __name__ == "__main__":
    main()
