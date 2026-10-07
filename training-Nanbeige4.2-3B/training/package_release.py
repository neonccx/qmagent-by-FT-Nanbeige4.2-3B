#!/usr/bin/env python3
"""Build a portable PEFT adapter bundle only from a passing audited run."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path


TOKENIZER_FILES = ("tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json")
REQUIRED_STAGES = (
    "training", "adapted_test", "adapted_controller_test", "comparison_test",
    "adapted_ood", "adapted_controller_ood", "comparison_ood", "closed_loop_adapted",
)


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def package(run_dir: Path, base_model_path: Path, base_model_id: str, model_card: Path) -> dict:
    run_dir, base_model_path = run_dir.resolve(), base_model_path.resolve()
    release = run_dir / "release"
    old_manifest = release / "release_manifest.json"
    gate_path = run_dir / "release_acceptance.json"
    gate, status = read_json(gate_path), read_json(run_dir / "status.json")
    if gate.get("all_passed") is not True or any(
        value is not True for group in gate["checks"].values() for value in group.values()
    ):
        raise ValueError("Release acceptance checks have not all passed")
    if status.get("status") not in {"completed", "completed_with_scientific_failures"}:
        raise ValueError("Training/evaluation pipeline has not finished")
    for stage in REQUIRED_STAGES:
        if not any(item.get("stage") == stage and item.get("status") == "completed"
                   for item in status["stages"]):
            raise ValueError(f"Required stage did not complete: {stage}")

    previous = read_json(old_manifest)
    version = previous["version"]
    if version != "1.0.0" or not base_model_id or "/" not in base_model_id:
        raise ValueError("Expected version 1.0.0 and a public base-model repository ID")
    source = run_dir / "training" / "final_adapter"
    weights = source / "adapter_model.safetensors"
    config = read_json(source / "adapter_config.json")
    if Path(config["base_model_name_or_path"]).resolve() != base_model_path:
        raise ValueError("Adapter was trained with a different base-model path")
    if config.get("peft_type") != "LORA" or config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Unexpected adapter format")
    weights_sha = sha256(weights)
    if (weights_sha != previous["adapter_sha256"] or
            weights_sha != sha256(release / "qcal-agent-1.0.0-lora.safetensors")):
        raise ValueError("Packaged adapter weights differ from the audited final adapter")
    tokenizer_hashes = {}
    for name in TOKENIZER_FILES:
        base_hash = sha256(base_model_path / name)
        if base_hash != sha256(source / name):
            raise ValueError(f"Adapter tokenizer differs from the base model: {name}")
        tokenizer_hashes[name] = base_hash
    dataset = Path(status["dataset"])
    dataset_manifest = read_json(dataset / "manifest.json")
    metrics = {}
    for split in ("test", "ood"):
        report = read_json(run_dir / f"adapted_{split}" / "metrics.json")
        if report["dataset_sha256"] != dataset_manifest["split_sha256"][split]:
            raise ValueError(f"{split} metrics do not use the frozen dataset")
        metrics[split] = {
            "sample_count": report["sample_count"],
            "valid_rate": report["valid_rate"],
            "next_tool_correct_rate": report["next_tool_correct_rate"],
            "arguments_correct_rate": report["arguments_correct_rate"],
            "controller_executable_rate": read_json(
                run_dir / f"adapted_{split}" / "controller_score.json"
            )["controller_executable_rate"],
        }
    closed_loop = read_json(run_dir / "closed_loop_adapted" / "summary.json")
    bundle_name = f"qcal-agent-{version}-adapter"
    bundle = release / bundle_name
    archive = release / f"{bundle_name}.tar.gz"
    manifest_path = release / "portable_manifest.json"
    if any(path.exists() for path in (bundle, archive, manifest_path)):
        raise FileExistsError("Portable release already exists; never overwrite a published candidate")

    with tempfile.TemporaryDirectory(prefix="bundle-staging-", dir=release) as temporary:
        staging = Path(temporary) / bundle_name
        staging.mkdir()
        shutil.copy2(weights, staging / "adapter_model.safetensors")
        portable_config = {**config, "base_model_name_or_path": base_model_id}
        (staging / "adapter_config.json").write_text(
            json.dumps(portable_config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        shutil.copy2(model_card, staging / "README.md")
        staged_archive = Path(temporary) / archive.name
        with tarfile.open(staged_archive, "w:gz") as tar:
            tar.add(staging, arcname=bundle_name)
        metadata = {
            "schema": "qcal-portable-adapter-release-1.0",
            "version": version,
            "scope": "Synthetic single-qubit research; no real-hardware or coupler validation",
            "base_model_id": base_model_id,
            "base_config_sha256": sha256(base_model_path / "config.json"),
            "tokenizer_sha256": tokenizer_hashes,
            "adapter_model_sha256": weights_sha,
            "adapter_config_sha256": sha256(staging / "adapter_config.json"),
            "bundle_sha256": sha256(staged_archive),
            "release_acceptance_sha256": sha256(gate_path),
            "source_release_manifest_sha256": sha256(old_manifest),
            "frozen_split_sha256": {split: dataset_manifest["split_sha256"][split]
                                    for split in ("test", "ood")},
            "metrics": metrics,
            "closed_loop_accepted": closed_loop["statuses"]["accepted"],
            "closed_loop_episodes": closed_loop["episodes"],
        }
        staging.rename(bundle)
        staged_archive.rename(archive)
        manifest_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--base-model-path", type=Path, required=True)
    parser.add_argument("--base-model-id", default="Nanbeige/Nanbeige4.2-3B")
    parser.add_argument("--model-card", type=Path, default=Path(__file__).resolve().parents[1] / "docs" / "RELEASE.md")
    args = parser.parse_args()
    print(json.dumps(package(args.run_dir, args.base_model_path, args.base_model_id, args.model_card), indent=2))
