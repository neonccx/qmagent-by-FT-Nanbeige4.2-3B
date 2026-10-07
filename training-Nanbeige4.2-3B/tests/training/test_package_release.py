import hashlib
import json
import tarfile
from pathlib import Path

import pytest

from training.package_release import TOKENIZER_FILES, package


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_run(tmp_path: Path) -> tuple[Path, Path, Path]:
    run = tmp_path / "run"
    base = tmp_path / "base"
    dataset = tmp_path / "dataset"
    base.mkdir()
    dataset.mkdir()
    (base / "config.json").write_text("base config", encoding="utf-8")
    adapter = run / "training" / "final_adapter"
    adapter.mkdir(parents=True)
    weights = b"fake LoRA weights"
    (adapter / "adapter_model.safetensors").write_bytes(weights)
    write_json(adapter / "adapter_config.json", {
        "base_model_name_or_path": str(base), "peft_type": "LORA", "task_type": "CAUSAL_LM"
    })
    for name in TOKENIZER_FILES:
        (base / name).write_text(name, encoding="utf-8")
        (adapter / name).write_text(name, encoding="utf-8")
    release = run / "release"
    release.mkdir()
    (release / "qcal-agent-1.0.0-lora.safetensors").write_bytes(weights)
    write_json(release / "release_manifest.json", {
        "version": "1.0.0", "adapter_sha256": hashlib.sha256(weights).hexdigest()
    })
    write_json(run / "release_acceptance.json", {
        "all_passed": True, "checks": {"test": {"valid": True}, "ood": {"valid": True}}
    })
    stages = [
        {"stage": name, "status": "completed"} for name in (
            "training", "adapted_test", "adapted_controller_test", "comparison_test",
            "adapted_ood", "adapted_controller_ood", "comparison_ood", "closed_loop_adapted",
        )
    ]
    write_json(run / "status.json", {
        "status": "completed_with_scientific_failures", "dataset": str(dataset), "stages": stages
    })
    write_json(dataset / "manifest.json", {"split_sha256": {"test": "test-hash", "ood": "ood-hash"}})
    for split in ("test", "ood"):
        write_json(run / f"adapted_{split}" / "metrics.json", {
            "dataset_sha256": f"{split}-hash", "sample_count": 160,
            "valid_rate": 1.0, "next_tool_correct_rate": 1.0, "arguments_correct_rate": 1.0,
        })
        write_json(run / f"adapted_{split}" / "controller_score.json", {
            "controller_executable_rate": 1.0
        })
    write_json(run / "closed_loop_adapted" / "summary.json", {
        "episodes": 3, "statuses": {"accepted": 3}
    })
    model_card = tmp_path / "model_card.md"
    model_card.write_text("Simulation only", encoding="utf-8")
    return run, base, model_card


def test_portable_release_contains_standard_peft_files_and_hashes(tmp_path: Path):
    run, base, model_card = fixture_run(tmp_path)
    manifest = package(run, base, "Nanbeige/Nanbeige4.2-3B", model_card)
    bundle = run / "release" / "qcal-agent-1.0.0-adapter"
    config = json.loads((bundle / "adapter_config.json").read_text())
    assert config["base_model_name_or_path"] == "Nanbeige/Nanbeige4.2-3B"
    assert (bundle / "adapter_model.safetensors").read_bytes() == b"fake LoRA weights"
    archive = run / "release" / "qcal-agent-1.0.0-adapter.tar.gz"
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == manifest["bundle_sha256"]
    with tarfile.open(archive) as tar:
        assert {Path(name).name for name in tar.getnames()} >= {
            "adapter_model.safetensors", "adapter_config.json", "README.md"
        }
    assert json.loads((run / "release" / "portable_manifest.json").read_text()) == manifest
    with pytest.raises(FileExistsError, match="already exists"):
        package(run, base, "Nanbeige/Nanbeige4.2-3B", model_card)


def test_portable_release_rejects_failed_gate(tmp_path: Path):
    run, base, model_card = fixture_run(tmp_path)
    write_json(run / "release_acceptance.json", {
        "all_passed": False, "checks": {"test": {"valid": False}}
    })
    with pytest.raises(ValueError, match="not all passed"):
        package(run, base, "Nanbeige/Nanbeige4.2-3B", model_card)
    assert not (run / "release" / "qcal-agent-1.0.0-adapter").exists()
