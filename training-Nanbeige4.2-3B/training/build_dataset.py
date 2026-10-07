#!/usr/bin/env python3
"""Build the immutable QCal Agent 1.0 next-action trajectory dataset."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import sys

from paths import AGENT_SRC, require_agent_source
sys.path.insert(0, str(require_agent_source()))

from qmagent.contracts import TOOLS, decision
from qmagent.physical_backend import PhysicalSimulator
from qmagent.policies import RulePolicy
from qmagent.protocol import PROTOCOL_VERSION, TOOLS_SCHEMA, assistant_call, policy_messages
from qmagent.runtime import AgentRunner
from qmagent.storage import digest, encoded


SPLITS = ("train", "validation", "test", "ood")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=True, allow_nan=False, indent=2)
        handle.write("\n")


class RecordingBackend(PhysicalSimulator):
    def __init__(self, *args, artifact_dir: Path, **kwargs):
        super().__init__(*args, **kwargs)
        self.artifact_dir = artifact_dir

    def acquire(self, *args):
        raw = super().acquire(*args)
        payload = encoded(raw)
        checksum = hashlib.sha256(payload).hexdigest()
        if checksum != digest(raw):
            raise RuntimeError("Canonical acquisition digest mismatch")
        path = self.artifact_dir / f"{checksum}.json"
        if not path.exists():
            path.write_bytes(payload)
        return raw


class ObservableTeacher(RulePolicy):
    """Teacher restricted to the same observable context as the trained policy."""

    def __init__(self, narrow_initial_scan: bool = False):
        self.narrow_initial_scan = narrow_initial_scan

    def decide(self, context):
        if not context["budget"]["remaining_experiments"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Global experiment budget exhausted")
        action = super().decide(context)
        if context["observation"] is None and self.narrow_initial_scan:
            action["parameter_action"]["scan"] = {"frequency_span_hz": 4e6}
        defaults = {
            "sq.s21_power2d": {"frequency_points": 161, "power_points": 21,
                                "power_min_dbm": -45.0, "power_max_dbm": -5.0},
            "sq.s21_zpa2d": {"zpa_points": 21, "frequency_points": 161},
            "sq.xeb": {"max_depth": 96, "depth_points": 12, "circuits_per_depth": 64},
        }
        if action["next_tool"] in defaults and not action["parameter_action"]["scan"]:
            action["parameter_action"]["scan"] = defaults[action["next_tool"]]
        name = action["next_tool"]
        if name in TOOLS and context["budget"]["tool_counts"].get(name, 0) >= context["budget"]["max_calls_per_tool"]:
            return decision("ESCALATE_HARDWARE_REVIEW", reason="Per-experiment retry budget exhausted")
        return action


def variants(split: str):
    if split == "ood":
        return (("drift", 1.0, False), ("quasistatic", 1.0, False),
                ("nominal", 30.0, False), ("flux_edge", 1.0, False),
                ("low_xeb", 0.5, False))
    common = (("nominal", 0.5, False), ("nominal", 1.0, False),
              ("nominal", 8.0, False), ("ambiguous", 1.0, False),
              ("nominal", 1.0, True))
    # The controller must learn that a reliable Ramsey fit can still require a
    # second Ramsey experiment after applying a correction >= 50 kHz.  The
    # original nominal-only train distribution never produced that branch,
    # while drift/quasistatic evaluation did.  Include those physical regimes
    # only for train/validation devices; keep the frozen test and OOD recipes
    # byte-for-byte stable so a rebuilt dataset remains comparable.
    if split in {"train", "validation"}:
        return common + (("drift", 1.0, False), ("quasistatic", 1.0, False))
    return common


def build(output: Path, devices: int = 64, frozen_eval_from: Path | None = None) -> dict:
    if devices < 8 or devices % 8:
        raise ValueError("devices must be a multiple of eight and at least eight")
    output.mkdir(parents=True, exist_ok=False)
    artifacts = output / "artifacts"
    artifacts.mkdir()
    rows = {name: [] for name in SPLITS}
    episodes, targets, outcomes = [], Counter(), Counter()
    total_devices = devices + devices // 4
    for device_index in range(total_devices):
        split = ("train" if device_index < devices * 3 // 4 else
                 "validation" if device_index < devices * 7 // 8 else
                 "test" if device_index < devices else "ood")
        seed = 2026091300 + device_index
        device_id = hashlib.sha256(f"qcal-device:{seed}".encode()).hexdigest()[:16]
        for replica, (profile, noise, narrow) in enumerate(variants(split)):
            episode_id = hashlib.sha256(f"qcal-episode:{seed}:{replica}".encode()).hexdigest()[:20]
            backend = RecordingBackend(seed, noise, profile, artifact_dir=artifacts)
            runner = AgentRunner(backend, ObservableTeacher(narrow), max_steps=40, max_tool_calls=10)
            transcript, sample_ids = [], []
            while runner.status == "active":
                messages = policy_messages(runner.context())
                action = runner.plan()
                if action is None:
                    break
                answer = assistant_call(action)
                sample_id = f"{episode_id}-{len(sample_ids):02d}"
                runner.step()
                if runner.status in {"invalid_action", "policy_error", "tool_error"}:
                    raise RuntimeError(f"Invalid teacher trajectory {episode_id}: {runner.reason}")
                rows[split].append({"id": sample_id, "task": "qcal_next_action", "device_id": device_id,
                    "episode_id": episode_id, "protocol_version": PROTOCOL_VERSION,
                    "tools": TOOLS_SCHEMA, "messages": messages + [answer]})
                sample_ids.append(sample_id)
                targets[action["next_tool"]] += 1
                transcript.extend([{"role": "user", "content": messages[1]["content"]}, answer,
                    {"role": "tool", "name": "calibration.step", "content": json.dumps({
                        "status": runner.status, "context": policy_messages(runner.context())[1]["content"]})}])
            result = runner.result()
            events = result.pop("events")
            write_json(output / "trajectories" / f"{episode_id}.json", {
                "device_id": device_id, "split": split, "messages": transcript,
                "sample_ids": sample_ids, "result": result})
            write_json(output / "audit" / f"{episode_id}.json", events)
            write_json(output / "evaluator_only" / f"{episode_id}.json", {
                "seed": seed, "profile": profile, "noise_scale": noise,
                "narrow_initial_scan": narrow, "truth": backend._truth})
            outcomes[f"{split}:{runner.status}"] += 1
            episodes.append({"episode_id": episode_id, "device_id": device_id, "split": split,
                             "status": runner.status, "samples": len(sample_ids)})
        print(f"device {device_index + 1}/{total_devices} {split}", flush=True)
    frozen_eval = None
    if frozen_eval_from is not None:
        frozen_eval_from = frozen_eval_from.resolve()
        frozen_manifest_path = frozen_eval_from / "manifest.json"
        if not frozen_manifest_path.is_file():
            raise ValueError("Frozen evaluation dataset is missing manifest.json")
        frozen_manifest = json.loads(frozen_manifest_path.read_text(encoding="utf-8"))
        frozen_episode_ids = set()
        for split in ("test", "ood"):
            source = frozen_eval_from / f"{split}.jsonl"
            expected_hash = frozen_manifest["split_sha256"][split]
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected_hash:
                raise ValueError(f"Frozen {split} checksum mismatch")
            rows[split] = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line]
            frozen_episode_ids.update(row["episode_id"] for row in rows[split])
            for row in rows[split]:
                context = json.loads(row["messages"][-2]["content"])
                reference = (context.get("observation") or {}).get("raw_reference", {}).get("sha256")
                if reference:
                    source_artifact = frozen_eval_from / "artifacts" / f"{reference}.json"
                    target_artifact = artifacts / source_artifact.name
                    if not source_artifact.is_file():
                        raise ValueError(f"Frozen artifact is missing: {reference}")
                    if not target_artifact.exists():
                        shutil.copyfile(source_artifact, target_artifact)
        for episode_id in frozen_episode_ids:
            for directory in ("trajectories", "audit", "evaluator_only"):
                source = frozen_eval_from / directory / f"{episode_id}.json"
                if not source.is_file():
                    raise ValueError(f"Frozen evaluation evidence is missing: {source}")
                shutil.copyfile(source, output / directory / source.name)
        episodes = ([item for item in episodes if item["split"] not in {"test", "ood"}] +
                    [item for item in frozen_manifest["episodes"] if item["split"] in {"test", "ood"}])
        frozen_eval = {
            "manifest_sha256": hashlib.sha256(frozen_manifest_path.read_bytes()).hexdigest(),
            "split_sha256": {split: frozen_manifest["split_sha256"][split] for split in ("test", "ood")},
        }
    targets = Counter(row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
                      for samples in rows.values() for row in samples)
    outcomes = Counter(f'{item["split"]}:{item["status"]}' for item in episodes)
    for split, samples in rows.items():
        with (output / f"{split}.jsonl").open("x", encoding="utf-8") as handle:
            for row in samples:
                handle.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")
    groups = {split: {row["device_id"] for row in samples} for split, samples in rows.items()}
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise RuntimeError("Device leakage across splits")
    source_root = AGENT_SRC / "qmagent"
    manifest = {
        "schema": PROTOCOL_VERSION, "runtime_schema": "runtime-1.0", "synthetic": True,
        "physics_version": PhysicalSimulator.backend_name, "teacher": "observable_only_rule_policy_1.0",
        "training_ready": True, "release_scope": "single_qubit_simulation_research_not_hardware_validated",
        "data_origin": "executed reduced flux-transmon/cQED/single-qubit-XEB simulator",
        "distribution_design": {
            "train_validation": "device-disjoint nominal, ambiguity, scan-coverage, drift and quasistatic regimes",
            "test": "held-out devices using the original in-distribution recipe",
            "ood": "held-out devices with severe noise and exclusive flux-edge/low-XEB regimes plus drift/quasistatic stress",
        },
        "frozen_evaluation": frozen_eval,
        "loss_scope": "one next assistant native call per sample; preceding context fully masked",
        "device_counts": {key: len(value) for key, value in groups.items()},
        "sample_counts": {key: len(value) for key, value in rows.items()},
        "outcomes": dict(outcomes), "target_counts": dict(targets), "episodes": episodes,
        "source_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in source_root.glob("*.py")},
        "split_sha256": {key: hashlib.sha256((output / f"{key}.jsonl").read_bytes()).hexdigest()
                         for key in rows},
        "limitations": ["Synthetic single-qubit XEB proxy, not multiqubit XEB",
            "No coupler or two-qubit physics", "Rule-teacher imitation is not hardware validation",
            "Evaluator-only simulator truth is never model input"],
    }
    write_json(output / "manifest.json", manifest)
    print(json.dumps({key: manifest[key] for key in
          ("device_counts", "sample_counts", "outcomes", "target_counts")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", type=int, default=64)
    parser.add_argument("--frozen-eval-from", type=Path,
                        help="Reuse immutable test/OOD rows and evidence from an audited dataset")
    arguments = parser.parse_args()
    build(arguments.output, arguments.devices, arguments.frozen_eval_from)
