#!/usr/bin/env python3
"""Build a train/validation-only safety curriculum without evaluation leakage."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from paths import require_agent_source
sys.path.insert(0, str(require_agent_source()))

from qmagent.contracts import TOOLS, decision
from qmagent.protocol import PROTOCOL_VERSION, assistant_call, policy_messages


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=True, allow_nan=False) + "\n")


def derived_row(source: dict, suffix: str, context: dict, action: dict) -> dict:
    return {
        **{key: source[key] for key in
           ("task", "device_id", "episode_id", "protocol_version", "tools")},
        "id": f'{source["id"]}-{suffix}',
        "source_sample_id": source["id"],
        "curriculum_derivation": suffix,
        "messages": policy_messages(context) + [assistant_call(action)],
    }


def curriculum_rows(source_rows: list[dict], split: str, ramsey_copies: int) -> list[dict]:
    by_episode = defaultdict(list)
    for row in source_rows:
        by_episode[row["episode_id"]].append(row)
    result = []
    # One budget-zero example per source episode, selected only from a context
    # whose original target was an experiment.  This covers the deterministic
    # terminal safety branch without borrowing any evaluation decisions.
    for episode_id in sorted(by_episode):
        candidates = []
        for row in by_episode[episode_id]:
            action = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
            if action["next_tool"] in TOOLS:
                candidates.append(row)
        if not candidates:
            continue
        source = candidates[-1]
        context = json.loads(source["messages"][1]["content"])
        context["budget"]["remaining_experiments"] = 0
        action = decision("ESCALATE_HARDWARE_REVIEW", reason="Global experiment budget exhausted")
        result.append(derived_row(source, "budget-zero", context, action))

    for source in source_rows:
        context = json.loads(source["messages"][1]["content"])
        observation = context.get("observation") or {}
        action = source["messages"][-1]["tool_calls"][0]["function"]["arguments"]
        correction = observation.get("fit_result", {}).get("frequency_correction_hz", 0)
        if (observation.get("tool") == "sq.ramsey_df" and abs(correction) >= 50_000
                and action["next_tool"] == "sq.ramsey_df"):
            for copy_index in range(ramsey_copies):
                result.append(derived_row(source, f"ramsey-repeat-{copy_index}", context, action))
    if not result:
        raise RuntimeError(f"No curriculum samples produced for {split}")
    return result


def build(source: Path, output: Path, ramsey_copies: int = 3) -> dict:
    source, output = source.resolve(), output.resolve()
    if ramsey_copies < 1:
        raise ValueError("ramsey-copies must be positive")
    source_manifest_path = source / "manifest.json"
    if not source_manifest_path.is_file():
        raise ValueError("Source dataset manifest is missing")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    output.mkdir(parents=True, exist_ok=False)
    artifacts = source / "artifacts"
    if not artifacts.is_dir():
        raise ValueError("Source artifact directory is missing")
    os.symlink(artifacts, output / "artifacts", target_is_directory=True)

    rows = {}
    for split in ("train", "validation"):
        path = source / f"{split}.jsonl"
        if sha256(path) != source_manifest["split_sha256"][split]:
            raise ValueError(f"Source {split} checksum mismatch")
        rows[split] = curriculum_rows(read_rows(path), split, ramsey_copies)
        write_rows(output / f"{split}.jsonl", rows[split])
    for split in ("test", "ood"):
        path = source / f"{split}.jsonl"
        if sha256(path) != source_manifest["split_sha256"][split]:
            raise ValueError(f"Frozen {split} checksum mismatch")
        shutil.copyfile(path, output / f"{split}.jsonl")
        rows[split] = read_rows(path)

    groups = {split: {row["device_id"] for row in values} for split, values in rows.items()}
    if any(groups[left] & groups[right] for left in groups for right in groups if left < right):
        raise RuntimeError("Device leakage across curriculum splits")
    targets = Counter(row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
                      for values in rows.values() for row in values)
    derivations = {split: dict(Counter(row.get("curriculum_derivation", "frozen_evaluation")
                                      for row in values)) for split, values in rows.items()}
    manifest = {
        "schema": PROTOCOL_VERSION,
        "runtime_schema": "runtime-1.0",
        "curriculum_schema": "qcal-repair-curriculum-1.0",
        "training_ready": True,
        "synthetic": True,
        "release_scope": "single_qubit_simulation_research_not_hardware_validated",
        "source_dataset_manifest_sha256": sha256(source_manifest_path),
        "source_split_sha256": source_manifest["split_sha256"],
        "split_sha256": {split: sha256(output / f"{split}.jsonl") for split in rows},
        "sample_counts": {split: len(values) for split, values in rows.items()},
        "device_counts": {split: len(values) for split, values in groups.items()},
        "target_counts": dict(targets),
        "derivation_counts": derivations,
        "frozen_evaluation": {split: source_manifest["split_sha256"][split]
                              for split in ("test", "ood")},
        "limitations": ["Safety-boundary curriculum derived only from source train/validation contexts",
                        "Frozen test/OOD rows are copied only for evaluation and are never curriculum sources"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: manifest[key] for key in
                      ("sample_counts", "device_counts", "derivation_counts", "target_counts")}, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ramsey-copies", type=int, default=3)
    arguments = parser.parse_args()
    build(arguments.source, arguments.output, arguments.ramsey_copies)
