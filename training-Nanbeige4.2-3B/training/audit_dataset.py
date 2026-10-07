#!/usr/bin/env python3
"""Fail closed on QCal Agent 1.0 dataset integrity or split leakage."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

from paths import require_agent_source
sys.path.insert(0, str(require_agent_source()))

from qmagent.protocol import PROTOCOL_VERSION


def load_lines(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def audit(dataset: Path) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["schema"] != PROTOCOL_VERSION or manifest["runtime_schema"] != "runtime-1.0":
        raise ValueError("Unexpected protocol/runtime schema")
    split_devices, ids, targets, references = {}, set(), Counter(), set()
    forbidden = {"sweet_zpa", "junction_asymmetry", "ej_sum_hz", "xeb_cycle_fidelity", "_truth"}
    for split in ("train", "validation", "test", "ood"):
        path = dataset / f"{split}.jsonl"
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError(f"{split} checksum mismatch")
        rows = load_lines(path)
        if len(rows) != manifest["sample_counts"][split]:
            raise ValueError(f"{split} count mismatch")
        split_devices[split] = {row["device_id"] for row in rows}
        for row in rows:
            if row["id"] in ids:
                raise ValueError("Duplicate sample id")
            ids.add(row["id"])
            if row["protocol_version"] != PROTOCOL_VERSION or row["task"] != "qcal_next_action":
                raise ValueError("Mixed protocol/task row")
            serialized = json.dumps(row, sort_keys=True)
            if any(name in serialized for name in forbidden):
                raise ValueError("Simulator truth leaked into a model-visible row")
            target = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
            targets[target] += 1
            context = json.loads(row["messages"][-2]["content"])
            observation = context.get("observation")
            if observation:
                if "measurement" in observation or "sweep" in observation:
                    raise ValueError("Raw arrays leaked into a prompt")
                reference = observation.get("raw_reference", {}).get("sha256")
                if reference:
                    references.add(reference)
    for left in split_devices:
        for right in split_devices:
            if left < right and split_devices[left] & split_devices[right]:
                raise ValueError(f"Device leakage: {left}/{right}")
    artifacts = list((dataset / "artifacts").glob("*.json"))
    for path in artifacts:
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            raise ValueError(f"Artifact content/hash mismatch: {path.name}")
        raw = json.loads(path.read_text())
        i_values, q_values = raw["measurement"]["i"], raw["measurement"]["q"]
        if raw["tool"] in {"sq.s21_power2d", "sq.s21_zpa2d"}:
            row_axis = "power_values_dbm" if raw["tool"] == "sq.s21_power2d" else "zpa_values"
            expected = (len(raw["sweep"][row_axis]), len(raw["sweep"]["frequency_values_hz"]))
            if (len(i_values) != expected[0] or len(q_values) != expected[0]
                    or any(len(row) != expected[1] for row in i_values + q_values)):
                raise ValueError(f"Malformed {raw['tool']} artifact shape")
        elif len(i_values) != len(q_values):
            raise ValueError("Malformed aligned I/Q artifact")
    if references - {path.stem for path in artifacts}:
        raise ValueError("Referenced raw artifacts are missing")
    required = {"sq.s21_power2d", "sq.s21_zpa2d", "sq.xeb"}
    if not required <= set(targets):
        raise ValueError("Power2D, ZPA2D or XEB is absent from native-call targets")
    report = {"protocol": PROTOCOL_VERSION, "samples": len(ids),
        "devices": {key: len(value) for key, value in split_devices.items()},
        "targets": dict(targets), "artifact_count": len(artifacts),
        "referenced_artifacts": len(references), "device_leakage": False,
        "truth_leakage": False, "artifact_hashes_valid": True,
        "power2d_shapes_valid": True, "zpa2d_shapes_valid": True,
        "required_tools_present": True}
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    arguments = parser.parse_args()
    audit(arguments.dataset)
