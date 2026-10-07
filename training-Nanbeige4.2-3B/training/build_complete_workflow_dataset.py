#!/usr/bin/env python3
"""Preserve all workflow examples and add modest train-only boundary coverage."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import sys

from paths import AGENT_SRC, require_agent_source
sys.path.insert(0, str(require_agent_source()))
from build_repair_curriculum import curriculum_rows, read_rows, sha256, write_rows
from qmagent.protocol import policy_messages


def build(source: Path, output: Path):
    source, output = source.resolve(), output.resolve()
    manifest = json.loads((source / "manifest.json").read_text())
    for split in ("train", "validation", "test", "ood"):
        if sha256(source / f"{split}.jsonl") != manifest["split_sha256"][split]:
            raise ValueError(f"Source checksum mismatch: {split}")
    output.mkdir(parents=True, exist_ok=False)
    os.symlink(source / "artifacts", output / "artifacts", target_is_directory=True)
    counts, derivations = {}, {}
    for split in ("train", "validation"):
        original = read_rows(source / f"{split}.jsonl")
        # One copy per boundary, not safety-only continuation or repeated oversampling.
        extra = curriculum_rows(original, split, ramsey_copies=1)
        rows = original + extra
        for row in rows:
            context = json.loads(row["messages"][1]["content"])
            answer = row["messages"][-1]
            row["messages"] = policy_messages(context) + [answer]
        write_rows(output / f"{split}.jsonl", rows)
        counts[split] = len(rows)
        derivations[split] = {"original_workflow": len(original), "boundary_examples": len(extra)}
        if len(original) / len(rows) < .85:
            raise ValueError("Normal workflow replay must remain at least 85%")
    for split in ("test", "ood"):
        shutil.copyfile(source / f"{split}.jsonl", output / f"{split}.jsonl")
        counts[split] = manifest["sample_counts"][split]
    updated = {**manifest, "sample_counts": counts,
               "split_sha256": {s: sha256(output / f"{s}.jsonl") for s in counts},
               "complete_workflow_schema": "qcal-complete-workflow-1.0",
               "source_manifest_sha256": sha256(source / "manifest.json"),
               "source_split_sha256": manifest["split_sha256"],
               "derivation_counts": derivations,
               "context_transform_sha256": sha256(AGENT_SRC / "qmagent/protocol.py"),
               "frozen_evaluation": {s: manifest["split_sha256"][s] for s in ("test", "ood")}}
    updated.pop("curriculum_schema", None)
    targets = Counter(row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
                      for split in counts for row in read_rows(output / f"{split}.jsonl"))
    updated["target_counts"] = dict(targets)
    (output / "manifest.json").write_text(json.dumps(updated, indent=2) + "\n")
    return updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.source, args.output)["derivation_counts"], indent=2))
