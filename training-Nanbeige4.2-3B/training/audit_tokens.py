#!/usr/bin/env python3
"""Audit prompt identity, native-call loss boundaries and tokenizer lengths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

from paths import AGENT_SRC, require_agent_source
sys.path.insert(0, str(require_agent_source()))

from qmagent.protocol import PROTOCOL_VERSION, parse_call, policy_messages


def audit(dataset: Path, model: Path | None = None) -> dict:
    manifest = json.loads((dataset / "manifest.json").read_text())
    if manifest["schema"] != PROTOCOL_VERSION:
        raise ValueError("Dataset protocol does not match this runtime")
    frozen_rebuild = manifest.get("complete_workflow_schema") == "qcal-complete-workflow-1.0"
    if frozen_rebuild:
        protocol_path = AGENT_SRC / "qmagent/protocol.py"
        if manifest.get("context_transform_sha256") != hashlib.sha256(protocol_path.read_bytes()).hexdigest():
            raise ValueError("Complete-workflow input transformation changed; rebuild training data")
    groups, counts, lengths, loss_lengths = {}, {}, [], []
    tokenizer = None
    if model:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model, trust_remote_code=True, local_files_only=True, use_fast=False)
    for split in ("train", "validation", "test", "ood"):
        raw = (dataset / f"{split}.jsonl").read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest["split_sha256"][split]:
            raise ValueError(f"{split} checksum mismatch")
        groups[split], counts[split] = set(), 0
        for line in raw.splitlines():
            row = json.loads(line)
            groups[split].add(row["device_id"])
            counts[split] += 1
            context = json.loads(row["messages"][1]["content"])
            online_messages = policy_messages(context)
            if frozen_rebuild and split in ("test", "ood"):
                source_hash = manifest.get("source_split_sha256", {}).get(split)
                if (manifest.get("frozen_evaluation", {}).get(split) != source_hash or
                        hashlib.sha256(raw).hexdigest() != source_hash):
                    raise ValueError("Frozen evaluation source checksum mismatch")
                # Match evaluate_policy: frozen contexts/answers, regenerated runtime prompt.
                row = {**row, "messages": online_messages + [row["messages"][-1]]}
            elif online_messages != row["messages"][:-1]:
                raise ValueError("Online/training prompt mismatch")
            visible = json.dumps(row["messages"])
            forbidden = ('"_truth"', '"ground_truth"', '"recommended_update"',
                         '"prepared_state": [', '"measurement":')
            if any(key in visible for key in forbidden):
                raise ValueError("Private simulator/raw data leaked into a prompt")
            observation = context.get("observation")
            if observation and not (dataset / "artifacts" /
                                    f"{observation['raw_reference']['sha256']}.json").is_file():
                raise ValueError("Prompt references a missing raw artifact")
            if tokenizer:
                options = {"tokenize": False, "enable_thinking": False,
                           "preserve_thinking": False, "tools": row["tools"],
                           "tool_call_format": "json"}
                prompt = tokenizer.apply_chat_template(
                    row["messages"][:-1], add_generation_prompt=True, **options)
                full = tokenizer.apply_chat_template(
                    row["messages"], add_generation_prompt=False, **options)
                prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
                full_ids = tokenizer(full, add_special_tokens=False)["input_ids"]
                if full_ids[:len(prompt_ids)] != prompt_ids or len(full_ids) <= len(prompt_ids):
                    raise ValueError("Assistant-only loss boundary mismatch")
                answer = tokenizer.decode(full_ids[len(prompt_ids):], skip_special_tokens=True).strip()
                expected = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
                if parse_call(answer) != expected:
                    raise ValueError("Tokenized native call does not round-trip")
                lengths.append(len(full_ids))
                loss_lengths.append(len(full_ids) - len(prompt_ids))
    if any(groups[a] & groups[b] for a in groups for b in groups if a < b):
        raise ValueError("Device leakage across splits")
    for path in (dataset / "artifacts").glob("*.json"):
        if hashlib.sha256(path.read_bytes()).hexdigest() != path.stem:
            raise ValueError("Artifact hash mismatch")
    return {"all_passed": True, "protocol": PROTOCOL_VERSION, "sample_counts": counts,
        "frozen_evaluation_runtime_prompts_rebuilt": frozen_rebuild,
        "artifact_count": len(list((dataset / "artifacts").glob("*.json"))),
        "device_counts": {key: len(value) for key, value in groups.items()},
        "tokenizer_checked": bool(tokenizer), "max_tokens": max(lengths) if lengths else None,
        "min_assistant_loss_tokens": min(loss_lengths) if loss_lengths else None,
        "max_assistant_loss_tokens": max(loss_lengths) if loss_lengths else None,
        "template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
                           if tokenizer else None}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    result = audit(arguments.dataset, arguments.model)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with arguments.output.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps(result, indent=2))
