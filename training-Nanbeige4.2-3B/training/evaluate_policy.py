#!/usr/bin/env python3
"""Evaluate native next-action calls on an immutable frozen-context split."""

from __future__ import annotations

import argparse
from collections import Counter
import datetime
import hashlib
import json
from pathlib import Path
import sys
import time

from paths import AGENT_SRC, require_agent_source
sys.path.insert(0, str(require_agent_source()))

from qmagent.policies import HuggingFacePolicy
from qmagent.protocol import PROTOCOL_VERSION, TOOLS_SCHEMA, parse_call, policy_messages


def select(rows: list[dict], limit: int) -> list[dict]:
    if not limit or limit >= len(rows):
        return rows
    groups = {}
    for row in rows:
        name = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]["next_tool"]
        groups.setdefault(name, []).append(row)
    selected = []
    while len(selected) < limit:
        before = len(selected)
        for name in sorted(groups):
            if groups[name] and len(selected) < limit:
                selected.append(groups[name].pop(0))
        if len(selected) == before:
            break
    return selected


def argument_match(actual: dict, expected: dict) -> bool:
    if actual["next_tool"] != expected["next_tool"]:
        return False
    for field in ("updates", "scan"):
        left, right = actual["parameter_action"][field], expected["parameter_action"][field]
        if set(left) != set(right):
            return False
        for key in left:
            tolerance = 1000.0 if key.endswith("_hz") else 0.01 if key.endswith("_us") else 1e-4
            if abs(left[key] - right[key]) > tolerance:
                return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--test-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--prompt-profile", choices=("minimal", "skill"), default="skill")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    raw = args.test_file.read_bytes()
    rows = select([json.loads(line) for line in raw.splitlines()], args.limit)
    if not rows:
        raise ValueError("Empty evaluation")
    first_context = json.loads(rows[0]["messages"][1]["content"])
    system_prompt = policy_messages(first_context, prompt_profile=args.prompt_profile)[0]["content"]
    config = {"protocol": PROTOCOL_VERSION, "model": args.model, "adapter": args.adapter,
        "test_sha256": hashlib.sha256(raw).hexdigest(),
        "selected_ids": [row["id"] for row in rows],
        "scope": "frozen-context imitation metrics; not closed-loop or hardware success",
        "sampling": "all rows" if not args.limit else "round-robin action-stratified subset",
        "batch_size": args.batch_size, "prompt_profile": args.prompt_profile,
        "system_prompt_sha256": hashlib.sha256(system_prompt.encode()).hexdigest(),
        "context_transform_sha256": hashlib.sha256(
            (AGENT_SRC / "qmagent/protocol.py").read_bytes()).hexdigest(),
        "tokenizer_sha256": hashlib.sha256(
            (Path(args.model) / "tokenizer_config.json").read_bytes()).hexdigest()}
    records = []
    if args.resume:
        recorded = json.loads((args.output / "config.json").read_text())
        immutable = ("protocol", "model", "adapter", "test_sha256", "selected_ids", "batch_size",
                     "prompt_profile", "system_prompt_sha256", "tokenizer_sha256", "context_transform_sha256")
        mismatches = [key for key in immutable if recorded.get(key) != config.get(key)]
        if mismatches:
            raise ValueError("Resume config mismatch: " + ", ".join(mismatches))
        records = [json.loads(line) for line in
                   (args.output / "predictions.jsonl").read_text().splitlines()]
        if [record.get("id") for record in records] != config["selected_ids"][:len(records)]:
            raise ValueError("Existing predictions are not the selected-ID prefix")
        with (args.output / "resume_history.jsonl").open("a") as history:
            history.write(json.dumps({"resumed_at": datetime.datetime.now(
                datetime.timezone.utc).isoformat(), "completed_prefix": len(records)}) + "\n")
    else:
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    policy = HuggingFacePolicy(args.model, args.adapter, max_new_tokens=512,
        decode_backend="hf", trust_remote_code=args.trust_remote_code,
        prompt_profile=args.prompt_profile)
    wall_seconds = sum(record.get("seconds", 0.0) for record in records)
    with (args.output / "predictions.jsonl").open("a" if args.resume else "x") as stream:
        for offset in range(len(records), len(rows), args.batch_size):
            batch = rows[offset:offset + args.batch_size]
            contexts = [json.loads(row["messages"][1]["content"]) for row in batch]
            message_batches = [policy_messages(context, prompt_profile=args.prompt_profile)
                               for context in contexts]
            started = time.monotonic()
            outputs, details = policy._generate_batch(message_batches, policy.max_new_tokens,
                                                       tools=TOOLS_SCHEMA)
            wall_seconds += time.monotonic() - started
            for row, context, output, detail in zip(batch, contexts, outputs, details):
                expected = row["messages"][-1]["tool_calls"][0]["function"]["arguments"]
                try:
                    prediction = parse_call(output, context=context)
                    record = {"id": row["id"], "valid": True, "prediction": prediction,
                        "next_tool_correct": prediction["next_tool"] == expected["next_tool"],
                        "arguments_correct": argument_match(prediction, expected)}
                except (ValueError, RuntimeError) as error:
                    record = {"id": row["id"], "valid": False, "error": str(error),
                              "next_tool_correct": False, "arguments_correct": False}
                record.update(expected=expected, seconds=detail["amortized_seconds"], generation=detail)
                records.append(record)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                print(f"{len(records)}/{len(rows)} valid={record['valid']} next={record['next_tool_correct']}", flush=True)
    metrics = {"sample_count": len(records), "scope": config["scope"],
        **{key + "_rate": sum(record[key] for record in records) / len(records)
           for key in ("valid", "next_tool_correct", "arguments_correct")},
        "mean_seconds": sum(record["seconds"] for record in records) / len(records),
        "wall_seconds": wall_seconds, "samples_per_second": len(records) / wall_seconds,
        "batch_size": args.batch_size,
        "target_counts": dict(Counter(record["expected"]["next_tool"] for record in records)),
        "dataset_sha256": config["test_sha256"], "adapter": args.adapter}
    (args.output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
