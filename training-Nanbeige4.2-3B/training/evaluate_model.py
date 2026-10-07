#!/usr/bin/env python3
"""Deterministic pre/post SFT benchmark for the calibration agent."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from evaluation_metrics import score_prediction, aggregate_scores, select_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter")
    parser.add_argument("--skip-generation", action="store_true", help="Teacher-forced NLL/PPL only; generation metrics remain null")
    parser.add_argument("--enable-cudnn-sdpa", action="store_true", help="Opt into cuDNN SDPA; slow decode planning observed on this H100 stack")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-task", type=int, default=2, help="0 evaluates all rows")
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--max-input-length", type=int, default=20480)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    args = parser.parse_args()
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(args.enable_cudnn_sdpa)

    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        parser.error("Output directory must be empty; refusing to overwrite baseline")
    if args.samples_per_task < 0 or min(args.max_input_length, args.max_new_tokens) <= 0:
        parser.error("Invalid sample/token limits")
    with open(args.data, encoding="utf-8") as handle:
        rows = select_rows([json.loads(line) for line in handle if line.strip()], args.samples_per_task, args.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    with (output_dir / "run_config.json").open("x", encoding="utf-8") as handle:
        json.dump(vars(args) | {"scoring_version": "task-aware-1.0", "selected_ids": [row["id"] for row in rows],
            "dataset_sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest(),
            "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                              for name in ("evaluate_model.py", "evaluation_metrics.py")},
            "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "accelerate", "datasets")},
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}, handle, ensure_ascii=False, indent=2)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto", low_cpu_mem_usage=True,
        local_files_only=True, attn_implementation=args.attn_implementation
    )
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter, local_files_only=True)
    model.eval()
    device = model.get_input_embeddings().weight.device
    template_kwargs = {"enable_thinking": False, "preserve_thinking": False}

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["task"]].append(row)

    totals = defaultdict(float)
    records: list[dict[str, Any]] = []
    started = time.time()
    for index, row in enumerate(rows, 1):
        messages = row["messages"]
        prompt = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **template_kwargs)
        full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **template_kwargs)
        prompt_ids = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")["input_ids"]
        full_ids = tokenizer(full, add_special_tokens=False, return_tensors="pt")["input_ids"]
        if full_ids.shape[1] > args.max_input_length:
            raise ValueError(f"{row['id']} has {full_ids.shape[1]} tokens > {args.max_input_length}; refusing truncation")
        prefix = prompt_ids.shape[1]
        if not torch.equal(full_ids[:, :prefix], prompt_ids) or full_ids.shape[1] <= prefix:
            raise ValueError(f"{row['id']}: template prefix mismatch or empty target")
        labels = full_ids.clone()
        labels[:, :prefix] = -100
        generated = None
        with torch.inference_mode():
            loss = model(input_ids=full_ids.to(device), labels=labels.to(device), use_cache=False).loss.item()
            if not args.skip_generation:
                generated = model.generate(
                    prompt_ids.to(device), attention_mask=torch.ones_like(prompt_ids).to(device),
                    max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                )
        prediction = tokenizer.decode(generated[0, prefix:], skip_special_tokens=True).strip() if generated is not None else None
        target = messages[-1]["content"].strip()
        if not math.isfinite(loss):
            raise ValueError(f"Nonfinite loss for {row['id']}")
        token_count = int((labels[:, 1:] != -100).sum())
        generated_tokens = generated.shape[1] - prefix if generated is not None else 0
        eos_ids = tokenizer.eos_token_id if isinstance(tokenizer.eos_token_id, list) else [tokenizer.eos_token_id]
        record = {
            "id": row["id"], "task": row["task"], "input_tokens": prefix,
            "target_tokens": token_count, "loss": loss, "prediction": prediction,
            "target": target, "generated_tokens": generated_tokens,
            "generation_skipped": args.skip_generation,
            "hit_output_limit": generated_tokens >= args.max_new_tokens and int(generated[0, -1]) not in eos_ids,
            **(score_prediction(row["task"], prediction, target) if prediction is not None else
               {key: None for key in ("json_valid", "exact_match", "action_match", "next_tool_match")}),
        }
        records.append(record)
        totals["weighted_nll"] += loss * token_count
        totals["tokens"] += token_count
        with (output_dir / "predictions.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        print(f"[{index}/{len(rows)}] {row['task']} {row['id']} loss={loss:.4f} json={record['json_valid']}", flush=True)

    n = len(records)
    mean_nll = totals["weighted_nll"] / max(totals["tokens"], 1)
    metrics = {
        "model": str(Path(args.model).resolve()), "dataset": str(Path(args.data).resolve()),
        "sample_count": n, "samples_per_task": args.samples_per_task,
        "tasks": {task: len(items) for task, items in sorted(grouped.items())},
        "assistant_token_nll": mean_nll, "assistant_token_perplexity": math.exp(min(mean_nll, 50)),
        **aggregate_scores(records),
        "by_task": {task: aggregate_scores([row for row in records if row["task"] == task]) for task in sorted(grouped)},
        "output_limit_count": sum(row["hit_output_limit"] for row in records),
        "scope": "teacher-forced NLL/PPL only; generation was skipped" if args.skip_generation else "offline sampled text benchmark; not full test or closed-loop IQ acceptance",
        "elapsed_seconds": time.time() - started,
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
                        "gpu_count": torch.cuda.device_count(), "hostname": platform.node()},
    }
    with open(output_dir / "metrics.json", "w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
