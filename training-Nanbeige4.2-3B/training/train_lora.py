#!/usr/bin/env python3
"""BF16 LoRA SFT for the superconducting-qubit calibration agent."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_rows(path: str, limit: int = 0) -> Dataset:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            # Preserve variable function arguments exactly; Arrow struct unification
            # would otherwise inject null parameter keys into native tool calls.
            rows.append({"id": row["id"], "task": row["task"], "messages": json.dumps(row["messages"]),
                         "tools_json": json.dumps(row.get("tools"))})
            if limit and len(rows) >= limit:
                break
    return Dataset.from_list(rows)


def common_prefix_length(left: list[int], right: list[int]) -> int:
    size = 0
    for a, b in zip(left, right):
        if a != b:
            break
        size += 1
    return size


@dataclass
class CalibrationCollator:
    tokenizer: Any
    pad_to_multiple_of: int = 8

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(feature["input_ids"]) for feature in features)
        multiple = self.pad_to_multiple_of
        max_len = ((max_len + multiple - 1) // multiple) * multiple
        input_ids, labels, attention = [], [], []
        pad_id = self.tokenizer.pad_token_id
        for feature in features:
            length = len(feature["input_ids"])
            padding = max_len - length
            input_ids.append(feature["input_ids"] + [pad_id] * padding)
            labels.append(feature["labels"] + [-100] * padding)
            attention.append([1] * length + [0] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention, dtype=torch.long),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--validation-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--initial-adapter",
                        help="Existing audited LoRA adapter to continue training from")
    parser.add_argument("--max-length", type=int, default=20480)
    parser.add_argument("--max-train-samples", type=int, default=0, help="0 means all; nonzero only for smoke tests")
    parser.add_argument("--max-validation-samples", type=int, default=0, help="0 means all; nonzero only for smoke tests")
    parser.add_argument("--baseline-metrics", help="Existing pre-training metrics.json to record in run config")
    parser.add_argument("--dataset-audit", help="Required tokenizer/artifact audit for native tool-call data")
    parser.add_argument("--assistant-only-projection", action="store_true", help="Audited Nanbeige-only sparse vocabulary projection; decoder context is unchanged")
    parser.add_argument("--load-best-model", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--attn-implementation", choices=("sdpa", "eager"), default="sdpa")
    parser.add_argument("--enable-cudnn-sdpa", action="store_true", help="Opt into cuDNN SDPA instead of native flash/efficient dispatch")
    parser.add_argument("--gpu-memory-fraction", type=float, default=1.0, help="Optional per-process allocator cap for bounded smoke tests")
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=-1, help="Positive values override epochs; useful for smoke tests.")
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=64)
    parser.add_argument("--micro-batch", type=int, default=1)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--logging-steps", type=int, default=5)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument("--allow-truncation", action="store_true")
    args = parser.parse_args()
    if hasattr(torch.backends.cuda, "enable_cudnn_sdp"):
        torch.backends.cuda.enable_cudnn_sdp(args.enable_cudnn_sdpa)
    if not 0 < args.gpu_memory_fraction <= 1:
        parser.error("gpu-memory-fraction must be in (0, 1]")
    if args.gpu_memory_fraction < 1 and torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(args.gpu_memory_fraction, int(os.environ.get("LOCAL_RANK", "0")))
    if args.max_train_samples < 0 or args.max_validation_samples < 0:
        parser.error("Sample limits must be nonnegative")
    output = Path(args.output_dir)
    if output.exists() and any(output.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; use a new directory or explicit resume")
    if args.baseline_metrics and not Path(args.baseline_metrics).is_file():
        parser.error("Pre-training metrics file not found")
    if args.initial_adapter and not (Path(args.initial_adapter) / "adapter_model.safetensors").is_file():
        parser.error("Initial adapter is missing adapter_model.safetensors")
    if args.baseline_metrics:
        baseline = json.loads(Path(args.baseline_metrics).read_text())
        if baseline.get("sample_count", 0) <= 0:
            parser.error("Pre-training metrics must contain evaluated samples")
    with Path(args.train_file).open() as handle:
        first_row = json.loads(next(handle))
    if first_row.get("tools"):
        manifest = json.loads((Path(args.train_file).parent/"manifest.json").read_text())
        if not args.baseline_metrics or not args.dataset_audit:
            parser.error("Native tool-call training requires a recorded pre-training baseline and dataset audit")
        audit = json.loads(Path(args.dataset_audit).read_text())
        if not manifest.get("training_ready") or not audit.get("all_passed") or not audit.get("tokenizer_checked"):
            parser.error("Dataset/tokenizer release checks have not passed")
        if baseline.get("dataset_sha256") != manifest["split_sha256"]["test"]:
            parser.error("Baseline uses a different frozen test split")
        baseline_config = json.loads((Path(args.baseline_metrics).parent/"config.json").read_text())
        if Path(baseline_config["model"]).resolve() != Path(args.model).resolve():
            parser.error("Baseline model differs from the training checkpoint")
        recorded_adapter = baseline_config.get("adapter")
        if args.initial_adapter:
            if not recorded_adapter or Path(recorded_adapter).resolve() != Path(args.initial_adapter).resolve():
                parser.error("Baseline adapter differs from --initial-adapter")
        elif baseline.get("adapter") is not None or recorded_adapter is not None:
            parser.error("Fresh LoRA training requires an unmodified-model baseline")
        for split, source in (("train", args.train_file), ("validation", args.validation_file)):
            if hashlib.sha256(Path(source).read_bytes()).hexdigest() != manifest["split_sha256"][split]:
                parser.error("Training/validation split changed after release")
        if audit["max_tokens"] > args.max_length or args.allow_truncation:
            parser.error("Native tool-call training requires complete, untruncated assistant targets")
    set_seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    def tokenize_row(row: dict[str, Any]) -> dict[str, Any]:
        messages = json.loads(row["messages"])
        template_kwargs = {"enable_thinking": False, "preserve_thinking": False}
        tools = json.loads(row["tools_json"])
        if tools:
            template_kwargs.update(tools=tools, tool_call_format="json")
        if messages[-1]["role"] != "assistant":
            raise ValueError("SFT sample must end with the assistant target")
        prompt_text = tokenizer.apply_chat_template(messages[:-1], tokenize=False, add_generation_prompt=True, **template_kwargs)
        full_text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False, **template_kwargs)
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
        prefix = common_prefix_length(prompt_ids, full_ids)
        if prefix != len(prompt_ids):
            raise ValueError(f"chat template prefix mismatch for {row['id']}: {prefix} != {len(prompt_ids)}")
        if len(full_ids) > args.max_length:
            if not args.allow_truncation:
                raise ValueError(
                    f"sample {row['id']} has {len(full_ids)} tokens, above max_length={args.max_length}; "
                    "rerun token inspection and choose a safe length"
                )
            full_ids = full_ids[: args.max_length]
        labels = [-100] * min(prefix, len(full_ids)) + full_ids[min(prefix, len(full_ids)) :]
        if all(label == -100 for label in labels):
            raise ValueError(f"sample {row['id']} has no assistant loss tokens")
        return {"input_ids": full_ids, "labels": labels, "length": len(full_ids)}

    train_rows = load_rows(args.train_file, args.max_train_samples)
    validation_rows = load_rows(args.validation_file, args.max_validation_samples)
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / ("resume_config.json" if args.resume_from_checkpoint else "run_config.json")
    config_path.write_text(json.dumps(vars(args) | {"baseline_sha256": hashlib.sha256(Path(args.baseline_metrics).read_bytes()).hexdigest() if args.baseline_metrics else None,
                          "initial_adapter_sha256": hashlib.sha256((Path(args.initial_adapter)/"adapter_model.safetensors").read_bytes()).hexdigest() if args.initial_adapter else None,
                          "training_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          "assistant_loss_sha256": hashlib.sha256(Path(__file__).with_name("assistant_loss.py").read_bytes()).hexdigest() if args.assistant_only_projection else None,
                          "tokenizer_config_sha256": hashlib.sha256((Path(args.model)/"tokenizer_config.json").read_bytes()).hexdigest(),
                          "selected_train_ids": train_rows["id"],
                          "selected_validation_ids": validation_rows["id"]}, ensure_ascii=False, indent=2), encoding="utf-8")
    train_data = train_rows.map(tokenize_row, remove_columns=train_rows.column_names)
    validation_data = validation_rows.map(tokenize_row, remove_columns=validation_rows.column_names)

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation=args.attn_implementation,
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    candidate_suffixes = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
    observed_suffixes = {name.rsplit(".", 1)[-1] for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    target_modules = sorted(candidate_suffixes & observed_suffixes)
    if not {"q_proj", "v_proj"}.issubset(target_modules):
        raise RuntimeError(f"unexpected model module names; observed linear suffixes={sorted(observed_suffixes)}")
    print(json.dumps({"lora_target_modules": target_modules}, ensure_ascii=False), flush=True)

    if args.initial_adapter:
        model = PeftModel.from_pretrained(
            model, args.initial_adapter, is_trainable=True, local_files_only=True)
    else:
        peft_config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=target_modules,
        )
        model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.micro_batch,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        max_grad_norm=1.0,
        bf16=True,
        tf32=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=3,
        load_best_model_at_end=args.load_best_model,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        seed=args.seed,
        data_seed=args.seed,
    )
    class AssistantTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            from assistant_loss import nanbeige_assistant_loss
            # The sparse path bypasses Accelerator's model.forward wrapper. Re-enter
            # its autocast context explicitly so LoRA matmuls use the declared BF16.
            with self.accelerator.autocast():
                if self.args.bf16 and self.accelerator.device.type == "cuda":
                    if not torch.is_autocast_enabled("cuda") or torch.get_autocast_dtype("cuda") != torch.bfloat16:
                        raise RuntimeError("Sparse BF16 training requires CUDA BF16 autocast")
                if not getattr(self, "_projection_precision_logged", False):
                    print(json.dumps({"assistant_only_projection": True,
                        "cuda_autocast": torch.is_autocast_enabled("cuda"),
                        "autocast_dtype": str(torch.get_autocast_dtype("cuda"))}), flush=True)
                    self._projection_precision_logged = True
                loss = nanbeige_assistant_loss(model, inputs)
            return (loss, {"loss": loss}) if return_outputs else loss

    trainer_class = AssistantTrainer if args.assistant_only_projection else Trainer
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=validation_data,
        data_collator=CalibrationCollator(tokenizer),
    )
    if args.resume_from_checkpoint:
        from checkpoint_rng import numpy_rng_loading
        with numpy_rng_loading():
            training_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        training_result = trainer.train()
    trainer.save_metrics("train", training_result.metrics)
    trainer.save_state()
    trainer.save_model(args.output_dir + "/final_adapter")
    tokenizer.save_pretrained(args.output_dir + "/final_adapter")


if __name__ == "__main__":
    main()
