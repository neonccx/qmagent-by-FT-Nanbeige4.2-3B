"""Check the real local Nanbeige checkpoint before enabling sparse projection."""
import argparse
import gc
import json
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed

from assistant_loss import nanbeige_assistant_loss
from train_lora import CalibrationCollator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError("Verification output already exists")
    set_seed(20260904)
    torch.backends.cuda.enable_cudnn_sdp(False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True,
                                             local_files_only=True, use_fast=False)
    tokenizer.pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    features = []
    for line in Path(args.dataset).read_text().splitlines():
        row = json.loads(line)
        options = dict(tools=row["tools"], tool_call_format="json",
                       enable_thinking=False, preserve_thinking=False)
        prompt = tokenizer.apply_chat_template(row["messages"][:-1], tokenize=True,
                                               add_generation_prompt=True, **options)
        full = tokenizer.apply_chat_template(row["messages"], tokenize=True,
                                             add_generation_prompt=False, **options)
        assert full[:len(prompt)] == prompt
        features.append(dict(input_ids=full, labels=[-100]*len(prompt)+full[len(prompt):]))
    selected = [min(features, key=lambda x: len(x["input_ids"])),
                max(features, key=lambda x: len(x["input_ids"]))]
    model = AutoModelForCausalLM.from_pretrained(
        args.model, trust_remote_code=True, local_files_only=True,
        torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = get_peft_model(model, LoraConfig(
        r=32, lora_alpha=64, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    # Nonzero B tests gradient paths through both LoRA factors, not only initial B.
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if "lora_B" in name:
                parameter.normal_(0, 0.001)
    model.train()
    report = dict(model=args.model, dropout=0.05, precision="bfloat16 autocast",
                  loss_atol=0.02, gradient_relative_l2_max=0.05, cases=[])
    for feature in selected:
        inputs = {key: value.cuda() for key, value in CalibrationCollator(tokenizer)([feature]).items()}
        case = dict(tokens=len(feature["input_ids"]), supervised=sum(x != -100 for x in feature["labels"]))
        reference = None
        for mode in ("full", "assistant_only"):
            model.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            set_seed(20260904)
            torch.cuda.synchronize()
            start = time.perf_counter()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(**inputs).loss if mode == "full" else nanbeige_assistant_loss(model, inputs)
            loss.backward()
            torch.cuda.synchronize()
            case[mode] = dict(loss=float(loss.detach()), seconds=time.perf_counter()-start,
                              peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20)
            gradients = {name: parameter.grad.detach().float().cpu() for name, parameter in model.named_parameters()
                         if parameter.requires_grad and parameter.grad is not None}
            if reference is None:
                reference = gradients
            else:
                assert gradients.keys() == reference.keys()
                numerator = sum(float((gradients[name]-reference[name]).square().sum()) for name in reference)
                denominator = sum(float(value.square().sum()) for value in reference.values())
                case["gradient_relative_l2"] = (numerator/max(denominator, 1e-30))**0.5
            del loss
        case["loss_difference"] = abs(case["full"]["loss"]-case["assistant_only"]["loss"])
        case["passed"] = case["loss_difference"] <= report["loss_atol"] and case["gradient_relative_l2"] <= report["gradient_relative_l2_max"]
        report["cases"].append(case)
        print(json.dumps(case), flush=True)
    report["all_passed"] = all(case["passed"] for case in report["cases"])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2)+"\n")
    if not report["all_passed"]:
        raise SystemExit("Numerical verification failed; do not enable sparse projection")


if __name__ == "__main__":
    main()
