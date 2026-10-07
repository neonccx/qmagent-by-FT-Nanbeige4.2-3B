# QCal Agent 1.0.0 — single-qubit simulation adapter

This is a PEFT LoRA adapter for [Nanbeige4.2-3B](https://huggingface.co/Nanbeige/Nanbeige4.2-3B).
The upstream model card declares Apache-2.0. The base weights are **not** included
in this bundle. The included `adapter_config.json` names the public base model;
the tokenizer is unchanged from that base model.

## Load

Extract `qcal-agent-1.0.0-adapter.tar.gz`. With a compatible PyTorch,
Transformers and PEFT installation:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

base_id = "Nanbeige/Nanbeige4.2-3B"
tokenizer = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
base = AutoModelForCausalLM.from_pretrained(
    base_id, torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto"
)
model = PeftModel.from_pretrained(base, "qcal-agent-1.0.0-adapter")
```

The model's raw output is **not** a safe instrument command. Run it through
the repository's bounded action protocol and deterministic controller. For a
local simulation, use `qm-agent run --policy hf --decode-backend hf --model
<local-base-model-path> --adapter <extracted-adapter-path> --trust-remote-code
--backend physical --output-dir runs/adapter-demo`.

## Evidence and limitations

The frozen test and OOD sets contain 160 examples each. Native-call validity,
next-tool accuracy and controller-executable rate were 100% on both; exact
bounded-argument accuracy was 98.75% (test) and 100% (OOD). Three of three
fresh-seed **simulated** closed-loop episodes were accepted with no invalid
actions, tool errors or policy errors. See `portable_manifest.json` and the
repository's evaluation scripts for checksums and definitions. The prior
candidate scored 98.75% test / 94.375% OOD next-tool accuracy on the same
selected IDs, but its context transform was not recorded identically; this is
not a controlled weights-only comparison.

This release has **not** been validated on a real quantum processor. Its XEB
step is a single-qubit simulation proxy, not a measured multi-qubit XEB
benchmark. There is no coupler, two-qubit calibration or autonomous hardware
operation. Hardware use requires device-specific limits, readback, locks,
timeouts, emergency shutdown and supervised approval.
