"""Project only supervised causal positions through the vocabulary head.

All decoder context and attention computation is unchanged. This is the same mean
cross-entropy objective as ignoring -100 labels after projecting the whole sequence.
"""
import torch
import torch.nn.functional as F


def assistant_loss(hidden_states, labels, head):
    targets = labels[..., 1:]
    selected = targets != -100
    if not selected.any():
        raise ValueError("No supervised causal positions")
    hidden = hidden_states[..., :-1, :][selected]
    logits = head(hidden).float()
    return F.cross_entropy(logits, targets[selected].to(logits.device))


def nanbeige_assistant_loss(model, inputs):
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if base.__class__.__name__ != "NanbeigeForCausalLM" or base.config.pretraining_tp != 1:
        raise ValueError("Sparse projection has only been audited for NanbeigeForCausalLM with pretraining_tp=1")
    labels = inputs["labels"]
    outputs = base.model(**{key: value for key, value in inputs.items() if key != "labels"},
                         use_cache=False, return_dict=True)
    return assistant_loss(outputs.last_hidden_state, labels, base.lm_head)
