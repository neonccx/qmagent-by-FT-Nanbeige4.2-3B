#!/usr/bin/env bash

export TRAINING_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export HF_HOME="$TRAINING_ROOT/cache/huggingface"
export HF_HUB_CACHE="$HF_HOME/hub"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
export TORCH_HOME="$TRAINING_ROOT/cache/torch"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

# Activate the Conda environment before sourcing this file. Keeping PATH out of
# this script makes the checkout portable across usernames and install paths.
