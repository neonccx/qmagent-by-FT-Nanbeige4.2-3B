# QCal Agent training assets

This directory preserves the public inputs and outputs for the QCal Agent 1.0
training release.

## Training code

- `../qcal-agent/`: Agent runtime, protocol, controller and simulator source used
  directly by the separate training tools.
- The immutable dataset release corresponds to the `dataset-v1.0.0` Git tag.

## Dataset

- Extracted dataset: `datasets/qcal-agent-v1.0.0/`
- Original release archive and checksum file: `releases/dataset-v1.0.0/`
- Source release:
  <https://github.com/neonccx/qcal-agent/releases/tag/dataset-v1.0.0>

Published split sizes:

- train: 4,637
- validation: 782
- test: 511
- OOD: 986

All six files listed by the release `SHA256SUMS` pass checksum verification.
The GitHub archive digest is
`083ce73c192ec50c05d190a67bc290157baae8e2c9cba9e48017236a9f54ad5b`.

Important limitation: the release archive does not contain the 6,375 raw JSON
artifacts referenced by dataset rows, although `tokenizer_audit.json` records
them. Therefore split integrity can be verified with the published checksums,
but the current `training/audit_dataset.py` raw-artifact completeness check fails.
Do not represent this public archive as a complete raw-artifact snapshot.

The training pipeline accepts this dataset directory through:

```bash
python training/run_training_pipeline.py \
  --model models/Nanbeige4.2-3B \
  --dataset datasets/qcal-agent-v1.0.0 \
  --run-dir /path/to/new-run
```

The raw-artifact limitation above must be resolved before rerunning the full
fail-closed pipeline unchanged.

## Published model

- Model-card Git repository: `qcal-agent-model/`
- Original LoRA release files: `releases/model-v1.0.0/`
- Source release:
  <https://github.com/neonccx/qcal-agent-model/releases/tag/v1.0.0>

The adapter archive and portable manifest pass the published SHA-256 checks.
The base Nanbeige model is intentionally not included in that release.
