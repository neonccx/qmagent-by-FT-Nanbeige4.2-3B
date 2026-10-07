# Server model configuration

This directory intentionally contains no sessions, reports, credentials, or
machine-specific `config.json`. Create the server configuration from the example:

```bash
cp qcal-agent-home/config.example.json qcal-agent-home/config.json
```

Replace both `/ABSOLUTE/PATH/TO/REPOSITORY` values with the absolute path of this
repository on the model server. `model-rpc` requires `policy` to be `hf`.
