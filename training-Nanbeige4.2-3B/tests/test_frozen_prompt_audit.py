import hashlib
import importlib.util
import json
from pathlib import Path
import pytest
from qmagent.contracts import decision
from qmagent.protocol import PROTOCOL_VERSION, assistant_call, policy_messages

root = Path(__file__).resolve().parents[1]
agent_root = root.parent / "qcal-agent"
spec = importlib.util.spec_from_file_location("token_audit", root / "training/audit_tokens.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def dataset(tmp_path):
    (tmp_path / "artifacts").mkdir()
    hashes = {}
    for split in ("train", "validation", "test", "ood"):
        messages = policy_messages({"state": {}, "observation": None})
        if split in ("test", "ood"):
            messages[0]["content"] = "legacy frozen instruction"
        row = {"device_id": split, "tools": [],
               "messages": messages + [assistant_call(decision("ESCALATE_HARDWARE_REVIEW", reason="test"))]}
        raw = (json.dumps(row) + "\n").encode()
        (tmp_path / f"{split}.jsonl").write_bytes(raw)
        hashes[split] = hashlib.sha256(raw).hexdigest()
    manifest = {"schema": PROTOCOL_VERSION, "split_sha256": hashes,
                "source_split_sha256": dict(hashes),
                "frozen_evaluation": {s: hashes[s] for s in ("test", "ood")},
                "complete_workflow_schema": "qcal-complete-workflow-1.0",
        "context_transform_sha256": hashlib.sha256((agent_root / "src/qmagent/protocol.py").read_bytes()).hexdigest()}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_frozen_prompt_rebuild_preserves_files(tmp_path):
    dataset(tmp_path)
    before = (tmp_path / "test.jsonl").read_bytes()
    assert module.audit(tmp_path)["frozen_evaluation_runtime_prompts_rebuilt"]
    assert (tmp_path / "test.jsonl").read_bytes() == before


def test_training_prompt_mismatch_still_fails(tmp_path):
    manifest = dataset(tmp_path)
    path = tmp_path / "train.jsonl"
    row = json.loads(path.read_text())
    row["messages"][0]["content"] = "wrong training prompt"
    path.write_text(json.dumps(row) + "\n")
    manifest["split_sha256"]["train"] = hashlib.sha256(path.read_bytes()).hexdigest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Online/training prompt mismatch"):
        module.audit(tmp_path)
