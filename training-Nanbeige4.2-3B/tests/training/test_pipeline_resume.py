from pathlib import Path

from training.run_training_pipeline import next_log_path, reconcile_interrupted_stages


def test_next_log_path_preserves_copied_attempts(tmp_path: Path):
    (tmp_path / "training.attempt_4.log").write_text("copied from H100", encoding="utf-8")
    selected = next_log_path(tmp_path, "training", 4)
    assert selected.name == "training.attempt_5.log"
    assert (tmp_path / "training.attempt_4.log").read_text(encoding="utf-8") == "copied from H100"


def test_reconcile_interrupted_stage_records_launch_error():
    state = {"error": "FileExistsError: copied log exists", "stages": [
        {"stage": "baseline_test", "status": "completed"},
        {"stage": "training", "status": "running"},
    ]}
    reconcile_interrupted_stages(state)
    assert state["stages"][0]["status"] == "completed"
    assert state["stages"][1] == {
        "stage": "training", "status": "failed", "error": "FileExistsError: copied log exists"
    }
