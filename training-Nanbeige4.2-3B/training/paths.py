"""Shared layout checks for the separate training and Agent repositories."""

from pathlib import Path

TRAINING_ROOT = Path(__file__).resolve().parents[1]
AGENT_ROOT = TRAINING_ROOT.parent / "qcal-agent"
AGENT_SRC = AGENT_ROOT / "src"


def require_agent_source() -> Path:
    if not (AGENT_SRC / "qmagent" / "__init__.py").is_file():
        raise RuntimeError(
            "Missing ../qcal-agent/src. Download qcal-agent beside "
            "training-Nanbeige4.2-3B before running training tools."
        )
    return AGENT_SRC
