"""Download the upstream base model and the released QCal Agent LoRA."""

from hashlib import sha256
from pathlib import Path
import tarfile
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
MODEL_ROOT = ROOT / "training-Nanbeige4.2-3B" / "models"
BASE_ID = "Nanbeige/Nanbeige4.2-3B"
ADAPTER_URL = (
    "https://github.com/neonccx/qmagent-by-FT-Nanbeige4.2-3B/"
    "releases/download/v1.0.0/qcal-agent-1.0.0-adapter.tar.gz"
)
ADAPTER_SHA256 = "94b3433e74c663dbe79c6d28a261d913831198891a0e22e47d1b57b3aab69337"


def main():
    from huggingface_hub import snapshot_download

    MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=BASE_ID, local_dir=MODEL_ROOT / "Nanbeige4.2-3B")
    archive = MODEL_ROOT / "qcal-agent-1.0.0-adapter.tar.gz"
    with urlopen(ADAPTER_URL) as response, archive.open("wb") as output:
        while chunk := response.read(1024 * 1024):
            output.write(chunk)
    if sha256(archive.read_bytes()).hexdigest() != ADAPTER_SHA256:
        archive.unlink(missing_ok=True)
        raise ValueError("LoRA archive checksum mismatch")
    with tarfile.open(archive) as bundle:
        bundle.extractall(MODEL_ROOT, filter="data")
    archive.unlink()
    print(f"Base model and LoRA are ready under {MODEL_ROOT}")


if __name__ == "__main__":
    main()
