#!/usr/bin/env python3
"""Download the complete QM Agent workspace from the GitHub Release."""

from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.request import urlopen


REPOSITORY = "neonccx/qmagent-by-FT-Nanbeige4.2-3B"
TAG = "v1.0.0"
ASSETS = [
    "qmagent-workspace.tar.zst.part-00",
    "qmagent-workspace.tar.zst.part-01",
    "qmagent-workspace.tar.zst.part-02",
    "qmagent-workspace.tar.zst.part-03",
    "SHA256SUMS",
    "FILE_MANIFEST.sha256",
]


def main() -> int:
    destination = Path(__file__).resolve().parent / "release-assets"
    destination.mkdir(exist_ok=True)
    base = f"https://github.com/{REPOSITORY}/releases/download/{TAG}"
    for name in ASSETS:
        target = destination / name
        print(f"Downloading {name} -> {target}", flush=True)
        with urlopen(f"{base}/{name}") as response, target.open("wb") as output:
            while block := response.read(8 * 1024 * 1024):
                output.write(block)

    expected: dict[str, str] = {}
    for line in (destination / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split(maxsplit=1)
        expected[name.lstrip("* ")] = digest
    for name, digest in expected.items():
        hasher = hashlib.sha256()
        with (destination / name).open("rb") as stream:
            while block := stream.read(8 * 1024 * 1024):
                hasher.update(block)
        if hasher.hexdigest() != digest:
            raise RuntimeError(f"SHA256 mismatch: {name}")
    print("Snapshot downloaded and verified. Run ./restore_workspace.sh")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
