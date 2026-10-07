#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [target-parent]" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
asset_dir="$repo_root/release-assets"
target_parent="${1:-$repo_root/restored}"

if [[ ! -d "$asset_dir" ]]; then
  echo "Missing $asset_dir; run: python download_workspace.py" >&2
  exit 1
fi

(cd "$asset_dir" && sha256sum --check SHA256SUMS)
mkdir -p "$target_parent"
cat "$asset_dir"/bishe-workspace.tar.zst.part-* | zstd -d | tar -xf - -C "$target_parent"

echo "Restored to: $target_parent/bishe"
echo "Verify files with: (cd '$target_parent/bishe' && sha256sum --check '$asset_dir/FILE_MANIFEST.sha256')"
