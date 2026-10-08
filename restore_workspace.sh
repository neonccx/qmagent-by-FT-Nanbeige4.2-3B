#!/usr/bin/env bash
set -euo pipefail

if [[ $# -gt 1 ]]; then
  echo "Usage: $0 [target-parent]" >&2
  exit 2
fi

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
asset_dir="$repo_root/release-assets"
target_parent="${1:-$repo_root/restored}"
target_dir="$target_parent/qmagent-workspace"

if [[ ! -d "$asset_dir" ]]; then
  echo "Missing $asset_dir; run: python download_workspace.py" >&2
  exit 1
fi

(cd "$asset_dir" && sha256sum --check SHA256SUMS)
mkdir -p "$target_dir"
cat "$asset_dir"/qmagent-workspace.tar.zst.part-* | zstd -d | tar -xf - --strip-components=1 -C "$target_dir"

echo "Restored to: $target_dir"
echo "Verify files with: (cd '$target_dir' && sha256sum --check '$asset_dir/FILE_MANIFEST.sha256')"
