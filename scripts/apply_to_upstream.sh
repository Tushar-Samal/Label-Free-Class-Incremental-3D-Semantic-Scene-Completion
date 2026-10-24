#!/usr/bin/env bash
# Build a runnable working copy: clone upstream DISC at the pinned commit, copy the new files of this
# repository into it, and apply the patches to the upstream files that were modified.
#
#   bash scripts/apply_to_upstream.sh [target_dir]      (default target_dir: DISC-dcil)
set -euo pipefail

UPSTREAM_URL="https://github.com/Enyu-Liu/DISC.git"
UPSTREAM_COMMIT="00ecc7fc592e07558b9f5516b4c9fbada6892b4e"

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:-DISC-dcil}"

[ -e "$OUT" ] && { echo "error: '$OUT' already exists" >&2; exit 1; }

git clone "$UPSTREAM_URL" "$OUT"
git -C "$OUT" checkout --quiet "$UPSTREAM_COMMIT"

# new files (same relative paths as in the upstream tree)
cp -r "$HERE/tools" "$HERE/ssc_ia" "$HERE/configs" "$OUT/"

# changes to existing upstream files
OUT_ABS="$(cd "$OUT" && pwd)"
(cd "$HERE" && find patches -name '*.patch' | sort) | while read -r p; do
    git -C "$OUT_ABS" apply --whitespace=nowarn "$HERE/$p"
done

echo "Done. Working copy: $OUT_ABS (upstream $UPSTREAM_COMMIT + this repository)"
echo "Next: install the requirements, put the datasets and pretrained weights in place, and export PYTHONPATH=\$(pwd)"
