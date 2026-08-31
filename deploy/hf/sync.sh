#!/usr/bin/env bash
# Assemble a Hugging Face Space from this repository.
#
# A Space is its own git repo and expects the Dockerfile at its root, so the
# pieces are copied rather than symlinked — a symlink into another checkout is
# not something `git push` to a Space can carry.
#
#   ./deploy/hf/sync.sh ~/hf/researchlens
#   cd ~/hf/researchlens && git add -A && git commit -m "update" && git push
set -euo pipefail

DEST="${1:?usage: sync.sh <path-to-space-checkout>}"
SRC="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [ ! -f "$SRC/data/bundle/chunks.jsonl" ]; then
  echo "No bundle at data/bundle. Build one first:" >&2
  echo "  python scripts/export_bundle.py --open-access-only" >&2
  exit 1
fi

mkdir -p "$DEST"
# Gradio, not Docker: the Docker SDK is a paid Space tier. The Dockerfile in
# this directory is kept for the Fly route and for anyone on a paid Space.
cp "$SRC/deploy/hf/app.py"           "$DEST/app.py"
cp "$SRC/deploy/hf/requirements.txt" "$DEST/requirements.txt"
cp "$SRC/deploy/hf/README.md"        "$DEST/README.md"
cp "$SRC/LICENSE"                    "$DEST/LICENSE"

for d in researchlens eval scripts; do
  rm -rf "${DEST:?}/$d"
  cp -R "$SRC/$d" "$DEST/$d"
done

rm -rf "$DEST/data"
mkdir -p "$DEST/data"
cp -R "$SRC/data/bundle" "$DEST/data/bundle"

find "$DEST" -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true

echo "Assembled in $DEST"
du -sh "$DEST/data/bundle" | sed 's/^/  bundle: /'
echo "  papers: $(python3 -c "import json;print(json.load(open('$DEST/data/bundle/manifest.json'))['papers'])")"
echo
echo "Next:"
echo "  cd $DEST && git add -A && git commit -m 'update' && git push"
