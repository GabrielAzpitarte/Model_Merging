#!/usr/bin/env bash
# Local backup (no network): git bundle of every ref + a tarball of the working
# tree (tracked and untracked, excluding the artifacts symlink and caches) on a
# second filesystem. Keeps the 20 most recent snapshots.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${CONDMERGE_BACKUP:-/shared/ssd_30T/azpitarte_condmerge_backup}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DEST/$STAMP"
cd "$REPO"
if git rev-parse --verify HEAD >/dev/null 2>&1; then
  git bundle create "$DEST/$STAMP/condmerge.bundle" --all >/dev/null 2>&1
fi
tar --exclude=./artifacts --exclude='*/__pycache__' --exclude=./.pytest_cache \
    --exclude=./.ruff_cache -czf "$DEST/$STAMP/worktree.tar.gz" .
git log --oneline -5 > "$DEST/$STAMP/HEAD.txt" 2>/dev/null || true
ls -1dt "$DEST"/2* | tail -n +21 | xargs -r rm -rf --
du -sh "$DEST/$STAMP" | sed 's/^/backup: /'
