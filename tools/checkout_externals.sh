#!/usr/bin/env bash
set -euo pipefail

# tools/checkout_externals.sh
# Reads extern_repos.txt and clones each repo into extern/<owner>_<repo>.
# By default this script only clones the repositories. If you want CI to
# pip-install or run tests for the external repos, modify the script to add
# commands after the clone step.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERN_FILE="$ROOT_DIR/extern_repos.txt"
DEST_DIR="$ROOT_DIR/extern"

if [ ! -f "$EXTERN_FILE" ] || [ ! -s "$EXTERN_FILE" ]; then
  echo "No extern_repos.txt or file is empty; skipping extern checkout"
  exit 0
fi

mkdir -p "$DEST_DIR"

while IFS= read -r line || [ -n "$line" ]; do
  repo=$(echo "$line" | sed 's/^\s*#.*$//g' | tr -d '\r' )
  repo=$(echo "$repo" | sed 's/^\s*//; s/\s*$//')
  if [ -z "$repo" ]; then
    continue
  fi
  # Normalize SSH-style URLs to HTTPS and strip trailing .git unless a @sha is present
  repo=$(echo "$repo" | sed 's#git@github.com:#https://github.com/#')
  # Extract owner_repo safe name for directory
  # If the entry has @sha suffix (e.g. ...git@sha) keep it for clone; for dir name remove @...
  clone_url="$repo"
  dir_name=$(echo "$repo" | sed 's/@.*$//' | sed 's#https://github.com/##' | sed 's#/$##' | tr '/.' '_' )
  safe_dir="$DEST_DIR/$dir_name"
  echo "Cloning $clone_url -> $safe_dir"
  # Use a shallow clone; if @sha is provided the git clone will still work
  git clone --depth 1 "$clone_url" "$safe_dir" || {
    echo "Shallow clone failed for $clone_url - trying full clone"
    rm -rf "$safe_dir"
    git clone "$clone_url" "$safe_dir"
  }

  # Optional: pip install the repo (editable) - uncomment if desired
  # echo "Installing $safe_dir"
  # python -m pip install -e "$safe_dir"

  # Optional: run the external repo's tests (if it has pytest configured)
  # echo "Running tests for $safe_dir"
  # (cd "$safe_dir" && pytest -q)

done < "$EXTERN_FILE"
