#!/usr/bin/env bash
# Run this once per clone, then, run
#  git core-pull
# or 
#  git core-push
# to sync with core subtree

set -euo pipefail

remote_url="https://github.com/pwspen/imgbench-core.git"

if git remote get-url core >/dev/null 2>&1; then
git remote set-url core "$remote_url"
else
git remote add core "$remote_url"
fi

git config alias.core-pull 'subtree pull --prefix core core main'
git config alias.core-push 'subtree push --prefix core core main'