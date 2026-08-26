#!/bin/sh
# Install repo hooks into .git/hooks (which git does not version).
cd "$(git rev-parse --show-toplevel)" || exit 1
for h in pre-commit; do
  cp "tools/hooks/$h" ".git/hooks/$h" && chmod +x ".git/hooks/$h"
  echo "installed .git/hooks/$h"
done
