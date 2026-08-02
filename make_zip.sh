#!/usr/bin/env bash
# Build the Colab upload bundle. Drag the result into
# My Drive/NeurIPS-CCAI-2026/ and re-run Step 2 of the notebook.
set -euo pipefail
cd "$(dirname "$0")"
rm -f phase1_1_repo.zip
# Everything git tracks, minus the cubes. Deriving the list from git means a
# new top-level package cannot be forgotten here again, which is how probes/
# shipped missing and broke test collection in Colab.
git ls-files -z \
  | grep -zv '^phase1_1_repo\.zip$' \
  | grep -zv '^data/raw/' \
  | xargs -0 zip -q phase1_1_repo.zip
echo "built phase1_1_repo.zip ($(du -h phase1_1_repo.zip | cut -f1))"
