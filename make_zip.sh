#!/usr/bin/env bash
# Build the Colab upload bundle. Drag the result into
# My Drive/NeurIPS-CCAI-2026/ and re-run Step 2 of the notebook.
set -euo pipefail
cd "$(dirname "$0")"
ZIP=phase1_2_repo.zip
rm -f "$ZIP"
# Everything git tracks, minus the cubes and older bundles. Deriving the list
# from git means a new top-level package cannot be forgotten here again, which
# is how probes/ shipped missing and broke test collection in Colab.
git ls-files -z \
  | grep -zEv '^phase1_[0-9]+_repo\.zip$' \
  | grep -zv '^data/raw/' \
  | grep -zv '^data/embeddings/' \
  | xargs -0 zip -q "$ZIP"
echo "built $ZIP ($(du -h "$ZIP" | cut -f1))"
