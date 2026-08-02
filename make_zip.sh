#!/usr/bin/env bash
# Build the Colab upload bundle. Drag the result into
# My Drive/NeurIPS-CCAI-2026/ and re-run Step 2 of the notebook.
set -euo pipefail
cd "$(dirname "$0")"
rm -f phase1_1_repo.zip
zip -q -r phase1_1_repo.zip \
    data tests notebooks requirements.txt pytest.ini README.md RUNBOOK.md \
    -x "*__pycache__*" "*.pytest_cache*" "data/raw/*"
echo "built phase1_1_repo.zip ($(du -h phase1_1_repo.zip | cut -f1))"
