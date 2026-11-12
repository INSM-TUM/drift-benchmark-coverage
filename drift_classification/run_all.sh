#!/usr/bin/env bash
# Regenerate results/*.csv. The DFG-based column needs armature (pip-installed or ARMATURE_REPO=/path).
set -euo pipefail
cd "$(dirname "$0")"
P=${PYTHON:-python3}
for b in own cdrift cdlg; do $P run_verification.py $b; done
if $P -c "import armature" 2>/dev/null || [ -n "${ARMATURE_REPO:-}" ]; then
  for b in own cdrift cdlg; do $P armature_variant.py $b; done
fi
$P make_tables.py
