#!/usr/bin/env bash
# Download the CDRIFT and CDLG benchmark logs (HuggingFace pm-science/cv4cdd_4d) into data/.
set -euo pipefail
cd "$(dirname "$0")"
R=https://huggingface.co/datasets/pm-science/cv4cdd_4d/resolve/main
get(){ # HF sometimes answers "maximum queue size reached"; retry
  for t in $(seq 1 40); do curl -sL -o "$2.part" "$R/$1"
    if [ "$(stat -c %s "$2.part" 2>/dev/null || stat -f %z "$2.part")" -gt 1000 ]; then mv "$2.part" "$2"; return; fi; sleep 5; done
  echo "failed: $1"; exit 1; }
mkdir -p data/cdrift data/cdlg/batches
[ -f data/cdlg/test_drift_info.csv ] || get input_cdlg/test/drift_info.csv data/cdlg/test_drift_info.csv
for i in $(seq -w 1 26); do f=data/cdlg/batches/xes_batch_00$i.zip; [ -f $f ] || get input_cdlg/test/xes_batch_00$i.zip $f; done
[ -d data/cdrift/inscope ] || { get input_cdrift/inscope.zip data/cdrift/inscope.zip; (cd data/cdrift && unzip -q inscope.zip && rm inscope.zip); }
