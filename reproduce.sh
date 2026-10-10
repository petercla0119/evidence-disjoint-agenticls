#!/usr/bin/env bash
# Recompute every reported number and Figure 1 from the stored responses. No API calls.
# Run inside the environment from environment.yml (Python 3.10); see README.
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONHASHSEED=0 MPLBACKEND=Agg SOURCE_DATE_EPOCH=0   # SOURCE_DATE_EPOCH: byte-stable PDFs
bash scripts/download_string.sh
py() { echo "+ $*"; python "scripts/evidence_eval/$@" > /dev/null; }
CALIB='gap>=0.059 | annotation margin 0-1.5'   # replication cell's parent bin (calibration half)

py audit_masking.py --json artifacts/evidence_eval/masking_audit.json
py arbitration.py noise-floor
py arbitration.py analyze
py arbitration.py analyze --provider azure
py tool_agent.py analyze
py tool_agent.py analyze --provider azure
py analyze_temp_arms.py
py analyze_control_retest.py
py exp20_fusion_baseline.py
py exp21_abstract_fusion_probe.py analyze
py exp22_headroom_scan.py
py exp22_headroom_scan.py --half calib
py exp23_complementary_cell.py analyze
py exp23_complementary_cell.py analyze --provider azure
py exp23_complementary_cell.py analyze --out complementary_calibsel --parent "$CALIB"
py exp23_complementary_cell.py analyze --out complementary_calibsel --parent "$CALIB" --provider azure
py exp7_scaffold_only.py analyze --set fair
py exp7_scaffold_only.py analyze --set booster
py validate_counterfactual.py analyze
py validate_counterfactual.py analyze --dir validate_booster
py reliability_calibration.py analyze
py pilot_counterfactual.py analyze
py headroom.py
py make_arbitration_figure.py
py test_reliability_comp.py
git diff --stat -- artifacts figures 2>/dev/null || true
