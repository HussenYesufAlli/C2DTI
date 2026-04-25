#!/usr/bin/env bash
set -euo pipefail

# Standardized workflow for share-ready C2DTI updates.
# 1) Validate configs
# 2) Run current 6 embedding-reference experiments
# 3) Compile one supervisor-facing report

cd "$(dirname "$0")/.."

echo "[STEP] Dry-run validation"
python3 scripts/run_binary.py --config configs/davis_binary_realemb.yaml --dry-run
python3 scripts/run_binary.py --config configs/kiba_binary_realemb.yaml --dry-run
python3 scripts/run_binary.py --config configs/bindingdb_kd_binary_realemb.yaml --dry-run
python3 scripts/run.py --config configs/davis_regression_realemb.yaml --dry-run
python3 scripts/run.py --config configs/kiba_regression_realemb.yaml --dry-run
python3 scripts/run.py --config configs/bindingdb_kd_regression_realemb.yaml --dry-run

echo "[STEP] Binary runs"
python3 scripts/run_binary.py --config configs/davis_binary_realemb.yaml --run-once
python3 scripts/run_binary.py --config configs/kiba_binary_realemb.yaml --run-once
python3 scripts/run_binary.py --config configs/bindingdb_kd_binary_realemb.yaml --run-once

echo "[STEP] Regression runs"
python3 scripts/run.py --config configs/davis_regression_realemb.yaml --run-once
python3 scripts/run.py --config configs/kiba_regression_realemb.yaml --run-once
python3 scripts/run.py --config configs/bindingdb_kd_regression_realemb.yaml --run-once

echo "[STEP] Compile supervisor report"
python3 scripts/compile_supervisor_report.py

echo "[DONE] See outputs/reports/SUPERVISOR_PROGRESS.md"
