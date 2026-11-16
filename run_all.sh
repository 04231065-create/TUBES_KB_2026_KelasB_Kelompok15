#!/usr/bin/env bash
# Menjalankan seluruh eksperimen offline secara berurutan (Linux/macOS/Git-Bash).
# Total waktu ~1 jam pada laptop 2 core. Log tersimpan di artifacts/logs/.
set -e
cd "$(dirname "$0")"
python scripts/01_eda.py
python scripts/02_baseline.py
python scripts/03_ga.py
python scripts/04_pso.py all
python scripts/04_pso.py ga
python scripts/05_compare.py
python scripts/06_fuzzy_eval.py
python scripts/07_figures.py
python scripts/09_validation_probe.py   # butuh artifacts/iter1_single_val (hasil iterasi 1)
python scripts/08_reports.py
python tests/test_core.py
