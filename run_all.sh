#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export SPARK_LOCAL_IP=127.0.0.1
export PYSPARK_SUBMIT_ARGS="--driver-memory 4g pyspark-shell"
export OPENBLAS_NUM_THREADS=1
export OMP_NUM_THREADS=1
mkdir -p logs
python -u run_pipeline.py --root . 2>&1 | tee logs/explicit.log
python -u run_implicit.py --root . 2>&1 | tee logs/implicit.log
python make_figures.py
python verify_results.py
echo '实验完成。运行 streamlit run app.py 查看展示页面。'
