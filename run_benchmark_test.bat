@echo off
echo Testing benchmark with conda environment...
call conda activate reasoning
python benchmark_improvement.py --baseline-model "Salesforce/codegen-350M-mono" --problems-file "data/custom_problems.json" --device "cpu" --baseline-only --output test_baseline.json
