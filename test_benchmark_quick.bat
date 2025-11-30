@echo off
REM Quick test of benchmark pipeline (5 minutes)

echo ==========================================
echo QUICK BENCHMARK TEST
echo ==========================================
echo Testing the benchmark pipeline with minimal settings
echo This will take about 5 minutes
echo.
echo Activating conda environment...
call conda activate reasoning
echo.

set MODEL=Salesforce/codegen-350M-mono
set PROBLEMS_FILE=data/custom_problems.json
set DEVICE=cpu

echo Step 1: Generate problems
python generate_custom_problems.py
if errorlevel 1 goto error

echo.
echo Step 2: Test baseline evaluation
python benchmark_improvement.py --baseline-model "%MODEL%" --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --baseline-only --output test_baseline.json
if errorlevel 1 goto error

echo.
echo ==========================================
echo QUICK TEST COMPLETE!
echo ==========================================
echo Baseline evaluation works correctly
echo You can now run the full benchmark with:
echo   run_cpu_benchmark_auto.bat
echo.
goto end

:error
echo.
echo ERROR: Test failed!
echo.
exit /b 1

:end
