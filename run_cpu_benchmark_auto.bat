@echo off
REM CPU-Optimized Benchmarking (30-40 minutes total) - AUTO MODE

echo ==========================================
echo CPU-OPTIMIZED ADVERSARIAL RL BENCHMARK
echo ==========================================
echo.
echo This will take approximately 30-40 minutes
echo Running in automatic mode...
echo.
echo Activating conda environment...
call conda activate reasoning
echo.

REM Configuration - Optimized for CPU speed
set MODEL=Salesforce/codegen-350M-mono
set PROBLEMS_FILE=data/function_problems.json
set DEVICE=cpu
set N_DISC_STEPS=2
set N_GEN_STEPS=2
set K_ALT_STEPS=1

echo Configuration:
echo   Model: %MODEL% (350M - fastest on CPU)
echo   Problems: %PROBLEMS_FILE%
echo   Device: %DEVICE%
echo   Training steps: %N_DISC_STEPS%-%N_GEN_STEPS%-%K_ALT_STEPS% (minimal for speed)
echo.

REM Step 1: Generate custom problems (using pre-made for speed)
echo ==========================================
echo STEP 1/4: Generating Custom Problems
echo ==========================================
echo Using pre-made custom problems...
python generate_custom_problems.py
if errorlevel 1 goto error
echo.

REM Step 2: Baseline evaluation
echo ==========================================
echo STEP 2/4: Evaluating Baseline Model
echo ==========================================
echo Time: ~5 minutes
echo Loading model and testing on 5 problems...
echo.
python benchmark_improvement.py --baseline-model "%MODEL%" --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --baseline-only --output baseline_results.json
if errorlevel 1 goto error
echo.

REM Step 3: Train with adversarial RL
echo ==========================================
echo STEP 3/4: Training with Adversarial RL
echo ==========================================
echo Time: ~20 minutes
echo Training 5 problems across 5 stages...
echo.
python run_training.py --generator-model "%MODEL%" --discriminator-model "%MODEL%" --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --n-discriminator-steps %N_DISC_STEPS% --n-generator-steps %N_GEN_STEPS% --k-alternating-steps %K_ALT_STEPS% --checkpoint-dir checkpoints_cpu
if errorlevel 1 goto error
echo.

REM Step 4: Evaluate trained model
echo ==========================================
echo STEP 4/4: Evaluating Trained Model
echo ==========================================
echo Time: ~5 minutes
echo Testing trained model on same problems...
echo.
python benchmark_improvement.py --baseline-model "%MODEL%" --trained-checkpoint checkpoints_cpu/checkpoint_best.pt --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --output comparison_results.json
if errorlevel 1 goto error
echo.

REM Success!
echo ==========================================
echo BENCHMARKING COMPLETE!
echo ==========================================
echo.
echo Results saved to:
echo   - baseline_results.json
echo   - comparison_results.json
echo.
echo Checkpoints saved to:
echo   - checkpoints_cpu/
echo.
echo Open comparison_results.json to see improvement metrics!
echo.
goto end

:error
echo.
echo ERROR: Benchmarking failed!
echo Check the error messages above
echo.
exit /b 1

:end
