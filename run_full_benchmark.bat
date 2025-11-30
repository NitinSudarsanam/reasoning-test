@echo off
REM Complete benchmarking workflow for demonstrating adversarial RL improvement

echo ==========================================
echo ADVERSARIAL RL BENCHMARKING WORKFLOW
echo ==========================================
echo.

REM Configuration
set MODEL=Qwen/Qwen2.5-Coder-0.5B
set PROBLEMS_FILE=data/custom_problems.json
set DEVICE=cpu
set N_DISC_STEPS=5
set N_GEN_STEPS=5
set K_ALT_STEPS=3

echo Configuration:
echo   Model: %MODEL%
echo   Problems: %PROBLEMS_FILE%
echo   Device: %DEVICE%
echo   Training steps: %N_DISC_STEPS%-%N_GEN_STEPS%-%K_ALT_STEPS%
echo.

REM Step 1: Generate custom problems
echo ==========================================
echo STEP 1: Generating Custom Problems
echo ==========================================
python generate_custom_problems.py
if errorlevel 1 goto error
echo.

REM Step 2: Baseline evaluation
echo ==========================================
echo STEP 2: Evaluating Baseline Model
echo ==========================================
python benchmark_improvement.py --baseline-model "%MODEL%" --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --baseline-only --output baseline_results.json
if errorlevel 1 goto error
echo.

REM Step 3: Train with adversarial RL
echo ==========================================
echo STEP 3: Training with Adversarial RL
echo ==========================================
python run_training.py --generator-model "%MODEL%" --discriminator-model "%MODEL%" --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --n-discriminator-steps %N_DISC_STEPS% --n-generator-steps %N_GEN_STEPS% --k-alternating-steps %K_ALT_STEPS% --checkpoint-dir checkpoints_benchmark
if errorlevel 1 goto error
echo.

REM Step 4: Evaluate trained model
echo ==========================================
echo STEP 4: Evaluating Trained Model
echo ==========================================
python benchmark_improvement.py --baseline-model "%MODEL%" --trained-checkpoint checkpoints_benchmark/checkpoint_best.pt --problems-file "%PROBLEMS_FILE%" --device "%DEVICE%" --output comparison_results.json
if errorlevel 1 goto error
echo.

REM Step 5: Summary
echo ==========================================
echo BENCHMARKING COMPLETE!
echo ==========================================
echo.
echo Results saved to:
echo   - baseline_results.json
echo   - comparison_results.json
echo.
echo Checkpoints saved to:
echo   - checkpoints_benchmark/
echo.
echo View detailed comparison in comparison_results.json
echo.
goto end

:error
echo.
echo ERROR: Benchmarking failed!
echo.
exit /b 1

:end
