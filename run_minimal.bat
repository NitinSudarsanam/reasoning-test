@echo off
echo ==========================================
echo MINIMAL TRAINING RUN
echo ==========================================
echo.
echo This will:
echo   - Use 2 problems only
echo   - Run 1 training iteration
echo   - Save all generated code to training_output/
echo   - Take about 5-10 minutes
echo.
echo Activating conda environment...
call conda activate reasoning
echo.
python run_minimal_training.py
echo.
pause
