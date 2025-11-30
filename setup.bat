@echo off
REM Setup script for adversarial RL reasoning system (Windows)

echo ==========================================
echo Adversarial RL System Setup
echo ==========================================
echo.

REM Check Python version
echo Checking Python version...
python --version
echo.

REM Create virtual environment
echo Creating virtual environment...
python -m venv venv
echo.

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Upgrade pip
echo Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Install dependencies
echo Installing dependencies...
pip install -r requirements.txt
echo.

REM Validate structure
echo Validating project structure...
python validate_structure.py
echo.

echo ==========================================
echo Setup complete!
echo ==========================================
echo.
echo Next steps:
echo 1. Activate virtual environment:
echo    venv\Scripts\activate.bat
echo 2. Run training:
echo    python run_training.py
echo.
pause
