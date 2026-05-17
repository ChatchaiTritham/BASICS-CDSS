@echo off
REM Activation script for BASICS-CDSS virtual environment (Windows)

echo Activating BASICS-CDSS virtual environment...
call venv\Scripts\activate.bat

echo.
echo ========================================
echo BASICS-CDSS Environment Activated
echo ========================================
echo.
echo Python:
python --version
echo.
echo Installed packages:
pip list | findstr "basics-cdss numpy pandas pytest"
echo.
echo Quick commands:
echo   pytest tests/ -v          : Run all tests
echo   jupyter lab               : Start JupyterLab
echo   python                    : Start Python REPL
echo.
