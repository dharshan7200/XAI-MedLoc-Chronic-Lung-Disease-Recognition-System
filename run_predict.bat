@echo off
cd /d "%~dp0"
echo Running prediction using isolated environment...
.\.venv\Scripts\python.exe predict.py
if errorlevel 1 (
    echo.
    echo Error: Prediction failed.
) else (
    echo.
    echo Success! Output saved to sample_output.png
)
pause
