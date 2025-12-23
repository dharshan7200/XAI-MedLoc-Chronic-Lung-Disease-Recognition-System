@echo off
cd /d "%~dp0"
.\.venv\Scripts\python.exe plot_metrics.py
pause
