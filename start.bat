@echo off
title AI Research Paper Assistant - Start
echo Starting AI Research Paper Assistant (app + Docker services)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0start.ps1" %*
if errorlevel 1 (
    echo.
    echo [start] Startup failed. See logs\start.log for details.
    pause
)
