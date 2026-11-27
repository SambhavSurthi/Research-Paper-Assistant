@echo off
title AI Research Paper Assistant - Stop
echo Stopping AI Research Paper Assistant (app + Docker services)...
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0stop.ps1" %*
if errorlevel 1 (
    echo.
    echo [stop] Shutdown completed with errors. See logs\stop.log for details.
    pause
)
