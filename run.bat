@echo off
title Live Face Recognition & Registration System
echo ======================================================================
echo    Starting Live Face Recognition & Registration System...
echo ======================================================================
echo.

:: Check for py launcher with Python 3.13 or default python
where py >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    py -3.13 main.py
) else (
    python main.py
)

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] Application closed with an error. Press any key to exit.
    pause >nul
)
