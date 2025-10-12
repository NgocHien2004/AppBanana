@echo off
title Banana API - Control Panel
color 0A

echo.
echo  ========================================
echo      BANANA API - DEPLOYMENT
echo  ========================================
echo.
echo  Public URL: https://terete-todd-gratulant.ngrok-free.dev
echo  Local URL:  http://localhost:8000
echo.
echo  ========================================
echo.

REM Kiểm tra ngrok
if not exist "ngrok.exe" (
    echo [ERROR] ngrok.exe not found in current directory!
    echo Please download ngrok to: D:\flutter_project\
    pause
    exit /b 1
)

REM Kiểm tra venv
if not exist "venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found!
    echo Please create venv: python -m venv venv
    pause
    exit /b 1
)

REM Kiểm tra backend files
if not exist "banana_backend\main.py" (
    echo [ERROR] Backend files not found!
    echo Please check banana_backend folder
    pause
    exit /b 1
)

echo [1/2] Starting Backend Server...
start "Backend Server" cmd /k "cd /d D:\flutter_project\banana_backend && call D:\flutter_project\venv\Scripts\activate && python main.py"

echo [INFO] Waiting for backend to start...
timeout /t 5 /nobreak >nul

echo [2/2] Starting Ngrok Tunnel...
start "Ngrok Tunnel" cmd /k "cd /d D:\flutter_project && ngrok.exe http 8000 --domain=terete-todd-gratulant.ngrok-free.dev"

echo.
echo  ========================================
echo      DEPLOYMENT COMPLETE!
echo  ========================================
echo.
echo  Backend Window: Check "Backend Server" window
echo  Ngrok Window:   Check "Ngrok Tunnel" window
echo.
echo  Test Backend:   http://localhost:8000/health
echo  Test Public:    https://terete-todd-gratulant.ngrok-free.dev/health
echo.
echo  Ngrok Inspector: http://127.0.0.1:4040
echo.
echo  Press any key to exit this control panel
echo  (Backend and Ngrok will keep running)
echo  ========================================
echo.
pause