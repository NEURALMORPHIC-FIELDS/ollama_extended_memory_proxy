@echo off
REM ============================================================
REM  Ollama Memory Proxy - Transparent Startup Script
REM ============================================================
REM  This script starts:
REM    1. Ollama server on port 11436 (hidden)
REM    2. Memory proxy on port 11434 (default Ollama port)
REM
REM  All clients (Ollama app, CLI, Open WebUI) connect to 11434
REM  and automatically get persistent memory.
REM ============================================================

echo ============================================================
echo   Starting Ollama Memory Proxy (Transparent Mode)
echo ============================================================

REM Kill any existing instances (including tray app that auto-starts a server)
echo [0/2] Killing existing Ollama processes...
taskkill /IM "ollama.exe" /F >nul 2>&1
taskkill /IM "ollama app.exe" /F >nul 2>&1

REM Wait for ports to free up
timeout /t 3 /nobreak >nul

REM Double-check: kill anything still holding port 11434
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":11434.*LISTENING"') do (
    echo      Killing leftover process on 11434: PID %%a
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM Step 1: Start Ollama server on port 11436 (ONLY this process gets the env var)
echo [1/2] Starting Ollama server on port 11436...
set OLLAMA_HOST=127.0.0.1:11436
start /B "Ollama Server" "%LOCALAPPDATA%\Programs\Ollama\ollama.exe" serve

REM Wait for Ollama to be ready
echo      Waiting for Ollama...
:wait_ollama
timeout /t 1 /nobreak >nul
curl -s http://127.0.0.1:11436/api/tags >nul 2>&1
if errorlevel 1 goto wait_ollama
echo      Ollama ready on port 11436

REM Kill any Ollama app that auto-started while we were waiting
REM (the tray app sometimes respawns and grabs port 11434)
for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":11434.*LISTENING"') do (
    echo      Killing Ollama auto-start on 11434: PID %%a
    taskkill /PID %%a /F >nul 2>&1
)
timeout /t 1 /nobreak >nul

REM Step 2: Start memory proxy on port 11434
echo [2/2] Starting Memory Proxy on port 11434...
set OLLAMA_HOST=
cd /d "%~dp0"
start /B "Memory Proxy" python run.py

REM Wait for proxy to be ready
echo      Waiting for proxy...
:wait_proxy
timeout /t 1 /nobreak >nul
curl -s http://127.0.0.1:11434/api/tags >nul 2>&1
if errorlevel 1 goto wait_proxy
echo      Proxy ready on port 11434

echo ============================================================
echo   READY! All clients connect to localhost:11434 automatically
echo   Memory is persistent across conversations.
echo ============================================================
echo.
echo   Press Ctrl+C or close this window to stop both services.
echo.

REM Keep window open (both processes run in background)
pause
