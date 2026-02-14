@echo off
echo ============================================================
echo   Ollama Memory Proxy - Windows Installer
echo ============================================================
echo.

:: Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Install Python 3.10+ from https://python.org
    pause
    exit /b 1
)

:: Check Ollama
ollama --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Ollama not found. Install from https://ollama.com
    echo          The proxy requires Ollama running on port 11434.
    echo.
)

:: Install dependencies
echo [1/3] Installing Python dependencies...
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install dependencies.
    pause
    exit /b 1
)

:: Download embedding model (first run)
echo.
echo [2/3] Pre-downloading embedding model (~80MB, one-time)...
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
if %errorlevel% neq 0 (
    echo [WARNING] Could not pre-download model. It will download on first proxy start.
)

:: Optional: add to startup
echo.
echo [3/3] Setup complete!
echo.
set /p AUTOSTART="Add to Windows startup (auto-run on boot)? [y/N]: "
if /i "%AUTOSTART%"=="y" (
    copy /Y start_proxy.vbs "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\ollama-memory-proxy.vbs" >nul
    echo Added to startup. Proxy will auto-start on boot.
) else (
    echo Skipped. Run manually with: python run.py
)

echo.
echo ============================================================
echo   Installation complete!
echo   Start the proxy:  python run.py
echo   Then point clients to http://localhost:11435
echo ============================================================
pause
