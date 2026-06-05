@echo off
setlocal EnableDelayedExpansion

:: ── Check for .env ────────────────────────────────────────────────────────────
if not exist ".env" (
    echo ERROR: .env file not found.
    echo Copy .env.example to .env and fill in your API keys.
    echo   copy .env.example .env
    echo.
    echo TIP: Store secrets securely in the OS keychain instead of .env:
    echo   uv run python setup_secrets.py
    echo.
    pause
    exit /b 1
)

:: ── Check for uv ──────────────────────────────────────────────────────────────
where uv >nul 2>&1
if errorlevel 1 (
    echo uv not found. Installing uv ^(one-time setup^)...
    powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"
    echo.
    echo uv installed. Please close this window and run run.bat again.
    pause
    exit /b 0
)

:: ── Run spotAIfy ──────────────────────────────────────────────────────────────
:: uv automatically creates a venv and installs dependencies on first run.
uv run --link-mode=copy python -m spotaify.agent %*
