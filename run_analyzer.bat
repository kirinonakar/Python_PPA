@echo off
title Pro Portfolio Analyzer Launcher
echo Starting Pro Portfolio Analyzer...
echo.

call .\.venv\Scripts\activate

:: Move to the application directory
cd /d "%~dp0portfolio_app"

:: Check if streamlit is installed
where streamlit >nul 2>nul
if %errorlevel% neq 0 (
    echo Error: Streamlit is not found in your PATH.
    echo Please ensure Python and Streamlit are installed.
    echo You can install it via: pip install streamlit yfinance pandas plotly scipy
    pause
    exit /b
)

:: Run the application
streamlit run app.py

pause
