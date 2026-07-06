@echo off
echo ========================================
echo   Bibliometrics Analysis Platform
echo   Starting Streamlit Server...
echo ========================================
echo.
cd /d "%~dp0"
streamlit run app.py --server.port 8501 --server.address localhost --server.headless true
pause
