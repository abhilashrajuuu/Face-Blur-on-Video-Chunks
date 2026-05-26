@echo off
echo ===================================================
echo   Face Blur Project - Auto Setup and Run Script
echo ===================================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in your PATH.
    echo Please install Python 3.11+ from python.org
    pause
    exit /b
)

:: Step 1: Create Virtual Environment
if not exist "venv" (
    echo [INFO] Creating virtual environment...
    python -m venv venv
)

:: Step 2: Activate Virtual Environment
echo [INFO] Activating virtual environment...
call venv\Scripts\activate.bat

:: Step 3: Install Dependencies
echo [INFO] Installing requirements (this may take a moment)...
pip install -r requirements.txt --quiet

:: Step 4: Download and Trim Sample Datasets
if not exist "sample_inputs\sample_image.jpg" (
    echo [INFO] Downloading sample datasets...
    python scripts\download_dataset.py
    echo [INFO] Trimming sample videos to 10 seconds...
    python scripts\trim_videos.py
)

:: Step 5: Run the Web UI
echo ===================================================
echo [INFO] Starting the Streamlit Web Interface...
echo ===================================================
streamlit run streamlit_app.py

pause
