@echo off
echo ===================================================
echo     Kidnefy-AI: Automated Setup and Run Script
echo ===================================================
echo.

:: 1. Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH! Please install Python (Python 3.9 - 3.12+ recommended).
    pause
    exit /b
)

:: 2. Create Virtual Environment if it doesn't exist
if not exist ".venv\Scripts\activate.bat" (
    echo [*] Creating virtual environment (.venv)...
    python -m venv .venv
)

:: 3. Activate Virtual Environment
echo [*] Activating virtual environment...
call .venv\Scripts\activate.bat

:: Ensure pip is installed and updated in virtual environment
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [*] pip is missing in the virtual environment. Installing pip...
    python -m ensurepip --default-pip
)
python -m pip install --upgrade pip

:: 4. Install Dependencies
echo [*] Installing dependencies from requirements.txt...
python -m pip install -r requirements.txt

:: 5. Handle .env file
if not exist ".env" (
    echo [*] .env file not found. Creating one from .env.example...
    copy .env.example .env
    echo.
    echo [WARNING] I created a .env file for you. 
    echo Please open it and add your GEMINI_API_KEY if you want the Chatbot to work!
    echo.
    pause
)

:: 6. Run the Server and Frontend
echo [*] Starting FastAPI Server in a separate window...
start "Kidnefy-AI Backend API" cmd /c "call .venv\Scripts\activate.bat && uvicorn api:app --host 127.0.0.1 --port 8000 --reload"

echo [*] Starting Streamlit Frontend...
echo [*] Once running, your browser should open automatically.
echo.
python -m streamlit run frontend/streamlit_app.py

pause
