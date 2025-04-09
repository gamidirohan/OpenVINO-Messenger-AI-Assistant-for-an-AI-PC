@echo off
echo Setting up OpenVINO Messenger AI Assistant...

REM Create virtual environment if it doesn't exist
if not exist backend\venv (
    echo Creating virtual environment...
    cd backend
    python -m venv venv
    cd ..
) else (
    echo Virtual environment already exists.
)

REM Activate virtual environment and install dependencies
echo Installing dependencies...
cd backend
call venv\Scripts\activate
pip install -r requirements.txt

echo Setup completed successfully!
echo To start the application, run start-app.bat
