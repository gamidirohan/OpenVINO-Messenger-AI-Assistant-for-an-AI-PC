@echo off
echo Installing dependencies for OpenVINO Messenger AI Assistant...

REM Activate virtual environment if it exists
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate.bat
) else (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
)

REM Install required packages
echo Installing required packages...
pip install -U fastapi uvicorn python-telegram-bot chromadb sentence-transformers python-dotenv
pip install -U transformers torch openvino

echo Dependencies installed successfully!
echo.
echo To convert the model to OpenVINO format, run:
echo python convert_model.py
echo.
echo To start the server, run:
echo uvicorn main:app --reload
