@echo off
echo Starting OpenVINO Messenger AI Assistant...

REM Start the backend server
start cmd /k "cd backend && python -m main"

REM Wait for the server to start
echo Waiting for the server to start...
timeout /t 5 /nobreak > nul

REM Open the frontend test page
echo Opening the frontend test page...
start frontend-test.html

echo Application started successfully!
echo Backend API: http://localhost:8000
echo Frontend Test UI: frontend-test.html
