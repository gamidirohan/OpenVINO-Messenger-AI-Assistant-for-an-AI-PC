@echo off
echo Starting OpenVINO Messenger AI Assistant...

REM Start the backend server
start cmd /k "cd backend && python -m main"

REM Wait for the server to start
echo Waiting for the server to start...
timeout /t 5 /nobreak > nul

REM Start the frontend development server
start cmd /k "cd ai-assistance-frontend && npm run dev"

echo Application started successfully!
echo Backend API: http://localhost:8000
echo Frontend: http://localhost:3000

echo Press any key to open the frontend in your browser...
pause > nul
start http://localhost:3000
