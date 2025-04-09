# OpenVINO Messenger AI Assistant

A desktop AI assistant for Telegram that uses OpenVINO for optimized inference. This application allows you to retrieve and analyze your Telegram messages, ask questions about your conversations, and get insights from your chat history.

## Features

- Retrieve and analyze Telegram messages with date range filtering
- Answer questions about your conversations using RAG (Retrieval-Augmented Generation)
- Store message embeddings in a local vector database for fast retrieval
- Optimized inference using Intel's OpenVINO toolkit for better performance
- Fallback mechanisms for graceful degradation when OpenVINO models are not available

## Architecture

The application consists of three main components:

1. **Frontend**: Next.js web application
2. **Backend**: FastAPI server with OpenVINO integration
3. **Electron Wrapper**: Desktop application wrapper

## Project Structure

```
openvino-messenger-ai-assistant/
├── main.js                           # Electron main process
├── package.json                      # Root package.json with scripts
├── ai-assistance-frontend/           # Next.js Frontend
│   ├── src/
│   │   ├── app/                      # Next.js pages and components
│   │   │   ├── page.tsx              # Main application page
│   │   │   ├── layout.tsx            # Root layout
│   │   │   └── globals.css           # Global styles
│   ├── public/                       # Static assets
│   ├── package.json                  # Frontend dependencies
│   └── next.config.ts                # Next.js configuration
└── backend/                          # Python FastAPI Backend
    ├── main.py                       # API endpoints and core functionality
    ├── telegram_integration.py       # Telegram API integration
    ├── vector_db.py                  # Vector database integration
    ├── openvino_integration.py       # OpenVINO model integration
    ├── .env                          # Environment variables
    └── backend-README.md             # Backend setup instructions
```

## Tech Stack

### Frontend
- [Next.js](https://nextjs.org/) (v15.2.4)
- [React](https://reactjs.org/) (v19.0.0)
- [TypeScript](https://www.typescriptlang.org/) (v5)
- [Tailwind CSS](https://tailwindcss.com/) (v4)

### Backend
- [FastAPI](https://fastapi.tiangolo.com/)
- [Telegram Bot API](https://core.telegram.org/bots/api)
- [ChromaDB](https://www.trychroma.com/) for vector storage
- [SentenceTransformer](https://www.sbert.net/) for embeddings
- [OpenVINO](https://docs.openvino.ai/) for optimized inference

### Desktop Application
- [Electron](https://www.electronjs.org/) (v28.2.1)

## Setup and Installation

### Prerequisites
- Node.js (latest LTS version)
- Python 3.8+
- Git
- OpenVINO Runtime (optional, will use fallbacks if not available)

### Quick Start
For a quick start, you can use the provided batch scripts:

```bash
# Install all dependencies and set up the project
.\setup.bat

# Run both backend and frontend
.\run_app.bat
```

### Frontend Setup
```bash
# Navigate to frontend directory
cd ai-assistance-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

### Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
.\venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies using the provided script
.\install_dependencies.bat

# Or install manually
pip install -r requirements.txt

# Start development server
uvicorn main:app --reload
```

### OpenVINO Model Setup (Optional)
To use OpenVINO optimized models:

```bash
# Navigate to backend directory with venv activated
cd backend
.\venv\Scripts\activate

# Run the model conversion script
python convert_model.py
```

### Full Application Setup
```bash
# Install root dependencies
npm install

# Start all components (frontend, backend, electron)
npm start
```

## Running the Application

### Development Mode
```bash
npm start
```

This will concurrently start:
- Next.js frontend on http://localhost:3000
- FastAPI backend on http://localhost:8000
- Electron application wrapper

### Production Build
```bash
npm run build
```

This will:
1. Build the Next.js frontend
2. Package everything into an Electron application

## API Endpoints

- `GET /`: Welcome message
- `GET /messages`: Retrieve messages within a time range
- `POST /rag-query`: Ask questions about your messages
- `POST /send-message`: Send a message to a Telegram chat
- `GET /health`: Check API health

## Contributing

### Making Changes to Frontend
```bash
# Navigate to frontend
cd ai-assistance-frontend

# Create a branch for your changes
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

### Making Changes to Backend or Electron Wrapper
```bash
# Create a branch
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit and push
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name
```

## License

ISC
