# OpenVINO Messenger AI Assistant

A desktop AI assistant application for Telegram that retrieves messages, generates daily digests, and answers questions using RAG (Retrieval-Augmented Generation) with OpenVINO optimization.

## Project Overview

This project is a desktop application built with Electron that combines:
- A Next.js frontend for the user interface
- A FastAPI Python backend for Telegram integration and AI processing
- OpenVINO for optimized AI inference

## Project Structure

```
openvino-messenger-ai-assistant/
├── main.js                     # Electron main process
├── package.json                # Root package.json with scripts
├── ai-assistance-frontend/     # Next.js Frontend (submodule)
│   ├── src/
│   │   ├── app/                # Next.js pages and components
│   │   │   ├── page.tsx        # Main application page
│   │   │   ├── layout.tsx      # Root layout
│   │   │   └── globals.css     # Global styles
│   ├── public/                 # Static assets
│   ├── package.json            # Frontend dependencies
│   └── next.config.ts          # Next.js configuration
└── backend/                    # Python FastAPI Backend
    ├── main.py                 # API endpoints and core functionality
    ├── .env                    # Environment variables
    └── backend-README.md       # Backend setup instructions
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
- Python 3.12+
- Git

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

# Install dependencies (create a requirements.txt file)
pip install fastapi uvicorn python-telegram-bot chromadb sentence-transformers

# Start development server
uvicorn main:app --reload
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

## Contributing

This project uses a Git submodule for the frontend:

```bash
# Clone the repository with submodules
git clone --recurse-submodules https://github.com/yourusername/openvino-messenger-ai-assistant.git

# If already cloned without submodules:
git submodule update --init --recursive
```

### Making Changes to Frontend
The frontend is maintained as a separate repository and linked as a submodule:

```bash
# Navigate to frontend
cd ai-assistance-frontend

# Create a branch for your changes
git checkout -b feature/your-feature-name

# Make your changes
# ...

# Commit and push to the frontend repo
git add .
git commit -m "Add your feature"
git push origin feature/your-feature-name

# Create a pull request in the frontend repository
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

# Create a pull request
```

## Features

- Telegram message retrieval within specified time ranges
- RAG-based question answering using message history
- Daily digest generation from message content
- Cross-platform desktop application support

## License

ISC