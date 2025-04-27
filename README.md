# OpenVINO Messenger AI Assistant

A desktop AI assistant for Telegram that uses OpenVINO for optimized inference. This application allows you to retrieve and analyze your Telegram messages, ask questions about your conversations, and get insights from your chat history using semantic search and retrieval-augmented generation.

## Overview

This project demonstrates how to build an AI-powered assistant that can process and analyze Telegram messages. It uses sentence-transformers for generating embeddings, with the option to optimize performance using Intel's OpenVINO toolkit. The application features a modern web interface built with Next.js and a robust backend API built with FastAPI.

## Features

- **Message Retrieval**: Fetch Telegram messages with date range filtering
- **Semantic Search**: Find relevant messages based on meaning, not just keywords
- **Question Answering**: Ask questions about your conversations using RAG (Retrieval-Augmented Generation)
- **Vector Database**: Store message embeddings in a local ChromaDB database for fast retrieval
- **OpenVINO Integration**: Optional optimized inference using Intel's OpenVINO toolkit
- **Fallback Mechanisms**: Graceful degradation when OpenVINO models are not available
- **Modern UI**: Clean, responsive interface with error handling and loading states

## Architecture

The application follows a client-server architecture with three main components:

1. **Frontend**: Next.js web application that provides the user interface
2. **Backend**: FastAPI server that handles API requests, Telegram integration, and AI processing
3. **Electron Wrapper**: Desktop application wrapper that packages the frontend and backend together

The backend also integrates with ChromaDB for vector storage and retrieval.

## Project Structure

```
openvino-messenger-ai-assistant/
├── .gitignore                        # Git ignore file
├── README.md                         # Project documentation
├── main.js                           # Electron main process
├── preload.js                        # Electron preload script
├── package.json                      # Root package.json with scripts
├── run_app.bat                       # Script to run both backend and frontend
├── ai-assistance-frontend/           # Next.js Frontend
│   ├── src/
│   │   ├── app/                      # Next.js pages and components
│   │   │   ├── page.tsx              # Main application page
│   │   │   ├── layout.tsx            # Root layout
│   │   │   └── globals.css           # Global styles
│   ├── public/                       # Static assets
│   ├── package.json                  # Frontend dependencies
│   └── next.config.js                # Next.js configuration
└── backend/                          # Python FastAPI Backend
    ├── main.py                       # API endpoints and core functionality
    ├── telegram_integration.py       # Telegram API integration
    ├── vector_db.py                  # Vector database integration
    ├── openvino_integration.py       # OpenVINO model integration
    ├── convert_model.py              # Script to convert models to OpenVINO format
    ├── install_dependencies.bat      # Script to install backend dependencies
    ├── requirements.txt              # Python dependencies
    ├── .env                          # Environment variables (not in repo)
    └── chroma_db/                    # Local vector database storage (not in repo)
```

## Tech Stack

### Frontend
- **[Next.js](https://nextjs.org/)**: React framework for building the user interface
- **[React](https://reactjs.org/)**: JavaScript library for building user interfaces
- **[TypeScript](https://www.typescriptlang.org/)**: Typed JavaScript for better code quality
- **[CSS Modules](https://github.com/css-modules/css-modules)**: Scoped CSS for component styling

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)**: Modern, fast web framework for building APIs with Python
- **[Telegram Bot API](https://core.telegram.org/bots/api)**: API for interacting with Telegram
- **[ChromaDB](https://www.trychroma.com/)**: Embedded vector database for storing and retrieving embeddings
- **[Sentence-Transformers](https://www.sbert.net/)**: Library for generating text embeddings
- **[OpenVINO](https://docs.openvino.ai/)**: Toolkit for optimizing and deploying AI models
- **[Python-Telegram-Bot](https://python-telegram-bot.org/)**: Python wrapper for the Telegram Bot API
- **[Uvicorn](https://www.uvicorn.org/)**: ASGI server for running the FastAPI application

### Desktop Application
- **[Electron](https://www.electronjs.org/)**: Framework for building cross-platform desktop applications
- **[Concurrently](https://www.npmjs.com/package/concurrently)**: Run multiple commands concurrently

### Development Tools
- **[Git](https://git-scm.com/)**: Version control system
- **[Python](https://www.python.org/)**: Programming language for the backend
- **[Node.js](https://nodejs.org/)**: JavaScript runtime for the frontend
- **[Visual Studio Code](https://code.visualstudio.com/)**: Recommended IDE for development

## Setup and Installation

### Prerequisites
- **Node.js**: Latest LTS version (for frontend)
- **Python 3.8+**: For backend development
- **Git**: For version control
- **Telegram Bot Token**: Create a bot via [BotFather](https://t.me/botfather)
- **OpenVINO Runtime**: Optional, will use fallbacks if not available

### Quick Start
For a quick start, you can use the provided batch script:

```bash
# Run both backend and frontend as web applications
.\run_app.bat

# Or run as a desktop application with Electron
npm start
```

### Step-by-Step Setup

#### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/openvino-messenger-ai-assistant.git
cd openvino-messenger-ai-assistant
```

#### 2. Backend Setup
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

# Create a .env file with your Telegram bot token
echo TELEGRAM_BOT_TOKEN=your_bot_token_here > .env

# Start development server
uvicorn main:app --reload
```

#### 3. Frontend Setup
```bash
# Navigate to frontend directory
cd ai-assistance-frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

#### 4. OpenVINO Model Setup (Optional)
To use OpenVINO optimized models:

```bash
# Navigate to backend directory with venv activated
cd backend
.\venv\Scripts\activate

# Run the model conversion script
python convert_model.py
```

### Configuration

Create a `.env` file in the backend directory with the following variables:

```
TELEGRAM_BOT_TOKEN=your_bot_token_here
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```

## Core Components

### Backend Modules

- **main.py**: Core API endpoints and application initialization
- **telegram_integration.py**: Handles communication with the Telegram API
- **vector_db.py**: Manages the vector database for storing and retrieving embeddings
- **openvino_integration.py**: Provides OpenVINO model integration with fallback mechanisms
- **convert_model.py**: Utility for converting models to OpenVINO format

### API Endpoints

- **`GET /`**: Welcome message and API status
- **`GET /messages`**: Retrieve messages within a specified time range
  - Query parameters: `start_time`, `end_time`, `chat_id` (optional)
- **`POST /rag-query`**: Ask questions about your messages using RAG
  - Body: `{"question": "Your question here"}`
- **`GET /health`**: Check API health status

### Frontend Components

- **page.tsx**: Main application page with UI components and API integration
- **globals.css**: Global styles for the application
- **layout.tsx**: Root layout component for the application

## Current Features and Limitations

### What Works
- ✅ Telegram message retrieval with date filtering
- ✅ Vector storage of message embeddings using ChromaDB
- ✅ Text embeddings using Sentence-Transformers
- ✅ Question answering using RAG (Retrieval-Augmented Generation)
- ✅ OpenVINO integration with fallback mechanisms
- ✅ Modern web interface with error handling

### Limitations
- ⚠️ Currently uses simulated responses for the LLM component
- ⚠️ Requires manual setup of a Telegram bot
- ⚠️ Limited to text messages (no media processing yet)

## Future Improvements

- Implement a real LLM for better question answering
- Add support for processing media messages (images, documents)
- Improve the UI with more visualization options
- Add user authentication and multi-user support
- Implement message summarization and topic clustering

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature-name`
3. Make your changes
4. Commit your changes: `git commit -m "Add your feature"`
5. Push to the branch: `git push origin feature/your-feature-name`
6. Open a pull request

## License

MIT
