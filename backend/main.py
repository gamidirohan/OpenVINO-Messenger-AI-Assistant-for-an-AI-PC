from fastapi import FastAPI, Depends, HTTPException, Query, Body
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List, Optional, Dict, Any
from datetime import datetime
import asyncio
import logging
import os
from dotenv import load_dotenv
from pydantic import BaseModel

# Import our custom modules
from telegram_integration import TelegramClient, create_telegram_client
from vector_db import VectorDatabase
from openvino_integration import OpenVINOModel, OpenVINOTextEmbedding, load_model

# Import sentence-transformers for embeddings
from sentence_transformers import SentenceTransformer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("app.log")
    ]
)

# Load environment variables
load_dotenv()

# Initialize FastAPI
app = FastAPI(
    title="OpenVINO Messenger AI Assistant",
    description="AI assistant for Telegram messages using OpenVINO",
    version="1.0.0"
)

# CORS Middleware with more permissive settings for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins in development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Configuration
VECTOR_DB_COLLECTION_NAME = os.getenv("VECTOR_DB_COLLECTION_NAME", "telegram_messages")
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')

# Initialize components
try:
    # Create Telegram client
    telegram_client = create_telegram_client()

    # Check if OpenVINO model exists
    embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH", "./models/embedding_model.xml")
    if os.path.exists(embedding_model_path) or os.path.exists(embedding_model_path.replace(".xml", "")):
        # Use OpenVINO for embeddings
        logging.info(f"Using OpenVINO for embeddings with model: {embedding_model_path}")
        embedding_model = OpenVINOTextEmbedding(
            model_path=embedding_model_path,
            use_fallback=True  # Fall back to sentence-transformers if OpenVINO fails
        )
    else:
        # Use sentence-transformers for real embeddings
        embedding_model_name = os.getenv("EMBEDDING_MODEL_NAME", 'all-MiniLM-L6-v2')
        logging.info(f"OpenVINO model not found. Using sentence-transformer model: {embedding_model_name}")
        embedding_model = SentenceTransformer(embedding_model_name)
        logging.info(f"Sentence-transformer model loaded successfully")

    # Initialize vector database
    vector_db = VectorDatabase(
        collection_name=VECTOR_DB_COLLECTION_NAME,
        embedding_model=embedding_model,
        client_type="persistent"  # Use persistent storage for production
    )

    # Initialize OpenVINO model with fallback
    openvino_model = load_model(use_fallback=True)

    logging.info("✅ All components initialized successfully")
except Exception as e:
    logging.error(f"Initialization error: {e}")
    # In production, you might want to exit the application here
    # For now, we'll continue and handle errors at the endpoint level


# --- Telegram API Interaction ---

async def get_telegram_messages(
    start_time: Annotated[datetime, Query(alias="start_time")],
    end_time: Annotated[datetime, Query(alias="end_time")],
    chat_id: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Retrieves Telegram messages within a specified time range.

    Args:
        start_time: Start time for message retrieval
        end_time: End time for message retrieval
        chat_id: Optional chat ID to filter messages

    Returns:
        List of message dictionaries
    """
    try:
        messages = await telegram_client.get_messages(
            start_time=start_time,
            end_time=end_time,
            chat_id=chat_id,
            limit=100
        )
        return messages
    except Exception as e:
        logging.error(f"Error retrieving Telegram messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")


# --- Request/Response Models ---

class RagQueryRequest(BaseModel):
    question: str
    chat_id: Optional[int] = None

class MessageIndexRequest(BaseModel):
    messages: List[Dict[str, Any]]

# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    """Root endpoint that returns a welcome message."""
    return {
        "message": "Welcome to the OpenVINO Messenger AI Assistant API",
        "version": "1.0.0",
        "status": "operational"
    }


@app.get("/messages")
async def get_messages(
    start_time: Annotated[datetime, Query(alias="start_time")],
    end_time: Annotated[datetime, Query(alias="end_time")],
    chat_id: Optional[int] = None,
    messages: List[Dict[str, Any]] = Depends(get_telegram_messages),
):
    """
    Endpoint to retrieve Telegram messages within a specified time range.

    Args:
        start_time: Start time for message retrieval
        end_time: End time for message retrieval
        chat_id: Optional chat ID to filter messages

    Returns:
        Dictionary with messages
    """
    # Index the retrieved messages in the vector database
    try:
        if messages:
            vector_db.index_messages(messages)
    except Exception as e:
        logging.warning(f"Failed to index messages: {e}")
        # Continue even if indexing fails

    return {"messages": messages}


@app.post("/index-messages")
async def index_messages(request: MessageIndexRequest):
    """
    Endpoint to manually index messages in the vector database.

    Args:
        request: Request containing messages to index

    Returns:
        Success message
    """
    try:
        vector_db.index_messages(request.messages)
        return {"status": "success", "message": f"Indexed {len(request.messages)} messages"}
    except Exception as e:
        logging.error(f"Error indexing messages: {e}")
        raise HTTPException(status_code=500, detail=f"Error indexing messages: {str(e)}")


@app.post("/rag-query")
async def rag_query(request: RagQueryRequest):
    """
    Endpoint to handle RAG queries.

    Args:
        request: Request containing the question and optional chat ID

    Returns:
        Dictionary with assistant response and daily digest
    """
    try:
        # 1. Query the vector database for relevant context
        results = vector_db.query(request.question, n_results=5)

        # 2. Prepare context for the LLM
        if results and 'documents' in results and results['documents'] and results['documents'][0]:
            context = "\n".join(results['documents'][0])
        else:
            context = "No relevant context found."

        # 3. Generate response using OpenVINO
        assistant_response = openvino_model.generate_response(request.question, context)

        # 4. Get daily digest
        today = datetime.now()
        daily_digest = vector_db.get_daily_digest(today)

        # 5. Format the digest as a string
        digest_text = "Daily Digest:\n\n"
        if daily_digest:
            for i, msg in enumerate(daily_digest[:5]):  # Limit to 5 messages
                digest_text += f"{i+1}. {msg.get('sender', 'Unknown')}: {msg.get('text', '')[:100]}...\n"
        else:
            digest_text += "No messages for today."

        response = {
            "daily_digest": digest_text,
            "assistant_response": assistant_response,
        }
        return response

    except Exception as e:
        logging.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing RAG query: {str(e)}")


@app.post("/send-message")
async def send_message(chat_id: int, text: str):
    """
    Endpoint to send a message to a Telegram chat.

    Args:
        chat_id: ID of the chat to send the message to
        text: Text of the message to send

    Returns:
        Dictionary with information about the sent message
    """
    try:
        result = await telegram_client.send_message(chat_id, text)
        return {"status": "success", "message": result}
    except Exception as e:
        logging.error(f"Error sending message: {e}")
        raise HTTPException(status_code=500, detail=f"Error sending message: {str(e)}")


@app.get("/health")
async def health_check():
    """
    Endpoint to check the health of the API.

    Returns:
        Dictionary with health status
    """
    return {
        "status": "healthy",
        "components": {
            "telegram": "operational",
            "vector_db": "operational",
            "openvino": "operational"
        }
    }


# --- Main Execution ---

if __name__ == "__main__":
    import uvicorn

    # Configure uvicorn server
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Enable auto-reload during development
        log_level="info"
    )