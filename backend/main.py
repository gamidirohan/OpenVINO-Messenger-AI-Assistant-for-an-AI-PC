from fastapi import FastAPI, Depends, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from typing import Annotated, List, Optional
from datetime import datetime
from telegram import Bot
from telegram.error import TelegramError
import asyncio
from chromadb import CloudClient
from sentence_transformers import SentenceTransformer
import logging

# Initialize FastAPI
app = FastAPI()

# CORS Middleware (Adjust allow_origins in production!)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration (Replace with your actual Telegram Bot Token and other settings)
TELEGRAM_BOT_TOKEN = "TELEGRAM_BOT_TOKEN"  # MUST REPLACE
VECTOR_DB_COLLECTION_NAME = "telegram_messages"
EMBEDDING_MODEL_NAME = 'all-MiniLM-L6-v2'  # Or your chosen model

# Initialize Telegram Bot, Vector DB, and Embedding Model (Do this once, ideally)
try:
    bot = Bot(token=TELEGRAM_BOT_TOKEN)
    chroma_client = CloudClient()
    collection = chroma_client.create_collection(VECTOR_DB_COLLECTION_NAME)  # Or get an existing collection
    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
except Exception as e:
    logging.error(f"Initialization error: {e}")
    # Handle initialization errors appropriately (e.g., exit the application)


# --- Telegram API Interaction ---

async def get_telegram_messages(
    start_time: Annotated[datetime, Query(alias="start_time")],
    end_time: Annotated[datetime, Query(alias="end_time")],
) -> List[dict]:
    """
    Retrieves Telegram messages within a specified time range using the Telegram Bot API.
    
    This example uses the python-telegram-bot library.  
    Consider Telethon or Pyrogram for more advanced features.
    
    This function requires significant adjustments to work correctly with 
    python-telegram-bot or any Telegram library due to the asynchronous nature 
    of Telegram API interactions.  This is a simplified illustration.
    """
    messages = []
    try:
        #  This part needs to be replaced with actual Telegram API calls
        #  using a library like python-telegram-bot.
        #  The example below is a placeholder and WILL NOT WORK as is.

        updates = await bot.get_updates()  # This is a simplified call

        for update in updates:
            if update.message and update.message.date >= start_time and update.message.date <= end_time:
                messages.append({
                    "sender": update.message.from_user.username if update.message.from_user else "Unknown",
                    "text": update.message.text,
                    "timestamp": update.message.date,
                })

    except TelegramError as e:
        logging.error(f"Telegram API Error: {e}")
        raise HTTPException(status_code=500, detail="Error communicating with Telegram API")
    except Exception as e:
        logging.error(f"Unexpected error retrieving messages: {e}")
        raise HTTPException(status_code=500, detail="Unexpected error")

    return messages


# --- FastAPI Endpoints ---

@app.get("/")
async def read_root():
    return {"message": "Hello from FastAPI Backend"}


@app.get("/messages")
async def get_messages(
    start_time: Annotated[datetime, Query(alias="start_time")],
    end_time: Annotated[datetime, Query(alias="end_time")],
    messages: List[dict] = Depends(get_telegram_messages),
):
    """
    Endpoint to retrieve Telegram messages within a specified time range.
    """
    return {"messages": messages}


@app.post("/rag-query")
async def rag_query(question: str):
    """
    Endpoint to handle RAG queries.
    """
    try:
        # 1. Generate embedding of the question
        question_embedding = embedding_model.encode(question).tolist()

        # 2. Query the vector database
        results = collection.query(
            query_embeddings=[question_embedding],
            n_results=5  # Number of relevant chunks to retrieve
        )

        # 3. Prepare context for LLM
        context = "\n".join(results['documents'][0])  # Join the retrieved chunks

        # 4. Integrate with LLM (Replace with your OpenVINO logic)
        # For now, simulate an LLM response:
        llm_response = f"Simulated LLM response based on context: {context} and question: {question}"

        response = {
            "daily_digest": "Simulated daily digest based on messages.",  # You'll need to generate this
            "assistant_response": llm_response,
        }
        return response

    except Exception as e:
        logging.error(f"Error processing RAG query: {e}")
        raise HTTPException(status_code=500, detail="Error processing RAG query")


# --- Main Execution ---

if __name__ == "__main__":
    import uvicorn
    import asyncio

    asyncio.run(uvicorn.run(app, host="0.0.0.0", port=8000))