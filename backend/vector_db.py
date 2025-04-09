"""
Vector Database Module for AI Assistant.

This module provides integration with ChromaDB for storing and retrieving
vector embeddings of messages for efficient semantic search.
"""

from chromadb import CloudClient, Client, PersistentClient
import logging
from typing import List, Dict, Any, Optional, Union
import os
from datetime import datetime
import json
import hashlib

class VectorDatabase:
    """
    Vector database for storing and retrieving message embeddings.

    This class provides methods to index messages and perform semantic
    search using vector embeddings.
    """

    def __init__(self, collection_name: str, embedding_model, client_type: str = "persistent"):
        """
        Initialize the vector database.

        Args:
            collection_name: Name of the collection to use
            embedding_model: Model to use for generating embeddings
            client_type: Type of ChromaDB client to use ("cloud", "persistent", or "memory")
        """
        self.embedding_model = embedding_model
        self.collection_name = collection_name

        try:
            # Default to persistent local storage
            db_path = os.getenv("CHROMADB_PATH", "./chroma_db")

            # Create the directory if it doesn't exist
            os.makedirs(db_path, exist_ok=True)

            logging.info(f"Initializing ChromaDB with {client_type} client")

            if client_type == "cloud":
                # For cloud deployment
                api_key = os.getenv("CHROMADB_API_KEY")
                api_url = os.getenv("CHROMADB_API_URL")
                if not api_key or not api_url:
                    logging.warning("Missing ChromaDB cloud credentials, falling back to persistent storage")
                    self.client = PersistentClient(path=db_path)
                else:
                    self.client = CloudClient(api_key=api_key, api_url=api_url)
            elif client_type == "persistent":
                # For local persistent storage
                logging.info(f"Using persistent ChromaDB at {db_path}")
                self.client = PersistentClient(path=db_path)
            else:
                # In-memory client for testing
                logging.info("Using in-memory ChromaDB (data will be lost when server restarts)")
                from chromadb import Client as MemoryClient
                self.client = MemoryClient()

            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"description": "Telegram message embeddings for AI assistant"}
            )

            logging.info(f"Vector database initialized with collection: {collection_name}")
        except Exception as e:
            logging.error(f"Failed to initialize vector database: {e}")
            raise

    def _generate_document_id(self, message: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a message.

        Args:
            message: Message dictionary

        Returns:
            Unique ID string
        """
        # Create a unique ID based on sender, timestamp, and text
        unique_string = f"{message.get('sender', 'unknown')}_{message.get('timestamp', '')}_{message.get('text', '')[:50]}"
        return hashlib.md5(unique_string.encode()).hexdigest()

    def index_messages(self, messages: List[Dict[str, Any]]) -> None:
        """
        Index a batch of messages in the vector database.

        Args:
            messages: List of message dictionaries with 'text', 'sender', and 'timestamp' keys
        """
        if not messages:
            logging.warning("No messages to index")
            return

        try:
            # Extract text and metadata
            texts = [msg["text"] for msg in messages if "text" in msg and msg["text"]]
            if not texts:
                logging.warning("No valid text content to index")
                return

            ids = [self._generate_document_id(msg) for msg in messages if "text" in msg and msg["text"]]

            # Prepare metadata
            metadatas = []
            for msg in messages:
                if "text" not in msg or not msg["text"]:
                    continue

                metadata = {
                    "sender": msg.get("sender", "unknown"),
                    "timestamp": str(msg.get("timestamp", datetime.now())),
                }

                # Add any additional metadata fields
                for key, value in msg.items():
                    if key not in ["text", "sender", "timestamp"] and isinstance(value, (str, int, float, bool)):
                        metadata[key] = str(value)

                metadatas.append(metadata)

            # Generate embeddings using sentence-transformers
            logging.info(f"Generating embeddings for {len(texts)} messages")
            embeddings = self.embedding_model.encode(texts, convert_to_tensor=False).tolist()

            # Add to collection
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )

            logging.info(f"Indexed {len(texts)} messages in vector database")
        except Exception as e:
            logging.error(f"Error indexing messages: {e}")
            raise

    def query(self, question: str, n_results: int = 5) -> Dict[str, Any]:
        """
        Query the vector database for relevant messages.

        Args:
            question: Query text
            n_results: Number of results to return

        Returns:
            Dictionary with query results
        """
        try:
            # Generate embedding for the question using sentence-transformers
            logging.info(f"Generating embedding for query: {question[:30]}...")
            question_embedding = self.embedding_model.encode(question, convert_to_tensor=False).tolist()

            # Query the collection
            results = self.collection.query(
                query_embeddings=[question_embedding],
                n_results=n_results
            )

            logging.info(f"Query returned {len(results.get('documents', [[]])[0])} results")
            return results
        except Exception as e:
            logging.error(f"Error querying vector database: {e}")
            raise

    def get_daily_digest(self, date: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """
        Get a digest of messages for a specific date.

        Args:
            date: Date to get digest for, or None for today

        Returns:
            List of message dictionaries
        """
        if date is None:
            date = datetime.now()

        date_str = date.strftime("%Y-%m-%d")

        try:
            # Query messages from the specified date
            results = self.collection.query(
                query_texts=["daily digest"],  # This is just a placeholder
                where={"timestamp": {"$regex": f"^{date_str}"}}  # Filter by date prefix
            )

            # Process results
            digest = []
            for i, doc in enumerate(results.get("documents", [[]])[0]):
                metadata = results.get("metadatas", [[]])[0][i] if i < len(results.get("metadatas", [[]])[0]) else {}
                digest.append({
                    "text": doc,
                    "sender": metadata.get("sender", "unknown"),
                    "timestamp": metadata.get("timestamp", "")
                })

            return digest
        except Exception as e:
            logging.error(f"Error getting daily digest: {e}")
            return []
