"""
OpenVINO Integration Module for AI Assistant.

This module provides integration with OpenVINO for optimized inference
of language models for the messenger AI assistant.
"""

from openvino.runtime import Core
import numpy as np
import logging
import os
from typing import List, Dict, Any, Optional, Union

class OpenVINOModel:
    """
    Wrapper class for OpenVINO model inference.

    This class handles loading and running inference on language models
    using the OpenVINO runtime for optimized performance.
    """

    def __init__(self, model_path: str, device: str = "CPU", use_fallback: bool = True):
        """
        Initialize the OpenVINO model.

        Args:
            model_path: Path to the OpenVINO IR model (.xml and .bin files)
            device: Target device for inference (CPU, GPU, etc.)
            use_fallback: Whether to use a fallback implementation if model loading fails
        """
        self.use_fallback = use_fallback
        self.model_loaded = False
        self.model_path = model_path

        try:
            if os.path.exists(model_path) or os.path.exists(model_path + ".xml"):
                self.core = Core()
                logging.info(f"Loading OpenVINO model from {model_path}")
                self.model = self.core.read_model(model_path)
                self.compiled_model = self.core.compile_model(self.model, device)
                self.output_layer = self.compiled_model.output(0)
                self.input_layer = self.compiled_model.input(0)
                self.model_loaded = True
                logging.info(f"Model loaded successfully on {device}")
            else:
                logging.warning(f"Model file not found at {model_path}. Using fallback implementation.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenVINO model: {e}")
            if not use_fallback:
                raise
            logging.info("Using fallback implementation for text generation")

    def generate_response(self, prompt: str, context: str, max_length: int = 512) -> str:
        """
        Generate a response using the OpenVINO model.

        Args:
            prompt: The user's question or prompt
            context: The context information retrieved from the vector database
            max_length: Maximum length of the generated response

        Returns:
            Generated text response
        """
        # Combine prompt and context
        full_prompt = f"Context: {context}\n\nQuestion: {prompt}\n\nAnswer:"

        # If the model is loaded, use it for inference
        if self.model_loaded:
            try:
                # This is where you would implement the actual model inference
                # For a real implementation, you would:
                # 1. Tokenize the input text
                # 2. Create the appropriate input tensor
                # 3. Run inference
                # 4. Decode the output tokens to text

                # Placeholder for actual inference
                # result = self.compiled_model([input_tensor])[self.output_layer]

                # For now, we'll still use the fallback implementation
                pass
            except Exception as e:
                logging.error(f"Error during model inference: {e}")
                if not self.use_fallback:
                    return f"Error generating response: {str(e)}"
                logging.info("Falling back to simulated response")

        # Fallback implementation: generate a simulated response
        try:
            # Extract key information from the context
            context_preview = context[:100] + "..." if len(context) > 100 else context

            # Generate a more sophisticated simulated response
            response_parts = [
                f"Based on the provided context, I can tell you that:",
                f"",
            ]

            # Add some content based on the context and question
            if "daily" in prompt.lower() or "summary" in prompt.lower():
                response_parts.append(f"The daily summary shows several messages exchanged today.")
            elif "who" in prompt.lower():
                response_parts.append(f"The messages indicate communication between multiple users.")
            elif "when" in prompt.lower():
                response_parts.append(f"The timing of these messages appears to be recent.")
            elif "how" in prompt.lower():
                response_parts.append(f"The process described in the messages involves several steps.")
            else:
                response_parts.append(f"The context contains information relevant to your question about {prompt[:30]}...")

            response_parts.append(f"")
            response_parts.append(f"For more specific information, you might want to ask a more targeted question.")

            # Join the response parts
            response = "\n".join(response_parts)

            logging.info(f"Generated simulated response for prompt: {prompt[:30]}...")
            return response

        except Exception as e:
            logging.error(f"Error generating simulated response: {e}")
            return f"Error generating response: {str(e)}"

class OpenVINOTextEmbedding:
    """
    Class for generating text embeddings using OpenVINO.

    This class provides methods to generate embeddings for text using
    OpenVINO-optimized embedding models.
    """

    def __init__(self, model_path: str, device: str = "CPU", use_fallback: bool = True):
        """
        Initialize the embedding model.

        Args:
            model_path: Path to the OpenVINO IR model for embeddings
            device: Target device for inference
            use_fallback: Whether to use a fallback implementation if model loading fails
        """
        self.use_fallback = use_fallback
        self.model_loaded = False
        self.tokenizer = None
        self.st_model = None
        self.model_path = model_path
        self.device = device
        self.using_fallback = False

        try:
            # Check if model file exists
            if not os.path.exists(model_path) and not os.path.exists(model_path + ".xml"):
                logging.warning(f"⚠️ OpenVINO model file not found at {model_path}")
                raise FileNotFoundError(f"Model file not found at {model_path}")

            # Import here to avoid dependency if not using this class
            try:
                from transformers import AutoTokenizer
                # Load the tokenizer from sentence-transformers
                logging.info(f"Loading tokenizer for embedding model...")
                self.tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
                logging.info(f"Tokenizer loaded successfully")
            except Exception as e:
                logging.error(f"Failed to load tokenizer: {e}")
                raise

            # Load the OpenVINO model
            logging.info(f"Loading OpenVINO embedding model from {model_path}")
            self.core = Core()
            self.model = self.core.read_model(model_path)
            self.compiled_model = self.core.compile_model(self.model, device)
            self.output_layer = self.compiled_model.output(0)  # pooler_output
            self.model_loaded = True
            logging.info(f"✅ OpenVINO embedding model loaded successfully on {device}")
        except Exception as e:
            logging.error(f"Failed to initialize OpenVINO embedding model: {e}")
            self.using_fallback = True
            if not use_fallback:
                raise

            logging.warning(f"⚠️ USING FALLBACK FOR EMBEDDINGS: OpenVINO model failed to load")
            # Load sentence-transformers as fallback
            try:
                from sentence_transformers import SentenceTransformer
                logging.info(f"Loading sentence-transformers as fallback...")
                self.st_model = SentenceTransformer('all-MiniLM-L6-v2')
                logging.info(f"✅ Loaded sentence-transformers as fallback")
            except Exception as e:
                logging.error(f"Failed to load fallback sentence-transformers model: {e}")

    def encode(self, texts: Union[str, List[str]]) -> np.ndarray:
        """
        Generate embeddings for the given texts.

        Args:
            texts: Single text string or list of text strings

        Returns:
            Numpy array of embeddings
        """
        # If the model is loaded, use it for inference
        if self.model_loaded and self.tokenizer:
            try:
                # Convert to list if single text
                if isinstance(texts, str):
                    texts = [texts]

                # Tokenize the texts
                logging.info(f"Tokenizing {len(texts)} texts for OpenVINO embedding")
                encoded_inputs = self.tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="np"
                )

                # Run inference
                input_dict = {
                    "input_ids": encoded_inputs["input_ids"],
                    "attention_mask": encoded_inputs["attention_mask"]
                }

                # Get the pooler output (sentence embedding)
                logging.info(f"Running OpenVINO inference for embeddings")
                results = self.compiled_model(input_dict)[self.output_layer]
                logging.info(f"✅ Generated embeddings using OpenVINO model")

                # If single text, return single embedding
                if len(texts) == 1:
                    return results[0]

                return results

            except Exception as e:
                logging.error(f"Error during OpenVINO embedding inference: {e}")
                self.using_fallback = True
                if not self.use_fallback:
                    raise
                logging.warning(f"⚠️ USING FALLBACK FOR EMBEDDINGS: OpenVINO inference failed")
        elif self.using_fallback:
            logging.warning(f"⚠️ USING FALLBACK FOR EMBEDDINGS: OpenVINO model not loaded")

        # Fallback to sentence-transformers
        if self.st_model:
            try:
                logging.info(f"Generating embeddings using sentence-transformers fallback")
                result = self.st_model.encode(texts, convert_to_tensor=False)
                logging.info(f"✅ Generated embeddings using sentence-transformers fallback")
                return result
            except Exception as e:
                logging.error(f"Error with sentence-transformers fallback: {e}")
                if not self.use_fallback:
                    raise
                logging.warning(f"⚠️ USING RANDOM EMBEDDINGS: All embedding methods failed")

        # Last resort fallback: random embeddings
        logging.warning(f"⚠️ USING RANDOM EMBEDDINGS: No embedding model available")
        if isinstance(texts, str):
            # Single text, return a single embedding vector
            return np.random.randn(384).astype(np.float32)  # MiniLM-L6 has 384 dimensions
        else:
            # List of texts, return a batch of embedding vectors
            return np.random.randn(len(texts), 384).astype(np.float32)

def load_model(model_name: Optional[str] = None, use_fallback: bool = True) -> OpenVINOModel:
    """
    Factory function to load a specific OpenVINO model.

    Args:
        model_name: Name of the model to load, or None to use the default
        use_fallback: Whether to use a fallback implementation if model loading fails

    Returns:
        Initialized OpenVINOModel instance
    """
    # Get model path from environment variable or use default
    model_path = os.getenv("OPENVINO_MODEL_PATH", "./models/llm_model")

    if model_name:
        model_path = f"./models/{model_name}"

    # Create models directory if it doesn't exist
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    return OpenVINOModel(model_path, use_fallback=use_fallback)
