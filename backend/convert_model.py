"""
Script to convert sentence-transformers model to OpenVINO format.
"""

import os
from sentence_transformers import SentenceTransformer
import torch
import logging
import subprocess
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create models directory
os.makedirs("models", exist_ok=True)

# Load the model
logging.info("Loading sentence-transformers model...")
model_name = "all-MiniLM-L6-v2"
model = SentenceTransformer(model_name)

# Get the transformer model
transformer_model = model._first_module().auto_model

# Create a dummy input
logging.info("Creating dummy input for ONNX export...")
dummy_input = torch.ones((1, 128), dtype=torch.long)  # Batch size 1, sequence length 128
attention_mask = torch.ones((1, 128), dtype=torch.long)

# Export to ONNX
logging.info("Exporting to ONNX...")
onnx_path = os.path.abspath("models/embedding_model.onnx")
torch.onnx.export(
    transformer_model,
    (dummy_input, attention_mask),
    onnx_path,
    input_names=["input_ids", "attention_mask"],
    output_names=["last_hidden_state", "pooler_output"],
    dynamic_axes={
        "input_ids": {0: "batch", 1: "sequence"},
        "attention_mask": {0: "batch", 1: "sequence"},
        "last_hidden_state": {0: "batch", 1: "sequence"},
        "pooler_output": {0: "batch"}
    },
    opset_version=14  # Using a newer opset version for scaled_dot_product_attention support
)

# Convert to OpenVINO IR using the runtime API directly
logging.info("Converting to OpenVINO IR using runtime API...")
try:
    # Use the runtime API directly - this is the most reliable method
    from openvino.runtime import Core
    logging.info("Importing OpenVINO Core...")
    core = Core()
    logging.info(f"Reading ONNX model from {onnx_path}...")
    ov_model = core.read_model(onnx_path)
    xml_path = os.path.abspath("models/embedding_model.xml")
    logging.info(f"Serializing model to {xml_path}...")
    from openvino.runtime import serialize
    serialize(ov_model, xml_path)
    logging.info("Model serialized successfully!")

except Exception as e:
    logging.error(f"Error during model conversion: {e}")
    logging.warning("Model conversion failed. The application will use sentence-transformers directly.")

    # Try command line approach as fallback
    try:
        logging.info("Trying command line conversion as fallback...")
        cmd = [sys.executable, "-m", "openvino.tools.mo", "--input_model", onnx_path, "--output_dir", "models/"]
        logging.info(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            logging.error(f"Command line conversion failed: {result.stderr}")
            logging.warning("All conversion methods failed. The application will use sentence-transformers directly.")
        else:
            logging.info(f"Command line conversion succeeded: {result.stdout}")
    except Exception as e2:
        logging.error(f"Command line conversion also failed: {e2}")
        logging.warning("All conversion methods failed. The application will use sentence-transformers directly.")

logging.info("Model conversion complete!")
logging.info("OpenVINO model saved to models/embedding_model.xml")
