import os
import json
import logging
from typing import List, Dict, Any, Union
from sentence_transformers import SentenceTransformer
from PIL import Image
import torch

log = logging.getLogger("embedder")

# Lazy loading models saves memory during fast API calls if not embedding
_text_model = None
_vision_model = None

def get_text_model():
    """Returns the BGE-large-en model for text and structured specs."""
    global _text_model
    if _text_model is None:
        log.info("Loading BGE Large En model...")
        _text_model = SentenceTransformer('BAAI/bge-large-en-v1.5')
    return _text_model

def get_vision_model():
    """Returns the CLIP model for images."""
    global _vision_model
    if _vision_model is None:
        log.info("Loading CLIP vision model...")
        _vision_model = SentenceTransformer('clip-ViT-B-32')
    return _vision_model

class Embedder:
    @staticmethod
    def embed_text(text: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        """Generate 1024d embeddings for raw text."""
        model = get_text_model()
        if isinstance(text, str):
            text = [text]
        # BGE recommends adding this prefix for retrieved queries, but for documents just use text
        embeddings = model.encode(text, normalize_embeddings=True)
        if len(embeddings) == 1:
            return embeddings[0].tolist()
        return embeddings.tolist()

    @staticmethod
    def embed_specs(specs: Dict[str, Any]) -> List[float]:
        """
        Convert structured JSON specs to a string representation 
        and generate 1024d embedding.
        """
        # Create a clean string representation of the specs
        spec_parts = []
        for k, v in specs.items():
            if not v or (isinstance(v, list) and not len(v)):
                continue
            if isinstance(v, list):
                val_str = ", ".join(str(x) for x in v)
            else:
                val_str = str(v)
            spec_parts.append(f"{k.replace('_', ' ').title()}: {val_str}")
            
        spec_string = ". ".join(spec_parts)
        log.debug(f"Structured Spec String: {spec_string}")
        
        model = get_text_model()
        return model.encode(spec_string, normalize_embeddings=True).tolist()

    @staticmethod
    def embed_image(image_path: str) -> List[float]:
        """Generate 512d CLIP embedding for a product or diagram image."""
        try:
            img = Image.open(image_path).convert("RGB")
            model = get_vision_model()
            embedding = model.encode(img)
            return embedding.tolist()
        except Exception as e:
            log.error(f"Failed to embed image {image_path}: {e}")
            raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    # print(len(Embedder.embed_text("High quality spotlight 12W")))
