from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
import os
from dotenv import load_dotenv
import logging

load_dotenv()
log = logging.getLogger("qdrant_db")

# Collection names for different embedding spaces
COLLECTION_TEXT = "starlight_text_vectors"
COLLECTION_SPEC = "starlight_spec_vectors"
COLLECTION_IMAGE = "starlight_image_vectors"

# Vector Dimensions
DIM_BGE_LARGE = 1024 
DIM_CLIP = 512

def get_qdrant_client():
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))
    return QdrantClient(host=host, port=port)

def init_qdrant_collections(client: QdrantClient = None):
    if client is None:
        client = get_qdrant_client()
        
    def recreate_if_not_exists(name, size, distance=Distance.COSINE):
        if not client.collection_exists(collection_name=name):
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=size, distance=distance)
            )
            log.info(f"Created Qdrant collection: {name} (Size: {size})")
        else:
            log.info(f"Qdrant collection {name} already exists.")

    try:
        # BGE-large-en outputs 1024d vectors
        recreate_if_not_exists(COLLECTION_TEXT, DIM_BGE_LARGE)
        recreate_if_not_exists(COLLECTION_SPEC, DIM_BGE_LARGE)
        
        # CLIP outputs 512d vectors
        recreate_if_not_exists(COLLECTION_IMAGE, DIM_CLIP)
        
    except Exception as e:
        log.error(f"Failed to initialize Qdrant collections: {e}")
        raise

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    init_qdrant_collections()
