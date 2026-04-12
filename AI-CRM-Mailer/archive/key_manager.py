import os
import time
import logging
from dotenv import load_dotenv
from google import genai

load_dotenv()
log = logging.getLogger("key_manager")

class GeminiKeyManager:
    def __init__(self):
        self.keys = []
        for i in range(1, 10):
            k = os.getenv(f"GEMINI_API_KEY_{i}")
            if k:
                self.keys.append(k)
        
        if not self.keys:
            k = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if k:
                self.keys.append(k)
                
        if not self.keys:
            log.warning("No GEMINI_API_KEYs found in environment.")
            
        self.models = ["gemini-2.5-flash"]
        self.current_key_index = 0
        self.current_model_index = 0
        
        if self.keys:
            self._apply_config(self.keys[self.current_key_index])

    def _apply_config(self, key):
        """Applies the current key and initializes the client."""
        self.client = genai.Client(api_key=key)
        os.environ["GEMINI_API_KEY"] = key
        os.environ["GOOGLE_API_KEY"] = key
        
    def get_current_model(self):
        return self.models[self.current_model_index]

    def rotate_resource(self):
        """
        Rotates through models first, then through keys.
        Returns True if rotation was successful, False if all resources exhausted.
        """
        # Try next model for current key
        if self.current_model_index < len(self.models) - 1:
            self.current_model_index += 1
            log.warning(f"🔄 Rotating Model: Switched to {self.get_current_model()} on current key.")
            return True
        
        # All models exhausted for current key, rotate key and reset model
        if self.keys and len(self.keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.keys)
            self.current_model_index = 0 # Reset to best model
            new_key = self.keys[self.current_key_index]
            log.warning(f"🔑 Rotating Key: Switched to key #{self.current_key_index + 1} and model {self.get_current_model()}")
            self._apply_config(new_key)
            return True
            
        log.error("❌ All keys and models exhausted.")
        return False

    def get_client(self):
        return self.client

key_manager = GeminiKeyManager()

def with_key_rotation(func):
    """Decorator to automatically rotate keys and models on ResourceExhausted or 429 exceptions."""
    import functools
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Calculate total possible attempts (keys * models)
        total_resources = len(key_manager.keys) * len(key_manager.models) if key_manager.keys else len(key_manager.models)
        attempts = 0
        
        while attempts < total_resources:
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_str = str(e).lower()
                # Check for rate limit or quota errors
                if any(x in err_str for x in ["429", "exhausted", "quota", "too many requests", "limit"]):
                    log.warning(f"⚠️ Quota/Rate limit hit. Error: {e}")
                    rotated = key_manager.rotate_resource()
                    if not rotated:
                        raise
                    attempts += 1
                    time.sleep(1) 
                    log.info(f"🔁 Retrying with {key_manager.get_current_model()} (Attempt {attempts}/{total_resources})...")
                else:
                    raise 
        
        raise Exception("❌ All API resources (keys and models) exhausted.")
        
    return wrapper
