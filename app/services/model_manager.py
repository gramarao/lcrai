import logging
import os
import time
from typing import Dict, List, Optional, Any, AsyncGenerator
import google.generativeai as genai
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    key: str
    provider: str
    display_name: str
    model_id: str
    cost_per_1k_tokens: float
    max_tokens: int
    description: str
    available: bool = False

class ModelManager:
    def __init__(self):
        self.models = {
            "gemini-2.5-pro": ModelConfig(
                key="gemini-2.5-pro",
                provider="google",
                display_name="Gemini 2.5 Pro",
                model_id="gemini-2.5-pro",
                cost_per_1k_tokens=0.0075,
                max_tokens=2048,
                description="High-quality, comprehensive responses"
            ),
            "gemini-2.5-flash": ModelConfig(
                key="gemini-2.5-flash",
                provider="google",
                display_name="Gemini 2.5 Flash",
                model_id="gemini-2.5-flash",
                cost_per_1k_tokens=0.00075,
                max_tokens=2048,
                description="Fast and efficient, 10x cheaper than Pro"
            ),
            "gemini-2.0-flash": ModelConfig(
                key="gemini-2.0-flash",
                provider="google",
                display_name="Gemini 2.0 Flash (Latest)",
                model_id="gemini-2.0-flash-exp",
                cost_per_1k_tokens=0.00075,
                max_tokens=2048,
                description="Latest model, excellent performance"
            )
        }
        
        self.check_availability()
    
    def check_availability(self):
        """Check which models are actually available"""
        # Check Google models
        google_key = os.getenv("GOOGLE_AI_API_KEY")
        logger.info("Checking model availability")
        if google_key:
            try:
                genai.configure(api_key=google_key)
                for modelkey in self.models:
                    logger.info(f"Check:{modelkey}")
                    if self.models[modelkey].provider == "google":
                        self.models[modelkey].available = True
                logger.info("Google models available")
            except Exception as e:
                logger.warning(f"Google models not available: {e}")
        else:
            logger.info(f"Google key {google_key}")
    
    def get_available_models(self) -> Dict[str, ModelConfig]:
        """Get only available models"""
        return {k: v for k, v in self.models.items() if v.available}
    
    def get_model(self, key: str) -> Optional[ModelConfig]:
        """Get specific model config"""
        return self.models.get(key)
    
    async def generate_response(self, model_key: str, prompt: str) -> AsyncGenerator[Dict, None]:
        """Generate response using specified model"""
        model_config = self.get_model(model_key)
        if not model_config or not model_config.available:
            yield {"type": "error", "message": f"Model {model_key} not available"}
            return
        
        try:
            if model_config.provider == "google":
                model = genai.GenerativeModel(model_config.model_id)
                response_stream = model.generate_content(
                    prompt,
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,
                        top_p=0.8,
                        max_output_tokens=model_config.max_tokens
                    ),
                    stream=True
                )
                
                for chunk in response_stream:
                    if hasattr(chunk, 'text') and chunk.text:
                        yield {
                            "type": "content",
                            "content": chunk.text,
                            "tokens": len(chunk.text.split())
                        }
        except Exception as e:
            logger.error(f"Generation failed for {model_key}: {e}")
            yield {"type": "error", "message": f"Generation failed: {str(e)}"}

# Global instance
model_manager = ModelManager()
