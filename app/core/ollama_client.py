# core/ollama_client.py

"""
Ollama API Client for local AI processing
"""
import json
import base64
import asyncio
from typing import Dict, List, Optional, Any
from pathlib import Path
import httpx
from PIL import Image
import numpy as np
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class OllamaConfig(BaseModel):
    """Ollama configuration"""
    base_url: str = "http://localhost:11434"
    timeout: int = 300
    max_retries: int = 3
    image_model: str = "llava:13b"
    embedding_model: str = "nomic-embed-text:latest"
    text_model: str = "llama3:latest"


class ImageAnalysisResult(BaseModel):
    """Structured image analysis result"""
    caption: str
    tags: List[str]
    objects: List[str]
    mood: str
    contains_text: bool = False
    is_sensitive: bool = False
    confidence: float = 0.0


class OllamaClient:
    """Client for interacting with local Ollama instance"""
    
    def __init__(self, config: OllamaConfig = None):
        self.config = config or OllamaConfig()
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout
        )
        self._models_loaded = set()
        
    async def ensure_model_loaded(self, model_name: str):
        """Ensure model is loaded in Ollama"""
        if model_name in self._models_loaded:
            return
            
        try:
            response = await self.client.post("/api/show", json={"name": model_name})
            if response.status_code == 200:
                self._models_loaded.add(model_name)
            else:
                # Try to pull the model
                await self.client.post("/api/pull", json={"name": model_name})
                self._models_loaded.add(model_name)
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
            
    async def analyze_image(self, image_path: Path) -> ImageAnalysisResult:
        """
        Analyze image using LLaVA or Moondream model
        Returns structured JSON with caption, tags, objects, mood
        """
        await self.ensure_model_loaded(self.config.image_model)
        
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        # Prepare prompt for structured JSON output
        prompt = """Describe this image in JSON format with the following keys:
        - caption: A detailed description of the scene
        - tags: List of 5-10 relevant tags (lowercase, hyphenated)
        - objects: List of objects visible in the image
        - mood: The overall mood/atmosphere (e.g., happy, peaceful, energetic)
        - contains_text: Boolean if image contains readable text
        - is_sensitive: Boolean if image contains sensitive/private content
        
        Return ONLY valid JSON, no other text.
        Example: {"caption": "...", "tags": ["tag1", "tag2"], "objects": ["obj1", "obj2"], "mood": "peaceful", "contains_text": false, "is_sensitive": false}
        """
        
        try:
            response = await self.client.post(
                "/api/generate",
                json={
                    "model": self.config.image_model,
                    "prompt": prompt,
                    "images": [image_data],
                    "stream": False,
                    "format": "json"
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract JSON from response
                response_text = result.get("response", "")
                
                # Try to parse JSON
                try:
                    # Sometimes the response might have markdown code blocks
                    if "```json" in response_text:
                        json_str = response_text.split("```json")[1].split("```")[0]
                    elif "```" in response_text:
                        json_str = response_text.split("```")[1].split("```")[0]
                    else:
                        json_str = response_text
                        
                    data = json.loads(json_str.strip())
                    return ImageAnalysisResult(**data)
                    
                except json.JSONDecodeError:
                    # Fallback: extract structured info from text
                    logger.warning(f"Failed to parse JSON from response: {response_text[:200]}")
                    return self._parse_unstructured_response(response_text)
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                raise Exception(f"Ollama API error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error analyzing image {image_path}: {e}")
            raise
            
    def _parse_unstructured_response(self, text: str) -> ImageAnalysisResult:
        """Fallback parser for unstructured responses"""
        # Simple heuristic parsing
        caption = text.split("\n")[0][:200]
        tags = []
        objects = []
        mood = "neutral"
        
        # Extract potential tags (words in quotes or after #)
        import re
        tags = re.findall(r'"([^"]+)"', text) or re.findall(r'#(\w+)', text)
        
        return ImageAnalysisResult(
            caption=caption,
            tags=tags[:10],
            objects=objects,
            mood=mood,
            contains_text=False,
            is_sensitive=False,
            confidence=0.5
        )
    
    async def generate_embedding(self, text: str) -> List[float]:
        """Generate vector embedding for text using nomic-embed-text"""
        await self.ensure_model_loaded(self.config.embedding_model)
        
        try:
            response = await self.client.post(
                "/api/embeddings",
                json={
                    "model": self.config.embedding_model,
                    "prompt": text
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding", [])
            else:
                logger.error(f"Embedding API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return []
    
    async def generate_caption(self, image_path: Path) -> str:
        """Generate simple caption for image"""
        await self.ensure_model_loaded(self.config.image_model)
        
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode('utf-8')
        
        try:
            response = await self.client.post(
                "/api/generate",
                json={
                    "model": self.config.image_model,
                    "prompt": "Describe this image in one sentence.",
                    "images": [image_data],
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("response", "").strip()
            return ""
            
        except Exception as e:
            logger.error(f"Error generating caption: {e}")
            return ""
    
    async def chat_completion(self, messages: List[Dict], system_prompt: str = None) -> str:
        """Chat completion with Llama3"""
        await self.ensure_model_loaded(self.config.text_model)
        
        payload = {
            "model": self.config.text_model,
            "messages": messages,
            "stream": False
        }
        
        if system_prompt:
            payload["system"] = system_prompt
            
        try:
            response = await self.client.post("/api/chat", json=payload)
            if response.status_code == 200:
                result = response.json()
                return result.get("message", {}).get("content", "")
            return ""
            
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            return ""
    
    async def close(self):
        """Close HTTP client"""
        await self.client.aclose()


# Singleton instance
_ollama_client = None

def get_ollama_client() -> OllamaClient:
    """Get singleton Ollama client instance"""
    global _ollama_client
    if _ollama_client is None:
        _ollama_client = OllamaClient()
    return _ollama_client