"""
ollama_client.py
----------------
Handles communication with the local Ollama service.
"""

import json
import urllib.request
import urllib.error
from typing import List, Dict, Any, Optional

from app import config


class OllamaClient:
    """Client for communicating with Ollama."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434"
    ):
        self.base_url = base_url.rstrip("/")

    def _post(
        self,
        endpoint: str,
        data: Dict[str, Any],
        timeout: int = 60
    ) -> Dict[str, Any]:

        request = urllib.request.Request(
            f"{self.base_url}{endpoint}",
            data=json.dumps(data).encode("utf-8"),
            headers={"Content-Type": "application/json"}
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=timeout
            ) as response:
                return json.loads(
                    response.read().decode("utf-8")
                )

        except urllib.error.URLError as error:
            raise ConnectionError(
                f"Ollama request failed.\n"
                f"URL: {self.base_url}{endpoint}\n"
                f"Error: {error}"
            )

    def get_embedding(
        self,
        text: str,
        model: Optional[str] = None
    ) -> List[float]:
        """Generate an embedding for a text."""

        model = model or config.EMBED_MODEL

        response = self._post(
            "/api/embed",
            {
                "model": model,
                "input": text.strip() or " "
            }
        )

        return response["embeddings"][0]

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.1
    ) -> str:
        """Generate a response using Ollama."""

        model = model or config.GEN_MODEL

        response = self._post(
            "/api/chat",
            {
                "model": model,
                "messages": messages,
                "options": {
                    "temperature": temperature
                },
                "stream": False
            },
            timeout=120
        )

        return response["message"]["content"].strip()


default_client = OllamaClient()