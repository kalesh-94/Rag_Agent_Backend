"""
rag/ollama_client.py
Handles communication with the local Ollama LLM server.
"""

import requests
from core.config import settings


OLLAMA_URL = f"{settings.OLLAMA_BASE_URL}/api/generate"


def generate_response(prompt: str) -> str:
    """
    Sends prompt to Ollama and returns generated response.
    """

    payload = {
        "model": settings.OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "top_p": 0.9,
            "num_ctx": 4096
        }
    }

    headers = {
        "Content-Type": "application/json"
    }

    try:
        print(f"Calling Ollama model: {settings.OLLAMA_MODEL}")

        response = requests.post(
            OLLAMA_URL,
            json=payload,
            headers=headers,
            timeout=120
        )

        # Raise error if request failed
        response.raise_for_status()

        data = response.json()

        if "response" not in data:
            return "Ollama returned an unexpected response."

        return data["response"].strip()

    except requests.exceptions.ConnectionError:
        return (
            "Cannot connect to Ollama.\n"
            "Make sure Ollama is installed and running."
        )

    except requests.exceptions.Timeout:
        return "Ollama request timed out."

    except requests.exceptions.HTTPError as e:
        return f"Ollama HTTP error: {str(e)}"

    except Exception as e:
        return f"Ollama error: {str(e)}"