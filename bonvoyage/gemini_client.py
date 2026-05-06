"""Centralized Gemini client initialization and utilities."""

import os
from google import genai

# Initialize client with API key
def get_client() -> genai.Client:
    """Get or create the Gemini client instance."""
    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    return genai.Client(api_key=api_key)


def generate_content(prompt: str, model: str = "gemini-3-flash-preview") -> str:
    """
    Generate content using Gemini API.
    
    Args:
        prompt: The prompt to send to the model
        model: Model identifier (default: gemini-3-flash-preview)
    
    Returns:
        The generated text response
    """
    client = get_client()
    response = client.models.generate_content(
        model=model,
        contents=prompt
    )
    return response.text.strip()
