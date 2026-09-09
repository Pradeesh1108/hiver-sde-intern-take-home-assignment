import os
from dotenv import load_dotenv
from openai import OpenAI
from groq import Groq

load_dotenv()

def generate_completion(messages: list, task_type: str = "fast", max_tokens: int = 150, temperature: float = 0.0) -> str:
    """
    Generate an LLM completion using the configured provider.
    
    Args:
        messages: List of message dicts (e.g., [{"role": "user", "content": "..."}])
        task_type: "fast" (for classification/escalation) or "draft" (for generating replies)
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
    """
    provider = os.getenv("LLM_PROVIDER", "groq").lower()

    if provider == "ollama":
        # Local Ollama Execution
        model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")
        client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama", # required, but unused by ollama
        )
        
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()

    else:
        # Groq Execution
        client = Groq()
        # Fallbacks to default groq models
        if task_type == "draft":
            model = "qwen/qwen3.8-27b" # originally gpt-oss-120b but changed to prevent TPM limits
        else:
            model = "qwen/qwen3.8-27b"

        # Rate Limit / Retry logic could go here if needed, 
        # but 04_eval.py sleeps mitigate this on the batch level.
        response = client.chat.completions.create(
            model=model,
            max_tokens=max_tokens,
            messages=messages,
            temperature=temperature
        )
        return response.choices[0].message.content.strip()
